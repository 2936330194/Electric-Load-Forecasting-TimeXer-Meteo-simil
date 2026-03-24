"""
similar_day_retriever_320.py

基于《相似日检索系统架构设计文档_单年精简版》的独立实现：

1. 使用 2025 年负荷数据前 2/3 训练段构建离线检索库
2. 将未来 10 个通道 96 步 的气象窗口编码为 128 维潜向量
3. 叠加 5 维时间特征并进行时间权重提权
4. 采用 Exact Inner Product Search 返回 Top-K 相似历史负荷曲线

说明：
- 本程序不改动 `convlstm_ae.py` 的训练流程，但在推理时依赖其定义的模型结构和训练出的权重。
- 128 维气象编码通过预训练模型完成：
  原始气象帧 -> 对数变换(log1p) -> 通道级标准化 -> ConvLSTM-AE Encoder -> 128 维潜映射(Latent)
- faiss 可用时优先使用 IndexFlatIP；不可用时自动回退到 NumPy 精确检索
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
import h5py
import faiss
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from convlstm_ae import (
    BATCH_SIZE as AE_BATCH_SIZE,
    BEST_MODEL_FILE as AE_BEST_MODEL_FILE,
    CHECKPOINT_DIR as AE_CHECKPOINT_DIR,
    ConvLSTMAutoEncoder,
    GPU_ID as AE_GPU_ID,
    HIDDEN_DIM as AE_HIDDEN_DIM,
    IN_CHANNELS as AE_IN_CHANNELS,
    LATENT_DIM as AE_LATENT_DIM,
    LOG1P_CHANNELS as AE_LOG1P_CHANNELS,
    NORM_STATS_FILE as AE_NORM_STATS_FILE,
    NUM_LAYERS as AE_NUM_LAYERS,
    USE_GPU as AE_USE_GPU,
    WINDOW_SIZE as AE_WINDOW_SIZE,
)


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_LOAD_CSV = ROOT_DIR / "data" / "湖南省电力负荷_unknow.csv"
DEFAULT_FUTURE_CSV = ROOT_DIR / "data" / "湖南省电力负荷_unknow_future.csv"
DEFAULT_WEATHER_H5 = ROOT_DIR / "data" / "hunan_grid_meteo_20250101_20260228.h5"
DEFAULT_ARTIFACT_DIR = ROOT_DIR / "artifacts" / "similar_day_retriever_ae_128"
DEFAULT_AE_CHECKPOINT = (ROOT_DIR / AE_CHECKPOINT_DIR / AE_BEST_MODEL_FILE).resolve()
DEFAULT_AE_NORM_STATS = (ROOT_DIR / AE_CHECKPOINT_DIR / AE_NORM_STATS_FILE).resolve()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(errors="replace")


def _require_h5py() -> None:
    """
    检查当前环境是否安装了 h5py 库。
    由于气象数据存储在 HDF5 文件中，因此该库是必须的。
    如果在没有 h5py 的环境下运行，将抛出 ImportError。
    """
    if h5py is None:
        raise ImportError("缺少 h5py，无法读取气象 HDF5 文件。")


def _find_first_4d_dataset(h5_obj):
    """
    递归遍历 HDF5 对象，找到第一个 4 维 (4D) 的数据集 (Dataset)。

    Args:
        h5_obj: HDF5 文件对象或组群 (Group) 对象。

    Returns:
        h5py.Dataset: 找到的第一个 4 维数据集；如果未找到则返回 None。
        
    说明:
        气象数据通常包含 (时间帧, 通道数, 高度, 宽度) 4个维度，因此使用这个方法可以自动定位气象数据所在位置，
        而无需硬编码数据集的名称。
    """
    for key in h5_obj.keys():
        item = h5_obj[key]
        # 如果是数据集，并且维度为4（T, C, H, W），则认为是我们要找的气象数据
        if isinstance(item, h5py.Dataset) and item.ndim == 4:
            return item
        # 如果是组群，则递归查找
        if isinstance(item, h5py.Group):
            found = _find_first_4d_dataset(item)
            if found is not None:
                return found
    return None


def _decode_h5_strings(values: np.ndarray) -> List[str]:
    """
    将 HDF5 中读取出的字节字符串 (bytes) 数组解码为普通的 Python 字符串列表。

    Args:
        values (np.ndarray): 从 HDF5 中读取出的包含字节串或字符串的 NumPy 数组。

    Returns:
        List[str]: 解码后的字符串列表。通常用于处理 HDF5 中的时间戳数组。
    """
    decoded: List[str] = []
    for item in values.tolist():
        # 处理字节类型的字符串
        if isinstance(item, bytes):
            decoded.append(item.decode("utf-8"))
        else:
            decoded.append(str(item))
    return decoded


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    对二维矩阵的每一行进行 L2 范数归一化。
    使得归一化后的每一行向量的 L2 范数(长度)等于 1，从而可以用内积 (Inner Product) 来等价计算余弦相似度。

    Args:
        matrix (np.ndarray): 需要归一化的二维数组，形状为 [N, D]。
        eps (float): 防止除以 0 的极小值。

    Returns:
        np.ndarray: L2 归一化后的矩阵。
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    # 沿着 axis=1 求每个行向量的 L2 范数
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # 避免除以绝对的 0
    norms = np.maximum(norms, eps)
    return matrix / norms


@dataclass
class RetrievalResult:
    """
    相似日检索结果的数据容器类。
    用于封装从离线特征库中检索到的相关历史负荷片段及相似度信息。
    """
    query_timestamp: str           # 查询(当前预测点)的时间戳，字符串形式
    historical_indices: List[int]  # 检索到的历史相似日在库中的索引(位置)
    historical_timestamps: List[str] # 检索到的历史相似日起始时间戳
    similarity_scores: List[float]   # 余弦相似度得分 (内积计算得出，最高通常为1.0)
    load_curves: np.ndarray          # 检索出的实际历史负荷曲线序列数据，形状为 [top_k, pred_len]

    def to_dict(self) -> Dict[str, object]:
        """将检索结果转换为字典格式，主要用于 JSON 序列化输出。"""
        return {
            "query_timestamp": self.query_timestamp,
            "historical_indices": self.historical_indices,
            "historical_timestamps": self.historical_timestamps,
            "similarity_scores": self.similarity_scores,
            "load_curves": self.load_curves.tolist(),
        }


class ExactInnerProductIndex:
    """
    精确内积检索库封装。
    优先使用性能极佳的 faiss.IndexFlatIP 进行向量相似度检索；
    如果在没有安装 faiss 的环境下执行，则自动回退利用 NumPy 进行精确内积计算。
    这种设计大大增加了代码在不同环境下的鲁棒性。
    """

    def __init__(self, dim: int):
        """
        Args:
            dim (int): 要存储的向量维度
        """
        self.dim = int(dim)
        # 根据 faiss 是否可用决定所用的后端
        self.backend = "faiss" if faiss is not None else "numpy"
        # 如果 faiss 存在，初始化 FlatIP 索引 (即精确的内积距离。若进行查询时使用归一化向量，等同于余弦相似度)
        self._faiss_index = faiss.IndexFlatIP(self.dim) if faiss is not None else None
        # 如果是 numpy 后端，在 _base_vectors 中存储所有已添加的底库向量
        self._base_vectors: Optional[np.ndarray] = None

    @property
    def ntotal(self) -> int:
        """获取当前索引库中总计包含的基础向量数量"""
        if self.backend == "faiss":
            return int(self._faiss_index.ntotal)
        if self._base_vectors is None:
            return 0
        return int(self._base_vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        """
        向索引库添加特征向量。

        Args:
            vectors (np.ndarray): 二维向量数组，形状期望为 [N, dim]
        """
        # 统一转为 float32 并且确保内存连续，这对 faiss 和 numpy 加速运算都很重要
        vectors = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"向量维度错误，期望 [N, {self.dim}]，实际 {vectors.shape}")
            
        if self.backend == "faiss":
            self._faiss_index.add(vectors)
            return
            
        # Numpy 后端逻辑：垂直堆叠追加向量
        if self._base_vectors is None:
            self._base_vectors = vectors
        else:
            self._base_vectors = np.vstack([self._base_vectors, vectors]).astype(np.float32, copy=False)

    def search(self, queries: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        在索引中搜索与查询向量具有最大内积分数的前 k 个向量。

        Args:
            queries (np.ndarray): 查询向量数组，形状为 [N_queries, dim]
            top_k (int): 返回匹配分数最高的前 k 个近邻

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - scores: 形状为 [N_queries, top_k] 的得分配列
                - ids:    形状为 [N_queries, top_k] 的对应近邻索引排列
        """
        queries = np.ascontiguousarray(np.asarray(queries, dtype=np.float32))
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError(f"查询向量维度错误，期望 [N, {self.dim}]，实际 {queries.shape}")
        if self.ntotal <= 0:
            raise RuntimeError("索引为空，无法执行检索。")

        # 限制 top_k 不超过索引库的最大总数
        top_k = min(int(top_k), self.ntotal)
        if top_k <= 0:
            raise ValueError("top_k 必须为正整数。")

        # faiss 提供直接搜索方法
        if self.backend == "faiss":
            return self._faiss_index.search(queries, top_k)

        # Numpy 后端实现高效求前 K 大相似度的逻辑
        # 1. 计算全部查询向量与基础向量之间的内积 [N_queries, N_total]
        scores = queries @ self._base_vectors.T
        
        # 2. 使用 argpartition (仅做前 K 划分而不完全排序，计算非常快)
        part = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]
        
        # 3. 提取前 K 大的分数
        part_scores = np.take_along_axis(scores, part, axis=1)
        
        # 4. 对这提取出的前 K 个分数在局部空间中重新进行倒序排序
        order = np.argsort(-part_scores, axis=1)
        
        # 5. 获得全局的 id 顺序
        ids = np.take_along_axis(part, order, axis=1).astype(np.int64)
        
        # 6. 获取全局的分数顺序
        sorted_scores = np.take_along_axis(part_scores, order, axis=1).astype(np.float32)
        return sorted_scores, ids


class HDF5WeatherSequenceStore:
    """
    针对当前项目气象 HDF5 的轻量顺序窗口读取器。
    提供了一种按需（按序列或按时间戳）轻量提取连续气象帧的机制，
    避免将巨大的气象数据一次性加载到内存中 (例如 11.5+ GiB 的张量)。
    """

    def __init__(self, h5_path: Union[str, Path]):
        """
        初始化读取器。
        
        Args:
            h5_path: 气象 HDF5 文件的路径。
        """
        _require_h5py()
        self.h5_path = Path(h5_path).resolve()
        if not self.h5_path.exists():
            raise FileNotFoundError(f"未找到气象 HDF5 文件: {self.h5_path}")

        self._file = None
        self.dataset = None
        self.dataset_name = ""
        self.timestamps: Optional[pd.DatetimeIndex] = None
        self.timestamp_to_index: Dict[int, int] = {}
        self.frame_shape: Optional[Tuple[int, int, int]] = None
        self.freq: Optional[pd.Timedelta] = None
        self._prepare()

    def _prepare(self) -> None:
        """打开 HDF5 文件并初始化元数据（如维度形状、时间戳索引映射等）。"""
        self._file = h5py.File(self.h5_path, "r")
        # 寻找气象数据集本身
        dataset = _find_first_4d_dataset(self._file)
        if dataset is None:
            raise ValueError(f"HDF5 中未找到 4D 气象数据集: {self.h5_path}")
        self.dataset = dataset
        self.dataset_name = dataset.name
        # 提取除去时间帧（第一个维度）之后的 (C, H, W) 作为形状参数
        self.frame_shape = tuple(int(x) for x in dataset.shape[1:])

        # 挂载对应的时间戳信息以供检索
        if "timestamps" not in self._file:
            raise ValueError(f"HDF5 中缺少 timestamps 数据集: {self.h5_path}")
        timestamps = pd.to_datetime(_decode_h5_strings(self._file["timestamps"][:]))
        self.timestamps = pd.DatetimeIndex(timestamps)
        # 构建从时间戳整数值（纳秒）到数据行索引的哈希映射
        self.timestamp_to_index = {int(ts.value): idx for idx, ts in enumerate(self.timestamps)}

        # 推断时间频率
        if len(self.timestamps) >= 2:
            self.freq = self.timestamps[1] - self.timestamps[0]

    def close(self) -> None:
        """安全关闭 HDF5 文件句柄。"""
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
        self._file = None
        self.dataset = None

    def __del__(self):
        self.close()

    def __len__(self) -> int:
        """返回 HDF5 中包含的总时间帧数。"""
        return int(self.dataset.shape[0])

    def verify_alignment(self, expected_dates: Sequence[pd.Timestamp]) -> None:
        """
        验证外部时间戳序列 (如负荷数据的 date 列) 与本 HDF5 存储内的时间戳序列是否在开头完全对齐。
        如果不一致则抛出异常。
        """
        expected = pd.DatetimeIndex(pd.to_datetime(expected_dates))
        actual = self.timestamps[: len(expected)]
        if len(actual) != len(expected):
            raise ValueError(
                f"HDF5 时间轴长度不足: 需要 {len(expected)}，实际仅有 {len(actual)}"
            )
        # asi8 表示为 int64 数组 (时间戳长整型)
        if not np.array_equal(actual.asi8, expected.asi8):
            mismatch = np.where(actual.asi8 != expected.asi8)[0]
            first_bad = int(mismatch[0])
            raise ValueError(
                "负荷 CSV 与 HDF5 时间轴未对齐: "
                f"位置 {first_bad}, csv={expected[first_bad]}, h5={actual[first_bad]}"
            )

    def lookup_index(self, timestamp: Union[str, pd.Timestamp]) -> int:
        """根据给定的时间戳对象或者字符串，反查其在 HDF5 文件中的行索引。"""
        ts = pd.Timestamp(timestamp)
        key = int(ts.value)
        if key not in self.timestamp_to_index:
            raise KeyError(f"时间戳不在气象 HDF5 覆盖范围内: {ts}")
        return self.timestamp_to_index[key]

    def get_block(self, start: int, end: int) -> np.ndarray:
        """
        按 [start, end) 索引切片，从磁盘中读取大块气象数据。
        
        Returns:
            np.ndarray: 取出的气象切片，类型被统一转换为 float32。
        """
        start = int(start)
        end = int(end)
        if start < 0 or end > len(self):
            raise IndexError(f"读取区间越界: [{start}, {end}) / {len(self)}")
        if end <= start:
            raise ValueError(f"非法读取区间: [{start}, {end})")
        return np.asarray(self.dataset[start:end], dtype=np.float32)

    def get_window(self, start_index: int, window_size: int) -> np.ndarray:
        """从某索引开始读取指定长度(window_size)的气象数据窗口。"""
        return self.get_block(start_index, start_index + int(window_size))

    def get_window_by_timestamp(
        self, timestamp: Union[str, pd.Timestamp], window_size: int
    ) -> np.ndarray:
        """从某个特定时间戳开始，向后提取所需步长的气象数据。"""
        start_index = self.lookup_index(timestamp)
        return self.get_window(start_index, int(window_size))


class StatisticalWeatherEncoder:
    """
    独立的统计型气象编码器（备选方案）。
    它负责将极高维度的气象图像序列(如 96小时 * 10通道 * 61高 * 62宽) 通过空间统计和 PCA 降维。

    核心处理流程式：
    1. 预处理 (log1p & Standard Scale): 对原始 [T, C, H, W] 气象块进行对数变换和 Z-Score 标准化。
    2. 空间聚合降维: 沿空间维度 (H, W) 提取像素单帧的全局统计量：mean/std/min/max，得到 [T, C] 分布。
    3. 时序拼接展开: 以指定的窗口步长 (例如 96 步) 滑动，收集窗口内的所有空间统计信息拼接为一个巨大的描述向量。
    4. 特征降维 (PCA): 用标准缩放器做二次处理最后交由增量PCA (IncrementalPCA) 解压到紧凑的低维子空间。
    """

    def __init__(
        self,
        weather_dim: int = AE_LATENT_DIM,
        window_size: int = 96,
        stats: Sequence[str] = ("mean", "std", "min", "max"),
        log1p_channels: Sequence[int] = (9,),
        batch_size: int = 384,
        channel_mean: Optional[np.ndarray] = None,
        channel_std: Optional[np.ndarray] = None,
    ):
        """
        初始化统计编码器。

        Args:
            weather_dim: 最终所需的输出维度，如果使用统计方案，该维度由 PCA 产生。
            window_size: 气象窗口长度 (时间步数)，对应于预测步长。
            stats: 要提取的空间维统计度量名列表。
            log1p_channels: 因为偏度极高，需要预先进行 log(1+x) 缩放转换的气象通道索引（例如降水通道）。
            batch_size: 在 IncrementalPCA 训练及预测时的批次大小。
            channel_mean: 全局已统计得出的像素通道均值数组。
            channel_std: 全局已统计得出的像素通道标准差数组。
        """
        self.requested_dim = int(weather_dim)
        self.window_size = int(window_size)
        self.stats = tuple(str(x) for x in stats)
        self.log1p_channels = tuple(int(x) for x in log1p_channels)
        self.batch_size = int(batch_size)
        self.channel_mean = None if channel_mean is None else np.asarray(channel_mean, dtype=np.float32)
        self.channel_std = None if channel_std is None else np.asarray(channel_std, dtype=np.float32)
        
        # 将被训练并用来对时序展开拼接后的超级长特征进行去均值缩放
        self.feature_scaler: Optional[StandardScaler] = None
        # 增量主成分分析，避免在内存中实例化巨型协方差矩阵
        self.pca: Optional[IncrementalPCA] = None
        self.output_dim: Optional[int] = None

        allowed = {"mean", "std", "min", "max"}
        unknown = [name for name in self.stats if name not in allowed]
        if unknown:
            raise ValueError(f"不支持的统计量: {unknown}")

    @property
    def channel_count(self) -> int:
        """从预先设置的均值数组中提取气象多通道数量。"""
        if self.channel_mean is None:
            raise RuntimeError("channel_mean 尚未设置。")
        return int(self.channel_mean.shape[1])

    @property
    def raw_feature_dim(self) -> int:
        """获取按时间窗和空间维度拼装展平处理后、降维之前对应的庞大原始特征维数。"""
        return len(self.stats) * self.window_size * self.channel_count

    def set_channel_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        """外部注入整个数据集在每个通道上的像素均值与标准差。"""
        self.channel_mean = np.asarray(mean, dtype=np.float32)
        self.channel_std = np.asarray(std, dtype=np.float32)

    def preprocess_frames(self, frames: np.ndarray) -> np.ndarray:
        """
        对气象张量图块执行预处理，包含部分通道的对数转换和平移缩放。

        Args:
            frames: 形状为 [T, C, H, W] 的原始帧切片。

        Returns:
            np.ndarray: 标准化后的气象张量块副本。
        """
        if self.channel_mean is None or self.channel_std is None:
            raise RuntimeError("请先设置 channel_mean/channel_std。")

        arr = np.asarray(frames, dtype=np.float32).copy()
        if arr.ndim != 4:
            raise ValueError(f"气象块必须为 [T, C, H, W]，实际 {arr.shape}")

        for channel in self.log1p_channels:
            # clip 截断负数以防止 log 计算发生非数值型错误
            arr[:, channel, :, :] = np.log1p(np.clip(arr[:, channel, :, :], a_min=0.0, a_max=None))

        # 广播减去均值并除以标准差
        arr = (arr - self.channel_mean) / self.channel_std
        return arr.astype(np.float32, copy=False)

    def extract_raw_features_from_block(
        self, frames: np.ndarray, expected_windows: Optional[int] = None
    ) -> np.ndarray:
        """
        执行降维的第一阶：消除 H 和 W 的空间维，得到随时间滑窗的原始海量特征。

        Args:
            frames: [T_chunk, C, H, W] 的大型连续气象帧，用于滑窗。
            expected_windows: 断言保障生成的窗口数量等于预期值。

        Returns:
            np.ndarray: 形状为 [N_windows, 96*C*num_stats] 的二维数组。
        """
        frames = self.preprocess_frames(frames)
        if frames.shape[0] < self.window_size:
            raise ValueError(
                f"气象块步数不足，至少需要 {self.window_size}，实际只有 {frames.shape[0]}"
            )

        metric_arrays: Dict[str, np.ndarray] = {}
        # 针对末尾的 H 和 W 轴计算全局统计抽象，将 [..., H, W] 压缩为 [...]
        if "mean" in self.stats:
            metric_arrays["mean"] = frames.mean(axis=(-1, -2), dtype=np.float32)
        if "std" in self.stats:
            metric_arrays["std"] = frames.std(axis=(-1, -2), dtype=np.float32)
        if "min" in self.stats:
            metric_arrays["min"] = frames.min(axis=(-1, -2))
        if "max" in self.stats:
            metric_arrays["max"] = frames.max(axis=(-1, -2))

        pieces: List[np.ndarray] = []
        for name in self.stats:
            metric = metric_arrays[name]
            # 生成随时间轴的滑动窗口视角，shape变为 [T_chunk - window_size + 1, C, window_size]
            window_view = sliding_window_view(metric, window_shape=self.window_size, axis=0)
            window_view = np.moveaxis(window_view, -1, 1) # 调整时间步维度至前面
            pieces.append(window_view.reshape(window_view.shape[0], -1).astype(np.float32, copy=False))

        # 将多个统计量沿特征维度直接横向拼接
        raw = np.concatenate(pieces, axis=1).astype(np.float32, copy=False)
        if expected_windows is not None and raw.shape[0] != int(expected_windows):
            raise RuntimeError(
                f"生成的窗口数异常，期望 {expected_windows}，实际 {raw.shape[0]}"
            )
        return raw

    def fit(self, raw_feature_batches_factory: Callable[[], Iterator[np.ndarray]], total_samples: int) -> None:
        """
        利用增量算法迭代地在全部原始描述向量总库上拟合尺度缩放器和 PCA 组件，防止内存泄漏。

        Args:
            raw_feature_batches_factory: 无参闭包，被调用时返回能逐批产生原始描述向量对列的迭代器。
            total_samples: 用来确保设定的 PCA 维度不会超过样本上限。
        """
        if total_samples <= 0:
            raise ValueError("total_samples 必须为正整数。")

        # 阶段 A：统计拟合 StandardScaler
        self.feature_scaler = StandardScaler()
        for batch_raw in raw_feature_batches_factory():
            self.feature_scaler.partial_fit(batch_raw)

        # 取需求维数、原维度限制、样本数目的最小值作为实际PCA提取维数
        effective_dim = min(self.requested_dim, self.raw_feature_dim, int(total_samples))
        if effective_dim < self.requested_dim:
            print(
                f"[编码器] 样本不足以支撑 {self.requested_dim} 维，"
                f"自动降为 {effective_dim} 维。"
            )

        self.output_dim = int(effective_dim)
        pca_batch_size = max(self.batch_size, self.output_dim)
        # 阶段 B：增量拟合 PCA
        self.pca = IncrementalPCA(n_components=self.output_dim, batch_size=pca_batch_size)
        for batch_raw in raw_feature_batches_factory():  # 此处重扫全量数据进行抽取
            batch_scaled = self.feature_scaler.transform(batch_raw).astype(np.float32, copy=False)
            self.pca.partial_fit(batch_scaled)

    def transform_raw(self, raw_features: np.ndarray) -> np.ndarray:
        """
        对给定的时空拼接版巨大向量进行压缩转换。

        Returns:
            np.ndarray: 被压缩后的稠密潜变量表达。
        """
        if self.feature_scaler is None or self.pca is None or self.output_dim is None:
            raise RuntimeError("编码器尚未完成 fit。")

        raw_features = np.asarray(raw_features, dtype=np.float32)
        scaled = self.feature_scaler.transform(raw_features).astype(np.float32, copy=False)
        encoded = self.pca.transform(scaled).astype(np.float32, copy=False)
        return encoded

    def transform_window(self, weather_window: np.ndarray) -> np.ndarray:
        """高层封装，从某个长度为预测区间的 4D tensor 直接获得降维潜向量。"""
        raw = self.extract_raw_features_from_block(weather_window, expected_windows=1)
        return self.transform_raw(raw)


class ConvLSTMAEWeatherEncoder:
    """
    基于预训练 ConvLSTM 自编码器的气象编码器。
    用于将高维的连续气象帧序列压缩降维。
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = DEFAULT_AE_CHECKPOINT,
        norm_stats_path: Union[str, Path] = DEFAULT_AE_NORM_STATS,
        window_size: int = AE_WINDOW_SIZE,
        latent_dim: int = AE_LATENT_DIM,
        batch_size: int = AE_BATCH_SIZE,
        hidden_dim: int = AE_HIDDEN_DIM,
        num_layers: int = AE_NUM_LAYERS,
        in_channels: int = AE_IN_CHANNELS,
        log1p_channels: Sequence[int] = AE_LOG1P_CHANNELS,
    ):
        """
        初始化自编码器封装。

        Args:
            checkpoint_path (Union[str, Path]): AE 模型权重检查点路径。
            norm_stats_path (Union[str, Path]): 用于预处理的归一化均值和标准差统计文件路径。
            window_size (int): 预测模型时间窗口长度 (包含多个气象图像帧)。
            latent_dim (int): 由 AE 模型提取的隐藏空间连续变量特征维度。
            batch_size (int): 数据通过 AE 模型的前向传播微批次规格以便防显存溢出。
            hidden_dim (int): ConvLSTM 中的内部通道数量。
            num_layers (int): ConvLSTM 的循环神经层堆叠深度。
            in_channels (int): 每帧输入网络的气象多变量原始通道总数。
            log1p_channels (Sequence[int]): 需以 log(1+x) 计算缓解右偏态极端值的降水或风速通道索引。
        """
        self.checkpoint_path = str(Path(checkpoint_path).resolve())
        self.norm_stats_path = str(Path(norm_stats_path).resolve())
        self.window_size = int(window_size)
        self.output_dim = int(latent_dim)
        self.batch_size = int(batch_size)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.in_channels = int(in_channels)
        self.log1p_channels = tuple(int(x) for x in log1p_channels)

        self.channel_mean: Optional[np.ndarray] = None
        self.channel_std: Optional[np.ndarray] = None
        self.frame_height: Optional[int] = None
        self.frame_width: Optional[int] = None

        self._model: Optional[ConvLSTMAutoEncoder] = None
        self._device: Optional[torch.device] = None
        self._use_amp = False

    def __getstate__(self) -> dict:
        """配置支持 Pickle 序列化以实现状态持久化；清理包含特定设备状态或网络权重的临时不可序列化指针对象。"""
        state = self.__dict__.copy()
        state["_model"] = None
        state["_device"] = None
        state["_use_amp"] = False
        return state

    def __setstate__(self, state: dict) -> None:
        """对象反序列化回调函数：重载基础成员并将底层依赖环境配置参数重置为空，交由运行时再次加载。"""
        self.__dict__.update(state)
        self._model = None
        self._device = None
        self._use_amp = False

    @property
    def channel_count(self) -> int:
        """获取训练集中提取的气象统计数据中的通道数。"""
        self._ensure_norm_stats_loaded()
        return int(self.channel_mean.shape[1])

    def _ensure_norm_stats_loaded(self) -> None:
        """加载用于归一化预处理的预置长效统计量数据如像素级均值及方差；这对应着我们在全局气象建库前的均值计算。"""
        if self.channel_mean is not None and self.channel_std is not None:
            return

        norm_stats_path = Path(self.norm_stats_path)
        if not norm_stats_path.exists():
            raise FileNotFoundError(f"未找到 ConvLSTM AE 归一化统计数据文件: {norm_stats_path}")

        stats = np.load(norm_stats_path)
        self.channel_mean = np.asarray(stats["mean"], dtype=np.float32)
        self.channel_std = np.asarray(stats["std"], dtype=np.float32)
        # 固定最小方差为 1，确保之后计算不产生除以 0 时发生的无穷大或空数据
        self.channel_std = np.where(self.channel_std < 1e-8, 1.0, self.channel_std).astype(np.float32)
        if "log1p_channels" in stats:
            self.log1p_channels = tuple(int(x) for x in np.asarray(stats["log1p_channels"]).tolist())

    def _resolve_device(self) -> torch.device:
        """根据超参数配置和环境实际资源推断运算应当运行的设备 (GPU/CPU)。"""
        if AE_USE_GPU and torch.cuda.is_available():
            return torch.device(f"cuda:{AE_GPU_ID}")
        return torch.device("cpu")

    def _ensure_model_loaded(self, frame_height: int, frame_width: int) -> None:
        """
        初始化神经网络结构。
        将训练保存好的 ConvLSTM AutoEncoder 节点还原，并将前向执行模式设好以便之后特征抽取使用。
        """
        self._ensure_norm_stats_loaded()

        if self._model is not None:
            if self.frame_height != int(frame_height) or self.frame_width != int(frame_width):
                raise ValueError(
                    f"ConvLSTM AE 帧尺寸不匹配: 期望 ({self.frame_height}, {self.frame_width})，"
                    f"实际 ({frame_height}, {frame_width})"
                )
            return

        checkpoint_path = Path(self.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"未找到 ConvLSTM AE 模型权重文件: {checkpoint_path}")

        self.frame_height = int(frame_height)
        self.frame_width = int(frame_width)
        self._device = self._resolve_device()
        self._use_amp = self._device.type == "cuda"  # 利用自动混合精度加速编码过程

        model = ConvLSTMAutoEncoder(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_dim,
            latent_dim=self.output_dim,
            num_layers=self.num_layers,
            seq_len=self.window_size,
            frame_height=self.frame_height,
            frame_width=self.frame_width,
        ).float().to(self._device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=self._device))
        model.eval()  # 固定 Batch Normalization、Dropout 的运行方式
        self._model = model

    def preprocess_frames(self, frames: np.ndarray) -> np.ndarray:
        """预处理：数据对齐校验 -> 偏度修正裁剪 -> 根据训练总集分布做标准缩放均一化。"""
        self._ensure_norm_stats_loaded()

        arr = np.asarray(frames, dtype=np.float32).copy()
        if arr.ndim != 4:
            raise ValueError(f"气象序列维度必须为 [T, C, H, W]，但获得的是 {arr.shape}")
        if arr.shape[1] != self.channel_count:
            raise ValueError(
                f"气象数据通道数错误：期望为 {self.channel_count}，但获得的是 {arr.shape[1]}"
            )

        for channel in self.log1p_channels:
            arr[:, channel, :, :] = np.log1p(np.clip(arr[:, channel, :, :], a_min=0.0, a_max=None))

        arr = (arr - self.channel_mean) / self.channel_std
        return arr.astype(np.float32, copy=False)

    def transform_frame_block(
        self,
        frames: np.ndarray,
        expected_windows: Optional[int] = None,
    ) -> np.ndarray:
        """
        核心气象序列嵌入函数（长块接口）。
        沿滑动窗口机制裁剪处理连续数据集并喂入 AutoEncoder 将其转换为固定尺寸特征表示以便比较或者建立检索树。
        """
        frames = self.preprocess_frames(frames)
        if frames.shape[0] < self.window_size:
            raise ValueError(
                f"气象输入序列总长度不足时间窗口的大小预设值: {frames.shape[0]} < {self.window_size}"
            )

        self._ensure_model_loaded(frames.shape[2], frames.shape[3])
        window_view = sliding_window_view(frames, window_shape=self.window_size, axis=0)
        window_view = np.moveaxis(window_view, -1, 1)
        num_windows = int(window_view.shape[0])
        if expected_windows is not None and num_windows != int(expected_windows):
            raise RuntimeError(f"预期获得 {expected_windows} 窗口的数据切片量但返回了 {num_windows} 个")

        encoded = np.empty((num_windows, self.output_dim), dtype=np.float32)
        micro_batch = max(1, self.batch_size)

        for start in range(0, num_windows, micro_batch):
            end = min(start + micro_batch, num_windows)
            batch_np = np.ascontiguousarray(window_view[start:end], dtype=np.float32)
            batch_tensor = torch.from_numpy(batch_np).to(device=self._device, dtype=torch.float32)
            # 无需反向传播，节约并且加快推理运行
            with torch.inference_mode():
                # 用混合精度降低运算负荷加速计算
                with torch.amp.autocast("cuda", enabled=self._use_amp):
                    batch_encoded = self._model.encode(batch_tensor)
            encoded[start:end] = batch_encoded.float().cpu().numpy()

        return encoded

    def transform_window(self, weather_window: np.ndarray) -> np.ndarray:
        """对单一查询或窗口直接转换处理，相当于 block 方法封装对预期 1 窗口断言控制。"""
        return self.transform_frame_block(weather_window, expected_windows=1)

    def fit(self, *args, **kwargs) -> None:
        """保留方法向后兼容以往旧设计接口(例如 PCA 的拟合函数)。实际上神经网络属于线下提前固化模型无需这一操作。"""
        return None


class SimilarDayRetriever:
    """
    相似日检索系统主类。
    
    整合并驱动了以下流程：
    1. 负荷/气象数据对齐加载和划分；
    2. 气象编码器的初始化与全量库特征转换；
    3. 时间特征向量生成及与气象特征的多模态加权融合 (时间特征由 alpha 参数提权) ；
    4. 建立底库索引 (ExactInnerProductIndex) ；
    5. 提供基于时间戳、在线气象窗口、或者未来的独立气象窗口的 Top-K 查询；
    6. 将所有的模型结构、数据集元模型等固化到本地 artifact 目录中以备快速加载复用。
    """

    def __init__(
        self,
        weather_dim: int = AE_LATENT_DIM,
        time_weight: float = 2.0,
        pred_len: int = 96,
        train_ratio: float = 2.0 / 3.0,
        freq: str = "15min",
        build_batch_size: int = 384,
        ae_checkpoint_path: Union[str, Path] = DEFAULT_AE_CHECKPOINT,
        ae_norm_stats_path: Union[str, Path] = DEFAULT_AE_NORM_STATS,
        encoder_batch_size: int = AE_BATCH_SIZE,
        stats: Sequence[str] = ("mean", "std", "min", "max"),
        log1p_channels: Sequence[int] = (9,),
    ):
        """
        初始化系统核心参数。

        Args:
            weather_dim: 降维后的气象特征目标维度。
            time_weight: 用于融合时强化时间特征（年/月/日等周期性模式被重点关注）的标量乘数 alpha。
            pred_len: 单次查询与返回的曲线点数 (如 96 点代表 24 小时 * 每 15 分钟 1 个点)。
            train_ratio: 按时间顺序划分的前部分序列作为建库集（由于检索的是历史发生过的片段）。
            freq: 采样频率字符串，用于辅助时间戳时间特征的提取。
            build_batch_size: 避免OOM的内部处理块大小。
            stats: 提取使用的统计特征集合。
            log1p_channels: 偏态需要对数缩放的指标维。
        """
        self.weather_dim = int(weather_dim)
        self.time_weight = float(time_weight)
        self.pred_len = int(pred_len)
        self.train_ratio = float(train_ratio)
        self.freq = str(freq)
        self.build_batch_size = int(build_batch_size)
        self.ae_checkpoint_path = str(Path(ae_checkpoint_path).resolve())
        self.ae_norm_stats_path = str(Path(ae_norm_stats_path).resolve())
        self.encoder_batch_size = int(encoder_batch_size)
        self.stats = tuple(stats)
        self.log1p_channels = tuple(int(x) for x in log1p_channels)

        # 核心功能组件，只有在 build() 或 load() 后才会被实例化或赋予数据
        self.weather_encoder: Optional[ConvLSTMAEWeatherEncoder] = None
        self.index: Optional[ExactInnerProductIndex] = None
        
        # 对应着库里所有的搜索底库和标签数据
        self.base_vectors: Optional[np.ndarray] = None
        self.load_curves: Optional[np.ndarray] = None
        self.start_timestamps: Optional[pd.DatetimeIndex] = None

        # 元数据追踪属性
        self.load_csv_path: Optional[str] = None
        self.weather_h5_path: Optional[str] = None
        self.train_frame_count: Optional[int] = None
        self.train_window_count: Optional[int] = None
        self.fused_dim: Optional[int] = None

    def _ensure_built(self) -> None:
        """断言类内部结构已经完全填充，否则报错阻截使用。"""
        if (
            self.weather_encoder is None
            or self.index is None
            or self.base_vectors is None
            or self.load_curves is None
            or self.start_timestamps is None
        ):
            raise RuntimeError("检索器尚未 build/load。")

    def _load_load_dataframe(self, load_csv_path: Union[str, Path]) -> pd.DataFrame:
        """
        读取并清洗电网目标负荷标量值数据表。确保其类型为 np.float32 且被适度重排序及索引复位。
        """
        load_csv_path = Path(load_csv_path).resolve()
        if not load_csv_path.exists():
            raise FileNotFoundError(f"未找到负荷 CSV 文件: {load_csv_path}")

        df = pd.read_csv(load_csv_path)
        if "date" not in df.columns:
            raise ValueError(f"负荷 CSV 缺少 date 列: {load_csv_path}")
        if "load" not in df.columns:
            raise ValueError(f"负荷 CSV 缺少 load 列: {load_csv_path}")

        df["date"] = pd.to_datetime(df["date"])
        # 按时间进行硬编码严格排序，防止读取时出现跳变干扰滑窗
        df = df.sort_values("date").reset_index(drop=True)
        df["load"] = df["load"].astype(np.float32)
        return df

    def _compute_channel_stats(
        self, weather_store: HDF5WeatherSequenceStore, frame_count: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        批处理计算所有历史用于建库的高清晰度气象图像的单通道像素级全局均值和方差。

        Args:
            weather_store: 数据存放读入句柄。
            frame_count: 用于计算这前多少帧。

        Returns:
            Tuple[np.ndarray, np.ndarray]: 形状为 (1, C, 1, 1) 的全局均值和标准差矩阵。
        """
        if frame_count <= 0:
            raise ValueError("frame_count 必须为正整数。")

        channel_count = int(weather_store.frame_shape[0])
        # 使用 float64 进行累加以抵御巨量数字之下的精度溢出
        total_sum = np.zeros((1, channel_count, 1, 1), dtype=np.float64)
        total_sq_sum = np.zeros((1, channel_count, 1, 1), dtype=np.float64)
        total_pixels = 0

        batch_frames = max(self.build_batch_size, 256)
        total_batches = math.ceil(frame_count / batch_frames)

        for batch_idx, start in enumerate(range(0, frame_count, batch_frames), start=1):
            end = min(frame_count, start + batch_frames)
            frames = weather_store.get_block(start, end)
            for channel in self.log1p_channels:
                frames[:, channel, :, :] = np.log1p(
                    np.clip(frames[:, channel, :, :], a_min=0.0, a_max=None)
                )

            # 沿所有空间 H 和 W 以及本批的时间帧 T 累加
            total_sum += frames.sum(axis=(0, 2, 3), keepdims=True, dtype=np.float64)
            total_sq_sum += np.square(frames, dtype=np.float64).sum(
                axis=(0, 2, 3), keepdims=True, dtype=np.float64
            )
            # 记录累积到底除以多少个像素点
            total_pixels += frames.shape[0] * frames.shape[2] * frames.shape[3]

            if batch_idx == 1 or batch_idx == total_batches or batch_idx % max(1, total_batches // 8) == 0:
                print(
                    f"[通道统计] 批次 {batch_idx}/{total_batches} "
                    f"frames={start}:{end}"
                )

        # 运用 E(X^2) - [E(X)]^2 的性质反推出方差，并加以 eps 防止其坍塌到负数边界（由于浮点误差）
        mean = (total_sum / total_pixels).astype(np.float32)
        variance = (total_sq_sum / total_pixels) - np.square(mean, dtype=np.float64)
        variance = np.maximum(variance, 1e-8)
        std = np.sqrt(variance).astype(np.float32)
        return mean, std

    def _iter_raw_feature_batches(
        self,
        weather_store: HDF5WeatherSequenceStore,
        encoder: StatisticalWeatherEncoder320,
        n_windows: int,
        stage_name: str,
    ) -> Iterator[np.ndarray]:
        """
        一个生成器，用于遍历指定数量的训练窗口，分块产生尚未经 PCA 压缩的时空描述向量。
        """
        total_batches = math.ceil(n_windows / self.build_batch_size)
        for batch_idx, start in enumerate(range(0, n_windows, self.build_batch_size), start=1):
            current_size = min(self.build_batch_size, n_windows - start)
            # 对于给定的窗口数量 N_win，需要读取 N_win + pred_len - 1 帧
            end = start + current_size + self.pred_len - 1
            frames = weather_store.get_block(start, end)
            raw = encoder.extract_raw_features_from_block(frames, expected_windows=current_size)

            if batch_idx == 1 or batch_idx == total_batches or batch_idx % max(1, total_batches // 8) == 0:
                print(
                    f"[{stage_name}] 批次 {batch_idx}/{total_batches} "
                    f"windows={start}:{start + current_size - 1}"
                )
            # 通过 yield 挂起，供调用方消费特征，节约内存
            yield raw

    def _iter_weather_embedding_batches(
        self,
        weather_store: HDF5WeatherSequenceStore,
        encoder: ConvLSTMAEWeatherEncoder,
        n_windows: int,
        stage_name: str,
    ) -> Iterator[np.ndarray]:
        """
        分批产生经过 ConvLSTM 自编码器嵌入后的特征向量流。
        """
        # 兼容性处理：部分环境下由于编码问题可能需要清洗阶段名称中的非 ASCII 字符（如在原始终端导出日志时）
        safe_stage_name = stage_name.encode("ascii", errors="ignore").decode("ascii").strip() or "weather_batch"
        total_batches = math.ceil(n_windows / self.build_batch_size)
        for batch_idx, start in enumerate(range(0, n_windows, self.build_batch_size), start=1):
            current_size = min(self.build_batch_size, n_windows - start)
            # 计算当前窗口组所需的总帧数范围
            end = start + current_size + self.pred_len - 1
            frames = weather_store.get_block(start, end)
            # 使用神经网络执行前向传播生成潜向量
            batch_vectors = encoder.transform_frame_block(frames, expected_windows=current_size)

            if batch_idx == 1 or batch_idx == total_batches or batch_idx % max(1, total_batches // 8) == 0:
                print(
                    f"[{stage_name}] 批次 {batch_idx}/{total_batches} "
                    f"windows={start}:{start + current_size - 1}"
                )
            yield batch_vectors

    def _build_time_vectors(self, timestamps: Sequence[pd.Timestamp]) -> np.ndarray:
        """从对应时间序列中提取周期性时间特征 embedding（例如：月、日、时等）"""
        timestamps = pd.DatetimeIndex(pd.to_datetime(timestamps))
        # time_features 返回尺寸为 [C, N] 的时间特征，随后转置成 [N, C]
        return time_features(timestamps, freq=self.freq).T.astype(np.float32)

    def _fuse_and_normalize(self, weather_vectors: np.ndarray, time_vectors: np.ndarray) -> np.ndarray:
        """
        核心的多模态融合计算环节：
        分别对气象向量做 L2 归一化；对时间特征乘以时间权重参数；
        拼接后再次对总体执行 L2 归一化。这保障了检索特征处于同一量级，并且可以使用高效的 faiss.IndexFlatIP。
        """
        weather_vectors = l2_normalize(np.asarray(weather_vectors, dtype=np.float32))
        time_vectors = np.asarray(time_vectors, dtype=np.float32)
        if time_vectors.shape[0] != weather_vectors.shape[0]:
            raise ValueError(
                f"气象向量与时间向量样本数不一致: {weather_vectors.shape[0]} vs {time_vectors.shape[0]}"
            )
        # 放大时间特征的权重以增强按相似节假日/相似周期检索的属性
        weighted_time = time_vectors * self.time_weight
        # 将归一化的气象表征和加权的时间特征在特征维度 (Axis=1) 水平拼接
        fused = np.hstack([weather_vectors, weighted_time]).astype(np.float32, copy=False)
        # 最后对混合的描述向量做 L2 归一化，以便于后续支持精确内积相似度
        return l2_normalize(fused)

    def build(
        self,
        load_csv_path: Union[str, Path] = DEFAULT_LOAD_CSV,
        weather_h5_path: Union[str, Path] = DEFAULT_WEATHER_H5,
        artifact_dir: Optional[Union[str, Path]] = None,
        max_train_windows: Optional[int] = None,
    ) -> "SimilarDayRetriever":
        """
        开始执行建库全流程。
        读取数据、切分训练数据、拟合编码器、压缩特征、融合构建索引并保存。
        """
        t0 = time.time()
        # 加载全量或目标电负荷序列 CSV 并解析日期
        load_df = self._load_load_dataframe(load_csv_path)
        # 打开气象数据集句柄
        weather_store = HDF5WeatherSequenceStore(weather_h5_path)

        total_rows = len(load_df)
        default_train_rows = int(total_rows * self.train_ratio)
        # 根据给定的预测长度推算可以产生的监督滑动窗口总数
        n_windows = default_train_rows - self.pred_len + 1
        if n_windows <= 0:
            raise ValueError(
                f"训练窗口数无效: total_rows={total_rows}, pred_len={self.pred_len}, train_ratio={self.train_ratio}"
            )
        # 在 Smoke Test 或 DEBUG 下可能会强行截断部分不训练
        if max_train_windows is not None:
            n_windows = min(int(max_train_windows), n_windows)
        if n_windows <= 0:
            raise ValueError("max_train_windows 设置后训练窗口数不大于 0。")

        # 验证气象和负荷在起跑线上是对齐的
        train_frame_count = n_windows + self.pred_len - 1
        weather_store.verify_alignment(load_df["date"].iloc[:train_frame_count])
        # 准备对应标签侧的负荷滑动窗口、起始时间戳列表
        start_timestamps = pd.DatetimeIndex(load_df["date"].iloc[:n_windows])
        load_values = load_df["load"].to_numpy(dtype=np.float32)
        # 使用 numpy 高效滑窗构建 ground truth 的历史曲线
        load_curves = sliding_window_view(load_values[:train_frame_count], self.pred_len)[:n_windows]
        load_curves = np.ascontiguousarray(load_curves.astype(np.float32))

        print("=" * 72)
        print("开始构建时空相似日检索库 (ConvLSTM-AE)")
        print(f"负荷数据: {Path(load_csv_path).resolve()}")
        print(f"气象数据: {Path(weather_h5_path).resolve()}")
        print(f"训练比例: {self.train_ratio:.6f}")
        print(f"训练帧数: {train_frame_count}")
        print(f"训练窗口数: {n_windows}")
        print(f"窗口长度: {self.pred_len}")
        print(f"时间权重 alpha: {self.time_weight}")
        print(f"气象统计量: {self.stats}")
        print("=" * 72)

        # 全局扫描计算历史训练区间内气象每通道像素方差分布
        # 实例化编码器核心
        encoder = ConvLSTMAEWeatherEncoder(
            checkpoint_path=self.ae_checkpoint_path,
            norm_stats_path=self.ae_norm_stats_path,
            latent_dim=self.weather_dim,
            window_size=self.pred_len,
            log1p_channels=self.log1p_channels,
            batch_size=self.encoder_batch_size,
        )
        if self.weather_dim != AE_LATENT_DIM:
            raise ValueError(
                f"weather_dim must match the trained ConvLSTM-AE latent dim {AE_LATENT_DIM}, "
                f"got {self.weather_dim}"
            )

        # 针对神经网络编码器（ConvLSTM-AE），其权重已在训练阶段固化，此处 fit 仅为保持接口一致性。
        # 如果切换回 StatisticalWeatherEncoder，则会在此处进行真实的 PCA 拟合。
        encoder.fit(
            raw_feature_batches_factory=lambda: self._iter_raw_feature_batches(
                weather_store=weather_store,
                encoder=encoder,
                n_windows=n_windows,
                stage_name="检查编码器状态",
            ),
            total_samples=n_windows,
        )

        # 此后分配底库潜向量内存（仅[N, Latent_Dim]大小所以可以直接全部放进内存）
        weather_vectors = np.empty((n_windows, encoder.output_dim), dtype=np.float32)
        cursor = 0
        # 第散遍扫描：产生用于建库的最终被降维表征
        for batch_vectors in self._iter_weather_embedding_batches(
            weather_store=weather_store,
            encoder=encoder,
            n_windows=n_windows,
            stage_name="向量生成",
        ):
            weather_vectors[cursor : cursor + len(batch_vectors)] = batch_vectors
            cursor += len(batch_vectors)

        # 组建并融合时间特征得到最终在向量空间待检索的“指纹”
        time_vectors = self._build_time_vectors(start_timestamps)
        fused_vectors = self._fuse_and_normalize(weather_vectors, time_vectors)
        
        # 装载底库入 IndexFlatIP / Numpy Backend
        index = ExactInnerProductIndex(fused_vectors.shape[1])
        index.add(fused_vectors)

        # 将产生的必要状态保存至类的实例属性中
        self.weather_encoder = encoder
        self.index = index
        self.base_vectors = fused_vectors
        self.load_curves = load_curves
        self.start_timestamps = start_timestamps
        self.load_csv_path = str(Path(load_csv_path).resolve())
        self.weather_h5_path = str(Path(weather_h5_path).resolve())
        self.train_frame_count = int(train_frame_count)
        self.train_window_count = int(n_windows)
        self.fused_dim = int(fused_vectors.shape[1])
        self.log1p_channels = tuple(self.weather_encoder.log1p_channels)

        weather_store.close()

        # 持久化输出
        if artifact_dir is not None:
            self.save(artifact_dir)

        print(
            f"[完成] 建库结束，用时 {time.time() - t0:.1f}s，"
            f"索引向量数={self.index.ntotal}，最终向量维度={self.fused_dim}"
        )
        return self

    def search_by_weather_embedding(
        self,
        weather_embedding: np.ndarray,
        query_timestamp: Union[str, pd.Timestamp],
        top_k: int = 3,
    ) -> RetrievalResult:
        """
        [底层接口] 输入已压缩好的潜向量表征，拼接时间特征后打分找相似日。

        Args:
            weather_embedding: 编码器产生的气象密集表示，尺寸应为 (weather_dim,)。
            query_timestamp: 预测发生的时间锚点，用于提取月、日、小时等周期规律进行加权。
            top_k: 设定返回分数最高的多少条历史片段。

        Returns:
            RetrievalResult: 封装了相似片段负荷序列以及评估得分的结果对象。
        """
        self._ensure_built()
        weather_embedding = np.asarray(weather_embedding, dtype=np.float32).reshape(1, -1)
        if weather_embedding.shape[1] != self.weather_encoder.output_dim:
            raise ValueError(
                f"气象编码维度错误，期望 {self.weather_encoder.output_dim}，实际 {weather_embedding.shape[1]}"
            )

        # 构建当前查询点的时间描述符并加权缩放
        time_vector = self._build_time_vectors([pd.Timestamp(query_timestamp)])
        # 合入气象表征
        fused_query = self._fuse_and_normalize(weather_embedding, time_vector)
        # 抛给高层检索器做精确比对（或者 faiss.IndexFlatIP）
        scores, ids = self.index.search(fused_query, top_k=top_k)

        # 从检索库元数据结构中通过 ID 直接拷贝切出匹配的负荷序列切片
        retrieved_ids = [int(idx) for idx in ids[0].tolist() if int(idx) >= 0]
        retrieved_scores = [float(s) for s in scores[0][: len(retrieved_ids)].tolist()]
        retrieved_loads = np.ascontiguousarray(self.load_curves[retrieved_ids])
        retrieved_times = [
            str(pd.Timestamp(ts)) for ts in self.start_timestamps[retrieved_ids]
        ]

        return RetrievalResult(
            query_timestamp=str(pd.Timestamp(query_timestamp)),
            historical_indices=retrieved_ids,
            historical_timestamps=retrieved_times,
            similarity_scores=retrieved_scores,
            load_curves=retrieved_loads,
        )

    def search_by_weather_window(
        self,
        weather_window: np.ndarray,
        query_timestamp: Union[str, pd.Timestamp],
        top_k: int = 3,
    ) -> RetrievalResult:
        """
        [中层接口] 直接受理时序未编码的 [pred_len, C, H, W] 的气象高维张量。将其编码降维后调度检索。
        常用于流式在线任务。
        """
        self._ensure_built()
        weather_window = np.asarray(weather_window, dtype=np.float32)
        if weather_window.ndim != 4 or weather_window.shape[0] != self.pred_len:
            raise ValueError(f"查询气象窗口必须为 [{self.pred_len}, C, H, W]，实际 {weather_window.shape}")
        if weather_window.shape[1] != self.weather_encoder.channel_count:
            raise ValueError(
                f"气象通道数错误，期望 {self.weather_encoder.channel_count}，实际 {weather_window.shape[1]}"
            )

        weather_embedding = self.weather_encoder.transform_window(weather_window)
        return self.search_by_weather_embedding(
            weather_embedding[0],
            query_timestamp=query_timestamp,
            top_k=top_k,
        )

    def search_by_timestamp(
        self,
        query_timestamp: Union[str, pd.Timestamp],
        top_k: int = 3,
        weather_store: Optional[HDF5WeatherSequenceStore] = None,
    ) -> RetrievalResult:
        """
        [高层接口] 用户仅需输入待查询时间的锚点。内部根据日期去气象库抽帧加载，进行相似性检索。
        常被测算验证环节(Backtest)使用。
        """
        self._ensure_built()
        created_store = False
        if weather_store is None:
            if self.weather_h5_path is None:
                raise RuntimeError("当前检索器没有保存 weather_h5_path，无法按时间戳取气象窗口。")
            weather_store = HDF5WeatherSequenceStore(self.weather_h5_path)
            # 是否在内层开启的标志
            created_store = True

        try:
            # 去气象数据表上反查得到图像序列
            weather_window = weather_store.get_window_by_timestamp(query_timestamp, self.pred_len)
            return self.search_by_weather_window(
                weather_window,
                query_timestamp=query_timestamp,
                top_k=top_k,
            )
        finally:
            if created_store:
                weather_store.close()

    def search_from_future_csv(
        self,
        future_csv_path: Union[str, Path] = DEFAULT_FUTURE_CSV,
        top_k: int = 3,
        weather_store: Optional[HDF5WeatherSequenceStore] = None,
    ) -> RetrievalResult:
        """
        [实用工具接口] 从给定的"未来数据表"中找到第一行包含的预测起点时间，基于此时间完成一键检索。
        """
        future_csv_path = Path(future_csv_path).resolve()
        if not future_csv_path.exists():
            raise FileNotFoundError(f"未找到未来负荷 CSV 文件: {future_csv_path}")
        future_df = pd.read_csv(future_csv_path)
        if "date" not in future_df.columns or len(future_df) == 0:
            raise ValueError(f"未来负荷 CSV 缺少 date 列或为空: {future_csv_path}")
        # 解析预测 CSV 表第一行即为要求搜索的时间戳起点
        query_timestamp = pd.Timestamp(future_df["date"].iloc[0])
        return self.search_by_timestamp(
            query_timestamp=query_timestamp,
            top_k=top_k,
            weather_store=weather_store,
        )

    def save(self, artifact_dir: Union[str, Path]) -> None:
        """
        持久化当前构建完毕的模型，方便在推理时极速加载 (免去算PCA的延误)。
        共分为三个产物文件：检索表数组(npz)、模型实例(pkl)、以及查询元配置(json)。
        """
        self._ensure_built()
        artifact_dir = Path(artifact_dir).resolve()
        artifact_dir.mkdir(parents=True, exist_ok=True)

        arrays_path = artifact_dir / "retriever_arrays.npz"
        encoder_path = artifact_dir / "weather_encoder.pkl"
        meta_path = artifact_dir / "metadata.json"

        # numpy 内置高速保存特征及标签池
        np.savez(
            arrays_path,
            base_vectors=self.base_vectors.astype(np.float32),
            load_curves=self.load_curves.astype(np.float32),
            start_timestamps_ns=self.start_timestamps.asi8.astype(np.int64),
        )

        # 序列化含有 StandardScaler 和 IncrementalPCA 内置学得权重的编码器实例
        with open(encoder_path, "wb") as f:
            pickle.dump(self.weather_encoder, f)

        # 记录关键超参等上下文信息
        metadata = {
            "weather_dim_requested": self.weather_dim,
            "weather_dim_effective": self.weather_encoder.output_dim,
            "time_weight": self.time_weight,
            "pred_len": self.pred_len,
            "train_ratio": self.train_ratio,
            "freq": self.freq,
            "build_batch_size": self.build_batch_size,
            "ae_checkpoint_path": self.ae_checkpoint_path,
            "ae_norm_stats_path": self.ae_norm_stats_path,
            "encoder_batch_size": self.encoder_batch_size,
            "stats": list(self.stats),
            "log1p_channels": list(self.log1p_channels),
            "load_csv_path": self.load_csv_path,
            "weather_h5_path": self.weather_h5_path,
            "train_frame_count": self.train_frame_count,
            "train_window_count": self.train_window_count,
            "fused_dim": self.fused_dim,
            "index_backend": self.index.backend,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"[保存] 检索库已写入: {artifact_dir}")

    @classmethod
    def load(cls, artifact_dir: Union[str, Path]) -> "SimilarDayRetriever320":
        """
        类方法，反向加载并初始化系统。从之前 `save(...)` 创建的存档目录中复苏全部模型状态结构和底库资源。
        """
        artifact_dir = Path(artifact_dir).resolve()
        arrays_path = artifact_dir / "retriever_arrays.npz"
        encoder_path = artifact_dir / "weather_encoder.pkl"
        meta_path = artifact_dir / "metadata.json"

        if not arrays_path.exists() or not encoder_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"检索库目录不完整: {artifact_dir}")

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        retriever = cls(
            weather_dim=int(metadata["weather_dim_requested"]),
            time_weight=float(metadata["time_weight"]),
            pred_len=int(metadata["pred_len"]),
            train_ratio=float(metadata["train_ratio"]),
            freq=str(metadata["freq"]),
            build_batch_size=int(metadata["build_batch_size"]),
            ae_checkpoint_path=metadata.get("ae_checkpoint_path", DEFAULT_AE_CHECKPOINT),
            ae_norm_stats_path=metadata.get("ae_norm_stats_path", DEFAULT_AE_NORM_STATS),
            encoder_batch_size=int(metadata.get("encoder_batch_size", AE_BATCH_SIZE)),
            stats=metadata["stats"],
            log1p_channels=metadata["log1p_channels"],
        )

        with open(encoder_path, "rb") as f:
            retriever.weather_encoder = pickle.load(f)

        arrays = np.load(arrays_path)
        retriever.base_vectors = arrays["base_vectors"].astype(np.float32)
        retriever.load_curves = arrays["load_curves"].astype(np.float32)
        retriever.start_timestamps = pd.to_datetime(arrays["start_timestamps_ns"].astype(np.int64))

        # 挂载并填充查询接口引擎
        retriever.index = ExactInnerProductIndex(retriever.base_vectors.shape[1])
        retriever.index.add(retriever.base_vectors)

        retriever.load_csv_path = metadata.get("load_csv_path")
        retriever.weather_h5_path = metadata.get("weather_h5_path")
        retriever.train_frame_count = metadata.get("train_frame_count")
        retriever.train_window_count = metadata.get("train_window_count")
        retriever.fused_dim = metadata.get("fused_dim")
        return retriever


def print_retrieval_result(result: RetrievalResult, print_json: bool = False) -> None:
    """
    可视化打印检索结果。
    
    参数:
        result: 检索结果对象。
        print_json: 是否以 JSON 格式输出（方便程序间对接）。
    """
    if print_json:
        # JSON 序列化，确保中文不乱码。
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return

    print("=" * 72)
    print(f"查询时间点 (Query): {result.query_timestamp}")
    # 遍历 Top-K 结果进行排版打印
    for rank, (idx, ts, score, curve) in enumerate(
        zip(
            result.historical_indices,
            result.historical_timestamps,
            result.similarity_scores,
            result.load_curves,
        ),
        start=1,
    ):
        # 仅打印负荷曲线的前 8 个点作为缩略展示
        head_values = ", ".join(f"{float(x):.4f}" for x in curve[:8])
        print(
            f"  [排名 {rank}] 索引: {idx:<6} | 起始时间: {ts} | 相似度: {score:.6f}"
        )
        print(f"    负荷曲线预览: [{head_values} ...]")
    print("=" * 72)


def add_build_arguments(parser: argparse.ArgumentParser) -> None:
    """向命令行解析器添加共享的建库配置及运行参数。"""
    parser.add_argument("--load-csv", type=str, default=str(DEFAULT_LOAD_CSV))
    parser.add_argument("--weather-h5", type=str, default=str(DEFAULT_WEATHER_H5))
    parser.add_argument("--artifact-dir", type=str, default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--weather-dim", type=int, default=AE_LATENT_DIM)
    parser.add_argument("--time-weight", type=float, default=2.0)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--train-ratio", type=float, default=2.0 / 3.0)
    parser.add_argument("--build-batch-size", type=int, default=384)
    parser.add_argument("--ae-checkpoint", type=str, default=str(DEFAULT_AE_CHECKPOINT))
    parser.add_argument("--ae-norm-stats", type=str, default=str(DEFAULT_AE_NORM_STATS))
    parser.add_argument("--encoder-batch-size", type=int, default=AE_BATCH_SIZE)
    parser.add_argument("--max-train-windows", type=int, default=None)


def resolve_query_timestamp(args: argparse.Namespace) -> pd.Timestamp:
    """智能推断查询时间点：优先读取显式参数，其次读取未来测试集期初，否则回退使用训练集期末。"""
    if getattr(args, "query_start", None):
        return pd.Timestamp(args.query_start)

    future_csv = Path(args.future_csv).resolve()
    if future_csv.exists():
        future_df = pd.read_csv(future_csv)
        if "date" in future_df.columns and len(future_df) > 0:
            return pd.Timestamp(future_df["date"].iloc[0])

    load_df = pd.read_csv(args.load_csv)
    load_df["date"] = pd.to_datetime(load_df["date"])
    train_rows = int(len(load_df) * float(args.train_ratio))
    if train_rows >= len(load_df):
        train_rows = len(load_df) - 1
    return pd.Timestamp(load_df["date"].iloc[train_rows])


def command_build(args: argparse.Namespace) -> None:
    """CLI 命令处理函数：触发并执行离线库构建流程。"""
    retriever = SimilarDayRetriever(
        weather_dim=args.weather_dim,
        time_weight=args.time_weight,
        pred_len=args.pred_len,
        train_ratio=args.train_ratio,
        build_batch_size=args.build_batch_size,
        ae_checkpoint_path=args.ae_checkpoint,
        ae_norm_stats_path=args.ae_norm_stats,
        encoder_batch_size=args.encoder_batch_size,
    )
    retriever.build(
        load_csv_path=args.load_csv,
        weather_h5_path=args.weather_h5,
        artifact_dir=args.artifact_dir,
        max_train_windows=args.max_train_windows,
    )


def command_query(args: argparse.Namespace) -> None:
    """CLI 命令处理函数：加载检索模型与离线相似特征库库并执行单次线上检索预测。"""
    retriever = SimilarDayRetriever.load(args.artifact_dir)
    query_timestamp = resolve_query_timestamp(args)
    weather_store = HDF5WeatherSequenceStore(args.weather_h5)
    try:
        result = retriever.search_by_timestamp(
            query_timestamp=query_timestamp,
            top_k=args.top_k,
            weather_store=weather_store,
        )
    finally:
        weather_store.close()
    print_retrieval_result(result, print_json=args.print_json)


def command_smoke_test(args: argparse.Namespace) -> None:
    """CLI 命令处理函数：在极小规模数据上串联执行构建及查询以校验整个检索系统运行畅通无阻。"""
    retriever = SimilarDayRetriever320(
        weather_dim=args.weather_dim,
        time_weight=args.time_weight,
        pred_len=args.pred_len,
        train_ratio=args.train_ratio,
        build_batch_size=args.build_batch_size,
        ae_checkpoint_path=args.ae_checkpoint,
        ae_norm_stats_path=args.ae_norm_stats,
        encoder_batch_size=args.encoder_batch_size,
    )
    retriever.build(
        load_csv_path=args.load_csv,
        weather_h5_path=args.weather_h5,
        artifact_dir=args.artifact_dir,
        max_train_windows=args.max_train_windows,
    )

    query_timestamp = resolve_query_timestamp(args)
    weather_store = HDF5WeatherSequenceStore(args.weather_h5)
    try:
        result = retriever.search_by_timestamp(
            query_timestamp=query_timestamp,
            top_k=args.top_k,
            weather_store=weather_store,
        )
    finally:
        weather_store.close()

    print("[Smoke Test] Top-K 检索已返回结果。")
    print_retrieval_result(result, print_json=args.print_json)


def build_parser() -> argparse.ArgumentParser:
    """构造整个脚本支持的包含多层级子命令的主命令行参数解析应用。"""
    parser = argparse.ArgumentParser(description="气象潜向量嵌入相似日检索系统")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_build = subparsers.add_parser("build", help="构建并保存离线相似日检索库")
    add_build_arguments(parser_build)

    parser_query = subparsers.add_parser("query", help="加载离线库并执行在线检索")
    parser_query.add_argument("--artifact-dir", type=str, default=str(DEFAULT_ARTIFACT_DIR))
    parser_query.add_argument("--weather-h5", type=str, default=str(DEFAULT_WEATHER_H5))
    parser_query.add_argument("--future-csv", type=str, default=str(DEFAULT_FUTURE_CSV))
    parser_query.add_argument("--load-csv", type=str, default=str(DEFAULT_LOAD_CSV))
    parser_query.add_argument("--train-ratio", type=float, default=2.0 / 3.0)
    parser_query.add_argument("--query-start", type=str, default=None)
    parser_query.add_argument("--top-k", type=int, default=3)
    parser_query.add_argument("--print-json", action="store_true")

    parser_smoke = subparsers.add_parser("smoke-test", help="小规模建库并验证检索流程")
    add_build_arguments(parser_smoke)
    parser_smoke.add_argument("--future-csv", type=str, default=str(DEFAULT_FUTURE_CSV))
    parser_smoke.add_argument("--query-start", type=str, default=None)
    parser_smoke.add_argument("--top-k", type=int, default=3)
    parser_smoke.add_argument("--print-json", action="store_true")
    parser_smoke.set_defaults(max_train_windows=384)

    return parser


def main() -> None:
    """程序入口：根据命令行输入的子命令执行相应逻辑。"""
    parser = build_parser()
    args = parser.parse_args()

    # 命令分发
    if args.command == "build":
        # 离线建库模式
        command_build(args)
    elif args.command == "query":
        # 在线查询模式
        command_query(args)
    elif args.command == "smoke-test":
        # 冒烟测试模式（快速验证）
        command_smoke_test(args)
    else:
        # 异常情况处理
        raise ValueError(f"未知子命令: {args.command}")


if __name__ == "__main__":
    # 执行主函数
    main()
