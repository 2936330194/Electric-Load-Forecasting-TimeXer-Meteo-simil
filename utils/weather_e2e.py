"""
weather_e2e.py - 气象特征与负荷端到端预测的数据处理与模型封装模块

该模块负责处理高维 4D 气象网格数据 (时间, 通道, 高度, 宽度)，将其与历史负荷数据对齐，
并提供模型（FullMapConvTimeXerQuantile）将气象图像特征提取和 TimeXer 时间序列预测进行结合，
实现了端到端的概率负荷预测。同时也包含用于大幅优化气象数据内存传输的去重对齐批处理工具。
"""

import os
import re
import time
import h5py
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Sampler

from models.TimeXer import Model as TimeXer
from utils.timefeatures import time_features


def _build_similar_day_prior_features(
    load_curves: np.ndarray,
    similarity_scores: Sequence[float],
    pred_len: int,
    top_k: int,
    shift_steps: int = 0,
) -> np.ndarray:
    """
    将检索得到的 Top-K 相似日曲线整理为预测头可直接融合的先验特征。

    该函数的主要逻辑包括：
    1. 对输入的相似日负荷曲线进行对齐处理（补齐或截断），确保长度符合预测长度 `pred_len`。
    2. 根据 `shift_steps` 对曲线进行循环移位，以适配可能的预测起始偏移（如 15min 步长的对齐）。
    3. 利用相似度评分通过 Softmax 归一化计算权重，合成一条“加权均值”先验曲线。
    4. 将加权先验曲线与 Top-K 各单条原始曲线拼接，形成最终的先验特征矩阵。

    参数:
        load_curves (np.ndarray): 检索到的相似日负荷曲线，形状通常为 [K, L] 或 [L,]。
        similarity_scores (Sequence[float]): 对应的相似度评分，用于计算 Softmax 融合权重。
        pred_len (int): 预测长度，输出特征的时间轴长度需与之对齐。
        top_k (int): 最终保留的相似日数量（特征矩阵的列数相关）。
        shift_steps (int, optional): 时刻偏移步数，用于循环对齐。默认 0。

    返回:
        np.ndarray: 先验特征矩阵，形状为 [pred_len, top_k + 1]。
                    - 第 0 列: 基于相似度 Softmax 加权后的融合均值先验。
                    - 第 1 到 K 列: Top-K 各单条经过对齐和移位后的相似日曲线。
    """
    # 强制转换参数类型并处理位移步数（防止超出预测长度范围）
    pred_len = int(pred_len)
    top_k = int(top_k)
    shift_steps = int(shift_steps) % max(1, pred_len)

    # 初始化存储容器：矩阵 [top_k, pred_len] 存储各曲线，向量 [pred_len] 存储加权均值
    prior_curves = np.zeros((top_k, pred_len), dtype=np.float32)
    weighted_prior = np.zeros((pred_len,), dtype=np.float32)

    # 规范化输入曲线形状，确保为二维矩阵
    curves = np.asarray(load_curves, dtype=np.float32)
    if curves.ndim == 1:
        curves = curves.reshape(1, -1)

    if curves.ndim == 2 and curves.size > 0:
        # 确定实际可供处理的相似日数量
        usable = min(top_k, curves.shape[0])
        
        # 遍历每一条相似日曲线进行长度对齐和循环移位
        for idx in range(usable):
            curve = curves[idx]
            # 长度处理：不足则补零，超出则截断到预测长度 pred_len
            if curve.shape[0] < pred_len:
                padded = np.zeros((pred_len,), dtype=np.float32)
                padded[: curve.shape[0]] = curve
                curve = padded
            else:
                curve = curve[:pred_len]
            
            # 使用 np.roll 实现时间轴的循环对齐，-shift_steps 表示向左（过去）偏移
            prior_curves[idx] = np.roll(curve.astype(np.float32, copy=False), -shift_steps)

        # 基于相似度分数计算 Softmax 权重，用于融合 Top-K 曲线
        scores = np.asarray(similarity_scores[:usable], dtype=np.float32)
        if scores.size > 0:
            # 减去最大值以维持数值稳定性（预防 exp 溢出）
            scores = scores - np.max(scores)
            weights = np.exp(scores).astype(np.float32, copy=False)
            weight_sum = float(np.sum(weights))
            
            if weight_sum > 0:
                # 归一化得到权重，并对相似日曲线进行加权求和
                weights = weights / weight_sum
                weighted_prior = np.sum(
                    prior_curves[:usable] * weights[:, None],
                    axis=0,
                    dtype=np.float32,
                )

    # 最终输出：[加权先验(1列), 原始曲线矩阵转置(K列)] 在维度 1 拼接
    # 生成形状为 [pred_len, top_k + 1] 的特征张量
    return np.concatenate(
        [weighted_prior[:, None], prior_curves.transpose(1, 0)],
        axis=1,
    ).astype(np.float32, copy=False)


def _require_weather_runtime() -> None:
    # 将 h5py 的硬依赖检查延后到运行期，避免仅导入本模块时就失败。
    if h5py is None:
        raise ImportError("缺少气象数据加载库h5py，请先安装：pip install h5py")


def _guess_year_start_from_path(file_path: str) -> pd.Timestamp:
    # 如果用户没有显式提供起始时间，则尝试从文件名中提取年份作为兜底。
    name = Path(file_path).stem
    match = re.search(r"(20\d{2})", name)
    if match:
        return pd.Timestamp(f"{match.group(1)}-01-01 00:00:00")
    return pd.Timestamp("2025-01-01 00:00:00")


def _find_first_4d_dataset(h5_obj):
    # 在 HDF5 树结构里递归寻找第一个 4D 数据集，
    # 约定其组织形式为 [time, channel, height, width]。
    for key in h5_obj.keys():
        item = h5_obj[key]
        if isinstance(item, h5py.Dataset) and item.ndim == 4:
            return item
        if isinstance(item, h5py.Group):
            found = _find_first_4d_dataset(item)
            if found is not None:
                return found
    return None


def _infer_weather_freq(n_steps: int, start_time: pd.Timestamp) -> pd.Timedelta:
    """
    推断气象数据的时间频率。
    通过尝试常见的分钟级别（60, 30, 15 分钟），检查数据总步数是否构成了完整的整数天。

    参数:
        n_steps (int): 数据集的总时间步数
        start_time (pd.Timestamp): 数据集的起始时间（可在此保留作扩展使用）

    返回:
        pd.Timedelta: 推断出的时间增量（频率）
    """
    del start_time
    # 当前只尝试常见的小时级/半小时级/15 分钟级频率。
    candidates_minutes = [60, 30, 15]
    for minutes in candidates_minutes:
        steps_per_day = 1440 // minutes
        n_days = n_steps / steps_per_day
        if abs(n_days - round(n_days)) < 0.01 and n_days >= 1:
            return pd.Timedelta(minutes=minutes)
    raise ValueError(
        f"无法从气象数据步数推断频率：n_steps={n_steps}。"
        "请在 weather_h5_specs 中显式提供频率。"
    )


class FullMapWeatherConvExtractor(nn.Module):
    """
    气象栅格数据的全景空间特征提取器。
    
    设计思路：
    气象数据通常以二维网格（如经纬度网格）形式存在。本模块通过一个特殊的 2D 卷积层，
    其卷积核大小（kernel_size）被设置为与输入气象网格的尺寸（Height * Width）完全一致。
    这种“全图卷积”操作能够一次性捕捉整个地理区域内的全局空间相关性，类似于 Vision Transformer 
    中的全局 Patch Embedding，但专门针对固定分辨率的气象物理场进行了简化。

    主要流程：
    1. 输入形状: [Batch, Channels, Height, Width]
    2. 全图卷积: 通过 H*W 大小的卷积核，将空间维度减约为 1x1，同时映射到目标特征通道。
    3. 特征处理: 包含 LayerNorm 归一化、GELU 激活以及 Dropout，增强特征的表达能力和训练稳定性。
    4. 输出形状: [Batch, out_channels]，即每个时刻对应一个紧凑的气象特征向量。
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_height: int,
        kernel_width: int,
        dropout: float = 0.1,
    ):
        """
        初始化全景卷积特征提取器。

        参数:
            in_channels (int): 输入气象数据的特征通道数（如温度、湿度、风速等物理量的个数）。
            out_channels (int): 输出特征向量的维度（即降维后的潜在空间维度）。
            kernel_height (int): 气象网格的垂直分辨率（高度），必须与输入张量严格匹配。
            kernel_width (int): 气象网格的水平分辨率（宽度），必须与输入张量严格匹配。
            dropout (float): 随机丢弃率，用于防止模型过拟合。
        """
        super().__init__()
        self.kernel_height = int(kernel_height)
        self.kernel_width = int(kernel_width)
        self.output_dim = int(out_channels)
        
        # 定义全图卷积层：stride 为 1 且不加 padding，卷积核刚好覆盖整个输入空间。
        # 输出形状将从 [B, C_in, H, W] 变为 [B, C_out, 1, 1]。
        self.full_map_conv = nn.Conv2d(
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            kernel_size=(self.kernel_height, self.kernel_width),
            bias=True,
        )
        
        # 归一化层：在通道维度上进行归一化，有助于时序序列的特征对齐。
        self.norm = nn.LayerNorm(out_channels)
        # 激活函数：使用 GELU 激活，提供平滑的非线性映射。
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行空间维度的主动降维和特征提取。
        
        参数:
            x (torch.Tensor): 气象输入张量，形状为 [Batch, Channels, Height, Width]。
            
        返回:
            torch.Tensor: 空间建模后的 1D 特征序列，形状为 [Batch, out_channels]。
        """
        # 输入维度校验
        if x.ndim != 4:
            raise ValueError(f"气象输入必须是 4D 张量 [B, C, H, W]，但得到形状 {tuple(x.shape)}")
        
        # 空间分辨率检查，确保卷积核大小与输入气象场一致
        if x.shape[-2] != self.kernel_height or x.shape[-1] != self.kernel_width:
            raise ValueError(
                f"输入气象帧分辨率 ({x.shape[-2]}, {x.shape[-1]}) 与 "
                f"预设分辨率 ({self.kernel_height}, {self.kernel_width}) 不匹配。"
            )

        # 1. 物理场全图卷积：通过匹配的分辨率卷积核执行空间聚合。
        # 2. 展平：将形状 [B, out_channels, 1, 1] 压缩为 [B, out_channels]。
        x = self.full_map_conv(x.float()).flatten(1)
        
        # 3. 特征精炼：归一化、激活和正则化。
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        return x


class WeatherGridStore:
    """
    用于高效读取和对齐大规模 4D 气象 HDF5 数据的存储管理器。
    
    设计与功能亮点：
    1. 多源支持：支持管理多个分断的 HDF5 文件源（如按月或按年存储的数据），将其统筹为一个无缝时间序列。
    2. 懒加载 (Lazy Loading)：在初始化阶段仅读取文件 metadata 并构建时间轴，只有在真正抽取数据时才实例化实际文件句柄。
    3. 时间戳对齐与插值：能够根据请求的目标时间序列，在气象文件时间轴上找到确切位置。遇到时间不精确对齐时，执行相邻两帧的时间维线性插值。
    4. 内存友好：支持通过分批 (`start`, `end`) 切片读取避免 OOM (Out of Memory)，并通过按源文件分桶批量读取降低 HDF5 大文件的跳转开销。
    """
    def __init__(
        self,
        h5_specs: Sequence[Tuple],
        expected_in_channels: int,
        fill_value: float = 0.0,
        use_channel_normalization: bool = False,
        log1p_channels: Optional[Sequence[int]] = None,
        normalization_eps: float = 1e-6,
    ):
        """
        初始化 WeatherGridStore。

        参数:
            h5_specs (Sequence[Tuple]): HDF5 气象文件配置规格。其元素为 Tuple，定义了各个文件的元数据：
                                        元素格式: (文件物理路径, 起始时间字符串, 频率字符串)
                                        如 `("/path/to/data.h5", "2024-01-01 00:00:00", "1H")`
            expected_in_channels (int): 每帧期望具有的物理量通道总数。用于健全性检查，确保所有拼接文件具有相同的特征维度。
            fill_value (float): 若查询的时间戳完全超出了现有 HDF5 文件覆盖的时间段时填充的默认回退值。
        """
        # 运行时延迟检查库依赖
        _require_weather_runtime()
        # 记录期望输入的特征通道总数，用于后续文件一致性校验
        self.expected_in_channels = int(expected_in_channels)
        # 初始化 HDF5 规格配置列表（物理路径、起始时间、频率等）
        self.h5_specs = []
        # 是否启用通道级全局标准化（Z-Score Normalization）
        self.use_channel_normalization = bool(use_channel_normalization)
        # 提取并格式化 log1p 通道列表，确保为有序且唯一的整数元组
        log1p_channels = () if log1p_channels is None else log1p_channels
        self.log1p_channels = tuple(sorted({int(ch) for ch in log1p_channels}))
        # 校验 log1p 通道索引是否在合法范围内 [0, expected_in_channels-1]
        for channel in self.log1p_channels:
            if channel < 0 or channel >= self.expected_in_channels:
                raise ValueError(
                    f"log1p channel index out of range: {channel}, "
                    f"expected within [0, {self.expected_in_channels - 1}]"
                )
        # 设置标准化计算时的数值稳定性极小值 epsilon，强制不低于 1e-12
        self.normalization_eps = max(float(normalization_eps), 1e-12)
        # 懒加载初始化：用于存储拟合后的通道均值与标准差
        self.channel_mean: Optional[np.ndarray] = None
        self.channel_std: Optional[np.ndarray] = None
        # 记录参与归一化统计量计算的样本总数
        self._normalization_sample_count = 0
        
        # 遍历规格配置，补充由于缺省可能丢失的开始时间和时间戳步长
        for spec in h5_specs:
            path = os.path.abspath(spec[0])
            start = pd.Timestamp(spec[1]) if spec[1] else _guess_year_start_from_path(spec[0])
            freq = spec[2] if len(spec) > 2 and spec[2] else None
            self.h5_specs.append((path, start, freq))

        self.fill_value = float(fill_value)
        # 存储组装完毕的所有有效源文件的元数据属性池
        self.sources: List[Dict[str, object]] = []
        # 保存第一张成功读取网格的拓扑形态，后续作为一致性参考
        self.frame_shape: Optional[Tuple[int, int, int]] = None
        
        # 运行时句柄缓存池
        self._file_handles: Dict[int, Any] = {}
        self._datasets: Dict[int, Any] = {}
        # 防止重复触发超出时间范围的告警导致日志刷屏
        self._warned_out_of_range = False
        
        # 触发准备阶段：扫描文件以建立全局时间轴元属性字典
        self.prepare()

    def prepare(self) -> None:
        """
        前期预备阶段：扫描所有配置文件。不展开庞大的气象阵列，
        而是搜集数据集名称、通道分辨率形态和完整的时间序列字典。
        """
        if self.sources:
            return

        for h5_path, start_time, explicit_freq in self.h5_specs:
            if not os.path.exists(h5_path):
                print(f"[气象] 未找到文件: {h5_path}")
                continue

            # 使用只读上下文打开文件探查基本属性
            with h5py.File(h5_path, "r") as h5_file:
                dataset = _find_first_4d_dataset(h5_file)
                if dataset is None:
                    raise ValueError(f"No 4D dataset found in {h5_path}")
                if dataset.shape[1] != self.expected_in_channels:
                    raise ValueError(
                        f"{h5_path} channel count mismatch: expected {self.expected_in_channels}, "
                        f"got {dataset.shape[1]}"
                    )

                # 获取数据的步长、通道数、空间高与空间宽
                n_steps, n_channels, height, width = dataset.shape
                dataset_name = dataset.name

            # 时间频率的推断或应用显式频率
            if explicit_freq:
                freq = pd.Timedelta(explicit_freq)
            else:
                freq = _infer_weather_freq(n_steps, start_time)
                
            # 根据起止频次，预先缓存出针对每个源文件的整个时间轴数组。
            # 这使得我们在后续执行跨越式的序列匹配对齐时可极速通过 Numpy 执行查找。
            timestamps = pd.date_range(start=start_time, periods=n_steps, freq=freq)
            
            # 使用纳秒级大整形表示时间以防止精度截断并在查询时最大化提速
            source = {
                "path": h5_path,
                "dataset_name": dataset_name,
                "n_steps": n_steps,
                "timestamps_ns": timestamps.asi8.copy(),
                "start_ns": int(timestamps[0].value),
                "end_ns": int(timestamps[-1].value),
                "freq": freq,
            }
            self.sources.append(source)

            # 初始全局形态校核记录或合法性核准
            if self.frame_shape is None:
                self.frame_shape = (n_channels, height, width)
            elif self.frame_shape != (n_channels, height, width):
                raise ValueError(
                    f"Inconsistent weather frame shape: {self.frame_shape} vs {(n_channels, height, width)}"
                )

            print(f"[气象] {Path(h5_path).name}: 时间步数={n_steps}, 频率={freq}")

        if not self.sources:
            raise FileNotFoundError("No available weather HDF5 file was found.")

        # 对内部持有的文件引用依据他们的真实起点进行排序，确保全局时间连续逻辑
        self.sources.sort(key=lambda x: x["start_ns"])
        start_ts = pd.Timestamp(min(source["start_ns"] for source in self.sources))
        end_ts = pd.Timestamp(max(source["end_ns"] for source in self.sources))
        print(f"[气象] 数据覆盖范围: {start_ts} ~ {end_ts}")

    def _get_dataset(self, source_idx: int):
        """
        惰性提取数据集对象。只有在发生访问碰撞时，才建立系统底层对该 HDF5 的游标连接句柄。
        
        参数:
            source_idx (int): 源文件的索引 ID
            
        返回:
            h5py.Dataset: 具体的数据集引用游标
        """
        if source_idx in self._datasets:
            return self._datasets[source_idx]

        # 延迟打开文件避免过早占用操作系统资源
        source = self.sources[source_idx]
        h5_file = h5py.File(source["path"], "r")
        dataset = h5_file[source["dataset_name"]]
        
        # 将建立好的操作把柄纳入缓存跟踪
        self._file_handles[source_idx] = h5_file
        self._datasets[source_idx] = dataset
        return dataset

    def close(self) -> None:
        """优雅关闭存储管代中的所有物理文件占用句柄"""
        for file_handle in self._file_handles.values():
            try:
                file_handle.close()
            except Exception:
                pass
        self._file_handles.clear()
        self._datasets.clear()

    def __del__(self):
        """实例析构时自我保护清理"""
        self.close()

    def has_fitted_channel_normalization(self) -> bool:
        """
        判断当前 WeatherGridStore 是否已经具备可用的通道级标准化统计量。
        
        若未启用标准化，则默认视为“已就绪”；若已启用，则需确保均值和标准差均已计算。
        """
        if not self.use_channel_normalization:
            return True
        return self.channel_mean is not None and self.channel_std is not None

    def _apply_log1p_transform_inplace(self, frames: np.ndarray) -> np.ndarray:
        """
        对指定长尾分布的气象通道执行 log1p 变换 (ln(1+x))。
        
        该操作通常用于具有长尾分布特征的物理量（如降水量），以减小数值波动范围并使其更接近正态分布。
        操作是原地（In-place）进行的。
        
        参数:
            frames (np.ndarray): 输入气象帧，形状 [N, C, H, W]。
            
        返回:
            np.ndarray: 变换后的气象帧。
        """
        if not self.log1p_channels:
            return frames
        for channel in self.log1p_channels:
            # 限制最小值为 0，防止 log 处理负值带来的数值异常
            frames[:, channel, :, :] = np.log1p(
                np.clip(frames[:, channel, :, :], a_min=0.0, a_max=None)
            )
        return frames

    def _apply_channel_normalization_inplace(self, frames: np.ndarray) -> np.ndarray:
        """
        基于预先拟合的全局统计量对气象帧各通道执行 Z-Score 标准化 (x' = (x - mean) / std)。
        
        操作是原地（In-place）进行的。若未拟合统计量则抛出异常。
        
        参数:
            frames (np.ndarray): 输入气象帧，形状 [N, C, H, W]。
            
        返回:
            np.ndarray: 标准化后的气象帧。
        """
        if not self.use_channel_normalization:
            return frames
        if self.channel_mean is None or self.channel_std is None:
            raise RuntimeError(
                "气象通道标准化统计量尚未拟合。请先调用 fit_channel_normalization_from_dates(...)。"
            )
        # 利用广播机制对所有空间位置执行线性变换
        frames -= self.channel_mean.reshape(1, -1, 1, 1)
        frames /= self.channel_std.reshape(1, -1, 1, 1)
        return frames

    def preprocess_weather_frames(
        self,
        frames: np.ndarray,
        apply_normalization: bool = True,
    ) -> np.ndarray:
        """
        对气象帧执行完整的预处理流水线：
        1. 物理量变换：对指定的长尾通道进行 log1p 处理。
        2. 数值标准化：对各通道执行基于训练集的 Z-Score 全局变换。
        
        该方法确保了推理阶段和训练阶段的数据预处理逻辑严格一致。
        
        参数:
            frames (np.ndarray): 原始气象帧阵列。
            apply_normalization (bool): 是否执行标准化步骤。默认为 True。

        返回:
            np.ndarray: 预处理后的 float32 格式阵列。
        """
        frames = np.asarray(frames, dtype=np.float32)
        if frames.ndim != 4:
            raise ValueError(f"气象帧必须为 4D 张量 [N, C, H, W]，但得到形状 {tuple(frames.shape)}")
        
        # 1. log1p 变换（若配置）
        self._apply_log1p_transform_inplace(frames)
        # 2. Z-Score 标准化（若启用）
        if apply_normalization:
            self._apply_channel_normalization_inplace(frames)
        return frames

    def _accumulate_channel_stats(self, frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        对单批次气象帧进行统计量累加计算（用于增量计算均值和方差）。
        
        计算各通道在当前批次下的：
        - 样本总和 (sum(x))
        - 样本平方和 (sum(x^2))
        - 总元素个数 (count)
        
        参数:
            frames (np.ndarray): 输入原始气象帧。
            
        返回:
            Tuple: (通道级和 [C], 通道级平方和 [C], 元素总数)
        """
        frames = np.asarray(frames, dtype=np.float32)
        if frames.ndim != 4:
            raise ValueError(f"气象帧必须为 4D 张量 [N, C, H, W]，但得到形状 {tuple(frames.shape)}")
        if frames.shape[0] == 0:
            zeros = np.zeros((self.expected_in_channels,), dtype=np.float64)
            return zeros, zeros.copy(), 0

        # 在统计之前先应用必要的非线性变换（如 log1p）
        self._apply_log1p_transform_inplace(frames)
        
        # 计算该批次各维度的统计信息
        channel_sum = frames.sum(axis=(0, 2, 3), dtype=np.float64)
        frames64 = frames.astype(np.float64, copy=False)
        channel_sq_sum = np.sum(frames64 * frames64, axis=(0, 2, 3), dtype=np.float64)
        element_count = int(frames.shape[0] * frames.shape[2] * frames.shape[3])
        
        return channel_sum, channel_sq_sum, element_count

    def _finalize_channel_stats(
        self,
        channel_sum: np.ndarray,
        channel_sq_sum: np.ndarray,
        element_count: int,
        sample_count: int,
        stage_name: str,
        elapsed_sec: float,
    ) -> None:
        """
        根据全局累计的和与平方和，计算最终的均值与标准差。
        
        采用数值稳定的方差计算方针：Var = E[X^2] - (E[X])^2。
        计算结果将直接更新到实例成员变量 `self.channel_mean` 和 `self.channel_std`。
        """
        if element_count <= 0:
            raise RuntimeError("拟合失败：未找到可用的气象元素。")

        # 1. 计算均值
        mean64 = channel_sum / float(element_count)
        # 2. 计算方差并确保非负（由于浮点精度问题可能出现微小负数，在此截断）
        var64 = np.maximum(channel_sq_sum / float(element_count) - mean64 ** 2, self.normalization_eps ** 2)
        # 3. 计算标准差
        std64 = np.sqrt(var64)

        # 写入结果并转换为 float32 节省内存空间
        self.channel_mean = mean64.astype(np.float32)
        self.channel_std = np.maximum(std64, self.normalization_eps).astype(np.float32)
        self._normalization_sample_count = int(sample_count)

        print(
            f"[气象] 通道归一化统计量拟合完成 ({stage_name}): "
            f"样本数={sample_count}, log1p通道={list(self.log1p_channels)}, "
            f"耗时={elapsed_sec:.1f}s"
        )

    def fit_channel_normalization_from_dates(
        self,
        dates: Sequence[pd.Timestamp],
        chunk_size: int = 512,
        stage_name: str = "train dates",
    ) -> None:
        """
        通过遍历指定的日期序列，分块从磁盘 HDF5 文件动态读取并拟合数据统计量。
        
        该方法适用于训练集规模巨大且无法一次性全部载入内存的情况。
        它会依据日期构建对齐计划，分批（chunk）抽取气象画面并在 CPU/内存中完成统计。
        
        参数:
            dates (Sequence[pd.Timestamp]): 用于拟合统计量的日期清单。
            chunk_size (int): 每次迭代加载的画面数量（防止 OOM）。
            stage_name (str): 用于日志打印的阶段标识。
        """
        if not self.use_channel_normalization or self.has_fitted_channel_normalization():
            return
        if self.frame_shape is None:
            raise RuntimeError("气象帧形制（frame_shape）未初始化，无法进行拟合。")

        # 1. 构建抽取执行计划
        alignment = self.build_alignment(dates)
        valid = np.asarray(alignment["valid"], dtype=bool)
        valid_count = int(valid.sum())
        if valid_count <= 0:
            raise RuntimeError("在提供的日期序列中未找到任何有效的气象覆盖点，无法拟合统计量。")

        # 初始化统计累加器
        channel_sum = np.zeros((self.expected_in_channels,), dtype=np.float64)
        channel_sq_sum = np.zeros((self.expected_in_channels,), dtype=np.float64)
        element_count = 0
        chunk_size = max(1, int(chunk_size))

        print(
            f"[气象] 正在分块拟合通道统计量 ({stage_name}): "
            f"目标点数={len(valid)}, 有效点数={valid_count}, 分块大小={chunk_size}"
        )
        t0 = time.time()
        
        # 2. 分块读取与统计循环
        for start in range(0, len(valid), chunk_size):
            end = min(start + chunk_size, len(valid))
            # 仅提取原始未预处理画面
            raw_chunk = self.fetch_raw_frames_from_alignment(alignment, start, end)
            # 过滤无效填充点
            if not valid[start:end].all():
                raw_chunk = raw_chunk[valid[start:end]]
            if raw_chunk.size == 0:
                continue
            
            # 执行本块统计
            chunk_sum, chunk_sq_sum, chunk_count = self._accumulate_channel_stats(raw_chunk)
            channel_sum += chunk_sum
            channel_sq_sum += chunk_sq_sum
            element_count += chunk_count

        # 3. 汇总并生成最终变换参数
        self._finalize_channel_stats(
            channel_sum=channel_sum,
            channel_sq_sum=channel_sq_sum,
            element_count=element_count,
            sample_count=valid_count,
            stage_name=stage_name,
            elapsed_sec=time.time() - t0,
        )

    def fit_channel_normalization_from_frames(
        self,
        raw_frames: np.ndarray,
        chunk_size: int = 512,
        stage_name: str = "train cache",
    ) -> None:
        """
        针对已经缓存在内存中的气象特征矩阵，直接执行统计量拟合。
        
        常用于气象特征已提前全量预加载完成的情境下快速初始化。
        
        参数:
            raw_frames (np.ndarray): 内存中的气象阵列, 形状 [N, C, H, W]。
            chunk_size (int): 统计分块大小。
            stage_name (str): 日志标识。
        """
        if not self.use_channel_normalization or self.has_fitted_channel_normalization():
            return

        raw_frames = np.asarray(raw_frames, dtype=np.float32)
        if raw_frames.ndim != 4:
            raise ValueError(f"气象画面阵列应为 4D，但得到 {tuple(raw_frames.shape)}")

        channel_sum = np.zeros((self.expected_in_channels,), dtype=np.float64)
        channel_sq_sum = np.zeros((self.expected_in_channels,), dtype=np.float64)
        element_count = 0
        chunk_size = max(1, int(chunk_size))

        print(
            f"[气象] 正在从内存缓存拟合统计量 ({stage_name}): "
            f"画面总数={raw_frames.shape[0]}, 块大小={chunk_size}"
        )
        t0 = time.time()
        
        for start in range(0, raw_frames.shape[0], chunk_size):
            end = min(start + chunk_size, raw_frames.shape[0])
            # 克隆并执行计算，避免对原始缓存数据的影响（这里 _accumulate_channel_stats 会进行 inplace log1p）
            chunk = np.array(raw_frames[start:end], dtype=np.float32, copy=True)
            chunk_sum, chunk_sq_sum, chunk_count = self._accumulate_channel_stats(chunk)
            channel_sum += chunk_sum
            channel_sq_sum += chunk_sq_sum
            element_count += chunk_count

        self._finalize_channel_stats(
            channel_sum=channel_sum,
            channel_sq_sum=channel_sq_sum,
            element_count=element_count,
            sample_count=int(raw_frames.shape[0]),
            stage_name=stage_name,
            elapsed_sec=time.time() - t0,
        )

    def build_alignment(self, dates: Sequence[pd.Timestamp]) -> Dict[str, np.ndarray]:
        """
        构建目标时间戳序列与底层 HDF5 文件数据时间轴的映射及插值系数对照表。
        此步骤不产生 I/O 读取开销，仅在内存中针对时间轴构建关联（相当于一个查询执行计划）。

        参数:
            dates (Sequence[pd.Timestamp]): 模型或数据集即将抽取的对应序列真实物理时刻表。

        返回:
            Dict[str, np.ndarray]: 字典含有五个 Numpy 1D 对齐数组描述本次计划：
                - `source_idx`: 时间戳命中的具体源文件序号。
                - `left_idx`: 用于插值的左侧相邻帧（较早时刻）的实际文件内部存储索引。
                - `right_idx`: 用于插值的右侧相邻帧（较晚时刻）的实际文件内部存储索引。
                - `alpha`: 取值 [0.0, 1.0] 的时间距离比重系数。0 表示严格贴近期望左帧。
                - `valid`: 判断对应位点是否有被气象底表承载包围的布尔掩码。
        """
        dates = pd.DatetimeIndex(pd.to_datetime(dates))
        # 统一转成最底层纳秒大整型进行高速扫描比较
        request_ns = dates.asi8.astype(np.int64)
        n = len(request_ns)

        # 初始化记录执行规划动作的容器数组
        source_idx = np.full(n, -1, dtype=np.int32)
        left_idx = np.zeros(n, dtype=np.int32)
        right_idx = np.zeros(n, dtype=np.int32)
        alpha = np.zeros(n, dtype=np.float32)
        valid = np.zeros(n, dtype=bool)

        for idx, source in enumerate(self.sources):
            # 将粗筛选限制在此源文件有效时区涵盖范围内以减少不必要的匹配
            mask = (request_ns >= source["start_ns"]) & (request_ns <= source["end_ns"])
            if not mask.any():
                continue

            ts_ns = source["timestamps_ns"]
            req = request_ns[mask]
            
            # 使用二分搜索快速确认期望时间落在缓存基轴标尺上的确切位段
            pos = np.searchsorted(ts_ns, req, side="left")

            # 处理完美对齐的情况（请求时间刚好等于气象帧的时间点）
            pos_clipped = np.clip(pos, 0, len(ts_ns) - 1)
            exact_mask = ts_ns[pos_clipped] == req

            current_indices = np.where(mask)[0]
            exact_indices = current_indices[exact_mask]
            
            source_idx[exact_indices] = idx
            left_idx[exact_indices] = pos_clipped[exact_mask]
            right_idx[exact_indices] = pos_clipped[exact_mask]
            alpha[exact_indices] = 0.0 # 完美踩点，无需求助右帧施加干预权重
            valid[exact_indices] = True

            # 对未命中的时刻，采用左右两帧之间的时间维线性插值。
            non_exact_indices = current_indices[~exact_mask]
            if len(non_exact_indices) == 0:
                continue

            non_exact_pos = pos[~exact_mask]
            # 决定用于差转调配的两端界碑
            right = np.clip(non_exact_pos, 1, len(ts_ns) - 1)
            left = np.clip(right - 1, 0, len(ts_ns) - 1)
            
            left_ts = ts_ns[left].astype(np.float64)
            right_ts = ts_ns[right].astype(np.float64)
            req_ts = req[~exact_mask].astype(np.float64)
            # alpha 越接近 1，表示请求时刻越靠近右侧参考帧。
            denom = np.maximum(right_ts - left_ts, 1.0)
            alpha_values = ((req_ts - left_ts) / denom).astype(np.float32)

            source_idx[non_exact_indices] = idx
            left_idx[non_exact_indices] = left
            right_idx[non_exact_indices] = right
            alpha[non_exact_indices] = alpha_values
            valid[non_exact_indices] = True

        # 时间逸出边际报告拦截与保护机制
        if (~valid).any() and not self._warned_out_of_range:
            print(
                f"[气象] 警告: {(~valid).sum()} 个时间戳超出气象数据范围; "
                f"将自动使用默认背景垫片 fill_value={self.fill_value} 进行填补缓冲。"
            )
            self._warned_out_of_range = True

        return {
            "source_idx": source_idx,
            "left_idx": left_idx,
            "right_idx": right_idx,
            "alpha": alpha,
            "valid": valid,
        }

    def fetch_raw_frames_from_alignment(
        self,
        alignment: Dict[str, np.ndarray],
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> np.ndarray:
        """
        利用已经运算建立好的执行计划配置表，发起实质性的 HDF5 磁盘大容量读取行动。
        通过时间维度比例分配执行前后参考帧的数值混合，实现时间的丝滑插值过渡效果。

        参数:
            alignment (Dict): build_alignment 函数抛出的计划详单字典。
            start (Optional[int]): batch/slice 请求游标的开启行号。
            end (Optional[int]): batch/slice 请求切片的终结行号。支持切割可以有效化解全局显存透支。

        返回:
            np.ndarray: 解码并融合成型的准确物理气象阵列栈，结构必然维持为 [N, Channels, Height, Width]
        """
        if self.frame_shape is None:
            raise RuntimeError("操作中断: 框架基本信息未得到初始化！")

        # 截取局部请求范围内对齐指令单
        sl = slice(start, end)
        source_idx = alignment["source_idx"][sl]
        left_idx = alignment["left_idx"][sl]
        right_idx = alignment["right_idx"][sl]
        alpha = alignment["alpha"][sl]
        valid = alignment["valid"][sl]

        n = len(source_idx)
        # 用缺省基底占位符刷漆作为出厂空白基岩，确保哪怕完全超出时间区数据尺寸格式绝不崩坏。
        frames = np.full((n,) + self.frame_shape, self.fill_value, dtype=np.float32)
        if not valid.any():
            return frames

        # 核心优化策略: 使用 Group-by File 聚集读取代替 Scatter Read 跳跃读取。
        # 此策略旨在避免磁头横跨多文件的频繁挂载，从而几何级加速 IO 带宽占用。
        for src in np.unique(source_idx[valid]):
            src_mask = valid & (source_idx == src)
            dataset = self._get_dataset(int(src))

            # 先集中吞入本组需求对应的全部偏左界定帧画面库表。
            left_frames = np.asarray(dataset[left_idx[src_mask]], dtype=np.float32)
            alpha_src = alpha[src_mask]

            # 浮点数比较允许细微抖动界限误差。如若确定其本身几乎就恰好骑在整点时间上，直接返还数据免去二次抓取打扰。
            if np.allclose(alpha_src, 0.0):
                frames[src_mask] = left_frames
                continue

            # 对于确认落在两采样截面中间点的不规则位置，开启右侧阵列装填，
            # 实施二维线性时间过渡公式: Final = (1 - ratio) * Past  + ratio * Future
            right_frames = np.asarray(dataset[right_idx[src_mask]], dtype=np.float32)
            alpha_view = alpha_src.reshape(-1, 1, 1, 1)
            frames[src_mask] = (1.0 - alpha_view) * left_frames + alpha_view * right_frames

        return frames

    def fetch_frames_from_alignment(
        self,
        alignment: Dict[str, np.ndarray],
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> np.ndarray:
        """
        在完成时间对齐/插值读取后，进一步施加主预测链路统一的气象预处理。
        返回值结构保持为 [N, C, H, W]。
        """
        frames = self.fetch_raw_frames_from_alignment(alignment, start=start, end=end)
        return self.preprocess_weather_frames(frames, apply_normalization=True)

    def fetch_frames_by_dates(self, dates: Sequence[pd.Timestamp]) -> np.ndarray:
        """
        全量便捷连打操作口：整合计划编制与调控提取于一役的辅助类糖方法。
        
        参数:
            dates (Sequence[pd.Timestamp]): 人类可读的目标抽取多时刻清单列表
            
        返回:
            np.ndarray: 直接产出就绪完毕的数值张量集群
        """
        alignment = self.build_alignment(dates)
        return self.fetch_frames_from_alignment(alignment)


class FullMapConvTimeXerQuantile(nn.Module):
    """
    集成了全图气象卷积特征提取器（FullMapWeatherConvExtractor）与
    TimeXer 时序预测骨干模型的端到端复合概率预测模型。

    【核心架构设计：外生变量分离模式】
    此模型设计支持同时摄入并处理两个具有不同时间跨度的序列输入：
    1. 内生变量（Target / Load 数据）：
       - 代表自身历史演变规律（如过去 672 小时的负荷曲线）。
       - 序列长度为历史回溯窗口 `seq_len`。
    2. 外生变量（External / Weather 气象驱动）：
       - 代表外部物理驱动力，其不仅包含了与历史负荷对应的历史气象观测，
         还扩展包含了未来预测区域的气象预报信息。
       - 序列长度为扩展视野 `seq_len + pred_len`（由 `weather_seq_len` 定义）。

    最终通过多层投影输出带有置信度区间的“分位数概率分布”序列，代替单调的点预测。
    """
    def __init__(self, configs, quantiles: Sequence[float]):
        """
        初始化端到端分位数预测模型。

        参数:
            configs: 包含全量超参数的配置对象（如 argparse.Namespace），
                     其中必须包含 TimeXer 核心构建参数以及气象卷积层的定制参数。
            quantiles (Sequence[float]): 需要执行回归的分位数列表。
                                         例如 [0.1, 0.5, 0.9] 代表下界 10%、中位数 50% 和上界 90%。
        """
        super().__init__()
        self.quantiles = list(quantiles)
        self.n_quantiles = len(self.quantiles)
        
        # 气象分支模块配置
        self.weather_feature_dim = int(configs.weather_feature_dim)
        # 为防止全景 4D 张量爆显存而设定的 Batch 切割大小
        self.encode_chunk_size = int(getattr(configs, "weather_encode_chunk_size", 512))
        
        # 相似日先验特征配置
        self.use_similar_day_prior = bool(getattr(configs, "use_similar_day_prior", False))
        self.similar_day_top_k = int(getattr(configs, "similar_day_top_k", 3))
        # 先验特征维度包含 1 个加权均值曲线和 K 个单独的相似日曲线
        self.similar_day_prior_dim = self.similar_day_top_k + 1 if self.use_similar_day_prior else 0

        # —— 组件 1: 气象空间域提取骨干网络 ——
        self.weather_backbone = FullMapWeatherConvExtractor(
            in_channels=int(getattr(configs, "weather_in_channels")),
            out_channels=self.weather_feature_dim,
            kernel_height=int(getattr(configs, "weather_kernel_height")),
            kernel_width=int(getattr(configs, "weather_kernel_width")),
            dropout=float(getattr(configs, "dropout", 0.1)),
        )

        # 确立气象序列扩展视野长度（默认使用 configs.seq_len 兜底，通常应配置为 seq_len + pred_len）。
        self.weather_seq_len = int(getattr(configs, "weather_seq_len", configs.seq_len))
        
        # 将分离处理要求强制注入配置字典，交予底层 TimeXer 实例化：
        # `exo_seq_len` 指示外生变量的时间轴长度。
        configs.exo_seq_len = self.weather_seq_len
        # `enc_in` 在此分离架构下退化，仅表征纯内生（负荷单序列）输入。
        configs.enc_in = 1

        # —— 组件 2: TimeXer 时空交会时序预测器 ——
        self.timexer = TimeXer(configs)
        
        # —— 组件 3: 相似日先验知识注意力融合头 ——
        if self.use_similar_day_prior:
            fusion_hidden_dim = int(
                getattr(
                    configs,
                    "similar_day_fusion_hidden_dim",
                    max(16, int(getattr(configs, "d_model", 128)) // 4),
                )
            )
            # 建立一个微型多层感知机（MLP）网络以自适应地将先验曲线融入现有预断点位。
            self.similar_day_fusion_head = nn.Sequential(
                nn.Linear(1 + self.similar_day_prior_dim, fusion_hidden_dim),
                nn.GELU(),
                nn.Dropout(float(getattr(configs, "dropout", 0.1))),
                nn.Linear(fusion_hidden_dim, 1),
            )
            # 初始化融合头使用零初始（Zero-init），确保训练初期等价于不干预模型原始输出（Residual 骨架）。
            with torch.no_grad():
                nn.init.zeros_(self.similar_day_fusion_head[-1].weight)
                nn.init.zeros_(self.similar_day_fusion_head[-1].bias)
        else:
            self.similar_day_fusion_head = None
            
        # —— 组件 4: 分位数输出发射头 ——
        self.quantile_head = nn.Linear(1, self.n_quantiles)
        with torch.no_grad():
            # 权重全置 1：要求各分位数在开始时统一等距离锚定在 TimeXer 计算出的点预估附近。
            self.quantile_head.weight.fill_(1.0)
            # 偏置微调平移：依据分位数占比进行阶梯展开，确保分布不交叉发散并迅速构成合理的区间约束。
            self.quantile_head.bias.copy_(torch.tensor([q - 0.5 for q in self.quantiles]) * 0.1)

    def _encode_weather_chunk(self, weather_chunk: torch.Tensor) -> torch.Tensor:
        """分拆调用底层气象特征编码，为日后扩展保留中间件插板槽实现。"""
        return self.weather_backbone(weather_chunk)

    def _encode_weather_frames(self, weather_frames: torch.Tensor) -> torch.Tensor:
        """
        全量推断分块器 (Chunking for OOM Protection)
        将庞杂的 N 个气象切片矩阵分割为若干批次轮番送交显卡编码。
        """
        if weather_frames.ndim != 4:
            raise ValueError(f"气象帧必须具有维度 [N, C, H, W]，此时收到 {tuple(weather_frames.shape)}")

        flat = weather_frames.float()
        encoded_chunks: List[torch.Tensor] = []
        
        # 通过预设安全切分上限防止海量栅格冲垮显存水位。
        for start in range(0, flat.shape[0], self.encode_chunk_size):
            end = min(start + self.encode_chunk_size, flat.shape[0])
            encoded_chunks.append(self._encode_weather_chunk(flat[start:end]))
            
        return torch.cat(encoded_chunks, dim=0)

    def _encode_weather_sequence(
        self,
        weather_seq: Optional[torch.Tensor],
        weather_index: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        高维时空气象网格序列至低维致密特征的转换桥接门。
        利用 `weather_index` 参数提供了内存极限优化的“Unique 帧表+组装字典”双路模式。

        参数:
            weather_seq: 气象源数据张量。
                         - 若带牵引 `weather_index`: 此序列退化为无重复图样的张量 [Unique图源数, C, ...]。
                         - 常规模式: 巨型重复包容大阵列 [Batch, Time_Len, C, H, W]。
            weather_index: 指引去重前帧画面的恢复路标, 维度为 [Batch, Time_Len]。

        返回:
            Optional[torch.Tensor]: 成功被空间卷积并压入序列维度的时序矩阵 [Batch, Time_Len, Feature_Dim]。
        """
        if weather_seq is None:
            return None

        # --- 优化内存通路: 唯一池去重投影 ---
        if weather_index is not None:
            if weather_seq.ndim != 4:
                raise ValueError(
                    f"在提供索引字典时气象矩阵应已去重为 4D [U, C, H, W]，但获得形状 {tuple(weather_seq.shape)}"
                )
            if weather_index.ndim != 2:
                raise ValueError(f"天气索引坐标必须是 [Batch, Time] 形态，获得 {tuple(weather_index.shape)}")

            batch_size, time_len = weather_index.shape
            
            # 第一步：仅仅提纯计算不重复帧底库 (U: Unique 图例总数) -> [U, D]
            encoded_frames = self._encode_weather_frames(weather_seq)
            
            # 第二步：使用 index_select 根据坐标字典按需拼接装填，廉价且高效地克隆构建回全周期状态 -> [Batch * Time, D]
            gathered = encoded_frames.index_select(0, weather_index.reshape(-1))
            
            # 重新拆分时序形态
            return gathered.reshape(batch_size, time_len, self.weather_feature_dim)

        # --- 常规通路: 直接计算展开的大矩阵 ---
        if weather_seq.ndim != 5:
            raise ValueError(f"直接常规推送的气象序列必须是 [B, T, C, H, W]，获得 {tuple(weather_seq.shape)}")

        bsz, time_len, channels, height, width = weather_seq.shape
        # 将 Batch 批次同时间轴揉碎推进入普通批次队列
        flat = weather_seq.reshape(bsz * time_len, channels, height, width)
        encoded = self._encode_weather_frames(flat)
        return encoded.reshape(bsz, time_len, self.weather_feature_dim)

    def forward(
        self,
        load_x: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_exo_mark: torch.Tensor,
        weather_x: torch.Tensor,
        weather_x_index: Optional[torch.Tensor] = None,
        similar_day_prior: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        端到端模型复合前向推理循环。
        
        分别引导非对等长度的内系统负荷序列及外循环气象预报进入编码空间，进而通过 TimeXer 统一发榜解构，最终施以分位数离散推测。

        参数:
            load_x (Tensor): [Batch, seq_len, 1] - 内生时序数据本底（如历史使用电负荷）。
            x_mark_enc (Tensor): [Batch, seq_len, T_dim] - 内生时间规律标记物（如几点、周几）。
            x_exo_mark (Tensor): [Batch, weather_seq_len, T_dim] - 外生气象视野（包含过去到未来）的时间标记物。
            weather_x (Tensor):  海量全景切片网格集群对象。
            weather_x_index (Optional[Tensor]): [B, weather_seq_len] 去重复用优化查表签号。
            similar_day_prior (Optional[Tensor]): [B, pred_len, top_k + 1] 由相似日重组提取机制派发的历史启发辅助先验矩阵。
            mask (Optional[Tensor]): 序列空洞阻绝掩码。
            
        返回:
            Tensor: [Batch, pred_len, n_quantiles] - 展开覆盖分位数面值的目标回归面（含未来确切步长）。
        """
        # 第一阶段：空间图像认知（CNN-Layer）
        # 将 4D 气象网格提炼为一个抽象的可作为时序的协变量。
        weather_feature = self._encode_weather_sequence(weather_x, weather_x_index)

        # 第二阶段：动态异质解耦引擎流转（TimeXer Transformer Core）
        # 负载序列自身独占内源渠道，而具有加长未来视窗的天气表占有 exo 特权渠道进位。
        point_pred = self.timexer(
            load_x,                 # 内生原点序列: [B, seq_len, 1]
            x_mark_enc,             # 内生时效拓印: [B, seq_len, T]
            None, None,
            mask=mask,
            x_exo=weather_feature,  # 外生拓扑延展: [B, weather_seq_len, weather_feature_dim]
            x_exo_mark=x_exo_mark,  # 外生时标护持: [B, weather_seq_len, T]
        )
        
        # 只裁剪剥离核心所需的未卜先知片段（未来欲求探测区段）
        point_pred = point_pred[:, -self.timexer.pred_len:, :]
        
        # 第三阶段（可选附赠机制）：先验记忆强力补充反馈
        if self.use_similar_day_prior and similar_day_prior is not None:
            if similar_day_prior.ndim != 3:
                raise ValueError(
                    f"相似日先验期望需对齐矩阵面必须是 [Batch, pred_len, {self.similar_day_prior_dim}]，"
                    f"而当下呈现出 {tuple(similar_day_prior.shape)}"
                )
            if similar_day_prior.shape[1] != self.timexer.pred_len:
                raise ValueError(
                    f"相似日志向预测视野产生撕裂（长度分歧）：计划长度 {self.timexer.pred_len} vs 载入长度 {similar_day_prior.shape[1]}"
                )
            if similar_day_prior.shape[2] != self.similar_day_prior_dim:
                raise ValueError(
                    f"相似日内在刻面特征规模出离基准: 蓝图期望 {self.similar_day_prior_dim} vs 实际送达 {similar_day_prior.shape[2]}"
                )
                
            # 并联拼合当前主线神经预设点位和纯历史抽帧统计趋势
            fusion_input = torch.cat([point_pred, similar_day_prior.float()], dim=-1)
            # 通过类似 ResNet 的旁路接驳机制向点位做渐微修正
            point_pred = point_pred + self.similar_day_fusion_head(fusion_input)
            
        # 第四阶段：确定性概率离散化推演投影
        # 将一个纯坐标位面扩列辐射至 N 个置信带分位元
        return self.quantile_head(point_pred)


class LoadWeatherEndToEndDataset(Dataset):
    """
    同时管理时间序列负荷数据及相关天气的混合端到端数据集。
    
    【核心职责】
    1. 在内存中一次性读取和截取与所需集合（Train/Val/Test）相对应的时间范围数据。
    2. 基于分离模式下的非对等序列要求，同步生成历史内生负荷序列（宽 `seq_len`）与外气象驱动序列（宽 `weather_seq_len`）。
    3. 支持外挂相似日检索引擎，在实例化阶段静态生成所有样本的先验融合曲线，实现推断期零开销查询。
    4. 高级滑窗提取优化：采用跨样本的气象帧去重交织技术，大幅化解连续步长采样带来的大体积张量深拷贝冗余。
    """
    def __init__(
        self,
        args,
        weather_store: WeatherGridStore,
        flag: str = "train",
        size: Optional[Sequence[int]] = None,
        target: Optional[str] = None,
        scale: bool = True,
        timeenc: int = 1,
        freq: Optional[str] = None,
    ):
        """
        初始化端到端负荷与气象混合的数据集。

        参数:
            args: 配置命名空间，包含序列维度、相似日配置等所有外部超参数。
            weather_store (WeatherGridStore): 已加载 HDF5 和时间对齐指令的 HDF5 数据网格管家。
            flag (str): 数据集用途枚举，限定为 "train", "val", "test" 其中之一。
            size (Optional[Sequence[int]]): [seq_len（历史回溯窗）, label_len（监督冗余重叠区）, pred_len（目标预测窗）]。
            target (Optional[str]): CSV里的需要目标列面（如负荷 OT）。
            scale (bool): 是否根据全局信息开启特征零均值单位方差的标准化缩放。
            timeenc (int): 时间戳特征模式（0 - 离散的单位维度，1 - TimeFeature 连续态特征）。
            freq (Optional[str]): pandas 中的观测粒度频次（如 '15min'）。
        """
        if size is None:
            size = [args.seq_len, args.label_len, args.pred_len]

        self.args = args
        self.seq_len = int(size[0])
        self.label_len = int(size[1])
        self.pred_len = int(size[2])
        self.target = target or args.target
        self.scale = bool(scale)
        self.timeenc = int(timeenc)
        self.freq = freq or args.freq
        self.weather_store = weather_store
        
        # 内生序列相对于某个 index 起点的局部偏移矩阵（如 0 到 671）
        self.seq_offsets = np.arange(self.seq_len, dtype=np.int64)
        # 模型输出的监督边界偏移计算：包含尾部回放部分及全新推演部分 (label_len + pred_len)
        self.target_offsets = (
            np.arange(self.label_len + self.pred_len, dtype=np.int64) + (self.seq_len - self.label_len)
        )

        # weather_seq_len 允许独立放长布置（通常需超过负荷视野并包含气象未来探测），为外循环预测赋能。
        self.weather_seq_len = int(getattr(args, 'weather_seq_len', self.seq_len))
        self.weather_seq_offsets = np.arange(self.weather_seq_len, dtype=np.int64)

        # 转换为对应的切分数组下标系
        flag_map = {"train": 0, "val": 1, "test": 2}
        if flag not in flag_map:
            raise ValueError(f"flag 参数定义越界，必须是 train/val/test 的枚举组合，获取错误： {flag}")
        self.set_type = flag_map[flag]

        self.scaler: Optional[StandardScaler] = None
        self.target_mean = 0.0
        self.target_scale = 1.0
        self.data_x: Optional[np.ndarray] = None
        self.data_y: Optional[np.ndarray] = None
        self.data_stamp: Optional[np.ndarray] = None
        self.raw_dates: Optional[pd.Series] = None
        
        # 气象对齐调度与内存缓存地带
        self.weather_alignment: Optional[Dict[str, np.ndarray]] = None
        self.weather_cache: Optional[np.ndarray] = None
        self.use_weather_normalization = bool(
            getattr(args, "use_weather_normalization", self.weather_store.use_channel_normalization)
        ) or self.weather_store.use_channel_normalization
        self.weather_norm_fit_chunk_size = int(getattr(args, "weather_norm_fit_chunk_size", 512))
        
        # 相似日检索引擎配置入口
        self.use_similar_day_prior = bool(getattr(args, "use_similar_day_prior", False))
        self.similar_day_top_k = int(getattr(args, "similar_day_top_k", 3))
        self.similar_day_artifact_dir = str(
            getattr(args, "similar_day_artifact_dir", "./artifacts/similar_day_retriever_ae_128")
        )
        self.similar_day_prior_cache: Optional[np.ndarray] = None

        self.__read_data__()

    def __read_data__(self) -> None:
        """从文件层级全盘接手和重塑负荷基础数列、归一化、并关联到对齐气象面"""
        csv_path = os.path.join(self.args.root_path, self.args.data_path)
        df_raw = pd.read_csv(csv_path)
        if "date" not in df_raw.columns:
            raise ValueError(f"缺失时间对正基准: {csv_path} 中没有 date 基准列")

        df_raw["date"] = pd.to_datetime(df_raw["date"])
        # 不受控外部表先执行严格顺排，否则会摧毁所有滑窗相关的位置偏移假设。
        df_raw = df_raw.sort_values("date").reset_index(drop=True)

        if self.target not in df_raw.columns and "Target" in df_raw.columns:
            df_raw = df_raw.rename(columns={"Target": self.target})
        if self.target not in df_raw.columns:
            raise ValueError(f"未找到要求的靶向列: 目标 {self.target} 未记录于 {csv_path}")

        total_len = len(df_raw)
        num_train = int(total_len * 2 / 3)   # 训练集占据总长度的绝对前 2/3（维持系统在各个任务周期对刻度划分的一致性）
        num_vali = int(total_len * 1 / 6)    # 验证集切分 1/6
        num_test = total_len - num_train - num_vali  # 收编尾迹做自然预测验证

        # [滑窗连续性边界偏移保护]
        # 不是简易地切分成三段互不相交的绝对值，而是额外向过去借用 seq_len 窗体的留白！
        # 这个操作避免了验证及测试集的首批样本因被历史跨步抹杀而出现空洞的问题。
        border1s = [
            0,
            max(0, num_train - self.seq_len),
            max(0, num_train + num_vali - self.seq_len),
        ]
        border2s = [num_train, num_train + num_vali, total_len]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        train_dates = pd.DatetimeIndex(df_raw["date"].iloc[: border2s[0]].to_numpy())

        # 负荷目标列始终在训练段上拟合 scaler；气象数据是否做预处理由 WeatherGridStore 配置决定。
        target_values = df_raw[[self.target]].values.astype(np.float32)
        if self.scale:
            self.scaler = StandardScaler()
            # 严格只在训练段上拟合 scaler，避免未来信息泄漏。
            self.scaler.fit(target_values[: border2s[0]]) # 仅在训练集上 fit
            target_values = self.scaler.transform(target_values).astype(np.float32)
            self.target_mean = float(self.scaler.mean_[0])
            self.target_scale = float(self.scaler.scale_[0]) if self.scaler.scale_[0] != 0 else 1.0

        # 时间特性提取 (Month, Day, Weekday, Hour, Minute)
        df_stamp = df_raw[["date"]].iloc[border1:border2].copy()
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp["date"].apply(lambda row: row.month)
            df_stamp["day"] = df_stamp["date"].apply(lambda row: row.day)
            df_stamp["weekday"] = df_stamp["date"].apply(lambda row: row.weekday())
            df_stamp["hour"] = df_stamp["date"].apply(lambda row: row.hour)
            df_stamp["minute"] = df_stamp["date"].apply(lambda row: row.minute)
            data_stamp = df_stamp.drop(columns=["date"]).values.astype(np.float32)
        else:
            # 使用官方 DataEmbedding 风格的时间特征
            data_stamp = time_features(pd.to_datetime(df_stamp["date"].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0).astype(np.float32)

        # 裁定分配至专有环境的本集可用内存段
        self.data_x = target_values[border1:border2]
        self.data_y = target_values[border1:border2]
        self.data_stamp = data_stamp
        self.raw_dates = df_raw["date"].iloc[border1:border2].reset_index(drop=True)
        # 先构造时间对齐计划，再一次性把该 split 对应的气象帧预取到内存。
        self.weather_alignment = self.weather_store.build_alignment(self.raw_dates)

        print(f"[数据集-{self.set_type}] 正在预加载气象帧...")
        t0 = time.time()
        # 将规划字典发送给执行管家发起海量帧读操作。若启用气象归一化，则先用训练段统计量拟合后再标准化。
        # 核心逻辑：若启用归一化且尚未拟合统计量，则触发拟合流程。确保验证集/测试集使用训练集的尺度。
        if self.use_weather_normalization and not self.weather_store.has_fitted_channel_normalization():
            # A. 若当前正在处理训练集 (set_type=0)，则直接利用内存中的原始数据进行拟合
            if self.set_type == 0:
                # 拉取原始气象帧到内存，用于高效计算均值和标准差
                raw_weather_cache = self.weather_store.fetch_raw_frames_from_alignment(
                    self.weather_alignment, 0, len(self.data_x)
                )
                # 执行拟合：计算通道级的 log1p 指数、均值和方差
                self.weather_store.fit_channel_normalization_from_frames(
                    raw_weather_cache,
                    chunk_size=self.weather_norm_fit_chunk_size,
                    stage_name="train split cache",
                )
                # 拟合完成后，对这批内存数据执行预处理（含标准化）并转为缓存
                self.weather_cache = self.weather_store.preprocess_weather_frames(
                    raw_weather_cache,
                    apply_normalization=True,
                )
            # B. 若当前是验证集/测试集，需回溯到训练集日期序列进行拟合，保证分布一致性
            else:
                self.weather_store.fit_channel_normalization_from_dates(
                    train_dates,
                    chunk_size=self.weather_norm_fit_chunk_size,
                    stage_name="train split dates",
                )
                # 拟合完成后，直接拉取并预处理本 split 特有的气象帧
                self.weather_cache = self.weather_store.fetch_frames_from_alignment(
                    self.weather_alignment, 0, len(self.data_x)
                )
        # C. 若无需拟合（已就绪或未启用），则常规拉取标准化预处理后的气象特征
        else:
            self.weather_cache = self.weather_store.fetch_frames_from_alignment(
                self.weather_alignment, 0, len(self.data_x)
            )
        mem_gb = self.weather_cache.nbytes / (1024 ** 3)
        print(
            f"[数据集-{self.set_type}] 气象缓存已就绪: shape={self.weather_cache.shape}, "
            f"内存={mem_gb:.2f} GB, 耗时={time.time() - t0:.1f}s"
        )
        print(
            f"[数据集-{self.set_type}] 样本行数={len(self.data_x)}, "
            f"负荷维度={self.data_x.shape[-1]}, 气象帧形状={self.weather_store.frame_shape}"
        )
        if self.use_similar_day_prior:
            self._build_similar_day_prior_cache()

    def _build_similar_day_prior_cache(self) -> None:
        """
        初始化相似日推荐字典面：
        利用离线编译引擎查询相似日记录，并通过每日单点聚合降低极端的查询复杂度，
        最终将生成的历史借鉴特征推向缓存层供推断直提。
        """
        sample_count = len(self)
        feature_dim = self.similar_day_top_k + 1
        self.similar_day_prior_cache = np.zeros(
            (sample_count, self.pred_len, feature_dim),
            dtype=np.float32,
        )
        if sample_count <= 0:
            return
        if self.raw_dates is None:
            raise RuntimeError("raw_dates 没有可用时间参考线去寻址推荐库。")

        artifact_dir = os.path.abspath(self.similar_day_artifact_dir)
        if not os.path.isdir(artifact_dir):
            raise FileNotFoundError(f"相似日特征指引工坊不存在: {artifact_dir}")

        try:
            from similar_day_retriever import HDF5WeatherSequenceStore, SimilarDayRetriever
        except Exception as exc:
            raise ImportError(f"加载 similar_day_retriever 函数体失控阻断: {exc}") from exc

        retriever = SimilarDayRetriever.load(artifact_dir)
        if retriever.weather_h5_path is None:
            raise RuntimeError("预装 Retriever 数据缺失底线支持的 h5 连接栈。")

        # 将每个散列细粒度查询时间通过 .normalize() 归一化至“一天”为主视角的独立查询块。
        query_positions = np.arange(sample_count, dtype=np.int64) + self.seq_len
        query_timestamps = pd.DatetimeIndex(self.raw_dates.iloc[query_positions].to_numpy())
        query_anchor_days = query_timestamps.normalize()
        unique_anchor_days = pd.DatetimeIndex(pd.unique(query_anchor_days)).sort_values()
        freq_ns = int(pd.Timedelta(self.freq).value)

        print(
            f"[数据集-{self.set_type}] 正在预计算相似日先验: "
            f"原始子段={sample_count}, 压缩按天查询总数={len(unique_anchor_days)}, 取用={self.similar_day_top_k}"
        )
        t0 = time.time()
        anchor_cache: Dict[int, Dict[str, object]] = {}
        weather_store = HDF5WeatherSequenceStore(retriever.weather_h5_path)
        try:
            total_anchors = len(unique_anchor_days)
            for anchor_idx, anchor_day in enumerate(unique_anchor_days, start=1):
                # 借助模型检索引擎获取最佳推荐时刻历史组
                result = retriever.search_by_timestamp(
                    query_timestamp=anchor_day,
                    top_k=self.similar_day_top_k,
                    weather_store=weather_store,
                    history_end_timestamp_exclusive=anchor_day,
                )
                curves = np.asarray(result.load_curves, dtype=np.float32)
                if curves.ndim == 1:
                    curves = curves.reshape(1, -1)
                    
                # 将所查到的长期大负荷按要求的推测限度截减 pred_len 并施加与主线相同的 ScaleZ 重构分布
                if curves.ndim == 2 and curves.size > 0:
                    curves = curves[:, : self.pred_len]
                    curves = self.scale_target(curves.reshape(-1, 1)).reshape(curves.shape[0], curves.shape[1])
                else:
                    curves = np.empty((0, self.pred_len), dtype=np.float32)
                    
                anchor_cache[int(anchor_day.value)] = {
                    "query_timestamp": pd.Timestamp(result.query_timestamp),
                    "curves": curves.astype(np.float32, copy=False),
                    "scores": np.asarray(result.similarity_scores, dtype=np.float32),
                }
                
                if (
                    anchor_idx == 1
                    or anchor_idx == total_anchors
                    or anchor_idx % max(1, total_anchors // 8) == 0
                ):
                    print(
                        f"[数据集-{self.set_type}] 相似日锚点加载批次 {anchor_idx}/{total_anchors} "
                        f"目前锚系={pd.Timestamp(anchor_day)}"
                    )
        finally:
            weather_store.close()

        # —— 将天粒度推荐报告解压扩展到原频率的所有分钟内样本 ——
        for sample_idx, (query_ts, anchor_day) in enumerate(zip(query_timestamps, query_anchor_days), start=0):
            cache_item = anchor_cache.get(int(anchor_day.value))
            if cache_item is None:
                continue
                
            anchor_start_ts = pd.Timestamp(cache_item["query_timestamp"])
            # 根据查询时刻具体偏移计算步差以对应在当天被搜索到的 24 小时宽表内正确的截点开端
            shift_steps = int((pd.Timestamp(query_ts).value - anchor_start_ts.value) // freq_ns) % self.pred_len
            
            # 存入被微粒调整和组合完毕的一块完整的预置加成阵列
            self.similar_day_prior_cache[sample_idx] = _build_similar_day_prior_features(
                load_curves=np.asarray(cache_item["curves"], dtype=np.float32),
                similarity_scores=np.asarray(cache_item["scores"], dtype=np.float32),
                pred_len=self.pred_len,
                top_k=self.similar_day_top_k,
                shift_steps=shift_steps,
            )

        mem_mb = self.similar_day_prior_cache.nbytes / (1024 ** 2)
        print(
            f"[数据集-{self.set_type}] 相似日先验缓存已就绪分布: "
            f"shape={self.similar_day_prior_cache.shape}, 内存占量={mem_mb:.1f} MB, "
            f"开销时间={time.time() - t0:.1f}s"
        )

    def __getitem__(self, index: int):
        # 取材只做虚假透传：真正的截取通过特供 DataLoader 下放列表的方式处理
        return int(index)

    def __len__(self) -> int:
        # 最大滑窗数量评估
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def build_overlap_batch(
        self,
        batch_indices: Sequence[int],
    ) -> Tuple[torch.Tensor, ...]:
        """按提供的下标群构造复合大阵批次（包含去重与对齐设计机制）

        在支持大规模气象的架构中，为防止大 Batch Size 切片气象画面被无谓复制上百次，此程序通过
        生成基于 unique 截点的相对词典重整了返回内容格局。
        
        返回:
            batch_x: [B, seq_len, 1] 时序历史主源。
            batch_y: [B, label_len+pred_len, 1] 时距验证对照。
            batch_x_mark: [B, seq_len, T] 主线时间标贴。
            batch_exo_mark: [B, weather_seq_len, T] 外围视窗预测时间标贴。
            weather_frames: [U, C, H, W] 去重版的气象纹理包合集，内聚度极高。
            weather_index: [B, weather_seq_len] 告诉模型如何在唯一库群里按照坐标拉取正确的重复序列。
            similar_day_prior: [B, pred_len, top_k+1]（若配置开启，送交先验指导权重库）。
        """
        if self.data_x is None or self.data_y is None or self.data_stamp is None or self.weather_cache is None:
            raise RuntimeError("系统未处在活跃周期！核心数据驻存段报失。")

        indices = np.asarray(batch_indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError(f"Batch 打包序列格式或长度有误，收到={indices.shape}")

        # 三套位置矩阵分别对应负荷输入窗、监督目标窗、气象外生窗。
        seq_positions = indices[:, None] + self.seq_offsets[None, :]   # 内生序列位置
        target_positions = indices[:, None] + self.target_offsets[None, :]  # 预测目标位置
        weather_positions = indices[:, None] + self.weather_seq_offsets[None, :]  # 气象序列位置

        # 使用高级索引一次性提取所有样本（效率远高于循环提取）
        batch_x = torch.from_numpy(np.ascontiguousarray(self.data_x[seq_positions]))
        batch_y = torch.from_numpy(np.ascontiguousarray(self.data_y[target_positions]))
        batch_x_mark = torch.from_numpy(np.ascontiguousarray(self.data_stamp[seq_positions]))
        # 外生时间标记要与 weather_seq_len 严格对齐，因此单独构造。
        batch_exo_mark = torch.from_numpy(np.ascontiguousarray(self.data_stamp[weather_positions]))
        
        similar_day_prior = None
        if self.use_similar_day_prior:
            if self.similar_day_prior_cache is None:
                raise RuntimeError("similar_day_prior_cache 未生成！")
            similar_day_prior = torch.from_numpy(
                np.ascontiguousarray(self.similar_day_prior_cache[indices])
            )

        # --- 气象帧提取优化策略 ---
        # 如果 batch 内索引是连续的，则其气象窗口也高度重叠。
        # 这时直接切一整段连续缓存，再配合相对索引恢复即可，代价最低。
        if len(indices) == 1 or np.all(indices[1:] == indices[:-1] + 1):
            first_index = int(indices[0])
            last_index = int(indices[-1])
            # 切下这唯一涵盖全部批次范围首尾的大片段作为 Unique 底本，极少发生复制消耗。
            weather_frames = torch.from_numpy(
                np.ascontiguousarray(
                    self.weather_cache[first_index : last_index + self.weather_seq_len]
                )
            )
            # 通过差值赋予其还原用的一维序列码。
            weather_index = torch.from_numpy(
                np.ascontiguousarray(
                    (indices - first_index)[:, None] + self.weather_seq_offsets[None, :]
                )
            )
            if similar_day_prior is not None:
                return (
                    batch_x,
                    batch_y,
                    batch_x_mark,
                    batch_exo_mark,
                    weather_frames,
                    weather_index,
                    similar_day_prior,
                )
            return batch_x, batch_y, batch_x_mark, batch_exo_mark, weather_frames, weather_index

        # 非连续 batch 无法整段切片，但仍可通过 unique 去除重复气象帧。
        unique_weather_idx, inverse = np.unique(weather_positions.reshape(-1), return_inverse=True)
        weather_frames = torch.from_numpy(np.ascontiguousarray(self.weather_cache[unique_weather_idx]))
        
        # 将压缩字典 `inverse` 重建回外生时间列维度
        weather_index = torch.from_numpy(
            np.ascontiguousarray(inverse.reshape(len(indices), self.weather_seq_len))
        )
        if similar_day_prior is not None:
            return (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_exo_mark,
                weather_frames,
                weather_index,
                similar_day_prior,
            )
        return batch_x, batch_y, batch_x_mark, batch_exo_mark, weather_frames, weather_index

    def scale_target(self, data: np.ndarray) -> np.ndarray:
        """根据当前数据集内部维护的 scaler 均值和防差，将输入的数据正向标准化归一。"""
        data = np.asarray(data, dtype=np.float32).reshape(-1, 1)
        if not self.scale or self.scaler is None:
            return data.astype(np.float32)
        return self.scaler.transform(data).astype(np.float32)

    def inverse_transform_target(self, data: np.ndarray) -> np.ndarray:
        """从标准差域内退回至其物理世界实际的单位值，常在验证和图表输出时调用。"""
        data = np.asarray(data, dtype=np.float32)
        if not self.scale:
            return data
        return data * self.target_scale + self.target_mean


class ContiguousWindowBatchSampler(Sampler[List[int]]):
    """
    连续时间窗口批次采样器 (OOM 防护核心组件)。
    
    【核心设计意图】
    为了优化多重叠的散点时间序列提取所产生的冗余大体积气象矩阵重复解码瓶颈，
    此采样器放弃了 PyTorch 默认的全局完全随机采样，而是采用“块级随机（Block-wise Shuffle）”。
    
    它强制将分配进同一个 Batch 内部的起始采样索引限制在严格相连（连续步长递增）的时间片段里（例如 [0, 1, 2... 31]）。
    这样一来，当缓存层和模型接收到请求时，其背后的气象时间大面是高度相互覆盖延展的。
    模型就可以通过 np.unique 等手段只提取极少数的有效物理帧，从而将显存和 IO 开支从 O(B*L) 级压缩到 O(B+L) 级。
    """
    def __init__(self, dataset_len: int, batch_size: int, drop_last: bool = False):
        """
        初始化连续批次块采样器。

        参数:
            dataset_len (int): 外部数据全集可供滑窗的总长度。
            batch_size (int): 每个批次包裹内希望容纳的连续样本个数。
            drop_last (bool): 结尾残差弃保开关。若开启，结尾凑不够一个完整 batch_size 的零散数据块将被彻底丢弃忽略。
        """
        if dataset_len <= 0:
            raise ValueError(f"要求的数据集游标边界长度必须为正，获得 {dataset_len}")
        if batch_size <= 0:
            raise ValueError(f"规定的批次包裹大小必须为正，获得 {batch_size}")
            
        self.dataset_len = int(dataset_len)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)

    def __iter__(self) -> Iterator[List[int]]:
        # 先划分连续块，再打乱块顺序，从而同时满足
        # “batch 内连续”与“epoch 间随机化”这两个目标。
        block_starts = np.arange(0, self.dataset_len, self.batch_size, dtype=np.int64)
        np.random.shuffle(block_starts)

        for start in block_starts.tolist():
            end = start + self.batch_size
            
            # 越界防卫探测：当切割的这一条碰巧位于整个漫长序列最末尾，可能会超出数据集实际拥有的可用下标。
            if end > self.dataset_len:
                if self.drop_last:
                    continue
                # 若不丢弃残余，兜底收拢截断到数据集的边际尽头。
                end = self.dataset_len
                
            # 发车：推送生成完整的内部步进单（比如 [128, 129, 130... 159]）
            yield list(range(start, end))

    def __len__(self) -> int:
        """测算当前数据集在一个完整迭代周期内部将能吐出多少个这样的连列批次箱。"""
        if self.drop_last:
            # 严格模式：直接地板除，容不下的边角料不计入预期发送总数。
            return self.dataset_len // self.batch_size
        # 宽松模式：天花板除（Ceil），哪怕最后一次运送只有 1 个包裹，也算独立的一批。
        return (self.dataset_len + self.batch_size - 1) // self.batch_size


class OverlapAwareBatchCollator:
    """
    重叠感知批次整理器 (Collate Fn)。
    
    【原理内涵】
    传统的深度学习 Loader（例如 PyTorch 默认的 default_collate）是盲目的，它仅傻瓜式地针对 `dataset[i]` 返回的张量
    执行栈式堆叠 (`torch.stack`)。这在处理“单张照片”时没问题。
    
    但当我们在做大跨度气象视频流/气象切面的连续预测时：
    1. 相邻序列会抓取大量相同的 `[Channels, Height, Width]` 巨型矩阵并放入独立内存。
    2. Default Collate 会真的把它们当成不相关的东西堆叠成极具冗余的 batch（显存暴涨爆炸）。

    因此，本类直接代理接管了 Collate 责任。
    由于此 Dataset 的特殊设计（`__getitem__` 仅返回虚假的 index 索引序号），本整理器会收集这些组装批次的序号群 `batch`，
    直接回调给 `dataset.build_overlap_batch()` 内部的 C/Numpy 一维查表重叠剔除算法，由根部统一结算返回打包好的结果组合。
    """
    def __init__(self, dataset: LoadWeatherEndToEndDataset):
        """记录绑定的数据集实例引用。"""
        self.dataset = dataset

    def __call__(
        self,
        batch: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        拦截 DataLoader 的批次组织回调。

        参数:
            batch (Sequence[int]): 本轮被 DataLoader/Sampler 选中派发的样本 ID 队列索引集。
            
        返回:
            拦截后经由优化重叠算法吐出的各种 Tensor Tuple（含负荷、气象字典表、还原指路签及可选相似日约束面）。
        """
        # 放权给拥有全局图谱概览视野的 Dataset 实体做合并切图操作
        return self.dataset.build_overlap_batch(batch)


def weather_data_provider(args, flag: str, weather_store: WeatherGridStore):
    """
    负责生产与分配包含外部长效天气信息端到端的混合数据加载适配器 (Data Provider/Loader)。
    能够自动检测配置属性决定是否开启连续性快速对齐及批重叠去除的并行组合方式 (Overlap/Contiguous Smart Batching)。

    参数:
        args: 通用全局超参数集，包含 `patch_len`, `batch_size`, `target` 以及各种开关等。
        flag (str): 决定此时要切割装载哪个目标段，取值范畴：'train', 'val' 或是 'test'。
        weather_store (WeatherGridStore): 全局掌控 4D HDF5 时空切片缓存盘口的实例连接对象引用。

    返回:
        Tuple[LoadWeatherEndToEndDataset, DataLoader]
        元组包，包含构建好对应状态参数的完整 Dataset 模型实例，以及裹挟它的可供迭代训练输出使用的 DataLoader 管道推流器。
    """
    # 判别选用连续时戳（TimeFeature 形式如余弦展开图谱）还是离散数值。
    timeenc = 0 if args.embed != "timeF" else 1
    
    # 只有处于训练集状态下，为了丰富模型鲁棒视野才会被允许随机洗牌 (shuffle)。
    shuffle_flag = flag == "train"
    
    # 【高阶核心开关判别】：
    # 验证与测试本身就是按自然时序步步推进的，因此自然处于天然接续连续态；
    # 唯有当明确指示要在开启了完全随机化的乱序“Train”域强行注入批内保护时，才挂载特殊 Sampler。
    use_contiguous_train_batches = flag == "train" and bool(
        getattr(args, "contiguous_train_batches", False)
    )
    
    # pin_memory 只在 CUDA 寻址训练时挂载以提高至显寸通道穿透率，纯 CPU 是反向开销。
    use_pin_memory = bool(getattr(args, "pin_memory", False)) and torch.cuda.is_available() and getattr(args, "use_gpu", False)

    # 1. 创建混合特征抽样数据集
    dataset = LoadWeatherEndToEndDataset(
        args=args,
        weather_store=weather_store,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        target=args.target,
        scale=True,
        timeenc=timeenc,
        freq=args.freq,
    )
    
    # 2. 绑定带有覆盖感知的防溢出批次汇总整理器
    collate_fn = OverlapAwareBatchCollator(dataset)

    # 3. 按照场景特性颁发分发器
    if use_contiguous_train_batches:
        # [极速省显存分支]：处于训练模式且明确要求对气象阵列减负
        # 激活自行研制的区域连续采样机来强硬替代并封死 DataLoader 原生自带的随机机制 (`shuffle` /默认 Sampler 被屏闭)。
        loader = DataLoader(
            dataset,
            batch_sampler=ContiguousWindowBatchSampler(
                dataset_len=len(dataset),
                batch_size=args.batch_size,
                drop_last=False,
            ),
            num_workers=args.num_workers,
            pin_memory=use_pin_memory,
            collate_fn=collate_fn,
        )
    else:
        # [默认常规分支]：处于无乱序需求（Val / Test 区段具有自然平滑重叠），或刻意追求极致混乱抖动的全随机态（未开开关的 Train）。
        # 恢复普通 DataLoader 语义定义，由底层驱动行为。
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            pin_memory=use_pin_memory,
            collate_fn=collate_fn,
            drop_last=False,
        )
    return dataset, loader


__all__ = [
    "ContiguousWindowBatchSampler",
    "FullMapConvTimeXerQuantile",
    "FullMapWeatherConvExtractor",
    "LoadWeatherEndToEndDataset",
    "OverlapAwareBatchCollator",
    "WeatherGridStore",
    "weather_data_provider",
]
