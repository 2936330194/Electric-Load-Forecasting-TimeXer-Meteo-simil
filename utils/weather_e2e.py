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


def _require_weather_runtime() -> None:
    """
    检查处理气象数据所需的运行时环境。
    验证是否已安装 h5py 库，如果缺失则抛出 ImportError。
    """
    if h5py is None:
        raise ImportError("缺少 h5py 库。气象 HDF5 数据加载需要 h5py。")


def _guess_year_start_from_path(file_path: str) -> pd.Timestamp:
    """
    通过文件路径猜测数据的开始年份。
    
    参数:
        file_path: 文件路径字符串。
        
    返回:
        pandas.Timestamp: 猜测到的该年 1 月 1 日 0 点的时间戳。
        如果路径中未发现 20xx 格式的年份，则默认为 2025-01-01。
    """
    # 搜索路径文件名中的 20xx 格式年份
    match = re.search(r"(20\d{2})", Path(file_path).stem)
    if match:
        return pd.Timestamp(f"{match.group(1)}-01-01 00:00:00")
    return pd.Timestamp("2024-01-01 00:00:00")


def _find_first_4d_dataset(h5_obj):
    """
    递归搜索 HDF5 对象（文件或组）中第一个 4 维数据集。
    通常用于自动识别气象数据的主数据集（形状一般为 [时间, 通道, 高, 宽]）。
    
    参数:
        h5_obj: h5py.File 或 h5py.Group 对象。
        
    返回:
        h5py.Dataset: 找到的第一个 4D 数据集，若未找到则返回 None。
    """
    for key in h5_obj.keys():
        item = h5_obj[key]
        if isinstance(item, h5py.Dataset) and item.ndim == 4:
            return item
        if isinstance(item, h5py.Group):
            # 递归搜索子组
            found = _find_first_4d_dataset(item)
            if found is not None:
                return found
    return None


def _find_named_1d_dataset(h5_obj, keyword: str, expected_len: Optional[int] = None):
    """
    递归搜索名称包含特定关键字且为 1 维的数据集。
    常用于寻找时间戳数据集。
    
    参数:
        h5_obj: HDF5 对象。
        keyword: 搜索关键字（不区分大小写）。
        expected_len: 期望的数据集长度（可选）。
        
    返回:
        h5py.Dataset: 符合条件的数据集，若未找到则返回 None。
    """
    keyword = str(keyword).lower()
    for key in h5_obj.keys():
        item = h5_obj[key]
        if isinstance(item, h5py.Dataset) and item.ndim == 1:
            # 匹配关键字且长度一致（如果指定了长度）
            if keyword in str(key).lower() and (expected_len is None or len(item) == expected_len):
                return item
        if isinstance(item, h5py.Group):
            found = _find_named_1d_dataset(item, keyword, expected_len)
            if found is not None:
                return found
    return None


def _load_timestamp_index(timestamp_dataset) -> pd.DatetimeIndex:
    """
    从 HDF5 数据集中加载时间戳并转换为 pandas DatetimeIndex。
    支持字符串格式或字节格式的时间戳。
    
    参数:
        timestamp_dataset: HDF5 数据集对象。
        
    返回:
        pd.DatetimeIndex: 转换后的时间索引。
    """
    try:
        # 尝试以字符串模式读取（针对 h5py 的 String 类型）
        raw_values = timestamp_dataset.asstr()[...]
    except Exception:
        # 回退到原始读取
        raw_values = timestamp_dataset[...]
    
    normalized = []
    # 扁平化处理并根据类型解码
    for value in np.asarray(raw_values).reshape(-1):
        if isinstance(value, (bytes, bytearray)):
            normalized.append(value.decode("utf-8"))
        else:
            normalized.append(str(value))
            
    timestamps = pd.DatetimeIndex(pd.to_datetime(normalized))
    # 检查是否存在非法时间戳
    if timestamps.isna().any():
        raise ValueError(f"在数据集 {timestamp_dataset.name} 中发现无效的气象时间戳")
    return timestamps


def _infer_step_from_timestamps(timestamps: pd.DatetimeIndex) -> pd.Timedelta:
    """
    通过时间戳序列推断数据的采样频率（步长）。
    取相邻时间戳之间最小的正时间间隔。
    
    参数:
        timestamps: 已排序的时间戳序列。
        
    返回:
        pd.Timedelta: 推断出的最小有效时间步长。
    """
    if len(timestamps) < 2:
        raise ValueError("推断气象频率至少需要两个时间戳。")
        
    # 计算纳秒级别的间隔差
    diffs = np.diff(timestamps.asi8)
    # 只考虑正数间隔（针对可能存在的非单调递增情况）
    positive_diffs = diffs[diffs > 0]
    if len(positive_diffs) == 0:
        raise ValueError("无法从非递增的时间戳中推断气象频率。")
        
    return pd.Timedelta(int(np.min(positive_diffs)), unit="ns")


def _ensure_timedelta(freq: Any) -> pd.Timedelta:
    """
    确保输入被转换为 pandas.Timedelta 类型。
    
    参数:
        freq: 可被转换为 Timedelta 的对象（字符串、Timedelta或纳秒值）。
    """
    if isinstance(freq, pd.Timedelta):
        return freq
    if freq is None:
        raise ValueError("频率不能为 None。")
    return pd.Timedelta(freq)


def _timedelta_to_freq_str(freq: Any) -> str:
    """
    将 pandas.Timedelta 转换为精简的频率字符串格式（如 '1h', '30min', '24d'）。
    常用于 pandas 的频率参数。
    
    参数:
        freq: 待转换的时间间隔。
    """
    freq = _ensure_timedelta(freq)
    total_seconds = int(freq.total_seconds())
    if total_seconds <= 0:
        raise ValueError(f"频率必须为正数，得到: {freq}。")
        
    # 优先转换天、小时、分钟，最后是秒
    if total_seconds % 86400 == 0:
        return f"{total_seconds // 86400}d"
    if total_seconds % 3600 == 0:
        return f"{total_seconds // 3600}h"
    if total_seconds % 60 == 0:
        return f"{total_seconds // 60}min"
    return f"{total_seconds}s"


def _take_h5_rows_in_original_order(dataset, indices: np.ndarray) -> np.ndarray:
    """
    从 HDF5 数据集中提取指定索引的行，并保持请求的原始顺序。
    HDF5 驱动程序直接访问非递增索引往往效率极低且有限制，此函数通过
    排序提取并反向映射来优化性能。
    
    参数:
        dataset: HDF5 数据集对象。
        indices: 需要提取的行索引数组。
        
    返回:
        np.array: 提取后的数据，形状为 [N, C, H, W]。
    """
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError(f"索引必须是1维的，得到形状: {indices.shape}")
    if len(indices) == 0:
        return np.empty((0,) + tuple(dataset.shape[1:]), dtype=np.float32)
        
    # 获取唯一索引及其在原始请求中的逆映射，以便一次性顺序提取提高 IO 效率
    unique_indices, inverse = np.unique(indices, return_inverse=True)
    fetched_unique = np.asarray(dataset[unique_indices], dtype=np.float32)
    # 使用逆映射恢复请求的原始顺序
    return fetched_unique[inverse]


def infer_weather_history_len(seq_len: int, load_freq: Any, weather_freq: Any) -> int:
    """
    根据负荷序列长度和各自的频率，推算回看历史时需要多少步气象数据。
    目的是确保气象窗口的时长与负荷历史序列时长严格一致。
    
    参数:
        seq_len: 负荷历史序列点的个数。
        load_freq: 负荷数据的采样频率。
        weather_freq: 气象数据的采样频率。
        
    返回:
        int: 气象数据需要回看的步数。
    """
    load_freq = _ensure_timedelta(load_freq)
    weather_freq = _ensure_timedelta(weather_freq)
    
    # 转换为纳秒进行精确计算
    load_history_ns = int(seq_len) * int(load_freq.value)
    weather_step_ns = int(weather_freq.value)
    
    history_len, remainder = divmod(load_history_ns, weather_step_ns)
    # 验证是否能整除，确保时间尺度对齐
    if load_history_ns <= 0 or weather_step_ns <= 0 or remainder != 0 or history_len <= 0:
        raise ValueError(
            "负荷历史时长无法被气象频率整除: "
            f"seq_len={seq_len}, load_freq={load_freq}, weather_freq={weather_freq}"
        )
    return int(history_len)


def build_weather_sequence_timestamps(
    target_start: pd.Timestamp,
    weather_seq_len: int,
    weather_history_len: int,
    weather_freq: Any,
    reference_timestamps_ns: Optional[np.ndarray] = None,
) -> pd.DatetimeIndex:
    """
    构建一个以 target_start 为准，对齐气象频率的完整气象时间窗口。
    该窗口通常包含两个阶段：用于编码器输入的“气象历史观测”和用于对齐预测目标的“气象未来信息”。
    
    参数:
        target_start: 预测的目标起始时间点（负荷序列中 pred_len 的第一个点）。
        weather_seq_len: 需要构建的气象总序列长度 (例如: 168h 历史 + 24h 未来 = 192)。
        weather_history_len: 其中属于历史部分的长度 (例如: 168)。
        weather_freq: 气象数据的步长频率 (例如: '1h')。
        reference_timestamps_ns: 可选。气象 HDF5 文件中真实存在的纳秒时间戳数组。
                                 如果提供，则优先从物理设备中对齐索引；若超出范围则执行线性外推。
        
    返回:
        pd.DatetimeIndex: 包含完整气象窗口（历史+未来）的时间戳序列。
    """
    # 1. 参数格式标准化与配置预检
    weather_freq = _ensure_timedelta(weather_freq)
    weather_seq_len = int(weather_seq_len)
    weather_history_len = int(weather_history_len)
    
    # 校验窗口配置的合法性：长度必须为正，且历史长度不能超过总长度
    if weather_seq_len <= 0 or weather_history_len <= 0 or weather_seq_len < weather_history_len:
        raise ValueError(
            f"无效的气象窗口配置: seq_len={weather_seq_len}, history_len={weather_history_len}"
        )

    # 2. 定位气象锚点 (Anchor)
    # 气象锚点是指：在气象频率格点上，与 target_start 最接近的“对齐时间点”。
    # 例如：负荷频率是 15min (2024-01-01 00:15)，气象频率是 1h，则锚点对齐为 2024-01-01 00:00。
    step_ns = int(weather_freq.value)
    anchor_ns = (int(pd.Timestamp(target_start).value) // step_ns) * step_ns
    
    # 构建偏移量数组 [0, 1, 2, ..., seq_len-1]
    offsets = np.arange(weather_seq_len, dtype=np.int64)

    # 3. 分支逻辑 A：无参考时间轴 (多用于理论窗口计算或合成数据)
    if reference_timestamps_ns is None:
        # 窗口起点 = 锚点 - 历史步长 * 频率
        start_ns = anchor_ns - weather_history_len * step_ns
        # 生成线性排列的时间轴
        return pd.DatetimeIndex(pd.to_datetime(start_ns + offsets * step_ns))

    # 4. 分支逻辑 B：基于物理时间轴 (HDF5 索引模式)
    # 这种模式下会尽量保证生成的时间戳是 HDF5 文件中真实存在的点，适用于非严格等间隔的观测数据。
    reference_timestamps_ns = np.asarray(reference_timestamps_ns, dtype=np.int64).reshape(-1)
    if reference_timestamps_ns.size == 0:
        raise ValueError("reference_timestamps_ns 不能为空。")

    # 4.1 在参考轴中查找锚点的索引位置
    anchor_pos = np.searchsorted(reference_timestamps_ns, anchor_ns, side="left")
    
    # 检查锚点是否精确对齐（气象观测数据必须覆盖负荷预测的起始点）
    if anchor_pos >= reference_timestamps_ns.size or reference_timestamps_ns[anchor_pos] != anchor_ns:
        raise ValueError(
            "target_start 对齐后的气象锚点未在气象时间轴中找到 (气象观测数据可能不包含此预设定点): "
            f"{pd.Timestamp(anchor_ns)}"
        )

    # 4.2 计算整个窗口相对于参考轴的物理下标
    # 窗口下标范围：[anchor_pos - history_len, anchor_pos + (seq_len - history_len)]
    requested_positions = anchor_pos - weather_history_len + offsets
    
    # 检查哪些下标在 HDF5 索引范围内
    in_bounds = (requested_positions >= 0) & (requested_positions < reference_timestamps_ns.size)
    requested_ns = np.empty_like(requested_positions, dtype=np.int64)
    
    # 对于在范围内的点，直接提取 HDF5 中存储的真实纳秒戳
    requested_ns[in_bounds] = reference_timestamps_ns[requested_positions[in_bounds]]
    
    # 4.3 边界外越界处理 (Extrapolation)
    # 如果窗口包含数据集记录范围之外的时间点（例如预测由于超出 H5 范围），则执行基准频率上的线性推演。
    if (~in_bounds).any():
        before = requested_positions < 0
        after = requested_positions >= reference_timestamps_ns.size
        
        # 头部越界：以参考轴第一个点为基准向左外推
        if before.any():
            requested_ns[before] = reference_timestamps_ns[0] + requested_positions[before] * step_ns
            
        # 尾部越界：以参考轴最后一个点为基准向右平移
        if after.any():
            requested_ns[after] = (
                reference_timestamps_ns[-1]
                + (requested_positions[after] - (reference_timestamps_ns.size - 1)) * step_ns
            )
            
    # 将纳秒数组统一转换为 pandas DatetimeIndex 返回
    return pd.DatetimeIndex(pd.to_datetime(requested_ns))


def _build_weather_window_schedule(
    load_dates: Sequence[pd.Timestamp],
    seq_len: int,
    pred_len: int,
    weather_seq_len: int,
    weather_history_len: int,
    weather_freq: Any,
    reference_timestamps_ns: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    高效批量预计算所有样本对应的气象窗口调度表。
    相比逐个样本计算时间轴，该函数通过构建一个覆盖整个数据集的大型“全局时间线 (Timeline)”，
    来最小化重复计算开销。这在处理包含数万个样本的训练集时非常关键。
    
    参数:
        load_dates: 整个负荷数据集中的原始日期序列。
        seq_len: 负荷回顾步长。
        pred_len: 负荷预测步长。
        weather_seq_len: 每个样本需要的气象总点数。
        weather_history_len: 每个样本窗口中的历史气象点数。
        weather_freq: 气象步进频率。
        reference_timestamps_ns: HDF5 物理数据中真实存在的时间戳索引数组。
        
    返回:
        timeline_ns (np.ndarray): 一个单调递增的长纳秒数组，包含该数据集所需的所有不重复气象时间点。
        window_positions (np.ndarray): 形状为 [N_Samples, weather_seq_len] 的矩阵。
                                      每一行代表一个样本，存储其窗口内的气象点在 timeline_ns 中的下标位置。
    """
    # 1. 初始化与样本规模确认
    load_dates = pd.DatetimeIndex(pd.to_datetime(load_dates))
    # 样本数量 = 总点数 - 历史长度 - 预测长度 + 1
    sample_count = len(load_dates) - int(seq_len) - int(pred_len) + 1
    if sample_count <= 0:
        raise ValueError("负荷时间戳不足，无法构建气象窗口。")
        
    weather_freq = _ensure_timedelta(weather_freq)
    step_ns = int(weather_freq.value)
    
    # 2. 确定所有样本的起始预测点及其对齐后的气象锚点
    # target_start_ns 形状为 [N_Samples]
    target_start_ns = load_dates.asi8[int(seq_len) : int(seq_len) + sample_count].astype(np.int64)
    # 将负荷时间戳向下对齐到气象频率的格点上
    anchor_ns = (target_start_ns // step_ns) * step_ns
    # 单个窗口内的步进偏移量 [0, 1, ..., seq_len-1]
    offsets = np.arange(int(weather_seq_len), dtype=np.int64)

    # 3. 分支分支 A：纯线性生成模式 (无参考物理索引)
    if reference_timestamps_ns is None:
        # 计算所有样本中最早和最晚的窗口起始点
        window_start_ns = anchor_ns - int(weather_history_len) * step_ns
        timeline_start_ns = int(window_start_ns.min())
        # 全局结束点 = 最晚起点 + 窗口总偏移
        timeline_end_ns = int(window_start_ns.max() + offsets[-1] * step_ns)
        
        # 生成贯穿整个数据集长度的全局时间线
        timeline_ns = np.arange(timeline_start_ns, timeline_end_ns + step_ns, step_ns, dtype=np.int64)
        
        # 利用广播机制计算每个样本在 timeline_ns 中的位置矩阵
        # [N, 1] + [1, W_Seq] -> [N, W_Seq]
        window_positions = (
            ((window_start_ns - timeline_start_ns) // step_ns)[:, None] + offsets[None, :]
        ).astype(np.int32)
        return timeline_ns, window_positions

    # 4. 分支分支 B：参考索引对齐模式 (基于 HDF5 时间轴)
    reference_timestamps_ns = np.asarray(reference_timestamps_ns, dtype=np.int64).reshape(-1)
    if reference_timestamps_ns.size == 0:
        raise ValueError("reference_timestamps_ns 不能为空。")

    # 4.1 批量查找锚点在 HDF5 中的索引
    anchor_pos = np.searchsorted(reference_timestamps_ns, anchor_ns, side="left")
    in_bounds = anchor_pos < reference_timestamps_ns.size
    
    # 严格校验：确保每一个通过负荷时间计算出的气象锚点在参考轴中均有精确对应
    exact = np.zeros_like(anchor_pos, dtype=bool)
    exact[in_bounds] = reference_timestamps_ns[anchor_pos[in_bounds]] == anchor_ns[in_bounds]
    if not exact.all():
        bad_idx = int(np.flatnonzero(~exact)[0])
        raise ValueError(
            "存在负荷样本的气象锚点无法在气象时间轴中精确找到 (请检查气象 HDF5 是否覆盖了负荷数据的所有区间): "
            f"sample={bad_idx}, anchor={pd.Timestamp(anchor_ns[bad_idx])}"
        )

    # 4.2 计算所有样本相对于参考轴的全局下标矩阵
    # 每一行表示该样本窗口所需的 HDF5 下标序列
    requested_positions = (anchor_pos.astype(np.int64) - int(weather_history_len))[:, None] + offsets[None, :]
    
    # 4.3 构建局部 Timeline 
    # 为了节省内存，我们只保留数据集覆盖范围内的那一截时间线
    timeline_start_pos = int(requested_positions.min())
    timeline_end_pos = int(requested_positions.max())
    timeline_positions = np.arange(timeline_start_pos, timeline_end_pos + 1, dtype=np.int64)

    # 4.4 提取/推演局部 Timeline 的真实纳秒数值
    timeline_ns = np.empty_like(timeline_positions, dtype=np.int64)
    # 处理范围内的真实记录
    valid = (timeline_positions >= 0) & (timeline_positions < reference_timestamps_ns.size)
    timeline_ns[valid] = reference_timestamps_ns[timeline_positions[valid]]
    
    # 处理超出文件记录范围（主要是预测未来部分）的线性推演
    if (~valid).any():
        before = timeline_positions < 0
        after = timeline_positions >= reference_timestamps_ns.size
        # 头部左外推
        if before.any():
            timeline_ns[before] = reference_timestamps_ns[0] + timeline_positions[before] * step_ns
        # 尾部右平移
        if after.any():
            timeline_ns[after] = (
                reference_timestamps_ns[-1]
                + (timeline_positions[after] - (reference_timestamps_ns.size - 1)) * step_ns
            )

    # 4.5 将全局下标矩阵转换为相对于局部 Timeline 的相对下标矩阵
    window_positions = (requested_positions - timeline_start_pos).astype(np.int32)
    return timeline_ns, window_positions



class FullMapWeatherConvExtractor(nn.Module):
    """
    全图气象卷积特征提取器。
    
    该模块负责将输入的 2D 气象网格数据（如风速、温度分布图）通过卷积操作
    压缩并转换为一维特征向量，以便后续与负荷数据结合。
    由于气象网格通常较小且具有全局空间相关性，该 extractor 默认卷积核
    大小与网格大小一致，实现“全图感知”。
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_height: int, kernel_width: int, dropout: float = 0.1):
        """
        初始化提取器。
        
        参数:
            in_channels: 输入气象数据的通道数（如包含多少个气象变量）。
            out_channels: 输出特征向量的维度。
            kernel_height: 气象网格的高度（像素/网格点数）。
            kernel_width: 气象网格的宽度（像素/网格点数）。
            dropout: 丢弃率，用于防止过拟合。
        """
        super().__init__()
        self.kernel_height = int(kernel_height)
        self.kernel_width = int(kernel_width)
        self.output_dim = int(out_channels)
        
        # 定义一个全图卷积：卷积核大小等于输入图像大小，从而输出为 1x1 的空间张量
        self.full_map_conv = nn.Conv2d(
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            kernel_size=(self.kernel_height, self.kernel_width),
            bias=True,
        )
        # 层归一化：稳定深层网络的训练
        self.norm = nn.LayerNorm(out_channels)
        # GELU 激活函数：现代 Transformer 架构常用的激活函数
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        参数:
            x: 输入张量，形状预计为 [B, C, H, W]（批量、通道、高、宽）。
            
        返回:
            torch.Tensor: 提取后的特征向量，形状为 [B, out_channels]。
        """
        # 数据完整性检查
        if x.ndim != 4:
            raise ValueError(f"气象输入数据必须是 [B, C, H, W] 格式，当前得到: {tuple(x.shape)}")
        if x.shape[-2] != self.kernel_height or x.shape[-1] != self.kernel_width:
            raise ValueError(
                f"输入的气象帧尺寸必须是 ({self.kernel_height}, {self.kernel_width})，"
                f"当前得到的是 ({x.shape[-2]}, {x.shape[-1]})"
            )
            
        # 1. 卷积提取：将 [B, C, H, W] 映射为 [B, out_channels, 1, 1]
        # 2. 展平：将多余的空间维度去除，得到 [B, out_channels]
        x = self.full_map_conv(x.float()).flatten(1)
        
        # 3. 后处理：归一化 -> 激活 -> Dropout
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x



class WeatherGridStore:
    """
    气象网格数据存储与高效读取器。
    
    该类负责管理多个 HDF5 格式的气象文件，支持：
    1. 自动识别并加载多个时间段的气象数据集。
    2. 提供基于时间戳的高效索引对齐（Alignment）。
    3. 实现数据预处理流水线，包括 Log1p 转换和 Z-Score 归一化。
    4. 采用分块读取和顺序 IO 优化，提升大规模网格数据的加载速度。
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
        初始化存储器。
        
        参数:
            h5_specs: 包含 (文件路径, 开始时间, 频率) 的元组序列。
            expected_in_channels: 预期的气象通道数（用于校验）。
            fill_value: 缺失数据时的填充值。
            use_channel_normalization: 是否启用通道归一化（Z-Score）。
            log1p_channels: 需要进行 log1p 转换的通道索引（如降水量）。
            normalization_eps: 归一化时的极小值，防止除零。
        """
        _require_weather_runtime()
        self.expected_in_channels = int(expected_in_channels)
        self.h5_specs = []
        # 解析配置参数
        for spec in h5_specs:
            path = os.path.abspath(spec[0])
            # 如果未指定开始时间，尝试从文件名解析
            start = pd.Timestamp(spec[1]) if spec[1] else _guess_year_start_from_path(spec[0])
            freq = spec[2] if len(spec) > 2 and spec[2] else None
            self.h5_specs.append((path, start, freq))
            
        self.fill_value = float(fill_value)
        self.use_channel_normalization = bool(use_channel_normalization)
        self.normalization_eps = float(normalization_eps)
        
        # 处理 log1p 通道白名单
        raw_log1p_channels = [] if log1p_channels is None else [int(ch) for ch in log1p_channels]
        invalid_channels = [ch for ch in raw_log1p_channels if ch < 0 or ch >= self.expected_in_channels]
        if invalid_channels:
            raise ValueError(
                f"log1p_channels 超出通道范围 (expected={self.expected_in_channels}): {invalid_channels}"
            )
        self.log1p_channels = tuple(sorted(set(raw_log1p_channels)))
        
        # 初始化状态变量
        self.sources: List[Dict[str, object]] = []
        self.frame_shape: Optional[Tuple[int, int, int]] = None
        self._file_handles: Dict[int, Any] = {}
        self._datasets: Dict[int, Any] = {}
        self._warned_out_of_range = False
        
        # 归一化统计量
        self.native_freq: Optional[pd.Timedelta] = None
        self.native_freq_str: Optional[str] = None
        self.channel_mean: Optional[np.ndarray] = None
        self.channel_std: Optional[np.ndarray] = None
        self._normalization_sample_count = 0
        
        # 执行初始化准备
        self.prepare()

    def prepare(self) -> None:
        """
        扫描所有 H5 文件，校验元数据并构建全局时间轴索引。
        """
        if self.sources:
            return
            
        for h5_path, start_time, explicit_freq in self.h5_specs:
            if not os.path.exists(h5_path):
                print(f"[weather] 找不到文件: {h5_path}")
                continue
                
            with h5py.File(h5_path, "r") as h5_file:
                # 1. 查找 4D 数据集 [T, C, H, W]
                dataset = _find_first_4d_dataset(h5_file)
                if dataset is None:
                    raise ValueError(f"在 {h5_path} 中未找到 4D 数据集")
                if dataset.shape[1] != self.expected_in_channels:
                    raise ValueError(
                        f"{h5_path} 通道数不匹配: 预期 {self.expected_in_channels}, 实际 {dataset.shape[1]}"
                    )
                
                n_steps, n_channels, height, width = dataset.shape
                dataset_name = dataset.name
                
                # 2. 尝试加载时间戳数据集
                timestamp_dataset = _find_named_1d_dataset(h5_file, "timestamp", expected_len=n_steps)
                if timestamp_dataset is not None:
                    timestamps = _load_timestamp_index(timestamp_dataset)
                    freq = _infer_step_from_timestamps(timestamps)
                else:
                    # 如果缺失时间戳，则根据配置的起始时间和频率生成
                    if not explicit_freq:
                        raise ValueError(
                            f"{h5_path} 不包含时间戳数据集，必须在配置中显式指定 weather_h5_specs 的频率。"
                        )
                    freq = pd.Timedelta(explicit_freq)
                    timestamps = pd.date_range(start=start_time, periods=n_steps, freq=freq)
            
            # 记录文件源配置
            source = {
                "path": h5_path,
                "dataset_name": dataset_name,
                "n_steps": n_steps,
                "timestamps_ns": timestamps.asi8.copy(), # 存储为纳秒长整型以加速比对
                "start_ns": int(timestamps[0].value),
                "end_ns": int(timestamps[-1].value),
                "freq": freq,
            }
            self.sources.append(source)
            
            # 校验网格尺寸一致性
            if self.frame_shape is None:
                self.frame_shape = (n_channels, height, width)
            elif self.frame_shape != (n_channels, height, width):
                raise ValueError(f"气象帧尺寸不一致: {self.frame_shape} vs {(n_channels, height, width)}")
            
            print(f"[weather] 已加载 {Path(h5_path).name}: steps={n_steps}, freq={freq}")
            
        if not self.sources:
            raise FileNotFoundError("未找到任何有效的气象 HDF5 文件。")
            
        # 3. 按时间排序所有源并确定全局频率
        self.sources.sort(key=lambda x: x["start_ns"])
        self.native_freq = sorted({source["freq"] for source in self.sources}, key=lambda value: int(value.value))[0]
        self.native_freq_str = _timedelta_to_freq_str(self.native_freq)
        
        start_ts = pd.Timestamp(min(source["start_ns"] for source in self.sources))
        end_ts = pd.Timestamp(max(source["end_ns"] for source in self.sources))
        print(f"[weather] 数据覆盖范围: {start_ts} -> {end_ts}")

    def _get_dataset(self, source_idx: int):
        """
        获取缓存的 H5 数据集对象，避免重复打开文件。
        """
        if source_idx in self._datasets:
            return self._datasets[source_idx]
            
        source = self.sources[source_idx]
        h5_file = h5py.File(source["path"], "r")
        dataset = h5_file[source["dataset_name"]]
        
        # 保持引用以防文件关闭
        self._file_handles[source_idx] = h5_file
        self._datasets[source_idx] = dataset
        return dataset

    def close(self) -> None:
        """
        显式关闭所有打开的 HDF5 文件。
        """
        for file_handle in self._file_handles.values():
            try:
                file_handle.close()
            except Exception:
                pass
        self._file_handles.clear()
        self._datasets.clear()

    def __del__(self):
        """对象销毁时自动关闭文件"""
        self.close()

    def has_fitted_channel_normalization(self) -> bool:
        """
        检查是否已计算或已拟合归一化统计量。
        """
        if not self.use_channel_normalization:
            return True
        return self.channel_mean is not None and self.channel_std is not None

    def _apply_log1p_transform_inplace(self, frames: np.ndarray) -> np.ndarray:
        """
        对指定通道执行 inplace 的 log1p 变换，用于平滑长尾分布（如降水）。
        """
        if not self.log1p_channels:
            return frames
        for channel in self.log1p_channels:
            # 裁剪负值，确保 log1p 安全
            frames[:, channel, :, :] = np.log1p(np.clip(frames[:, channel, :, :], a_min=0.0, a_max=None))
        return frames

    def _apply_channel_normalization_inplace(self, frames: np.ndarray) -> np.ndarray:
        """
        执行 inplace 的 Z-Score 归一化。
        """
        if not self.use_channel_normalization:
            return frames
        if self.channel_mean is None or self.channel_std is None:
            raise RuntimeError(
                "气象通道归一化统计量尚未拟合。请先调用 fit_channel_normalization_xxx。"
            )
        # 使用广播机制进行减均值和除标准差
        frames -= self.channel_mean.reshape(1, -1, 1, 1)
        frames /= self.channel_std.reshape(1, -1, 1, 1)
        return frames

    def preprocess_weather_frames(self, frames: np.ndarray, apply_normalization: bool = True) -> np.ndarray:
        """
        执行完整的气象预处理流水线。
        
        参数:
            frames: 原始气象张量 [N, C, H, W]。
            apply_normalization: 是否执行 Z-Score 归一化。
        """
        frames = np.asarray(frames, dtype=np.float32)
        if frames.ndim != 4:
            raise ValueError(f"气象帧必须是 4D [N, C, H, W]，得到形状: {tuple(frames.shape)}")
            
        # 1. Log1p 变换
        self._apply_log1p_transform_inplace(frames)
        # 2. Z-Score 归一化
        if apply_normalization:
            self._apply_channel_normalization_inplace(frames)
        return frames

    def _accumulate_channel_stats(self, frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        分块累加通道统计量（单步均值与平方和均值计算基础）。
        
        返回:
            channel_sum: 通道累加和。
            channel_sq_sum: 通道平方累加和。
            element_count: 参与计算的像素总点数。
        """
        frames = np.asarray(frames, dtype=np.float32)
        if frames.ndim != 4:
            raise ValueError(f"气象帧必须是 4D [N, C, H, W]，得到形状: {tuple(frames.shape)}")
            
        if frames.shape[0] == 0:
            zeros = np.zeros((self.expected_in_channels,), dtype=np.float64)
            return zeros, zeros.copy(), 0
            
        # 先进行 log1p 转换（如果有），确保归一化是基于转换后的分布
        self._apply_log1p_transform_inplace(frames)
        
        # 为了保证精度，使用 float64 进行累加
        channel_sum = frames.sum(axis=(0, 2, 3), dtype=np.float64)
        frames64 = frames.astype(np.float64, copy=False)
        channel_sq_sum = np.sum(frames64 * frames64, axis=(0, 2, 3), dtype=np.float64)
        
        # 计算总元素个数：样本数 * 高 * 宽
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
        利用累加值计算最终的均值与标准差。
        使用 Var(X) = E[X^2] - (E[X])^2 公式。
        """
        if element_count <= 0:
            raise RuntimeError("拟合失败：未发现有效的数据点。")
            
        mean64 = channel_sum / float(element_count)
        # 计算方差，保证非负
        var64 = np.maximum(
            channel_sq_sum / float(element_count) - mean64 ** 2,
            self.normalization_eps ** 2,
        )
        std64 = np.sqrt(var64)
        
        # 存储为 float32 用于推理
        self.channel_mean = mean64.astype(np.float32)
        self.channel_std = np.maximum(std64, self.normalization_eps).astype(np.float32)
        self._normalization_sample_count = int(sample_count)
        
        print(
            f"[weather] 归一化统计量拟合完成 ({stage_name}): "
            f"样本数={sample_count}, log1p通道={list(self.log1p_channels)}, 耗时={elapsed_sec:.1f}s"
        )

    def fit_channel_normalization_from_dates(
        self,
        dates: Sequence[pd.Timestamp],
        chunk_size: int = 512,
        stage_name: str = "train dates",
    ) -> None:
        """
        根据指定的日期列表从 H5 文件中读取数据并拟合归一化统计量。
        通常用于训练集。
        """
        if not self.use_channel_normalization or self.has_fitted_channel_normalization():
            return
            
        if self.frame_shape is None:
            raise RuntimeError("frame_shape 未初始化，无法拟合。")
            
        # 1. 寻找所有日期的索引位置
        alignment = self.build_alignment(dates)
        valid = np.asarray(alignment["valid"], dtype=bool)
        valid_count = int(valid.sum())
        
        if valid_count <= 0:
            raise RuntimeError("在气象文件中未找到所请求日期的有效数据。")
            
        channel_sum = np.zeros((self.expected_in_channels,), dtype=np.float64)
        channel_sq_sum = np.zeros((self.expected_in_channels,), dtype=np.float64)
        element_count = 0
        chunk_size = max(1, int(chunk_size))
        
        print(
            f"[weather] 正在从文件拟合归一化参数 ({stage_name}): "
            f"目标点数={len(valid)}, 有效点数={valid_count}, 块大小={chunk_size}"
        )
        
        t0 = time.time()
        # 2. 分块读取数据并累加统计量
        for start in range(0, len(valid), chunk_size):
            end = min(start + chunk_size, len(valid))
            raw_chunk = self.fetch_raw_frames_from_alignment(alignment, start, end)
            
            # 仅处理有效的帧
            if not valid[start:end].all():
                raw_chunk = raw_chunk[valid[start:end]]
                
            if raw_chunk.size == 0:
                continue
                
            chunk_sum, chunk_sq_sum, chunk_count = self._accumulate_channel_stats(raw_chunk)
            channel_sum += chunk_sum
            channel_sq_sum += chunk_sq_sum
            element_count += chunk_count
            
        # 3. 计算最终统计量
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
        使用已加载到内存中的原始帧数据快速拟合归一化统计量。
        通常用于已 Preload 的内存数据集。
        """
        if not self.use_channel_normalization or self.has_fitted_channel_normalization():
            return
            
        raw_frames = np.asarray(raw_frames, dtype=np.float32)
        if raw_frames.ndim != 4:
            raise ValueError(f"气象帧必须是 4D [N, C, H, W]，得到形状: {tuple(raw_frames.shape)}")
            
        channel_sum = np.zeros((self.expected_in_channels,), dtype=np.float64)
        channel_sq_sum = np.zeros((self.expected_in_channels,), dtype=np.float64)
        element_count = 0
        chunk_size = max(1, int(chunk_size))
        
        print(
            f"[weather] 正在从内存数据拟合归一化参数 ({stage_name}): "
            f"帧数={raw_frames.shape[0]}, 块大小={chunk_size}"
        )
        
        t0 = time.time()
        for start in range(0, raw_frames.shape[0], chunk_size):
            end = min(start + chunk_size, raw_frames.shape[0])
            # 复制分块以确保 IO 和计算隔离
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
        将请求的日期序列映射到 HDF5 文件中的特定行索引。
        支持跨文件的时间序列对齐。
        
        返回:
            Dict 包含:
                source_idx: 每个请求日期对应的文件源索引（-1 表示未找到）。
                row_idx: 对应文件中的行号。
                valid: 布尔遮罩，指示该日期是否有效。
        """
        dates = pd.DatetimeIndex(pd.to_datetime(dates))
        request_ns = dates.asi8.astype(np.int64)
        
        source_idx = np.full(len(request_ns), -1, dtype=np.int32)
        row_idx = np.zeros(len(request_ns), dtype=np.int32)
        valid = np.zeros(len(request_ns), dtype=bool)
        
        for idx, source in enumerate(self.sources):
            # 找出落在此文件范围内的尚未匹配的时间戳
            mask = (~valid) & (request_ns >= source["start_ns"]) & (request_ns <= source["end_ns"])
            if not mask.any():
                continue
                
            ts_ns = source["timestamps_ns"]
            req = request_ns[mask]
            # 计算请求时间在文件时间序列中的位置
            pos = np.searchsorted(ts_ns, req, side="left")
            
            in_bounds = pos < len(ts_ns)
            exact = np.zeros_like(pos, dtype=bool)
            # 检查时间戳是否精确匹配
            exact[in_bounds] = ts_ns[pos[in_bounds]] == req[in_bounds]
            
            if not exact.any():
                continue
                
            matched_indices = np.where(mask)[0][exact]
            source_idx[matched_indices] = idx
            row_idx[matched_indices] = pos[exact]
            valid[matched_indices] = True
            
        if (~valid).any() and not self._warned_out_of_range:
            print(
                f"[weather] 警告: 有 {(~valid).sum()} 个请求的气象时间戳未找到，"
                f"将填入默认值: fill_value={self.fill_value}"
            )
            self._warned_out_of_range = True
            
        return {"source_idx": source_idx, "row_idx": row_idx, "valid": valid}

    def fetch_raw_frames_from_alignment(
        self,
        alignment: Dict[str, np.ndarray],
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> np.ndarray:
        """
        基于对齐信息从 H5 文件中高效读取原始帧数据。
        按文件源进行分组读取，通过优化提取提高性能。
        """
        if self.frame_shape is None:
            raise RuntimeError("frame_shape 未初始化。")
            
        sl = slice(start, end)
        source_idx = alignment["source_idx"][sl]
        row_idx = alignment["row_idx"][sl]
        valid = alignment["valid"][sl]
        
        # 初始化输出张量
        frames = np.full((len(source_idx),) + self.frame_shape, self.fill_value, dtype=np.float32)
        
        if not valid.any():
            return frames
            
        # 对每个涉及的文件源，批量顺序提取行
        for src in np.unique(source_idx[valid]):
            src_mask = valid & (source_idx == src)
            dataset = self._get_dataset(int(src))
            # 核心优化：调用专用函数加速 H5 多行非连续读取
            frames[src_mask] = _take_h5_rows_in_original_order(dataset, row_idx[src_mask])
            
        return frames

    def fetch_frames_from_alignment(
        self,
        alignment: Dict[str, np.ndarray],
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> np.ndarray:
        """
        基于对齐信息读取数据并执行预处理逻辑（归一化）。
        """
        frames = self.fetch_raw_frames_from_alignment(alignment, start, end)
        return self.preprocess_weather_frames(frames, apply_normalization=True)

    def fetch_raw_frames_by_dates(self, dates: Sequence[pd.Timestamp]) -> np.ndarray:
        """根据日期列表读取原始数据（便捷接口）。"""
        return self.fetch_raw_frames_from_alignment(self.build_alignment(dates))

    def fetch_frames_by_dates(self, dates: Sequence[pd.Timestamp]) -> np.ndarray:
        """根据日期列表读取预处理后的数据（便捷接口）。"""
        return self.fetch_frames_from_alignment(self.build_alignment(dates))



class FullMapConvTimeXerQuantile(nn.Module):
    """
    基于全图卷积与 TimeXer 的分位数负荷预测模型。
    
    该模型集成了气象空间特征提取器与时间序列预测模型，能够：
    1. 提取 2D 气象网格的空间特征。
    2. 将气象特征作为外部变量 (Exogenous Variables) 输入 TimeXer。
    3. 输出多概率分布下的分位数预测结果。
    """
    def __init__(self, configs, quantiles: Sequence[float]):
        """
        初始化模型。
        
        参数:
            configs: 包含模型超参数的配置对象。
            quantiles: 需要预测的分位数列表（如 [0.1, 0.5, 0.9]）。
        """
        super().__init__()
        self.quantiles = list(quantiles)
        self.n_quantiles = len(self.quantiles)
        self.weather_feature_dim = int(configs.weather_feature_dim)
        # 分块编码大小，防止显存溢出
        self.encode_chunk_size = int(getattr(configs, "weather_encode_chunk_size", 512))
        
        # 1. 初始化气象特征提取器（Backbone）
        self.weather_backbone = FullMapWeatherConvExtractor(
            in_channels=int(getattr(configs, "weather_in_channels")),
            out_channels=self.weather_feature_dim,
            kernel_height=int(getattr(configs, "weather_kernel_height")),
            kernel_width=int(getattr(configs, "weather_kernel_width")),
            dropout=float(getattr(configs, "dropout", 0.1)),
        )
        
        self.weather_seq_len = int(getattr(configs, "weather_seq_len", configs.seq_len))
        # 更新配置以匹配 TimeXer 的外部变量维度需求
        configs.exo_seq_len = self.weather_seq_len
        configs.enc_in = 1 # 负荷数据一般为单变量
        
        # 2. 初始化预测主干模型 (TimeXer)
        self.timexer = TimeXer(configs)
        
        # 3. 初始化分位数回归头：将点预测投影到多个分位数平面
        self.quantile_head = nn.Linear(1, self.n_quantiles)
        
        # 初始化权重：使初始预测接近中性，提高训练稳定性
        with torch.no_grad():
            self.quantile_head.weight.fill_(1.0)
            self.quantile_head.bias.copy_(torch.tensor([q - 0.5 for q in self.quantiles]) * 0.1)

    def _encode_weather_frames(self, weather_frames: torch.Tensor) -> torch.Tensor:
        """
        使用 Backbone 对气象帧进行批量编码。
        采用分块处理机制，兼顾效率与显存占用。
        """
        if weather_frames.ndim != 4:
            raise ValueError(f"气象帧维度错误，预期 [N, C, H, W]，实际: {tuple(weather_frames.shape)}")
            
        encoded_chunks: List[torch.Tensor] = []
        for start in range(0, weather_frames.shape[0], self.encode_chunk_size):
            end = min(start + self.encode_chunk_size, weather_frames.shape[0])
            # float() 确保数据类型正确
            encoded_chunks.append(self.weather_backbone(weather_frames[start:end].float()))
        return torch.cat(encoded_chunks, dim=0)

    def _encode_weather_sequence(
        self,
        weather_seq: Optional[torch.Tensor],
        weather_index: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        将原始气象输入转换为语义特征序列 [B, T, D]。
        支持两种模式：
        1. 索引模式：输入为独特的帧池 [U, C, H, W] 与采样索引 [B, T]。
        2. 序列模式：直接输入 5D 序列张量 [B, T, C, H, W]。
        """
        if weather_seq is None:
            return None
            
        # 模式 1: 使用索引从帧池中提取
        if weather_index is not None:
            if weather_seq.ndim != 4 or weather_index.ndim != 2:
                raise ValueError("索引模式下，气象输入需为 [U,C,H,W] 且索引需为 [B,T]。")
            batch_size, time_len = weather_index.shape
            # 提取全量特征
            encoded_frames = self._encode_weather_frames(weather_seq)
            # 根据索引 Gather 到对应的位置
            gathered = encoded_frames.index_select(0, weather_index.reshape(-1))
            return gathered.reshape(batch_size, time_len, self.weather_feature_dim)
            
        # 模式 2: 直接对全量序列进行编码
        if weather_seq.ndim != 5:
            raise ValueError(f"序列模式下，气象输入需为 [B, T, C, H, W]，实际: {tuple(weather_seq.shape)}")
        batch_size, time_len, channels, height, width = weather_seq.shape
        # 展平批次和时间维度以进行高效并发提取
        flat = weather_seq.reshape(batch_size * time_len, channels, height, width)
        encoded = self._encode_weather_frames(flat)
        return encoded.reshape(batch_size, time_len, self.weather_feature_dim)

    def forward(
        self,
        load_x: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_exo_mark: torch.Tensor,
        weather_x: torch.Tensor,
        weather_x_index: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        端到端预测流程。
        """
        # 1. 编码气象特征
        weather_feature = self._encode_weather_sequence(weather_x, weather_x_index)
        
        # 2. TimeXer 预测：融合负荷、时间戳、气象特征
        point_pred = self.timexer(
            load_x,
            x_mark_enc,
            None,
            None,
            mask=mask,
            x_exo=weather_feature,
            x_exo_mark=x_exo_mark,
        )
        
        # 3. 截取预测窗口并应用分位数回归头
        point_pred = point_pred[:, -self.timexer.pred_len :, :]
        return self.quantile_head(point_pred)



class LoadWeatherEndToEndDataset(Dataset):
    """
    端到端负荷与气象网格耦合数据集。
    
    该类实现了负荷时间序列（CSV）与空间气象网格（HDF5）的深度集成。其核心特性包括：
    1. 自动时间对齐：确保每个负荷预测窗口对应正确的气象历史与未来窗口。
    2. 气象预处理缓存：将气象网格预先加载至内存，极大提升训练时的 IO 吞吐。
    3. 索引池化技术：在构建 Batch 时，通过 Unique 索引提取独立的帧池，通过索引映射减少显存占用。
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
        初始化数据集。
        """
        if size is None:
            size = [args.seq_len, args.label_len, args.pred_len]
        self.args = args
        self.seq_len = int(size[0])    # 历史步长
        self.label_len = int(size[1])  # 标签步长 (Informalers 等模型需要)
        self.pred_len = int(size[2])   # 预测步长
        self.target = target or args.target
        self.scale = bool(scale)
        self.timeenc = int(timeenc)
        self.freq = freq or args.freq
        self.weather_store = weather_store
        
        self.load_freq = _ensure_timedelta(self.freq)
        # 预计算相对偏移数组，用于后续快速切片
        self.seq_offsets = np.arange(self.seq_len, dtype=np.int64)
        self.target_offsets = (
            np.arange(self.label_len + self.pred_len, dtype=np.int64) + (self.seq_len - self.label_len)
        )

        # 配置气象序列参数
        self.weather_seq_len = int(getattr(args, "weather_seq_len", self.seq_len))
        weather_step_override = getattr(args, "weather_step_freq", None)
        if weather_step_override is not None:
            self.weather_freq = _ensure_timedelta(weather_step_override)
            self.weather_freq_str = (
                weather_step_override
                if isinstance(weather_step_override, str)
                else _timedelta_to_freq_str(self.weather_freq)
            )
        elif self.weather_store.native_freq is not None:
            self.weather_freq = self.weather_store.native_freq
            self.weather_freq_str = self.weather_store.native_freq_str or _timedelta_to_freq_str(
                self.weather_freq
            )
        else:
            raise RuntimeError("无法获取气象数据的原始频率。")

        # 推算气象窗口中的历史步长，确保时间跨度与负荷序列一致
        default_weather_history_len = infer_weather_history_len(
            seq_len=self.seq_len,
            load_freq=self.load_freq,
            weather_freq=self.weather_freq,
        )
        self.weather_history_len = int(getattr(args, "weather_history_len", default_weather_history_len))
        if self.weather_seq_len <= 0 or self.weather_history_len <= 0 or self.weather_seq_len < self.weather_history_len:
            raise ValueError(
                f"无效的气象窗口配置: seq_len={self.weather_seq_len}, history_len={self.weather_history_len}"
            )
        self.weather_future_len = self.weather_seq_len - self.weather_history_len

        # 气象编码特征的时间戳频率
        weather_mark_override = getattr(args, "weather_mark_freq", None)
        if weather_mark_override is not None:
            self.weather_mark_freq = (
                weather_mark_override
                if isinstance(weather_mark_override, str)
                else _timedelta_to_freq_str(weather_mark_override)
            )
        else:
            self.weather_mark_freq = self.weather_freq_str

        self.use_weather_normalization = bool(
            getattr(args, "use_weather_normalization", self.weather_store.use_channel_normalization)
        ) or self.weather_store.use_channel_normalization
        self.weather_norm_fit_chunk_size = int(getattr(args, "weather_norm_fit_chunk_size", 2048))

        # 数据集切分选项
        flag_map = {"train": 0, "val": 1, "test": 2}
        if flag not in flag_map:
            raise ValueError(f"flag 必须是 train/val/test, 得到: {flag}")
        self.set_type = flag_map[flag]

        # 初始化数据容器
        self.scaler: Optional[StandardScaler] = None
        self.target_mean = 0.0
        self.target_scale = 1.0
        self.data_x: Optional[np.ndarray] = None
        self.data_y: Optional[np.ndarray] = None
        self.data_stamp: Optional[np.ndarray] = None
        self.raw_dates: Optional[pd.Series] = None
        self.weather_timestamps: Optional[pd.DatetimeIndex] = None
        self.weather_stamp: Optional[np.ndarray] = None
        self.weather_window_positions: Optional[np.ndarray] = None
        self.weather_lookup: Optional[Dict[str, np.ndarray]] = None
        self.weather_cache: Optional[np.ndarray] = None

        self.__read_data__()

    def __read_data__(self) -> None:
        """
        核心数据加载函数：处理负荷数据切分、归一化、时间特征提取以及气象数据的内存预加载。
        """
        csv_path = os.path.join(self.args.root_path, self.args.data_path)
        df_raw = pd.read_csv(csv_path)
        if "date" not in df_raw.columns:
            raise ValueError(f"CSV 文件 {csv_path} 缺失 date 列")
        df_raw["date"] = pd.to_datetime(df_raw["date"])
        df_raw = df_raw.sort_values("date").reset_index(drop=True)
        # 统一目标列名
        if self.target not in df_raw.columns and "Target" in df_raw.columns:
            df_raw = df_raw.rename(columns={"Target": self.target})
        if self.target not in df_raw.columns:
            raise ValueError(f"CSV 缺失目标列 {self.target}")

        # 1. 划分数据集边界
        total_len = len(df_raw)
        num_train = int(total_len * 2 / 3)
        num_test = int(total_len * 1 / 6)
        num_vali = total_len - num_train - num_test
        border1s = [0, max(0, num_train - self.seq_len), max(0, num_train + num_vali - self.seq_len)]
        border2s = [num_train, num_train + num_vali, total_len]
        train_dates = pd.DatetimeIndex(df_raw["date"].iloc[: border2s[0]].to_numpy())
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 2. 负荷数据归一化
        target_values = df_raw[[self.target]].values.astype(np.float32)
        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(target_values[: border2s[0]])
            target_values = self.scaler.transform(target_values).astype(np.float32)
            self.target_mean = float(self.scaler.mean_[0])
            self.target_scale = float(self.scaler.scale_[0]) if self.scaler.scale_[0] != 0 else 1.0

        # 3. 负荷时间特征提取
        df_stamp = df_raw[["date"]].iloc[border1:border2].copy()
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp["date"].apply(lambda row: row.month)
            df_stamp["day"] = df_stamp["date"].apply(lambda row: row.day)
            df_stamp["weekday"] = df_stamp["date"].apply(lambda row: row.weekday())
            df_stamp["hour"] = df_stamp["date"].apply(lambda row: row.hour)
            df_stamp["minute"] = df_stamp["date"].apply(lambda row: row.minute)
            data_stamp = df_stamp.drop(columns=["date"]).values.astype(np.float32)
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp["date"].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0).astype(np.float32)

        self.data_x = target_values[border1:border2]
        self.data_y = target_values[border1:border2]
        self.data_stamp = data_stamp
        self.raw_dates = df_raw["date"].iloc[border1:border2].reset_index(drop=True)

        # 4. 构建气象时间窗口调度表
        reference_weather_timestamps_ns = np.concatenate(
            [source["timestamps_ns"] for source in self.weather_store.sources],
            axis=0,
        )
        weather_timeline_ns, weather_window_positions = _build_weather_window_schedule(
            load_dates=self.raw_dates,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            weather_seq_len=self.weather_seq_len,
            weather_history_len=self.weather_history_len,
            weather_freq=self.weather_freq,
            reference_timestamps_ns=reference_weather_timestamps_ns,
        )
        self.weather_timestamps = pd.DatetimeIndex(pd.to_datetime(weather_timeline_ns))
        self.weather_window_positions = weather_window_positions
        # 预先执行一次对齐查询，锁定 H5 的行索引
        self.weather_lookup = self.weather_store.build_alignment(self.weather_timestamps)
        # 提取气象对应的时间特征
        self.weather_stamp = time_features(
            pd.to_datetime(self.weather_timestamps.values),
            freq=self.weather_mark_freq,
        ).transpose(1, 0).astype(np.float32)

        # 5. 气象网格预加载到内存 (重要优化)
        print(f"[dataset-{self.set_type}] 正在预加载气象网格...")
        t0 = time.time()
        if self.use_weather_normalization and not self.weather_store.has_fitted_channel_normalization():
            # 如果是训练集且尚未拟合，则先拟合统计量
            if self.set_type == 0:
                raw_weather_cache = self.weather_store.fetch_raw_frames_from_alignment(
                    self.weather_lookup, 0, len(self.weather_timestamps)
                )
                self.weather_store.fit_channel_normalization_from_frames(
                    raw_weather_cache,
                    chunk_size=self.weather_norm_fit_chunk_size,
                    stage_name="train cache",
                )
                self.weather_cache = self.weather_store.preprocess_weather_frames(
                    raw_weather_cache,
                    apply_normalization=True,
                )
            else:
                # 验证/测试集：如果尚未拟合，需强制使用训练集时间段进行拟合
                train_weather_timeline_ns, _ = _build_weather_window_schedule(
                    load_dates=train_dates,
                    seq_len=self.seq_len,
                    pred_len=self.pred_len,
                    weather_seq_len=self.weather_seq_len,
                    weather_history_len=self.weather_history_len,
                    weather_freq=self.weather_freq,
                    reference_timestamps_ns=reference_weather_timestamps_ns,
                )
                self.weather_store.fit_channel_normalization_from_dates(
                    pd.DatetimeIndex(pd.to_datetime(train_weather_timeline_ns)),
                    chunk_size=self.weather_norm_fit_chunk_size,
                    stage_name="train dates",
                )
                self.weather_cache = self.weather_store.fetch_frames_from_alignment(
                    self.weather_lookup, 0, len(self.weather_timestamps)
                )
        else:
            # 常规加载流水线
            self.weather_cache = self.weather_store.fetch_frames_from_alignment(
                self.weather_lookup, 0, len(self.weather_timestamps)
            )

        # 内存占用统计
        mem_gb = self.weather_cache.nbytes / (1024 ** 3)
        print(
            f"[dataset-{self.set_type}] 气象缓存就绪: 形状={self.weather_cache.shape}, "
            f"占用显存={mem_gb:.2f} GB, 耗时={time.time() - t0:.1f}s"
        )
        print(
            f"[dataset-{self.set_type}] 窗口配置: step={self.weather_freq_str}, "
            f"seq_len={self.weather_seq_len}, 历史={self.weather_history_len}, "
            f"未来={self.weather_future_len}"
        )

    def __getitem__(self, index: int):
        """
        这里返回索引，具体的数据构筑由 build_overlap_batch 或 Collator 完成。
        """
        return int(index)

    def __len__(self) -> int:
        """返回数据集的样本总量"""
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def build_overlap_batch(
        self,
        batch_indices: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        高效批次构筑函数。
        
        该方法针对深度学习模型对气象网格的处理进行了专门优化：
        1. 针对一个 Batch 中的多个样本，提取其负荷、时间戳、气象标记数据。
        2. 气象网格优化：
           - 识别 Batch 中所有时间步请求的所有气象帧。
           - 提取唯一的帧集合（帧池），大大减少了多个样本重合时间带来的数据冗余。
           - 返回帧池和对应的池内索引映射关系。
        """
        if (
            self.data_x is None
            or self.data_y is None
            or self.data_stamp is None
            or self.weather_cache is None
            or self.weather_stamp is None
            or self.weather_window_positions is None
        ):
            raise RuntimeError("数据集尚未正确初始化。")
            
        indices = np.asarray(batch_indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError(f"batch_indices 必须是非空的一维数组，当前形状: {indices.shape}")
            
        # 1. 快速切片构建负荷窗口与目标窗口
        seq_positions = indices[:, None] + self.seq_offsets[None, :]
        target_positions = indices[:, None] + self.target_offsets[None, :]
        weather_positions = self.weather_window_positions[indices]
        
        batch_x = torch.from_numpy(np.ascontiguousarray(self.data_x[seq_positions]))
        batch_y = torch.from_numpy(np.ascontiguousarray(self.data_y[target_positions]))
        batch_x_mark = torch.from_numpy(np.ascontiguousarray(self.data_stamp[seq_positions]))
        
        # 2. 气象标记特征提取
        batch_exo_mark = torch.from_numpy(np.ascontiguousarray(self.weather_stamp[weather_positions]))
        
        # 3. 气象帧池化优化 (Pool & Index)
        # 获取 Batch 内请求的所有互异气象点
        unique_weather_idx, inverse = np.unique(weather_positions.reshape(-1), return_inverse=True)
        # 仅将 Batch 内需要的帧提取出来（大幅减少数据传输开销）
        weather_frames = torch.from_numpy(np.ascontiguousarray(self.weather_cache[unique_weather_idx]))
        # 重新映射索引到帧池
        weather_index = torch.from_numpy(
            np.ascontiguousarray(inverse.reshape(len(indices), self.weather_seq_len))
        )
        
        return batch_x, batch_y, batch_x_mark, batch_exo_mark, weather_frames, weather_index

    def scale_target(self, data: np.ndarray) -> np.ndarray:
        """对输入负荷数据应用归一化。"""
        data = np.asarray(data, dtype=np.float32).reshape(-1, 1)
        if not self.scale or self.scaler is None:
            return data.astype(np.float32)
        return self.scaler.transform(data).astype(np.float32)

    def inverse_transform_target(self, data: np.ndarray) -> np.ndarray:
        """对归一化后的负荷数据执行反换算（用于获得原始数值）。"""
        data = np.asarray(data, dtype=np.float32)
        if not self.scale:
            return data
        return data * self.target_scale + self.target_mean



class ContiguousWindowBatchSampler(Sampler[List[int]]):
    """
    连续窗口批次采样器。
    
    与 PyTorch 默认随机采样不同，该采样器以“块 (Block)”为单位进行采样：
    1. 每个块内部的样本索引是连续的（例如 [0,1,2,3], [4,5,6,7]）。
    2. 在不同块之间运行随机洗牌。
    
    设计目的：在某些模型中，保持 Batch 内样本的时间连续性有助于利用本地缓存或符合特定的时间演化假设。
    """
    def __init__(self, dataset_len: int, batch_size: int, drop_last: bool = False):
        if dataset_len <= 0 or batch_size <= 0:
            raise ValueError(f"无效的采样器参数: dataset_len={dataset_len}, batch_size={batch_size}")
        self.dataset_len = int(dataset_len)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)

    def __iter__(self) -> Iterator[List[int]]:
        # 1. 计算所有连续块的起始位置
        block_starts = np.arange(0, self.dataset_len, self.batch_size, dtype=np.int64)
        # 2. 对块的起始顺序进行洗牌
        np.random.shuffle(block_starts)
        
        for start in block_starts.tolist():
            end = start + self.batch_size
            if end > self.dataset_len:
                if self.drop_last:
                    continue
                end = self.dataset_len
            # 生成当前块的全部连续索引
            yield list(range(start, end))

    def __len__(self) -> int:
        if self.drop_last:
            return self.dataset_len // self.batch_size
        return (self.dataset_len + self.batch_size - 1) // self.batch_size


class OverlapAwareBatchCollator:
    """
    Overlap 感知批次整理器。
    
    该类作为一个简单的桥接器，将 DataLoader 提供的索引列表传递给数据集的 
    `build_overlap_batch` 方法。它是实现“池化索引”气象帧提取的关键入口。
    """
    def __init__(self, dataset: LoadWeatherEndToEndDataset):
        self.dataset = dataset

    def __call__(
        self,
        batch: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 委托数据集执行复杂的 Batch 构筑逻辑
        return self.dataset.build_overlap_batch(batch)


def weather_data_provider(args, flag: str, weather_store: WeatherGridStore):
    """
    气象耦合负荷预测数据提供者（Factory 函数）。
    
    参数:
        args: 全局配置对象。
        flag: 数据集类型 ('train', 'val', 'test')。
        weather_store: 已初始化的气象存储库实例。
        
    返回:
        dataset: 初始化后的 LoadWeatherEndToEndDataset 实例。
        loader: 配置好的 DataLoader 实例。
    """
    timeenc = 0 if args.embed != "timeF" else 1
    shuffle_flag = flag == "train" # 仅训练集默认洗牌
    
    # 检查是否使用特殊的连续窗口采样
    use_contiguous_train_batches = flag == "train" and bool(getattr(args, "contiguous_train_batches", False))
    # GPU 内存锁定优化
    use_pin_memory = bool(getattr(args, "pin_memory", False)) and torch.cuda.is_available() and bool(getattr(args, "use_gpu", False))
    
    # 1. 构造数据集实例
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
    
    # 2. 构造批次整理器
    collate_fn = OverlapAwareBatchCollator(dataset)
    
    # 3. 根据配置构造 DataLoader
    if use_contiguous_train_batches:
        # 使用自定义的块采样器
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
        # 使用标准的随机采样器
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
    "build_weather_sequence_timestamps",
    "infer_weather_history_len",
    "weather_data_provider",
]
