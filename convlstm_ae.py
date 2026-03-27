"""
ConvLSTM AutoEncoder for meteorological spatiotemporal sequence encoding.

Usage:
    python convlstm_ae.py
    python convlstm_ae.py --smoke-test
    python convlstm_ae.py --test-only
"""

import argparse
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import h5py

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from utils.tools import EarlyStopping, adjust_learning_rate

# =============================================================================
# 全局配置 (Global Config) - 超参数、路径、数据字段等的收口定义
# =============================================================================

# 指定气象网格数据集存储路径
H5_PATH = "./data/hunan_grid_2024_2025_filtered.h5"
LEGACY_H5_PATH = "./data/hunan_grid_meteo_20250101_20260228.h5"

# 旧版 10 通道气象字段定义，作为变量名缺失时的兜底映射
LEGACY_CHANNEL_NAMES = [
    "temperature_2m",         # 0: 2米温度
    "relative_humidity_2m",   # 1: 2米相对湿度
    "apparent_temperature",   # 2: 体感温度
    "dew_point_2m",           # 3: 2米露点温度
    "surface_pressure",       # 4: 地表气压
    "cloud_cover",            # 5: 云量
    "wind_speed_10m",         # 6: 10米风速
    "shortwave_radiation",    # 7: 短波辐射
    "direct_radiation",       # 8: 直接辐射
    "precipitation",          # 9: 降水
]

# 降水等符合长尾分布规律的指标极其容易受极端天气 (如洪涝暴雨) 产生巨大数值偏移
# 使用 log1p(x+1) 能有效将极值拖缩回去，在网络训练中更容易学到连续特征
LOG1P_VARIABLE_NAMES = {"precipitation"}
DEFAULT_WINDOW_HOURS = 24
DEFAULT_STRIDE_HOURS = 6

# ----------------- 模型超参数 (Model Hyperparameters) -----------------
WINDOW_SIZE = 24     # 默认窗口大小：对 hourly 数据为 1 天；15min 数据会在运行时自动推断为 96
HIDDEN_DIM = 16      # ConvLSTM 的隐藏特征通道层数，提取出的特征更抽象，参数也随之成平方倍增加
LATENT_DIM = 128      # 最后编码得到的潜向量长度映射 (决定了后续在检索库搜索时向量的体积大小)
NUM_LAYERS = 1       # 级联的 ConvLSTM 提取深度层数 (1层通常具备基础表征，多层计算较慢)
IN_CHANNELS = 5      # 默认兼容 hourly 版；实际训练/测试时会根据 H5 元数据自动覆盖

# ----------------- 训练控制超参数 (Training Hyperparameters) ----------
BATCH_SIZE = 16      # 每批次包含的天数样本，受限于 4D 张量 (B, T, C, H, W) 的极长的时间步长，极易发生显存崩塌
LEARNING_RATE = 1e-3 # Adam优化器的初始学习率
EPOCHS = 70          # 模型全参数迭代的最高轮数
TRAIN_RATIO = 0.8    # 时序前 80% 用于自回归拟合生成，后 20% 用于客观衡量它是否学到了"重建规律"
NUM_WORKERS = 0      # DataLoader 读取线程，0 表示主线程取，在 Windows 和复杂 H5 索引时避免死锁首选

# ----------------- 路径超参数 (Path Configuration) --------------------
CHECKPOINT_DIR = "./checkpoints_ae/"          # 模型权重和统计参数落地的路径
BEST_MODEL_FILE = "convlstm_ae_best.pth"      # 最优权重备份名称
NORM_STATS_FILE = "norm_stats.npz"            # 特别保存好的均值 (mean) /方差 (std)，测试验证必须挂靠使用

# ----------------- 计算设备超参数 (Device Hyperparameters) ------------
USE_GPU = True       # 标志开关：能开显卡绝对要开显卡
GPU_ID = 0           # 在多显卡环境下的定点指向


# =============================================================================
# Model
# =============================================================================


class ConvLSTMCell(nn.Module):
    """
    ConvLSTM 单元：将传统 LSTM 的全连接运算替换为 2D 卷积运算，
    能够在时序门控机制中保留空间局部特征（气象网格的空间相关性）。
    
    参数:
        in_channels: 每次输入的通道数（例如原始气象变量数或上一层的隐藏通道数）
        hidden_channels: 隐藏状态序列的通道数
        kernel_size: 卷积核大小，默认为3（3x3卷积）
    """
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2  # 保持卷积前后空间大小一致（Same Padding）
        
        # 将输入序列特征图与上一步的隐藏状态组合后共同做卷积
        # 输出通道数为 4 * hidden_channels，分别对应 i(输入门), f(遗忘门), o(输出门), g(细胞状态候选)
        self.gates = nn.Conv2d(
            in_channels=in_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播计算单步时间的一帧。
        输入:
            x_t: 当前时间步 t 的输入，形状 [Batch, in_channels, H, W]
            h_prev: 前一时间步 t-1 的隐藏状态，形状 [Batch, hidden_channels, H, W]
            c_prev: 前一时间步 t-1 的细胞状态，形状 [Batch, hidden_channels, H, W]
        输出:
            h_t: 当前时间步 t 的隐藏状态，形状 [Batch, hidden_channels, H, W]
            c_t: 当前时间步 t 的细胞状态，形状 [Batch, hidden_channels, H, W]
        """
        # 沿通道维度 (dim=1) 拼接输入与上一时刻隐状态 -> [B, in+hidden, H, W]
        combined = torch.cat([x_t, h_prev], dim=1)
        
        # 通过卷积同时计算四个门 -> [B, 4*hidden, H, W]
        gates = self.gates(combined)
        
        # 沿着通道维度切分为四等份，分别给四种门控信号
        i, f, o, g = gates.chunk(4, dim=1)

        # 门控信号使用 Sigmoid 激活函数，将其值压缩到 [0, 1] 之间（代表开启程度）
        i = torch.sigmoid(i)  # Input Gate，决定吸取多少新的当前输入信息
        f = torch.sigmoid(f)  # Forget Gate，决定遗忘多少上一时刻的细胞状态信息
        o = torch.sigmoid(o)  # Output Gate，决定输出多少细胞状态作为当前的隐状态
        
        # 候选信号使用 Tanh 激活函数，将其值压缩到 [-1, 1] 之间
        g = torch.tanh(g)     # Cell Update Candidate，当前输入所提供的新信息

        # 更新当前时刻的细胞状态：保留上一时刻未被遗忘的部分 + 加入当前的新信息
        c_t = f * c_prev + i * g
        
        # 更新当前时刻的隐藏状态：输出门决定输出比例，tanh压缩细胞状态
        h_t = o * torch.tanh(c_t)
        
        return h_t, c_t


class ConvLSTM(nn.Module):
    """
    多层 ConvLSTM 序列处理模块（封装完整的时间轴迭代逻辑）。
    
    参数:
        in_channels: 输入多变量时间序列帧的通道数
        hidden_channels: 隐藏层提取的特征通道数
        num_layers: 堆叠的 ConvLSTMCell 层数
        kernel_size: 卷积核大小
        return_all_steps: 若为True，返回每一个时间步的隐状态（Decoder使用）；若为False，只返回最后一个时间步（Encoder使用）。
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 1,
        kernel_size: int = 3,
        return_all_steps: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.return_all_steps = return_all_steps

        # 构建多层 ConvLSTMCell
        layers = []
        for i in range(num_layers):
            # 首层输入为实际通道数，其余层输入为上一层的隐藏通道数
            layer_in = in_channels if i == 0 else hidden_channels
            layers.append(ConvLSTMCell(layer_in, hidden_channels, kernel_size))
        self.layers = nn.ModuleList(layers)

    def _init_hidden(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        初始化每个 ConvLSTM 层的隐藏状态 (h) 和细胞状态 (c) 为零张量。
        返回 list: [(h_0, c_0) for layer 1, (h_0, c_0) for layer 2, ...]
        """
        states = []
        for _ in range(self.num_layers):
            h = torch.zeros(batch_size, self.hidden_channels, height, width, device=device, dtype=dtype)
            c = torch.zeros(batch_size, self.hidden_channels, height, width, device=device, dtype=dtype)
            states.append((h, c))
        return states

    def forward(
        self,
        x: torch.Tensor,
        initial_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """
        输入:
            x: 时空序列输入，形状 [Batch, seq_len(时间步数), Channels, Height, Width]
        输出:
            return_all_steps=True: 形状 [Batch, seq_len, hidden_channels, Height, Width]
            return_all_steps=False: 形状 [Batch, hidden_channels, Height, Width] (仅最后一步)
        """
        batch_size, seq_len, _, height, width = x.shape
        
        # 1. 状态初始化：若没有传入，则初始化为全零状态
        states = initial_states or self._init_hidden(batch_size, height, width, x.device, x.dtype)

        all_outputs = []
        
        # 2. 沿着时间步(t)逐步展开序列
        for t in range(seq_len):
            x_t = x[:, t]  # 获取第 t 步输入，形状 [Batch, Channels, H, W]
            
            # 3. 数据穿过多个隐层
            for layer_idx, cell in enumerate(self.layers):
                h_prev, c_prev = states[layer_idx]
                
                # 执行单步 ConvLSTM 计算
                h_t, c_t = cell(x_t, h_prev, c_prev)
                states[layer_idx] = (h_t, c_t) # 保存更新后的状态
                
                # 当前层的隐状态输出作为下一层的输入
                x_t = h_t
                
            # 记录最后一层在时刻 t 的输出隐状态
            if self.return_all_steps:
                all_outputs.append(h_t)

        # 根据配置决定返回整个序列还是单个时间的隐状态
        if self.return_all_steps:
            return torch.stack(all_outputs, dim=1) # 拼成序列维度
        return h_t # 返回最后时间步


class Encoder(nn.Module):
    """
    编码器：负责将高维气象时空序列 [B, 96, 10, H, W] 压缩为低维连续潜向量 [B, latent_dim]
    """
    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        hidden_channels: int = HIDDEN_DIM,
        latent_dim: int = LATENT_DIM,
        num_layers: int = NUM_LAYERS,
    ):
        super().__init__()
        # 使用 ConvLSTM 提取包含时空信息的特征
        self.convlstm = ConvLSTM(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            return_all_steps=False, # 编码器只需取时间序列最后的全局表示
        )
        # 自适应平均池化，由于直接降到 1x1 会丢失所有拓扑空间信号，改为保留 4x4 的特征代表值
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        # 将池化后的隐藏层 (原 hidden_channels * 4 * 4) 映射为潜向量维度
        self.fc = nn.Linear(hidden_channels * 16, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 过 ConvLSTM 处理完整序列长度，获取最后时间步 -> [B, hidden_channels, H, W]
        h_last = self.convlstm(x)
        
        # 2. 空间拉平 (Pooling) -> 去除空间网格特征 -> [B, hidden_channels, 1, 1]
        pooled = self.pool(h_last)
        
        # 3. 展平并得到纯全连接向量 -> [B, hidden_channels]
        flat = pooled.flatten(1)
        
        # 4. 映射到指定的表示维度 -> [B, latent_dim]
        return self.fc(flat)


class Decoder(nn.Module):
    """
    解码器：负责将低维潜向量 [B, latent_dim] 解压、广播并重建回气象序列 [B, 96, 10, H, W]
    主要是为了强制潜向量保留关键的生成信息，以用于相似日度量和对比学习约束。
    """
    def __init__(
        self,
        out_channels: int = IN_CHANNELS,
        hidden_channels: int = HIDDEN_DIM,
        latent_dim: int = LATENT_DIM,
        num_layers: int = NUM_LAYERS,
        seq_len: int = WINDOW_SIZE,
        frame_height: int = 62,
        frame_width: int = 61,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_channels = hidden_channels
        self.frame_height = frame_height
        self.frame_width = frame_width

        # 将低维向量直接映射放出原始的完整粗略空间拓扑张量 [hidden_channels * H * W]
        self.fc = nn.Linear(latent_dim, hidden_channels * frame_height * frame_width)
        
        # 解码的 ConvLSTM，作用是注入时域和空域特征
        self.convlstm = ConvLSTM(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            return_all_steps=True, # 解码阶段需回放重建每一个时刻
        )
        
        # 最后通过 1x1 卷积把隐藏特征映射为原始气象的特定多通道 -> [Batch, ...., out_channels, H, W]
        self.output_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        batch_size = latent.shape[0]
        
        # 1. 映射回包含了粗略地理空间拓扑的密集张量 -> [B, hidden_channels * H * W]
        feat = self.fc(latent)
        
        # 2. 折叠重建成真实的二维空间地理地图数组 -> [B, hidden_channels, H, W]
        feat = feat.view(batch_size, self.hidden_channels, self.frame_height, self.frame_width)
        
        # 3. 增加单帧时间维度 -> [B, 1, hidden_channels, H, W]
        feat = feat.unsqueeze(1)
        
        # 4. 在时间跨度序列上应用均匀的广播 (网络自己依此单帧画面推演地理时流)
        # -> [B, seq_len, hidden_channels, H, W]
        feat = feat.expand(
            batch_size,
            self.seq_len,
            self.hidden_channels,
            self.frame_height,
            self.frame_width,
        )

        # 3. 交给多层 ConvLSTM 重建成包含有时间动态变化的过程 -> [B, seq_len, hidden_channels, H, W]
        decoded = self.convlstm(feat)
        
        # 4. 把 Batch 和 Seq 取平铺作为伪 Batch 传入 1x1 Conv，用以从隐藏通道映射出各气象通道
        batch_size, seq_len, hidden_dim, height, width = decoded.shape
        decoded_flat = decoded.reshape(batch_size * seq_len, hidden_dim, height, width)
        
        # [B*seq_len, out_channels, H, W]
        output_flat = self.output_conv(decoded_flat)
        
        # 5. 形变恢复原来的五维张量返回 -> [B, seq_len, out_channels, H, W]
        return output_flat.reshape(batch_size, seq_len, -1, height, width)


class ConvLSTMAutoEncoder(nn.Module):
    """
    气象数据的时空自编码器总控类 (AutoEncoder = Encoder + Decoder)
    """
    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        hidden_channels: int = HIDDEN_DIM,
        latent_dim: int = LATENT_DIM,
        num_layers: int = NUM_LAYERS,
        seq_len: int = WINDOW_SIZE,
        frame_height: int = 62,
        frame_width: int = 61,
    ):
        super().__init__()
        # 实例化编码和解码模块
        self.encoder = Encoder(in_channels, hidden_channels, latent_dim, num_layers)
        self.decoder = Decoder(
            out_channels=in_channels,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            num_layers=num_layers,
            seq_len=seq_len,
            frame_height=frame_height,
            frame_width=frame_width,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        纯编码阶段，通常在特征提取 (检索过程) 时使用。
        """
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        全量前向传播，用于自建重构损失进行无监督训练。
        """
        # 第一阶段将空间输入压缩为定长向量
        latent = self.encoder(x)
        # 第二阶段利用向量信息重构图像输入
        return self.decoder(latent)


# =============================================================================
# Dataset and preprocessing
# =============================================================================


class MeteoDataset(Dataset):
    """
    负责将预处理后的全量气象数据 [Total_Steps, 10, H, W] 
    切分成适合送入模型的固定窗口 (window_size) 样本集。
    
    采用滑动窗口 (Sliding Window) 策略可以成十倍百倍地增加可用训练样本。
    """
    def __init__(self, data: np.ndarray, window_size: int = WINDOW_SIZE, stride: int = 24):
        super().__init__()
        self.window_size = int(window_size)
        self.stride = int(stride)
        if self.window_size <= 0:
            raise ValueError("window_size 必须为正数。")
        if self.stride <= 0:
            raise ValueError("stride 必须为正数。")
        
        # 保留下数据的引用
        self.data = data
        if len(self.data) < self.window_size:
            raise ValueError(
                f"数据长度 {len(self.data)} 小于窗口长度 {self.window_size}，无法构造滑动窗口样本。"
            )
        
        # 计算滑动窗口游标能截取到的样本总个数
        self.n_samples = (len(self.data) - self.window_size) // self.stride + 1

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int) -> torch.Tensor:
        # 在获取单条样本时再执行内存切片，转化为 GPU 计算所需的 Tensor
        start_idx = index * self.stride
        end_idx = start_idx + self.window_size
        return torch.from_numpy(self.data[start_idx:end_idx])

def _find_first_4d_dataset(h5_obj):
    """
    递归遍历 HDF5 对象，找到第一个形如 [T, C, H, W] 即 4 维的数据集 (Dataset)。
    由于 HDF5 本质是一棵类似于文件系统的树，我们通常将目标数据存在某一个叶子节点里。
    """
    for key in h5_obj.keys():
        item = h5_obj[key]
        if isinstance(item, h5py.Dataset) and item.ndim == 4:
            return item
        if isinstance(item, h5py.Group):
            found = _find_first_4d_dataset(item)
            if found is not None:
                return found
    return None


def _find_named_1d_dataset(h5_obj, candidate_names: Sequence[str]):
    """
    递归遍历 HDF5 对象，找到名称匹配的 1 维数据集。
    主要用于提取 variables / timestamps 之类的元数据。
    """
    normalized_candidates = tuple(str(name).lower() for name in candidate_names)
    for key in h5_obj.keys():
        item = h5_obj[key]
        key_lower = str(key).lower()
        if isinstance(item, h5py.Dataset) and item.ndim == 1:
            if any(candidate in key_lower for candidate in normalized_candidates):
                return item
        if isinstance(item, h5py.Group):
            found = _find_named_1d_dataset(item, candidate_names)
            if found is not None:
                return found
    return None


def _decode_string_array(values) -> List[str]:
    """
    将 HDF5 中的 bytes / object / string 数组统一转为 Python 字符串列表。
    """
    decoded = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, (bytes, bytearray)):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def _build_default_channel_names(channels: int) -> List[str]:
    """
    当 H5 内没有可靠的变量名时，构造一个稳定的兜底通道名列表。
    """
    if channels == len(LEGACY_CHANNEL_NAMES):
        return list(LEGACY_CHANNEL_NAMES)
    return [f"channel_{idx}" for idx in range(channels)]


def _infer_step_seconds_from_timestamps(timestamp_values: Sequence[str]) -> Optional[int]:
    """
    从时间戳数组中推断原始气象数据步长（秒）。
    """
    if len(timestamp_values) < 2:
        return None

    try:
        timestamps_ns = np.asarray(timestamp_values, dtype="datetime64[ns]").astype(np.int64)
    except (TypeError, ValueError):
        return None

    diffs = np.diff(timestamps_ns)
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.size == 0:
        return None

    step_seconds = int(positive_diffs.min() // 1_000_000_000)
    return step_seconds if step_seconds > 0 else None


def _infer_steps_for_hours(step_seconds: Optional[int], hours: int, fallback: int) -> int:
    """
    将“若干小时”的业务窗口换算为当前采样频率下的步数。
    例如:
    - 15min 数据: 24 小时 -> 96 步, 6 小时 -> 24 步
    - 1h 数据:    24 小时 -> 24 步, 6 小时 -> 6 步
    """
    if step_seconds is None or step_seconds <= 0:
        return int(fallback)

    total_seconds = int(hours) * 3600
    steps, remainder = divmod(total_seconds, step_seconds)
    if steps <= 0:
        return 1
    if remainder != 0:
        return max(1, int(round(total_seconds / step_seconds)))
    return int(steps)


def _resolve_log1p_channels(channel_names: Sequence[str]) -> List[int]:
    """
    根据变量名自动确定哪些通道应该做 log1p 平滑。
    """
    resolved = []
    for idx, channel_name in enumerate(channel_names):
        lowered = str(channel_name).lower()
        if any(token in lowered for token in LOG1P_VARIABLE_NAMES):
            resolved.append(idx)
    return resolved


def _resolve_h5_path(h5_path: str) -> str:
    """
    将相对路径解析为相对当前脚本目录的绝对路径。
    """
    if os.path.isabs(h5_path):
        return h5_path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, h5_path)


@dataclass
class MeteoRuntimeConfig:
    h5_path: str
    dataset_name: str
    total_steps: int
    in_channels: int
    frame_height: int
    frame_width: int
    channel_names: List[str]
    log1p_channels: List[int]
    step_seconds: Optional[int]
    window_size: int
    stride: int

    @property
    def dataset_tag(self) -> str:
        return os.path.splitext(os.path.basename(self.h5_path))[0]

    @property
    def step_desc(self) -> str:
        if self.step_seconds is None:
            return "unknown"
        if self.step_seconds % 3600 == 0:
            return f"{self.step_seconds // 3600}h"
        if self.step_seconds % 60 == 0:
            return f"{self.step_seconds // 60}min"
        return f"{self.step_seconds}s"


def inspect_h5_runtime_config(
    h5_path: str,
    window_size: Optional[int] = None,
    stride: Optional[int] = None,
) -> MeteoRuntimeConfig:
    """
    读取 H5 元数据，得到模型和数据管线真正需要的运行时配置。
    """
    resolved_h5_path = _resolve_h5_path(h5_path)
    if not os.path.exists(resolved_h5_path):
        raise FileNotFoundError(f"找不到 HDF5 文件: {resolved_h5_path}")

    with h5py.File(resolved_h5_path, "r") as f:
        dataset = _find_first_4d_dataset(f)
        if dataset is None:
            raise ValueError(f"HDF5 文件中未找到 4D 数据集: {resolved_h5_path}")

        total_steps, in_channels, frame_height, frame_width = dataset.shape

        variable_dataset = _find_named_1d_dataset(f, ("variables", "variable", "channel_names", "channels"))
        if variable_dataset is not None:
            channel_names = _decode_string_array(variable_dataset[...])
        else:
            channel_names = _build_default_channel_names(in_channels)

        if len(channel_names) != in_channels:
            print(
                f"[数据] 变量名数量 {len(channel_names)} 与通道数 {in_channels} 不一致，"
                "将回退到默认通道命名。"
            )
            channel_names = _build_default_channel_names(in_channels)

        timestamp_dataset = _find_named_1d_dataset(f, ("timestamps", "timestamp", "times", "time"))
        timestamp_values = _decode_string_array(timestamp_dataset[...]) if timestamp_dataset is not None else []

    step_seconds = _infer_step_seconds_from_timestamps(timestamp_values)
    effective_window_size = (
        int(window_size)
        if window_size is not None and int(window_size) > 0
        else _infer_steps_for_hours(step_seconds, DEFAULT_WINDOW_HOURS, WINDOW_SIZE)
    )
    effective_stride = (
        int(stride)
        if stride is not None and int(stride) > 0
        else _infer_steps_for_hours(step_seconds, DEFAULT_STRIDE_HOURS, max(1, effective_window_size // 4))
    )
    effective_log1p_channels = _resolve_log1p_channels(channel_names)

    return MeteoRuntimeConfig(
        h5_path=resolved_h5_path,
        dataset_name=dataset.name,
        total_steps=int(total_steps),
        in_channels=int(in_channels),
        frame_height=int(frame_height),
        frame_width=int(frame_width),
        channel_names=channel_names,
        log1p_channels=effective_log1p_channels,
        step_seconds=step_seconds,
        window_size=int(effective_window_size),
        stride=int(effective_stride),
    )


def load_and_preprocess(
    runtime_config: MeteoRuntimeConfig,
    train_ratio: float = TRAIN_RATIO,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    从 HDF5 中加载高维气象场数据，并执行预处理，包含两步：
    1. log1p 对指定长尾分布特征进行缩放。
    2. 基于分离出的训练集，在各个通道 (Channel) 维度计算并进行独立的 Z-Score 标准化。
    """
    if h5py is None:
        raise ImportError("需要先安装 h5py: pip install h5py")

    h5_path = runtime_config.h5_path
    print(f"[数据] 加载 HDF5: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        dataset = _find_first_4d_dataset(f)
        if dataset is None:
            raise ValueError(f"HDF5 文件中未找到 4D 数据集: {h5_path}")
        print(f"[数据] 数据集: {dataset.name}, shape={dataset.shape}")
        
        # 提取至内存，转为更省空间的 float32
        data = dataset[:].astype(np.float32)

    total_steps, channels, height, width = data.shape
    print(f"[数据] 总步数: {total_steps}, 通道数: {channels}, 网格: {height}x{width}")

    # 第 1 步：平滑处理
    # 大部分时候气象降水 (precipitation) 数据呈现严重的右偏长尾分布 (非常多 0，少数极端大暴雨)
    # log1p 即 log(x+1)，能压制极值，使得网络更容易学到降水的特征而不是被极值干扰导致的严重 loss 震荡
    for ch_idx in runtime_config.log1p_channels:
        channel_name = runtime_config.channel_names[ch_idx]
        print(f"[预处理] channel {ch_idx} ({channel_name}): 取 log1p(x+1) 进行异常值平滑")
        data[:, ch_idx, :, :] = np.log1p(np.maximum(data[:, ch_idx, :, :], 0.0))

    # 第 2 步：依据比例分割时序数据作为训练集和验证集
    n_train = int(total_steps * train_ratio)
    train_data = data[:n_train]
    val_data = data[n_train:]
    print(f"[数据] 训练集: {n_train} 步 | 验证集: {total_steps - n_train} 步")

    # 第 3 步：仅仅使用训练集的数据计算均值与标准差 (防止验证集/测试集信息泄露给模型)
    # 沿着时间 (dim=0), 高度 (dim=2), 宽度 (dim=3) 做求值，使得保留各个独立的通道均值 -> 存为 [1, Channels, 1, 1]
    mean = train_data.mean(axis=(0, 2, 3), keepdims=True)
    std = train_data.std(axis=(0, 2, 3), keepdims=True)
    
    # 避免有些通道全是0 (例如某地永远无雨) 的极端情况导致除零异常
    std = np.where(std < 1e-8, 1.0, std)

    print(f"[预处理] mean: {mean.flatten()}")
    print(f"[预处理] std:  {std.flatten()}")

    # 第 4 步：Z-Score 标准化操作，让模型对数据尺度不敏感，加速模型收敛
    train_data = (train_data - mean) / std
    val_data = (val_data - mean) / std

    # 返回预处理后的数据，同时返回参与处理的统计算子使得以后能逆向恢复真实值
    return train_data, val_data, mean, std, height, width


# =============================================================================
# Training
# =============================================================================


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
) -> float:
    """训练循环：训练一个完整的 Epoch 并返回平均 Loss"""
    model.train() # 切换至训练模式，启用 Dropout 等等
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        # 将取出的数据推入计算设备 (GPU或CPU)
        batch = batch.to(device=device, dtype=torch.float32)
        
        # 1. 梯度清零，防止多步累加。设置 set_to_none=True 比 =0 更快释放显存
        optimizer.zero_grad(set_to_none=True)

        # 2. 混合精度上下文 (AMP)
        # 启用自动混合精度 autocast ，自动判断用 Float16 计算和 Float32 ，极大下降显存消耗提升算力
        with torch.amp.autocast("cuda", enabled=use_amp):
            # 将原始完整图像时空结构丢给模型进行压缩与重构
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch) # 自监督无标签训练，原图也就是标签

        # 3. 反向传播与优化器步进
        # 因为在 FP16 下算部分梯度可能因为极小导致产生下溢出(Underflow)，所以引入 GradScaler 将其按比例放大到正常值
        scaler.scale(loss).backward()  # 先缩放 Loss 再反传
        scaler.step(optimizer)         # 优化器收到解缩放复原后的梯度并进行参数修正
        scaler.update()                # 自己根据当前迭代的结果自检更新比例因子

        total_loss += loss.item() # 只取数值，防止历史梯图残留在计算图中拖累显存
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> float:
    """
    验证循环：负责验证集上的性能评估测试。
    `@torch.inference_mode()` 装饰器彻底禁用了梯度追踪 (比 no_grad 更快)，提升预测速度。
    """
    model.eval() # 开启验证模式，禁止 Dropout 等等会改变状态的操作
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device=device, dtype=torch.float32)
        
        # 验证阶段同样可以使用半精度来加速前向传播的速度
        with torch.amp.autocast("cuda", enabled=use_amp):
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            
        total_loss += loss.item()
        n_batches += 1

    # 返回验证集的平均重构 Loss
    return total_loss / max(n_batches, 1)


@torch.inference_mode()
def evaluate_reconstruction_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> dict:
    """
    计算验证集上的重构指标（MSE, MAE, RMSE, R2）。
    
    参数:
        model: 待评估的 ConvLSTM AutoEncoder 模型
        loader: 验证集的数据加载器
        device: 运行设备 (CPU/CUDA)
        use_amp: 是否开启自动混合精度 (AMP) 推理
        
    返回:
        包含各项指标的字典 {"mse": ..., "mae": ..., "rmse": ..., "r2": ...}
    """
    model.eval()
    total_squared_error = 0.0  # 累计平方误差 (L2)
    total_absolute_error = 0.0 # 累计绝对误差 (L1)
    total_elements = 0         # 总像素点/特征点数
    target_sum = 0.0           # 目标值的总和 (计算 R2 用)
    target_squared_sum = 0.0   # 目标值的平方和 (计算 R2 用)

    for batch in loader:
        # 将批次数据移至指定设备
        batch = batch.to(device=device, dtype=torch.float32)

        # 开启自动混合精度推理
        with torch.amp.autocast("cuda", enabled=use_amp):
            reconstructed = model(batch)

        # 计算残差 (重建值 - 原始值)
        diff = reconstructed - batch
        
        # 统计各项误差
        total_squared_error += diff.square().sum().item()
        total_absolute_error += diff.abs().sum().item()
        total_elements += diff.numel()
        
        # 统计标签数据的基本信息，用于后续计算方差
        target_sum += batch.sum().item()
        target_squared_sum += batch.square().sum().item()

    # 计算均方误差 (MSE) 和 平均绝对误差 (MAE)
    mse = total_squared_error / max(total_elements, 1)
    mae = total_absolute_error / max(total_elements, 1)
    # 计算均方根误差 (RMSE)
    rmse = float(np.sqrt(mse))
    
    # 计算 R2 分数 (决定系数)
    # R2 = 1 - SSR / SST
    target_mean = target_sum / max(total_elements, 1)
    # 总体平方和 (Total Sum of Squares)
    total_variance = target_squared_sum - total_elements * (target_mean ** 2)
    
    if total_variance <= 1e-12:
        r2 = 0.0  # 防止除以 0
    else:
        r2 = 1.0 - (total_squared_error / total_variance)
        
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}


def _resolve_checkpoint_dir(base_checkpoint_dir: str, runtime_config: MeteoRuntimeConfig) -> str:
    """
    默认情况下按数据集文件名拆分 checkpoint 目录，避免不同通道配置互相覆盖。
    如果用户显式传了自定义目录，则保持原样。
    """
    normalized_base = os.path.normpath(base_checkpoint_dir)
    default_base = os.path.normpath(CHECKPOINT_DIR)
    if normalized_base == default_base:
        dataset_dir = os.path.join(base_checkpoint_dir, runtime_config.dataset_tag)
        return dataset_dir
    return base_checkpoint_dir


def run_training(args: argparse.Namespace) -> None:
    """
    执行完整的模型训练流程。
    包括设备检测、数据预处理、模型加载、主循环训练、验证及早停。
    """
    # 1. 基本参数合法性检查
    if args.train_epochs <= 0:
        raise ValueError("train_epochs 必须为正数")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate 必须为正数")
    if args.patience <= 0:
        raise ValueError("patience 必须为正数")

    # 2. 设置训练设备 (GPU 或 CPU)
    if USE_GPU and torch.cuda.is_available():
        device = torch.device(f"cuda:{GPU_ID}")
        print(f"[设备] 环境检测完毕，使用 GPU: {device}")
    else:
        device = torch.device("cpu")
        print("[设备] 未检测到 GPU 或未配置使用，采用 CPU 训练")

    # 3. 数据加载与标准化预处理
    runtime_config = inspect_h5_runtime_config(
        args.h5_path,
        window_size=args.window_size,
        stride=args.stride,
    )
    train_data, val_data, mean, std, height, width = load_and_preprocess(
        runtime_config, TRAIN_RATIO
    )

    # 4. 初始化 Dataset 和 DataLoader
    train_dataset = MeteoDataset(
        train_data,
        window_size=runtime_config.window_size,
        stride=runtime_config.stride,
    )
    val_dataset = MeteoDataset(
        val_data,
        window_size=runtime_config.window_size,
        stride=runtime_config.stride,
    )
    print(f"[数据] 训练集样本: {len(train_dataset)} | 验证集样本: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True, # 训练时开启打乱
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, # 验证时不打乱
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    # 5. 初始化模型 (ConvLSTM AutoEncoder)
    model = ConvLSTMAutoEncoder(
        in_channels=runtime_config.in_channels,
        hidden_channels=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_layers=NUM_LAYERS,
        seq_len=runtime_config.window_size,
        frame_height=height,
        frame_width=width,
    ).float().to(device)

    # 打印模型参数规模
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[模型] 总参数: {total_params:,} | 可训练参数: {trainable_params:,}")

    # 6. 设置评估函数、优化器及 AMP 缩放器
    criterion = nn.MSELoss() # 重构任务通常使用 MSE 作为 Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    use_amp = (device.type == "cuda") # 仅在 GPU 上开启 AMP
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 7. 准备 Checkpoint 目录并保存标准化统计参数
    checkpoint_dir = _resolve_checkpoint_dir(args.checkpoints, runtime_config)
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, BEST_MODEL_FILE)
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth") # EarlyStopping 使用的临时存档
    norm_stats_path = os.path.join(checkpoint_dir, NORM_STATS_FILE)
    
    # 初始化早停工具
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    # 保存均值和标准差，供后续推理/测试阶段进行逆向变换或一致性预处理
    np.savez(
        norm_stats_path,
        mean=mean,
        std=std,
        log1p_channels=np.array(runtime_config.log1p_channels, dtype=np.int64),
        channel_names=np.array(runtime_config.channel_names, dtype="<U64"),
        h5_path=np.array([runtime_config.h5_path], dtype="<U512"),
        window_size=np.array([runtime_config.window_size], dtype=np.int64),
        stride=np.array([runtime_config.stride], dtype=np.int64),
        step_seconds=np.array(
            [-1 if runtime_config.step_seconds is None else runtime_config.step_seconds],
            dtype=np.int64,
        ),
    )
    print(f"[持久化] 标准化参数已存至: {norm_stats_path}")

    # 8. 打印训练配置总览
    print("\n" + "=" * 72)
    print(">>> 启动 ConvLSTM AutoEncoder 训练任务")
    print(f"    - 网络参数: Hidden={HIDDEN_DIM}, Latent={LATENT_DIM}, Layers={NUM_LAYERS}")
    print(f"    - 训练超参: Batch={BATCH_SIZE}, LR={args.learning_rate}, Epochs={args.train_epochs}")
    print(
        f"    - 数据配置: H5={runtime_config.h5_path}, Dataset={runtime_config.dataset_name}, "
        f"Step={runtime_config.step_desc}, Window={runtime_config.window_size}, "
        f"Stride={runtime_config.stride}, Channels={runtime_config.in_channels}"
    )
    print(f"    - 策略配置: LR_Adj={args.lradj}, EarlyStop_Patience={args.patience}")
    print(f"    - 系统状态: AMP={use_amp}, Device={device}")
    print("=" * 72)

    # 9. 主训练循环 (Training Loop)
    for epoch in range(1, args.train_epochs + 1):
        t_start = time.time()

        # 执行一个 Epoch 的训练
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, use_amp
        )
        # 执行验证
        val_loss = validate(model, val_loader, criterion, device, use_amp)

        t_end = time.time()
        lr_current = optimizer.param_groups[0]["lr"]
        
        # 实时日志输出
        print(
            f"Epoch {epoch:3d}/{args.train_epochs} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"LR: {lr_current:.2e} | 耗时: {t_end - t_start:.1f}s"
        )

        # 检查是否满足早停条件并保存进度
        early_stopping(val_loss, model, checkpoint_dir)
        
        if early_stopping.early_stop:
            print(f"[早停] 在第 {epoch} 轮触发，模型性能已不再显著提升。")
            break

        # 调整下一轮的学习率
        adjust_learning_rate(optimizer, epoch, args)

    # 10. 训练结束，将最优检查点固化为最终模型文件
    if os.path.exists(checkpoint_path):
        # 加载 EarlyStopping 保存的最优状态字典
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        # 另存为正式的 best_model 文件
        torch.save(model.state_dict(), best_model_path)
        best_val_loss = early_stopping.val_loss_min
    else:
        best_val_loss = float("inf")

    print("\n" + "=" * 72)
    print(f"训练结束！最优验证 Loss (MSE): {best_val_loss:.6f}")
    print(f"权重路径: {best_model_path}")
    print(f"统计文件: {norm_stats_path}")
    print("=" * 72)


def run_test(args: argparse.Namespace) -> None:
    """
    独立测试环节：加载已有权重，在验证集上全面评估重构质量。
    """
    # 1. 设备设置
    if USE_GPU and torch.cuda.is_available():
        device = torch.device(f"cuda:{GPU_ID}")
        print(f"[设备] 测试模式使用 GPU: {device}")
    else:
        device = torch.device("cpu")
        print("[设备] 测试模式使用 CPU")

    # 2. 数据准备 (只需验证部分数据)
    runtime_config = inspect_h5_runtime_config(
        args.h5_path,
        window_size=args.window_size,
        stride=args.stride,
    )
    _, val_data, mean, std, height, width = load_and_preprocess(
        runtime_config, TRAIN_RATIO
    )

    val_dataset = MeteoDataset(
        val_data,
        window_size=runtime_config.window_size,
        stride=runtime_config.stride,
    )
    if len(val_dataset) <= 0:
        raise ValueError("验证集样本数为 0，请检查数据分流或路径。")

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    # 3. 构建模型并加载预训练权重
    model = ConvLSTMAutoEncoder(
        in_channels=runtime_config.in_channels,
        hidden_channels=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_layers=NUM_LAYERS,
        seq_len=runtime_config.window_size,
        frame_height=height,
        frame_width=width,
    ).float().to(device)

    # 确定权重路径：优先使用命令行指定的路径，否则使用默认路径
    checkpoint_dir = _resolve_checkpoint_dir(args.checkpoints, runtime_config)
    checkpoint_path = args.test_checkpoint or os.path.join(checkpoint_dir, BEST_MODEL_FILE)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到需要测试的模型权重文件: {checkpoint_path}")

    print(f"[测试] 正在加载权重: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 4. 执行评估
    criterion = nn.MSELoss()
    use_amp = (device.type == "cuda")
    
    # 获取原始 MSE Loss
    val_loss = validate(model, val_loader, criterion, device, use_amp)
    # 获取详细的重构物理指标
    metrics = evaluate_reconstruction_metrics(model, val_loader, device, use_amp)

    # 5. 打印测试报告
    print("\n" + "=" * 72)
    print("ConvLSTM AutoEncoder 气象数据重构性能测试报告")
    print(f"  测试权重: {checkpoint_path}")
    print(
        f"  数据配置: Step={runtime_config.step_desc}, Window={runtime_config.window_size}, "
        f"Stride={runtime_config.stride}, Channels={runtime_config.in_channels}"
    )
    print(f"  样本数量: {len(val_dataset)}")
    print(f"  标准化态: Mean={tuple(mean.shape)}, Std={tuple(std.shape)}")
    print("-" * 72)
    print(f"  1. MSE Loss (Criterion): {val_loss:.8f}")
    print(f"  2. MAE (物理绝对值误差): {metrics['mae']:.8f}")
    print(f"  3. RMSE (均方根误差):   {metrics['rmse']:.8f}")
    print(f"  4. R2 Score (决定系数):  {metrics['r2']:.8f}")
    print("=" * 72)


# =============================================================================
# Smoke test
# =============================================================================


def run_smoke_test(args: argparse.Namespace) -> None:
    """
    冒烟测试 (Smoke Test)：
    用于在正式训练前，使用随机生成的伪造数据验证整个模型的前向传播逻辑和张量形状变换是否正确。
    可以快速排查出 OOM (显存溢出) 或者是 Shape 不匹配等低级代码错误。
    """
    print("=" * 72)
    print("Smoke Test: 验证 ConvLSTM AutoEncoder 模型的张量维度变动")
    print("=" * 72)

    try:
        runtime_config = inspect_h5_runtime_config(
            args.h5_path,
            window_size=args.window_size,
            stride=args.stride,
        )
        height, width = runtime_config.frame_height, runtime_config.frame_width
        in_channels = runtime_config.in_channels
        window_size = runtime_config.window_size
        print(
            f"[数据] 基于 H5 元数据构造冒烟测试: "
            f"Step={runtime_config.step_desc}, Window={window_size}, Channels={in_channels}"
        )
    except Exception as exc:
        runtime_config = None
        height, width = 62, 61
        in_channels = IN_CHANNELS
        window_size = WINDOW_SIZE
        print(f"[数据] 读取 H5 元数据失败，回退到默认形状执行冒烟测试: {exc}")

    batch_size = 2 # 仅测试用，不需要太大尺寸

    # 自动探测环境：有 GPU 且启动 GPU 配置则放入 cuda:0，否则用 CPU
    device = torch.device("cuda:0" if (USE_GPU and torch.cuda.is_available()) else "cpu")
    print(f"[设备] 运行设备定为: {device}")

    # 1. 初始化模型并上移设备
    model = ConvLSTMAutoEncoder(
        in_channels=in_channels,
        hidden_channels=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_layers=NUM_LAYERS,
        seq_len=window_size,
        frame_height=height,
        frame_width=width,
    ).float().to(device)

    # 2. 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[模型] 总参数量规模: {total_params:,}")

    # 3. 构造随机假输入 [Batch, 序列长, 气象通道, 网格高, 网格宽]
    x = torch.randn(batch_size, window_size, in_channels, height, width, device=device)
    print(f"[输入] x.shape 必须为: {tuple(x.shape)}")

    # 4. 执行不追踪梯度的纯粹前向推理
    with torch.inference_mode():
        # 测试：全流程 (Encode -> Decode)
        recon = model(x)
        print(f"[重建] recon.shape 为: {tuple(recon.shape)}")
        # 强制断言输出形状应与输入完全保持一致，不一致直接抛出异常中断
        assert recon.shape == x.shape, f"重建形状不匹配，期待 {x.shape}，实际得到 {recon.shape}"

        # 测试：纯编码功能 (用于将来提取特征向量供相似日匹配库比对)
        latent = model.encode(x)
        print(f"[潜向量] latent.shape 必须被挤压至: {tuple(latent.shape)}")
        assert latent.shape == (batch_size, LATENT_DIM), f"纯编码形状出错: 期待 {(batch_size, LATENT_DIM)} 得到 {latent.shape}"

    print("\n[OK] Smoke Test passed. 恭喜，所有形状断言均通过验证！")
    print("=" * 72)


if __name__ == "__main__":
    # 配置命令行解析器
    parser = argparse.ArgumentParser(description="ConvLSTM AutoEncoder - 气象时序时空特征重构与编码系统")
    
    # 模式选择
    parser.add_argument("--smoke-test", action="store_true", help="冒烟测试：使用虚拟数据验证模型结构和显存占用")
    parser.add_argument("--test-only", action="store_true", help="测试模式：加载最优权重并在验证集上跑分")
    
    # 训练关键参数
    parser.add_argument("--train-epochs", type=int, default=EPOCHS, help="最大训练轮数 (Epochs)")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="学习率 (Learning Rate)")
    parser.add_argument("--patience", type=int, default=5, help="早停耐心值 (Patience)，超过此轮数 Loss 不降则停止")
    parser.add_argument(
        "--h5-path",
        type=str,
        default=H5_PATH,
        help=(
            "训练/测试所使用的气象 HDF5 路径。默认使用 hourly 版；"
            f"旧版 10 通道 15min 可指定为 {LEGACY_H5_PATH}"
        ),
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=0,
        help="滑动窗口长度。默认按 H5 时间分辨率自动推断：hourly=24, 15min=96。",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=0,
        help="滑动窗口步长。默认按 H5 时间分辨率自动推断：hourly=6, 15min=24。",
    )
    parser.add_argument(
        "--lradj",
        type=str,
        default="cosine",
        choices=["type1", "type2", "cosine", "none"],
        help="学习率衰减策略",
    )
    
    # 路径配置
    parser.add_argument("--checkpoints", type=str, default=CHECKPOINT_DIR, help="模型权重和统计量的保存目录")
    parser.add_argument(
        "--test-checkpoint",
        type=str,
        default="",
        help="测试模式下的特定权重文件路径 (不填则默认加载保存的最优模型)",
    )
    
    cli_args = parser.parse_args()

    # 根据参数进入不同分支
    if cli_args.smoke_test:
        # 进入模型验证冒烟测试
        run_smoke_test(cli_args)
    elif cli_args.test_only:
        # 进入纯推理测试评估模式
        run_test(cli_args)
    else:
        # 进入常规训练/验证流程
        run_training(cli_args)
