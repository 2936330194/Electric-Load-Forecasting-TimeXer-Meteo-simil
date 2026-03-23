"""
ConvLSTM AutoEncoder for meteorological spatiotemporal sequence encoding.

Usage:
    python convlstm_ae.py
    python convlstm_ae.py --smoke-test
"""

import argparse
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from utils.tools import EarlyStopping, adjust_learning_rate

try:
    import h5py
except ImportError:
    h5py = None


# =============================================================================
# 全局配置 (Global Config) - 超参数、路径、数据字段等的收口定义
# =============================================================================

# 指定气象网格数据集存储路径
H5_PATH = "./data/hunan_grid_meteo_20250101_20260228.h5"

# 气象通道字段定义 (按照通道 index 顺序严格对应)
CHANNEL_NAMES = [
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
LOG1P_CHANNELS = [9]

# ----------------- 模型超参数 (Model Hyperparameters) -----------------
WINDOW_SIZE = 96     # 时间窗大小：96代表1天的数据包含着每15分钟一个步长组成的96个连续帧
HIDDEN_DIM = 16      # ConvLSTM 的隐藏特征通道层数，提取出的特征更抽象，参数也随之成平方倍增加
LATENT_DIM = 64      # 最后编码得到的潜向量长度映射 (决定了后续在检索库搜索时向量的体积大小)
NUM_LAYERS = 1       # 级联的 ConvLSTM 提取深度层数 (1层通常具备基础表征，多层计算较慢)
IN_CHANNELS = 10     # 对应上面的 CHANNEL_NAMES

# ----------------- 训练控制超参数 (Training Hyperparameters) ----------
BATCH_SIZE = 16      # 每批次包含的天数样本，受限于 4D 张量 (B, T, C, H, W) 的极长的时间步长，极易发生显存崩塌
LEARNING_RATE = 1e-3 # Adam优化器的初始学习率
EPOCHS = 50          # 模型全参数迭代的最高轮数
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
        # 自适应平均池化，将任意 HxW 的空间图压缩为 1x1 标量维度
        self.pool = nn.AdaptiveAvgPool2d(1)
        # 将池化后的隐藏层映射为用户设定的潜在维度（Latent Space）
        self.fc = nn.Linear(hidden_channels, latent_dim)

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

        # 映射回隐藏层维度
        self.fc = nn.Linear(latent_dim, hidden_channels)
        
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
        
        # 1. 扩展潜向量成基础表征图 -> [B, hidden_channels]
        feat = self.fc(latent)
        
        # 2. 通过 Expand 的方式广播回时间序列维度与空间网格维度
        # view 先插入 seq_len, H, W 对应的伪维度 1 -> [B, 1, hidden_channels, 1, 1]
        feat = feat.view(batch_size, 1, self.hidden_channels, 1, 1)
        # expand 执行广播，不耗费大量额外显存 -> [B, seq_len, hidden_channels, H, W]
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
    负责将预处理后的全量 numpy 数据 [Total_Time_Steps, 10, H, W] 
    切分成适合送入模型的固定窗口 (window_size) 样本集。

    核心逻辑是平移切割：假如序列长 1000，window_size=96，则切割出 1000//96 个非重叠样本。
    """
    def __init__(self, data: np.ndarray, window_size: int = WINDOW_SIZE):
        super().__init__()
        self.window_size = window_size
        
        # 为了严格保证每个样本时序长度等于 window_size，采用向下取整舍弃多余的首尾部分
        n_samples = len(data) // window_size
        
        # 将连续的一维时序截断后 Reshape 成 [样本数量, 时间窗口大小, 通道数, H, W] 的格式
        self.data = data[: n_samples * window_size].reshape(
            n_samples,
            window_size,
            data.shape[1],
            data.shape[2],
            data.shape[3],
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        # PyTorch 的 Dataset 要求返回的是 Tensor 或者字典，这里将 NumPy 数据转化为 Tensor
        return torch.from_numpy(self.data[index])


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


def load_and_preprocess(
    h5_path: str,
    log1p_channels: List[int],
    train_ratio: float = TRAIN_RATIO,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    从 HDF5 中加载高维气象场数据，并执行预处理，包含两步：
    1. log1p 对指定长尾分布特征进行缩放。
    2. 基于分离出的训练集，在各个通道 (Channel) 维度计算并进行独立的 Z-Score 标准化。
    """
    if h5py is None:
        raise ImportError("需要先安装 h5py: pip install h5py")

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
    for ch_idx in log1p_channels:
        print(f"[预处理] channel {ch_idx} ({CHANNEL_NAMES[ch_idx]}): 取 log1p(x+1) 进行异常值平滑")
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


def run_training(args: argparse.Namespace) -> None:
    if args.train_epochs <= 0:
        raise ValueError("train_epochs must be positive.")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if args.patience <= 0:
        raise ValueError("patience must be positive.")

    if USE_GPU and torch.cuda.is_available():
        device = torch.device(f"cuda:{GPU_ID}")
        print(f"[设备] 使用 GPU: cuda:{GPU_ID}")
    else:
        device = torch.device("cpu")
        print("[设备] 使用 CPU")

    h5_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), H5_PATH)
    train_data, val_data, mean, std, height, width = load_and_preprocess(
        h5_path, LOG1P_CHANNELS, TRAIN_RATIO
    )

    train_dataset = MeteoDataset(train_data, WINDOW_SIZE)
    val_dataset = MeteoDataset(val_data, WINDOW_SIZE)
    print(f"[数据集] 训练样本: {len(train_dataset)} | 验证样本: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model = ConvLSTMAutoEncoder(
        in_channels=IN_CHANNELS,
        hidden_channels=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_layers=NUM_LAYERS,
        seq_len=WINDOW_SIZE,
        frame_height=height,
        frame_width=width,
    ).float().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[模型] 总参数: {total_params:,} | 可训练: {trainable_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(args.checkpoints, exist_ok=True)
    best_val_loss = float("inf")
    best_model_path = os.path.join(args.checkpoints, BEST_MODEL_FILE)
    checkpoint_path = os.path.join(args.checkpoints, "checkpoint.pth")
    norm_stats_path = os.path.join(args.checkpoints, NORM_STATS_FILE)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    np.savez(
        norm_stats_path,
        mean=mean,
        std=std,
        log1p_channels=np.array(LOG1P_CHANNELS),
    )
    print(f"[保存] 标准化统计量: {norm_stats_path}")

    print("\n" + "=" * 72)
    print("开始训练 ConvLSTM AutoEncoder")
    print(f"  hidden_dim={HIDDEN_DIM}, latent_dim={LATENT_DIM}, num_layers={NUM_LAYERS}")
    print(f"  batch_size={BATCH_SIZE}, lr={args.learning_rate}, epochs={args.train_epochs}")
    print(f"  lradj={args.lradj}, patience={args.patience}")
    print(f"  use_amp={use_amp}")
    print("=" * 72)

    for epoch in range(1, args.train_epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, use_amp
        )
        val_loss = validate(model, val_loader, criterion, device, use_amp)

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{args.train_epochs} | "
            f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
            f"LR: {lr_now:.2e} | Time: {elapsed:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  Best model saved (val_loss={val_loss:.6f})")

        early_stopping(val_loss, model, args.checkpoints)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        adjust_learning_rate(optimizer, epoch, args)

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        torch.save(model.state_dict(), best_model_path)
        best_val_loss = min(best_val_loss, early_stopping.val_loss_min)

    print("\n" + "=" * 72)
    print(f"训练完成！最优验证 Loss: {best_val_loss:.6f}")
    print(f"模型权重: {best_model_path}")
    print(f"标准化统计量: {norm_stats_path}")
    print("=" * 72)


# =============================================================================
# Smoke test
# =============================================================================


def run_smoke_test() -> None:
    """
    冒烟测试 (Smoke Test)：
    用于在正式训练前，使用随机生成的伪造数据验证整个模型的前向传播逻辑和张量形状变换是否正确。
    可以快速排查出 OOM (显存溢出) 或者是 Shape 不匹配等低级代码错误。
    """
    print("=" * 72)
    print("Smoke Test: 验证 ConvLSTM AutoEncoder 模型的张量维度变动")
    print("=" * 72)

    height, width = 62, 61
    batch_size = 2 # 仅测试用，不需要太大尺寸

    # 自动探测环境：有 GPU 且启动 GPU 配置则放入 cuda:0，否则用 CPU
    device = torch.device("cuda:0" if (USE_GPU and torch.cuda.is_available()) else "cpu")
    print(f"[设备] 运行设备定为: {device}")

    # 1. 初始化模型并上移设备
    model = ConvLSTMAutoEncoder(
        in_channels=IN_CHANNELS,
        hidden_channels=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_layers=NUM_LAYERS,
        seq_len=WINDOW_SIZE,
        frame_height=height,
        frame_width=width,
    ).float().to(device)

    # 2. 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[模型] 总参数量规模: {total_params:,}")

    # 3. 构造随机假输入 [Batch, 序列长, 气象通道, 网格高, 网格宽]
    x = torch.randn(batch_size, WINDOW_SIZE, IN_CHANNELS, height, width, device=device)
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
    parser = argparse.ArgumentParser(description="ConvLSTM AutoEncoder 气象时空特征编码器")
    parser.add_argument("--smoke-test", action="store_true", help="使用随机数据验证模型结构")
    parser.add_argument("--train-epochs", type=int, default=EPOCHS, help="训练轮数")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="初始学习率")
    parser.add_argument("--patience", type=int, default=7, help="早停耐心轮数")
    parser.add_argument(
        "--lradj",
        type=str,
        default="cosine",
        choices=["type1", "type2", "cosine", "none"],
        help="学习率调整策略",
    )
    parser.add_argument("--checkpoints", type=str, default=CHECKPOINT_DIR, help="checkpoint 保存目录")
    cli_args = parser.parse_args()

    if cli_args.smoke_test:
        run_smoke_test()
    else:
        run_training(cli_args)
