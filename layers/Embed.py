"""
Embed.py - 嵌入层模块

本模块实现了 Transformer 时序模型所需的各种嵌入层，主要包括：
1. PositionalEmbedding: 正弦余弦位置编码
2. TokenEmbedding: 值嵌入（1D 卷积）
3. FixedEmbedding: 固定位置编码查找表
4. TemporalEmbedding: 时间特征嵌入（离散）
5. TimeFeatureEmbedding: 时间特征嵌入（连续）
6. DataEmbedding: 完整的数据嵌入（值 + 位置 + 时间）
7. DataEmbedding_inverted: 反转嵌入（用于 iTransformer）
8. DataEmbedding_wo_pos: 无位置编码的嵌入
9. PatchEmbedding: Patch 嵌入（用于 PatchTST）

嵌入层的作用：
    将原始输入（时间序列值、时间戳）转换为高维向量表示，
    使 Transformer 能够处理和理解输入数据。

位置编码公式（Sinusoidal Positional Encoding）：
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    其中：
    - pos: 位置索引 (0, 1, 2, ...)
    - i: 维度索引
    - d_model: 模型维度

嵌入层叠加示意图：
    ┌─────────────────────────────────────────────────────────┐
    │  输入 x [B, L, C]     输入 x_mark [B, L, T]             │
    │       ↓                      ↓                          │
    │  ┌─────────────┐       ┌─────────────┐                  │
    │  │TokenEmbedding│       │TemporalEmbed│                  │
    │  └─────────────┘       └─────────────┘                  │
    │       ↓                      ↓                          │
    │       └──────────────────────┘                          │
    │                    ↓                                    │
    │              ┌─────────────┐                            │
    │              │PositionalEmbed│                          │
    │              └─────────────┘                            │
    │                    ↓                                    │
    │    output = value_emb + temporal_emb + position_emb    │
    │                    ↓                                    │
    │              ┌─────────────┐                            │
    │              │   Dropout   │                            │
    │              └─────────────┘                            │
    │                    ↓                                    │
    │             输出 [B, L, d_model]                        │
    └─────────────────────────────────────────────────────────┘
"""

import torch                           # PyTorch 深度学习框架
import torch.nn as nn                  # 神经网络模块
import torch.nn.functional as F        # 函数式接口
from torch.nn.utils import weight_norm # 权重归一化
import math                            # 数学函数


class PositionalEmbedding(nn.Module):
    """
    正弦余弦位置编码（Sinusoidal Positional Encoding）
    
    为序列中的每个位置生成唯一的编码向量，使模型能够感知位置信息。
    这是 "Attention is All You Need" 论文中提出的经典位置编码方法。
    
    特点：
    - 固定编码，不需要学习
    - 可以处理任意长度的序列（受 max_len 限制）
    - 周期性变化，不同位置有不同的编码模式
    
    公式：
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model, max_len=5000):
        """
        初始化位置编码
        
        参数:
            d_model (int): 模型维度（编码向量的维度）
            max_len (int): 最大序列长度
        """
        super(PositionalEmbedding, self).__init__()
        
        # 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False  # 位置编码不需要梯度

        # 位置索引：[0, 1, 2, ..., max_len-1]，形状 [max_len, 1]
        position = torch.arange(0, max_len).float().unsqueeze(1)
        
        # 计算分母项：10000^(2i/d_model)，使用对数形式计算更稳定
        # div_term = exp(2i * (-log(10000) / d_model))
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        # 偶数维度使用 sin，奇数维度使用 cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 2i
        pe[:, 1::2] = torch.cos(position * div_term)  # 2i+1

        # 添加 batch 维度：[1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # 注册为 buffer（不作为参数，但会保存到 state_dict）
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x (Tensor): 输入张量 [B, L, ...]，仅用于获取序列长度
        
        返回:
            Tensor: 位置编码 [1, L, d_model]（会自动广播到 batch 维度）
        """
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """
    Token/值嵌入层
    
    使用 1D 卷积将原始时间序列值映射到高维空间。
    卷积核大小为 3，可以捕获局部时序模式。
    
    维度变化：
        输入: [B, L, c_in]
        输出: [B, L, d_model]
    """
    
    def __init__(self, c_in, d_model):
        """
        初始化 Token 嵌入
        
        参数:
            c_in (int): 输入特征维度（变量数量）
            d_model (int): 模型维度（输出维度）
        """
        super(TokenEmbedding, self).__init__()
        
        # PyTorch 版本兼容性处理
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        
        # 1D 卷积：kernel_size=3 捕获局部模式
        # circular padding：循环填充，处理序列边界
        self.tokenConv = nn.Conv1d(
            in_channels=c_in, 
            out_channels=d_model,
            kernel_size=3, 
            padding=padding, 
            padding_mode='circular', 
            bias=False
        )
        
        # Kaiming 初始化：适用于 ReLU/LeakyReLU 激活
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        """
        前向传播
        
        参数:
            x (Tensor): 输入张量 [B, L, c_in]
        
        返回:
            Tensor: Token 嵌入 [B, L, d_model]
        """
        # [B, L, C] -> [B, C, L] -> Conv1d -> [B, d_model, L] -> [B, L, d_model]
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    """
    固定位置编码嵌入（用于时间特征）
    
    与 PositionalEmbedding 类似，但使用 nn.Embedding 查找表实现。
    用于将离散的时间索引（如月份、日期）映射到固定的编码向量。
    
    特点：
    - 参数不可训练（固定）
    - 使用正弦余弦编码
    """
    
    def __init__(self, c_in, d_model):
        """
        初始化固定嵌入
        
        参数:
            c_in (int): 输入类别数（如月份数=13，小时数=24）
            d_model (int): 模型维度
        """
        super(FixedEmbedding, self).__init__()

        # 创建固定权重矩阵
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        # 使用正弦余弦编码
        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        # 创建 Embedding 层并设置为固定权重
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x (Tensor): 时间索引张量 [B, L] 或 [B, L, 1]
        
        返回:
            Tensor: 时间嵌入 [B, L, d_model]
        """
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """
    时间特征嵌入（离散方式）
    
    将时间戳的各个组成部分（月、日、周几、时、分）
    分别嵌入后求和，得到综合的时间表示。
    
    时间特征组成：
        - 月份 (month): 1-12
        - 日期 (day): 1-31
        - 周几 (weekday): 0-6
        - 小时 (hour): 0-23
        - 分钟 (minute): 0-59（仅当 freq='t' 时）
    
    输入格式：
        x_mark: [B, L, 4] 或 [B, L, 5]
        包含 [month, day, weekday, hour, (minute)]
    """
    
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        """
        初始化时间特征嵌入
        
        参数:
            d_model (int): 模型维度
            embed_type (str): 嵌入类型
                             'fixed' - 固定正弦余弦编码
                             其他 - 可学习的嵌入
            freq (str): 时间频率
                       'h' - 小时级
                       't' - 分钟级（15分钟）
        """
        super(TemporalEmbedding, self).__init__()

        # 各时间组件的类别数（+1 用于索引对齐）
        minute_size = 4      # 分钟（15分钟间隔：0, 15, 30, 45）
        hour_size = 24       # 小时：0-23
        weekday_size = 7     # 周几：0-6
        day_size = 32        # 日期：1-31（+1 为了索引从 1 开始）
        month_size = 13      # 月份：1-12（+1 为了索引从 1 开始）

        # 选择嵌入类型
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        
        # 分钟嵌入（仅分钟级数据）
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        
        # 其他时间组件嵌入
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x (Tensor): 时间标记张量 [B, L, 4/5]
                       格式：[month, day, weekday, hour, (minute)]
        
        返回:
            Tensor: 时间嵌入 [B, L, d_model]
        """
        x = x.long()  # 转为整数索引
        
        # 提取各时间组件并嵌入
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.  # 分钟（如果有）
        hour_x = self.hour_embed(x[:, :, 3])      # 小时
        weekday_x = self.weekday_embed(x[:, :, 2]) # 周几
        day_x = self.day_embed(x[:, :, 1])         # 日期
        month_x = self.month_embed(x[:, :, 0])     # 月份

        # 各组件求和得到综合时间嵌入
        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    """
    时间特征嵌入（连续方式）
    
    使用线性层将连续的时间特征（如归一化后的时间值）
    映射到模型维度。
    
    与 TemporalEmbedding 的区别：
    - TemporalEmbedding：离散索引 → 查找表
    - TimeFeatureEmbedding：连续特征 → 线性变换
    
    输入特征维度（根据 freq）：
        'h' (小时): 4 维 [hour_sin, hour_cos, day_sin, day_cos, ...]
        't' (分钟): 5 维
        'd' (天): 3 维
        'w' (周): 2 维
        'm' (月): 1 维
    """
    
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        """
        初始化时间特征嵌入
        
        参数:
            d_model (int): 模型维度
            embed_type (str): 嵌入类型（仅用于接口一致性）
            freq (str): 时间频率，决定输入特征维度
        """
        super(TimeFeatureEmbedding, self).__init__()

        # 不同频率对应的特征维度
        freq_map = {
            'h': 4,  # 小时级：hour_of_day, day_of_week, day_of_month, month_of_year
            't': 5,  # 分钟级：增加 minute_of_hour
            's': 6,  # 秒级
            'm': 1,  # 月级
            'a': 1,  # 年级
            'w': 2,  # 周级
            'd': 3,  # 天级
            'b': 3   # 工作日级
        }
        d_inp = freq_map[freq]
        
        # 线性映射：d_inp → d_model
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x (Tensor): 时间特征张量 [B, L, d_inp]
        
        返回:
            Tensor: 时间嵌入 [B, L, d_model]
        """
        return self.embed(x)


class DataEmbedding(nn.Module):
    """
    完整的数据嵌入层
    
    将输入数据转换为 Transformer 可处理的表示，包含三个组成部分：
    1. 值嵌入（TokenEmbedding）：编码时间序列值
    2. 位置嵌入（PositionalEmbedding）：编码序列位置
    3. 时间嵌入（Temporal/TimeFeatureEmbedding）：编码时间特征
    
    最终输出 = 值嵌入 + 位置嵌入 + 时间嵌入
    """
    
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        """
        初始化数据嵌入层
        
        参数:
            c_in (int): 输入特征维度
            d_model (int): 模型维度
            embed_type (str): 时间嵌入类型
                             'fixed' - 固定正弦余弦编码
                             'learned' - 可学习嵌入
                             'timeF' - 连续时间特征
            freq (str): 时间频率
            dropout (float): Dropout 比率
        """
        super(DataEmbedding, self).__init__()

        # 值嵌入：时间序列值 → d_model
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        
        # 位置嵌入：序列位置 → d_model
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        
        # 时间嵌入：根据 embed_type 选择不同的时间编码方式
        self.temporal_embedding = TemporalEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq
        ) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq
        )
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        前向传播
        
        参数:
            x (Tensor): 输入值 [B, L, c_in]
            x_mark (Tensor or None): 时间标记 [B, L, T]
        
        返回:
            Tensor: 数据嵌入 [B, L, d_model]
        """
        if x_mark is None:
            # 没有时间标记：只使用值嵌入和位置嵌入
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            # 有时间标记：三者相加
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    """
    反转数据嵌入层（用于 iTransformer）
    
    与标准 DataEmbedding 不同，这里将时间维度和变量维度互换：
    - 标准：每个时间步的所有变量作为一个 token
    - 反转：每个变量的所有时间步作为一个 token
    
    维度变化：
        输入: [B, L, C] (L=序列长度, C=变量数)
        转置: [B, C, L] (每个变量作为一个 token)
        嵌入: [B, C, d_model]
    """
    
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        """
        初始化反转嵌入层
        
        参数:
            c_in (int): 序列长度（作为嵌入输入维度）
            d_model (int): 模型维度
            embed_type, freq: 保留参数（这里未使用）
            dropout (float): Dropout 比率
        """
        super(DataEmbedding_inverted, self).__init__()
        
        # 值嵌入：序列长度 → d_model
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        前向传播
        
        参数:
            x (Tensor): 输入值 [B, L, C]
            x_mark (Tensor or None): 时间标记 [B, L, T]
        
        返回:
            Tensor: 反转嵌入 [B, C, d_model] 或 [B, C+T, d_model]
        """
        # 转置：[B, L, C] → [B, C, L]
        x = x.permute(0, 2, 1)
        
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # 将时间特征也作为额外的"变量"
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    """
    无位置嵌入的数据嵌入层
    
    与 DataEmbedding 类似，但在输出中不加入位置嵌入。
    适用于某些不需要显式位置编码的模型。
    
    用途：
    - PatchTST：位置信息通过 Patch 顺序隐式编码
    - 某些预训练场景
    """
    
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        """
        初始化无位置嵌入层
        
        参数:
            c_in (int): 输入特征维度
            d_model (int): 模型维度
            embed_type (str): 时间嵌入类型
            freq (str): 时间频率
            dropout (float): Dropout 比率
        """
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)  # 保留但不使用
        self.temporal_embedding = TemporalEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq
        ) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        前向传播
        
        注意：这里不使用位置嵌入
        
        参数:
            x (Tensor): 输入值 [B, L, c_in]
            x_mark (Tensor or None): 时间标记
        
        返回:
            Tensor: 数据嵌入 [B, L, d_model]
        """
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    """
    Patch 嵌入层（用于 PatchTST）
    
    PatchTST 的核心创新：将时间序列分割成多个 Patch，
    每个 Patch 作为一个 Token 输入 Transformer。
    
    处理流程：
        1. 边界填充：确保序列能被完整分割
        2. 分割 Patch：使用 unfold 操作
        3. 值嵌入：将每个 Patch 映射到 d_model
        4. 位置嵌入：为每个 Patch 添加位置信息
    
    Patch 分割示意图（patch_len=4, stride=2）：
        原始序列: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        Patch 1: [1, 2, 3, 4]     (位置 0-3)
        Patch 2: [3, 4, 5, 6]     (位置 2-5)
        Patch 3: [5, 6, 7, 8]     (位置 4-7)
        Patch 4: [7, 8, 9, 10]    (位置 6-9)
        
        注意：stride < patch_len 时会有重叠
    
    优点：
    - 降低序列长度：L/stride 个 Patch
    - 降低计算复杂度：O(L²) → O((L/stride)²)
    - 更好的局部模式捕获
    """
    
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        """
        初始化 Patch 嵌入层
        
        参数:
            d_model (int): 模型维度
            patch_len (int): 每个 Patch 的长度
            stride (int): Patch 滑动步长（stride < patch_len 时有重叠）
            padding (int): 序列末尾填充长度（确保完整分割）
            dropout (float): Dropout 比率
        """
        super(PatchEmbedding, self).__init__()
        
        self.patch_len = patch_len
        self.stride = stride
        
        # 边界填充层：使用复制填充（ReplicationPad）
        # (0, padding) 表示只在序列末尾填充
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # 值嵌入：将 patch_len 维的 Patch 映射到 d_model
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        
        # 位置嵌入
        self.position_embedding = PositionalEmbedding(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x (Tensor): 输入张量 [B, C, L]
                       B: batch_size
                       C: 变量数（Channel-Independent，每个变量独立处理）
                       L: 序列长度
        
        返回:
            x (Tensor): Patch 嵌入 [B*C, num_patches, d_model]
            n_vars (int): 变量数量
        
        维度变化：
            输入: [B, C, L]
            填充: [B, C, L+padding]
            分割: [B, C, num_patches, patch_len]
            重塑: [B*C, num_patches, patch_len]（Channel-Independent）
            嵌入: [B*C, num_patches, d_model]
        """
        # 获取变量数
        n_vars = x.shape[1]
        
        # 边界填充
        x = self.padding_patch_layer(x)
        
        # 使用 unfold 分割 Patch
        # unfold(dim, size, step): 在指定维度上滑动窗口
        # 结果: [B, C, num_patches, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        # 重塑为 Channel-Independent 形式
        # [B, C, num_patches, patch_len] → [B*C, num_patches, patch_len]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        
        # 值嵌入 + 位置嵌入
        x = self.value_embedding(x) + self.position_embedding(x)
        
        return self.dropout(x), n_vars

