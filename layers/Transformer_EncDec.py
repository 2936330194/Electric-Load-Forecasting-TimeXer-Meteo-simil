"""
Transformer_EncDec.py - Transformer 编码器-解码器架构模块

本模块实现了 Transformer 模型的核心组件：
1. ConvLayer: 卷积下采样层（用于 Informer 等模型的蒸馏）
2. EncoderLayer: 编码器层（自注意力 + 前馈网络）
3. Encoder: 编码器（堆叠多个编码器层）
4. DecoderLayer: 解码器层（自注意力 + 交叉注意力 + 前馈网络）
5. Decoder: 解码器（堆叠多个解码器层）

Transformer 架构概览：
                                    
    ┌─────────────────────────────────────────────────────────────┐
    │                        ENCODER                              │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │  EncoderLayer 1                                       │  │
    │  │  ┌─────────────────┐    ┌─────────────────┐           │  │
    │  │  │ Self-Attention  │───→│ Feed Forward    │           │  │
    │  │  └─────────────────┘    └─────────────────┘           │  │
    │  └───────────────────────────────────────────────────────┘  │
    │                          ↓                                  │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │  EncoderLayer 2  ...                                  │  │
    │  └───────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────┘
                               ↓ (encoder output)
    ┌─────────────────────────────────────────────────────────────┐
    │                        DECODER                              │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │  DecoderLayer 1                                       │  │
    │  │  ┌─────────────┐  ┌───────────────┐  ┌──────────────┐ │  │
    │  │  │Self-Attn    │→ │Cross-Attn     │→ │Feed Forward  │ │  │
    │  │  │(masked)     │  │(with encoder) │  │              │ │  │
    │  │  └─────────────┘  └───────────────┘  └──────────────┘ │  │
    │  └───────────────────────────────────────────────────────┘  │
    │                          ↓                                  │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │  DecoderLayer 2  ...                                  │  │
    │  └───────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────┘
                               ↓
                           Output

残差连接和层归一化：
    每个子层（注意力、前馈网络）都使用残差连接和层归一化：
    output = LayerNorm(x + Sublayer(x))
"""

import torch                    # PyTorch 深度学习框架
import torch.nn as nn           # 神经网络模块
import torch.nn.functional as F # 函数式接口（激活函数等）


class ConvLayer(nn.Module):
    """
    卷积下采样层
    
    用于 Informer 模型中的自注意力蒸馏（Self-attention Distilling）。
    通过卷积和池化操作减少序列长度，降低计算复杂度。
    
    处理流程：
        1. 1D 卷积（保持通道数）
        2. 批归一化
        3. ELU 激活
        4. 最大池化（序列长度减半）
    
    输入输出维度变化：
        输入: [B, L, d_model]
        输出: [B, L/2, d_model]（序列长度减半）
    """
    
    def __init__(self, c_in):
        """
        初始化卷积下采样层
        
        参数:
            c_in (int): 输入通道数（等于 d_model）
        """
        super(ConvLayer, self).__init__()
        
        # 1D 卷积层
        # kernel_size=3: 卷积核大小为 3
        # padding=2, padding_mode='circular': 循环填充，处理序列边界
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=3,
            padding=2,
            padding_mode='circular'
        )
        
        # 批归一化：稳定训练
        self.norm = nn.BatchNorm1d(c_in)
        
        # ELU 激活函数：比 ReLU 更平滑，允许负值
        self.activation = nn.ELU()
        
        # 最大池化：stride=2 使序列长度减半
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x (Tensor): 输入张量 [B, L, d_model]
        
        返回:
            Tensor: 下采样后的张量 [B, L/2, d_model]
        """
        # 转换维度：[B, L, C] -> [B, C, L]（Conv1d 需要通道在前）
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        # 转回：[B, C, L/2] -> [B, L/2, C]
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    """
    Transformer 编码器层
    
    每个编码器层包含两个子层：
    1. 多头自注意力层
    2. 前馈神经网络（两层 1D 卷积实现）
    
    结构：
        x → Self-Attention → Add & Norm → FFN → Add & Norm → output
        └──────────────────────┘       └────────────────┘
              残差连接                    残差连接
    
    前馈网络（FFN）结构：
        输入 [B, L, d_model]
          ↓ Conv1d (d_model → d_ff)
          ↓ 激活函数 (ReLU/GELU)
          ↓ Dropout
          ↓ Conv1d (d_ff → d_model)
          ↓ Dropout
        输出 [B, L, d_model]
    """
    
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        """
        初始化编码器层
        
        参数:
            attention (nn.Module): 多头注意力模块（AttentionLayer 实例）
            d_model (int): 模型维度
            d_ff (int): 前馈网络隐藏层维度，默认为 4 * d_model
            dropout (float): Dropout 比率
            activation (str): 激活函数，"relu" 或 "gelu"
        """
        super(EncoderLayer, self).__init__()
        
        # FFN 隐藏层维度，默认是模型维度的 4 倍
        d_ff = d_ff or 4 * d_model
        
        # 自注意力模块
        self.attention = attention
        
        # 前馈网络：使用两个 1D 卷积（kernel_size=1 等价于全连接）
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)  # 注意力后的归一化
        self.norm2 = nn.LayerNorm(d_model)  # FFN 后的归一化
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 激活函数
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        """
        前向传播
        
        参数:
            x (Tensor): 输入张量 [B, L, d_model]
            attn_mask: 注意力掩码
            tau, delta: 预留参数（用于 De-stationary Attention）
        
        返回:
            output (Tensor): 编码器层输出 [B, L, d_model]
            attn (Tensor): 注意力权重
        """
        # ==================== 自注意力子层 ====================
        # Q=K=V=x（自注意力）
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        # 残差连接 + Dropout
        x = x + self.dropout(new_x)

        # 第一次层归一化（Post-LN 架构）
        y = x = self.norm1(x)
        
        # ==================== 前馈网络子层 ====================
        # Conv1d 需要 [B, C, L] 格式，所以需要转置
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        # 残差连接 + 第二次层归一化
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    """
    Transformer 编码器
    
    由多个编码器层堆叠而成，可选地包含卷积下采样层（用于 Informer）。
    
    结构（无卷积层）：
        x → EncoderLayer 1 → EncoderLayer 2 → ... → EncoderLayer N → LayerNorm → output
    
    结构（带卷积层，Informer）：
        x → EncoderLayer 1 → ConvLayer → EncoderLayer 2 → ConvLayer → ... → EncoderLayer N → output
        
        卷积层用于自注意力蒸馏，逐层减少序列长度。
    """
    
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        """
        初始化编码器
        
        参数:
            attn_layers (list): 编码器层列表
            conv_layers (list): 卷积下采样层列表（可选，用于 Informer）
            norm_layer (nn.Module): 最终的归一化层（可选）
        """
        super(Encoder, self).__init__()
        
        # 将层列表转换为 ModuleList，确保正确注册参数
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        """
        前向传播
        
        参数:
            x (Tensor): 输入张量 [B, L, d_model]
            attn_mask: 注意力掩码
            tau, delta: 预留参数
        
        返回:
            x (Tensor): 编码器输出 [B, L', d_model]
                       L' = L（无卷积层）或 L / 2^n（有卷积层）
            attns (list): 各层的注意力权重列表
        """
        attns = []  # 存储各层的注意力权重
        
        if self.conv_layers is not None:
            # ==================== 带卷积层的编码（Informer） ====================
            # 注意：attn_layers 比 conv_layers 多一层
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                # delta 只在第一层使用
                delta = delta if i == 0 else None
                # 注意力层
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                # 卷积下采样
                x = conv_layer(x)
                attns.append(attn)
            
            # 最后一个注意力层（没有对应的卷积层）
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            # ==================== 标准编码（无卷积下采样） ====================
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        # 最终归一化（可选）
        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Transformer 解码器层
    
    每个解码器层包含三个子层：
    1. 掩码自注意力层（防止看到未来信息）
    2. 交叉注意力层（关注编码器输出）
    3. 前馈神经网络
    
    结构：
        x → Masked Self-Attn → Add & Norm 
          → Cross-Attn (with encoder) → Add & Norm
          → FFN → Add & Norm → output
    
    与编码器层的区别：
    - 使用掩码自注意力（因果掩码）
    - 额外的交叉注意力层，Query 来自解码器，Key/Value 来自编码器
    """
    
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        """
        初始化解码器层
        
        参数:
            self_attention (nn.Module): 自注意力模块（带因果掩码）
            cross_attention (nn.Module): 交叉注意力模块
            d_model (int): 模型维度
            d_ff (int): 前馈网络隐藏层维度
            dropout (float): Dropout 比率
            activation (str): 激活函数
        """
        super(DecoderLayer, self).__init__()
        
        d_ff = d_ff or 4 * d_model
        
        # 两个注意力模块
        self.self_attention = self_attention    # 自注意力（掩码）
        self.cross_attention = cross_attention  # 交叉注意力
        
        # 前馈网络
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        
        # 三个层归一化（对应三个子层）
        self.norm1 = nn.LayerNorm(d_model)  # 自注意力后
        self.norm2 = nn.LayerNorm(d_model)  # 交叉注意力后
        self.norm3 = nn.LayerNorm(d_model)  # FFN 后
        
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        """
        前向传播
        
        参数:
            x (Tensor): 解码器输入 [B, L_dec, d_model]
            cross (Tensor): 编码器输出 [B, L_enc, d_model]
            x_mask: 自注意力掩码（因果掩码，防止看到未来）
            cross_mask: 交叉注意力掩码
            tau, delta: 预留参数
        
        返回:
            output (Tensor): 解码器层输出 [B, L_dec, d_model]
        """
        # ==================== 掩码自注意力子层 ====================
        # Q=K=V=x（自注意力），使用因果掩码
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])  # [0] 取注意力输出，忽略注意力权重
        x = self.norm1(x)

        # ==================== 交叉注意力子层 ====================
        # Q=x（来自解码器），K=V=cross（来自编码器）
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        # 第二次归一化
        y = x = self.norm2(x)
        
        # ==================== 前馈网络子层 ====================
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        # 残差连接 + 第三次归一化
        return self.norm3(x + y)


class Decoder(nn.Module):
    """
    Transformer 解码器
    
    由多个解码器层堆叠而成。
    
    结构：
        x → DecoderLayer 1 → DecoderLayer 2 → ... → DecoderLayer N 
          → LayerNorm → Projection → output
    
    与编码器的区别：
    - 每层接收编码器输出作为交叉注意力的 Key/Value
    - 可选的投影层用于将输出映射到目标维度
    """
    
    def __init__(self, layers, norm_layer=None, projection=None):
        """
        初始化解码器
        
        参数:
            layers (list): 解码器层列表
            norm_layer (nn.Module): 最终归一化层（可选）
            projection (nn.Module): 输出投影层（可选）
        """
        super(Decoder, self).__init__()
        
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection  # 用于将 d_model 映射到输出维度

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        """
        前向传播
        
        参数:
            x (Tensor): 解码器输入 [B, L_dec, d_model]
            cross (Tensor): 编码器输出 [B, L_enc, d_model]
            x_mask: 自注意力掩码
            cross_mask: 交叉注意力掩码
            tau, delta: 预留参数
        
        返回:
            output (Tensor): 解码器输出
                            [B, L_dec, d_model]（无投影层）
                            [B, L_dec, c_out]（有投影层）
        """
        # 依次通过各解码器层
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        # 最终归一化（可选）
        if self.norm is not None:
            x = self.norm(x)

        # 输出投影（可选）
        if self.projection is not None:
            x = self.projection(x)
            
        return x

