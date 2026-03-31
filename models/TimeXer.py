"""
TimeXer.py - TimeXer 时间序列预测模型

论文: "TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables"
链接: https://arxiv.org/abs/2402.19072

TimeXer 核心创新：
    1. 内生变量（Endogenous）和外生变量（Exogenous）分离处理
    2. 全局 Token（Global Token）用于捕获序列整体信息
    3. 交叉注意力（Cross-Attention）融合外生变量信息
    
定义说明：
    - 内生变量（Endogenous）: 需要预测的目标变量，如电力负荷
    - 外生变量（Exogenous）: 辅助预测的协变量，如温度、湿度、时间特征
    
模型架构示意图：
    ┌─────────────────────────────────────────────────────────────────────┐
    │                            TimeXer                                  │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │  输入: x_enc [B, L, C] = [内生变量 + 外生变量]                       │
    │                                                                     │
    │  ┌────────────────────────┐    ┌────────────────────────┐          │
    │  │    内生变量处理         │    │    外生变量处理         │          │
    │  │  (Endogenous Branch)   │    │  (Exogenous Branch)    │          │
    │  │                        │    │                        │          │
    │  │  1. Patching           │    │  1. Inverted Embedding │          │
    │  │  2. Value Embedding    │    │  (每个变量作为 Token)   │          │
    │  │  3. Position Embedding │    │                        │          │
    │  │  4. Global Token       │    │  输出: [B, C-1, d_model]│          │
    │  │                        │    │                        │          │
    │  │  输出: [B*n_vars,      │    │                        │          │
    │  │   patch_num+1, d_model]│    │                        │          │
    │  └────────────────────────┘    └────────────────────────┘          │
    │              ↓                            ↓                        │
    │  ┌─────────────────────────────────────────────────────────────┐   │
    │  │                     EncoderLayer × N                        │   │
    │  │  ┌─────────────────┐                                        │   │
    │  │  │  Self-Attention │  (内生变量 Patch 之间的注意力)          │   │
    │  │  └─────────────────┘                                        │   │
    │  │           ↓                                                 │   │
    │  │  ┌─────────────────┐                                        │   │
    │  │  │ Cross-Attention │  (Global Token 关注外生变量)           │   │
    │  │  └─────────────────┘                                        │   │
    │  │           ↓                                                 │   │
    │  │  ┌─────────────────┐                                        │   │
    │  │  │  Feed Forward   │                                        │   │
    │  │  └─────────────────┘                                        │   │
    │  └─────────────────────────────────────────────────────────────┘   │
    │              ↓                                                     │
    │  ┌─────────────────────────────────────────────────────────────┐   │
    │  │  Flatten Head: 展平 + 线性投影到 pred_len                    │   │
    │  └─────────────────────────────────────────────────────────────┘   │
    │              ↓                                                     │
    │  ┌─────────────────────────────────────────────────────────────┐   │
    │  │  Denormalization: 反归一化恢复原始尺度                       │   │
    │  └─────────────────────────────────────────────────────────────┘   │
    │              ↓                                                     │
    │  输出: [B, pred_len, 1]（单变量预测）                              │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘

Global Token 的作用：
    1. 作为内生变量序列的"汇总"表示
    2. 通过交叉注意力融合外生变量信息
    3. 避免 Patch Token 直接参与交叉注意力，降低计算复杂度

与其他模型的区别：
    - PatchTST: Channel-Independence，变量独立处理
    - iTransformer: 变量作为 Token，捕获变量间关系
    - TimeXer: 内生/外生分离，通过 Global Token 融合
"""

import torch                                                    # PyTorch 深度学习框架
import torch.nn as nn                                           # 神经网络模块
import torch.nn.functional as F                                 # 函数式接口
from layers.SelfAttention_Family import FullAttention, AttentionLayer  # 自注意力层
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding   # 嵌入层
import numpy as np                                              # 数值计算


class FlattenHead(nn.Module):
    """
    展平预测头
    
    将编码器输出展平并投影到目标预测长度。
    
    处理流程：
        输入: [B, n_vars, d_model, patch_num+1]
        展平: [B, n_vars, d_model * (patch_num+1)]
        线性: [B, n_vars, pred_len]
        Dropout: [B, n_vars, pred_len]
    """
    
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        """
        初始化预测头
        
        参数:
            n_vars (int): 变量数量
            nf (int): 展平后的特征维度 (d_model * (patch_num + 1))
            target_window (int): 目标预测长度
            head_dropout (float): Dropout 比率
        """
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)  # 展平最后两个维度
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        """
        前向传播
        
        参数:
            x (Tensor): 编码器输出 [B, n_vars, d_model, patch_num+1]
        
        返回:
            Tensor: 预测输出 [B, n_vars, pred_len]
        """
        x = self.flatten(x)   # [B, n_vars, d_model*(patch_num+1)]
        x = self.linear(x)    # [B, n_vars, pred_len]
        x = self.dropout(x)
        return x


class EnEmbedding(nn.Module):
    """
    内生变量嵌入层（Endogenous Embedding）
    
    处理目标变量（需要预测的变量）：
    1. Patching: 将序列分割成固定长度的 Patch
    2. Value Embedding: 将每个 Patch 映射到 d_model
    3. Position Embedding: 添加位置编码
    4. Global Token: 添加可学习的全局 Token
    
    Global Token 示意图：
        Patch Tokens: [P1, P2, P3, ..., Pn]
        加入 Global:  [P1, P2, P3, ..., Pn, GLB]
        
        GLB 通过交叉注意力融合外生变量信息
    """
    
    def __init__(self, n_vars, d_model, patch_len, dropout):
        """
        初始化内生变量嵌入
        
        参数:
            n_vars (int): 内生变量数量
            d_model (int): 模型维度
            patch_len (int): Patch 长度
            dropout (float): Dropout 比率
        """
        super(EnEmbedding, self).__init__()
        
        # Patch 参数
        self.patch_len = patch_len

        # 值嵌入：将 patch_len 维的 Patch 映射到 d_model
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        
        # 全局 Token：可学习的参数，用于汇总序列信息
        # 形状: [1, n_vars, 1, d_model]，会复制到每个样本
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        
        # 位置编码
        self.position_embedding = PositionalEmbedding(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x (Tensor): 内生变量输入 [B, n_vars, L]
        
        返回:
            x (Tensor): 嵌入后的 Patch + Global Token [B*n_vars, patch_num+1, d_model]
            n_vars (int): 变量数量
        
        维度变化：
            输入: [B, n_vars, L]
            Patching: [B, n_vars, patch_num, patch_len]
            重塑: [B*n_vars, patch_num, patch_len]
            嵌入: [B*n_vars, patch_num, d_model]
            恢复: [B, n_vars, patch_num, d_model]
            加 GLB: [B, n_vars, patch_num+1, d_model]
            最终: [B*n_vars, patch_num+1, d_model]
        """
        # 获取变量数
        n_vars = x.shape[1]
        
        # 复制 Global Token 到每个样本
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))  # [B, n_vars, 1, d_model]

        # ==================== Patching ====================
        # unfold: 将序列分割成不重叠的 Patch
        # 输入: [B, n_vars, L] → 输出: [B, n_vars, patch_num, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        
        # 重塑为 Channel-Independent 形式
        # [B, n_vars, patch_num, patch_len] → [B*n_vars, patch_num, patch_len]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        
        # ==================== Input Encoding ====================
        # 值嵌入 + 位置编码
        x = self.value_embedding(x) + self.position_embedding(x)  # [B*n_vars, patch_num, d_model]
        
        # 恢复变量维度
        # [B*n_vars, patch_num, d_model] → [B, n_vars, patch_num, d_model]
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        
        # ==================== 添加 Global Token ====================
        # 在 Patch 序列末尾添加 Global Token
        # [B, n_vars, patch_num, d_model] + [B, n_vars, 1, d_model] → [B, n_vars, patch_num+1, d_model]
        x = torch.cat([x, glb], dim=2)
        
        # 再次重塑为 Channel-Independent 形式
        # [B, n_vars, patch_num+1, d_model] → [B*n_vars, patch_num+1, d_model]
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        
        return self.dropout(x), n_vars


class Encoder(nn.Module):
    """
    TimeXer 编码器
    
    由多个 EncoderLayer 堆叠而成。
    每层接收内生变量嵌入和外生变量嵌入，通过交叉注意力融合。
    """
    
    def __init__(self, layers, norm_layer=None, projection=None):
        """
        初始化编码器
        
        参数:
            layers (list): 编码器层列表
            norm_layer (nn.Module): 最终归一化层
            projection (nn.Module): 输出投影层（可选）
        """
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        """
        前向传播
        
        参数:
            x (Tensor): 内生变量嵌入 [B*n_vars, patch_num+1, d_model]
            cross (Tensor): 外生变量嵌入 [B, C-1, d_model]
            x_mask: 自注意力掩码
            cross_mask: 交叉注意力掩码
            tau, delta: 预留参数
        
        返回:
            x (Tensor): 编码器输出 [B*n_vars, patch_num+1, d_model]
        """
        # 依次通过各编码器层
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        # 最终归一化
        if self.norm is not None:
            x = self.norm(x)

        # 输出投影（如果有）
        if self.projection is not None:
            x = self.projection(x)
        return x


class EncoderLayer(nn.Module):
    """
    TimeXer 编码器层
    
    每层包含三个子层：
    1. 自注意力：内生变量 Patch 之间的注意力
    2. 交叉注意力：Global Token 关注外生变量
    3. 前馈网络
    
    关键设计：
    - 只有 Global Token 参与交叉注意力，Patch Token 不直接关注外生变量
    - 这样可以减少计算量，同时让 Global Token 作为"桥梁"融合信息
    
    交叉注意力结构：
        Query: Global Token [B, 1, d_model]（每个变量一个）
        Key/Value: 外生变量 [B, C-1, d_model]
        
        只有 Global Token 从外生变量获取信息，然后通过后续层传递给 Patch Token
    """
    
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        """
        初始化编码器层
        
        参数:
            self_attention (nn.Module): 自注意力模块
            cross_attention (nn.Module): 交叉注意力模块
            d_model (int): 模型维度
            d_ff (int): 前馈网络隐藏层维度
            dropout (float): Dropout 比率
            activation (str): 激活函数
        """
        super(EncoderLayer, self).__init__()
        
        d_ff = d_ff or 4 * d_model
        
        # 两个注意力模块
        self.self_attention = self_attention    # 内生变量自注意力
        self.cross_attention = cross_attention  # Global Token 交叉注意力
        
        # 前馈网络
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        
        # 三个层归一化
        self.norm1 = nn.LayerNorm(d_model)  # 自注意力后
        self.norm2 = nn.LayerNorm(d_model)  # 交叉注意力后
        self.norm3 = nn.LayerNorm(d_model)  # FFN 后
        
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        """
        前向传播
        
        参数:
            x (Tensor): 内生变量嵌入 [B*n_vars, patch_num+1, d_model]
                       最后一个位置是 Global Token
            cross (Tensor): 外生变量嵌入 [B, C-1, d_model]
            x_mask: 自注意力掩码
            cross_mask: 交叉注意力掩码
            tau, delta: 预留参数
        
        返回:
            Tensor: 编码器层输出 [B*n_vars, patch_num+1, d_model]
        """
        # 获取 batch 相关维度
        B, L, D = cross.shape  # B: 原始 batch_size, L: 外生变量数, D: d_model
        
        # ==================== 1. 自注意力子层 ====================
        # 所有 Patch Token + Global Token 一起参与自注意力
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        # ==================== 2. 交叉注意力子层 ====================
        # 只提取 Global Token（最后一个位置）
        x_glb_ori = x[:, -1, :].unsqueeze(1)  # [B*n_vars, 1, d_model]
        
        # 重塑：[B*n_vars, 1, d_model] → [B, n_vars, d_model]
        x_glb = torch.reshape(x_glb_ori, (B, -1, D))
        
        # Global Token 作为 Query，外生变量作为 Key/Value
        # Query: [B, n_vars, d_model], Key/Value: [B, C-1, d_model]
        x_glb_attn = self.dropout(self.cross_attention(
            x_glb, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])
        
        # 重塑回去：[B, n_vars, d_model] → [B*n_vars, 1, d_model]
        x_glb_attn = torch.reshape(x_glb_attn,
                                   (x_glb_attn.shape[0] * x_glb_attn.shape[1], x_glb_attn.shape[2])).unsqueeze(1)
        
        # 残差连接
        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        # ==================== 3. 组合 Patch Token 和更新后的 Global Token ====================
        # 用更新后的 Global Token 替换原来的
        # x[:, :-1, :] 是所有 Patch Token
        # x_glb 是更新后的 Global Token
        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)

        # ==================== 4. 前馈网络子层 ====================
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Model(nn.Module):
    """
    TimeXer 模型
    
    用于带外生变量的时间序列预测。
    
    核心特点：
    1. 内生变量（目标变量）使用 Patching + Global Token
    2. 外生变量使用 Inverted Embedding（每个变量作为 Token）
    3. 通过交叉注意力融合外生变量信息
    
    支持的预测模式：
    - MS (Multivariate to Single): 多变量输入，单变量输出
      最后一个变量是目标，其他是外生变量
    - M (Multivariate): 多变量输入，多变量输出
      所有变量都是内生变量
    """

    def __init__(self, configs):
        """
        初始化 TimeXer 模型
        
        参数:
            configs: 配置对象，包含以下属性：
                - task_name: 任务名称
                - features: 预测模式 ('M' 或 'MS')
                - seq_len: 输入序列长度
                - pred_len: 预测序列长度
                - use_norm: 是否使用归一化
                - patch_len: Patch 长度
                - enc_in: 输入变量数
                - d_model: 模型维度
                - n_heads: 注意力头数
                - e_layers: 编码器层数
                - d_ff: 前馈网络隐藏层维度
                - dropout: Dropout 比率
                - activation: 激活函数
                - embed: 嵌入类型
                - freq: 时间频率
        """
        super(Model, self).__init__()
        
        # 保存配置
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.patch_len = configs.patch_len
        self.enc_in = configs.enc_in
        
        # 外生变量序列长度（可与内生变量不同，例如包含未来气象数据）
        # 当 exo_seq_len > seq_len 时，外生变量可以覆盖预测窗口的未来信息
        self.exo_seq_len = int(getattr(configs, 'exo_seq_len', configs.seq_len))
        
        # 计算 Patch 数量（不重叠分割）
        self.patch_num = int(configs.seq_len // configs.patch_len)
        
        # 内生变量数量
        # MS 模式：只有最后一个变量是内生的
        # M 模式：所有变量都是内生的
        self.n_vars = 1 if configs.features == 'MS' else configs.enc_in

        # ==================== 嵌入层 ====================
        # 内生变量嵌入：Patching + Global Token
        self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout)

        # 外生变量嵌入：Inverted Embedding（每个变量作为 Token）
        # 使用 exo_seq_len 而非 seq_len，以支持不同长度的外生变量输入
        self.ex_embedding = DataEmbedding_inverted(self.exo_seq_len, configs.d_model, configs.embed, configs.freq,
                                                   configs.dropout)

        # ==================== 编码器 ====================
        self.encoder = Encoder(
            [
                EncoderLayer(
                    # 自注意力：内生变量 Patch 之间
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    # 交叉注意力：Global Token → 外生变量
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # ==================== 预测头 ====================
        # 输入维度：d_model * (patch_num + 1)，+1 是 Global Token
        self.head_nf = configs.d_model * (self.patch_num + 1)
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_exo=None, x_exo_mark=None, x_exo_extra_tokens=None):
        """
        MS 模式预测：多变量输入，单变量输出
        
        支持两种模式：
        1. 合并模式（x_exo=None）：x_enc 包含内生+外生变量，长度相同
        2. 分离模式（x_exo≠None）：内生/外生变量分开传入，可以有不同序列长度
           用于外生变量包含未来信息的场景（如 672+96=768 步气象数据）
        
        参数:
            x_enc (Tensor): 合并模式 [B, seq_len, C] 或 分离模式 [B, seq_len, n_endo_vars]
            x_mark_enc (Tensor): 内生变量时间标记 [B, seq_len, T]
            x_dec, x_mark_dec: 解码器输入（未使用）
            x_exo (Tensor, optional): 外生变量 [B, exo_seq_len, C_exo]
            x_exo_mark (Tensor, optional): 外生变量时间标记 [B, exo_seq_len, T]
        
        返回:
            dec_out (Tensor): 预测输出 [B, pred_len, 1]
        """
        # ==================== 1. Instance Normalization（内生变量） ====================
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()  # [B, 1, C]
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        _, _, N = x_enc.shape  # N: 变量数

        if x_exo is not None:
            # ======== 分离模式：内生/外生变量有不同序列长度 ========
            # 外生变量独立归一化
            if self.use_norm:
                exo_means = x_exo.mean(1, keepdim=True).detach()
                x_exo = x_exo - exo_means
                exo_stdev = torch.sqrt(torch.var(x_exo, dim=1, keepdim=True, unbiased=False) + 1e-5)
                x_exo = x_exo / exo_stdev

            # 内生变量嵌入（与合并模式相同，取最后一个变量）
            en_embed, n_vars = self.en_embedding(x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1))
            # 外生变量嵌入（使用独立的 exo_seq_len）
            ex_embed = self.ex_embedding(x_exo, x_exo_mark)
        else:
            # ======== 合并模式：内生/外生变量在同一张量中 ========
            # 内生变量嵌入
            en_embed, n_vars = self.en_embedding(x_enc[:, :, -1].unsqueeze(-1).permute(0, 2, 1))
            # 外生变量嵌入
            ex_embed = self.ex_embedding(x_enc[:, :, :-1], x_mark_enc)

        if x_exo_extra_tokens is not None:
            if x_exo_extra_tokens.ndim != 3:
                raise ValueError(
                    f"x_exo_extra_tokens should be [B, N_extra, d_model], got {tuple(x_exo_extra_tokens.shape)}"
                )
            if x_exo_extra_tokens.shape[0] != ex_embed.shape[0]:
                raise ValueError(
                    "x_exo_extra_tokens batch dimension does not match exogenous embeddings: "
                    f"{x_exo_extra_tokens.shape[0]} vs {ex_embed.shape[0]}"
                )
            if x_exo_extra_tokens.shape[2] != ex_embed.shape[2]:
                raise ValueError(
                    "x_exo_extra_tokens feature dimension does not match d_model: "
                    f"{x_exo_extra_tokens.shape[2]} vs {ex_embed.shape[2]}"
                )
            ex_embed = torch.cat(
                [
                    ex_embed,
                    x_exo_extra_tokens.to(device=ex_embed.device, dtype=ex_embed.dtype),
                ],
                dim=1,
            )

        # ==================== 3. 编码器 ====================
        enc_out = self.encoder(en_embed, ex_embed)
        
        # 维度恢复：[B*n_vars, patch_num+1, d_model] → [B, n_vars, patch_num+1, d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # 转置：[B, n_vars, patch_num+1, d_model] → [B, n_vars, d_model, patch_num+1]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # ==================== 4. 预测头 ====================
        dec_out = self.head(enc_out)  # [B, n_vars, pred_len]
        dec_out = dec_out.permute(0, 2, 1)  # [B, pred_len, n_vars]

        # ==================== 5. Denormalization ====================
        if self.use_norm:
            # 只恢复目标变量（最后一个变量）的尺度
            dec_out = dec_out * (stdev[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, -1:].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        M 模式预测：多变量输入，多变量输出
        
        所有变量都作为内生变量，同时预测多个变量。
        
        参数:
            x_enc (Tensor): 编码器输入 [B, L, C]
            x_mark_enc (Tensor): 时间标记 [B, L, T]
            x_dec, x_mark_dec: 解码器输入（未使用）
        
        返回:
            dec_out (Tensor): 预测输出 [B, pred_len, C]
        """
        # ==================== 1. Instance Normalization ====================
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        _, _, N = x_enc.shape

        # ==================== 2. 所有变量作为内生变量 ====================
        # x_enc: [B, L, C] → permute → [B, C, L]
        en_embed, n_vars = self.en_embedding(x_enc.permute(0, 2, 1))
        
        # 外生变量嵌入（这里用全部变量作为外生信息）
        ex_embed = self.ex_embedding(x_enc, x_mark_enc)

        # ==================== 3. 编码器 ====================
        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # ==================== 4. 预测头 ====================
        dec_out = self.head(enc_out)  # [B, n_vars, pred_len]
        dec_out = dec_out.permute(0, 2, 1)  # [B, pred_len, n_vars]

        # ==================== 5. Denormalization ====================
        if self.use_norm:
            # 恢复所有变量的尺度
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, x_exo=None, x_exo_mark=None, x_exo_extra_tokens=None):
        """
        模型前向传播（统一入口）
        
        根据 task_name 和 features 分发到不同的处理方法。
        
        参数:
            x_enc (Tensor): 编码器输入 [B, L, C]
            x_mark_enc (Tensor): 编码器时间标记
            x_dec (Tensor): 解码器输入
            x_mark_dec (Tensor): 解码器时间标记
            mask (Tensor): 掩码（未使用）
            x_exo (Tensor, optional): 外生变量 [B, exo_seq_len, C_exo]（分离模式）
            x_exo_mark (Tensor, optional): 外生变量时间标记 [B, exo_seq_len, T]
        
        返回:
            预测输出：
            - MS 模式: [B, pred_len, 1]
            - M 模式: [B, pred_len, C]
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if self.features == 'M':
                # 多变量预测模式
                dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
                return dec_out[:, -self.pred_len:, :]  # [B, pred_len, C]
            else:
                # MS 模式：多变量输入，单变量输出（支持分离外生变量）
                dec_out = self.forecast(
                    x_enc,
                    x_mark_enc,
                    x_dec,
                    x_mark_dec,
                    x_exo=x_exo,
                    x_exo_mark=x_exo_mark,
                    x_exo_extra_tokens=x_exo_extra_tokens,
                )
                return dec_out[:, -self.pred_len:, :]  # [B, pred_len, 1]
        else:
            return None
