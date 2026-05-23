"""
SelfAttention_Family.py - 自注意力机制模块

本模块实现了 Transformer 模型中的自注意力机制，主要包括：
1. FullAttention: 标准的全注意力（Scaled Dot-Product Attention）
2. AttentionLayer: 多头注意力层的封装

自注意力机制核心公式：
    Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
    
    其中：
    - Q (Query): 查询向量，表示"我在找什么"
    - K (Key): 键向量，表示"我有什么"
    - V (Value): 值向量，表示"我的内容是什么"
    - d_k: Key 的维度，用于缩放防止梯度消失

多头注意力机制：
    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W^O
    where head_i = Attention(Q × W_i^Q, K × W_i^K, V × W_i^V)
    
    多头注意力允许模型同时关注不同位置的不同表示子空间。

数据流示意图：
    输入 [B, L, d_model]
           ↓
    ┌──────────────────────────────────────┐
    │  Linear Projections (Q, K, V)        │
    └──────────────────────────────────────┘
           ↓
    [B, L, H, d_k] (reshape to multi-head)
           ↓
    ┌──────────────────────────────────────┐
    │  Scaled Dot-Product Attention        │
    │  scores = Q × K^T / √d_k             │
    │  attn = softmax(scores) × V          │
    └──────────────────────────────────────┘
           ↓
    [B, L, H, d_v] → reshape → [B, L, d_model]
           ↓
    ┌──────────────────────────────────────┐
    │  Output Projection                    │
    └──────────────────────────────────────┘
           ↓
    输出 [B, L, d_model]

参数说明：
    B: batch_size (批次大小)
    L: query 序列长度
    S: key/value 序列长度（自注意力时 S = L）
    H: n_heads (注意力头数)
    E/d_k: 每个头的 key 维度
    d_v: 每个头的 value 维度
"""

import torch               # PyTorch 深度学习框架
import torch.nn as nn      # 神经网络模块
import numpy as np
from math import sqrt      # 平方根函数，用于缩放因子计算
from utils.masking import TriangularCausalMask, ProbMask


class FullAttention(nn.Module):
    """
    全注意力模块（Scaled Dot-Product Attention）
    
    实现标准的缩放点积注意力机制，是 Transformer 的核心组件。
    
    计算过程：
    1. 计算注意力分数：scores = Q × K^T
    2. 缩放：scores = scores / √d_k
    3. 掩码（可选）：将被掩码位置设为 -inf
    4. Softmax：attn_weights = softmax(scores)
    5. 加权求和：output = attn_weights × V
    
    属性:
        scale (float): 缩放因子，默认为 1/√d_k
        mask_flag (bool): 是否使用因果掩码
        output_attention (bool): 是否返回注意力权重
        dropout (nn.Dropout): 注意力权重的 Dropout
    """
    
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        """
        初始化全注意力模块
        
        参数:
            mask_flag (bool): 是否使用因果掩码
                             True - 解码器自注意力（防止看到未来）
                             False - 编码器自注意力或交叉注意力
            factor (int): 预留参数（用于稀疏注意力的采样因子，这里未使用）
            scale (float): 自定义缩放因子，None 则使用 1/√d_k
            attention_dropout (float): 注意力权重的 Dropout 比率
            output_attention (bool): 是否在输出中包含注意力权重
        """
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        """
        前向传播：计算注意力
        
        参数:
            queries (Tensor): 查询张量 [B, L, H, E]
            keys (Tensor): 键张量 [B, S, H, E]
            values (Tensor): 值张量 [B, S, H, D]
            attn_mask: 注意力掩码，None 则使用因果掩码
            tau, delta: 预留参数（用于某些变体，如 De-stationary Attention）
        
        返回:
            V (Tensor): 注意力输出 [B, L, H, D]
            A (Tensor or None): 注意力权重 [B, H, L, S]（如果 output_attention=True）
        
        维度变化:
            queries: [B, L, H, E]
            keys: [B, S, H, E]
            scores = Q × K^T: [B, H, L, S]
            A = softmax(scores): [B, H, L, S]
            values: [B, S, H, D]
            V = A × values: [B, L, H, D]
        """
        # 获取维度信息
        B, L, H, E = queries.shape  # B: batch, L: query长度, H: 头数, E: 每头维度
        _, S, _, _ = values.shape   # S: key/value 长度
        
        # 计算缩放因子：1/√d_k，防止点积过大导致 softmax 梯度消失
        scale = self.scale or 1. / sqrt(E)

        # ==================== 计算注意力分数 ====================
        # 使用 einsum 进行高效的批量矩阵乘法
        # "blhe,bshe->bhls" 表示：
        # - b: batch, l: query位置, h: head, e: embedding
        # - s: key位置
        # 结果: [B, H, L, S] 表示每个query对每个key的注意力分数
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # ==================== 应用掩码 ====================
        if self.mask_flag:
            if attn_mask is None:
                # 使用三角因果掩码（用于解码器自注意力）
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            # 将掩码位置填充为 -inf，softmax 后变为 0
            scores.masked_fill_(attn_mask.mask, -torch.inf)

        # ==================== Softmax 和 Dropout ====================
        # 缩放后应用 softmax，在 key 维度（最后一维）上归一化
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        
        # ==================== 加权求和 ====================
        # "bhls,bshd->blhd" 表示：
        # - A: [B, H, L, S] 注意力权重
        # - values: [B, S, H, D]
        # 结果: [B, L, H, D] 注意力输出
        V = torch.einsum("bhls,bshd->blhd", A, values)

        # 返回结果
        if self.output_attention:
            return V.contiguous(), A
        return V.contiguous(), None



class ProbAttention(nn.Module):
    """ProbSparse attention used by Informer."""

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k), device=K.device)
        K_sample = K_expand[:, :, torch.arange(L_Q, device=K.device).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        Q_reduce = Q[
            torch.arange(B, device=Q.device)[:, None, None],
            torch.arange(H, device=Q.device)[None, :, None],
            M_top,
            :,
        ]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            context = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            assert L_Q == L_V
            context = V.cumsum(dim=-2)
        return context

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)
        context_in[
            torch.arange(B, device=V.device)[:, None, None],
            torch.arange(H, device=V.device)[None, :, None],
            index,
            :,
        ] = torch.matmul(attn, V).type_as(context_in)

        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V], device=attn.device) / L_V).type_as(attn)
            attns[
                torch.arange(B, device=V.device)[:, None, None],
                torch.arange(H, device=V.device)[None, :, None],
                index,
                :,
            ] = attn
            return context_in, attns
        return context_in, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype("int").item()
        u = self.factor * np.ceil(np.log(L_Q)).astype("int").item()
        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        scale = self.scale or 1.0 / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        return context.contiguous(), attn
class AttentionLayer(nn.Module):
    """
    多头注意力层
    
    封装了完整的多头注意力计算流程：
    1. 线性投影：将输入投影到 Q, K, V
    2. 分头：将投影结果重塑为多头形式
    3. 注意力计算：调用内部注意力模块
    4. 合并：将多头输出合并并投影回原始维度
    
    属性:
        inner_attention: 内部注意力模块（如 FullAttention）
        query_projection: Q 投影层
        key_projection: K 投影层
        value_projection: V 投影层
        out_projection: 输出投影层
        n_heads: 注意力头数
    """
    
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        """
        初始化多头注意力层
        
        参数:
            attention (nn.Module): 内部注意力模块实例
            d_model (int): 模型维度（输入/输出维度）
            n_heads (int): 注意力头数
            d_keys (int): 每个头的 key 维度，默认为 d_model // n_heads
            d_values (int): 每个头的 value 维度，默认为 d_model // n_heads
        
        约束:
            d_model 应该能被 n_heads 整除
        """
        super(AttentionLayer, self).__init__()

        # 设置每个头的维度，默认均分
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        # 内部注意力模块
        self.inner_attention = attention
        
        # 线性投影层：将 d_model 投影到 n_heads * d_keys/d_values
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        
        # 输出投影层：将多头输出投影回 d_model
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        """
        前向传播：多头注意力计算
        
        参数:
            queries (Tensor): 查询输入 [B, L, d_model]
            keys (Tensor): 键输入 [B, S, d_model]
            values (Tensor): 值输入 [B, S, d_model]
            attn_mask: 注意力掩码
            tau, delta: 预留参数
        
        返回:
            output (Tensor): 注意力输出 [B, L, d_model]
            attn (Tensor or None): 注意力权重
        
        处理流程:
            1. 线性投影 + 分头：[B, L, d_model] -> [B, L, H, d_k]
            2. 注意力计算：[B, L, H, d_k] -> [B, L, H, d_v]
            3. 合并 + 输出投影：[B, L, H, d_v] -> [B, L, d_model]
        """
        # 获取维度信息
        B, L, _ = queries.shape  # B: batch, L: query长度
        _, S, _ = keys.shape     # S: key/value长度
        H = self.n_heads         # H: 头数

        # ==================== 线性投影 + 分头 ====================
        # 投影后重塑：[B, L, d_model] -> [B, L, H*d_k] -> [B, L, H, d_k]
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # ==================== 注意力计算 ====================
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta,
        )
        
        # ==================== 合并多头 + 输出投影 ====================
        # 重塑：[B, L, H, d_v] -> [B, L, H*d_v]
        out = out.view(B, L, -1)
        
        # 输出投影：[B, L, H*d_v] -> [B, L, d_model]
        return self.out_projection(out), attn

