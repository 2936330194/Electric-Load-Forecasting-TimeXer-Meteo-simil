import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from math import sqrt
import os


class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    
    自相关机制（AutoCorrelation Mechanism），包含以下两个主要阶段：
    (1) 基于周期的依赖发现（找出序列中隐含的周期性）
    (2) 时间延迟聚合（根据找到的周期，聚合相似的子序列）
    该模块设计为可以无缝替换标准的自注意力（Self-Attention）系列机制。
    """

    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor  # top_k 延迟（延迟聚合的数量）的一个缩放因子
        self.scale = scale    # 缩放系数
        self.mask_flag = mask_flag  # 是否使用注意力掩码
        self.output_attention = output_attention  # 是否输出注意力/相关性权重矩阵
        self.dropout = nn.Dropout(attention_dropout)  # attention 的 dropout 层（在此实现中似乎未被显式调用）

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        
        自相关的时间延迟聚合（训练阶段使用的加速版本）。
        采用类似批量归一化（batch-normalization）的风格设计，主要是对不同延迟量进行加权求和。
        
        参数:
            values: 值矩阵，形状为 [batch, head, channel, length] (注意：输入时经过了 permute)
            corr: 自相关性系数（通过 FFT 计算得到），表示不同时间延迟下的相关性强度
        返回:
            delays_agg: 经过时间延迟聚合后的输出张量
        """
        head = values.shape[1]   # 多头注意力中的头数
        channel = values.shape[2]# 通道维度
        length = values.shape[3] # 序列长度
        
        # find top k (找出相关性最高的 top_k 个时间延迟)
        top_k = int(self.factor * math.log(length))
        
        # 对自相关系数在 channel 和 head 维度上取平均，得到每个延迟的平均相关性
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        
        # 在 batch 维度上再取平均，然后找出最相关的 top_k 个延迟量的索引 (index)
        # 训练阶段使用全局（基于 entire batch）的平均相关性来选择延迟索引，这有助于提升训练速度和稳定性
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        
        # 根据找出的 top_k 个延迟索引，提取对应的相关性权重
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        
        # update corr (更新自相关权重，使用 softmax 将其归一化为概率分布)
        tmp_corr = torch.softmax(weights, dim=-1)
        
        # aggregation (时间延迟聚合)
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()  # 初始化聚合结果为全 0 张量
        for i in range(top_k):
            # 获取第 i 个最相关的延迟量对应的序列模式
            # 使用 torch.roll 操作将序列循环平移，移动的步数即为对应的延迟量 index[i]
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            
            # 将平移后的序列模式乘以对应的归一化权重，并累加到聚合结果中
            # tmp_corr 形状通过一系列 unsqueeze 和 repeat 扩展，使其与 pattern 形状一致
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        
        自相关的时间延迟聚合（推理阶段使用的加速版本）。
        
        参数:
            values: 值矩阵
            corr: 自相关性系数
        返回:
            delays_agg: 聚合后的张量
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        
        # index init (初始化索引表，用于后续在 gather 操作中选取时间步)
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).to(values.device)
        
        # find top k (找出每个样本最相关的 top_k 个时间延迟)
        # 注意推理阶段不对 batch 维度取平均，而是为每个 batch 独立找到最相关的延迟
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)  # 对 head 和 channel 取平均
        weights, delay = torch.topk(mean_value, top_k, dim=-1)   # 获取 top_k 对应的权重和延迟量
        
        # update corr (对 top_k 权重使用 softmax 归一化)
        tmp_corr = torch.softmax(weights, dim=-1)
        
        # aggregation (时间延迟聚合)
        # 将 values 的 length 维度复制一次，形状变为 [batch, head, channel, 2 * length]
        # 这是为了在应用延迟时充当循环填充的作用
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            # 计算平移后的索引，结合了基础序列索引和当前找到的延迟量
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            # 使用 gather 函数按照计算出的延迟索引提取序列模式
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            # 加权求和
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        
        标准版本的自相关聚合运算。它不对 channel 和 head 维度求平均来选延迟，
        而是针对每个维度独立寻找 top_k 延迟。计算开销可能更大。
        
        参数:
            values: 值矩阵
            corr: 计算出的完整自相关度
        返回:
            delays_agg: 聚合结果张量
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        
        # index init (初始化索引)
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).to(values.device)
        
        # find top k
        top_k = int(self.factor * math.log(length))
        # 直接在原 corr 张量（不求均值）中寻找 top_k 延迟
        weights, delay = torch.topk(corr, top_k, dim=-1)
        
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        
        # aggregation
        # 在长度维度复制以处理循环平移
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            # 将基础索引加上在 delay 矩阵中找到的延迟索引
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            # 使用 torch.gather 提取延迟后的模式
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            # 乘以归一化后的相关性权重累加
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        """
        前向传播函数。
        用基于快速傅里叶变换（FFT）的自相关代替点积注意力（Dot-Product Attention）。
        
        参数:
            queries: 查询张量, 形状为 [Batch, L, Head, D_keys]
            keys: 键张量, 形状为 [Batch, S, Head, D_keys]
            values: 值张量, 形状为 [Batch, S, Head, D_values]
            attn_mask: 注意力掩码（这里暂未使用，为了保持与标准注意力层接口兼容）
            
        返回:
            (V, corr): 聚合后的特征 V 和（可选的）相关度矩阵 corr
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        
        # 如果 query 的长度大于 key/value 的长度（例如在 Decoder 的交叉注意力中），则用 0 对 keys/values 进行补齐
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        # 如果 query 长度较短，则对 keys/values 进行截断保证长度一致
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies (基于周期的依赖计算)
        # 通过傅里叶变换来计算自相关性。首先将序列转换到频域。
        # 先通过 permute 把长度 L 换到最后一个维度以便进行 FFT 计算
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        
        # 在频域下计算点积（使用共轭相乘）。频域的点积等价于时域中的循环卷积（也就是自相关函数定理）
        res = q_fft * torch.conj(k_fft)
        
        # 再通过逆傅里叶变换转回时域，得到时域下的序列自相关系数
        corr = torch.fft.irfft(res, dim=-1)

        # time delay agg (时间延迟聚合：找到重要周期并对 values 进行聚合)
        if self.training:
            # 训练阶段使用 time_delay_agg_training 以类似批归一化的方式选延迟，计算更稳定
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            # 推理阶段使用 time_delay_agg_inference 为每个独立样本寻找和计算延迟
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        # 返回聚合完成的特征输出。如果配置了输出 attention，则将相关度 corr 也一并返回
        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)


class AutoCorrelationLayer(nn.Module):
    """
    自动相关层的封装模块，包含了线性投影层（类似 Multi-Head Attention 中的 Q, K, V 处理层），
    内部封装了具体的 AutoCorrelation 运算模块。
    """
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        # 如果未指定 d_keys 或是 d_values，则默认按照 d_model // n_heads 大小等分
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation  # 内部相关的核心操作（即前面的 AutoCorrelation 类实例）
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)  # Q 投影层
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)    # K 投影层
        self.value_projection = nn.Linear(d_model, d_values * n_heads)# V 投影层
        self.out_projection = nn.Linear(d_values * n_heads, d_model)  # 输出 O 投影层
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        """
        前向传播
        参数:
            queries: 查询张量, 形状 [Batch, L, d_model]
            keys:    键张量,   形状 [Batch, S, d_model]
            values:  值张量,   形状 [Batch, S, d_model]
            attn_mask: 序列掩码（这里暂未使用，保证接口统一）
        返回:
            投影计算后的特征张量（形状 [Batch, L, d_model]）和相关度分数（attn）
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # 通过线性层将输入投影为多头特征，并变形为 [Batch, Seq_len, n_heads, d_k]
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # 调用具体的自相关机制（inner_correlation）来进行特征提取
        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )
        
        # 将多头的输出特征拼接到一起，变形为 [Batch, L, d_values * n_heads]
        out = out.view(B, L, -1)

        # 最后经过一个线性投影层后返回整合的特征
        return self.out_projection(out), attn
