"""
masking.py - 注意力掩码模块

本模块实现了 Transformer 模型中使用的注意力掩码，主要包括：
1. TriangularCausalMask: 三角因果掩码，用于自回归解码
2. ProbMask: 概率稀疏掩码，用于 Informer 的 ProbSparse 自注意力

掩码的作用：
    在注意力计算中，掩码用于控制每个位置可以"看到"哪些位置的信息。
    被掩码的位置（mask=True）在计算 softmax 前会被设置为 -inf，
    从而在 softmax 后变为 0，实现信息屏蔽。

掩码类型说明：
    1. 因果掩码（Causal Mask）：
       - 确保位置 i 只能看到位置 0, 1, ..., i-1 的信息（禁止看到未来）
       - 用于解码器的自注意力，防止信息泄露
       
    2. 概率稀疏掩码（ProbMask）：
       - Informer 提出的稀疏注意力机制
       - 只选择最重要的 Query 进行完整注意力，其余使用均值
       - 大幅降低计算复杂度：O(n²) → O(n log n)

注意力掩码示意图（因果掩码，L=5）：
    Query/Key:  0   1   2   3   4
           0  [ 0   1   1   1   1 ]  ← 位置 0 只能看自己
           1  [ 0   0   1   1   1 ]  ← 位置 1 可以看 0, 1
           2  [ 0   0   0   1   1 ]  ← 位置 2 可以看 0, 1, 2
           3  [ 0   0   0   0   1 ]  ← 位置 3 可以看 0, 1, 2, 3
           4  [ 0   0   0   0   0 ]  ← 位置 4 可以看所有
    
    0 = 可访问（not masked）
    1 = 被屏蔽（masked，注意力权重为 0）
"""

import torch  # PyTorch 深度学习框架


class TriangularCausalMask():
    """
    三角因果掩码
    
    生成上三角掩码矩阵，用于自回归解码器的自注意力层。
    确保每个位置只能关注它之前（包括自身）的位置，防止看到未来信息。
    
    属性:
        _mask (Tensor): 布尔掩码张量 [B, 1, L, L]
                       True 表示该位置被屏蔽，False 表示可访问
    
    使用场景:
        - Transformer 解码器的自注意力
        - GPT 等自回归语言模型
        - 时序预测的解码器
    
    示例:
        mask = TriangularCausalMask(B=32, L=100, device='cuda')
        # 在注意力计算中使用
        attn_scores = attn_scores.masked_fill(mask.mask, float('-inf'))
    """
    
    def __init__(self, B, L, device="cpu"):
        """
        初始化三角因果掩码
        
        参数:
            B (int): 批次大小 (Batch size)
            L (int): 序列长度 (Sequence length)
            device (str): 设备类型，'cpu' 或 'cuda'
        
        生成的掩码形状: [B, 1, L, L]
            - B: 批次维度
            - 1: 头维度（广播到所有注意力头）
            - L, L: Query 和 Key 的序列长度
        """
        mask_shape = [B, 1, L, L]
        
        # torch.no_grad() 确保掩码不参与梯度计算
        with torch.no_grad():
            # torch.triu: 生成上三角矩阵
            # diagonal=1: 主对角线以上的元素为 True（不包括主对角线）
            # 这意味着位置 i 可以看到位置 0 到 i（包括自己）
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), 
                diagonal=1
            ).to(device)

    @property
    def mask(self):
        """
        获取掩码张量
        
        返回:
            Tensor: 布尔掩码 [B, 1, L, L]
        """
        return self._mask


class ProbMask():
    """
    概率稀疏掩码（ProbSparse Mask）
    
    Informer 模型提出的稀疏注意力掩码。
    在 ProbSparse 自注意力中，只选择最重要的 Top-k Query 进行完整注意力计算，
    其余 Query 使用 Key 的均值代替，从而降低计算复杂度。
    
    工作原理:
        1. 计算每个 Query 的"稀疏性度量"（Sparsity Measurement）
        2. 选择 Top-u 个最重要的 Query（u = c * ln(L)）
        3. 只对这些 Query 进行完整的注意力计算
        4. 其余 Query 使用均值注意力
    
    属性:
        _mask (Tensor): 布尔掩码张量，形状与注意力分数相同
    
    参考:
        "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting"
        https://arxiv.org/abs/2012.07436
    """
    
    def __init__(self, B, H, L, index, scores, device="cpu"):
        """
        初始化概率稀疏掩码
        
        参数:
            B (int): 批次大小 (Batch size)
            H (int): 注意力头数 (Number of heads)
            L (int): 序列长度 (Sequence length)
            index (Tensor): 被选中的 Top-u Query 的索引 [B, H, u]
            scores (Tensor): 注意力分数张量，用于确定掩码形状
            device (str): 设备类型
        
        原理:
            1. 创建基础上三角掩码（因果掩码）
            2. 根据选中的 Query 索引，提取对应的掩码行
            3. 重塑为与 scores 相同的形状
        """
        # 创建基础上三角掩码 [L, S]，S 是 Key 的长度
        # triu(1) 生成严格上三角（不包括对角线）
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        
        # 扩展掩码到 [B, H, L, S] 形状
        # None 添加新维度，expand 进行广播扩展
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        
        # 根据 index 提取被选中 Query 对应的掩码
        # 使用高级索引从扩展掩码中选取特定行
        # torch.arange(B)[:, None, None] 生成批次索引
        # torch.arange(H)[None, :, None] 生成头索引
        # index 提供 Query 位置索引
        indicator = _mask_ex[
            torch.arange(B)[:, None, None],   # 批次索引 [B, 1, 1]
            torch.arange(H)[None, :, None],   # 头索引 [1, H, 1]
            index,                             # Query 索引 [B, H, u]
            :                                  # 所有 Key 位置
        ].to(device)
        
        # 重塑为与 scores 相同的形状
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        """
        获取掩码张量
        
        返回:
            Tensor: 布尔掩码，形状与 scores 相同
        """
        return self._mask

