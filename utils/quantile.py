"""
quantile.py - 分位数损失函数（Pinball Loss）实现模块

该模块包含了用于概率预测的分位数损失函数实现。通过优化此损失函数，
模型可以学习到预测目标在不同概率水平下的取值，从而提供预测的不确定度估计。
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn


class QuantileLoss(nn.Module):
    """
    分位数损失函数（Pinball Loss），用于估计指定分位数的数值。
    
    对于给定的分位数 q (0 < q < 1)，Pinball Loss 的公式为：
    L(y, f) = max(q * (y - f), (q - 1) * (y - f))
    其中 y 是真实值，f 是预测值。
    """

    def __init__(self, quantiles: Optional[Sequence[float]] = None):
        """
        初始化分位数损失函数。

        参数:
            quantiles (Optional[Sequence[float]]): 需要优化的分位数列表。
                例如 [0.1, 0.5, 0.9] 表示模型将预测 10%、50% 和 90% 的分位数。
        """
        super().__init__()
        if quantiles is None:
            raise ValueError("必须提供 quantiles 列表。")
        self.quantiles = list(quantiles)
        
        # 将分位数注册为 buffer，这样它会跟随模型移动到同一设备（CPU/GPU），
        # 但不会被视为模型的可选参数进行更新。
        self.register_buffer(
            "quantiles_tensor",
            torch.tensor(self.quantiles, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算预测值与真实值之间的分位数损失。

        参数:
            predictions (torch.Tensor): 模型输出的预测张量，形状通常为 [Batch, SeqLen, NQuantiles]。
            targets (torch.Tensor): 真实值张量，形状通常为 [Batch, SeqLen, 1]。

        返回:
            torch.Tensor: 所有样本和所有分位数的平均损失标量。
        """
        # 确保真实值的维度与预测值对齐（最后一维通常是特征或分位数维）
        if targets.dim() == 2:
            targets = targets.unsqueeze(-1)

        # 计算误差（真实值 - 预测值）
        errors = targets - predictions
        
        # 将分位数张量转换为与预测值相同的数据类型
        quantiles_tensor = self.quantiles_tensor.to(
            device=predictions.device,
            dtype=predictions.dtype,
        )
        
        # Pinball Loss 核心计算：
        # 对于误差 > 0 (真实值大于预测值)，损失贡献为 q * errors
        # 对于误差 < 0 (真实值小于预测值)，损失贡献为 (q - 1) * errors
        # 取两者最大值即实现了上述逻辑
        losses = torch.max(quantiles_tensor * errors, (quantiles_tensor - 1.0) * errors)
        
        return losses.mean()
