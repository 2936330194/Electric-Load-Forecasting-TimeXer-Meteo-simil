"""
tmp.py - TimeXer primary forecast + similar-day prior correction gate

Compared with test5_smp.py:
1. TimeXer predicts the absolute load directly.
2. The weighted similar-day prior is converted into a bounded correction direction:
   gap = prior_mean - timexer_pred
3. A sigmoid gate beta uses model/prior agreement and prior spread to decide
   how much prior correction to accept:
   y_hat = timexer_pred + beta * gap
"""

import argparse
import os
import random
import time
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from models.TimeXer import Model as TimeXer
import tmp_base as base
from utils.forecast_visualization import plot_pred_vs_true
from utils.metrics import metric
from utils.quantile import QuantileLoss
from utils.tools import adjust_learning_rate
from utils.weather_e2e import FullMapWeatherConvExtractor, WeatherGridStore, weather_data_provider


# ================= 相似日检索模块与 TimeXer 主体纠偏门控方案的专属参数配置 =================
# 指定相似日检索模型的缓存目录（包含离线训练好的 PCA 分解器和 Faiss 向量索引库）。
SIMILAR_DAY_ARTIFACT_DIR: Optional[str] = None

# 在检索历史中最相近日期的天气时，选取前 K(这里是3) 个最相似的日期来生成先验负荷曲线。
# 一般 K 选取 3~5 左右能起到较好的去噪及平滑效果。
SIMILAR_DAY_TOP_K = 3

# 这是一个总开关，决定接下来的测试或训练环节中，是否要启用并组装这条相似日先验特征进入端到端模型。
USE_SIMILAR_DAY_PRIOR = True

# 针对先验纠偏门控 (Dynamic Prior-Correction Gating) 的隐藏层尺寸参数。
# 控制 Sigmoid 门控网络的非线性宽度，越大则门控决策越复杂。
SIMILAR_DAY_GATE_HIDDEN_DIM = 32

# 网络初始化时给先验纠偏比例的权重锚点。
# 0.1 表示初始时仅采纳 10% 的相似日纠偏，让 TimeXer 先以主体预测稳定起步，
# 随后网络再通过反向传播自动调节先验的介入比例。
SIMILAR_DAY_GATE_INIT_BETA = 0.05


# ================= 从基础实验模块导入常用工具函数 =================
# 为了保持代码简洁并保证逻辑与基础版本严格对齐，这里大量借用了 test4_smp.py (下称 base) 中写好的底层支持函数。
_use_non_blocking_transfer = base._use_non_blocking_transfer  # 用于判断是否开启异步显存传输加速（non_blocking=True）
_to_float_device = base._to_float_device                      # 将 Float 数据安全地发送给设定的硬件（CPU 或 CUDA）
_to_long_device = base._to_long_device                        # 将 Long/Int 数据发送给特定计算硬件
extract_target = base.extract_target                          # 用来自动抽取出用于计算 Loss (只保留真值通道) 的数据片段
_parse_cli_args = base._parse_cli_args                        # 用于解析用户从终端传入的模型维度和训练参数等设置
_resolve_weather_h5_specs = base._resolve_weather_h5_specs    # 读取不同区域的气象变量对应说明书 (告诉网络通道有多少个气象因数)
_configure_runtime_weather_args = base._configure_runtime_weather_args # 动态初始化网络时，根据实际气象包尺寸微调部分配置
export_similar_day_baseline = base.export_similar_day_baseline# 一键在训练结束后将单纯依靠该先验所生成的对比结果图表保存


def _unpack_weather_batch(
    batch: Sequence[torch.Tensor],
) -> Tuple[
    torch.Tensor,           # batch_x (输入历史负荷的序列)
    torch.Tensor,           # batch_y (未来需要预测负荷目标的真实序列)
    torch.Tensor,           # batch_x_mark (输入历史负载在日历上的各个时间分量标签)
    torch.Tensor,           # batch_exo_mark (气象等外生变量的时间标签序列)
    torch.Tensor,           # batch_weather_frames (气象图：或者是批量裁剪出来的序列，或者是唯一的图像全集)
    torch.Tensor,           # batch_weather_index (用来在上面的唯一气象图像中按索引查找重现序列的一维索引表)
    Optional[torch.Tensor], # similar_day_prior (我们的重头戏：相似日序列张量 [Batch, pred_len, TopK+1])
]:
    """
    自适应长度的数据集解包流线 (通用 DataLoader batch 拆包工具)。
    用于将来自 PyTorch DataLoader 返回的数据迭代器拆分成模型需要的各部位张量。
    根据元组中 Tensor 的数量，可以无缝向下兼容无论是带有外生相似日先验的“新 Dataset”还是缺省先验的“旧 Dataset”。
    """
    # 如果传来的是 6 个元素，说明调用的是未开启使用相似日的常规 Weather End-to-End Dataset。
    if len(batch) == 6:
        batch_x, batch_y, batch_x_mark, batch_exo_mark, batch_weather_frames, batch_weather_index = batch
        return (
            batch_x,
            batch_y,
            batch_x_mark,
            batch_exo_mark,
            batch_weather_frames,
            batch_weather_index,
            None, # 由于不含先验信息，这第七个输出槽强制空置
        )
        
    # 如果传来的是 7 个元素，表明数据集中已经集成了检索、权重运算合并完的相似日先验特征。
    if len(batch) == 7:
        (
            batch_x,
            batch_y,
            batch_x_mark,
            batch_exo_mark,
            batch_weather_frames,
            batch_weather_index,
            similar_day_prior,
        ) = batch
        return (
            batch_x,
            batch_y,
            batch_x_mark,
            batch_exo_mark,
            batch_weather_frames,
            batch_weather_index,
            similar_day_prior, # 直接往外原封递出该预测区间的经验合成结果
        )
        
    # 如果数据结构发生未知的突变（例如底层 __getitem__ 操作被更改而不自知），报错避免灾难蔓延。
    raise ValueError(f"Unexpected batch size: expected 6 or 7 tensors, got {len(batch)} (未预料的数据集解包维数！)")


class FullMapConvTimeXerPriorCorrectionGateQuantile(nn.Module):
    """
    端到端架构类：包含“TimeXer 主预测 + 相似日先验纠偏门控”机制的概率预测网络。
    核心思想：
    1. TimeXer 直接预测绝对负荷值，始终作为主预测分支。
    2. 相似日先验不再充当硬基线，而是提供纠偏方向 gap = prior_mean - timexer_pred。
    3. 门控器会结合模型输出、先验差异、差异幅度、Top-K 离散度等证据，生成 β ∈ (0, 1)，
       决定要吸收多少先验修正：y = timexer_pred + β * gap。
    """
    def __init__(self, configs, quantiles: Sequence[float]):
        super().__init__()
        # ------- 1. 基础参数配置与初始化 -------
        self.quantiles = list(quantiles)                                            # 概率预测的分位数列表（如 [0.1, 0.5, 0.9]）
        self.n_quantiles = len(self.quantiles)                                      # 需要预测的分位数数量
        self.weather_feature_dim = int(configs.weather_feature_dim)                 # 气象特征经过 CNN 降维后的向量维度大小
        self.encode_chunk_size = int(getattr(configs, "weather_encode_chunk_size", 512)) # 气象特征编码的切块大小，用于防止显存溢出 (OOM)
        
        # ------- 2. 相似日先验相关配置 -------
        self.use_similar_day_prior = bool(getattr(configs, "use_similar_day_prior", False)) # 是否启用相似日先验纠偏逻辑
        self.similar_day_top_k = int(getattr(configs, "similar_day_top_k", 3))              # 提取的历史相似日数量 (Top-K)
        # 相似日先验的特征维度大小：TopK 条相似日曲线 + 1 条通过加权平均得到的综合基准曲线
        self.similar_day_prior_dim = self.similar_day_top_k + 1 if self.use_similar_day_prior else 0

        # ------- 3. 气象特征提取主干网络 (CNN backbone) -------
        self.weather_backbone = FullMapWeatherConvExtractor(
            in_channels=int(getattr(configs, "weather_in_channels")),               # 输入气象数据的通道数
            out_channels=self.weather_feature_dim,                                  # CNN 提取特征后的输出通道数
            kernel_height=int(getattr(configs, "weather_kernel_height")),           # 气象网格的高度
            kernel_width=int(getattr(configs, "weather_kernel_width")),             # 气象网格的宽度
            dropout=float(getattr(configs, "dropout", 0.1)),                        # CNN 模块的 Dropout 比例
        )

        # ------- 4. 核心时序预测模型 (TimeXer) 的配置 -------
        self.weather_seq_len = int(getattr(configs, "weather_seq_len", configs.seq_len)) # 气象序列的长度
        configs.exo_seq_len = self.weather_seq_len                                  # 设定 TimeXer 接收的外生变量序列长度
        configs.enc_in = 1                                                          # 设定 TimeXer 编码器的输入维度（单变量负荷）
        self.timexer = TimeXer(configs)                                             # 实例化底层 TimeXer 预测模型

        # ------- 5. 构建增强的相似日先验纠偏门控单元 (Gating Mechanism) -------
        if self.use_similar_day_prior:
            # 门控网络的隐藏层维度：如果未在 configs 中配置，则在 16 和 d_model/4 之间取一个合理的最大值
            gate_hidden_dim = int(
                getattr(
                    configs,
                    "similar_day_gate_hidden_dim",
                    max(16, int(getattr(configs, "d_model", 128)) // 4),
                )
            )
            
            # 【重要技巧：门控偏差初始化 (Bias Initialization)】
            # 目的：通过反解 Sigmoid 函数来设置初始偏置项，使得网络在初始化时倾向于产生较低的先验采纳比例 β。
            # 默认初始 beta (gate_init_beta) 设为 0.1，意味着在模型训练初期，模型主要依赖 TimeXer 分支的主干预测能力。
            # 先验信息只被非常轻微（温和）地引入进行修正，以防初期先验特征带来的抖动引起模型崩溃。
            gate_init_beta = float(getattr(configs, "similar_day_gate_init_beta", 0.05))
            gate_init_beta = min(max(gate_init_beta, 1e-3), 1.0 - 1e-3)             # 截断以避免 log(0) 问题
            gate_bias = float(np.log(gate_init_beta / (1.0 - gate_init_beta)))      # 反解 Sigmoid：bias = ln(beta / (1 - beta))

            # 构建门控多层感知机 (MLP)
            # 门控输入的设计 (共 5 + TopK + 1 维):
            #   1维: timexer_pred     (TimeXer 的初步预测值)
            #   1维: prior_mean       (相似日先验曲线的平均值)
            #   1维: gap              (先验差异：prior_mean - timexer_pred)
            #   1维: abs_gap          (差异幅度：|gap|)
            #   1维: prior_spread     (先验离散度：Top-K 相似日之间的标准差)
            #   TopK+1维: similar_day_prior (TopK 序列和综合基准序列的值)
            self.similar_day_gate = nn.Sequential(
                nn.Linear(3, gate_hidden_dim),
                nn.GELU(),
                nn.Dropout(float(getattr(configs, "dropout", 0.1))),
                nn.Linear(gate_hidden_dim, 1),
                nn.Sigmoid(), # 最后一层使用 Sigmoid，将纠偏权重强行压缩约束到 (0, 1) 的比例区间
            )
            
            # 将输出层（即 nn.Linear(gate_hidden_dim, 1)）初始化为常数门控结构：
            # 1. 权重置为 0；
            # 2. 偏置置为计算出的 gate_bias；
            # 这样网络在刚开始训练时，前向推导出的 gate β 值严格等于 gate_init_beta (0.1)，后续随着梯度更新再逐步学习各个特征的权重。
            with torch.no_grad():
                nn.init.zeros_(self.similar_day_gate[-2].weight)
                nn.init.constant_(self.similar_day_gate[-2].bias, gate_bias)
        else:
            self.similar_day_gate = None                                            # 不启用相似日先验纠偏逻辑

        # ------- 6. 联合分位数回归预测头 (Quantile Regression Head) -------
        # 负责将校正后的 1 维单点负荷预测扩展映射为多维的不同置信区间（分位数）的负荷预测
        self.quantile_head = nn.Linear(1, self.n_quantiles)
        
        # 分位数头初始化的先验约束：
        # 将权重全置为 1.0：表明各分位数上的初始估计值都等同于单点绝对纠偏值 (y = x)
        # 将偏置设为按分位数大小比例递增：如 q=0.1 偏置为负，q=0.9 偏置为正，从一开始就构造好上下分位数的自然间隔结构。
        with torch.no_grad():
            self.quantile_head.weight.fill_(1.0)
            self.quantile_head.bias.copy_(torch.tensor([q - 0.5 for q in self.quantiles]) * 0.1)

    def _encode_weather_frames(self, weather_frames: torch.Tensor) -> torch.Tensor:
        if weather_frames.ndim != 4:
            raise ValueError(
                f"Weather frames should have shape [N, C, H, W], got {tuple(weather_frames.shape)}"
            )

        encoded_chunks: List[torch.Tensor] = []
        for start in range(0, weather_frames.shape[0], self.encode_chunk_size):
            end = min(start + self.encode_chunk_size, weather_frames.shape[0])
            encoded_chunks.append(self.weather_backbone(weather_frames[start:end].float()))
        return torch.cat(encoded_chunks, dim=0)

    def _encode_weather_sequence(
        self,
        weather_seq: Optional[torch.Tensor],
        weather_index: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if weather_seq is None:
            return None

        if weather_index is not None:
            if weather_seq.ndim != 4 or weather_index.ndim != 2:
                raise ValueError(
                    "Indexed weather mode expects weather_seq [U, C, H, W] and weather_index [B, T]."
                )
            batch_size, time_len = weather_index.shape
            encoded_frames = self._encode_weather_frames(weather_seq)
            gathered = encoded_frames.index_select(0, weather_index.reshape(-1))
            return gathered.reshape(batch_size, time_len, self.weather_feature_dim)

        if weather_seq.ndim != 5:
            raise ValueError(
                f"Sequential weather mode expects [B, T, C, H, W], got {tuple(weather_seq.shape)}"
            )
        batch_size, time_len, channels, height, width = weather_seq.shape
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
        similar_day_prior: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        端到端先验纠偏门控的前向网络路由 (Forward Pass)
        
        参数:
            load_x (Tensor): 过去一段时间的历史负荷序列，形状为 [Batch, seq_len, 1]
            x_mark_enc (Tensor): 负荷序列对应的时间特征编码（如 星期几、小时 等），形状为 [Batch, seq_len, mark_dim]
            x_exo_mark (Tensor): 外生气象变量的时间特征编码，形状为 [Batch, exo_seq_len, mark_dim]
            weather_x (Tensor): 全天候多要素气象网格数据，形状为 [Batch, exo_seq_len, C, H, W]（或预计算过的气象索引池）
            weather_x_index (Optional[Tensor]): 当采用预计算的大规模气象池时所对应的气象文件索引位置
            similar_day_prior (Optional[Tensor]): 外挂传入的历史相似日负荷先验曲线，包含均值线及 TopK 原片，形状为 [Batch, pred_len, similar_day_prior_dim]
            mask (Optional[Tensor]): 时序填充掩码遮罩
        
        返回:
            torch.Tensor: 支持多置信区间的最终预测结果，形状为 [Batch, pred_len, n_quantiles]
        """
        # --- 1. 气象表征降维提取：正常提纯出长周期的气象 Token ---
        weather_feature = self._encode_weather_sequence(weather_x, weather_x_index)

        # --- 2. TimeXer 主分支推导：直接预测未来目标时段绝对负荷值 ---
        timexer_pred = self.timexer(
            load_x,                     # 截取过去的纯历史负荷标量序列
            x_mark_enc,                 # 时间标记（大背景锚点：如周期节气、节假日、星期等）
            None,
            None,
            mask=mask,
            x_exo=weather_feature,      # 灌入通过 CNN 或池化后解析出的高阶外生气象环境特征序列 [Batch, exo_seq_len, weather_feature_dim]
            x_exo_mark=x_exo_mark,      # 气象序列的协变量编码
        )
        
        # 将 TimeXer 底层出来的时序拼接到需要预测的 seq 长度 (仅截取未来预测时段 pred_len 的输出部分)
        timexer_pred = timexer_pred[:, -self.timexer.pred_len :, :]

        # 托底降级选项：如果关闭或者未传入相似日先验修正逻辑，则最终预测输出直接依赖纯净的 TimeXer 本地预测
        point_pred = timexer_pred 
        
        # --- 3. 发动核心架构：相似日先验柔性纠偏门控 (Residual Gating Fusion) ---
        if self.use_similar_day_prior and similar_day_prior is not None:
            # ======= 数据安全性与维度校验 =======
            if similar_day_prior.ndim != 3:
                raise ValueError(
                    f"传入的最邻近相似日先验数据维度错误：应为 [Batch, pred_len, 维度({self.similar_day_prior_dim})], "
                    f"但实际得到 {tuple(similar_day_prior.shape)}"
                )
            if similar_day_prior.shape[1] != self.timexer.pred_len:
                raise ValueError(
                    f"相似日先验的时间步长与预测步长不吻合: "
                    f"先验维度 {similar_day_prior.shape[1]} vs 模型预测域长 {self.timexer.pred_len}"
                )
            if similar_day_prior.shape[2] != self.similar_day_prior_dim:
                raise ValueError(
                    "相似日先验的特征通道数与初始化配置不同步: "
                    f"当前传递的通道数 {similar_day_prior.shape[2]} vs 配置设定的通量为 {self.similar_day_prior_dim}"
                )
                
            # ======= 提纯出供 Gate 推理的多维度参考物理量 =======
            similar_day_prior = similar_day_prior.float()
            # 索引0位置：是根据某种距离策略算出的相似日加权融合后作为锚点的先验均值曲线 (综合基准基线)
            prior_mean = similar_day_prior[:, :, :1]
            
            # 索引1及以后：未经处理的纯正 Top-K 历史真值负荷切片（供计算内部扰动和离散边界）
            topk_curves = similar_day_prior[:, :, 1:]
            
            if topk_curves.shape[-1] > 0:
                # 衡量先验信息的置信度：TopK 序列自身的无偏标准差。
                # 标准差越大，说明挑选出来的相似日彼此走势分歧强烈，先验的可信度就相对存疑。
                prior_spread = torch.std(topk_curves, dim=-1, keepdim=True, unbiased=False)
            else:
                prior_spread = torch.zeros_like(prior_mean)

            # --- 4. 差距度量表与动态β加权融合 ---
            # 门控网络核心物理学视角下的 "纠偏方向向量 (gap)"，目标是从 timexer_pred 出发指向 prior_mean 的修正幅度向量
            gap = prior_mean - timexer_pred
            
            # 聚合所有的判定线索
            gate_input = torch.cat(
                [
                    timexer_pred,       # [B, L, 1] 主分支当前预估体量（负荷基数大小）
                    prior_mean,         # [B, L, 1] 相似日先验综合均值基线
                    gap,                # [B, L, 1] 原始差距（具有正负符号，体现高估或低估）
                    gap.abs(),          # [B, L, 1] 绝对规模位移差（要校正的程度多猛烈）
                    prior_spread,       # [B, L, 1] Top-K 自身的不一致性（先验信噪系数指标）
                    similar_day_prior,  # [B, L, TopK+1] 完整的全幅先验环境分布特征
                ],
                dim=-1,
            )
            
            # 使用门控多层感知机及末端 Sigmoid 推理出动态采纳比例 beta ∈ (0, 1)
            # β -> 0: 说明模型认定没必要大调，充分采信 TimeXer 主支结果，先验权当参考。
            # β -> 1: 说明模型认定需要大幅度吸纳外挂库相似日经验，做强制的拉平纠正预估（多见于气候巨变或非标事件如节假日异常断层）。
            gate_input = torch.cat(
                [
                    prior_mean,
                    gap.detach(),
                    prior_spread,
                ],
                dim=-1,
            )
            beta = self.similar_day_gate(gate_input)
            
            # 执行校正合成公式：最终输出 = 主线原始预测 + 修正采纳比例 * 去往先验的差距量
            point_pred = timexer_pred + beta * gap

        # --- 5. 置信区间多阶段散射 ---
        # 抛给量化回归头：根据加权校正后的中心点预估分布，分化成诸如 10%, 50%, 90% 各级不同保护圈大小的区间分布负荷
        return self.quantile_head(point_pred)

def validate_quantile(model, data_loader, criterion, args, device, use_amp: bool = False) -> float:
    model.eval()
    total_loss = []
    use_non_blocking = _use_non_blocking_transfer(args, device)

    with torch.inference_mode():
        for batch in data_loader:
            (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_exo_mark,
                batch_weather_frames,
                batch_weather_index,
                similar_day_prior,
            ) = _unpack_weather_batch(batch)

            batch_x = _to_float_device(batch_x, device, non_blocking=use_non_blocking)
            batch_y = _to_float_device(batch_y, device, non_blocking=use_non_blocking)
            batch_x_mark = _to_float_device(batch_x_mark, device, non_blocking=use_non_blocking)
            batch_exo_mark = _to_float_device(batch_exo_mark, device, non_blocking=use_non_blocking)
            batch_weather_frames = _to_float_device(batch_weather_frames, device, non_blocking=use_non_blocking)
            batch_weather_index = _to_long_device(batch_weather_index, device, non_blocking=use_non_blocking)
            if similar_day_prior is not None:
                similar_day_prior = _to_float_device(similar_day_prior, device, non_blocking=use_non_blocking)

            with torch.amp.autocast("cuda", enabled=use_amp):
                model_kwargs = {
                    "load_x": batch_x,
                    "x_mark_enc": batch_x_mark,
                    "x_exo_mark": batch_exo_mark,
                    "weather_x": batch_weather_frames,
                    "weather_x_index": batch_weather_index,
                }
                if similar_day_prior is not None:
                    model_kwargs["similar_day_prior"] = similar_day_prior
                outputs = model(**model_kwargs)
                batch_y_target = extract_target(batch_y[:, -args.pred_len :, :])
                loss = criterion(outputs, batch_y_target)
            total_loss.append(loss.item())

    model.train()
    return float(np.average(total_loss)) if total_loss else np.nan


def train_quantile_model(model, args, device, weather_store: WeatherGridStore):
    _, train_loader = weather_data_provider(args, "train", weather_store)
    _, vali_loader = weather_data_provider(args, "test", weather_store)

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.abspath(args.model_path)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = QuantileLoss(args.quantiles).to(device)

    use_amp = bool(getattr(args, "use_amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    use_non_blocking = _use_non_blocking_transfer(args, device)
    best_vali_loss = float("inf")
    best_epoch = 0
    best_state_dict = None
    patience_counter = 0

    print("\n" + "=" * 72)
    print("Start training TimeXer-primary + similar-day prior-correction quantile model")
    print("validation split: test")
    print(f"best model path: {model_path}")
    print(f"early stopping patience: {args.patience}")
    print(f"quantiles: {args.quantiles}")
    print(f"weather_feature_dim: {args.weather_feature_dim}")
    print(f"weather_kernel_size: ({args.weather_kernel_height}, {args.weather_kernel_width})")
    print(
        f"weather_seq_len: {args.weather_seq_len} "
        f"(history={args.weather_history_len}, future={args.weather_seq_len - args.weather_history_len}, "
        f"step={getattr(args, 'weather_step_freq', 'native')})"
    )
    print(f"use_similar_day_prior: {bool(getattr(args, 'use_similar_day_prior', False))}")
    if bool(getattr(args, "use_similar_day_prior", False)):
        print(
            "similar_day_gate_config: "
            f"top_k={getattr(args, 'similar_day_top_k', 0)}, "
            f"gate_hidden_dim={getattr(args, 'similar_day_gate_hidden_dim', 0)}, "
            f"gate_init_beta={float(getattr(args, 'similar_day_gate_init_beta', 0.0)):.3f}, "
            f"artifact_dir={getattr(args, 'similar_day_artifact_dir', None)}"
        )
    print(f"batch_size: {args.batch_size}")
    print(f"use_amp: {use_amp}")
    if bool(getattr(args, "contiguous_train_batches", False)):
        dense_weather_frames = args.batch_size * args.weather_seq_len
        print(f"overlap-aware weather batching: on (dense {dense_weather_frames} exogenous frames/batch)")
    print("=" * 72)

    for epoch in range(args.train_epochs):
        model.train()
        train_loss = []
        epoch_time = time.time()

        for i, batch in enumerate(train_loader):
            (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_exo_mark,
                batch_weather_frames,
                batch_weather_index,
                similar_day_prior,
            ) = _unpack_weather_batch(batch)

            optimizer.zero_grad(set_to_none=True)
            batch_x = _to_float_device(batch_x, device, non_blocking=use_non_blocking)
            batch_y = _to_float_device(batch_y, device, non_blocking=use_non_blocking)
            batch_x_mark = _to_float_device(batch_x_mark, device, non_blocking=use_non_blocking)
            batch_exo_mark = _to_float_device(batch_exo_mark, device, non_blocking=use_non_blocking)
            batch_weather_frames = _to_float_device(batch_weather_frames, device, non_blocking=use_non_blocking)
            batch_weather_index = _to_long_device(batch_weather_index, device, non_blocking=use_non_blocking)
            if similar_day_prior is not None:
                similar_day_prior = _to_float_device(similar_day_prior, device, non_blocking=use_non_blocking)

            with torch.amp.autocast("cuda", enabled=use_amp):
                model_kwargs = {
                    "load_x": batch_x,
                    "x_mark_enc": batch_x_mark,
                    "x_exo_mark": batch_exo_mark,
                    "weather_x": batch_weather_frames,
                    "weather_x_index": batch_weather_index,
                }
                if similar_day_prior is not None:
                    model_kwargs["similar_day_prior"] = similar_day_prior
                outputs = model(**model_kwargs)
                batch_y_target = extract_target(batch_y[:, -args.pred_len :, :])
                loss = criterion(outputs, batch_y_target)

            train_loss.append(loss.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (i + 1) % 50 == 0:
                print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")

        vali_loss = validate_quantile(model, vali_loader, criterion, args, device, use_amp=use_amp)
        train_loss_avg = float(np.average(train_loss)) if train_loss else np.nan
        print(
            f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.1f}s | "
            f"Train: {train_loss_avg:.7f} Vali(Test): {vali_loss:.7f}"
        )

        if np.isfinite(vali_loss) and vali_loss < best_vali_loss:
            best_vali_loss = float(vali_loss)
            best_epoch = epoch + 1
            patience_counter = 0
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            torch.save(best_state_dict, model_path)
            print(
                f"Save best test-validated model at epoch {best_epoch}: "
                f"loss={best_vali_loss:.7f}"
            )
        else:
            patience_counter += 1
            print(
                f"No improvement. Early-stop counter: "
                f"{patience_counter}/{args.patience}"
            )
            if patience_counter >= args.patience:
                print("Early stopping")
                break
        adjust_learning_rate(optimizer, epoch + 1, args)

    if best_state_dict is None:
        best_state_dict = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        torch.save(best_state_dict, model_path)
        print("Validation loss is not finite in all epochs, saved last model weights.")

    model.load_state_dict(best_state_dict)
    print(
        f"Loaded best model weights from epoch {best_epoch if best_epoch > 0 else args.train_epochs}: "
        f"{model_path}"
    )
    return model


def test_quantile_model(model, args, device, weather_store: WeatherGridStore) -> str:
    test_data, test_loader = weather_data_provider(args, "test", weather_store)

    folder_path = args.output_dir
    os.makedirs(folder_path, exist_ok=True)

    preds_p50 = []
    trues = []
    quantile_preds_all = []

    use_amp = bool(getattr(args, "use_amp", False)) and device.type == "cuda"
    use_non_blocking = _use_non_blocking_transfer(args, device)

    model.eval()
    with torch.inference_mode():
        for batch in test_loader:
            (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_exo_mark,
                batch_weather_frames,
                batch_weather_index,
                similar_day_prior,
            ) = _unpack_weather_batch(batch)

            batch_x = _to_float_device(batch_x, device, non_blocking=use_non_blocking)
            batch_y = _to_float_device(batch_y, device, non_blocking=use_non_blocking)
            batch_x_mark = _to_float_device(batch_x_mark, device, non_blocking=use_non_blocking)
            batch_exo_mark = _to_float_device(batch_exo_mark, device, non_blocking=use_non_blocking)
            batch_weather_frames = _to_float_device(batch_weather_frames, device, non_blocking=use_non_blocking)
            batch_weather_index = _to_long_device(batch_weather_index, device, non_blocking=use_non_blocking)
            if similar_day_prior is not None:
                similar_day_prior = _to_float_device(similar_day_prior, device, non_blocking=use_non_blocking)

            with torch.amp.autocast("cuda", enabled=use_amp):
                model_kwargs = {
                    "load_x": batch_x,
                    "x_mark_enc": batch_x_mark,
                    "x_exo_mark": batch_exo_mark,
                    "weather_x": batch_weather_frames,
                    "weather_x_index": batch_weather_index,
                }
                if similar_day_prior is not None:
                    model_kwargs["similar_day_prior"] = similar_day_prior
                outputs = model(**model_kwargs)

            batch_y_target = extract_target(batch_y[:, -args.pred_len :, :])
            p50_pred = outputs.float()[:, :, base.P50_IDX : base.P50_IDX + 1]

            quantile_preds_all.append(outputs.float().detach().cpu().numpy())
            preds_p50.append(p50_pred.detach().cpu().numpy())
            trues.append(batch_y_target.detach().cpu().numpy())

    preds_p50 = np.concatenate(preds_p50, axis=0)
    trues = np.concatenate(trues, axis=0)
    quantile_preds_all = np.concatenate(quantile_preds_all, axis=0)

    print(
        f"Test shape: preds={preds_p50.shape}, "
        f"trues={trues.shape}, quantiles={quantile_preds_all.shape}"
    )

    np.save(os.path.join(folder_path, "pred.npy"), preds_p50)
    np.save(os.path.join(folder_path, "true.npy"), trues)
    np.save(os.path.join(folder_path, "quantile_preds.npy"), quantile_preds_all)

    if test_data.scale:
        shape = trues.shape
        preds_inv = test_data.inverse_transform_target(preds_p50.reshape(shape[0] * shape[1], -1)).reshape(shape)
        trues_inv = test_data.inverse_transform_target(trues.reshape(shape[0] * shape[1], -1)).reshape(shape)

        q_shape = quantile_preds_all.shape
        quantile_inv = np.zeros_like(quantile_preds_all)
        for qi in range(base.N_QUANTILES):
            q_slice = quantile_preds_all[:, :, qi : qi + 1]
            q_inv = test_data.inverse_transform_target(
                q_slice.reshape(q_shape[0] * q_shape[1], -1)
            ).reshape(q_shape[0], q_shape[1], 1)
            quantile_inv[:, :, qi] = q_inv[:, :, 0]

        np.save(os.path.join(folder_path, "pred_inv.npy"), preds_inv)
        np.save(os.path.join(folder_path, "true_inv.npy"), trues_inv)
        np.save(os.path.join(folder_path, "quantile_preds_inv.npy"), quantile_inv)

    if test_data.scale and getattr(args, "inverse_eval", False):
        mae, mse, rmse, mape, mspe = metric(preds_inv, trues_inv)
        metrics_text = (
            f"P50 Test Metrics (Inverse): "
            f"MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}, "
            f"MAPE={mape:.6f}, MSPE={mspe:.6f}"
        )
    else:
        mae, mse, rmse, mape, mspe = metric(preds_p50, trues)
        metrics_text = (
            f"P50 Test Metrics (Normalized): "
            f"MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}, "
            f"MAPE={mape:.6f}, MSPE={mspe:.6f}"
        )

    print(metrics_text)
    with open(os.path.join(folder_path, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(metrics_text + "\n")

    return folder_path


def main() -> None:
    fix_seed = 2026
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    cli_args = _parse_cli_args()
    selected_weather_source = cli_args.weather_source
    selected_weather_h5_specs = _resolve_weather_h5_specs(selected_weather_source)

    args = argparse.Namespace(
        task_name=base.TASK_NAME,
        is_training=1 if base.TRAIN_MODE else 0,
        model_id=f"{base.MODEL_ID_PREFIX}_sdv2",
        model=base.MODEL,
        des=base.DES,
        itr=base.ITR,
        data="custom",
        root_path=base.ROOT_PATH,
        data_path=base.DATA_PATH,
        features=base.FEATURES,
        target=base.TARGET,
        target_channel_idx=0,
        freq=base.LOAD_FREQ,
        embed="timeF",
        output_dir="./tmp",
        model_path="./tmp/checkpoint.pth",
        seq_len=base.SEQ_LEN,
        label_len=base.LABEL_LEN,
        pred_len=base.PRED_LEN,
        enc_in=base.ENC_IN,
        c_out=base.C_OUT,
        d_model=base.D_MODEL,
        n_heads=base.N_HEADS,
        e_layers=base.E_LAYERS,
        d_ff=base.D_FF,
        factor=base.FACTOR,
        dropout=base.DROPOUT,
        activation=base.ACTIVATION,
        patch_len=base.PATCH_LEN,
        use_norm=base.USE_NORM,
        weather_source=selected_weather_source,
        weather_h5_specs=selected_weather_h5_specs,
        weather_in_channels=base.WEATHER_IN_CHANNELS,
        weather_feature_dim=base.WEATHER_FEATURE_DIM,
        weather_grid_height=base.WEATHER_GRID_HEIGHT,
        weather_grid_width=base.WEATHER_GRID_WIDTH,
        weather_kernel_height=base.WEATHER_KERNEL_HEIGHT,
        weather_kernel_width=base.WEATHER_KERNEL_WIDTH,
        weather_encode_chunk_size=base.WEATHER_ENCODE_CHUNK_SIZE,
        use_weather_normalization=True,
        num_workers=base.NUM_WORKERS,
        pin_memory=base.PIN_MEMORY,
        contiguous_train_batches=base.CONTIGUOUS_TRAIN_BATCHES,
        train_epochs=base.TRAIN_EPOCHS,
        batch_size=base.BATCH_SIZE,
        patience=base.PATIENCE,
        learning_rate=base.LEARNING_RATE,
        loss="Quantile",
        lradj="cosine",
        use_amp=True,
        inverse_eval=base.INVERSE_EVAL,
        use_gpu=base.USE_GPU,
        gpu=base.GPU,
        use_multi_gpu=False,
        devices="0,1,2,3",
        quantiles=base.QUANTILES,
        n_quantiles=base.N_QUANTILES,
        use_similar_day_prior=USE_SIMILAR_DAY_PRIOR,
        similar_day_top_k=SIMILAR_DAY_TOP_K,
        similar_day_artifact_dir=SIMILAR_DAY_ARTIFACT_DIR,
        similar_day_gate_hidden_dim=SIMILAR_DAY_GATE_HIDDEN_DIM,
        similar_day_gate_init_beta=SIMILAR_DAY_GATE_INIT_BETA,
    )

    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU: cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    weather_store = WeatherGridStore(
        args.weather_h5_specs,
        expected_in_channels=args.weather_in_channels,
        fill_value=base.WEATHER_FILL_VALUE,
        use_channel_normalization=True,
    )
    try:
        args = _configure_runtime_weather_args(args, weather_store, selected_weather_source)

        if weather_store.frame_shape is None:
            raise RuntimeError("weather_store.frame_shape is not initialized.")
        _, frame_height, frame_width = weather_store.frame_shape
        if (frame_height, frame_width) != (args.weather_kernel_height, args.weather_kernel_width):
            raise ValueError(
                "Weather frame size does not match full-map kernel size: "
                f"frame=({frame_height}, {frame_width}), "
                f"kernel=({args.weather_kernel_height}, {args.weather_kernel_width})"
            )

        model = FullMapConvTimeXerPriorCorrectionGateQuantile(args, quantiles=base.QUANTILES).float().to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"TimeXer-primary + prior-correction total params: {total_params:,}")
        print(f"TimeXer-primary + prior-correction trainable params: {trainable_params:,}")
        os.makedirs(args.output_dir, exist_ok=True)
        if base.TRAIN_MODE:
            print("\n>>> Start training with test-set validation")
            model = train_quantile_model(model, args, device, weather_store)

            print("\n>>> Start testing on test split")
            results_dir = test_quantile_model(model, args, device, weather_store)
        else:
            ckpt_path = os.path.abspath(args.model_path)
            if os.path.exists(ckpt_path):
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                print(f"Loaded model: {ckpt_path}")
            else:
                raise FileNotFoundError(
                    f"Model file not found: {ckpt_path}. Please set TRAIN_MODE = True first."
                )

            print("\n>>> Test only on test split")
            results_dir = test_quantile_model(model, args, device, weather_store)

        plot_pred_vs_true(
            results_dir,
            use_inverse=base.INVERSE_EVAL,
            quantiles=args.quantiles,
            title_prefix="TimeXer-Primary + Similar-Day Prior-Correction Test Prediction",
            y_label="Load (MW)",
        )
    finally:
        weather_store.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
