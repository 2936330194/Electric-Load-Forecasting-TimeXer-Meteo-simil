"""
DLinear.py - DLinear 时间序列预测模型

本模块实现了 DLinear（Decomposition-Linear）模型，这是一个简单但高效的
时间序列预测模型，在多个基准数据集上达到了与 Transformer 模型相当甚至更好的性能。

核心思想：
    1. 序列分解：将输入序列分解为趋势项（Trend）和季节项（Seasonal）
    2. 线性映射：分别对趋势和季节成分进行线性变换
    3. 结果合成：将两个成分的预测结果相加得到最终预测

模型架构：
    输入序列 X [B, L, C]
         ↓
    序列分解 (series_decomp)
         ↓
    ┌─────────────────┬─────────────────┐
    │   季节项 X_s    │    趋势项 X_t   │
    │  [B, L, C]      │   [B, L, C]     │
    └────────┬────────┴────────┬────────┘
             ↓                 ↓
    ┌────────────────┐ ┌────────────────┐
    │ Linear_Seasonal│ │  Linear_Trend  │
    │   L → P        │ │    L → P       │
    └────────┬───────┘ └────────┬───────┘
             ↓                 ↓
    ┌────────────────┐ ┌────────────────┐
    │  Y_s [B,P,C]   │ │  Y_t [B,P,C]   │
    └────────┬───────┘ └────────┬───────┘
             └────────┬────────┘
                      ↓
              Y = Y_s + Y_t
               [B, P, C]

参数说明：
    B: batch_size (批次大小)
    L: seq_len (输入序列长度)
    P: pred_len (预测序列长度)
    C: channels (特征/变量数量)

模式说明：
    individual=True:  每个变量使用独立的线性层（Channel-Independent）
    individual=False: 所有变量共享同一个线性层（Channel-Shared）

支持的任务：
    - long_term_forecast: 长期预测
    - short_term_forecast: 短期预测
    - imputation: 缺失值填补
    - anomaly_detection: 异常检测
    - classification: 时序分类

参考论文：
    "Are Transformers Effective for Time Series Forecasting?"
    https://arxiv.org/abs/2205.13504
"""

import torch            # PyTorch 深度学习框架
import torch.nn as nn   # 神经网络模块
from layers.Autoformer_EncDec import series_decomp  # 序列分解模块


class Model(nn.Module):
    """
    DLinear 模型类
    
    一个基于分解的线性时间序列预测模型。
    通过将输入分解为趋势和季节成分，分别进行线性映射后合成预测结果。
    
    主要优势：
        1. 简单高效：只使用线性层，计算复杂度低
        2. 可解释性强：分解后的趋势和季节成分具有明确的物理意义
        3. 性能优异：在多个基准数据集上超越复杂的 Transformer 模型
    
    属性:
        task_name (str): 任务类型
        seq_len (int): 输入序列长度
        pred_len (int): 预测序列长度
        decomposition (series_decomp): 序列分解模块
        individual (bool): 是否为每个变量使用独立的线性层
        channels (int): 输入特征/变量数量
        Linear_Seasonal: 季节成分的线性映射层
        Linear_Trend: 趋势成分的线性映射层
    """

    def __init__(self, configs):
        """
        初始化 DLinear 模型
        
        参数:
            configs: 配置对象，需要包含以下属性：
                - task_name (str): 任务类型
                - seq_len (int): 输入序列长度
                - pred_len (int): 预测长度（预测任务）
                - moving_avg (int): 移动平均窗口大小，用于序列分解
                - enc_in (int): 输入特征数量
                - individual (bool, optional): 是否使用独立线性层，默认 False
                - num_class (int, optional): 分类任务的类别数
        """
        super().__init__()
        
        # 保存任务类型和序列长度
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        
        # 根据任务类型设置预测长度
        # 对于分类、异常检测、填补任务，输出长度等于输入长度
        if self.task_name in ("classification", "anomaly_detection", "imputation"):
            self.pred_len = configs.seq_len
        else:
            # 预测任务使用配置的预测长度
            self.pred_len = configs.pred_len

        # ==================== 序列分解模块 ====================
        # 使用移动平均将序列分解为趋势项和季节项
        self.decomposition = series_decomp(configs.moving_avg)
        
        # 是否为每个变量（通道）使用独立的线性层
        # True: Channel-Independent 模式，参数更多但可能更灵活
        # False: Channel-Shared 模式，参数共享更高效
        self.individual = getattr(configs, "individual", False)
        
        # 输入特征/变量数量
        self.channels = configs.enc_in
        self.target_channel_idx = int(getattr(configs, "target_channel_idx", self.channels - 1))
        if self.channels > 0:
            self.target_channel_idx = max(0, min(self.target_channel_idx, self.channels - 1))
        else:
            self.target_channel_idx = 0

        # 条件式未来协变量分支参数
        self.use_future_covariates = bool(getattr(configs, "use_future_covariates", False))
        default_cov_dim = self.channels - 1 if getattr(configs, "features", "MS") == "MS" else 0
        self.future_cov_dim = int(getattr(configs, "future_cov_dim", default_cov_dim))
        max_cov_dim = max(0, self.channels - 1)
        self.future_cov_dim = max(0, min(self.future_cov_dim, max_cov_dim))
        self.future_cov_head = None
        self.delta_scale = None

        # ==================== 构建线性层 ====================
        if self.individual:
            # Channel-Independent 模式：每个变量有独立的线性层
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for _ in range(self.channels):
                # 为每个通道创建独立的线性层
                # 输入维度: seq_len, 输出维度: pred_len
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
                
                # 权重初始化：使用均匀分布初始化
                # 初始化为 1/seq_len 相当于初始时做简单平均
                # 这种初始化有助于模型稳定训练
                self.Linear_Seasonal[-1].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
                )
                self.Linear_Trend[-1].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
                )
        else:
            # Channel-Shared 模式：所有变量共享同一个线性层
            # 这种方式参数量更少，适合变量之间有相似模式的场景
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            
            # 权重初始化
            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
            )
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
            )

        # ==================== 分类任务的投影层 ====================
        if self.task_name == "classification":
            # 将编码器输出展平后映射到类别数
            self.projection = nn.Linear(configs.enc_in * configs.seq_len, configs.num_class)

        # ==================== 未来协变量校正分支 ====================
        if (
            self.task_name in ("long_term_forecast", "short_term_forecast")
            and self.use_future_covariates
            and self.future_cov_dim > 0
        ):
            hidden_dim = max(16, min(128, self.future_cov_dim * 8))
            self.future_cov_head = nn.Sequential(
                nn.Linear(self.future_cov_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(getattr(configs, "future_cov_dropout", 0.1)),
                nn.Linear(hidden_dim, 1),
            )
            self.delta_scale = nn.Parameter(torch.tensor(1.0))

    def encoder(self, x):
        """
        编码器：DLinear 的核心处理逻辑
        
        处理流程：
        1. 序列分解：将输入分解为季节项和趋势项
        2. 维度转换：调整维度以适配线性层
        3. 线性映射：分别对两个成分进行时间维度的线性变换
        4. 结果合成：将两个成分的输出相加
        
        参数:
            x (Tensor): 输入序列 [batch_size, seq_len, channels]
        
        返回:
            Tensor: 预测结果 [batch_size, pred_len, channels]
        """
        # ==================== 序列分解 ====================
        # 将输入分解为季节项（高频波动）和趋势项（低频趋势）
        seasonal_init, trend_init = self.decomposition(x)
        # 形状: [B, seq_len, C] -> [B, seq_len, C]
        
        # 维度转换：将时间维度移到最后，以便线性层处理
        # [B, seq_len, C] -> [B, C, seq_len]
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        # ==================== 线性映射 ====================
        if self.individual:
            # Channel-Independent 模式：逐通道处理
            # 预分配输出张量
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype,
                device=seasonal_init.device,
            )
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_len],
                dtype=trend_init.dtype,
                device=trend_init.device,
            )
            
            # 对每个通道使用对应的线性层
            for i in range(self.channels):
                # seasonal_init[:, i, :] 形状: [B, seq_len]
                # Linear 输出形状: [B, pred_len]
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            # Channel-Shared 模式：所有通道共享同一线性层
            # 线性层会自动在最后一个维度上操作
            # [B, C, seq_len] -> [B, C, pred_len]
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        # ==================== 结果合成 ====================
        # 将季节成分和趋势成分的预测结果相加
        x = seasonal_output + trend_output
        # 形状: [B, C, pred_len]
        
        # 维度还原：[B, C, pred_len] -> [B, pred_len, C]
        return x.permute(0, 2, 1)

    def _compute_future_delta(self, x_fut_known):
        """
        根据未来已知外生变量计算目标序列校正项 delta。
        """
        if self.future_cov_head is None or x_fut_known is None:
            return None
        if x_fut_known.dim() != 3:
            return None

        if x_fut_known.size(-1) != self.future_cov_dim:
            if x_fut_known.size(-1) > self.future_cov_dim:
                x_fut_known = x_fut_known[..., :self.future_cov_dim]
            else:
                pad = torch.zeros(
                    x_fut_known.size(0),
                    x_fut_known.size(1),
                    self.future_cov_dim - x_fut_known.size(-1),
                    dtype=x_fut_known.dtype,
                    device=x_fut_known.device,
                )
                x_fut_known = torch.cat([x_fut_known, pad], dim=-1)

        delta = self.future_cov_head(x_fut_known)
        if self.delta_scale is not None:
            delta = delta * self.delta_scale
        return delta

    def forecast(self, x_enc, x_fut_known=None):
        """
        预测任务入口
        
        参数:
            x_enc (Tensor): 编码器输入 [batch_size, seq_len, channels]
        
        返回:
            Tensor: 预测结果 [batch_size, pred_len, channels]
        """
        base_out = self.encoder(x_enc)  # y_base
        delta = self._compute_future_delta(x_fut_known)
        if delta is None:
            return base_out

        if base_out.size(-1) == 1:
            return base_out + delta

        corrected = base_out.clone()
        corrected[..., self.target_channel_idx:self.target_channel_idx + 1] = (
            corrected[..., self.target_channel_idx:self.target_channel_idx + 1] + delta
        )
        return corrected

    def imputation(self, x_enc):
        """
        缺失值填补任务
        
        参数:
            x_enc (Tensor): 带缺失值的输入序列 [batch_size, seq_len, channels]
        
        返回:
            Tensor: 填补后的序列 [batch_size, seq_len, channels]
        """
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        """
        异常检测任务
        
        参数:
            x_enc (Tensor): 输入序列 [batch_size, seq_len, channels]
        
        返回:
            Tensor: 重构序列 [batch_size, seq_len, channels]
                   （可通过比较原始序列和重构序列检测异常）
        """
        return self.encoder(x_enc)

    def classification(self, x_enc):
        """
        时序分类任务
        
        参数:
            x_enc (Tensor): 输入序列 [batch_size, seq_len, channels]
        
        返回:
            Tensor: 分类 logits [batch_size, num_class]
        """
        # 获取编码器输出
        enc_out = self.encoder(x_enc)
        # 形状: [B, seq_len, C]
        
        # 展平为一维向量
        output = enc_out.reshape(enc_out.shape[0], -1)
        # 形状: [B, seq_len * C]
        
        # 通过投影层映射到类别数
        output = self.projection(output)
        # 形状: [B, num_class]
        
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, x_fut_known=None):
        """
        模型前向传播（统一接口）
        
        该方法提供了与其他时序模型（如 Transformer）统一的接口，
        但 DLinear 只使用编码器输入 x_enc，其他参数被忽略。
        
        参数:
            x_enc (Tensor): 编码器输入 [batch_size, seq_len, channels]
            x_mark_enc (Tensor): 编码器时间特征（DLinear 不使用）
            x_dec (Tensor): 解码器输入（DLinear 不使用）
            x_mark_dec (Tensor): 解码器时间特征（DLinear 不使用）
            mask (Tensor, optional): 掩码（用于缺失值填补任务）
        
        返回:
            Tensor: 根据任务类型返回不同形状的输出
                   - 预测任务: [batch_size, pred_len, channels]
                   - 填补/异常检测: [batch_size, seq_len, channels]
                   - 分类: [batch_size, num_class]
        """
        # 预测任务：长期预测或短期预测
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            dec_out = self.forecast(x_enc, x_fut_known=x_fut_known)
            # 只返回预测部分（最后 pred_len 个时间步）
            return dec_out[:, -self.pred_len:, :]
        
        # 缺失值填补任务
        if self.task_name == "imputation":
            return self.imputation(x_enc)
        
        # 异常检测任务
        if self.task_name == "anomaly_detection":
            return self.anomaly_detection(x_enc)
        
        # 分类任务
        if self.task_name == "classification":
            return self.classification(x_enc)
        
        # 未知任务类型
        return None
