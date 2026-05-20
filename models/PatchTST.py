"""
PatchTST.py - PatchTST 时间序列预测模型

论文: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
链接: https://arxiv.org/pdf/2211.14730.pdf

PatchTST 核心创新：
    1. Patching（分块）：将时间序列分割成多个 Patch，大幅降低序列长度
    2. Channel-Independence（通道独立）：每个变量独立处理，提高泛化能力
    3. Instance Normalization（实例归一化）：对输入进行标准化，提高稳定性

模型架构示意图：
    ┌─────────────────────────────────────────────────────────────────┐
    │                         PatchTST                                │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  输入: [B, L, C]                                                │
    │         ↓                                                       │
    │  ┌─────────────────────────────────────────────────────────┐    │
    │  │  Instance Normalization (RevIN)                         │    │
    │  │  x = (x - mean) / std                                   │    │
    │  └─────────────────────────────────────────────────────────┘    │
    │         ↓                                                       │
    │  维度变换: [B, L, C] → [B, C, L]                                │
    │         ↓                                                       │
    │  ┌─────────────────────────────────────────────────────────┐    │
    │  │  Patch Embedding                                        │    │
    │  │  将序列分割成 Patch 并嵌入                                │
    │  │  [B, C, L] → [B*C, num_patches, d_model]                │    │
    │  └─────────────────────────────────────────────────────────┘    │
    │         ↓                                                       │
    │  ┌─────────────────────────────────────────────────────────┐    │
    │  │  Transformer Encoder                                    │    │
    │  │  多层自注意力 + 前馈网络                                  │
    │  │  [B*C, num_patches, d_model]                            │    │
    │  └─────────────────────────────────────────────────────────┘    │
    │         ↓                                                       │
    │  维度恢复: [B*C, num_patches, d_model] → [B, C, d_model, num_patches] │
    │         ↓                                                       │
    │  ┌─────────────────────────────────────────────────────────┐    │
    │  │  Flatten Head                                           │    │
    │  │  展平 + 线性投影到预测长度                                │
    │  │  [B, C, d_model*num_patches] → [B, C, pred_len]         │    │
    │  └─────────────────────────────────────────────────────────┘    │
    │         ↓                                                       │
    │  ┌─────────────────────────────────────────────────────────┐    │
    │  │  Denormalization (反归一化)                             │    │
    │  │  output = output * std + mean                           │    │
    │  └─────────────────────────────────────────────────────────┘    │
    │         ↓                                                       │
    │  输出: [B, pred_len, C]                                         │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

Channel-Independence 的优势：
    1. 每个变量独立学习，避免变量间的噪声干扰
    2. 同一个模型可以处理不同数量的变量
    3. 减少参数量，提高训练效率
    4. 更好的迁移学习能力

Patch 的优势：
    1. 降低序列长度：L → L/stride，大幅减少计算量
    2. 捕获局部语义信息：每个 Patch 包含多个时间点
    3. 类似于 NLP 中的"词"，每个 Patch 是一个语义单元
    4. 减少信息冗余

支持的任务：
    - long_term_forecast / short_term_forecast: 长/短期预测
    - imputation: 缺失值填充
    - anomaly_detection: 异常检测
    - classification: 时序分类
"""

import torch                                                    # PyTorch 深度学习框架
from torch import nn                                            # 神经网络模块
from layers.Transformer_EncDec import Encoder, EncoderLayer     # Transformer 编码器
from layers.SelfAttention_Family import FullAttention, AttentionLayer  # 自注意力层
from layers.Embed import PatchEmbedding                         # Patch 嵌入层


class Transpose(nn.Module):
    """
    维度转置辅助模块
    
    用于在 nn.Sequential 中插入转置操作。
    主要用于 BatchNorm1d，因为它需要通道维度在中间位置。
    
    使用场景：
        nn.Sequential(
            Transpose(1, 2),      # [B, L, C] → [B, C, L]
            nn.BatchNorm1d(C),    # 在通道维度上归一化
            Transpose(1, 2)       # [B, C, L] → [B, L, C]
        )
    """
    
    def __init__(self, *dims, contiguous=False):
        """
        初始化
        
        参数:
            *dims: 要交换的维度索引，如 (1, 2) 表示交换第 1 和第 2 维
            contiguous (bool): 是否确保内存连续
        """
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        """
        前向传播：执行转置
        
        参数:
            x (Tensor): 输入张量
        
        返回:
            Tensor: 转置后的张量
        """
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    """
    展平预测头
    
    将编码器输出展平并投影到目标长度。
    这是 PatchTST 的输出层，用于生成最终预测。
    
    处理流程：
        输入: [B, C, d_model, num_patches]
        展平: [B, C, d_model * num_patches]
        线性: [B, C, target_window]
        Dropout: [B, C, target_window]
    """
    
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        """
        初始化预测头
        
        参数:
            n_vars (int): 变量数量（未使用，保留参数）
            nf (int): 展平后的特征维度（d_model * num_patches）
            target_window (int): 目标窗口长度（预测长度或序列长度）
            head_dropout (float): Dropout 比率
        """
        super().__init__()
        self.n_vars = n_vars
        
        # 展平最后两个维度：[..., d_model, num_patches] → [..., d_model*num_patches]
        self.flatten = nn.Flatten(start_dim=-2)
        
        # 线性投影到目标长度
        self.linear = nn.Linear(nf, target_window)
        
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        """
        前向传播
        
        参数:
            x (Tensor): 编码器输出 [B, C, d_model, num_patches]
        
        返回:
            Tensor: 预测输出 [B, C, target_window]
        """
        x = self.flatten(x)   # [B, C, d_model*num_patches]
        x = self.linear(x)    # [B, C, target_window]
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    PatchTST 模型
    
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    
    核心思想：
    - Patching: 将时间序列分割成固定长度的 Patch
    - Channel-Independence: 每个变量独立处理
    - Instance Normalization: 输入归一化 + 输出反归一化
    
    模型组件：
    - patch_embedding: Patch 嵌入层
    - encoder: Transformer 编码器（多层 EncoderLayer）
    - head: 预测头（FlattenHead 或分类投影）
    """

    def __init__(self, configs, patch_len=16, stride=8):
        """
        初始化 PatchTST 模型
        
        参数:
            configs: 配置对象，包含以下属性：
                - task_name: 任务名称（forecast/imputation/anomaly_detection/classification）
                - seq_len: 输入序列长度
                - pred_len: 预测序列长度
                - d_model: 模型维度
                - n_heads: 注意力头数
                - e_layers: 编码器层数
                - d_ff: 前馈网络隐藏层维度
                - dropout: Dropout 比率
                - activation: 激活函数（relu/gelu）
                - enc_in: 输入变量数
                - factor: 注意力因子（未使用）
            patch_len (int): Patch 长度，默认 16
            stride (int): Patch 步长，默认 8（重叠率 50%）
        """
        super().__init__()
        
        # 保存配置
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        # ==================== 未来已知外生变量分支配置 ====================
        # 该分支用于条件式多步直推：在基础预测 y_base 上学习残差 delta
        self.use_future_covariates = bool(getattr(configs, "use_future_covariates", False))
        self.future_cov_dim = int(getattr(configs, "future_cov_dim", 0))
        default_target_idx = self.enc_in - 1 if self.enc_in > 0 else 0
        self.target_channel_idx = int(getattr(configs, "target_channel_idx", default_target_idx))
        self.future_cov_dropout = float(getattr(configs, "future_cov_dropout", configs.dropout))

        # 从配置中读取 patch 参数（如果有的话）
        patch_len = getattr(configs, "patch_len", patch_len)
        stride = getattr(configs, "stride", stride)
        padding = stride  # 填充长度等于步长，确保完整分割

        # ==================== Patch Embedding ====================
        # 将序列分割成 Patch 并嵌入到 d_model 维空间
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout
        )

        # ==================== Transformer Encoder ====================
        # 构建多层编码器
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        # 使用标准全注意力，不使用因果掩码（双向注意力）
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)  # 堆叠 e_layers 层
            ],
            # 归一化层：使用 BatchNorm1d（需要转置以适配维度）
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(configs.d_model), Transpose(1, 2))
        )

        # ==================== 预测头 ====================
        # 计算 head 输入维度：d_model * num_patches
        # num_patches = (seq_len - patch_len) / stride + 2（+2 是因为 padding）
        self.head_nf = configs.d_model * int((configs.seq_len - patch_len) / stride + 2)
        
        # 根据任务类型选择不同的预测头
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            # 预测任务：输出长度为 pred_len
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
        elif self.task_name in ("imputation", "anomaly_detection"):
            # 填充/异常检测：输出长度为 seq_len（与输入相同）
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == "classification":
            # 分类任务：全局池化 + 线性分类器
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(self.head_nf * configs.enc_in, configs.num_class)

        # 仅在预测任务启用未来外生分支，其他任务保持原样
        if (
            self.task_name in ("long_term_forecast", "short_term_forecast")
            and self.use_future_covariates
            and self.future_cov_dim > 0
        ):
            hidden_dim = max(configs.d_model // 2, 32)
            self.future_cov_head = nn.Sequential(
                nn.LayerNorm(self.future_cov_dim),
                nn.Linear(self.future_cov_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(self.future_cov_dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.future_cov_head = None

    def _compute_future_delta(self, x_fut_known):
        """
        基于未来已知外生变量计算残差项 delta。

        参数:
            x_fut_known (Tensor): [B, pred_len, future_cov_dim]

        返回:
            Tensor or None: [B, pred_len]，若未启用则返回 None
        """
        if self.future_cov_head is None or x_fut_known is None:
            return None

        if x_fut_known.ndim != 3:
            raise ValueError(
                f"x_fut_known 维度错误，期望 [B, pred_len, cov_dim]，实际 {tuple(x_fut_known.shape)}"
            )

        if x_fut_known.shape[-1] < self.future_cov_dim:
            raise ValueError(
                f"x_fut_known 最后一维不足: got={x_fut_known.shape[-1]}, need={self.future_cov_dim}"
            )

        future_cov = x_fut_known[:, :, : self.future_cov_dim]
        delta = self.future_cov_head(future_cov).squeeze(-1)  # [B, pred_len]
        return delta

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_fut_known=None):
        """
        预测任务的前向传播
        
        参数:
            x_enc (Tensor): 编码器输入 [B, L, C]
            x_mark_enc: 编码器时间标记（未使用）
            x_dec: 解码器输入（未使用）
            x_mark_dec: 解码器时间标记（未使用）
        
        返回:
            dec_out (Tensor): 预测输出 [B, pred_len, C]
        
        处理流程：
            1. Instance Normalization：归一化输入
            2. Patch Embedding：分割成 Patch 并嵌入
            3. Transformer Encoder：提取时序特征
            4. Flatten Head：生成预测
            5. Denormalization：反归一化恢复原始尺度
        """
        # ==================== 1. Instance Normalization ====================
        # 计算均值和标准差
        means = x_enc.mean(1, keepdim=True).detach()  # [B, 1, C]
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)  # [B, 1, C]
        x_enc /= stdev

        # ==================== 2. 维度变换 + Patch Embedding ====================
        # [B, L, C] → [B, C, L]（Channel-Independence 需要这种格式）
        x_enc = x_enc.permute(0, 2, 1)
        # Patch Embedding: [B, C, L] → [B*C, num_patches, d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)
        
        # ==================== 3. Transformer Encoder ====================
        # [B*C, num_patches, d_model] → [B*C, num_patches, d_model]
        enc_out, _ = self.encoder(enc_out)
        
        # ==================== 4. 维度恢复 ====================
        # [B*C, num_patches, d_model] → [B, C, num_patches, d_model]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # [B, C, num_patches, d_model] → [B, C, d_model, num_patches]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # ==================== 5. Flatten Head ====================
        # [B, C, d_model, num_patches] → [B, C, pred_len]
        dec_out = self.head(enc_out)
        # [B, C, pred_len] → [B, pred_len, C]
        dec_out = dec_out.permute(0, 2, 1)

        # ==================== 6. Denormalization ====================
        # 恢复原始尺度
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        # ==================== 7. 未来外生残差修正（可选） ====================
        # y_hat = y_base + delta，仅修正目标通道
        delta = self._compute_future_delta(x_fut_known)
        if delta is not None:
            target_idx = max(0, min(self.target_channel_idx, dec_out.shape[-1] - 1))
            dec_out = dec_out.clone()
            dec_out[:, :, target_idx] = dec_out[:, :, target_idx] + delta
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """
        缺失值填充任务的前向传播
        
        参数:
            x_enc (Tensor): 编码器输入 [B, L, C]（包含缺失值）
            x_mark_enc: 时间标记（未使用）
            x_dec, x_mark_dec: 解码器输入（未使用）
            mask (Tensor): 掩码 [B, L, C]，1=有效，0=缺失
        
        返回:
            dec_out (Tensor): 填充后的序列 [B, L, C]
        """
        # ==================== Instance Normalization（考虑掩码） ====================
        # 只对有效值计算均值
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)  # [B, C]
        means = means.unsqueeze(1).detach()  # [B, 1, C]
        x_enc = x_enc - means
        # 将缺失值置零
        x_enc = x_enc.masked_fill(mask == 0, 0)
        # 只对有效值计算标准差
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()  # [B, 1, C]
        x_enc /= stdev

        # ==================== Patch Embedding + Encoder ====================
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out, _ = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # ==================== Flatten Head + Denormalization ====================
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        # 恢复原始尺度（输出长度为 seq_len）
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        """
        异常检测任务的前向传播
        
        原理：模型学习重建正常序列，异常点的重建误差较大
        
        参数:
            x_enc (Tensor): 输入序列 [B, L, C]
        
        返回:
            dec_out (Tensor): 重建序列 [B, L, C]
        """
        # ==================== Instance Normalization ====================
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # ==================== Patch Embedding + Encoder ====================
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out, _ = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # ==================== Flatten Head + Denormalization ====================
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)

        # 恢复原始尺度
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        """
        时序分类任务的前向传播
        
        参数:
            x_enc (Tensor): 输入序列 [B, L, C]
            x_mark_enc: 时间标记（未使用）
        
        返回:
            output (Tensor): 分类 logits [B, num_class]
        """
        # ==================== Instance Normalization ====================
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # ==================== Patch Embedding + Encoder ====================
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out, _ = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # ==================== 分类头 ====================
        # 展平所有特征
        output = self.flatten(enc_out)  # [B, C, d_model*num_patches]
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)  # [B, C*d_model*num_patches]
        # 线性分类器
        output = self.projection(output)  # [B, num_class]
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, x_fut_known=None):
        """
        模型前向传播（统一入口）
        
        根据 task_name 分发到不同的任务处理方法。
        
        参数:
            x_enc (Tensor): 编码器输入 [B, L, C]
            x_mark_enc (Tensor): 编码器时间标记
            x_dec (Tensor): 解码器输入
            x_mark_dec (Tensor): 解码器时间标记
            mask (Tensor): 缺失值掩码（仅 imputation 任务）
        
        返回:
            根据任务类型返回不同形状的输出：
            - forecast: [B, pred_len, C]
            - imputation: [B, seq_len, C]
            - anomaly_detection: [B, seq_len, C]
            - classification: [B, num_class]
        """
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, x_fut_known=x_fut_known)
            return dec_out[:, -self.pred_len:, :]  # 只返回预测部分
        if self.task_name == "imputation":
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        if self.task_name == "anomaly_detection":
            return self.anomaly_detection(x_enc)
        if self.task_name == "classification":
            return self.classification(x_enc, x_mark_enc)
        return None
