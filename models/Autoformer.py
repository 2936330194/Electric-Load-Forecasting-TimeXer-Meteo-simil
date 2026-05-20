import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


class Model(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    Paper link: https://openreview.net/pdf?id=I55UqU-M11y
    
    Autoformer 模型主体。
    Autoformer 首次提出了在时间序列中进行 "系列级别 (series-wise)" 的连接和处理，
    利用基于 FFT 的自相关机制 (AutoCorrelation) 达到了 O(L log L) 的计算复杂度。
    同时，它引入了深度时间序列分解（Deep Decomposition）结构。
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name       # 任务名称（如长/短期预测、插补、分类、异常检测等）
        self.seq_len = configs.seq_len           # 编码器输入序列长度
        self.label_len = configs.label_len       # 解码器输入的历史已知序列长度
        self.pred_len = configs.pred_len         # 预测序列长度 

        # Decomp (序列分解模块)
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size) # 使用移动平均来拆分趋势和季节性

        # Embedding (编码器嵌入层)
        # 不使用绝对位置编码 (DataEmbedding_wo_pos)，因为 Autoformer 处理的是时间序列的内生周期，
        # 位置编码可能会在自相关操作的 FFT 频域阶段造成干扰。
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        # Encoder (编码器构建)
        # 包含了多个 EncoderLayer，每个 EncoderLayer 包含一个 AutoCorrelationLayer 和一系列操作
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model) # 采用了专门针对季节性的自定 LayerNorm
        )
        
        # Decoder (解码器构建，仅在时间序列预测时需要)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # 解码器的输入嵌入
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        # 第一个 AutoCorrelationLayer: 处理解码器输入序列的自相关操作
                        AutoCorrelationLayer(
                            AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        # 第二个 AutoCorrelationLayer: 交叉相关操作，处理解码器输入与编码器输出 (跨序列)
                        AutoCorrelationLayer(
                            AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.c_out,
                        configs.d_ff,
                        moving_avg=configs.moving_avg,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=my_Layernorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )
            
        # 针对不同任务的特定输出投影层
        if self.task_name == 'imputation':
            # 插补任务直接用线性层还原维度
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            # 异常检测任务重建原始输入
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            # 分类任务
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """预测任务前向传播"""
        # decomp init (解码器输入分解初始化)
        # 求编码器输入的均值，将其作为未来预测趋势的一个简单基准参考
        mean = torch.mean(x_enc, dim=1).unsqueeze(
            1).repeat(1, self.pred_len, 1)
        # 对于未来的预测季节性项，使用 0 来初始化 (因为季节项的均值趋近于0)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len,
                             x_dec.shape[2]], device=x_enc.device)
        # 用移动平均把历史输入进行一次初始分解
        seasonal_init, trend_init = self.decomp(x_enc)
        
        # decoder input (拼装解码器输入)
        # 趋势输入项: 截取历史标签序列趋势 ([..., -label_len:, ...]) 加上对未来的均值预测
        trend_init = torch.cat(
            [trend_init[:, -self.label_len:, :], mean], dim=1)
        # 季节输入项: 截取历史标签序列季节项 加上对未来初始化的 0 序列
        seasonal_init = torch.cat(
            [seasonal_init[:, -self.label_len:, :], zeros], dim=1)
            
        # enc (编码过程)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        
        # dec (解码过程)
        # 只对初始化的季节成分做 embedding
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        # 送入解码器，产出预测好的季节性特征和累计修正好的趋势项
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None,
                                                 trend=trend_init)
        # final (组装最终预测结果)
        dec_out = trend_part + seasonal_part
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        """插补任务前向传播（仅需 Encoder）"""
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final: 直接由 encoder 隐空间投射回原空间大小
        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        """异常检测任务前向传播（仅需 Encoder 进行序列重构）"""
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # final
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        """分类任务前向传播"""
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        # output transformer encoder/decoder embeddings don't include non-linearity
        # transformer 输出默认没有激活函数，因此手动加一个 gelu
        output = self.act(enc_out)
        output = self.dropout(output)
        
        # zero-out padding embeddings (基于 padding mask 把无用部分置 0)
        output = output * x_mark_enc.unsqueeze(-1)
        
        # 压平(flatten)：将 (batch_size, seq_length, d_model) 展平为 (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        
        # 分类预测：线性降维到类别数
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """统一的总前向传播入口，根据任务名称分发到对应的方法"""
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # 只返回对未来的预测步长特征 [B, pred_len, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
