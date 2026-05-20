import torch
import torch.nn as nn
import torch.nn.functional as F


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    
    专为季节性（seasonal）部分设计的层归一化（LayerNorm）。
    它在标准 LayerNorm 的基础上，减去了在序列长度（dim=1）上的均值，
    这有助于进一步消除数据中的全局趋势影响，突显季节性波动。
    """

    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        # 先进行标准的层归一化计算
        x_hat = self.layernorm(x)
        # 计算每个特征维度（channel）上的时间步序列均值
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        # 减去该偏差，返回更纯粹的季节性特征
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    
    移动平均模块（Moving Average Block），通过平均池化（Average Pooling）
    来平滑时间序列，从而提取出数据的主导趋势（Trend）。
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        # 核心操作是 1D 平均池化
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        # 为了保持使用滑动平均后的序列长度不变，对序列的首尾进行 Padding
        # 重复第一帧数据和最后一帧数据作为填充
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        # 将填充的部分拼接到原序列上
        x = torch.cat([front, x, end], dim=1)
        # nn.AvgPool1d 要求输入维度为 [Batch, Channels, Length]，因此先 permute
        x = self.avg(x.permute(0, 2, 1))
        # 池化完后再变回 [Batch, Length, Channels]
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    
    序列分解模块。将输入时间序列分解为：
    1. 趋势周期项（Trend-Cyclical part），通过滑动平均获取
    2. 季节性项（Seasonal part），原序列减去趋势项后的残差
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        # 使用上面的滑动平均模块，步长固定为 1
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        # 提取趋势（平滑过的序列）
        moving_mean = self.moving_avg(x)
        # 用原序列减去趋势，得到剩余的季节性部分（高频部分）
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Multiple Series decomposition block from FEDformer
    
    来自 FEDformer 的多尺度序列分解模块。
    通过使用多个不同大小的卷积核（kernel_size 列表）进行滑动平均，
    并将多个尺度下分解得到的季节项和趋势项分别求平均，以获取更鲁棒的分解结果。
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        # 生成多个不同核大小的分解模块列表
        self.series_decomp = [series_decomp(kernel) for kernel in kernel_size]

    def forward(self, x):
        moving_mean = []
        res = []
        # 分别用每一个 kernel 尺寸的分解器对序列进行分解
        for func in self.series_decomp:
            sea, moving_avg = func(x)
            moving_mean.append(moving_avg)
            res.append(sea)

        # 把多个不同尺度的结果加起来取平均
        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean


class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    
    Autoformer 编码器层，具有渐进式序列分解架构。
    与普通 Transformer 编码器层不同的是：
    1. 使用了自相关机制（AutoCorrelation）或其他 Attention。
    2. 用 `series_decomp` 代替了原来的 `LayerNorm+Add` 中的 LayerNorm，
       让模型在做完注意力和前馈网络后，逐步地提取和抛弃趋势信息，只保留季节性特征。
    """

    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        # FFN（前馈神经网络）的两层 1D 卷积，这里没有使用标准的 Linear 层而是用的 Conv1d
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        # 定义了两个序列分解模块
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # 1. Attention 计算
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        # 2. Add: 加上残差连接
        x = x + self.dropout(new_x)
        
        # 3. 第一次分解：将 Attention + 残差后的结果进行序列分解
        # 提取季节性特征 x (即残差部分)，将提取出的趋势部分舍弃（_）
        x, _ = self.decomp1(x)
        
        # 4. FFN (前馈网络)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        # 5. Add & 进一步分解
        # 将 FFN 后的输出与上一步的季节性特征（当作残差）相加，再进行第二次分解
        res, _ = self.decomp2(x + y)
        
        # 返回季节性特征 res (也是最终向后传递的表示)，以及 attention 权重
        return res, attn


class Encoder(nn.Module):
    """
    Autoformer encoder
    
    Autoformer 的编码器模块，堆叠多个 EncoderLayer。
    有可能包含序列下采样的卷积层（conv_layers），不过这通常在 Informer 中使用得较多。
    """

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)  # 所有的 Attention / Encoder 层的列表
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer  # 顶层最终的归一化模块

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            # 如果配置了下采样卷基层，就交替进行 attention 和下采样
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            # 最后一个 attention 层之后不再跟卷积
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            # 正常堆叠 Encoder 层
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        # 如果有 norm 层，最后通过一下
        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    
    Autoformer 解码器层，同样具有渐进式序列分解架构。
    解码器层负责融合自身历史输入和来自编码器的表示。
    
    核心流程包含三部分，每一步都会生成残差/季节项(x)，并提取出局部趋势项(trend)，
    最后将三步提取出的所有局部趋势累加，作为该层的总辅助趋势预测输出。
    """

    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        
        self.self_attention = self_attention      # 自注意力层
        self.cross_attention = cross_attention    # 交叉注意力层（接收编码器输出）
        
        # FFN 前馈网络层
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        
        # 三个序列分解模块
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        
        self.dropout = nn.Dropout(dropout)
        
        # 一维卷积投影层，用于最终把这层聚合的趋势量（trend）映射回实际输出变量的通道数（c_out）
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # 第一部分：Self-Attention 与序列分解
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend1 = self.decomp1(x)  # 分解出季节项 x 和趋势项 trend1
        
        # 第二部分：Cross-Attention 与序列分解（与 Encoder 的输出交互）
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        x, trend2 = self.decomp2(x)  # 分解出季节项 x 和趋势项 trend2
        
        # 第三部分：FFN（前馈网络）与序列分解
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y) # 分解出最终的季节项 x 和趋势项 trend3

        # 将三部分提取到的部分趋势值相加，合成当前层的总捕获趋势值
        residual_trend = trend1 + trend2 + trend3
        # 使用 1D 卷积投影将隐变量维度的趋势变为目标输出变量的通道数 c_out
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        
        # 返回传递给下一层的季节性表示 x，以及本层产出的局部预测趋势项 residual_trend
        return x, residual_trend


class Decoder(nn.Module):
    """
    Autoformer decoder (这里的类定义注解原文可能笔误写成了 encoder，实际是解码器 Decoder)
    
    由多层 DecoderLayer 堆叠而成。
    解码器的一个核心思想是将：
      1. 季节成分（通过层层相乘与非线性变换，类似标准 Transformer 继续流动）
      2. 趋势成分（每一层分解出的趋势通过简单相加来不断累积）
    分离开来，并在最后才合入对应的投影层。
    """

    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer            # 对最终出提取的所有季节特征进行归一化
        self.projection = projection      # 将最终出提取的季节特征映射为预测变量维度

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        # 遍历每一层解码器
        for layer in self.layers:
            # 持续传入随流更新的季节变量 x；以及始终来自 Encoder 的源源不断的 cross
            # 每层都会返回更新过一次的 x，并且产生本层的局部预测趋势 residual_trend
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            
            # 将该层输出的局部预测趋势累加到总趋势 (trend) 中去
            trend = trend + residual_trend

        # 对最终的残差季节部分进行特定的层归一化
        if self.norm is not None:
            x = self.norm(x)

        # 对最终的残差季节部分投影到所需的 channel
        if self.projection is not None:
            x = self.projection(x)
            
        # 返回处理完的季节特征 x，以及不断累加融合出来的趋势主线 trend
        # 在最终的 Autoformer 模型中通常会将两者简单相加作为最终的预测结果。
        return x, trend
