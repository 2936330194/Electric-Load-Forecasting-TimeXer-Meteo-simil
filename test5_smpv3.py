"""
test5_smpv3.py - 气象特征与相似日先验特征的早期融合 (Early fusion of weather tokens and similar-day prior token)

与旧版 test5_smp.py 的对比 (核心升级点):
1. 数据提纯: 历史检索到的加权相似日曲线在这里被缩减为单纯的一条 96 步预测窗口长度的先验曲线，抛弃了无用的历史冗余。
2. 专属映射通道: 引入一个专门的线性映射层 (Linear 层)，将这 96 步的相似日曲线映射、压缩成一个统一维度 (d_model) 的独立外生变量 Token。
3. 注意力融合机制: 在数据被送入 TimeXer 核心的 Cross-Attention 前，这个专门生成的“相似日 Token”将与气象模型的输出 Tokens 拼接在一起。
   这种架构直接允许 TimeXer 的注意力机制根据当前态势自动决策该信赖气象模型还是相似日经验，彻底消除了末端直接“残差相加”带来的尺度紊乱。
"""

import argparse
import hashlib
import json
import os
import random
import time
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from models.TimeXer import Model as TimeXer
import test4_smp as base
from utils.forecast_visualization import plot_pred_vs_true, predict_future_load_from_csv
from utils.metrics import metric
from utils.quantile import QuantileLoss
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.weather_e2e import FullMapWeatherConvExtractor, WeatherGridStore, weather_data_provider


# ================= 相似日检索参数配置 =================
# 指定相似日检索模型的缓存目录（包含离线训练好的 PCA 分解器和 Faiss 向量索引库）。
# 如果设为 None，程序通常会在 _parse_cli_args 中回落到命令行参数里的默认路径。
SIMILAR_DAY_ARTIFACT_DIR: Optional[str] = None

# 在检索历史中最相近日期的天气时，选取前 K(这里是3) 个最相似的日期来生成先验负荷曲线。
# 一般 K 选取 3~5 左右能起到较好的去噪及平滑效果。
SIMILAR_DAY_TOP_K = 3

# 这是一个总开关，决定接下来的测试或训练环节中，是否要启用并组装这条相 似日先验特征进入端到端模型。
USE_SIMILAR_DAY_PRIOR = True
TRAIN_MODE = base.TRAIN_MODE

LOAD_FROM_OPTUNA = False
OPTUNA_DIR = "./optuna"
OPTUNA_BEST_PARAMS_FILE = "best_params5.json"
OPTUNA_BEST_CONFIG_FILE = "best_config5.json"
OPTUNA_BEST_WEIGHT_FILE = "best_model5.pth"
OPTUNA_BEST_TRIAL_FILE = "best_trial_result5.json"

TUNABLE_PARAM_MAP = {
    "SIMILAR_DAY_TOP_K": "similar_day_top_k",
    "WEATHER_FEATURE_DIM": "weather_feature_dim",
    "D_MODEL": "d_model",
    "N_HEADS": "n_heads",
    "E_LAYERS": "e_layers",
    "D_FF": "d_ff",
    "DROPOUT": "dropout",
    "PATCH_LEN": "patch_len",
    "BATCH_SIZE": "batch_size",
    "LEARNING_RATE": "learning_rate",
}

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


def _load_json_file(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _apply_optuna_artifacts(args: argparse.Namespace) -> argparse.Namespace:
    optuna_dir = os.path.abspath(OPTUNA_DIR)
    config_path = os.path.join(optuna_dir, OPTUNA_BEST_CONFIG_FILE)
    params_path = os.path.join(optuna_dir, OPTUNA_BEST_PARAMS_FILE)
    weight_path = os.path.join(optuna_dir, OPTUNA_BEST_WEIGHT_FILE)

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"./optuna model weight file not found: {weight_path}")

    if os.path.exists(config_path):
        payload = _load_json_file(config_path)
        if not isinstance(payload, dict):
            raise ValueError(f"./optuna config file must be a JSON object: {config_path}")
        for key, value in payload.items():
            setattr(args, key, value)
    elif os.path.exists(params_path):
        payload = _load_json_file(params_path)
        if not isinstance(payload, dict):
            raise ValueError(f"./optuna params file must be a JSON object: {params_path}")
        for raw_key, value in payload.items():
            key = TUNABLE_PARAM_MAP.get(str(raw_key), str(raw_key))
            setattr(args, key, value)
    else:
        raise FileNotFoundError(
            f"./optuna missing both {OPTUNA_BEST_CONFIG_FILE} and {OPTUNA_BEST_PARAMS_FILE}"
        )

    args.is_training = 0
    args.load_weight_path = weight_path
    args.quantiles = list(args.quantiles)
    args.n_quantiles = len(args.quantiles)
    print(f"Loaded saved artifacts from ./optuna: {weight_path}")
    return args


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


class FullMapConvTimeXerSimilarTokenQuantile(nn.Module):
    """
    端到端架构类：包含早期特征拼接机制的综合概率预测神经网络。
    - 主干1: FullMapWeatherConvExtractor （用 3D 卷积负责抽取网格气象在空间上的纹理关联特征）。
    - 主干2: 全新的 Similar-Day Embedding（用 Linear 层将预测区的 96 步特征映射为一个高维 Token）。
    - 处理器: TimeXer （此时它的 Cross-Attention 负责综合评判内生特征、气象特征以及相似日先脸特征）。
    """
    def __init__(self, configs, quantiles: Sequence[float]):
        super().__init__()
        # ------- 参数配置与初始化 -------
        self.quantiles = list(quantiles)                                            # 分位数列表 (如 [0.1, 0.5, 0.9])
        self.n_quantiles = len(self.quantiles)                                      # 需要预测的分位数数量
        self.weather_feature_dim = int(configs.weather_feature_dim)                 # CNN 抽取的每一帧气象图像降阶后的向量维度
        self.encode_chunk_size = int(getattr(configs, "weather_encode_chunk_size", 512)) # 图像分块送入 GPU 的大小限制 (防 OOM)
        
        # ------- 相似日超参数 -------
        self.use_similar_day_prior = bool(getattr(configs, "use_similar_day_prior", False))
        self.similar_day_top_k = int(getattr(configs, "similar_day_top_k", 3))
        # 先验特征深度: 原材料里包含了 TopK 条单天曲线，再加上 1 条综合加权后的最终最优底线曲线
        self.similar_day_prior_dim = self.similar_day_top_k + 1 if self.use_similar_day_prior else 0
        self.pred_len = int(configs.pred_len)                                       # 未来预测的时间步 (通常是 96)
        self.d_model = int(configs.d_model)                                         # TimeXer 核心隐空间的维度大小

        # ------- 构建子网络 1: 气象图片降维卷积核 -------
        self.weather_backbone = FullMapWeatherConvExtractor(
            in_channels=int(getattr(configs, "weather_in_channels")),
            out_channels=self.weather_feature_dim,
            kernel_height=int(getattr(configs, "weather_kernel_height")),
            kernel_width=int(getattr(configs, "weather_kernel_width")),
            dropout=float(getattr(configs, "dropout", 0.1)),
        )

        # ------- 部署主力时序模型 TimeXer -------
        self.weather_seq_len = int(getattr(configs, "weather_seq_len", configs.seq_len))
        configs.exo_seq_len = self.weather_seq_len # 提示 TimeXer 准备好接收这个长度的外生变量序列
        configs.enc_in = 1                         # 单变量自身预估 (输入序列只含有 1 维的 Load)
        self.timexer = TimeXer(configs)            # 初始化主角模型

        # ------- 构建子网络 2: 方案 2 专用的相似日高维投射器 -------
        if self.use_similar_day_prior:
            # 作用是将 [Batch, 96] 直接压缩并表达为 [Batch, d_model]
            self.similar_day_token_proj = nn.Linear(self.pred_len, self.d_model)
        else:
            self.similar_day_token_proj = None

        # ------- 最终阶段: 不确定性（多概率）重构映射头 -------
        self.quantile_head = nn.Linear(1, self.n_quantiles)
        # 初始化权重：让初始阶段所有分位数尽量重合在真实点附近，偏置给予微小的随机发散，帮助网络迅速撕开上下限包络
        with torch.no_grad():
            self.quantile_head.weight.fill_(1.0)
            self.quantile_head.bias.copy_(torch.tensor([q - 0.5 for q in self.quantiles]) * 0.1)

    def _encode_weather_frames(self, weather_frames: torch.Tensor) -> torch.Tensor:
        """
        （最底盘的计算池层）
        接收四维张量 [被裁剪/抽离出的独立图像数量, Channels, Height, Width]，
        用设定好的 chunks 步长循环调用气象 CNN 抽取器。
        有效避免在训练大批量序列（比如 Batch=16, seq=768）时把这上万张图一口气全塞进显卡打爆显存。
        """
        if weather_frames.ndim != 4:
            raise ValueError(
                f"Weather frames should have shape [N, C, H, W], got {tuple(weather_frames.shape)}"
            )

        encoded_chunks: List[torch.Tensor] = []
        # 分而治之，每次只压入 encode_chunk_size(默认512) 张图片
        for start in range(0, weather_frames.shape[0], self.encode_chunk_size):
            end = min(start + self.encode_chunk_size, weather_frames.shape[0])
            encoded_chunks.append(self.weather_backbone(weather_frames[start:end].float()))
        # 将结果按原样拼接回一条总的一阶张量序列
        return torch.cat(encoded_chunks, dim=0)

    def _encode_weather_sequence(
        self,
        weather_seq: Optional[torch.Tensor],
        weather_index: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        （中层调度层）
        接收外部传来的图像时序载体，并负责在通过 _encode_weather_frames 获取高维特征后整平、恢复出具备时序性的张量结构。
        """
        if weather_seq is None:
            return None

        # 模式A：【高效去重模式 / Context 索引模式】 
        # 数据集抛来的是去重过的图片字典(weather_seq)以及一个二维时序向字典取值的索引表(weather_index)
        if weather_index is not None:
            if weather_seq.ndim != 4 or weather_index.ndim != 2:
                raise ValueError(
                    "Indexed weather mode expects weather_seq [U, C, H, W] and weather_index [B, T]."
                )
            batch_size, time_len = weather_index.shape
            encoded_frames = self._encode_weather_frames(weather_seq)  # 获取所有不重复的独特时刻气象特征 [N_unique, d_out]
            gathered = encoded_frames.index_select(0, weather_index.reshape(-1)) # 根据时钟剧本重新选取填槽
            return gathered.reshape(batch_size, time_len, self.weather_feature_dim) # [Batch, Seq_Len, d_out]

        # 模式B：【原教旨直连模式 / 傻瓜模式】
        # DataLoader 不考虑图象重叠与复用，直接抛来一整条全冗余序列集 [Batch, Seq_Len, C, H, W]
        if weather_seq.ndim != 5:
            raise ValueError(
                f"Sequential weather mode expects [B, T, C, H, W], got {tuple(weather_seq.shape)}"
            )
        batch_size, time_len, channels, height, width = weather_seq.shape
        flat = weather_seq.reshape(batch_size * time_len, channels, height, width) # 先压平成 4D 交由底层计算
        encoded = self._encode_weather_frames(flat)
        return encoded.reshape(batch_size, time_len, self.weather_feature_dim)    # 算完之后按规矩重新归档还原 3D 骨架

    def _build_similar_day_token(self, similar_day_prior: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        [本程序的核心创新点 - 相似日经验专属高维表征器]
        将 Dataset 提供的一条具有预测区间长度 (96步) 的先验曲线信息，
        利用自有的投影通道 (Linear 层) 压缩成一个供给 TimeXer 吞吐的规范特征 Token。
        """
        if not self.use_similar_day_prior or similar_day_prior is None:
            return None
            
        # --------------------- 严格的输入形状与尺寸哨兵防御 ---------------------
        # 必须是三维的 Batch 张量
        if similar_day_prior.ndim != 3:
            raise ValueError(
                f"similar_day_prior should be [B, pred_len, {self.similar_day_prior_dim}], "
                f"got {tuple(similar_day_prior.shape)}"
            )
        # 序列的时间维度长必须强制要求等于未来待预测步数 (抛弃了原先 672 的无用尾巴)
        if similar_day_prior.shape[1] != self.pred_len:
            raise ValueError(
                f"similar_day_prior time dimension does not match pred_len: {similar_day_prior.shape[1]} vs {self.pred_len}"
            )
        # 通道数应该符合设置的 TopK + 1 
        if similar_day_prior.shape[2] != self.similar_day_prior_dim:
            raise ValueError(
                "similar_day_prior feature dimension does not match configuration: "
                f"{similar_day_prior.shape[2]} vs {self.similar_day_prior_dim}"
            )
            
        # --------------------- 构造并生成 Token ---------------------
        # 取第 0 号切片（第 0 维通常存放由多条候选天动态 Softmax 加权融合的最优平滑合成基准线）
        weighted_prior = similar_day_prior.float()[:, :, 0]  # Shape: [Batch, pred_len]
        
        # 将其一次性通过专属映射层，从长度为预测区间的曲线转为 TimeXer 可理解的隐模型向量。
        # .unsqueeze(1) 是为了构造序列感，即将其变幻为一个 Sequence Length 为 1 的 Token: [Batch, 1, d_model]
        return self.similar_day_token_proj(weighted_prior).unsqueeze(1)

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
        端到端早融合前向骨干路线（Early Fusion Control Room）
        这是贯穿全模型脉络的总控开关，将各个子网络抽取的特征按特定方式塞入主脑 TimeXer。
        """
        # 第一阶段：各自为营，针对异构原生数据执行特征提纯
        # 1. 启动气象骨干，提取的是高维网格经 CNN 后的展平时序，形状 -> [Batch, weather_seq_len, feature_dim]
        weather_feature = self._encode_weather_sequence(weather_x, weather_x_index)
        # 2. 启动相似日专属投影，获取高度凝练的一站式的先验特征经验 Token -> [Batch, 1, d_model]
        similar_day_token = self._build_similar_day_token(similar_day_prior)

        # 第二阶段：气象变量 Token 集团 与 相似日先验 Token 的跨模态大聚会 
        # TimeXer 的 _forecast 内部已经开辟了接口，如果接到 x_exo_extra_tokens，
        # 会在 Cross-Attention 之际将这具有不同物理意义的数据拼接在一个序列槽里互相交织碰撞。
        point_pred = self.timexer(
            load_x,                     # 截断的内部负荷历史记录 (主要内生基准)
            x_mark_enc,                 # 时间标记集（向模型提供当前的星期、日期、节假日等）
            None,                       # decoder 对于非自回归网络弃用，赋空
            None,
            mask=mask,                  # padding 盲区罩子
            x_exo=weather_feature,      # 常规的由 3D 卷积提纯产生的气象外部长时特征序列
            x_exo_mark=x_exo_mark,      # 这段气象特征对应的未来和历史时间点标记
            x_exo_extra_tokens=similar_day_token,  # [核心升级] 强行插队：追加的用于表达历史“业务经验”的高维载体
        )
        
        # 第三阶段：切除不需要的历史时间点预测废渣，只提取对应面向未来那段区间的目标预估
        point_pred = point_pred[:, -self.timexer.pred_len :, :]
        
        # 返回前走一遭分位数发散头，将唯一的一条预测线映射为 n_quantiles 条不确定性边界包络网
        return self.quantile_head(point_pred)


def validate_quantile(model, data_loader, criterion, args, device, use_amp: bool = False) -> float:
    """计算单个 Loader 中的标准验证集测试/防过拟合过程（不进行 Backwards 梯度回传）"""
    model.eval()                                                  # 挂起所有的 Dropout 和 Batch Normalization 等训练特有的网络惩罚扰乱机制
    total_loss = []
    # 如果允许且在 CUDA 上，则使用非阻塞指令（将 CPU 计算张量向显卡上异步抛送，防止 GPU 大量时间浪费在挂起等待同步上）
    use_non_blocking = _use_non_blocking_transfer(args, device)

    # 替换原本老旧的 torch.no_grad()，inference_mode 提供更好的底层推理指令优化以及大幅降低显存和资源申请占用
    with torch.inference_mode():
        for batch in data_loader:
            # 1. 拆包: 将包含先验与气象的混合 7 维度或 6 维度特装子弹盒提取并分类
            (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_exo_mark,
                batch_weather_frames,
                batch_weather_index,
                similar_day_prior,
            ) = _unpack_weather_batch(batch)

            # 2. 内存推移: 将解包出来的各种张量依照规则向推理所在设备 (常见: cuda:0) 进行寄送转移
            batch_x = _to_float_device(batch_x, device, non_blocking=use_non_blocking)
            batch_y = _to_float_device(batch_y, device, non_blocking=use_non_blocking)
            batch_x_mark = _to_float_device(batch_x_mark, device, non_blocking=use_non_blocking)
            batch_exo_mark = _to_float_device(batch_exo_mark, device, non_blocking=use_non_blocking)
            batch_weather_frames = _to_float_device(batch_weather_frames, device, non_blocking=use_non_blocking)
            batch_weather_index = _to_long_device(batch_weather_index, device, non_blocking=use_non_blocking)
            
            # 由于验证集有时可能配置不使用先验，先做一个空判断防爆，再执行显卡投递
            if similar_day_prior is not None:
                similar_day_prior = _to_float_device(similar_day_prior, device, non_blocking=use_non_blocking)

            # 3. 执行推演: 配合 AMP 混合精度推理，榨干新一代显卡特有 Tensor Core 的计算频率以获取半精度的大幅速度提升
            with torch.amp.autocast("cuda", enabled=use_amp):
                # 利用字典（**Kwargs）的形式动态装配传参，以避免形参位置出现问题或者不需要先验的时候反而因为 None 取点报错的情况
                model_kwargs = {
                    "load_x": batch_x,
                    "x_mark_enc": batch_x_mark,
                    "x_exo_mark": batch_exo_mark,
                    "weather_x": batch_weather_frames,
                    "weather_x_index": batch_weather_index,
                }
                # 如果开启并实际拿到了基于目标天气特征索取的当前高信心相似日信息，便挂载这块“额外经验先验包”
                if similar_day_prior is not None:
                    model_kwargs["similar_day_prior"] = similar_day_prior
                    
                # 【前向全开】：模型启动计算，此时对于深层 TimeXer 它会综合内生特征和两种外生先验作出对应的分位数预测面
                outputs = model(**model_kwargs)
                
                # 由于输入模型参与验证的 batch_y 往往是由原本长度的历史特征(X部分)+未来预测标签(Y部分)冗长拼装而成的组合矩阵，
                # 我们需要在评判误差时截断不属于自己预测的区域：只截取其代表未来 96 长度的预想端点
                batch_y_target = extract_target(batch_y[:, -args.pred_len :, :])
                
                # 将基于各概率包络线产生的多条点预测集合与唯一的实际真值点 Y 坐标进行针对于特定 Quantile 损失标准计算得分
                loss = criterion(outputs, batch_y_target)
                
            # 将这个小批次的误差打分计算结果作为一条存根抽取出来作为该 Epoch 的成绩单记录之一
            total_loss.append(loss.item())

    model.train() # 所有批次验证巡回结束，解冻 dropout 等模块将模型重新拉起为训练态，继续准备下一轮的反向求导推演 Backwards
    # 汇总成绩单，计算该网络在整个验证集上的严格平铺平均 loss 用作最后对验证集表现优劣的防过拟合 Early Stopping 的天平判决衡量准则
    return float(np.average(total_loss)) if total_loss else np.nan


def train_quantile_model(model, args, device, weather_store: WeatherGridStore):
    _, train_loader = weather_data_provider(args, "train", weather_store)
    _, vali_loader = weather_data_provider(args, "val", weather_store)
    _, test_loader = weather_data_provider(args, "test", weather_store)

    setting = _get_setting(args)
    path = os.path.join(args.checkpoints, setting)
    os.makedirs(path, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = QuantileLoss(args.quantiles).to(device)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    use_amp = bool(getattr(args, "use_amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    use_non_blocking = _use_non_blocking_transfer(args, device)

    print("\n" + "=" * 72)
    print("Start training weather-token + similar-day-token early-fusion model")
    print(f"setting: {setting}")
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
            "similar_day_token_config (先验组件启动): "
            f"top_k={getattr(args, 'similar_day_top_k', 0)}, "
            f"token_proj=Linear({args.pred_len}->{args.d_model}), "
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
        test_loss = validate_quantile(model, test_loader, criterion, args, device, use_amp=use_amp)
        train_loss_avg = float(np.average(train_loss)) if train_loss else np.nan
        print(
            f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.1f}s | "
            f"Train: {train_loss_avg:.7f} Vali: {vali_loss:.7f} Test: {test_loss:.7f}"
        )

        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        adjust_learning_rate(optimizer, epoch + 1, args)

    best_model_path = os.path.join(path, "checkpoint.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"Loaded best model weights: {best_model_path}")
    return model


def test_quantile_model(model, args, device, weather_store: WeatherGridStore) -> str:
    test_data, test_loader = weather_data_provider(args, "test", weather_store)

    setting = _get_setting(args)
    folder_path = os.path.join(getattr(args, "results_root", "./results/"), setting)
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
        print(f"P50 Test Metrics (Inverse): MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}")
    else:
        mae, mse, rmse, mape, mspe = metric(preds_p50, trues)
        print(f"P50 Test Metrics (Normalized): MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}")

    return folder_path


def _get_setting(args, itr: int = 0) -> str:
    des_raw = str(getattr(args, "des", "exp"))
    des_norm = "".join(ch for ch in des_raw.lower() if ch.isalnum()) or "exp"
    des_short = "op" if des_norm.startswith("optuna") else des_norm[:6]
    signature = (
        f"{args.task_name}_{args.model_id}_{args.model}_e2e_sdv3_"
        f"sl{args.seq_len}_pl{args.pred_len}_dm{args.d_model}_"
        f"el{args.e_layers}_wd{args.weather_feature_dim}_"
        f"wsl{args.weather_seq_len}_wh{args.weather_history_len}_"
        f"wk{args.weather_kernel_height}x{args.weather_kernel_width}_"
        f"sdp{int(bool(getattr(args, 'use_similar_day_prior', False)))}_"
        f"sdk{int(getattr(args, 'similar_day_top_k', 0))}_"
        f"lr{args.learning_rate}_bs{args.batch_size}_{args.des}_{itr}"
    )
    digest = hashlib.md5(signature.encode("utf-8")).hexdigest()[:8]
    return (
        f"sdv3_sl{args.seq_len}_pl{args.pred_len}_"
        f"wd{args.weather_feature_dim}_sdk{int(getattr(args, 'similar_day_top_k', 0))}_"
        f"bs{args.batch_size}_{des_short}{int(itr):03d}_{digest}"
    )


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
        is_training=1 if TRAIN_MODE else 0,
        model_id=f"{base.MODEL_ID_PREFIX}_sdv3",
        model=base.MODEL,
        des=base.DES,
        itr=base.ITR,
        data="custom",
        root_path=base.ROOT_PATH,
        data_path=base.DATA_PATH,
        future_path=base.FUTURE_PATH,
        features=base.FEATURES,
        target=base.TARGET,
        target_channel_idx=0,
        freq=base.LOAD_FREQ,
        embed="timeF",
        checkpoints="./checkpoints_test5_v3/",
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
    )

    args.results_root = "./results/"
    args.load_weight_path = None

    if not args.is_training and LOAD_FROM_OPTUNA:
        args = _apply_optuna_artifacts(args)
        selected_weather_source = getattr(args, "weather_source", selected_weather_source)
        selected_weather_h5_specs = getattr(args, "weather_h5_specs", selected_weather_h5_specs)

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

        model = FullMapConvTimeXerSimilarTokenQuantile(args, quantiles=args.quantiles).float().to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Weather-token + similar-day-token early-fusion total params: {total_params:,}")
        print(f"Weather-token + similar-day-token early-fusion trainable params: {trainable_params:,}")

        setting = _get_setting(args)
        if args.is_training:
            print(f"\n>>> Start training {setting}")
            model = train_quantile_model(model, args, device, weather_store)

            print(f"\n>>> Start testing {setting}")
            results_dir = test_quantile_model(model, args, device, weather_store)
        else:
            ckpt_path = getattr(args, "load_weight_path", None)
            if ckpt_path is None:
                ckpt_path = os.path.join(args.checkpoints, setting, "checkpoint.pth")
            if os.path.exists(ckpt_path):
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                print(f"Loaded model: {ckpt_path}")
            else:
                raise FileNotFoundError(
                    f"Model file not found: {ckpt_path}. Please set TRAIN_MODE = True first."
                )

            print(f"\n>>> Test only {setting}")
            results_dir = test_quantile_model(model, args, device, weather_store)

        plot_pred_vs_true(
            results_dir,
            use_inverse=args.inverse_eval,
            quantiles=args.quantiles,
            title_prefix="Weather + Similar-Day Token Early-Fusion Prediction",
            y_label="Load (MW)",
        )

        similar_day_result = export_similar_day_baseline(
            results_dir=results_dir,
            future_path=getattr(args, "future_path", base.FUTURE_PATH),
            args=args,
            artifact_dir=getattr(args, "similar_day_artifact_dir", SIMILAR_DAY_ARTIFACT_DIR),
            top_k=int(getattr(args, "similar_day_top_k", SIMILAR_DAY_TOP_K)),
        )
        predict_future_load_from_csv(
            model=model,
            args=args,
            device=device,
            weather_store=weather_store,
            results_dir=results_dir,
            future_path=getattr(args, "future_path", base.FUTURE_PATH),
            steps=args.pred_len,
            use_inverse=args.inverse_eval,
            quantiles=args.quantiles,
            data_provider_fn=weather_data_provider,
            model_label="Weather + Similar-Day Token Early-Fusion",
            y_label="Load (MW)",
            similar_day_result=similar_day_result,
        )
    finally:
        weather_store.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
