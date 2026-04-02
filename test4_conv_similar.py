"""
test5_smpv4.py - TimeXer primary forecast + similar-day prior correction gate

This version adds two-stage fine-tuning:
1. Load the tuned Full-Map Conv + TimeXer backbone from Optuna artifacts.
2. Freeze the backbone and train the similar-day gate + quantile head first.
3. Unfreeze the full model and run low-LR joint fine-tuning.

Compared with test5_smp.py:
1. TimeXer predicts the absolute load directly.
2. The weighted similar-day prior is converted into a bounded correction direction:
   gap = prior_mean - timexer_pred
3. A sigmoid gate beta uses model/prior agreement and prior spread to decide
   how much prior correction to accept:
   y_hat = timexer_pred + beta * gap
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from models.TimeXer import Model as TimeXer
import test4_base as base
from utils.forecast_visualization import plot_pred_vs_true, predict_future_load_from_csv
from utils.metrics import metric
from utils.quantile import QuantileLoss
from utils.tools import EarlyStopping, adjust_learning_rate
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
# 门控决策本质上是简单的标量回归，不需要大容量网络，过大容易过拟合。
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


TRAIN_MODE = base.TRAIN_MODE
ENABLE_TWO_STAGE_FINETUNE = True
# In v4 this flag controls training-time Optuna backbone initialization.
# Test-only mode still loads the local fine-tuned checkpoint by default.
LOAD_FROM_OPTUNA = True
OPTUNA_DIR = "./optuna"
OPTUNA_BEST_PARAMS_FILE = "best_params3.json"
OPTUNA_BEST_CONFIG_FILE = "best_config3.json"
OPTUNA_BEST_WEIGHT_FILE = "best_model3.pth"
OPTUNA_BEST_TRIAL_FILE = "best_trial_result3.json"

STAGE1_EPOCHS = 10
STAGE1_PATIENCE = 3
STAGE1_GATE_LR = 1e-3
STAGE1_HEAD_LR = 2e-4

STAGE2_EPOCHS = 15
STAGE2_PATIENCE = 5
STAGE2_BACKBONE_LR = 2e-5
STAGE2_GATE_LR_SCALE = 5.0
STAGE2_USE_COSINE_LR = True

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

OPTUNA_CONFIG_SKIP_KEYS = {
    "is_training",
    "checkpoints",
    "results_root",
    "load_weight_path",
    "des",
    "itr",
    "model_id",
    "root_path",
    "data_path",
    "future_path",
}
BACKBONE_LOAD_PREFIXES = (
    "weather_backbone.",
    "timexer.",
    "quantile_head.",
)
STAGE1_TRAINABLE_PREFIXES = (
    "similar_day_gate.",
    "quantile_head.",
)
STAGE2_FAST_PREFIXES = (
    "similar_day_gate.",
    "quantile_head.",
)


def _load_json_file(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _apply_optuna_payload_to_args(
    args: argparse.Namespace,
    payload: Dict[str, Any],
    skip_keys: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    skip = set(skip_keys or ())
    for raw_key, value in payload.items():
        key = TUNABLE_PARAM_MAP.get(str(raw_key), str(raw_key))
        if key in skip:
            continue
        setattr(args, key, value)
    return args


def _apply_optuna_backbone_config(args: argparse.Namespace) -> argparse.Namespace:
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
        _apply_optuna_payload_to_args(args, payload, skip_keys=OPTUNA_CONFIG_SKIP_KEYS)
    elif os.path.exists(params_path):
        payload = _load_json_file(params_path)
        if not isinstance(payload, dict):
            raise ValueError(f"./optuna params file must be a JSON object: {params_path}")
        _apply_optuna_payload_to_args(args, payload)
    else:
        raise FileNotFoundError(
            f"./optuna missing both {OPTUNA_BEST_CONFIG_FILE} and {OPTUNA_BEST_PARAMS_FILE}"
        )

    args.quantiles = list(args.quantiles)
    args.n_quantiles = len(args.quantiles)
    args.optuna_backbone_weight_path = weight_path
    print(f"Loaded Optuna backbone config from ./optuna: {weight_path}")
    return args


def _load_state_dict_file(weight_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    state = torch.load(weight_path, map_location=device)
    if isinstance(state, dict):
        for candidate_key in ("state_dict", "model_state_dict", "model"):
            candidate = state.get(candidate_key)
            if isinstance(candidate, dict):
                state = candidate
                break

    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: expected dict, got {type(state)}")

    normalized_state: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        clean_key = str(key)
        if clean_key.startswith("module."):
            clean_key = clean_key[len("module.") :]
        normalized_state[clean_key] = value
    return normalized_state


def _load_backbone_from_optuna(
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    weight_path = getattr(args, "optuna_backbone_weight_path", None)
    if weight_path is None:
        raise RuntimeError("Optuna backbone weight path is not configured.")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Optuna backbone weight not found: {weight_path}")

    source_state = _load_state_dict_file(weight_path, device)
    model_state = model.state_dict()

    matched_keys: List[str] = []
    shape_mismatch_keys: List[str] = []
    skipped_new_keys: List[str] = []

    filtered_state: Dict[str, torch.Tensor] = {}
    for key, target_tensor in model_state.items():
        if not key.startswith(BACKBONE_LOAD_PREFIXES):
            continue
        source_tensor = source_state.get(key)
        if source_tensor is None:
            skipped_new_keys.append(key)
            continue
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            shape_mismatch_keys.append(
                f"{key}: source{tuple(source_tensor.shape)} != target{tuple(target_tensor.shape)}"
            )
            continue
        filtered_state[key] = source_tensor
        matched_keys.append(key)

    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
    missing_keys = [key for key in missing_keys if key.startswith(BACKBONE_LOAD_PREFIXES)]
    unexpected_keys = [key for key in unexpected_keys if key.startswith(BACKBONE_LOAD_PREFIXES)]

    print(f"[two-stage] Loaded Optuna backbone weights from: {weight_path}")
    print(f"[two-stage]   matched keys: {len(matched_keys)}")
    print(f"[two-stage]   missing backbone keys: {len(missing_keys)}")
    print(f"[two-stage]   new module keys skipped: {len(skipped_new_keys)}")
    print(f"[two-stage]   shape mismatches: {len(shape_mismatch_keys)}")
    if unexpected_keys:
        print(f"[two-stage]   unexpected checkpoint keys ignored: {len(unexpected_keys)}")
    if shape_mismatch_keys:
        for item in shape_mismatch_keys[:8]:
            print(f"[two-stage]     shape mismatch: {item}")


def _freeze_backbone_for_stage1(model: FullMapConvTimeXerPriorCorrectionGateQuantile) -> None:
    if model.similar_day_gate is None:
        raise RuntimeError("Two-stage fine-tuning requires similar_day_gate to be enabled.")

    for param in model.parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if name.startswith(STAGE1_TRAINABLE_PREFIXES):
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"[two-stage] Stage 1 trainable params: {trainable_params:,} / {total_params:,} "
        f"({100.0 * trainable_params / max(total_params, 1):.4f}%)"
    )


def _apply_stage1_train_mode(model: FullMapConvTimeXerPriorCorrectionGateQuantile) -> None:
    model.weather_backbone.eval()
    model.timexer.eval()
    if model.similar_day_gate is not None:
        model.similar_day_gate.train()
    model.quantile_head.train()


def _unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def _build_stage1_optimizer(
    model: FullMapConvTimeXerPriorCorrectionGateQuantile,
    args: argparse.Namespace,
) -> optim.Optimizer:
    param_groups = []
    if model.similar_day_gate is not None:
        gate_params = [p for p in model.similar_day_gate.parameters() if p.requires_grad]
        if gate_params:
            param_groups.append({"params": gate_params, "lr": args.stage1_gate_lr})

    head_params = [p for p in model.quantile_head.parameters() if p.requires_grad]
    if head_params:
        param_groups.append({"params": head_params, "lr": args.stage1_head_lr})

    if not param_groups:
        raise RuntimeError("Stage 1 optimizer has no trainable parameters.")
    return optim.Adam(param_groups)


def _build_stage2_optimizer(
    model: nn.Module,
    args: argparse.Namespace,
) -> Tuple[optim.Optimizer, List[float]]:
    backbone_params = []
    fast_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(STAGE2_FAST_PREFIXES):
            fast_params.append(param)
        else:
            backbone_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.stage2_backbone_lr})
    if fast_params:
        param_groups.append(
            {"params": fast_params, "lr": args.stage2_backbone_lr * args.stage2_gate_lr_scale}
        )

    if not param_groups:
        raise RuntimeError("Stage 2 optimizer has no trainable parameters.")

    optimizer = optim.Adam(param_groups)
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    return optimizer, base_lrs


def _apply_cosine_lr_schedule(
    optimizer: optim.Optimizer,
    base_lrs: Sequence[float],
    epoch: int,
    total_epochs: int,
) -> None:
    if total_epochs <= 0:
        return
    factor = 0.5 * (1.0 + np.cos(np.pi * float(epoch) / float(total_epochs)))
    updated_lrs = []
    for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
        lr = float(base_lr) * float(factor)
        param_group["lr"] = lr
        updated_lrs.append(lr)
    print(f"[Stage2] Updated learning rates: {updated_lrs}")


def _prepare_model_batch(
    batch: Sequence[torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    use_non_blocking: bool,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
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
    batch_weather_frames = _to_float_device(
        batch_weather_frames, device, non_blocking=use_non_blocking
    )
    batch_weather_index = _to_long_device(
        batch_weather_index, device, non_blocking=use_non_blocking
    )
    if similar_day_prior is not None:
        similar_day_prior = _to_float_device(
            similar_day_prior, device, non_blocking=use_non_blocking
        )

    model_kwargs: Dict[str, torch.Tensor] = {
        "load_x": batch_x,
        "x_mark_enc": batch_x_mark,
        "x_exo_mark": batch_exo_mark,
        "weather_x": batch_weather_frames,
        "weather_x_index": batch_weather_index,
    }
    if similar_day_prior is not None:
        model_kwargs["similar_day_prior"] = similar_day_prior

    batch_y_target = extract_target(batch_y[:, -args.pred_len :, :])
    return model_kwargs, batch_y_target


def _run_train_epoch(
    model: nn.Module,
    data_loader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
    use_non_blocking: bool,
    epoch: int,
    stage_label: str,
    train_mode_callback=None,
) -> Tuple[float, float]:
    model.train()
    if train_mode_callback is not None:
        train_mode_callback(model)

    train_loss = []
    epoch_time = time.time()

    for i, batch in enumerate(data_loader):
        optimizer.zero_grad(set_to_none=True)
        model_kwargs, batch_y_target = _prepare_model_batch(
            batch, args, device, use_non_blocking
        )

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(**model_kwargs)
            loss = criterion(outputs, batch_y_target)

        train_loss.append(loss.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if (i + 1) % 50 == 0:
            print(f"[{stage_label}] iters: {i + 1}, epoch: {epoch} | loss: {loss.item():.7f}")

    train_loss_avg = float(np.average(train_loss)) if train_loss else np.nan
    epoch_cost = time.time() - epoch_time
    return train_loss_avg, epoch_cost


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
            gate_init_beta = float(getattr(configs, "similar_day_gate_init_beta", 0.1))
            gate_init_beta = min(max(gate_init_beta, 1e-3), 1.0 - 1e-3)             # 截断以避免 log(0) 问题
            gate_bias = float(np.log(gate_init_beta / (1.0 - gate_init_beta)))      # 反解 Sigmoid：bias = ln(beta / (1 - beta))

            # 构建门控多层感知机 (MLP)
            # 门控输入精简为 3 维（去除冗余特征，避免梯度竞争）:
            #   1维: prior_mean       (相似日先验曲线的平均值)
            #   1维: gap              (先验差异，必须 detach 切断梯度回流)
            #   1维: prior_spread     (先验离散度：Top-K 相似日之间的标准差)
            # 说明：timexer_pred 与 gap 线性相关无需重复；abs(gap) 信息已隐含在 gap 中；
            #       similar_day_prior[:,0] 就是 prior_mean。
            self.similar_day_gate = nn.Sequential(
                nn.Linear(3, gate_hidden_dim),
                nn.GELU(),
                nn.Dropout(float(getattr(configs, "dropout", 0.1))),
                nn.Linear(gate_hidden_dim, 1),
                nn.Sigmoid(),
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

            # --- 4. 差距度量表与动态beta加权融合 ---
            # 门控网络核心物理学视角下的 "纠偏方向向量 (gap)"，目标是从 timexer_pred 出发指向 prior_mean 的修正幅度向量
            gap = prior_mean - timexer_pred
            
            # 聚合门控判定线索（精简为 3 维，全部对 TimeXer 主干切断梯度）
            # 核心原则：门控是"旁观者"，它根据先验值、gap 大小和先验自信度来决定采纳比例，
            #           但不允许通过门控反向传播去修改 TimeXer 主干的预测输出。
            #           否则 TimeXer 会收到"竞争性梯度"信号，导致 loss 居高不下。
            gate_input = torch.cat(
                [
                    prior_mean,         # [B, L, 1] 相似日先验综合均值基线（来自外部数据，无梯度问题）
                    gap.detach(),       # [B, L, 1] 纠偏方向（必须 detach！切断门控到 TimeXer 的梯度回流）
                    prior_spread,       # [B, L, 1] Top-K 自身的不一致性（先验信噪系数指标）
                ],
                dim=-1,
            )
            
            # 使用门控多层感知机及末端 Sigmoid 推理出动态采纳比例 beta ∈ (0, 1)
            # β -> 0: 说明模型认定没必要大调，充分采信 TimeXer 主支结果，先验权当参考。
            # β -> 1: 说明模型认定需要大幅度吸纳外挂库相似日经验，做强制的拉平纠正预估（多见于气候巨变或非标事件如节假日异常断层）。
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
            model_kwargs, batch_y_target = _prepare_model_batch(
                batch, args, device, use_non_blocking
            )
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(**model_kwargs)
                loss = criterion(outputs, batch_y_target)
            total_loss.append(loss.item())

    model.train()
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
    print("Start training TimeXer-primary + similar-day prior-correction quantile model")
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
        train_loss_avg, epoch_cost = _run_train_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            args=args,
            device=device,
            use_amp=use_amp,
            use_non_blocking=use_non_blocking,
            epoch=epoch + 1,
            stage_label="Train",
        )
        vali_loss = validate_quantile(model, vali_loader, criterion, args, device, use_amp=use_amp)
        test_loss = validate_quantile(model, test_loader, criterion, args, device, use_amp=use_amp)
        print(
            f"Epoch: {epoch + 1} cost time: {epoch_cost:.1f}s | "
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


def train_two_stage_model(model, args, device, weather_store: WeatherGridStore):
    _, train_loader = weather_data_provider(args, "train", weather_store)
    _, vali_loader = weather_data_provider(args, "val", weather_store)
    _, test_loader = weather_data_provider(args, "test", weather_store)

    setting = _get_setting(args)
    path = os.path.join(args.checkpoints, setting)
    stage1_path = os.path.join(path, "stage1")
    stage2_path = os.path.join(path, "stage2")
    final_checkpoint_path = os.path.join(path, "checkpoint.pth")
    os.makedirs(stage1_path, exist_ok=True)
    os.makedirs(stage2_path, exist_ok=True)

    criterion = QuantileLoss(args.quantiles).to(device)
    use_amp = bool(getattr(args, "use_amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    use_non_blocking = _use_non_blocking_transfer(args, device)

    print("\n" + "=" * 72)
    print("Start two-stage fine-tuning for TimeXer-primary + prior-correction model")
    print(f"setting: {setting}")
    print(f"optuna_backbone: {getattr(args, 'optuna_backbone_weight_path', None)}")
    print(
        f"stage1: epochs={args.stage1_epochs}, patience={args.stage1_patience}, "
        f"gate_lr={args.stage1_gate_lr}, head_lr={args.stage1_head_lr}"
    )
    print(
        f"stage2: epochs={args.stage2_epochs}, patience={args.stage2_patience}, "
        f"backbone_lr={args.stage2_backbone_lr}, "
        f"fast_lr_scale={args.stage2_gate_lr_scale}, "
        f"cosine={bool(getattr(args, 'stage2_use_cosine_lr', False))}"
    )
    print("=" * 72)

    print("\n" + "=" * 72)
    print("STAGE 1: Freeze backbone, train similar_day_gate + quantile_head")
    print("=" * 72)
    _freeze_backbone_for_stage1(model)
    stage1_optimizer = _build_stage1_optimizer(model, args)
    stage1_early_stopping = EarlyStopping(patience=args.stage1_patience, verbose=True)

    for epoch in range(args.stage1_epochs):
        train_loss_avg, epoch_cost = _run_train_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=stage1_optimizer,
            criterion=criterion,
            scaler=scaler,
            args=args,
            device=device,
            use_amp=use_amp,
            use_non_blocking=use_non_blocking,
            epoch=epoch + 1,
            stage_label="Stage1",
            train_mode_callback=_apply_stage1_train_mode,
        )
        vali_loss = validate_quantile(model, vali_loader, criterion, args, device, use_amp=use_amp)
        test_loss = validate_quantile(model, test_loader, criterion, args, device, use_amp=use_amp)
        print(
            f"[Stage1] Epoch: {epoch + 1} cost time: {epoch_cost:.1f}s | "
            f"Train: {train_loss_avg:.7f} Vali: {vali_loss:.7f} Test: {test_loss:.7f}"
        )

        stage1_early_stopping(vali_loss, model, stage1_path)
        if stage1_early_stopping.early_stop:
            print("[Stage1] Early stopping")
            break

    stage1_ckpt_path = os.path.join(stage1_path, "checkpoint.pth")
    if not os.path.exists(stage1_ckpt_path):
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {stage1_ckpt_path}")
    model.load_state_dict(torch.load(stage1_ckpt_path, map_location=device))
    stage1_best_val = float(stage1_early_stopping.val_loss_min)
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"[Stage1] Loaded best checkpoint: {stage1_ckpt_path}")
    print(f"[Stage1] Best vali loss: {stage1_best_val:.7f}")

    print("\n" + "=" * 72)
    print("STAGE 2: Unfreeze all, run low-LR joint fine-tuning")
    print("=" * 72)
    _unfreeze_all(model)
    stage2_optimizer, stage2_base_lrs = _build_stage2_optimizer(model, args)
    stage2_early_stopping = EarlyStopping(patience=args.stage2_patience, verbose=True)
    best_overall_val = stage1_best_val

    for epoch in range(args.stage2_epochs):
        train_loss_avg, epoch_cost = _run_train_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=stage2_optimizer,
            criterion=criterion,
            scaler=scaler,
            args=args,
            device=device,
            use_amp=use_amp,
            use_non_blocking=use_non_blocking,
            epoch=epoch + 1,
            stage_label="Stage2",
        )
        vali_loss = validate_quantile(model, vali_loader, criterion, args, device, use_amp=use_amp)
        test_loss = validate_quantile(model, test_loader, criterion, args, device, use_amp=use_amp)
        print(
            f"[Stage2] Epoch: {epoch + 1} cost time: {epoch_cost:.1f}s | "
            f"Train: {train_loss_avg:.7f} Vali: {vali_loss:.7f} Test: {test_loss:.7f}"
        )

        stage2_early_stopping(vali_loss, model, stage2_path)
        if np.isfinite(vali_loss) and float(vali_loss) < float(best_overall_val):
            best_overall_val = float(vali_loss)
            torch.save(model.state_dict(), final_checkpoint_path)
            print(f"[Stage2] New overall best checkpoint saved: {final_checkpoint_path}")
        if stage2_early_stopping.early_stop:
            print("[Stage2] Early stopping")
            break
        if bool(getattr(args, "stage2_use_cosine_lr", False)):
            _apply_cosine_lr_schedule(
                stage2_optimizer,
                stage2_base_lrs,
                epoch=epoch + 1,
                total_epochs=args.stage2_epochs,
            )

    stage2_ckpt_path = os.path.join(stage2_path, "checkpoint.pth")
    if os.path.exists(stage2_ckpt_path):
        print(f"[Stage2] Best stage checkpoint: {stage2_ckpt_path}")
        print(f"[Stage2] Best vali loss: {float(stage2_early_stopping.val_loss_min):.7f}")

    model.load_state_dict(torch.load(final_checkpoint_path, map_location=device))
    print(f"[two-stage] Loaded overall best model: {final_checkpoint_path}")
    print(f"[two-stage] Overall best vali loss: {best_overall_val:.7f}")
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
            model_kwargs, batch_y_target = _prepare_model_batch(
                batch, args, device, use_non_blocking
            )
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(**model_kwargs)
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
        for qi in range(args.n_quantiles):
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
    stage_signature = ""
    if bool(getattr(args, "enable_two_stage_finetune", False)):
        stage_signature = (
            f"_ts1_s1e{int(getattr(args, 'stage1_epochs', 0))}"
            f"_s1g{float(getattr(args, 'stage1_gate_lr', 0.0))}"
            f"_s1h{float(getattr(args, 'stage1_head_lr', 0.0))}"
            f"_s2e{int(getattr(args, 'stage2_epochs', 0))}"
            f"_s2b{float(getattr(args, 'stage2_backbone_lr', 0.0))}"
            f"_s2x{float(getattr(args, 'stage2_gate_lr_scale', 0.0))}"
        )
    signature = (
        f"{args.task_name}_{args.model_id}_{args.model}_e2e_sdv4_"
        f"sl{args.seq_len}_pl{args.pred_len}_dm{args.d_model}_"
        f"el{args.e_layers}_wd{args.weather_feature_dim}_"
        f"wsl{args.weather_seq_len}_wh{args.weather_history_len}_"
        f"wk{args.weather_kernel_height}x{args.weather_kernel_width}_"
        f"sdp{int(bool(getattr(args, 'use_similar_day_prior', False)))}_"
        f"sdk{int(getattr(args, 'similar_day_top_k', 0))}_"
        f"sdgh{int(getattr(args, 'similar_day_gate_hidden_dim', 0))}_"
        f"sdga{int(round(1000.0 * float(getattr(args, 'similar_day_gate_init_beta', 0.0))))}_"
        f"lr{args.learning_rate}_bs{args.batch_size}{stage_signature}_{args.des}_{itr}"
    )
    digest = hashlib.md5(signature.encode("utf-8")).hexdigest()[:8]
    return (
        f"sdv4_sl{args.seq_len}_pl{args.pred_len}_"
        f"wd{args.weather_feature_dim}_sdk{int(getattr(args, 'similar_day_top_k', 0))}_"
        f"ts{int(bool(getattr(args, 'enable_two_stage_finetune', False)))}_"
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
        model_id=f"{base.MODEL_ID_PREFIX}_sdv4",
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
        checkpoints="./checkpoints_test5_v4/",
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
        enable_two_stage_finetune=ENABLE_TWO_STAGE_FINETUNE,
        stage1_epochs=STAGE1_EPOCHS,
        stage1_patience=STAGE1_PATIENCE,
        stage1_gate_lr=STAGE1_GATE_LR,
        stage1_head_lr=STAGE1_HEAD_LR,
        stage2_epochs=STAGE2_EPOCHS,
        stage2_patience=STAGE2_PATIENCE,
        stage2_backbone_lr=STAGE2_BACKBONE_LR,
        stage2_gate_lr_scale=STAGE2_GATE_LR_SCALE,
        stage2_use_cosine_lr=STAGE2_USE_COSINE_LR,
    )

    args.results_root = "./results/"
    args.load_weight_path = None
    args.optuna_backbone_weight_path = None

    if LOAD_FROM_OPTUNA:
        args = _apply_optuna_backbone_config(args)
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

        model = FullMapConvTimeXerPriorCorrectionGateQuantile(args, quantiles=args.quantiles).float().to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"TimeXer-primary + prior-correction total params: {total_params:,}")
        print(f"TimeXer-primary + prior-correction trainable params: {trainable_params:,}")

        setting = _get_setting(args)
        if args.is_training:
            if LOAD_FROM_OPTUNA:
                print("\n>>> Load Optuna backbone before training")
                _load_backbone_from_optuna(model, args, device)

            if bool(getattr(args, "enable_two_stage_finetune", False)):
                print(f"\n>>> Start two-stage training {setting}")
                model = train_two_stage_model(model, args, device, weather_store)
            else:
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
            title_prefix="TimeXer-Primary + Similar-Day Prior-Correction Prediction",
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
            model_label="TimeXer-Primary + Similar-Day Prior-Correction",
            y_label="Load (MW)",
            similar_day_result=similar_day_result,
        )
    finally:
        weather_store.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
