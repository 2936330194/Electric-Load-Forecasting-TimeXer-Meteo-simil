"""
test4_conv_similar.py - Optuna超参数优化 + 外生气象变量 + TimeXer 基础预测 + 相似日先验修正门控系统

此版本引入了两阶段微调（Two-stage Fine-tuning）策略：
1. 模型加载：从 Optuna 实验记录中加载已调优的 Full-Map Conv + TimeXer 骨干网络权重。
2. 阶段一（预热）：冻结骨干网络，优先训练新增的相似日门控单元（Gate）和量化回归输出头。
3. 阶段二（联合微调）：解冻全模型参数，以极低的学习率进行全量联合微调，实现特征深度适配。

核心思路：
1. 预测逻辑：TimeXer 骨干网络直接预测未来的电力负荷绝对值。
2. 纠偏机制：将加权的相似日先验转化为一个有界的修正方向（Gap）：
   gap = prior_mean (先验均值) - timexer_pred (模型原始预测)
3. 动态融合：引入 Sigmoid 门控系数 beta，根据模型预测与先验的一致性以及先验自身的离散度（Spread）
   动态决定纠偏力度：
   最终输出 y_hat = timexer_pred + beta * gap
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
from utils.metrics import cal_eval, metric
from utils.quantile import QuantileLoss
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.weather_e2e import FullMapWeatherConvExtractor, WeatherGridStore, weather_data_provider


# ================= 相似日检索模块与 TimeXer 主体纠偏门控方案的专属参数配置 =================
SIMILAR_DAY_ARTIFACT_DIR: Optional[str] = None

# 在检索历史中最相近日期的天气时，选取前 K(这里是3) 个最相似的日期来生成先验负荷曲线。
SIMILAR_DAY_TOP_K = 3

# 这是一个总开关，决定接下来的测试或训练环节中，是否要启用并组装这条相似日先验特征进入端到端模型。
USE_SIMILAR_DAY_PRIOR = True

# 针对先验纠偏门控 (Dynamic Prior-Correction Gating) 的隐藏层尺寸参数。
SIMILAR_DAY_GATE_HIDDEN_DIM = 64

# 网络初始化时给先验纠偏比例的权重锚点。
# 0.1 表示初始时仅采纳 10% 的相似日纠偏，让 TimeXer 先以主体预测稳定起步，随后网络再通过反向传播自动调节先验的介入比例。
SIMILAR_DAY_GATE_INIT_BETA = 0.10


# ================= 从基础实验模块导入常用工具函数 =================
# 为了保持代码简洁并保证逻辑与基础版本严格对齐，这里大量借用了 test4_smp.py (下称 base) 中写好的底层支持函数。
_use_non_blocking_transfer = base._use_non_blocking_transfer  
_to_float_device = base._to_float_device                      
_to_long_device = base._to_long_device                        
extract_target = base.extract_target                          
_parse_cli_args = base._parse_cli_args                        
_resolve_weather_h5_specs = base._resolve_weather_h5_specs    
_configure_runtime_weather_args = base._configure_runtime_weather_args 
export_similar_day_baseline = base.export_similar_day_baseline


# ================= 训练模式与 Optuna 预训练骨干加载配置 =================
TRAIN_MODE = base.TRAIN_MODE                          
ENABLE_TWO_STAGE_FINETUNE = True                      
LOAD_FROM_OPTUNA = True                               
OPTUNA_DIR = "./optuna_15min_7_1"
OPTUNA_BEST_PARAMS_FILE = "best_params_fullmap.json"
OPTUNA_BEST_CONFIG_FILE = "best_config_fullmap.json"
OPTUNA_BEST_WEIGHT_FILE = "best_model_fullmap.pth"
OPTUNA_BEST_TRIAL_FILE = "best_trial_result_fullmap.json"

# ================= 第一阶段微调超参数（冻结骨干，仅训练门控 + 分位数头）=================
STAGE1_EPOCHS = 15           # 第一阶段最大训练轮数
STAGE1_PATIENCE = 4          # 第一阶段的早停耐心值（连续4轮验证集无改善即停止）
STAGE1_GATE_LR = 7e-4        # 第一阶段中相似日门控网络的学习率（稍低，减少门控震荡）
STAGE1_HEAD_LR = 1e-4        # 第一阶段中分位数预测头的学习率（更保守地校准预测头）

# ================= 第二阶段微调超参数（解冻全部参数，低学习率联合精调）=================
STAGE2_EPOCHS = 20           # 第二阶段最大训练轮数
STAGE2_PATIENCE = 6          # 第二阶段的早停耐心值（略宽松，允许更充分的联合收敛）
STAGE2_BACKBONE_LR = 1e-5    # 第二阶段中骨干网络（CNN + TimeXer）的基础学习率（更低，降低破坏预训练特征的风险）
STAGE2_GATE_LR_SCALE = 10.0  # 第二阶段中门控/预测头的学习率倍率（保持门控/预测头约 1e-4 的有效学习率）
STAGE2_USE_COSINE_LR = True  # 第二阶段是否启用余弦退火学习率调度（平滑衰减，避免末期震荡）

# ================= Optuna 超参数键名映射表 =================
# 将 Optuna JSON 产物中使用的大写键名映射到本脚本 argparse.Namespace 中的小写属性名。
# 例如 Optuna 保存的 "D_MODEL" 对应到 args.d_model，确保无缝加载。
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

# ================= 从 Optuna 配置中需要跳过的键集合 =================
# 这些键属于运行时路径/标识类设置，不应被 Optuna 调优结果覆盖，
# 否则会导致检查点路径、数据路径等被错误篡改。
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

# ================= 模型参数前缀过滤规则 =================
# BACKBONE_LOAD_PREFIXES: 从 Optuna 权重文件中加载时，只匹配这些前缀的参数。
#   即气象 CNN、TimeXer 主干和分位数头的权重会被加载，门控网络参数会被跳过（因为是新增模块）。
BACKBONE_LOAD_PREFIXES = (
    "weather_backbone.",
    "timexer.",
    "quantile_head.",
)
# STAGE1_TRAINABLE_PREFIXES: 第一阶段中仅解冻这些前缀对应的参数，其余全部冻结。
#   只训练门控网络和分位数预测头，骨干网络保持 Optuna 预训练状态不动。
STAGE1_TRAINABLE_PREFIXES = (
    "similar_day_gate.",
    "quantile_head.",
)
# STAGE2_FAST_PREFIXES: 第二阶段中这些前缀的参数使用更高的学习率（加速收敛），
#   而骨干网络使用极低学习率进行保守微调。
STAGE2_FAST_PREFIXES = (
    "similar_day_gate.",
    "quantile_head.",
)


def _load_json_file(json_path: str):
    """从指定路径读取 JSON 文件并返回解析后的 Python 对象。"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _apply_optuna_payload_to_args(
    args: argparse.Namespace,
    payload: Dict[str, Any],
    skip_keys: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    """
    将 Optuna 导出的超参数字典逐项写入 argparse.Namespace 对象。
    
    工作流程：
    1. 遍历 payload 中的每个键值对；
    2. 通过 TUNABLE_PARAM_MAP 将 Optuna 使用的键名转换为本脚本的属性名；
    3. 跳过 skip_keys 中指定的保护键（如路径类配置）；
    4. 使用 setattr 动态设置 args 的属性值。
    """
    skip = set(skip_keys or ())
    for raw_key, value in payload.items():
        key = TUNABLE_PARAM_MAP.get(str(raw_key), str(raw_key))
        if key in skip:
            continue
        setattr(args, key, value)
    return args


def _apply_optuna_backbone_config(args: argparse.Namespace) -> argparse.Namespace:
    """
    从 Optuna 最优试验的产物目录中加载骨干网络的配置参数。
    
    加载优先级：best_config3.json（完整配置） > best_params3.json（仅超参数）。
    加载完成后，将 Optuna 最优权重文件路径写入 args.optuna_backbone_weight_path，
    供后续 _load_backbone_from_optuna() 函数使用。
    """
    optuna_dir = os.path.abspath(OPTUNA_DIR)
    config_path = os.path.join(optuna_dir, OPTUNA_BEST_CONFIG_FILE)   # 完整配置文件路径
    params_path = os.path.join(optuna_dir, OPTUNA_BEST_PARAMS_FILE)   # 超参数文件路径
    weight_path = os.path.join(optuna_dir, OPTUNA_BEST_WEIGHT_FILE)   # 模型权重文件路径

    # 权重文件是必须存在的，否则后续的骨干加载无法进行
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"./optuna model weight file not found: {weight_path}")

    # 优先从完整配置文件加载（包含所有 args 属性），回退到仅超参数文件
    if os.path.exists(config_path):
        payload = _load_json_file(config_path)
        if not isinstance(payload, dict):
            raise ValueError(f"./optuna config file must be a JSON object: {config_path}")
        # 使用跳过键集合，防止路径类配置被 Optuna 产物覆盖
        _apply_optuna_payload_to_args(args, payload, skip_keys=OPTUNA_CONFIG_SKIP_KEYS)
    elif os.path.exists(params_path):
        payload = _load_json_file(params_path)
        if not isinstance(payload, dict):
            raise ValueError(f"./optuna params file must be a JSON object: {params_path}")
        # 仅超参数文件没有路径类键，无需跳过
        _apply_optuna_payload_to_args(args, payload)
    else:
        raise FileNotFoundError(
            f"./optuna missing both {OPTUNA_BEST_CONFIG_FILE} and {OPTUNA_BEST_PARAMS_FILE}"
        )

    # 确保分位数相关属性与配置同步
    args.quantiles = list(args.quantiles)
    args.n_quantiles = len(args.quantiles)
    # 记录权重路径，供后续骨干权重加载使用
    args.optuna_backbone_weight_path = weight_path
    print(f"Loaded Optuna backbone config from ./optuna: {weight_path}")
    return args


def _load_state_dict_file(weight_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    通用的模型权重文件加载器，支持多种 checkpoint 封装格式。
    
    功能：
    1. 从磁盘加载 .pth 文件到指定设备；
    2. 自动检测并解包 'state_dict'、'model_state_dict'、'model' 等常见嵌套键；
    3. 去除 DataParallel/DistributedDataParallel 遮罩的 'module.' 前缀；
    4. 过滤非 Tensor 类型的条目（如优化器状态、训练步计数等）。
    
    返回：
        归一化后的 {key: Tensor} 字典，可直接用于 model.load_state_dict()。
    """
    state = torch.load(weight_path, map_location=device)
    # 检测是否为嵌套封装格式（很多框架会将 state_dict 嵌套在外层字典中）
    if isinstance(state, dict):
        for candidate_key in ("state_dict", "model_state_dict", "model"):
            candidate = state.get(candidate_key)
            if isinstance(candidate, dict):
                state = candidate
                break

    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: expected dict, got {type(state)}")

    # 归一化处理：移除 DataParallel 产生的 'module.' 前缀，只保留 Tensor 类型的条目
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
    """
    从 Optuna 最优试验的权重文件中择性加载骨干网络参数。
    
    加载策略：
    1. 只加载 BACKBONE_LOAD_PREFIXES 指定前缀的参数（气象CNN、TimeXer、分位数头）；
    2. 跳过新增模块（如门控网络）的参数，这些参数保持随机初始化；
    3. 跳过形状不匹配的参数（可能由于架构调整引起）；
    4. 使用 strict=False 加载，允许缺失和多余的键。
    
    加载完成后会输出详细的匹配/跳过/失配统计信息。
    """
    weight_path = getattr(args, "optuna_backbone_weight_path", None)
    if weight_path is None:
        raise RuntimeError("Optuna backbone weight path is not configured.")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Optuna backbone weight not found: {weight_path}")

    # 加载并归一化源权重文件
    source_state = _load_state_dict_file(weight_path, device)
    model_state = model.state_dict()

    # 统计跟踪各类参数的匹配情况
    matched_keys: List[str] = []             # 成功匹配并加载的参数键
    shape_mismatch_keys: List[str] = []      # 键名匹配但形状不一致的参数
    skipped_new_keys: List[str] = []         # 新增模块中在源权重中找不到的参数

    # 筛选可加载的参数：只处理骨干前缀的参数，跳过新增模块和形状不匹配的参数
    filtered_state: Dict[str, torch.Tensor] = {}
    for key, target_tensor in model_state.items():
        if not key.startswith(BACKBONE_LOAD_PREFIXES):
            continue  # 跳过非骨干前缀的参数（如门控网络）
        source_tensor = source_state.get(key)
        if source_tensor is None:
            skipped_new_keys.append(key)  # 源权重中找不到该键，说明是新增的参数
            continue
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            shape_mismatch_keys.append(
                f"{key}: source{tuple(source_tensor.shape)} != target{tuple(target_tensor.shape)}"
            )
            continue  # 形状不匹配时跳过，防止加载报错
        filtered_state[key] = source_tensor
        matched_keys.append(key)

    # 使用 strict=False 部分加载：允许目标模型中存在未被加载的新增参数
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
    # 只报告骨干前缀范围内的缺失/意外键，忽略新增模块的警告
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
    """
    为第一阶段微调冻结骨干网络参数。
    
    策略：先将所有参数的 requires_grad 置为 False，
    然后仅解冻 STAGE1_TRAINABLE_PREFIXES 指定前缀的参数（门控 + 分位数头）。
    这样骨干网络在训练时不会被更新，保持 Optuna 预训练的特征提取能力。
    """
    if model.similar_day_gate is None:
        raise RuntimeError("Two-stage fine-tuning requires similar_day_gate to be enabled.")

    # 第一步：全部冻结
    for param in model.parameters():
        param.requires_grad = False

    # 第二步：选择性解冻门控和分位数头
    for name, param in model.named_parameters():
        if name.startswith(STAGE1_TRAINABLE_PREFIXES):
            param.requires_grad = True

    # 输出可训练参数的统计信息，便于确认冻结是否生效
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"[two-stage] Stage 1 trainable params: {trainable_params:,} / {total_params:,} "
        f"({100.0 * trainable_params / max(total_params, 1):.4f}%)"
    )


def _apply_stage1_train_mode(model: FullMapConvTimeXerPriorCorrectionGateQuantile) -> None:
    """
    第一阶段专用的训练模式回调函数。
    
    将骨干网络（气象 CNN 和 TimeXer）设为 eval 模式（固定 BatchNorm/Dropout），
    仅将门控和分位数头设为 train 模式。
    这样即使骨干参数已冻结，BN层的滑动统计量也不会被破坏。
    """
    model.weather_backbone.eval()   # 气象 CNN 保持评估模式
    model.timexer.eval()            # TimeXer 主干保持评估模式
    if model.similar_day_gate is not None:
        model.similar_day_gate.train()  # 门控网络开启训练模式
    model.quantile_head.train()     # 分位数头开启训练模式


def _unfreeze_all(model: nn.Module) -> None:
    """解冻模型的所有参数，用于第二阶段联合微调前的准备工作。"""
    for param in model.parameters():
        param.requires_grad = True


def _build_stage1_optimizer(
    model: FullMapConvTimeXerPriorCorrectionGateQuantile,
    args: argparse.Namespace,
) -> optim.Optimizer:
    """
    构建第一阶段专用的 Adam 优化器。
    
    使用差异化学习率策略：
    - 门控网络使用较大的学习率 (stage1_gate_lr)，以便快速学习先验采纳比例；
    - 分位数头使用较小的学习率 (stage1_head_lr)，微调即可。
    """
    param_groups = []
    # 收集门控网络的可训练参数
    if model.similar_day_gate is not None:
        gate_params = [p for p in model.similar_day_gate.parameters() if p.requires_grad]
        if gate_params:
            param_groups.append({"params": gate_params, "lr": args.stage1_gate_lr})

    # 收集分位数预测头的可训练参数
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
    """
    构建第二阶段联合微调的 Adam 优化器。
    
    使用双速学习率策略：
    - 骨干网络（CNN + TimeXer）使用极低学习率 (stage2_backbone_lr)，保守微调；
    - 门控 + 分位数头使用加速学习率 (backbone_lr * gate_lr_scale)，继续快速优化。
    
    返回：
        (optimizer, base_lrs): 优化器和各参数组的基础学习率列表（供余弦调度使用）。
    """
    backbone_params = []  # 骨干网络参数（低学习率组）
    fast_params = []      # 门控/预测头参数（高学习率组）

    # 根据参数名称前缀将参数分派到不同的学习率组
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
    # 记录各参数组的基础学习率，供余弦退火调度器计算衰减因子时使用
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    return optimizer, base_lrs


def _apply_cosine_lr_schedule(
    optimizer: optim.Optimizer,
    base_lrs: Sequence[float],
    epoch: int,
    total_epochs: int,
) -> None:
    """
    应用余弦退火学习率调度。
    
    公式：lr = base_lr * 0.5 * (1 + cos(π * epoch / total_epochs))
    效果：学习率从 base_lr 平滑衰减到接近 0，避免训练末期学习率过大引起的权重震荡。
    """
    if total_epochs <= 0:
        return
    # 计算余弦衰减因子：从 1.0 平滑下降到 0.0
    factor = 0.5 * (1.0 + np.cos(np.pi * float(epoch) / float(total_epochs)))
    updated_lrs = []
    # 对每个参数组独立应用衰减因子
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
    """
    将 DataLoader 输出的原始 batch 转换为模型前向传播所需的输入字典和目标张量。
    
    工作流程：
    1. 解包原始 batch 为各组件张量（自动兼容 6/7 元素格式）；
    2. 将所有张量转换为正确的数据类型并发送到计算设备；
    3. 组装模型输入关键字参数字典（model_kwargs）；
    4. 提取目标区间的真实负荷值作为监督信号。
    
    返回：
        (model_kwargs, batch_y_target): 模型输入参数字典和目标张量。
    """
    # 解包原始 batch（自动处理有/无相似日先验的两种情况）
    (
        batch_x,
        batch_y,
        batch_x_mark,
        batch_exo_mark,
        batch_weather_frames,
        batch_weather_index,
        similar_day_prior,
    ) = _unpack_weather_batch(batch)

    # 将各张量转换为正确数据类型并传输到计算设备（GPU/CPU）
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

    # 组装模型前向传播的关键字参数字典
    model_kwargs: Dict[str, torch.Tensor] = {
        "load_x": batch_x,                      # 历史负荷序列
        "x_mark_enc": batch_x_mark,              # 历史负荷的时间特征编码
        "x_exo_mark": batch_exo_mark,            # 外生变量的时间特征编码
        "weather_x": batch_weather_frames,       # 气象网格数据
        "weather_x_index": batch_weather_index,  # 气象数据的索引位置
    }
    # 如果存在相似日先验，也加入输入参数
    if similar_day_prior is not None:
        model_kwargs["similar_day_prior"] = similar_day_prior

    # 提取目标预测区间的真实负荷值（仅保留目标通道）
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
    """
    执行一个完整的训练 epoch，支持混合精度训练 (AMP) 和自定义训练模式回调。
    
    参数：
        model: 待训练的神经网络模型
        data_loader: 训练数据加载器
        optimizer: 优化器实例
        criterion: 损失函数（分位数损失）
        scaler: AMP 梯度缩放器
        args: 配置参数
        device: 计算设备
        use_amp: 是否启用混合精度
        use_non_blocking: 是否启用异步数据传输
        epoch: 当前 epoch 编号
        stage_label: 训练阶段标签（如 "Train"、"Stage1"、"Stage2"）
        train_mode_callback: 可选的训练模式设置回调（用于第一阶段的差异化 train/eval 设置）
    
    返回：
        (train_loss_avg, epoch_cost): 平均训练损失和 epoch 耗时（秒）
    """
    model.train()
    # 如果提供了回调，则覆盖默认的 train 模式（用于第一阶段将骨干置为 eval）
    if train_mode_callback is not None:
        train_mode_callback(model)

    train_loss = []
    epoch_time = time.time()

    for i, batch in enumerate(data_loader):
        optimizer.zero_grad(set_to_none=True)  # 清空梯度（set_to_none=True 更高效）
        # 准备模型输入和监督目标
        model_kwargs, batch_y_target = _prepare_model_batch(
            batch, args, device, use_non_blocking
        )

        # 混合精度前向传播 + 损失计算
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(**model_kwargs)
            loss = criterion(outputs, batch_y_target)

        train_loss.append(loss.item())
        # AMP 梯度缩放 + 反向传播 + 参数更新
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 每 50 个 batch 输出一次训练进度
        if (i + 1) % 50 == 0:
            print(f"[{stage_label}] iters: {i + 1}, epoch: {epoch} | loss: {loss.item():.7f}")

    # 计算平均损失和 epoch 耗时
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
        """
        将原始气象网格帧通过 CNN 骨干网络编码为低维特征向量。
        
        为了防止显存溢出，当帧数较多时采用分块 (chunk) 编码策略。
        
        参数：
            weather_frames: 形状为 [N, C, H, W] 的气象网格数据
        返回：
            形状为 [N, weather_feature_dim] 的编码后特征向量
        """
        if weather_frames.ndim != 4:
            raise ValueError(
                f"Weather frames should have shape [N, C, H, W], got {tuple(weather_frames.shape)}"
            )

        # 分块编码：每次处理 encode_chunk_size 个帧，避免大批量一次性进入 GPU 引起 OOM
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
        """
        将气象序列编码为时序特征序列，支持两种输入模式。
        
        模式 1 - 索引模式（内存高效）：
            weather_seq [U, C, H, W]: 去重后的气象帧池
            weather_index [B, T]: 每个样本每个时间步对应的帧索引
            先对帧池统一编码，再按索引查找重组。
        
        模式 2 - 序列模式（通用）：
            weather_seq [B, T, C, H, W]: 每个样本的完整气象时序
            将时序展平后统一编码，再重新组装为 batch 形状。
        
        返回：
            形状为 [B, T, weather_feature_dim] 的气象特征序列
        """
        if weather_seq is None:
            return None

        # 模式 1：索引模式 —— 气象帧池 + 时间索引映射
        if weather_index is not None:
            if weather_seq.ndim != 4 or weather_index.ndim != 2:
                raise ValueError(
                    "Indexed weather mode expects weather_seq [U, C, H, W] and weather_index [B, T]."
                )
            batch_size, time_len = weather_index.shape
            # 对去重帧池统一编码，避免重复计算相同时刻的气象特征
            encoded_frames = self._encode_weather_frames(weather_seq)
            # 按索引查找对应时刻的编码特征，重组为 [B, T, dim] 形状
            gathered = encoded_frames.index_select(0, weather_index.reshape(-1))
            return gathered.reshape(batch_size, time_len, self.weather_feature_dim)

        # 模式 2：序列模式 —— 每个样本独立的气象时序
        if weather_seq.ndim != 5:
            raise ValueError(
                f"Sequential weather mode expects [B, T, C, H, W], got {tuple(weather_seq.shape)}"
            )
        batch_size, time_len, channels, height, width = weather_seq.shape
        # 将 [B, T, C, H, W] 展平为 [B*T, C, H, W] 进行统一编码
        flat = weather_seq.reshape(batch_size * time_len, channels, height, width)
        encoded = self._encode_weather_frames(flat)
        # 重新组装为 [B, T, dim]
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
    """
    在验证/测试集上评估模型的分位数损失。
    
    使用 torch.inference_mode() 禁止梯度计算，提高推理速度并减少显存占用。
    评估完成后自动恢复模型为 train 模式。
    
    返回：
        平均分位数损失值（float）。
    """
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

    model.train()  # 评估完成后恢复训练模式
    return float(np.average(total_loss)) if total_loss else np.nan


def train_quantile_model(model, args, device, weather_store: WeatherGridStore):
    """
    常规单阶段训练流程：全参数端到端训练 + 早停机制。
    
    工作流程：
    1. 创建训练/验证/测试数据加载器；
    2. 配置 Adam 优化器、分位数损失函数、早停器；
    3. 每个 epoch 依次训练→验证→测试，并根据验证损失决定是否早停；
    4. 加载最优检查点并返回模型。
    """
    # 创建三个数据分割的 DataLoader
    _, train_loader = weather_data_provider(args, "train", weather_store)
    _, vali_loader = weather_data_provider(args, "val", weather_store)
    _, test_loader = weather_data_provider(args, "test", weather_store)

    # 创建检查点保存目录
    setting = _get_setting(args)
    path = os.path.join(args.checkpoints, setting)
    os.makedirs(path, exist_ok=True)

    # 配置基础优化器、损失函数与验证集早停监控器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = QuantileLoss(args.quantiles).to(device)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    # 判断是否开启混合精度计算 (AMP)，以降低显存占用及加快训练速度
    use_amp = bool(getattr(args, "use_amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    # 获取是否开启异步显存传输以避免 CPU->GPU 拷贝阻塞
    use_non_blocking = _use_non_blocking_transfer(args, device)

    # ================= 打印并记录当次训练的所有关键实验配置 =================
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

    # ================= 开启全量训练迭代循环 =================
    for epoch in range(args.train_epochs):
        # 1. 独立运行一个 epoch 的前向、反向及学习步骤
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
        
        # 2. 每个 epoch 结束后调用验证与测试流程 (不影响网络梯度)
        vali_loss = validate_quantile(model, vali_loader, criterion, args, device, use_amp=use_amp)
        test_loss = validate_quantile(model, test_loader, criterion, args, device, use_amp=use_amp)
        print(
            f"Epoch: {epoch + 1} cost time: {epoch_cost:.1f}s | "
            f"Train: {train_loss_avg:.7f} Vali: {vali_loss:.7f} Test: {test_loss:.7f}"
        )

        # 3. 将验证集 Loss 反馈给早停监视器并保存当前最优的模型参数
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break
            
        # 4. 根据设定策略进行学习率衰减调节
        adjust_learning_rate(optimizer, epoch + 1, args)

    # ================= 训练终结回收 =================
    # 即便早停也重新加载验证集表现最好那一轮存下的稳定权重，防止模型偏转震荡
    best_model_path = os.path.join(path, "checkpoint.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"Loaded best model weights: {best_model_path}")
    return model


def train_two_stage_model(model, args, device, weather_store: WeatherGridStore):
    """
    两阶段微调训练流程：专为预训练的骨干网络（Optuna）和新增的门控网络设计。
    
    工作流程：
    - Stage 1（微调预热）：冻结骨干网络，仅以较大学习率训练新增的门控单元和预测头，使其迅速收敛到一个合理的作用域；
    - Stage 2（全量联合微调）：解冻所有网络层，对骨干网络施加极低学习率保护，对门控网络使用加速退火学习率，全面适配任务。
    
    说明：这个策略有效避免了由于新增的随机初始化模块在初期产生极端误差从而拉垮骨干网络的问题。
    """
    # 1. 拆分准备训练、验证、测试 Dataloader
    _, train_loader = weather_data_provider(args, "train", weather_store)
    _, vali_loader = weather_data_provider(args, "val", weather_store)
    _, test_loader = weather_data_provider(args, "test", weather_store)

    # 2. 为两阶段训练独立构建隔离的检查点（Checkpoint）保存目录
    setting = _get_setting(args)
    path = os.path.join(args.checkpoints, setting)
    stage1_path = os.path.join(path, "stage1")
    stage2_path = os.path.join(path, "stage2")
    final_checkpoint_path = os.path.join(path, "checkpoint.pth")
    os.makedirs(stage1_path, exist_ok=True)
    os.makedirs(stage2_path, exist_ok=True)

    # 3. 混合精度 (AMP)、损失函数和显存异步传输的基础设定
    criterion = QuantileLoss(args.quantiles).to(device)
    use_amp = bool(getattr(args, "use_amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    use_non_blocking = _use_non_blocking_transfer(args, device)

    # 4. 打印两阶段策略运行面板信息
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
    # 调用回调冻结主干特征提取与 TimeXer 映射层的参数梯度回传
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
            train_mode_callback=_apply_stage1_train_mode,  # 第一阶段特殊 eval 强制挂载钩子，防止 BN 滑动
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

    # Stage 1 完毕：必须重新拉起验证集表现最优的阶段检查点，为联合微调 (Stage 2) 打好最优基础锚点。
    stage1_ckpt_path = os.path.join(stage1_path, "checkpoint.pth")
    if not os.path.exists(stage1_ckpt_path):
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {stage1_ckpt_path}")
    model.load_state_dict(torch.load(stage1_ckpt_path, map_location=device))
    stage1_best_val = float(stage1_early_stopping.val_loss_min)
    # 将 Stage1 产生的最好状态作为整体测试的最优垫底基线保存。
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"[Stage1] Loaded best checkpoint: {stage1_ckpt_path}")
    print(f"[Stage1] Best vali loss: {stage1_best_val:.7f}")

    print("\n" + "=" * 72)
    print("STAGE 2: Unfreeze all, run low-LR joint fine-tuning")
    print("=" * 72)
    # 解除所有参数锁，准许反向微调干预
    _unfreeze_all(model)
    # 构造带有双速学习率（骨干网极低 lr，门控组件中等 lr）参数组分类的专属 Adam 调度器
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
        # 不仅仅监控 Stage2 内最佳，一旦跨阶段超越了 Stage1 的极限，便认定并覆写全局大检查点记录
        if np.isfinite(vali_loss) and float(vali_loss) < float(best_overall_val):
            best_overall_val = float(vali_loss)
            torch.save(model.state_dict(), final_checkpoint_path)
            print(f"[Stage2] New overall best checkpoint saved: {final_checkpoint_path}")
            
        if stage2_early_stopping.early_stop:
            print("[Stage2] Early stopping")
            break
            
        # 对第二阶段采用 Cosine 余弦退火策略来规避后期过拟合波动
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

    # 二阶段全终局收尾：装载两阶段跨越验证所取得的全局历史最优网络全状态权重
    model.load_state_dict(torch.load(final_checkpoint_path, map_location=device))
    print(f"[two-stage] Loaded overall best model: {final_checkpoint_path}")
    print(f"[two-stage] Overall best vali loss: {best_overall_val:.7f}")
    return model


def test_quantile_model(model, args, device, weather_store: WeatherGridStore) -> str:
    """
    对训练好的概率型模型在不可见的测试集上进行全量推理、评估及结果落盘。
    
    工作流程：
    1. 前向推理并收集各时间步的所有预测分位数（包括 10%, 50%, 90% 区间）；
    2. 从多维预测中单独提取出 P50（中位预测，用于充当传统点预测），并与真实标签对齐；
    3. 如果在预处理时做了标准化，这里将进行彻底的逆归一化（Inverse Transform）还原成真实物理单位（如 MW）；
    4. 计算 RMSE、MAE 等传统验证指标并落盘 numpy 结果供后处理画图。
    """
    # 获取专属于不可见时间段的 Dataset 与 DataLoader
    test_data, test_loader = weather_data_provider(args, "test", weather_store)

    # 在 ./results/ 目录下，依据此次特殊超参设定创建唯一名称的文件保存目录
    setting = _get_setting(args)
    folder_path = os.path.join(getattr(args, "results_root", "./results/"), setting)
    os.makedirs(folder_path, exist_ok=True)

    # 结果缓存容器
    preds_p50 = []           # 用作传统点预测验证的中位数序列
    trues = []               # 目标真实序列 (Ground Truth)
    quantile_preds_all = []  # 包含上下界的所有分位预测的完整集合 [B, pred_len, n_quantiles]

    use_amp = bool(getattr(args, "use_amp", False)) and device.type == "cuda"
    use_non_blocking = _use_non_blocking_transfer(args, device)

    # =============== 1. 批量前向推理 ===============
    model.eval()
    with torch.inference_mode():
        for batch in test_loader:
            model_kwargs, batch_y_target = _prepare_model_batch(
                batch, args, device, use_non_blocking
            )
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(**model_kwargs)
            
            # 从 Quantile Head 输出的多通道分位结果中，按配置设定的 P50 索引抽出中心预测结果
            p50_pred = outputs.float()[:, :, base.P50_IDX : base.P50_IDX + 1]

            quantile_preds_all.append(outputs.float().detach().cpu().numpy())
            preds_p50.append(p50_pred.detach().cpu().numpy())
            trues.append(batch_y_target.detach().cpu().numpy())

    # 拼接所有批次的集合，生成连续时序全貌
    preds_p50 = np.concatenate(preds_p50, axis=0)            # 形状：[Samples, pred_len, 1]
    trues = np.concatenate(trues, axis=0)                    # 形状：[Samples, pred_len, 1]
    quantile_preds_all = np.concatenate(quantile_preds_all, axis=0)

    print(
        f"Test shape: preds={preds_p50.shape}, "
        f"trues={trues.shape}, quantiles={quantile_preds_all.shape}"
    )

    # 留存标准化空间的原始网络输出
    np.save(os.path.join(folder_path, "pred.npy"), preds_p50)
    np.save(os.path.join(folder_path, "true.npy"), trues)
    np.save(os.path.join(folder_path, "quantile_preds.npy"), quantile_preds_all)

    # =============== 2. 逆归一化（还原物理量） ===============
    if test_data.scale:
        shape = trues.shape
        # 数据集提供方通常只支持二维数组的逆归一化，所以先将时间步打平展开
        preds_inv = test_data.inverse_transform_target(preds_p50.reshape(shape[0] * shape[1], -1)).reshape(shape)
        trues_inv = test_data.inverse_transform_target(trues.reshape(shape[0] * shape[1], -1)).reshape(shape)

        q_shape = quantile_preds_all.shape
        quantile_inv = np.zeros_like(quantile_preds_all)
        
        # 由于所有分位数结果共用相同的标准化范围（与目标变量一致），因此逐个分位通道单独进行逆向还原
        for qi in range(args.n_quantiles):
            q_slice = quantile_preds_all[:, :, qi : qi + 1]
            q_inv = test_data.inverse_transform_target(
                q_slice.reshape(q_shape[0] * q_shape[1], -1)
            ).reshape(q_shape[0], q_shape[1], 1)
            quantile_inv[:, :, qi] = q_inv[:, :, 0]

        # 留存对应具有物理单位（MW/KW）的真实世界数据
        np.save(os.path.join(folder_path, "pred_inv.npy"), preds_inv)
        np.save(os.path.join(folder_path, "true_inv.npy"), trues_inv)
        np.save(os.path.join(folder_path, "quantile_preds_inv.npy"), quantile_inv)

    origin_pred = preds_inv if test_data.scale else preds_p50
    origin_true = trues_inv if test_data.scale else trues
    origin_eval_df = cal_eval(origin_true, origin_pred)
    print("[origin Eval] metrics:")
    print(origin_eval_df)

    # =============== 3. 统计误差打印 ===============
    # 通常要求在原始量纲（Inverse）空间内进行最终测绘以评判模型业务价值
    if test_data.scale and getattr(args, "inverse_eval", False):
        mae, mse, rmse, mape, mspe = metric(preds_inv, trues_inv)
        print(f"P50 Test Metrics (Inverse): MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}")
    else:
        mae, mse, rmse, mape, mspe = metric(preds_p50, trues)
        print(f"P50 Test Metrics (Normalized): MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}")

    return folder_path


def _get_setting(args, itr: int = 0) -> str:
    """
    生成一个具有唯一性的系统级字符串标识（Setting string），用于命名当前实验存放检查点（Checkpoints）和预测结果的目录。
    
    【核心挑战解释】
    在具有复杂超参数空间（如气象通道维度、相似日门控参数、两阶段学习率等）的深度学习实验中，如果直接将所有超参数拼接作为文件夹名称，
    极易导致操作系统路径长度溢出（Windows 默认限制为 260 字符），甚至因为参数含特殊字符引发文件读写报错。
    
    【防碰撞设计哈希策略】
    为了解决上述痛点，该方法采用"显式核心参数前缀 + 隐式全量参数MD5哈希后缀"的混合编码策略：
    - 显式部分（肉眼可读）：保留部分最具区分度的实验属性（如：任务名、窗长、Batch Size、是否两阶段微调截断描述符等），方便快速人工查阅。
    - 隐式部分（绝对唯一）：将所有可能影响到网络权重或训练动力学的核心参数（包括：维度定义、学习率、门控设计等）拼接并进行 MD5 哈希摘要，随后截取前8位。
      这在前缀相同的情况下，提供了近 16^8（约 42.9 亿）种独立哈希槽位，几乎完全避免了不同微小配置覆盖掉历史权重的"参数撞车"问题，并缩短了路径长度。

    Args:
        args (argparse.Namespace): 包含所有命令行或全局解析配置参数的对象，含各种动态属性。
        itr (int): 实验迭代轮次（多次重复实验中的第 N 轮），默认值为 0。

    Returns:
        str: 基于参数计算出的一段高度浓缩且安全的唯一标识字符串。
    """
    # -------------------------------------------------------------------------
    # 步骤一：提取和清洗基础实验描述符（Experiment Descriptor）
    # 去除特殊字符以防止底层文件系统挂载时触发无效路径异常
    # -------------------------------------------------------------------------
    des_raw = str(getattr(args, "des", "exp"))
    # 从原始描述中过滤，仅保留英文字母和数字。若过滤后为空则降级退回 'exp' 标记
    des_norm = "".join(ch for ch in des_raw.lower() if ch.isalnum()) or "exp"
    
    # -------------------------------------------------------------------------
    # 步骤二：缩略显示，针对大规模超参探索框架的特定优化
    # Optuna 等 HPO 工具往往自动生成极长的前缀描述，因此需做精简
    # -------------------------------------------------------------------------
    # 若发现带有 'optuna' 标签，直接简写为 'op'；否则保留纯净序列的前 6 个字符防喧宾夺主
    des_short = "op" if des_norm.startswith("optuna") else des_norm[:6]
    
    # -------------------------------------------------------------------------
    # 步骤三：提取复杂训练范式（两阶段微调）下极高灵敏度的学习率超参签名
    # 对处于第一/二阶段的 epoch 设定及各网络基件的学习率参数单独记录
    # -------------------------------------------------------------------------
    stage_signature = ""
    # 使用 getattr 防御性获取属性；只有明确开启了两阶段微调的配置才会注入此详细签名
    if bool(getattr(args, "enable_two_stage_finetune", False)):
        stage_signature = (
            f"_ts1_s1e{int(getattr(args, 'stage1_epochs', 0))}"
            f"_s1g{float(getattr(args, 'stage1_gate_lr', 0.0))}"
            f"_s1h{float(getattr(args, 'stage1_head_lr', 0.0))}"
            f"_s2e{int(getattr(args, 'stage2_epochs', 0))}"
            f"_s2b{float(getattr(args, 'stage2_backbone_lr', 0.0))}"
            f"_s2x{float(getattr(args, 'stage2_gate_lr_scale', 0.0))}"
        )
        
    # -------------------------------------------------------------------------
    # 步骤四：聚合拼装底层完整的特征属性大签名（The Full Detailed Signature）
    # 穷举在特征工程、主干网络与门控机制上起决定性作用的所有组合因子。
    # 包含了模型基建：骨干结构, 窗长，以及所有的相似日寻回与 Gate 相关超参数配置。
    # -------------------------------------------------------------------------
    signature = (
        f"{args.task_name}_{args.model_id}_{args.model}_e2e_sdv4_"
        f"sl{args.seq_len}_pl{args.pred_len}_dm{args.d_model}_"
        f"el{args.e_layers}_wd{args.weather_feature_dim}_"
        f"wsl{args.weather_seq_len}_wh{args.weather_history_len}_"
        f"wk{args.weather_kernel_height}x{args.weather_kernel_width}_"
        # 相似日先验是否启用
        f"sdp{int(bool(getattr(args, 'use_similar_day_prior', False)))}_"
        # 相似日的检索召回数量 Top-K
        f"sdk{int(getattr(args, 'similar_day_top_k', 0))}_"
        # 残差映射门控层（Gate）隐含层的神经投影维度
        f"sdgh{int(getattr(args, 'similar_day_gate_hidden_dim', 0))}_"
        # 将门控初始基准 Bias 扩大 1000 倍转化为整数（防浮点进度精度漂移导致哈希撞车）
        f"sdga{int(round(1000.0 * float(getattr(args, 'similar_day_gate_init_beta', 0.0))))}_"
        f"lr{args.learning_rate}_bs{args.batch_size}{stage_signature}_{args.des}_{itr}"
    )
    
    # -------------------------------------------------------------------------
    # 步骤五：哈希签名压缩并生成全局抗冲撞 UID
    # 由于原始 signature 可能超 200 个字符长度，强制计算其 UTF-8 的 MD5 摘要，
    # 并取其前 8 个16进制字符输出。不仅能收敛路径长度，更是版本追踪的最佳水印。
    # -------------------------------------------------------------------------
    digest = hashlib.md5(signature.encode("utf-8")).hexdigest()[:8]
    
    # -------------------------------------------------------------------------
    # 步骤六：组装精炼前缀与 UID 的最终组合结果
    # “短明文”暴露了开发者最关心的日常维度（序列长、预测长、气象维、两阶段设定等），
    # 并由唯一 digest 收尾，确保即使前缀一致但细节参数异动也能保证绝对的输出物理隔离。
    # -------------------------------------------------------------------------
    return (
        f"sdv4_sl{args.seq_len}_pl{args.pred_len}_"
        f"wd{args.weather_feature_dim}_sdk{int(getattr(args, 'similar_day_top_k', 0))}_"
        f"ts{int(bool(getattr(args, 'enable_two_stage_finetune', False)))}_"
        f"bs{args.batch_size}_{des_short}{int(itr):03d}_{digest}"
    )


def main() -> None:
    """
    程序的全局总入口：贯穿数据准备、模型装配、环境探测、双阶段训练调度以及预测分析。
    """
    # 1. 初始化全局确定性（Deterministic Engine）
    # 固定随机种子是复现比较性实验（如对比有无先验引入的涨点情况）的绝对前置条件
    fix_seed = 2026
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # 2. 从命令行接收气象大数据的挂载源，并解析该源对应的 HDF5 详细物理配置
    cli_args = _parse_cli_args()
    selected_weather_source = cli_args.weather_source
    selected_weather_h5_specs = _resolve_weather_h5_specs(selected_weather_source)

    # 3. 聚合全局配置 (Configurations Aggregation)
    # 结合命令行与 base.py 中大量预设的物理学、网络结构和训练法则超参数
    args = argparse.Namespace(
        task_name=base.TASK_NAME,
        is_training=1 if TRAIN_MODE else 0,                  # 控制执行树分支（Train or Test Only）
        model_id=f"{base.MODEL_ID_PREFIX}_sdv4",             # 定义版本流水线代号为 sdv4 (Similar Day V4)
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
        embed="timeF",                                       # 采用高阶时间特征嵌入模式
        checkpoints="./checkpoints_test4/",                  # 网络权重存储仓库点
        seq_len=base.SEQ_LEN,                                # 观测的历史窗口宽度
        label_len=base.LABEL_LEN,
        pred_len=base.PRED_LEN,                              # 预测的时段长度
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
        
        # ==== 气象流通道专门的设置 ====
        weather_source=selected_weather_source,
        weather_h5_specs=selected_weather_h5_specs,
        weather_in_channels=base.WEATHER_IN_CHANNELS,        # 原始气象通道数目（例如：温、湿度、降水、风速等）
        weather_feature_dim=base.WEATHER_FEATURE_DIM,        # 经过 CNN 骨干降维后的特征输出数
        weather_grid_height=base.WEATHER_GRID_HEIGHT,
        weather_grid_width=base.WEATHER_GRID_WIDTH,
        weather_kernel_height=base.WEATHER_KERNEL_HEIGHT,
        weather_kernel_width=base.WEATHER_KERNEL_WIDTH,
        weather_encode_chunk_size=base.WEATHER_ENCODE_CHUNK_SIZE,
        use_weather_normalization=True,
        num_workers=base.NUM_WORKERS,
        pin_memory=base.PIN_MEMORY,
        contiguous_train_batches=base.CONTIGUOUS_TRAIN_BATCHES,
        
        # ==== 优化器与训练周期相关计算 ====
        train_epochs=base.TRAIN_EPOCHS,
        batch_size=base.BATCH_SIZE,
        patience=base.PATIENCE,
        learning_rate=base.LEARNING_RATE,
        loss="Quantile",
        lradj="cosine",
        use_amp=True,                                        # 默认开启 PyTorch 原生混合精度计算降低显存峰值
        inverse_eval=base.INVERSE_EVAL,                      # 预测评估时是否反归一化还原回绝对物理量纲（MW）
        use_gpu=base.USE_GPU,
        gpu=base.GPU,
        use_multi_gpu=False,
        devices="0,1,2,3",
        quantiles=base.QUANTILES,
        n_quantiles=base.N_QUANTILES,
        
        # ==== 相似日门控纠偏系统的核心参数 ====
        use_similar_day_prior=USE_SIMILAR_DAY_PRIOR,
        similar_day_top_k=SIMILAR_DAY_TOP_K,
        similar_day_artifact_dir=SIMILAR_DAY_ARTIFACT_DIR,
        similar_day_gate_hidden_dim=SIMILAR_DAY_GATE_HIDDEN_DIM,
        similar_day_gate_init_beta=SIMILAR_DAY_GATE_INIT_BETA,
        
        # ==== 两阶段调度机制及相关各部位学习率 (防灾难性遗忘与震荡设计) ====
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

    # 4. Optuna 劫持覆盖逻辑 (Optuna Override)
    # 若系统当前受 HPO 超参搜索框架接管，会自动将最优实验参数强行改写当前设置
    if LOAD_FROM_OPTUNA:
        args = _apply_optuna_backbone_config(args)
        selected_weather_source = getattr(args, "weather_source", selected_weather_source)
        selected_weather_h5_specs = getattr(args, "weather_h5_specs", selected_weather_h5_specs)

    # 5. 算力设备自动感知与分配
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU: cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # 6. 初始化巨型气象缓存池 (Weather Hub Memory Store)
    # 通过 H5 持久化链接获取去重后的气象时序网格
    weather_store = WeatherGridStore(
        args.weather_h5_specs,
        expected_in_channels=args.weather_in_channels,
        fill_value=base.WEATHER_FILL_VALUE,
        use_channel_normalization=True,
    )
    
    # Python 的 finally 块用来保证无论模型内部引发什么异常，底层 H5 文件描述符都会被安全析构
    try:
        # 基于从文件读出的真实气象分辨率对 args 执行运行时补丁修正
        args = _configure_runtime_weather_args(args, weather_store, selected_weather_source)

        if weather_store.frame_shape is None:
            raise RuntimeError("weather_store.frame_shape is not initialized.")
        _, frame_height, frame_width = weather_store.frame_shape
        # 断言气象帧与预设的 CNN 感受野分析窗尺寸是否匹配
        if (frame_height, frame_width) != (args.weather_kernel_height, args.weather_kernel_width):
            raise ValueError(
                "Weather frame size does not match full-map kernel size: "
                f"frame=({frame_height}, {frame_width}), "
                f"kernel=({args.weather_kernel_height}, {args.weather_kernel_width})"
            )

        # 7. 组装实例化具有“门控残差纠偏”和“多置信区间量化”的 FullMap TimeXer 模型架构
        model = FullMapConvTimeXerPriorCorrectionGateQuantile(args, quantiles=args.quantiles).float().to(device)
        
        # 参数规模热区诊断监控（方便查阅其模型复杂度）
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"TimeXer-primary + prior-correction total params: {total_params:,}")
        print(f"TimeXer-primary + prior-correction trainable params: {trainable_params:,}")

        setting = _get_setting(args)
        
        # ==================== 执行生命周期核心主分支（训练 / 推理） ====================
        if args.is_training:
            # 策略 1：如果是做先验强化挂载研究，需要载入预训练已经收敛好的 TimeXer 权重垫底
            if LOAD_FROM_OPTUNA:
                print("\n>>> Load Optuna backbone before training")
                _load_backbone_from_optuna(model, args, device)

            # 策略 2：控制是否拉起高阶的《两阶逐级释放微调法》护城河（防初期随机参数导致的梯度爆炸摧毁骨干网）
            # 或者回退到传统的《从头端到端联合微调法》
            if bool(getattr(args, "enable_two_stage_finetune", False)):
                print(f"\n>>> Start two-stage training {setting}")
                model = train_two_stage_model(model, args, device, weather_store)
            else:
                print(f"\n>>> Start training {setting}")
                model = train_quantile_model(model, args, device, weather_store)

            # 模型训练成型且收敛后，自动进入全量测试流程提取业务报告
            print(f"\n>>> Start testing {setting}")
            results_dir = test_quantile_model(model, args, device, weather_store)
        else:
            # === 非训练态：回退至纯预测验证（Inference Deployment Mode） ===
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
            # 进行盲区序列检测，评估真实性能
            results_dir = test_quantile_model(model, args, device, weather_store)

        # 8. 自动化周边业务工具链联动 -- 生成可视化对比曲线走势图
        # 图谱中心为 P50 中心线，附带根据其他分位数的置信阴影区域
        plot_pred_vs_true(
            results_dir,
            use_inverse=args.inverse_eval,
            quantiles=args.quantiles,
            title_prefix="TimeXer-Primary + Similar-Day Prior-Correction Prediction",
            y_label="Load (MW)",
        )

        # 9. 导出并落盘当前配置下最纯天然未过网络的相似日基准（Baseline），以便在后续业务汇报中打靶对比
        similar_day_result = export_similar_day_baseline(
            results_dir=results_dir,
            future_path=getattr(args, "future_path", base.FUTURE_PATH),
            args=args,
            artifact_dir=getattr(args, "similar_day_artifact_dir", SIMILAR_DAY_ARTIFACT_DIR),
            top_k=int(getattr(args, "similar_day_top_k", SIMILAR_DAY_TOP_K)),
        )
        
        # 10. 落盘未来无标签时刻表生成工业落地（Production Use）格式的 .csv 正规负荷预测结论文件
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
        # 【关键兜底动作】
        # 释放系统底下的文件描述符防爆表，清理残存的 VRAM 以避免在连续执行外层 Python 循环脚本时的内存泄漏
        weather_store.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
