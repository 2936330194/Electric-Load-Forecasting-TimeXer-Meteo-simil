"""
test4_smp.py - 气象全图卷积编码器 + TimeXer 分位点预测端到端流水线主程序

该程序是整个电力负荷预测项目的核心入口。它集成了：
1. 全图气象卷积特征提取（Full-Map Weather Conv Encoder）
2. TimeXer 分离模式下的概率时间序列预测（Quantile Forecasting）
3. 扩展气象窗口支持（使用 seq_len + pred_len = 768 步的气象数据输入）
4. 增强的 I/O 优化（连续批次采样与去重气象索引）

项目模块依赖：
- utils.weather_e2e: 气象数据加载、对齐、模型封装等核心逻辑
- utils.quantile: 分位数损失函数实现
- utils.forecast_visualization: 预测结果可视化与 CSV 预测工具
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
from torch import optim

from utils.forecast_visualization import (
    plot_pred_vs_true,
    plot_similar_day_curves,
    predict_future_load_from_csv,
)
from utils.metrics import metric
from utils.quantile import QuantileLoss
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.weather_e2e import FullMapConvTimeXerQuantile, WeatherGridStore, weather_data_provider


# 分位点配置：用于概率区间预测
QUANTILES = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
N_QUANTILES = len(QUANTILES)
P10_IDX = QUANTILES.index(0.1)  # 10% 分位数索引
P50_IDX = QUANTILES.index(0.5)  # 中位数（点预测值）索引
P90_IDX = QUANTILES.index(0.9)  # 90% 分位数索引


# 基础实验 ID 与路径配置
TASK_NAME = "long_term_forecast"
MODEL = "TimeXer"
MODEL_ID = "HunanLoad_uk_672_96_FullMapConv_E2E"

ROOT_PATH = "./data/"
DATA_PATH = "湖南省电力负荷_unknow.csv"
FUTURE_PATH = "./data/湖南省电力负荷_unknow_future.csv"
TARGET = "load"         # 目标列名
FEATURES = "MS"         # 任务类型：MS 表示多变量输入单变量输出
SIMILAR_DAY_ARTIFACT_DIR = "./artifacts/similar_day_retriever_ae_128" # 相似日检索特征库目录
SIMILAR_DAY_TOP_K = 3                                               # 检索 Top-K 个相似日
USE_SIMILAR_DAY_PRIOR = True                                         # 是否在主模型中使用相似日作为先验
SIMILAR_DAY_FUSION_HIDDEN_DIM = 128                                 # 相似日特征融合的隐藏维度

# 时间窗口配置
SEQ_LEN = 96 * 7        # 历史观测窗口长度（672步，对应7天）
LABEL_LEN = 0           # 标签长度（TimeXer 分离模式下设为 0）
PRED_LEN = 96           # 预测窗口长度（24小时）
WEATHER_SEQ_LEN = SEQ_LEN + PRED_LEN    # 重要：气象数据序列包含未来 24 小时预报，总长 768

# 气象 HDF5 配置
WEATHER_H5_SPECS: List[Tuple[str, str, str]] = [
    ("./data/hunan_grid_meteo_20250101_2 0260228.h5", "2025-01-01 00:00:00", "15min"),
]
WEATHER_IN_CHANNELS = 10        # 气象网格通道数（10个气象参数）
WEATHER_GRID_HEIGHT = 62        # 网格高度
WEATHER_GRID_WIDTH = 61         # 网格宽度
WEATHER_KERNEL_HEIGHT = 62      # 全图卷积核高度
WEATHER_KERNEL_WIDTH = 61       # 全图卷积核宽度
WEATHER_FEATURE_DIM = 3         # 气象降维后的特征维度
WEATHER_ENCODE_CHUNK_SIZE = 512 # 模型并行时的 chunk 大小
WEATHER_FILL_VALUE = 0.0        # 缺失值填充
USE_WEATHER_NORMALIZATION = True  # 是否启用气象通道级标准化处理
WEATHER_LOG1P_CHANNELS = [9]      # 需要先做 log1p 变换的通道索引列表
WEATHER_NORM_FIT_CHUNK_SIZE = 512 # 计算标准化统计量时的分块大小（用于大规模数据）
WEATHER_NORMALIZATION_EPS = 1e-6  # 标准差下界保护，防止除零或过小导致数值不稳定

# TimeXer 网络超参数
ENC_IN = 1      # 内生变量通道数（分离模式下仅负荷本身）
C_OUT = 1       # 输出通道数
D_MODEL = 512   # 隐藏层维度
N_HEADS = 4     # 注意力头数
E_LAYERS = 2    # 编码器层数
D_FF = 2048     # FFN 维度
FACTOR = 3      # 注意力因子
DROPOUT = 0.1   # 丢弃率
ACTIVATION = "gelu" # 激活函数
PATCH_LEN = 96    # PatchEmbedding 的长度
USE_NORM = 1      # 是否使用归一化

# 训练配置
TRAIN_EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
PATIENCE = 5            # 早期停止耐心值
NUM_WORKERS = 0         # 建议在 Windows 下设为 0 以防 HDF5 多线程冲突

# GPU 与模式配置
USE_GPU = True
GPU = 0
DES = "Exp"             # 实验描述
ITR = 1                 # 运行迭代次数
INVERSE_EVAL = True     # 指标评估时是否反标准化回到原始量纲
TRAIN_MODE = True       # True 为训练模式，False 为仅推理测试

# 性能优化参数
PIN_MEMORY = True               # 开启固定内存加速数据搬运
CONTIGUOUS_TRAIN_BATCHES = True # 开启连续批次采样以大幅优化气象数据加载速度

# /use 导入配置
LOAD_FROM_USE = False  # 是否从 /use 导入最优参数和权重（仅测试模式有效）
USE_DIR = "./use"
USE_BEST_PARAMS_FILE = "best_params.json"
USE_BEST_CONFIG_FILE = "best_config.json"
USE_BEST_WEIGHT_FILE = "best_model.pth"
TUNABLE_PARAM_MAP = {
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


def _use_non_blocking_transfer(args, device: torch.device) -> bool:
    """判断是否可以使用非阻塞的数据传输，通常在 CUDA 且使用固定内存时开启"""
    return (
        device.type == "cuda"
        and bool(getattr(args, "pin_memory", False))
        and torch.cuda.is_available()
        and args.use_gpu
    )


def _to_float_device(
    tensor: torch.Tensor, device: torch.device, non_blocking: bool = False
) -> torch.Tensor:
    """将张量移动到设备并转为浮点数，支持非阻塞传输"""
    return tensor.to(device=device, dtype=torch.float32, non_blocking=non_blocking)


def _to_long_device(
    tensor: torch.Tensor, device: torch.device, non_blocking: bool = False
) -> torch.Tensor:
    """将张量移动到设备并转为长整型（用于索引），支持非阻塞传输"""
    return tensor.to(device=device, dtype=torch.long, non_blocking=non_blocking)


def extract_target(batch_y: torch.Tensor) -> torch.Tensor:
    """从回归输出中提取目标负荷列（最后一列）"""
    return batch_y[:, :, -1:]


def _unpack_weather_batch(
    batch: Sequence[torch.Tensor],
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
]:
    """
    解包天气批次数据，兼容有无相似日先验的两种格式。
    
    支持两种输入格式：
    - 6元组：不包含相似日先验信息
    - 7元组：包含相似日先验信息
    
    Returns:
        元组包含 (batch_x, batch_y, batch_x_mark, batch_exo_mark, 
                 batch_weather_frames, batch_weather_index, similar_day_prior)
    """
    if len(batch) == 6:
        # 标准批次格式（不使用相似日先验）
        batch_x, batch_y, batch_x_mark, batch_exo_mark, batch_weather_frames, batch_weather_index = batch
        return (
            batch_x,
            batch_y,
            batch_x_mark,
            batch_exo_mark,
            batch_weather_frames,
            batch_weather_index,
            None,  # 相似日先验为 None
        )
    elif len(batch) == 7:
        # 扩展批次格式（包含相似日先验）
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
            similar_day_prior,  # 返回相似日先验
        )
    
    # 异常处理：不支持的批次格式
    raise ValueError(f"Unexpected batch size: expected 6 or 7 tensors, got {len(batch)}")


def _format_int_list(values: Sequence[int]) -> str:
    """将整数列表压缩为紧凑的签名文本。"""
    values = [int(v) for v in values]
    return "-".join(str(v) for v in values) if values else "none"


def _load_json_file(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _apply_use_artifacts(args: argparse.Namespace) -> argparse.Namespace:
    """
    从 /use 目录导入最优模型参数、配置和权重文件。
    
    该函数在测试模式（TRAIN_MODE=False）下调用，用于加载之前训练保存的最优模型配置。
    支持两种导入方式：
    1. 优先导入 best_config.json（完整配置）
    2. 备选导入 best_params.json（超参数映射）
    
    Args:
        args: 命令行参数命名空间对象
        
    Returns:
        更新后的 args 对象，包含从 /use 导入的所有参数和权重路径
        
    Raises:
        FileNotFoundError: 权重文件或配置文件不存在时抛出
        ValueError: 配置文件内容不是有效 JSON 对象时抛出
    """
    # 1. 获取 /use 目录的绝对路径
    use_dir = os.path.abspath(USE_DIR)
    
    # 2. 构建完整配置、超参数、权重文件的路径
    config_path = os.path.join(use_dir, USE_BEST_CONFIG_FILE)      # best_config.json
    params_path = os.path.join(use_dir, USE_BEST_PARAMS_FILE)      # best_params.json
    weight_path = os.path.join(use_dir, USE_BEST_WEIGHT_FILE)      # best_model.pth

    # 3. 强制检查权重文件存在性（权重文件是必需的）
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"/use 权重文件不存在: {weight_path}")

    # 4. 尝试优先导入完整配置文件（best_config.json）
    if os.path.exists(config_path):
        # 加载 JSON 格式的完整配置
        payload = _load_json_file(config_path)
        
        # 验证配置内容为字典格式
        if not isinstance(payload, dict):
            raise ValueError(f"/use 配置文件内容不是 JSON 对象: {config_path}")
        
        # 将配置中的每个键值对映射到 args 对象属性
        for key, value in payload.items():
            setattr(args, key, value)
    
    # 5. 备选方案：导入超参数映射文件（best_params.json）
    elif os.path.exists(params_path):
        # 加载 JSON 格式的超参数字典
        payload = _load_json_file(params_path)
        
        # 验证超参数内容为字典格式
        if not isinstance(payload, dict):
            raise ValueError(f"/use 超参数文件内容不是 JSON 对象: {params_path}")
        
        # 遍历超参数并映射到标准参数名称
        for raw_key, value in payload.items():
            # 使用 TUNABLE_PARAM_MAP 将原始参数名映射到规范名称
            # 例如："WEATHER_FEATURE_DIM" -> "weather_feature_dim"
            key = TUNABLE_PARAM_MAP.get(str(raw_key), str(raw_key))
            setattr(args, key, value)
    
    # 6. 都不存在时抛出异常
    else:
        raise FileNotFoundError(
            f"/use 中未找到 {USE_BEST_CONFIG_FILE} 或 {USE_BEST_PARAMS_FILE}"
        )

    # 7. 强制设置测试模式相关参数
    args.is_training = 0  # 设为测试模式（不进行训练）
    args.load_weight_path = weight_path  # 指定权重加载路径
    
    # 8. 确保气象序列长度正确设置（覆盖可能的旧值）
    args.weather_seq_len = int(getattr(args, "weather_seq_len", args.seq_len + args.pred_len))
    
    # 9. 设置分位数数量（用于分位数预测模型）
    args.n_quantiles = len(args.quantiles)
    
    # 10. 终端输出确认成功导入
    print(f"已从 /use 导入参数与权重: {weight_path}")
    
    return args


def export_similar_day_baseline(
    results_dir: str,
    future_path: str,
    artifact_dir: str = SIMILAR_DAY_ARTIFACT_DIR,
    top_k: int = SIMILAR_DAY_TOP_K,
) -> Optional[Any]:
    """
    基于离线构建好的相似日检索库，为未来预测起点检索 Top-K 历史负荷曲线,
    并导出曲线图、宽表 CSV 与检索元信息 JSON，作为系统的保底输出。
    """
    # 打印分隔符和执行提示
    print("\n" + "=" * 60)
    print(f"相似日检索基线：读取自 {future_path}")
    print("=" * 60)

    # 1. 解析目标文件的绝对路径并检查是否存在
    abs_future_path = os.path.abspath(future_path)
    if not os.path.exists(abs_future_path):
        print(f"未找到未来数据文件，跳过相似日检索: {abs_future_path}")
        return None

    # 2. 动态导入相似日检索模块，增加独立运行的鲁棒性
    try:
        from similar_day_retriever import DEFAULT_ARTIFACT_DIR, SimilarDayRetriever, print_retrieval_result
    except Exception as exc:
        print(f"导入 similar_day_retriever 失败，跳过检索: {exc}")
        return None

    # 3. 确定并校验离线特征库目录（结合参数传入与模块默认值）
    resolved_artifact_dir = os.path.abspath(
        str(artifact_dir) if artifact_dir is not None else str(DEFAULT_ARTIFACT_DIR)
    )
    if not os.path.isdir(resolved_artifact_dir):
        print(f"未找到相似日模型库目录，跳过检索: {resolved_artifact_dir}")
        return None

    # 4. 尝试加载检索模型并执行基于预测起点文件的 Top-K 检索
    try:
        retriever = SimilarDayRetriever.load(resolved_artifact_dir)
        result = retriever.search_from_future_csv(abs_future_path, top_k=int(top_k))
    except Exception as exc:
        print(f"执行相似日检索失败: {exc}")
        return None

    # 5. 在终端展示检索结果并进行文件记录与绘图生成
    print_retrieval_result(result)
    plot_similar_day_curves(
        results_dir=results_dir,                         # 输出目录
        retrieval_result=result,                         # 检索返回的结果数据对象
        out_name="similar_day_retrieval.png",             # 导出的可视化图像文件名
        csv_name="similar_day_retrieval.csv",             # 导出的负荷数值表格
        json_name="similar_day_retrieval.json",           # 导出的查询元配置数据
        title_prefix="相似日负荷检索基线",               # 保存图表的标题文案
        y_label="电负荷 (MW)",                           # Y 轴展示标签
        freq="15min",                                    # 时序默认的步长解析规则
    )
    return result


def validate_quantile(model, data_loader, criterion, args, device, use_amp: bool = False) -> float:
    """
    分位数预测验证函数。
    
    在验证/测试集上计算模型的平均分位数损失（Quantile Loss）。
    该函数用于评估模型在概率区间预测任务上的性能。
    
    Args:
        model: 训练好的 TimeXer 分位数预测模型实例
        data_loader: 验证/测试数据加载器（包含天气和负荷数据）
        criterion: 分位数损失函数对象（QuantileLoss）
        args: 包含模型超参数和配置的命名空间对象
        device: 计算设备（cuda 或 cpu）
        use_amp: 是否使用自动混合精度加速，默认 False
        
    Returns:
        float: 加权平均分位数损失值，如果无数据则返回 NaN
    """
    # 1. 设置模型为评估模式（禁用 dropout 和 batch norm 更新）
    model.eval()
    total_loss = []
    
    # 2. 根据设备和内存配置判断是否使用异步数据传输
    use_non_blocking = _use_non_blocking_transfer(args, device)

    # 3. 禁用梯度计算以节省显存（推理阶段不需要梯度）
    with torch.inference_mode():
        # 4. 遍历验证/测试数据加载器中的每个批次
        for batch in data_loader:
            # 5. 解包批次数据，兼容有无相似日先验的两种格式
            (
                batch_x,                    # 历史负荷序列 [batch_size, seq_len, 1]
                batch_y,                    # 目标负荷序列 [batch_size, pred_len, 1]
                batch_x_mark,               # 负荷时间标记 [batch_size, seq_len, time_dims]
                batch_exo_mark,             # 气象时间标记 [batch_size, pred_len, time_dims]
                batch_weather_frames,       # 气象网格数据 [batch_size, weather_seq_len, channels, H, W]
                batch_weather_index,        # 气象时间索引 [batch_size, weather_seq_len]
                similar_day_prior,          # 相似日先验 [batch_size, top_k, pred_len] 或 None
            ) = _unpack_weather_batch(batch)
            
            # 6. 将所有张量异步移动到目标设备（GPU 或 CPU）
            batch_x = _to_float_device(batch_x, device, non_blocking=use_non_blocking)
            batch_y = _to_float_device(batch_y, device, non_blocking=use_non_blocking)
            batch_x_mark = _to_float_device(batch_x_mark, device, non_blocking=use_non_blocking)
            batch_exo_mark = _to_float_device(batch_exo_mark, device, non_blocking=use_non_blocking)
            batch_weather_frames = _to_float_device(
                batch_weather_frames, device, non_blocking=use_non_blocking
            )
            # 气象索引使用长整型（用于 embedding 查表）
            batch_weather_index = _to_long_device(
                batch_weather_index, device, non_blocking=use_non_blocking
            )
            # 相似日先验如果存在则转换为浮点类型
            if similar_day_prior is not None:
                similar_day_prior = _to_float_device(
                    similar_day_prior, device, non_blocking=use_non_blocking
                )

            # 7. 条件激活 AMP（自动混合精度）以加速推理
            with torch.amp.autocast("cuda", enabled=use_amp):
                # 8. 前向传播：调用端到端模型生成多分位数预测
                # 输入：分离模式下仅使用负荷序列、时间标记、气象数据、相似日先验
                # 输出：预测的全分位数矩阵 [batch_size, pred_len, n_quantiles]
                outputs = model(
                    load_x=batch_x,                      # 历史负荷输入
                    x_mark_enc=batch_x_mark,             # 负荷时间特征
                    x_exo_mark=batch_exo_mark,           # 气象时间特征
                    weather_x=batch_weather_frames,      # 气象网格（已编码）
                    weather_x_index=batch_weather_index,  # 气象时间对齐索引
                    similar_day_prior=similar_day_prior,  # 相似日先验（可选）
                )
                
                # 9. 从目标负荷序列中提取预测窗口对应的目标值
                # batch_y[:, -args.pred_len:, :] 取最后 pred_len 步作为目标
                # extract_target() 从多变量中提取单变量目标列（最后一列：负荷）
                batch_y_target = extract_target(batch_y[:, -args.pred_len :, :])
                
                # 10. 计算分位数损失
                # criterion 为 QuantileLoss，计算所有分位数的加权损失
                # 输入 outputs [batch_size, pred_len, n_quantiles]
                # 输入 batch_y_target [batch_size, pred_len, 1]
                loss = criterion(outputs, batch_y_target)

            # 11. 收集当前批次的损失值（转为 Python 标量）
            total_loss.append(loss.item())

    # 12. 恢复模型为训练模式
    model.train()
    
    # 13. 计算并返回所有批次损失的平均值
    # 如果没有数据则返回 NaN（空列表情形）
    return float(np.average(total_loss)) if total_loss else np.nan


def train_quantile_model(model, args, device, weather_store: WeatherGridStore):
    """
    主训练循环函数。
    
    执行完整的模型训练流程，包括：
    - 数据加载（训练/验证集）
    - 优化器与损失函数初始化
    - 混合精度加速（AMP）配置
    - Epoch 级别的训练循环
    - 验证损失监控与早期停止
    - 学习率衰减调度
    - 最优权重保存与加载
    
    Args:
        model: 端到端的 TimeXer 分位数预测模型实例
        args: 包含所有超参数和配置的命名空间对象
        device: 计算设备（cuda 或 cpu）
        weather_store: 气象网格数据存储池（用于批次数据加载）
        
    Returns:
        训练完毕后的模型，已加载最优权重
    """
    # 1. 从数据提供器构建训练/验证数据加载器
    _, train_loader = weather_data_provider(args, "train", weather_store)
    _, vali_loader = weather_data_provider(args, "val", weather_store)

    # 2. 生成唯一的实验配置标识（包含所有关键超参数）
    setting = _get_setting(args)
    # 3. 创建检查点保存目录
    path = os.path.join(args.checkpoints, setting)
    os.makedirs(path, exist_ok=True)

    # 4. 初始化优化器（Adam 优化器，应用学习率）
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # 5. 初始化分位数损失函数并移动到目标设备
    criterion = QuantileLoss(args.quantiles).to(device)
    # 6. 初始化早期停止监控对象（监控验证损失，设定耐心值）
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    # 7. 配置自动混合精度（AMP）加速
    # 仅在 CUDA 设备上启用，可显著加快 RTX 30/40 系列显卡的训练速度
    use_amp = bool(getattr(args, "use_amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    # 8. 判断是否使用异步数据传输（加快 GPU 数据搬运）
    use_non_blocking = _use_non_blocking_transfer(args, device)

    # 9. 打印训练配置信息（用于记录实验参数）
    print("\n" + "=" * 72)
    print("Start training Full-Map Conv + TimeXer end-to-end quantile model")
    print(f"setting: {setting}")
    print(f"quantiles: {args.quantiles}")
    print(f"weather_feature_dim: {args.weather_feature_dim}")
    print(f"weather_seq_len: {getattr(args, 'weather_seq_len', args.seq_len)} (seq_len={args.seq_len} + pred_len={args.pred_len})")
    print(f"weather_kernel_size: ({args.weather_kernel_height}, {args.weather_kernel_width})")
    print(f"batch_size: {args.batch_size}")
    print(f"use_amp: {use_amp}")
    print(f"use_weather_normalization: {bool(getattr(args, 'use_weather_normalization', False))}")
    if bool(getattr(args, "use_weather_normalization", False)):
        print(
            "weather_normalization_config: "
            f"log1p_channels={list(getattr(args, 'weather_log1p_channels', []))}, "
            f"fit_chunk_size={int(getattr(args, 'weather_norm_fit_chunk_size', 0))}"
        )
    print(f"use_similar_day_prior: {bool(getattr(args, 'use_similar_day_prior', False))}")
    # 打印相似日先验的配置（如果启用）
    if bool(getattr(args, "use_similar_day_prior", False)):
        print(
            "similar_day_prior_config: "
            f"top_k={getattr(args, 'similar_day_top_k', 0)}, "
            f"fusion_hidden_dim={getattr(args, 'similar_day_fusion_hidden_dim', 0)}"
        )
    
    # 10. 打印性能优化相关信息
    weather_seq_len = getattr(args, 'weather_seq_len', args.seq_len)
    # 显示连续批次采样优化效果（减少气象数据加载重复）
    if bool(getattr(args, "contiguous_train_batches", False)):
        dense_weather_frames = args.batch_size * weather_seq_len
        unique_weather_frames = weather_seq_len + args.batch_size - 1
        print(
            "overlap-aware weather batching: "
            f"on (dense {dense_weather_frames} frames/batch -> about {unique_weather_frames} unique frames/batch)"
        )
    print("=" * 72)

    # 11. Epoch 级别的外层循环
    for epoch in range(args.train_epochs):
        # 12. 设置模型为训练模式（启用 dropout 和 batch norm 更新）
        model.train()
        train_loss = []
        epoch_time = time.time()

        # 13. Batch 级别的内层循环：遍历训练数据加载器
        for i, batch in enumerate(train_loader):
            # 14. 解包批次数据，兼容有无相似日先验的两种格式
            (
                batch_x,                    # 历史负荷序列 [batch_size, seq_len, 1]
                batch_y,                    # 目标负荷序列 [batch_size, pred_len, 1]
                batch_x_mark,               # 负荷时间标记 [batch_size, seq_len, time_dims]
                batch_exo_mark,             # 气象时间标记 [batch_size, pred_len, time_dims]
                batch_weather_frames,       # 气象网格数据 [batch_size, weather_seq_len, channels, H, W]
                batch_weather_index,        # 气象时间索引 [batch_size, weather_seq_len]
                similar_day_prior,          # 相似日先验 [batch_size, top_k, pred_len] 或 None
            ) = _unpack_weather_batch(batch)
            
            # 15. 清空优化器梯度（使用 set_to_none=True 以提高效率）
            optimizer.zero_grad(set_to_none=True)

            # 16. 将所有张量异步移动到目标设备（GPU 或 CPU）
            batch_x = _to_float_device(batch_x, device, non_blocking=use_non_blocking)
            batch_y = _to_float_device(batch_y, device, non_blocking=use_non_blocking)
            batch_x_mark = _to_float_device(batch_x_mark, device, non_blocking=use_non_blocking)
            batch_exo_mark = _to_float_device(batch_exo_mark, device, non_blocking=use_non_blocking)
            batch_weather_frames = _to_float_device(
                batch_weather_frames, device, non_blocking=use_non_blocking
            )
            # 气象索引使用长整型（用于 embedding 查表）
            batch_weather_index = _to_long_device(
                batch_weather_index, device, non_blocking=use_non_blocking
            )
            # 相似日先验如果存在则转换为浮点类型
            if similar_day_prior is not None:
                similar_day_prior = _to_float_device(
                    similar_day_prior, device, non_blocking=use_non_blocking
                )

            # 17. 条件激活 AMP（自动混合精度）以加速前向传播
            with torch.amp.autocast("cuda", enabled=use_amp):
                # 18. 前向传播：调用端到端模型生成多分位数预测
                # 输出：预测的全分位数矩阵 [batch_size, pred_len, n_quantiles]
                outputs = model(
                    load_x=batch_x,                      # 历史负荷输入
                    x_mark_enc=batch_x_mark,             # 负荷时间特征
                    x_exo_mark=batch_exo_mark,           # 气象时间特征
                    weather_x=batch_weather_frames,      # 气象网格（已编码）
                    weather_x_index=batch_weather_index,  # 气象时间对齐索引
                    similar_day_prior=similar_day_prior,  # 相似日先验（可选）
                )
                
                # 19. 从目标负荷序列中提取预测窗口对应的目标值
                # 取最后 pred_len 步作为目标
                batch_y_target = extract_target(batch_y[:, -args.pred_len :, :])
                
                # 20. 计算分位数损失
                # 输入 outputs [batch_size, pred_len, n_quantiles]
                # 输入 batch_y_target [batch_size, pred_len, 1]
                loss = criterion(outputs, batch_y_target)

            # 21. 收集当前批次的损失值
            train_loss.append(loss.item())
            
            # 22. AMP 梯度更新流程
            # scaler.scale() - 损失值缩放以防止下溢
            # backward() - 反向传播计算梯度
            # scaler.step() - 梯度反缩放并执行优化器步长
            # scaler.update() - 动态调整缩放因子
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 23. 定期打印训练进度（每 20 个 batch 打印一次）
            if (i + 1) % 20 == 0:
                print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")

        # 24. 每个 Epoch 结束后进行验证
        vali_loss = validate_quantile(model, vali_loader, criterion, args, device, use_amp=use_amp)
        # 25. 计算训练集的平均损失
        train_loss_avg = float(np.average(train_loss)) if train_loss else np.nan
        # 26. 打印 Epoch 统计信息（时间、训练损失、验证损失）
        print(
            f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.1f}s | "
            f"Train: {train_loss_avg:.7f} Vali: {vali_loss:.7f}"
        )

        # 27. 早期停止检查逻辑
        # 如果验证损失在 patience 个 epoch 内未改善，则停止训练
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # 28. 学习率衰减调度（根据 epoch 逐步降低学习率）
        adjust_learning_rate(optimizer, epoch + 1, args)

    # 29. 导出并加载最优模型权重（早期停止保存的最佳模型）
    best_model_path = os.path.join(path, "checkpoint.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"Loaded best model weights: {best_model_path}")
    
    return model


def test_quantile_model(model, args, device, weather_store: WeatherGridStore) -> str:
    """
    模型测试与评估函数。
    
    在测试集上执行模型推理，计算 P50（中位数）的误差指标（MAE/MSE/RMSE）,
    并保存所有分位点的预测结果到 .npy 文件供后续可视化和分析使用。
    
    Args:
        model: 训练完成的 TimeXer 分位数预测模型实例
        args: 包含所有超参数和配置的命名空间对象
        device: 计算设备（cuda 或 cpu）
        weather_store: 气象网格数据存储池（用于批次数据加载）
        
    Returns:
        str: 结果保存目录路径（包含所有 .npy 文件和后续生成的图表）
    """
    # 1. 从数据提供器获取测试数据和测试加载器
    test_data, test_loader = weather_data_provider(args, "test", weather_store)

    # 2. 生成唯一的实验配置标识（用于区分不同实验的结果文件夹）
    setting = _get_setting(args)
    # 3. 创建结果保存目录
    folder_path = os.path.join("./results/", setting)
    os.makedirs(folder_path, exist_ok=True)

    # 4. 初始化列表容器：存储预测结果、真实值和全分量数结果
    preds_p50 = []       # 存储 P50（中位数）点预测值
    trues = []           # 存储真实负荷目标值
    quantile_preds_all = []  # 存储所有分位数的预测结果

    # 5. 根据设备和内存配置判断是否使用异步数据传输
    use_amp = bool(getattr(args, "use_amp", False)) and device.type == "cuda"
    use_non_blocking = _use_non_blocking_transfer(args, device)

    # 6. 设置模型为评估模式（禁用 dropout 和 batch norm 更新）
    model.eval()
    
    # 7. 禁用梯度计算以节省显存（推理阶段不需要梯度）
    with torch.inference_mode():
        # 8. 遍历测试数据加载器中的每个批次
        for batch in test_loader:
            # 9. 解包批次数据，兼容有无相似日先验的两种格式
            (
                batch_x,                    # 历史负荷序列 [batch_size, seq_len, 1]
                batch_y,                    # 目标负荷序列 [batch_size, pred_len, 1]
                batch_x_mark,               # 负荷时间标记 [batch_size, seq_len, time_dims]
                batch_exo_mark,             # 气象时间标记 [batch_size, pred_len, time_dims]
                batch_weather_frames,       # 气象网格数据 [batch_size, weather_seq_len, channels, H, W]
                batch_weather_index,        # 气象时间索引 [batch_size, weather_seq_len]
                similar_day_prior,          # 相似日先验 [batch_size, top_k, pred_len] 或 None
            ) = _unpack_weather_batch(batch)
            
            # 10. 将所有张量异步移动到目标设备（GPU 或 CPU）
            batch_x = _to_float_device(batch_x, device, non_blocking=use_non_blocking)
            batch_y = _to_float_device(batch_y, device, non_blocking=use_non_blocking)
            batch_x_mark = _to_float_device(batch_x_mark, device, non_blocking=use_non_blocking)
            batch_exo_mark = _to_float_device(batch_exo_mark, device, non_blocking=use_non_blocking)
            batch_weather_frames = _to_float_device(
                batch_weather_frames, device, non_blocking=use_non_blocking
            )
            # 气象索引使用长整型（用于 embedding 查表）
            batch_weather_index = _to_long_device(
                batch_weather_index, device, non_blocking=use_non_blocking
            )
            # 相似日先验如果存在则转换为浮点类型
            if similar_day_prior is not None:
                similar_day_prior = _to_float_device(
                    similar_day_prior, device, non_blocking=use_non_blocking
                )

            # 11. 条件激活 AMP（自动混合精度）以加速推理
            with torch.amp.autocast("cuda", enabled=use_amp):
                # 12. 前向传播：调用端到端模型生成多分位数预测
                # 输出：预测的全分位数矩阵 [batch_size, pred_len, n_quantiles]
                outputs = model(
                    load_x=batch_x,                      # 历史负荷输入
                    x_mark_enc=batch_x_mark,             # 负荷时间特征
                    x_exo_mark=batch_exo_mark,           # 气象时间特征
                    weather_x=batch_weather_frames,      # 气象网格（已编码）
                    weather_x_index=batch_weather_index,  # 气象时间对齐索引
                    similar_day_prior=similar_day_prior,  # 相似日先验（可选）
                )

            # 13. 从目标负荷序列中提取预测窗口对应的目标值
            batch_y_target = extract_target(batch_y[:, -args.pred_len :, :])
            
            # 14. 从全分位数预测张量中提取 P50（中位数）分位数
            # P50_IDX 是分位数列表中中位数的索引位置
            p50_pred = outputs.float()[:, :, P50_IDX : P50_IDX + 1]

            # 15. 收集当前批次的结果
            # 全分位数结果：[batch_size, pred_len, n_quantiles]
            quantile_preds_all.append(outputs.float().detach().cpu().numpy())
            # P50 点预测：[batch_size, pred_len, 1]
            preds_p50.append(p50_pred.detach().cpu().numpy())
            # 真实目标值：[batch_size, pred_len, 1]
            trues.append(batch_y_target.detach().cpu().numpy())

    # 16. 合并所有批次的数据沿着 batch 维度（axis=0）
    preds_p50 = np.concatenate(preds_p50, axis=0)         # [n_samples, pred_len, 1]
    trues = np.concatenate(trues, axis=0)                 # [n_samples, pred_len, 1]
    quantile_preds_all = np.concatenate(quantile_preds_all, axis=0)  # [n_samples, pred_len, n_quantiles]

    # 17. 打印最终数据形状信息（用于验证数据完整性）
    print(
        f"Test shape: preds={preds_p50.shape}, "
        f"trues={trues.shape}, quantiles={quantile_preds_all.shape}"
    )

    # 18. 保存原始（标准化后）数据到 .npy 文件
    # 这些数据仍然处于标准化的数值范围内
    np.save(os.path.join(folder_path, "pred.npy"), preds_p50)
    np.save(os.path.join(folder_path, "true.npy"), trues)
    np.save(os.path.join(folder_path, "quantile_preds.npy"), quantile_preds_all)

    # 19. 如果测试数据经过标准化，则反转回原始量纲进行保存
    if test_data.scale:
        # 获取数据形状信息
        shape = trues.shape  # [n_samples, pred_len, 1]
        
        # 20. 反标准化 P50 点预测值
        # reshape：展平为 [n_samples * pred_len, 1] 进行批量反标准化
        preds_inv = test_data.inverse_transform_target(
            preds_p50.reshape(shape[0] * shape[1], -1)
        ).reshape(shape)
        
        # 21. 反标准化真实目标值
        trues_inv = test_data.inverse_transform_target(
            trues.reshape(shape[0] * shape[1], -1)
        ).reshape(shape)

        # 22. 反标准化所有分量数预测值（逐个分量数处理）
        q_shape = quantile_preds_all.shape  # [n_samples, pred_len, n_quantiles]
        quantile_inv = np.zeros_like(quantile_preds_all)  # 初始化容器
        
        # 对每个分量数独立进行反标准化
        for qi in range(N_QUANTILES):
            # 提取第 qi 个分量数的数据：[n_samples, pred_len, 1]
            q_slice = quantile_preds_all[:, :, qi : qi + 1]
            # 反标准化：展平 -> 反标准化 -> 还原形状
            q_inv = test_data.inverse_transform_target(
                q_slice.reshape(q_shape[0] * q_shape[1], -1)
            ).reshape(q_shape[0], q_shape[1], 1)
            # 存储反标准化后的结果
            quantile_inv[:, :, qi] = q_inv[:, :, 0]

        # 23. 保存反标准化后的数据到 .npy 文件（恢复到原始量纲）
        np.save(os.path.join(folder_path, "pred_inv.npy"), preds_inv)
        np.save(os.path.join(folder_path, "true_inv.npy"), trues_inv)
        np.save(os.path.join(folder_path, "quantile_preds_inv.npy"), quantile_inv)

    # 24. 计算 P50 点预测的标准误差指标
    # metric() 函数计算：MAE、MSE、RMSE、MAPE、MSPE
    mae, mse, rmse, mape, mspe = metric(preds_p50, trues)
    
    # 25. 打印评估指标（用于快速了解模型性能）
    print(f"P50 Test Metrics: MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}")
    
    # 26. 返回结果保存目录路径（供后续可视化和结果处理使用）
    return folder_path


def _get_setting(args, itr: int = 0) -> str:
    """生成唯一的实验配置标识字符串，包含所有关键超参数，用于区分检查点和结果文件夹"""
    weather_log1p_tag = _format_int_list(getattr(args, "weather_log1p_channels", []))
    signature = (
        f"{args.task_name}_{args.model_id}_{args.model}_e2e_"
        f"sl{args.seq_len}_pl{args.pred_len}_dm{args.d_model}_"
        f"el{args.e_layers}_wd{args.weather_feature_dim}_"
        f"wzn{int(bool(getattr(args, 'use_weather_normalization', False)))}_"
        f"wlog{weather_log1p_tag}_"
        f"sdp{int(bool(getattr(args, 'use_similar_day_prior', False)))}_"
        f"sdk{int(getattr(args, 'similar_day_top_k', 0))}_"
        f"sdfh{int(getattr(args, 'similar_day_fusion_hidden_dim', 0))}_"
        f"wk{args.weather_kernel_height}x{args.weather_kernel_width}_"
        f"lr{args.learning_rate}_"
        f"bs{args.batch_size}_{args.des}_{itr}"
    )
    # 使用 MD5 摘要缩短过长的文件夹名
    digest = hashlib.md5(signature.encode("utf-8")).hexdigest()[:8]
    return (
        f"TimeXerE2E_sl{args.seq_len}_pl{args.pred_len}_"
        f"wd{args.weather_feature_dim}_"
        f"wzn{int(bool(getattr(args, 'use_weather_normalization', False)))}_"
        f"sdp{int(bool(getattr(args, 'use_similar_day_prior', False)))}_"
        f"sdk{int(getattr(args, 'similar_day_top_k', 0))}_"
        f"wk{args.weather_kernel_height}x{args.weather_kernel_width}_"
        f"bs{args.batch_size}_{args.des}_{itr}_{digest}"
    )


def main() -> None:
    """
    程序主入口：环境初始化、数据集构建、模型定义及工作流执行。
    
    执行流程：
    1. 固定随机种子保证实验可重复性
    2. 构建命令行参数配置
    3. 加载可选的最优参数（测试模式）
    4. 初始化计算设备和气象数据存储
    5. 实例化端到端模型并验证配置
    6. 执行训练或测试工作流
    7. 生成预测结果可视化和基线对比
    """
    # 1. 固定随机种子，保证实验可重复性
    # 所有随机操作（数据划分、权重初始化、dropout 等）都会使用同一种子
    fix_seed = 2026
    random.seed(fix_seed)         # Python 内置随机数生成器
    torch.manual_seed(fix_seed)   # PyTorch CPU 随机数生成器
    np.random.seed(fix_seed)      # NumPy 随机数生成器

    # 2. 构造 argparse Namespace 模拟命令行输入参数
    # 这样做的好处是可以直接修改常量 TRAIN_MODE、BATCH_SIZE 等来改变实验配置
    args = argparse.Namespace(
        # 2.1 基础任务配置
        task_name=TASK_NAME,                # 任务名称："long_term_forecast"
        is_training=1 if TRAIN_MODE else 0, # 训练模式标志：1=训练，0=测试
        model_id=MODEL_ID,                  # 模型 ID："HunanLoad_uk_672_96_FullMapConv_E2E"
        model=MODEL,                        # 模型架构："TimeXer"
        data="custom",                      # 数据类型："custom" 表示自定义数据加载器
        
        # 2.2 数据路径配置
        root_path=ROOT_PATH,                # 数据根目录："./data/"
        data_path=DATA_PATH,                # 数据文件：湖南省电力负荷 CSV 文件
        features=FEATURES,                  # 任务类型："MS" 多变量输入单变量输出
        target=TARGET,                      # 目标列名："load"（电力负荷）
        freq="15min",                       # 数据频率：15分钟一条记录
        embed="timeF",                      # 时间嵌入方式："timeF" 表示使用时间特征
        checkpoints="./checkpoints_quantile/",  # 检查点保存目录
        
        # 2.3 时间窗口配置
        seq_len=SEQ_LEN,                    # 历史序列长度：672 步（7 天 × 96）
        label_len=LABEL_LEN,                # 标签长度：0（分离模式不需要标签）
        pred_len=PRED_LEN,                  # 预测长度：96 步（24 小时）
        
        # 2.4 TimeXer 网络超参数
        enc_in=ENC_IN,                      # 编码器输入通道：1（仅负荷本身）
        c_out=C_OUT,                        # 输出通道：1（单变量输出）
        d_model=D_MODEL,                    # 隐藏层维度：512
        n_heads=N_HEADS,                    # 注意力头数：4
        e_layers=E_LAYERS,                  # 编码器层数：2
        d_ff=D_FF,                          # FFN 维度：2048
        factor=FACTOR,                      # 注意力因子：3（稀疏注意力）
        dropout=DROPOUT,                    # 规则化丢弃率：0.1
        activation=ACTIVATION,              # 激活函数："gelu"
        patch_len=PATCH_LEN,                # Patch 长度：96
        use_norm=USE_NORM,                  # 是否使用层归一化：1
        
        # 2.5 气象参数扩展（全图卷积编码器）
        weather_h5_specs=WEATHER_H5_SPECS,  # HDF5 气象数据文件规范
        weather_in_channels=WEATHER_IN_CHANNELS,  # 气象通道数：10
        weather_feature_dim=WEATHER_FEATURE_DIM,  # 气象特征降维维度：3
        weather_grid_height=WEATHER_GRID_HEIGHT,  # 网格高度：62
        weather_grid_width=WEATHER_GRID_WIDTH,    # 网格宽度：61
        weather_kernel_height=WEATHER_KERNEL_HEIGHT,  # 卷积核高度：62（全图）
        weather_kernel_width=WEATHER_KERNEL_WIDTH,    # 卷积核宽度：61（全图）
        weather_encode_chunk_size=WEATHER_ENCODE_CHUNK_SIZE,  # 模型并行 chunk 大小：512
        weather_seq_len=WEATHER_SEQ_LEN,   # 气象序列长度：768 步（包含未来 24 小时预报）
        use_weather_normalization=USE_WEATHER_NORMALIZATION,  # 启用气象通道级标准化：True
        weather_log1p_channels=WEATHER_LOG1P_CHANNELS,        # 先做 log1p 的长尾通道：[9]
        weather_norm_fit_chunk_size=WEATHER_NORM_FIT_CHUNK_SIZE,  # 拟合统计量分块大小：512
        weather_normalization_eps=WEATHER_NORMALIZATION_EPS,  # 标准差下界保护
        
        # 2.6 数据加载配置
        num_workers=NUM_WORKERS,            # 数据加载线程数：0（Windows 下防止 HDF5 冲突）
        pin_memory=PIN_MEMORY,              # 固定内存加速：True
        contiguous_train_batches=CONTIGUOUS_TRAIN_BATCHES,  # 连续批次采样：True
        
        # 2.7 训练配置
        itr=ITR,                            # 迭代次数：1
        train_epochs=TRAIN_EPOCHS,          # 训练 epoch 数：30
        batch_size=BATCH_SIZE,              # 批次大小：64
        patience=PATIENCE,                  # 早期停止耐心值：5
        learning_rate=LEARNING_RATE,        # 学习率：1e-4
        des=DES,                            # 实验描述："Exp"
        loss="Quantile",                    # 损失函数类型："Quantile"
        lradj="type1",                      # 学习率衰减策略："type1"
        use_amp=True,                       # 自动混合精度：True
        inverse_eval=INVERSE_EVAL,          # 反标准化评估：True
        
        # 2.8 GPU 与设备配置
        use_gpu=USE_GPU,                    # 使用 GPU：True
        gpu=GPU,                            # GPU 设备号：0
        use_multi_gpu=False,                # 多 GPU 模式：False
        devices="0,1,2,3",                  # 可用 GPU 列表
        
        # 2.9 分位数预测配置
        quantiles=QUANTILES,                # 分位数列表：[0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        n_quantiles=N_QUANTILES,            # 分位数数量：7
        
        # 2.10 相似日先验配置
        use_similar_day_prior=USE_SIMILAR_DAY_PRIOR,  # 使用相似日先验：True
        similar_day_top_k=SIMILAR_DAY_TOP_K,          # 检索 Top-K：3
        similar_day_artifact_dir=SIMILAR_DAY_ARTIFACT_DIR,  # 离线特征库目录
        similar_day_fusion_hidden_dim=SIMILAR_DAY_FUSION_HIDDEN_DIM,  # 融合隐藏维度：128
    )

    # 3. 如果处于测试模式且启用了从 /use 导入参数，则加载最优参数和权重
    if not TRAIN_MODE and LOAD_FROM_USE:
        args = _apply_use_artifacts(args)

    # 4. CUDA 设备检测与初始化
    if torch.cuda.is_available() and args.use_gpu:
        # 如果 CUDA 可用且配置允许使用 GPU，则使用指定的 GPU
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU: cuda:{args.gpu}")
    else:
        # 否则使用 CPU
        device = torch.device("cpu")
        print("Using CPU")

    # 5. 初始化气象数据存储池
    # WeatherGridStore 负责从 HDF5 文件中高效读取气象网格数据
    weather_store = WeatherGridStore(
        args.weather_h5_specs,                      # HDF5 文件规范
        expected_in_channels=args.weather_in_channels,  # 预期通道数：10
        fill_value=WEATHER_FILL_VALUE,              # 缺失值填充为 0.0
        use_channel_normalization=bool(getattr(args, "use_weather_normalization", False)),
        log1p_channels=getattr(args, "weather_log1p_channels", ()),
        normalization_eps=float(getattr(args, "weather_normalization_eps", 1e-6)),
    )
    
    try:
        # 6. 验证气象网格尺寸与卷积核尺寸是否匹配（全图卷积要求一致）
        if weather_store.frame_shape is None:
            raise RuntimeError("weather_store.frame_shape is not initialized.")
        _, frame_height, frame_width = weather_store.frame_shape
        if (frame_height, frame_width) != (args.weather_kernel_height, args.weather_kernel_width):
            raise ValueError(
                "Weather frame size does not match full-map kernel size: "
                f"frame=({frame_height}, {frame_width}), "
                f"kernel=({args.weather_kernel_height}, {args.weather_kernel_width})"
            )

        # 7. 实例化端到端模型（全图卷积编码器 + TimeXer）
        model = FullMapConvTimeXerQuantile(args, quantiles=args.quantiles).float().to(device)
        
        # 8. 打印模型参数统计信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Full-Map Conv + TimeXer total params: {total_params:,}")
        print(f"Full-Map Conv + TimeXer trainable params: {trainable_params:,}")

        # 9. 生成实验配置标识（用于区分检查点和结果文件夹）
        setting = _get_setting(args)
        
        # 10. 执行训练或测试工作流
        if TRAIN_MODE:
            # 10.1 训练模式：从头开始训练模型
            print(f"\n>>> Start training {setting}")
            # 执行训练流程（返回加载了最优权重的模型）
            model = train_quantile_model(model, args, device, weather_store)

            # 10.2 训练完毕后，在测试集上进行评估
            print(f"\n>>> Start testing {setting}")
            results_dir = test_quantile_model(model, args, device, weather_store)
        else:
            # 10.3 测试模式：加载现有权重后进行推理
            # 优先级：getattr(args, "load_weight_path") > 默认路径
            ckpt_path = getattr(args, "load_weight_path", None)
            if ckpt_path is None:
                ckpt_path = os.path.join(args.checkpoints, setting, "checkpoint.pth")
            
            # 10.4 加载权重文件
            if os.path.exists(ckpt_path):
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                print(f"Loaded model: {ckpt_path}")
            else:
                raise FileNotFoundError(
                    f"Model file not found: {ckpt_path}. Please set TRAIN_MODE = True first."
                )

            # 10.5 执行测试评估
            print(f"\n>>> Test only {setting}")
            results_dir = test_quantile_model(model, args, device, weather_store)

        # 11. 生成预测结果对比图（P50 点预测 vs 真实值）
        plot_pred_vs_true(
            results_dir,                            # 结果保存目录
            use_inverse=INVERSE_EVAL,               # 是否反标准化到原始量纲
            quantiles=args.quantiles,               # 分位数列表
            title_prefix="Full-Map Conv + TimeXer Prediction",  # 图表标题前缀
            y_label="Load (MW)",                    # Y 轴标签
        )
        
        # 12. 执行相似日检索基线（离线特征库 Top-K 检索）
        # 为未来预测起点检索 Top-K 个历史相似日的负荷曲线
        similar_day_result = export_similar_day_baseline(
            results_dir=results_dir,                 # 结果保存目录
            future_path=FUTURE_PATH,                 # 未来数据 CSV 文件
            artifact_dir=SIMILAR_DAY_ARTIFACT_DIR,   # 离线特征库目录
            top_k=SIMILAR_DAY_TOP_K,                 # 检索 Top-K
        )
        
        # 13. 基于真实气象预报数据的未来负荷预测（生成 CSV 导出）
        # 使用模型在未来真实气象条件下预测接下来 96 小时（24 小时整天）的负荷
        predict_future_load_from_csv(
            model=model,                            # 训练好的模型
            args=args,                              # 参数配置
            device=device,                          # 计算设备
            weather_store=weather_store,            # 气象数据存储
            results_dir=results_dir,                # 结果保存目录
            future_path=FUTURE_PATH,                # 未来数据 CSV 文件（包含气象条件）
            steps=PRED_LEN,                         # 预测步长：96
            use_inverse=INVERSE_EVAL,               # 是否反标准化
            quantiles=args.quantiles,               # 分位数列表
            data_provider_fn=weather_data_provider, # 数据加载函数
            model_label="Full-Map Conv + TimeXer",  # 模型标签
            y_label="Load (MW)",                    # Y 轴标签
            similar_day_result=similar_day_result,  # 相似日检索结果（用于对比）
        )
    
    finally:
        # 14. 清理资源（确保 HDF5 文件句柄被正确关闭）
        weather_store.close()
        
        # 15. 清空 GPU 缓存（释放显存，防止内存泄漏）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # 程序入口点：执行主函数
    main()
