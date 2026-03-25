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
from typing import List, Tuple

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
SIMILAR_DAY_ARTIFACT_DIR = "./artifacts/similar_day_retriever_ae_128"
SIMILAR_DAY_TOP_K = 3

# 时间窗口配置
SEQ_LEN = 96 * 7        # 历史观测窗口长度（672步，对应7天）
LABEL_LEN = 0           # 标签长度（TimeXer 分离模式下设为 0）
PRED_LEN = 96           # 预测窗口长度（24小时）
WEATHER_SEQ_LEN = SEQ_LEN + PRED_LEN    # 重要：气象数据序列包含未来 24 小时预报，总长 768

# 气象 HDF5 配置
WEATHER_H5_SPECS: List[Tuple[str, str, str]] = [
    ("./data/hunan_grid_meteo_20250101_20260228.h5", "2025-01-01 00:00:00", "15min"),
]
WEATHER_IN_CHANNELS = 10        # 气象网格通道数（10个气象参数）
WEATHER_GRID_HEIGHT = 62        # 网格高度
WEATHER_GRID_WIDTH = 61         # 网格宽度
WEATHER_KERNEL_HEIGHT = 62      # 全图卷积核高度
WEATHER_KERNEL_WIDTH = 61       # 全图卷积核宽度
WEATHER_FEATURE_DIM = 3         # 气象降维后的特征维度
WEATHER_ENCODE_CHUNK_SIZE = 512 # 模型并行时的 chunk 大小
WEATHER_FILL_VALUE = 0.0        # 缺失值填充

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
TRAIN_MODE = False       # True 为训练模式，False 为仅推理测试

# 性能优化参数
PIN_MEMORY = True               # 开启固定内存加速数据搬运
CONTIGUOUS_TRAIN_BATCHES = True # 开启连续批次采样以大幅优化气象数据加载速度

# /use 导入配置
LOAD_FROM_USE = True  # 是否从 /use 导入最优参数和权重（仅测试模式有效）
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


def _load_json_file(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _apply_use_artifacts(args: argparse.Namespace) -> argparse.Namespace:
    use_dir = os.path.abspath(USE_DIR)
    config_path = os.path.join(use_dir, USE_BEST_CONFIG_FILE)
    params_path = os.path.join(use_dir, USE_BEST_PARAMS_FILE)
    weight_path = os.path.join(use_dir, USE_BEST_WEIGHT_FILE)

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"/use 权重文件不存在: {weight_path}")

    if os.path.exists(config_path):
        payload = _load_json_file(config_path)
        if not isinstance(payload, dict):
            raise ValueError(f"/use 配置文件内容不是 JSON 对象: {config_path}")
        for key, value in payload.items():
            setattr(args, key, value)
    elif os.path.exists(params_path):
        payload = _load_json_file(params_path)
        if not isinstance(payload, dict):
            raise ValueError(f"/use 超参数文件内容不是 JSON 对象: {params_path}")
        for raw_key, value in payload.items():
            key = TUNABLE_PARAM_MAP.get(str(raw_key), str(raw_key))
            setattr(args, key, value)
    else:
        raise FileNotFoundError(
            f"/use 中未找到 {USE_BEST_CONFIG_FILE} 或 {USE_BEST_PARAMS_FILE}"
        )

    args.is_training = 0
    args.load_weight_path = weight_path
    args.weather_seq_len = int(getattr(args, "weather_seq_len", args.seq_len + args.pred_len))
    args.n_quantiles = len(args.quantiles)
    print(f"已从 /use 导入参数与权重: {weight_path}")
    return args


def export_similar_day_baseline(
    results_dir: str,
    future_path: str,
    artifact_dir: str = SIMILAR_DAY_ARTIFACT_DIR,
    top_k: int = SIMILAR_DAY_TOP_K,
) -> None:
    """
    基于离线构建好的相似日检索库，为未来预测起点检索 Top-K 历史负荷曲线，
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
        return

    # 2. 动态导入相似日检索模块，增加独立运行的鲁棒性
    try:
        from similar_day_retriever import DEFAULT_ARTIFACT_DIR, SimilarDayRetriever, print_retrieval_result
    except Exception as exc:
        print(f"导入 similar_day_retriever 失败，跳过检索: {exc}")
        return

    # 3. 确定并校验离线特征库目录（结合参数传入与模块默认值）
    resolved_artifact_dir = os.path.abspath(
        str(artifact_dir) if artifact_dir is not None else str(DEFAULT_ARTIFACT_DIR)
    )
    if not os.path.isdir(resolved_artifact_dir):
        print(f"未找到相似日模型库目录，跳过检索: {resolved_artifact_dir}")
        return

    # 4. 尝试加载检索模型并执行基于预测起点文件的 Top-K 检索
    try:
        retriever = SimilarDayRetriever.load(resolved_artifact_dir)
        result = retriever.search_from_future_csv(abs_future_path, top_k=int(top_k))
    except Exception as exc:
        print(f"执行相似日检索失败: {exc}")
        return

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


def validate_quantile(model, data_loader, criterion, args, device, use_amp: bool = False) -> float:
    """
    分位数预测验证函数。
    计算测试集或验证集上的平均 Quantile Loss。
    """
    model.eval()
    total_loss = []
    use_non_blocking = _use_non_blocking_transfer(args, device)

    with torch.inference_mode():
        for batch_x, batch_y, batch_x_mark, batch_exo_mark, batch_weather_frames, batch_weather_index in data_loader:
            # 数据异步搬运到 GPU
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

            with torch.amp.autocast("cuda", enabled=use_amp):
                # 前向传播调用（分离模式：负荷、负荷时间标、气象时间标、气象帧、气象索引）
                outputs = model(
                    load_x=batch_x,
                    x_mark_enc=batch_x_mark,
                    x_exo_mark=batch_exo_mark,
                    weather_x=batch_weather_frames,
                    weather_x_index=batch_weather_index,
                )
                batch_y_target = extract_target(batch_y[:, -args.pred_len :, :])
                loss = criterion(outputs, batch_y_target)

            total_loss.append(loss.item())

    model.train()
    return float(np.average(total_loss)) if total_loss else np.nan


def train_quantile_model(model, args, device, weather_store: WeatherGridStore):
    """
    主训练训练循环逻辑。
    包含优化器设置、AMP 混合精度、分位数 Loss 计算以及早期停止检查。
    """
    _, train_loader = weather_data_provider(args, "train", weather_store)
    _, vali_loader = weather_data_provider(args, "val", weather_store)

    setting = _get_setting(args)
    path = os.path.join(args.checkpoints, setting)
    os.makedirs(path, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = QuantileLoss(args.quantiles).to(device)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    # 混合精度加速（针对 30/40 系显卡显著提速）
    use_amp = bool(getattr(args, "use_amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    use_non_blocking = _use_non_blocking_transfer(args, device)

    print("\n" + "=" * 72)
    print("Start training Full-Map Conv + TimeXer end-to-end quantile model")
    print(f"setting: {setting}")
    print(f"quantiles: {args.quantiles}")
    print(f"weather_feature_dim: {args.weather_feature_dim}")
    print(f"weather_seq_len: {getattr(args, 'weather_seq_len', args.seq_len)} (seq_len={args.seq_len} + pred_len={args.pred_len})")
    print(f"weather_kernel_size: ({args.weather_kernel_height}, {args.weather_kernel_width})")
    print(f"batch_size: {args.batch_size}")
    print(f"use_amp: {use_amp}")
    
    # 打印优化提速相关的信息
    weather_seq_len = getattr(args, 'weather_seq_len', args.seq_len)
    if bool(getattr(args, "contiguous_train_batches", False)):
        dense_weather_frames = args.batch_size * weather_seq_len
        unique_weather_frames = weather_seq_len + args.batch_size - 1
        print(
            "overlap-aware weather batching: "
            f"on (dense {dense_weather_frames} frames/batch -> about {unique_weather_frames} unique frames/batch)"
        )
    print("=" * 72)

    for epoch in range(args.train_epochs):
        model.train()
        train_loss = []
        epoch_time = time.time()

        for i, (batch_x, batch_y, batch_x_mark, batch_exo_mark, batch_weather_frames, batch_weather_index) in enumerate(
            train_loader
        ):
            optimizer.zero_grad(set_to_none=True)

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

            with torch.amp.autocast("cuda", enabled=use_amp):
                # 调用端到端模型
                outputs = model(
                    load_x=batch_x,
                    x_mark_enc=batch_x_mark,
                    x_exo_mark=batch_exo_mark,
                    weather_x=batch_weather_frames,
                    weather_x_index=batch_weather_index,
                )
                batch_y_target = extract_target(batch_y[:, -args.pred_len :, :])
                loss = criterion(outputs, batch_y_target)

            train_loss.append(loss.item())
            
            # AMP 梯度更新
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (i + 1) % 20 == 0:
                print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")

        # 每一个 epoch 结束后进行验证
        vali_loss = validate_quantile(model, vali_loader, criterion, args, device, use_amp=use_amp)
        train_loss_avg = float(np.average(train_loss)) if train_loss else np.nan
        print(
            f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.1f}s | "
            f"Train: {train_loss_avg:.7f} Vali: {vali_loss:.7f}"
        )

        # 早期停止逻辑
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # 学习率衰减
        adjust_learning_rate(optimizer, epoch + 1, args)

    # 导出并加载最优模型权重
    best_model_path = os.path.join(path, "checkpoint.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"Loaded best model weights: {best_model_path}")
    return model


def test_quantile_model(model, args, device, weather_store: WeatherGridStore) -> str:
    """
    模型测试与评估函数。
    计算 P50（中位数）的 MAE/MSE 指标，并保存所有分位点的预测结果到 .npy 文件。
    """
    test_data, test_loader = weather_data_provider(args, "test", weather_store)

    setting = _get_setting(args)
    folder_path = os.path.join("./results/", setting)
    os.makedirs(folder_path, exist_ok=True)

    preds_p50 = []
    trues = []
    quantile_preds_all = []

    use_amp = bool(getattr(args, "use_amp", False)) and device.type == "cuda"
    use_non_blocking = _use_non_blocking_transfer(args, device)

    model.eval()
    with torch.inference_mode():
        for batch_x, batch_y, batch_x_mark, batch_exo_mark, batch_weather_frames, batch_weather_index in test_loader:
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

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(
                    load_x=batch_x,
                    x_mark_enc=batch_x_mark,
                    x_exo_mark=batch_exo_mark,
                    weather_x=batch_weather_frames,
                    weather_x_index=batch_weather_index,
                )

            batch_y_target = extract_target(batch_y[:, -args.pred_len :, :])
            # 提取 P50 分位数作为点预测对比
            p50_pred = outputs.float()[:, :, P50_IDX : P50_IDX + 1]

            quantile_preds_all.append(outputs.float().detach().cpu().numpy())
            preds_p50.append(p50_pred.detach().cpu().numpy())
            trues.append(batch_y_target.detach().cpu().numpy())

    # 合并批次数据
    preds_p50 = np.concatenate(preds_p50, axis=0)
    trues = np.concatenate(trues, axis=0)
    quantile_preds_all = np.concatenate(quantile_preds_all, axis=0)

    print(
        f"Test shape: preds={preds_p50.shape}, "
        f"trues={trues.shape}, quantiles={quantile_preds_all.shape}"
    )

    # 保存原始（缩放后）数据
    np.save(os.path.join(folder_path, "pred.npy"), preds_p50)
    np.save(os.path.join(folder_path, "true.npy"), trues)
    np.save(os.path.join(folder_path, "quantile_preds.npy"), quantile_preds_all)

    # 如果有标准化，则反转回原始量纲进行保存
    if test_data.scale:
        shape = trues.shape
        preds_inv = test_data.inverse_transform_target(preds_p50.reshape(shape[0] * shape[1], -1)).reshape(
            shape
        )
        trues_inv = test_data.inverse_transform_target(trues.reshape(shape[0] * shape[1], -1)).reshape(shape)

        q_shape = quantile_preds_all.shape
        quantile_inv = np.zeros_like(quantile_preds_all)
        for qi in range(N_QUANTILES):
            q_slice = quantile_preds_all[:, :, qi : qi + 1]
            q_inv = test_data.inverse_transform_target(
                q_slice.reshape(q_shape[0] * q_shape[1], -1)
            ).reshape(q_shape[0], q_shape[1], 1)
            quantile_inv[:, :, qi] = q_inv[:, :, 0]

        np.save(os.path.join(folder_path, "pred_inv.npy"), preds_inv)
        np.save(os.path.join(folder_path, "true_inv.npy"), trues_inv)
        np.save(os.path.join(folder_path, "quantile_preds_inv.npy"), quantile_inv)

    # 计算标准误差指标
    mae, mse, rmse, mape, mspe = metric(preds_p50, trues)
    print(f"P50 Test Metrics: MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}")
    return folder_path


def _get_setting(args, itr: int = 0) -> str:
    """生成唯一的实验配置标识字符串，包含所有关键超参数，用于区分检查点和结果文件夹"""
    signature = (
        f"{args.task_name}_{args.model_id}_{args.model}_e2e_"
        f"sl{args.seq_len}_pl{args.pred_len}_dm{args.d_model}_"
        f"el{args.e_layers}_wd{args.weather_feature_dim}_"
        f"wk{args.weather_kernel_height}x{args.weather_kernel_width}_"
        f"lr{args.learning_rate}_"
        f"bs{args.batch_size}_{args.des}_{itr}"
    )
    # 使用 MD5 摘要缩短过长的文件夹名
    digest = hashlib.md5(signature.encode("utf-8")).hexdigest()[:8]
    return (
        f"TimeXerE2E_sl{args.seq_len}_pl{args.pred_len}_"
        f"wd{args.weather_feature_dim}_"
        f"wk{args.weather_kernel_height}x{args.weather_kernel_width}_"
        f"bs{args.batch_size}_{args.des}_{itr}_{digest}"
    )


def main() -> None:
    """程序主入口：环境初始化、数据集构建、模型定义及工作流执行"""
    # 固定随机种子，保证实验可重复性
    fix_seed = 2026
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # 构造 argparse Namespace 模拟命令行输入参数
    args = argparse.Namespace(
        task_name=TASK_NAME,
        is_training=1 if TRAIN_MODE else 0,
        model_id=MODEL_ID,
        model=MODEL,
        data="custom",
        root_path=ROOT_PATH,
        data_path=DATA_PATH,
        features=FEATURES,
        target=TARGET,
        freq="15min",
        embed="timeF",
        checkpoints="./checkpoints_quantile/",
        seq_len=SEQ_LEN,
        label_len=LABEL_LEN,
        pred_len=PRED_LEN,
        enc_in=ENC_IN,
        c_out=C_OUT,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        e_layers=E_LAYERS,
        d_ff=D_FF,
        factor=FACTOR,
        dropout=DROPOUT,
        activation=ACTIVATION,
        patch_len=PATCH_LEN,
        use_norm=USE_NORM,
        # 气象参数扩展
        weather_h5_specs=WEATHER_H5_SPECS,
        weather_in_channels=WEATHER_IN_CHANNELS,
        weather_feature_dim=WEATHER_FEATURE_DIM,
        weather_grid_height=WEATHER_GRID_HEIGHT,
        weather_grid_width=WEATHER_GRID_WIDTH,
        weather_kernel_height=WEATHER_KERNEL_HEIGHT,
        weather_kernel_width=WEATHER_KERNEL_WIDTH,
        weather_encode_chunk_size=WEATHER_ENCODE_CHUNK_SIZE,
        weather_seq_len=WEATHER_SEQ_LEN, # 768步
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        contiguous_train_batches=CONTIGUOUS_TRAIN_BATCHES,
        itr=ITR,
        train_epochs=TRAIN_EPOCHS,
        batch_size=BATCH_SIZE,
        patience=PATIENCE,
        learning_rate=LEARNING_RATE,
        des=DES,
        loss="Quantile",
        lradj="type1",
        use_amp=True,
        inverse_eval=INVERSE_EVAL,
        use_gpu=USE_GPU,
        gpu=GPU,
        use_multi_gpu=False,
        devices="0,1,2,3",
        quantiles=QUANTILES,
        n_quantiles=N_QUANTILES,
    )

    if not TRAIN_MODE and LOAD_FROM_USE:
        args = _apply_use_artifacts(args)

    # CUDA 设备检测
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU: cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # 初始化气象数据存储池
    weather_store = WeatherGridStore(
        args.weather_h5_specs,
        expected_in_channels=args.weather_in_channels,
        fill_value=WEATHER_FILL_VALUE,
    )
    try:
        # 验证气象网格尺寸与卷积核尺寸是否匹配
        if weather_store.frame_shape is None:
            raise RuntimeError("weather_store.frame_shape is not initialized.")
        _, frame_height, frame_width = weather_store.frame_shape
        if (frame_height, frame_width) != (args.weather_kernel_height, args.weather_kernel_width):
            raise ValueError(
                "Weather frame size does not match full-map kernel size: "
                f"frame=({frame_height}, {frame_width}), "
                f"kernel=({args.weather_kernel_height}, {args.weather_kernel_width})"
            )

        # 实例化端到端模型
        model = FullMapConvTimeXerQuantile(args, quantiles=args.quantiles).float().to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Full-Map Conv + TimeXer total params: {total_params:,}")
        print(f"Full-Map Conv + TimeXer trainable params: {trainable_params:,}")

        setting = _get_setting(args)
        if TRAIN_MODE:
            # 执行训练流
            print(f"\n>>> Start training {setting}")
            model = train_quantile_model(model, args, device, weather_store)

            # 执行测试流
            print(f"\n>>> Start testing {setting}")
            results_dir = test_quantile_model(model, args, device, weather_store)
        else:
            # 仅测试模式：加载现有权重
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

        # 预测结果对比图绘制
        plot_pred_vs_true(
            results_dir,
            use_inverse=INVERSE_EVAL,
            quantiles=args.quantiles,
            title_prefix="Full-Map Conv + TimeXer Prediction",
            y_label="Load (MW)",
        )
        export_similar_day_baseline(
            results_dir=results_dir,
            future_path=FUTURE_PATH,
            artifact_dir=SIMILAR_DAY_ARTIFACT_DIR,
            top_k=SIMILAR_DAY_TOP_K,
        )
        # 运行基于真实气象预报的未来预测（CSV 导出）
        predict_future_load_from_csv(
            model=model,
            args=args,
            device=device,
            weather_store=weather_store,
            results_dir=results_dir,
            future_path=FUTURE_PATH,
            steps=PRED_LEN,
            use_inverse=INVERSE_EVAL,
            quantiles=args.quantiles,
            data_provider_fn=weather_data_provider,
            model_label="Full-Map Conv + TimeXer",
            y_label="Load (MW)",
        )
    finally:
        # 确保 HDF5 文件句柄被正确关闭
        weather_store.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
