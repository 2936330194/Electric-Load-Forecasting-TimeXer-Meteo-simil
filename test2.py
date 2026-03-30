"""
test4_smp_old_metro_672.py - Simple Full-Map Conv + TimeXer end-to-end training

核心改动：
1. 用单层全图卷积替换 ConvNeXt-Tiny。
2. 每个卷积核覆盖整个 62x61 网格，等价于对全省全部格点做一组可学习的加权汇总。
3. 气象卷积模块、TimeXer、Quantile Head 统一组成一个模型并统一保存/加载权重。
4. 历史气象特征进入 TimeXer 编码器输入；预测阶段完全采用 encoder-only 路径。

注意：
- 这是端到端版本，DataLoader 仍然直接返回原始气象网格。
- 全图卷积比 ConvNeXt 简单很多，但原始天气张量仍然较大，建议保持较小 batch_size。
- HDF5 读取建议保持 num_workers=0。
"""

import argparse
import hashlib
import os
import random
import time
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import optim

from utils.forecast_visualization import plot_pred_vs_true, predict_future_load_from_csv
from utils.metrics import metric
from utils.quantile import QuantileLoss
from utils.weather_e2e import (
    FullMapConvTimeXerQuantile as ExogenousFullMapConvTimeXerQuantile,
    WeatherGridStore,
    infer_weather_history_len,
    weather_data_provider,
)
from utils.tools import EarlyStopping, adjust_learning_rate


# ==================== 分位数配置 ====================
QUANTILES = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]  # 需要输出的分位点列表
N_QUANTILES = len(QUANTILES)  # 分位点个数
P50_IDX = QUANTILES.index(0.5)  # P50 在分位点列表中的位置
P10_IDX = QUANTILES.index(0.1)  # P10 在分位点列表中的位置
P90_IDX = QUANTILES.index(0.9)  # P90 在分位点列表中的位置


# ==================== 基础任务配置 ====================
TASK_NAME = "long_term_forecast"  # 任务类型：长时序预测
MODEL = "TimeXer"  # 主体时序模型名称
MODEL_ID_PREFIX = "HunanLoad_2024_672"  # 实验标识名前缀，实际会根据天气分辨率动态拼接
CHECKPOINTS_DIR = "./checkpoints_test2/"


# ==================== 数据配置 ====================
ROOT_PATH = "./data/"  # 数据根目录
DATA_PATH = "湖南省电力负荷2024.csv"  # 历史负荷数据文件
FUTURE_PATH = "./data/湖南省电力负荷2024_future.csv"  # 未来预测时间文件
TARGET = "load"  # 目标列名
FEATURES = "MS"  # 多变量输入、单变量输出


# ==================== 时序长度配置 ====================
SEQ_LEN = 96 * 7  # 输入历史长度：7 天
LABEL_LEN = 0  # 解码端已知标签长度
PRED_LEN = 96  # 预测长度：1 天
LOAD_FREQ = "15min"  # 负荷序列采样频率


# ==================== Full-Map Conv 气象配置 ====================
WEATHER_SOURCE_CONFIGS = {
    "15min": [
        ("./data/hunan_grid_2024_filtered_15min.h5", "2024-01-01 00:00:00", "15min"),
    ],
    "1h": [
        ("./data/hunan_grid_2024_2025_filtered.h5", "2024-01-01 00:00:00", "1h"),
    ],
}  # (气象HDF5路径, 文件起始时间, 时间分辨率)
DEFAULT_WEATHER_SOURCE = "15min"  # 气象变量采样频率
WEATHER_IN_CHANNELS = 5  # 气象变量通道数
WEATHER_GRID_HEIGHT = 62  # 气象网格高度
WEATHER_GRID_WIDTH = 61  # 气象网格宽度
WEATHER_KERNEL_HEIGHT = 62  # 全图卷积核高度
WEATHER_KERNEL_WIDTH = 61  # 全图卷积核宽度
WEATHER_FEATURE_DIM = 3  # 每个时刻输出的气象特征维度
WEATHER_ENCODE_CHUNK_SIZE = 2048  # 气象帧分块编码大小
WEATHER_FILL_VALUE = 0.0  # 气象缺失时的填充值
WEATHER_FUTURE_LEN = 0  # 当前脚本仍只使用历史天气；未来天气长度保持为 0


# ==================== TimeXer 模型配置 ====================
ENC_IN = 1  # 外生模式下内生输入仅包含负荷
C_OUT = 1  # 输出通道数
D_MODEL = 256  # 隐藏层特征维度
N_HEADS = 4  # 多头注意力头数
E_LAYERS = 3  # 编码器层数
D_FF = 1024  # 前馈网络维度
FACTOR = 3  # 注意力因子
DROPOUT = 0.1  # dropout 比例
ACTIVATION = "gelu"  # 激活函数
PATCH_LEN = 96  # TimeXer patch 长度
USE_NORM = 1  # 是否启用归一化


# ==================== 训练配置 ====================
TRAIN_EPOCHS = 50  # 最大训练轮数
BATCH_SIZE = 32  # 训练批大小
LEARNING_RATE = 1e-4  # 学习率
PATIENCE = 5  # 早停容忍轮数
NUM_WORKERS = 0  # DataLoader 进程数


# ==================== 硬件配置 ====================
USE_GPU = True  # 是否使用 GPU
GPU = 0  # 使用的 GPU 编号


# ==================== 运行配置 ====================
DES = "Exp"  # 实验描述后缀
ITR = 1  # 实验重复次数
INVERSE_EVAL = True  # 评估时是否反标准化
TRAIN_MODE = False  # True 为训练+测试，False 为仅加载测试


# ==================== 数据通路配置 ====================
PIN_MEMORY = True  # CUDA DataLoader 使用锁页内存，配合 non_blocking 传输
CONTIGUOUS_TRAIN_BATCHES = True  # 训练阶段按连续窗口分块组 batch，提升重叠帧复用率

# 说明：
# - 当 FEATURES="MS" 时，TimeXer 的输入会被组织成“气象特征 + 历史负荷”，输出只预测目标负荷。

def _use_non_blocking_transfer(args, device: torch.device) -> bool:
    return (
        device.type == "cuda"
        and bool(getattr(args, "pin_memory", False))
        and torch.cuda.is_available()
        and args.use_gpu
    )


def _to_float_device(tensor: torch.Tensor, device: torch.device, non_blocking: bool = False) -> torch.Tensor:
    return tensor.to(device=device, dtype=torch.float32, non_blocking=non_blocking)


def _to_long_device(tensor: torch.Tensor, device: torch.device, non_blocking: bool = False) -> torch.Tensor:
    return tensor.to(device=device, dtype=torch.long, non_blocking=non_blocking)


def extract_target(batch_y: torch.Tensor) -> torch.Tensor:
    # 当前数据集只有一个监督目标“负荷”。
    # 保留最后一维是为了与 [B, pred_len, 1] 的模型输出保持一致。
    return batch_y[:, :, -1:]


def _parse_cli_args() -> argparse.Namespace:
    """
    解析脚本入口参数，支持在 1h / 15min 两套气象文件之间直接切换。
    """
    parser = argparse.ArgumentParser(
        description="Full-Map Conv + TimeXer quantile forecast with switchable weather resolutions."
    )
    parser.add_argument(
        "--weather-source",
        type=str,
        choices=sorted(WEATHER_SOURCE_CONFIGS.keys()),
        default=DEFAULT_WEATHER_SOURCE,
        help="选择使用的气象数据源：1h 或 15min。",
    )
    return parser.parse_args()


def _resolve_weather_h5_specs(weather_source: str) -> List[Tuple[str, str, str]]:
    """
    根据 CLI 指定的数据源名称，返回对应的 HDF5 配置。
    """
    if weather_source not in WEATHER_SOURCE_CONFIGS:
        raise ValueError(
            f"Unsupported weather_source={weather_source}. "
            f"Available: {sorted(WEATHER_SOURCE_CONFIGS.keys())}"
        )
    return list(WEATHER_SOURCE_CONFIGS[weather_source])


def _configure_runtime_weather_args(
    args: argparse.Namespace,
    weather_store: WeatherGridStore,
    weather_source: str,
) -> argparse.Namespace:
    """
    根据当前选中的气象 HDF5 的原生频率，自动推导天气窗口长度。
    这样 1h 数据会自动得到 168 步天气历史，15min 数据会自动得到 672 步天气历史。
    """
    if weather_store.native_freq is None:
        raise RuntimeError("weather_store.native_freq is not initialized.")

    weather_history_len = infer_weather_history_len(
        seq_len=args.seq_len,
        load_freq=args.freq,
        weather_freq=weather_store.native_freq,
    )
    weather_seq_len = int(weather_history_len + WEATHER_FUTURE_LEN)
    weather_step_freq = weather_store.native_freq_str or str(weather_store.native_freq)

    args.weather_source = weather_source
    args.weather_step_freq = weather_step_freq
    args.weather_mark_freq = weather_step_freq
    args.weather_history_len = int(weather_history_len)
    args.weather_seq_len = int(weather_seq_len)
    args.model_id = (
        f"{MODEL_ID_PREFIX}_{weather_source}_{weather_step_freq}Wx_"
        f"wh{args.weather_history_len}_pl{args.pred_len}_FullMapConv_Exo"
    )

    print(
        "[weather-config] "
        f"source={weather_source}, step={weather_step_freq}, "
        f"history={args.weather_history_len}, future={args.weather_seq_len - args.weather_history_len}, "
        f"seq={args.weather_seq_len}"
    )
    return args


def validate_quantile(model, data_loader, criterion, args, device, use_amp=False):
    """
    网络验证流程。执行给定的验证数据集的前向推理而不参与梯度下降更新。
    计算所有有效批次上预测值和标签的指定分位数综合损失。

    Args:
        model: 待验证的深度学习模型实例。
        data_loader: 用于提供验证数据的 DataLoader。
        criterion: 损失函数，通常为 QuantileLoss。
        args: 全局配置参数对象。
        device: 执行计算的计算设备（CPU 或 CUDA）。
        use_amp: 是否在推理时启用自动混合精度（AMP）。

    Returns:
        float: 验证集上的平均损失值。如果数据为空则返回 np.nan。
    """
    # 将模型切换到评估模式 (Inference Mode)
    # 这会停用 Dropout、BatchNorm 的更新等仅在训练时需要的行为
    model.eval()
    total_loss = []
    
    # 根据硬件配置决定是否使用非阻塞数据传输
    use_non_blocking = _use_non_blocking_transfer(args, device)

    # 使用 inference_mode 禁用梯度计算，比 no_grad 更轻量高效
    with torch.inference_mode():
        for batch_x, batch_y, batch_x_mark, batch_exo_mark, batch_weather_frames, batch_weather_index in data_loader:
            # 将该批次的各类张量异步或同步转移至目标计算设备
            # batch_x: 历史负荷序列 [B, L, 1]
            # batch_y: 目标负荷序列 (包含历史和预测) [B, L+P, 1]
            # batch_x_mark: 编码器端时间特征 (如小时、星期)
            # batch_exo_mark: 外生变量特征 (如节假日)
            # batch_weather_frames: 原始气象格点数据 [B, T, C, H, W]
            # batch_weather_index: 气象帧的时间戳索引，用于模型内部对齐
            batch_x = _to_float_device(batch_x, device, non_blocking=use_non_blocking)
            batch_y = _to_float_device(batch_y, device, non_blocking=use_non_blocking)
            batch_x_mark = _to_float_device(batch_x_mark, device, non_blocking=use_non_blocking)
            batch_exo_mark = _to_float_device(batch_exo_mark, device, non_blocking=use_non_blocking)
            batch_weather_frames = _to_float_device(batch_weather_frames, device, non_blocking=use_non_blocking)
            batch_weather_index = _to_long_device(batch_weather_index, device, non_blocking=use_non_blocking)

            # 启用 GPU 端的自动混合精度 (AMP)，通过 FP16 加速推理并减少显存占用
            with torch.amp.autocast('cuda', enabled=use_amp):
                # 验证阶段走与训练阶段完全相同的前向路径
                # 传入端到端模型，内部会先由全图卷积提取气象特征，再送入 TimeXer 做预测
                outputs = model(
                    load_x=batch_x,
                    x_mark_enc=batch_x_mark,
                    x_exo_mark=batch_exo_mark,
                    weather_x=batch_weather_frames,
                    weather_x_index=batch_weather_index,
                )

                # 从 batch_y 中提取预测目标部分的真实值 (通常是序列最后 pred_len 个点)
                batch_y_target = extract_target(batch_y[:, -args.pred_len :, :])
                
                # 计算预测输出与真实值的损失 (Quantile Loss)
                loss = criterion(outputs, batch_y_target)
            
            # 记录批次损失值
            total_loss.append(loss.item())

    # 验证结束后务必将模型恢复至训练模式，以便后续可能的训练循环继续
    model.train()
    
    # 计算并返回所有批次的平均损失
    return float(np.average(total_loss)) if total_loss else np.nan


def train_quantile_model(model, args, device, weather_store: WeatherGridStore):
    """
    执行基于全气象卷积模块和 TimeXer 的端到端模型训练调度主流程。
    主要使用了早停技术 (Early Stopping) 和混合精度运算 (AMP) 来做优化加速。
    在训练周期内不断通过 DataLoader 循环提供不同分段的联合训练数据送入模型以利用分位数损失拟合并迭代参数。

    Args:
        model: 待训练的端到端模型实例（包含天气卷积和时序预测部分）。
        args: 全局配置参数，包含学习率、Batch Size、训练轮数等。
        device: 计算设备（CPU 或 CUDA）。
        weather_store: 气象数据存储方案，提供高效的格点气象检索。

    Returns:
        torch.nn.Module: 训练完成并加载了最佳权重后的模型实例。
    """
    # 初始化训练集与验证集的数据加载器 (DataLoader)
    # 采用端到端模式，气象格点数据将在训练过程中实时从 HDF5 中读取并组装
    _, train_loader = weather_data_provider(args, "train", weather_store)
    _, vali_loader = weather_data_provider(args, "val", weather_store)
    _, test_loader = weather_data_provider(args, "test", weather_store)

    # 获取实验配置指纹并创建对应的 Checkpoint 模型权重保存路径
    setting = _get_setting(args)
    path = os.path.join(args.checkpoints, setting)
    os.makedirs(path, exist_ok=True)

    # 初始化 Adam 优化器，通过学习率控制权重更新步长
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 初始化分位数损失函数，用于指导模型学习不同概率分布下的不确定性
    criterion = QuantileLoss(args.quantiles).to(device)
    
    # 初始化早停器，防止模型在验证集上性能退化导致过拟合
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    # 自动混合精度 (AMP) 配置：仅在检测到 CUDA 环境时启用，以平衡计算速度与数值精度
    use_amp = bool(getattr(args, 'use_amp', False)) and device.type == 'cuda'
    # GradScaler 用于在混合精度下缩放梯度，防止 FP16 精度导致的梯度下溢
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    use_non_blocking = _use_non_blocking_transfer(args, device)

    # 打印训练启动配置信息
    print("\n" + "=" * 72)
    print("Start training Full-Map Conv + TimeXer end-to-end quantile model")
    print(f"setting: {setting}")
    print(f"quantiles: {args.quantiles}")
    print(f"weather_feature_dim: {args.weather_feature_dim}")
    print(f"weather_kernel_size: ({args.weather_kernel_height}, {args.weather_kernel_width})")
    print(
        f"weather_seq_len: {args.weather_seq_len} "
        f"(history={args.weather_history_len}, future={args.weather_seq_len - args.weather_history_len}, "
        f"step={getattr(args, 'weather_step_freq', 'native')})"
    )
    print(f"batch_size: {args.batch_size}")
    print(f"use_amp: {use_amp}")
    
    # 如果启用了基于重叠感知的气象批处理，则计算每批次的格点总数（用于性能监控）
    if bool(getattr(args, "contiguous_train_batches", False)):
        dense_weather_frames = args.batch_size * args.weather_seq_len
        print(
            "overlap-aware weather batching: "
            f"on (dense {dense_weather_frames} exogenous frames/batch)"
        )
    print("=" * 72)

    # 开始跨 Epoch 迭代训练
    for epoch in range(args.train_epochs):
        model.train()  # 确保模型处于训练模式，启用 Dropout/BatchNorm
        train_loss = []
        epoch_time = time.time()

        # 遍历数据加载器中的所有批次数据
        for i, (batch_x, batch_y, batch_x_mark, batch_exo_mark, batch_weather_frames, batch_weather_index) in enumerate(train_loader):
            # 清除旧梯度，set_to_none=True 能带来微弱的显存和速度提升
            optimizer.zero_grad(set_to_none=True)

            # 数据分发至设备
            batch_x = _to_float_device(batch_x, device, non_blocking=use_non_blocking)
            batch_y = _to_float_device(batch_y, device, non_blocking=use_non_blocking)
            batch_x_mark = _to_float_device(batch_x_mark, device, non_blocking=use_non_blocking)
            batch_exo_mark = _to_float_device(batch_exo_mark, device, non_blocking=use_non_blocking)
            batch_weather_frames = _to_float_device(batch_weather_frames, device, non_blocking=use_non_blocking)
            batch_weather_index = _to_long_device(batch_weather_index, device, non_blocking=use_non_blocking)

            # 使用混合精度上下文管理器进行前向计算
            with torch.amp.autocast('cuda', enabled=use_amp):
                # 调用模型 forward 接口
                # 流程：格点气象 -> 全图卷积 -> 特征映射 -> TimeXer Encoder -> Quantile Head
                outputs = model(
                    load_x=batch_x,
                    x_mark_enc=batch_x_mark,
                    x_exo_mark=batch_exo_mark,
                    weather_x=batch_weather_frames,
                    weather_x_index=batch_weather_index,
                )

                # 提取目标标签并计算分位数损失
                batch_y_target = extract_target(batch_y[:, -args.pred_len :, :])
                loss = criterion(outputs, batch_y_target)

            # 存储该批次的标量损失值
            train_loss.append(loss.item())

            # 执行混合精度的反向传播与优化器更新
            # 1. 缩放损失值
            # 2. 计算缩放后的梯度
            # 3. 检查是否有 Inf/NaN，若无则执行 optimizer.step() 否则跳过
            # 4. 更新缩放因子
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 每隔 50 个迭代打印一次进度
            if (i + 1) % 100 == 0:
                print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")

        # 完成一个 Epoch 训练后，在验证集上评估当前模型性能
        # 注意：validate_quantile 内部会执行 model.eval()，执行完后会复原为 model.train()
        vali_loss = validate_quantile(model, vali_loader, criterion, args, device, use_amp=use_amp)
        test_loss = validate_quantile(model, test_loader, criterion, args, device, use_amp=use_amp)
        train_loss_avg = float(np.average(train_loss)) if train_loss else np.nan
        print(
            f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.1f}s | "
            f"Train: {train_loss_avg:.7f} Vali: {vali_loss:.7f} Test: {test_loss:.7f}"
        )

        # 提交验证损失给早停触发器
        # 若验证性能优于历史最佳，则会保存当前模型权重至 Checkpoint 路径
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # 每个 Epoch 结束后根据策略（如 Decay）灵活调整学习率
        adjust_learning_rate(optimizer, epoch + 1, args)

    # 整个训练过程结束后，自动加载本次训练过程中验证性能最好的那份模型权重
    best_model_path = os.path.join(path, "checkpoint.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"Loaded best model weights: {best_model_path}")
    
    return model


def test_quantile_model(model, args, device, weather_store: WeatherGridStore):
    """
    模型测试流程：
    运用训练完毕的最佳模型在独立的测试集上验证泛化预测性能。
    保存预测结果与对应的所有分位数值、真实值以供后续计算或可视化程序使用。

    Args:
        model: 训练完毕后的模型实例。
        args: 全局配置参数。
        device: 计算设备。
        weather_store: 气象数据存储方案。

    Returns:
        str: 存储测试结果的文件夹路径。
    """
    # 初始化测试集数据加载器，该数据加载器包含未参与训练的样本
    # test_data 对象包含数据标准化器，后续用于反标准化 (inverse_transform)
    test_data, test_loader = weather_data_provider(args, "test", weather_store)

    # 按照实验设置生成唯一的文件夹名，用于隔离不同实验的结果
    setting = _get_setting(args)
    folder_path = os.path.join("./results/", setting)
    os.makedirs(folder_path, exist_ok=True)

    # 初始化存储预测值的容器
    preds_p50 = []           # 中位数预测结果 (P50)
    trues = []               # 真实负荷值
    quantile_preds_all = []  # 所有预测分位点结果

    # 确定是否在推理阶段启用 AMP 和非阻塞传输，以保持与训练环境的一致性
    use_amp = bool(getattr(args, 'use_amp', False)) and device.type == 'cuda'
    use_non_blocking = _use_non_blocking_transfer(args, device)

    # 进入评估模式并禁用梯度追踪，减少计算开销
    model.eval()
    with torch.inference_mode():
        for batch_x, batch_y, batch_x_mark, batch_exo_mark, batch_weather_frames, batch_weather_index in test_loader:
            # 数据迁移至目标计算设备
            batch_x = _to_float_device(batch_x, device, non_blocking=use_non_blocking)
            batch_y = _to_float_device(batch_y, device, non_blocking=use_non_blocking)
            batch_x_mark = _to_float_device(batch_x_mark, device, non_blocking=use_non_blocking)
            batch_exo_mark = _to_float_device(batch_exo_mark, device, non_blocking=use_non_blocking)
            batch_weather_frames = _to_float_device(batch_weather_frames, device, non_blocking=use_non_blocking)
            batch_weather_index = _to_long_device(batch_weather_index, device, non_blocking=use_non_blocking)

            # 混合精度前向推理
            with torch.amp.autocast('cuda', enabled=use_amp):
                # 获取模型输出，维度通常为 [Batch, Pred_Len, N_Quantiles]
                outputs = model(
                    load_x=batch_x,
                    x_mark_enc=batch_x_mark,
                    x_exo_mark=batch_exo_mark,
                    weather_x=batch_weather_frames,
                    weather_x_index=batch_weather_index,
                )

            # 提取真实值标签（取预测长度部分）
            batch_y_target = extract_target(batch_y[:, -args.pred_len :, :])
            
            # 单独提取 P50 (中位数) 预测，用于标准点估计指标计算
            p50_pred = outputs.float()[:, :, P50_IDX : P50_IDX + 1]

            # 将张量转为 NumPy 数组并移至 CPU，加入结果列表
            quantile_preds_all.append(outputs.float().detach().cpu().numpy())
            preds_p50.append(p50_pred.detach().cpu().numpy())
            trues.append(batch_y_target.detach().cpu().numpy())

    # 将各批次的列表拼接为完整的 NumPy 矩阵
    preds_p50 = np.concatenate(preds_p50, axis=0)
    trues = np.concatenate(trues, axis=0)
    quantile_preds_all = np.concatenate(quantile_preds_all, axis=0)

    # 打印测试集的总样本规模和预测维度
    print(
        f"Test shape: preds={preds_p50.shape}, "
        f"trues={trues.shape}, quantiles={quantile_preds_all.shape}"
    )

    # 将原始预测结果（标准化后的空间）保存至硬盘，便于离线分析
    np.save(os.path.join(folder_path, "pred.npy"), preds_p50)
    np.save(os.path.join(folder_path, "true.npy"), trues)
    np.save(os.path.join(folder_path, "quantile_preds.npy"), quantile_preds_all)

    # 如果数据曾被标准化，则需要进行反标准化 (Inverse Transform) 还原到 MW 量级
    if test_data.scale:
        shape = trues.shape
        # 对 P50 预测和真实值做全局反标准化
        preds_inv = test_data.inverse_transform_target(preds_p50.reshape(shape[0] * shape[1], -1)).reshape(shape)
        trues_inv = test_data.inverse_transform_target(trues.reshape(shape[0] * shape[1], -1)).reshape(shape)

        q_shape = quantile_preds_all.shape
        quantile_inv = np.zeros_like(quantile_preds_all)
        # 核心逻辑：每个分位点都必须单独做反标准化。
        # 理由：多变量反标准化器通常期望最后一维是不同的物理变量，
        # 而此处最后一维是同一变量的不同分位数，混合处理可能导致维度含义冲突。
        for qi in range(N_QUANTILES):
            q_slice = quantile_preds_all[:, :, qi : qi + 1]
            q_inv = test_data.inverse_transform_target(
                q_slice.reshape(q_shape[0] * q_shape[1], -1)
            ).reshape(q_shape[0], q_shape[1], 1)
            quantile_inv[:, :, qi] = q_inv[:, :, 0]

        # 保存反标准化后的物理意义数值结果
        np.save(os.path.join(folder_path, "pred_inv.npy"), preds_inv)
        np.save(os.path.join(folder_path, "true_inv.npy"), trues_inv)
        np.save(os.path.join(folder_path, "quantile_preds_inv.npy"), quantile_inv)

    # 计算并打印 P50 预测的点估计评价指标 (MSE, MAE, RMSE)
    if test_data.scale and getattr(args, 'inverse_eval', False):
        mae, mse, rmse, mape, mspe = metric(preds_inv, trues_inv)
        print(f"P50 Test Metrics (Inverse): MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}")
    else:
        mae, mse, rmse, mape, mspe = metric(preds_p50, trues)
        print(f"P50 Test Metrics (Normalized): MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}")

    # 返回结果存储路径，供后续绘图流程调用
    return folder_path


def _get_setting(args, itr=0):
    """
    按照各种超参设置格式化生成特征标识短串以用作模型相关存储记录的主目录名。
    包含特征指纹哈希计算，以防止不一样的参数训练相互覆盖文件。
    """
    # 使用短目录名 + hash，避免 Windows 下路径过长带来的访问异常问题。
    signature = (
        f"{args.task_name}_{args.model_id}_{args.model}_e2e_"
        f"sl{args.seq_len}_pl{args.pred_len}_dm{args.d_model}_"
        f"el{args.e_layers}_wd{args.weather_feature_dim}_"
        f"wsl{args.weather_seq_len}_wh{args.weather_history_len}_"
        f"wk{args.weather_kernel_height}x{args.weather_kernel_width}_"
        f"lr{args.learning_rate}_"
        f"bs{args.batch_size}_{args.des}_{itr}"
    )
    digest = hashlib.md5(signature.encode("utf-8")).hexdigest()[:8]
    return (
        f"TimeXerE2E_sl{args.seq_len}_pl{args.pred_len}_"
        f"wd{args.weather_feature_dim}_"
        f"wsl{args.weather_seq_len}_wh{args.weather_history_len}_"
        f"wk{args.weather_kernel_height}x{args.weather_kernel_width}_"
        f"bs{args.batch_size}_{args.des}_{itr}_{digest}"
    )


def main():
    """
    主程序执行入口。
    用于建立全局配置与种子分配机制、整合创建环境依赖实例，依次调度执行整个过程(建立存储->验证/训练->测试数据检验->未来数值直推测绘等)。
    """
    # 统一初始化随机种子以确保实验结果的可复现性
    fix_seed = 2026
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    cli_args = _parse_cli_args()
    selected_weather_source = cli_args.weather_source
    selected_weather_h5_specs = _resolve_weather_h5_specs(selected_weather_source)

    # 构造 argparse.Namespace 对象，集中管理模型的所有超参数
    # 这种方式简化了参数在函数间的传递，无需手动维护繁琐的参数列表
    args = argparse.Namespace(
        # ---------- 基础任务与实验配置 ----------
        task_name=TASK_NAME,             # 任务名称，如长时序预测
        is_training=1 if TRAIN_MODE else 0, # 标识当前是否处于训练状态
        model_id=MODEL_ID_PREFIX,        # 实验唯一标识 ID，后续会根据天气频率动态补全
        model=MODEL,                     # 模型名称 (TimeXer)
        des=DES,                         # 实验描述后缀
        itr=ITR,                         # 实验重复运行次数

        # ---------- 数据集路径与目标列设置 ----------
        data="custom",                   # 使用自定义数据集格式
        root_path=ROOT_PATH,             # 数据根目录
        data_path=DATA_PATH,             # 历史负荷数据 CSV 文件名
        features=FEATURES,               # 特征模式 (MS: 多变量输入，单变量输出)
        target=TARGET,                   # 负荷目标列名
        target_channel_idx=0,            # 目标列在多变量矩阵中的索引位置
        freq=LOAD_FREQ,                  # 数据采样频率 (15分钟一个点)
        embed="timeF",                   # 时间编码方式 (TimeFeature)
        checkpoints=CHECKPOINTS_DIR, # 权重保存基目录

        # ---------- 时序长度配置 (负荷端) ----------
        seq_len=SEQ_LEN,                 # 历史负荷回顾窗口长度
        label_len=LABEL_LEN,             # TimeXer Decoder 开始时的重叠起始长度 (本实验中为 0)
        pred_len=PRED_LEN,               # 预测未来的长度 (96 = 1天)

        # ---------- TimeXer 模型架构参数 ----------
        enc_in=ENC_IN,                   # 编码器输入通道数 (仅负荷)
        c_out=C_OUT,                     # 解码器输出通道数
        d_model=D_MODEL,                 # 隐藏层特征维度
        n_heads=N_HEADS,                 # 多头注意力头数
        e_layers=E_LAYERS,               # 编码器层数
        d_ff=D_FF,                       # 前馈全连接层维度
        factor=FACTOR,                   # ProbSparse 注意力因子
        dropout=DROPOUT,                 # Dropout 比例
        activation=ACTIVATION,           # 激活函数 (GELU)
        patch_len=PATCH_LEN,             # TimeXer 切片 (Patch) 长度
        use_norm=USE_NORM,               # 是否使用归一化层

        # ---------- Full-Map Conv 气象格点参数 ----------
        weather_source=selected_weather_source,  # 当前选中的气象数据源标签
        weather_h5_specs=selected_weather_h5_specs, # 气象 HDF5 文件配置列表
        weather_in_channels=WEATHER_IN_CHANNELS, # 原始气象通道数 (如温、压、湿、风)
        weather_feature_dim=WEATHER_FEATURE_DIM, # 卷积压缩后输出的各时刻气象特征维度
        weather_grid_height=WEATHER_GRID_HEIGHT, # 输入格点高度
        weather_grid_width=WEATHER_GRID_WIDTH,   # 输入格点宽度
        weather_kernel_height=WEATHER_KERNEL_HEIGHT, # 卷积核高度 (与格点对齐实现全图覆盖)
        weather_kernel_width=WEATHER_KERNEL_WIDTH,   # 卷积核宽度
        weather_encode_chunk_size=WEATHER_ENCODE_CHUNK_SIZE, # 气象序列分块推理大小，防止 OOM
        use_weather_normalization=True,          # 显式开启气象数据通道归一化

        # ---------- 训练与优化配置 ----------
        num_workers=NUM_WORKERS,         # Pytorch DataLoader 并行线程数
        pin_memory=PIN_MEMORY,           # 是否使用锁页内存加速 CUDA 数据搬运
        contiguous_train_batches=CONTIGUOUS_TRAIN_BATCHES, # 是否启用重叠气象帧复用策略
        train_epochs=TRAIN_EPOCHS,       # 最大训练轮数
        batch_size=BATCH_SIZE,           # 训练批大小
        patience=PATIENCE,               # 早停容忍步数
        learning_rate=LEARNING_RATE,     # 初始学习率
        loss="Quantile",                 # 损失函数类型：分位数损失
        lradj="cosine",                   # 学习率调整策略 cosine

        # ---------- 硬件加速与 GPU 配置 ----------
        use_amp=True,                    # 是否开启自动混合精度训练 (AMP)
        inverse_eval=INVERSE_EVAL,       # 评估指标计算前是否执行反标准化
        use_gpu=USE_GPU,                 # 是否启用 GPU
        gpu=GPU,                         # 使用第几个 GPU 编号
        use_multi_gpu=False,             # 是否启用多卡分布式并行 (本实验暂不启用)
        devices="0,1,2,3",               # 多卡时的设备 ID 列表

        # ---------- 分位数预测设置 ----------
        quantiles=QUANTILES,             # 分位点列表 (P2, P10, P25, P50, P75, P90, P98)
        n_quantiles=N_QUANTILES,         # 分位点数量
    )

    # 计算设备选择：优先使用 CUDA，否则回退到 CPU
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU: cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # WeatherGridStore 在主函数中只创建一次，这是全局唯一的气象数据管理中枢
    # 其内部缓存了 HDF5 文件句柄和时间索引，避免在不同阶段重复加载大型元数据
    # train/val/test/future_predict 全部复用此实例以提升 IO 效率
    weather_store = WeatherGridStore(
        args.weather_h5_specs,
        expected_in_channels=args.weather_in_channels,
        fill_value=WEATHER_FILL_VALUE,
        use_channel_normalization=True,
    )
    try:
        args = _configure_runtime_weather_args(args, weather_store, selected_weather_source)

        # 全图卷积 (Full-Map Conv) 对维度要求极严：卷积核尺寸必须与输入气象网格尺寸一致
        # 此处在模型初始化前执行硬检查，防止后续矩阵乘法时报维度不匹配错误
        if weather_store.frame_shape is None:
            raise RuntimeError("weather_store.frame_shape is not initialized.")
        _, frame_height, frame_width = weather_store.frame_shape
        if (frame_height, frame_width) != (args.weather_kernel_height, args.weather_kernel_width):
            raise ValueError(
                "Weather frame size does not match full-map kernel size: "
                f"frame=({frame_height}, {frame_width}), "
                f"kernel=({args.weather_kernel_height}, {args.weather_kernel_width})"
            )

        # 实例化端到端模型，并将参数转移至目标设备
        model = ExogenousFullMapConvTimeXerQuantile(args, quantiles=QUANTILES).float().to(device)
        
        # 统计并展示模型的总参数量和可训练参数量，这有助于评估服务器显存开销
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Full-Map Conv + TimeXer total params: {total_params:,}")
        print(f"Full-Map Conv + TimeXer trainable params: {trainable_params:,}")

        # 获取实验唯一标识名
        setting = _get_setting(args)

        if TRAIN_MODE:
            # ==================== 训练模式分支 ====================
            # 流程：模型训练 -> 最佳权重加载 -> 测试集泛化性评估 -> 可视化 -> 未来负荷外推
            
            print(f"\n>>> Start training {setting}")
            # 执行端到端训练，分位数损失会同步更新天气特征提取和时序挖掘两个模块
            model = train_quantile_model(model, args, device, weather_store)

            print(f"\n>>> Start testing {setting}")
            # 在独立测试集上计算 MSE/MAE 以及各分位数覆盖表现
            results_dir = test_quantile_model(model, args, device, weather_store)

            # 绘制测试集真实值与 P50 预测及其置信区间 (P10-P90) 的对比图
            plot_pred_vs_true(
                results_dir,
                use_inverse=INVERSE_EVAL,
                quantiles=args.quantiles,
                title_prefix="Full-Map Conv + TimeXer Prediction",
                y_label="Load (MW)",
            )
            
            # 使用 future.csv 指引的时间线，调取对应时刻的气象格点做未来 24-96 小时的外推预测
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
        else:
            # ==================== 仅推理/测试模式分支 ====================
            # 用于加载已训练好的成品模型，直接进行测试或部署应用
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

            # 同样生成可视化图表
            plot_pred_vs_true(
                results_dir,
                use_inverse=INVERSE_EVAL,
                quantiles=args.quantiles,
                title_prefix="Full-Map Conv + TimeXer Prediction",
                y_label="Load (MW)",
            )
            
            # 执行外推预测
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
        # ==================== 资源释放 ====================
        # 显式关闭 HDF5 文件句柄，防止多进程下出现文件锁定异常
        weather_store.close()
        # 清理显存碎片，确保不影响后续进程使用 GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
