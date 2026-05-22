"""
test1.py - TimeXer 概率预测脚本 (分位数回归) - 湖南省电力负荷预测

将 TimeXer 从点预测升级为概率预测：
- 使用分位数回归 (Quantile Regression) 方法
- 输出 7 个分位数: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
- P50 (中位数) 作为点预测，P10-P90 作为置信区间
- 在预测图中绘制 P10-P90 置信区间带

核心结构：
1. TimeXerQuantile 包装器：在原始 TimeXer 输出层后添加分位数映射头
2. QuantileLoss：替代 MSE 的分位数损失函数
3. 自定义训练/测试流程：不依赖 Exp_Long_Term_Forecast
4. 图表增加 P10-P90 置信区间带
"""

# ==================== 导入依赖模块 ====================
import argparse
import hashlib
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import os
import time
import json
import matplotlib.pyplot as plt

# 导入原始 TimeXer 模型和数据工厂
from models.TimeXer import Model as TimeXer
from data_provider.data_factory import data_provider
from utils.forecast_visualization import (
    plot_pred_vs_true as shared_plot_pred_vs_true,
    predict_future_load_from_csv as shared_predict_future_load_from_csv,
)
from utils.quantile import QuantileLoss as SharedQuantileLoss
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, cal_eval, append_probabilistic_eval
from utils.timefeatures import time_features

CHECKPOINTS_DIR = "./checkpoints_test1/"

# ==================== 分位数配置 ====================
# 概率预测所使用的分位数列表，覆盖从极端低值(0.02)到极端高值(0.98)的分布范围。
# - 0.02 / 0.98：近似 96% 置信区间的上下界，用于极端情况预警
# - 0.10 / 0.90：常用的 80% 置信区间边界（P10-P90），报告与可视化的主要区间
# - 0.25 / 0.75：50% 置信区间（四分位距 IQR），反映预测的核心不确定性
# - 0.50：中位数（P50），作为概率预测中的"点预测"代表值
QUANTILES = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
N_QUANTILES = len(QUANTILES) # 分位数总数（7个），用于模型输出头维度设定和循环遍历

# 预先计算常用分位数在列表中的索引位置，避免重复查找：
P50_IDX = QUANTILES.index(0.5)   # 索引 3
P10_IDX = QUANTILES.index(0.1)   # 索引 1
P90_IDX = QUANTILES.index(0.9)   # 索引 5

# ==================== 任务与模型配置 ====================

TASK_NAME = "long_term_forecast" # 任务名称：长期预测任务（long_term_forecast），决定数据加载器和实验流程的分支逻辑
MODEL = "TimeXer" # 使用的基础模型架构名称，对应 models/TimeXer.py 中的 Model 类
MODEL_ID = "HunanLoad_2024_672_96" # 模型标识符，用于生成检查点/结果目录名，格式: {数据集}_{年份}_{输入长度}_{预测长度}

# ==================== 数据集配置 ====================
DATA = "custom" # 数据集类型标识：使用 "custom" 表示自定义 CSV 数据集（非内置基准数据集）
ROOT_PATH = "./data/" # 数据文件的根目录路径（相对路径）
DATA_PATH = "湖南省电力负荷2024.csv" # 原始电力负荷数据的 CSV 文件名，包含历史负荷时序数据
# 特征类型：
#   "S" = 单变量（仅使用目标变量自身的历史值进行预测）
#   "M" = 多变量（使用所有特征列）
#   "MS" = 多变量输入、单变量输出
FEATURES = "S"
TARGET = "load" # 预测目标列名，对应 CSV 文件中的负荷数据列
FUTURE_PATH = "./data/湖南省电力负荷2024_future.csv" # 未来时段数据文件路径，用于生成滚动式的超前预测（不含真实标签的纯输入数据）

# ==================== 序列长度配置 ====================
# 编码器输入序列长度（回看窗口）：
# 96 个时间步/天 × 7 天 = 672 个 15 分钟间隔 = 7 天的历史数据
# 选择 7 天以捕获完整的周周期性模式
SEQ_LEN = 96 * 7     # 672 个 15min = 7 天
# 解码器标签长度（label_len）：设置为 0 表示不向解码器提供已知的真实值前缀，
# 解码器输入完全由零填充组成（纯自回归预测模式）
LABEL_LEN = 0
# 预测长度：96 个 15 分钟间隔 = 24 小时 = 1 天的超前预测
PRED_LEN = 96 * 1       # 96 个 15min = 1 天

# ==================== 模型架构参数 ====================
ENC_IN = 1 # 编码器输入特征维度：单变量模式下为 1（仅负荷值）
C_OUT = 1 # 输出特征维度：预测单一目标变量，维度为 1
D_MODEL = 512  # 隐藏层特征维度
N_HEADS = 4  # 多头注意力头数
E_LAYERS = 3  # 编码器层数
D_FF = 2048  # 前馈网络维度
FACTOR = 3  # 注意力因子
DROPOUT = 0.1 # Dropout 比率：训练时随机丢弃 10% 的神经元，防止过拟合
ACTIVATION = "gelu" # 激活函数类型："gelu"（高斯误差线性单元），比 ReLU 更平滑，在 Transformer 架构中广泛使用
PATCH_LEN = 96 # Patch 长度：将输入序列按 96 步（1 天）分割为多个 patch，每个 patch 作为一个 token 输入 Transformer 编码器
USE_NORM = 1 # 是否启用实例归一化（RevIN）：1 = 启用，0 = 禁用

# ==================== 训练超参数 ====================

TRAIN_EPOCHS = 50 # 最大训练轮数：模型最多训练 50 个 epoch（若未被早停机制提前终止）
BATCH_SIZE = 32 # 每个 mini-batch 的样本数：32 是常用的批量大小
LEARNING_RATE = 1e-4 # 初始学习率：使用 Adam 优化器的初始步长为 1e-4
PATIENCE = 5 # 早停耐心值：验证集损失连续 5 个 epoch 未改善时停止训练，防止过拟合
NUM_WORKERS = 0 # 数据加载器的并行工作线程数：0 表示在主线程中同步加载数据

# ==================== 设备配置 ====================
USE_GPU = True # 是否启用 GPU 加速：True 表示优先使用 CUDA GPU 进行训练和推理
GPU = 0 # 指定使用的 GPU 设备编号（cuda:0 为第一块显卡）

# ==================== 其他配置 ====================
DES = "Exp" # 实验描述标签，用于生成检查点/结果目录名中的描述字段
ITR = 1 # 实验重复次数：当 ITR > 1 时可评估模型的训练稳定性和结果方差
INVERSE_EVAL = True # 是否在评估时对预测值和真实值做反标准化
TRAIN_MODE = True    # 训练模式开关

# ==================== /optuna 导入配置 ====================
# 可调参数名称映射表：将 best_params.json 中的大写键名映射为 args 中对应的小写属性名。
LOAD_FROM_OPTUNA = False
OPTUNA_DIR = "./optuna"
OPTUNA_BEST_PARAMS_FILE = "best_params1.json"
OPTUNA_BEST_CONFIG_FILE = "best_config1.json"
OPTUNA_BEST_WEIGHT_FILE = "best_model1.pth"
OPTUNA_BEST_TRIAL_FILE = "best_trial_result1.json"

TUNABLE_PARAM_MAP = {
    "D_MODEL": "d_model",       # Transformer 隐含层维度
    "N_HEADS": "n_heads",       # 多头注意力头数
    "E_LAYERS": "e_layers",     # 编码器层数
    "D_FF": "d_ff",             # 前馈网络隐含层维度
    "DROPOUT": "dropout",       # Dropout 比率
    "PATCH_LEN": "patch_len",   # Patch 分割长度
    "BATCH_SIZE": "batch_size", # 批量大小
    "LEARNING_RATE": "learning_rate",  # 学习率
}


def _load_json_file(json_path: str):
    """
    从指定路径读取并解析 JSON 文件。

    Args:
        json_path: JSON 文件的绝对或相对路径

    Returns:
        解析后的 Python 对象（通常为 dict 或 list）
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# TimeXer 分位数包装模型
# =============================================================================
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


class TimeXerQuantile(nn.Module):
    """
    TimeXer 分位数包装模型。

    在原始 TimeXer 模型的点预测输出之上，添加一个线性分位数输出头，
    将 [B, pred_len, 1] 的点预测映射为 [B, pred_len, n_quantiles] 的分位数预测。

    架构:
        输入 → TimeXer → [B, pred_len, 1]
                                ↓
                       quantile_head (Linear: 1 → n_quantiles)
                                ↓
                       [B, pred_len, n_quantiles]
    """
    def __init__(self, configs, quantiles=None):
        super(TimeXerQuantile, self).__init__()
        self.quantiles = quantiles if quantiles is not None else QUANTILES
        self.n_quantiles = len(self.quantiles)

        # 原始 TimeXer 模型
        self.timexer = TimeXer(configs)

        # 分位数输出头: 将 1 维点预测映射为 n_quantiles 维分位数预测
        self.quantile_head = nn.Linear(1, self.n_quantiles)

        # 初始化分位数头的偏置，权重接近1
        with torch.no_grad():
            self.quantile_head.weight.fill_(1.0)
            self.quantile_head.bias.copy_(
                torch.tensor([q - 0.5 for q in self.quantiles]) * 0.1
            )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        前向传播
        输入: 与原始 TimeXer 相同
        输出: [B, pred_len, n_quantiles]
        """
        # TimeXer 点预测: [B, pred_len, c_out]
        point_pred = self.timexer(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        # 取最后 pred_len 步
        point_pred = point_pred[:, -self.timexer.pred_len:, :]

        # 分位数映射: [B, pred_len, 1] → [B, pred_len, n_quantiles]
        quantile_pred = self.quantile_head(point_pred)

        return quantile_pred


# =============================================================================
# 工具函数
# =============================================================================
def restore_sliding_window_2d(data_2d: np.ndarray) -> np.ndarray:
    """还原二维滑动窗口数据为一维序列"""
    if len(data_2d) == 0:
        return np.array([])
    restored = list(data_2d[0, :])
    for i in range(1, len(data_2d)):
        restored.append(data_2d[i, -1])
    return np.asarray(restored)


def restore_sliding_window_3d(data_3d: np.ndarray) -> np.ndarray:
    """还原三维滑动窗口数据为二维序列"""
    if len(data_3d) == 0:
        return np.array([])
    restored = list(data_3d[0, :, :])
    for i in range(1, len(data_3d)):
        restored.append(data_3d[i, -1, :])
    return np.asarray(restored)


def _load_ordered_dataframe(csv_path: str, target: str) -> pd.DataFrame:
    """读取并整理数据列顺序，保证目标列在最后一列"""
    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        raise ValueError(f"缺少 date 列: {csv_path}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if target not in df.columns and "Target" in df.columns:
        df = df.rename(columns={"Target": target})
    if target not in df.columns:
        raise ValueError(f"缺少目标列 {target}: {csv_path}")
    other_cols = [c for c in df.columns if c not in ("date", target)]
    return df[["date"] + other_cols + [target]]


# =============================================================================
# 自定义训练流程
# =============================================================================
def train_quantile_model(model, args, device):
    """
    自定义训练流程，使用 QuantileLoss (分位数损失) 训练 TimeXerQuantile 概率预测模型。
    
    该函数实现了完整的 epoch 迭代训练方案，包括：
    1. 自回归解码器的 padding 构建
    2. 分位数损失计算与反向传播
    3. 验证集评估与早停（Early Stopping）机制
    4. 学习率动态衰减（Cosine Annealing）

    Args:
        model: 待训练的 TimeXerQuantile 模型实例
        args: 包含各类超参数的 Namespace 对象
        device: 计算设备（CPU 或 GPU）

    Returns:
        加载了验证集上具有最小损失（最佳权重）的模型
    """
    # ==================== 1. 加载数据 ====================
    # 调用数据工厂函数，获取训练集、验证集和测试集的 Dataset 和 DataLoader
    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    # ==================== 2. 创建检查点目录 ====================
    # 生成当前实验配置的唯一 hash 目录名，用于保存最优模型权重 (checkpoint.pth)
    setting = _get_setting(args)
    path = os.path.join(args.checkpoints, setting)
    os.makedirs(path, exist_ok=True)

    # ==================== 3. 核心组件初始化 ====================
    # 使用 Adam 优化器，学习率由 args.learning_rate 确定
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # 实例化自定义的分位数损失函数，传入定义的 7 个分位数列表
    criterion = SharedQuantileLoss(args.quantiles)
    # 初始化早停器：如果在 args.patience 设定的 epoch 数内，验证集 Loss 未降低，则停止训练
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    print(f"\n{'='*60}")
    print(f"开始训练 TimeXer-Quantile 模型")
    print(f"分位数: {QUANTILES}")
    print(f"{'='*60}")

    # ==================== 4. 开始 Epoch 循环训练 ====================
    train_start_time = time.time()
    for epoch in range(args.train_epochs):
        model.train()  # 切换模型到训练模式：启用 Dropout 和 BatchNorm/LayerNorm 更新
        train_loss = []
        epoch_time = time.time()  # 记录当前 epoch 的开始时间

        # 遍历训练数据加载器中的所有 mini-batch
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            # 清空上一轮迭代的梯度
            optimizer.zero_grad()

            # 将特征数据 (负荷时序) 和时间标记转移到计算设备
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # ----- 构造解码器输入 (Decoder Input) -----
            # 标准 Informer/TimeXer 的推理模式：已知部分（label_len）+ 未知部分（pred_len）
            # 对于未知部分，用全零张量填充占位；已知部分使用真实的历史观测值。
            # 注：本文档设 LABEL_LEN = 0，即属于纯自回归模式（无已知部分前缀）
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            # ----- 前向传播 -----
            # 将历史序列、时间标志及解码占位符输入模型
            # 输出形状: [Batch_size, pred_len, n_quantiles]
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            # ----- 取目标受试维度 -----
            # 如果特征模式是 'MS' (多变量输入单变量输出)，则只取最后一个通道（即 Target 列）
            f_dim = -1 if args.features == 'MS' else 0
            # 从真实标签 batch_y 中截取实际需要预测的 target 片段，维度 [B, pred_len, 1]
            batch_y_target = batch_y[:, -args.pred_len:, f_dim:]

            # ----- 计算分位数损失 并 梯度反传 -----
            # criterion 接受 [B, pred_len, n_quantiles] 的预测与 [B, pred_len, 1] 的目标，
            # 内部使用广播机制计算每个分位数对应的 Pinball Loss 并求均值
            loss = criterion(outputs, batch_y_target)
            train_loss.append(loss.item())

            loss.backward()  # 根据 Loss 计算梯度
            optimizer.step() # 使用优化器更新模型权重

            # 打印每 100 次迭代 (100 batches) 后的瞬时训练损失
            if (i + 1) % 100 == 0:
                print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")

        # ==================== 5. 验证与早停检查 ====================
        # 完成一个 epoch 的数据遍历后，计算模型在验证集和测试集上的整体分位数损失
        vali_loss = validate_quantile(model, vali_loader, criterion, args, device)
        test_loss = validate_quantile(model, test_loader, criterion, args, device)

        # 统计平均训练损失
        train_loss_avg = np.average(train_loss)
        print(f"Epoch: {epoch+1} cost time: {time.time()-epoch_time:.1f}s | "
              f"Train: {train_loss_avg:.7f} Vali: {vali_loss:.7f} Test: {test_loss:.7f}")

        # 将验证集 loss 传入早停器；若当前损失为历史最低，早停器会将当前权重保存至 path
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # ==================== 6. 学习率衰减 ====================
        # 按指定策略（如余弦退火）逐步减小学习率，帮助模型后期在损失曲面上更精细地收敛
        adjust_learning_rate(optimizer, epoch + 1, args)

    total_train_time = time.time() - train_start_time
    print(f"Total training time: {total_train_time:.1f}s ({total_train_time / 60:.2f} min)")

    # ==================== 7. 加载并返回最佳模型 ====================
    # 无论是正常跑完所有 epoch，还是中途触发早停，最终都需要回退到验证集性能最好的那组参数
    best_model_path = os.path.join(path, 'checkpoint.pth')
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"已加载最佳模型: {best_model_path}")

    return model


def validate_quantile(model, data_loader, criterion, args, device):
    """
    在验证集或测试集上评估模型的平均分位数损失 (Quantile/Pinball Loss)。

    Args:
        model: TimeXerQuantile 概率预测模型
        data_loader: 验证集或测试集的 DataLoader
        criterion: 损失函数 (QuantileLoss)
        args: 包含预测长度 (pred_len) 等超参数的配置对象
        device: 计算设备 (CPU/GPU)

    Returns:
        float: 数据集上的平均分位数损失
    """
    model.eval()  # 切换至评估模式，关闭 Dropout，固定 Batch Normalization 等层的状态
    total_loss = []
    
    with torch.no_grad():  # 禁用梯度计算，加速前向传播并节省显存
        for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
            # 将输入数据和时间标记迁移到对应计算设备
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # ----- 构造解码器占位输入 -----
            # 全零填充预测部分 (pred_len) ，历史部分使用真实标签中的序列 (label_len)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            # 模型前向传播，输出形状: [B, pred_len, n_quantiles]
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            # ----- 目标维度提取 -----
            # 'MS' 特征配置下，只预测多变量的最后一个通道，否则预测所有通道
            f_dim = -1 if args.features == 'MS' else 0
            batch_y_target = batch_y[:, -args.pred_len:, f_dim:].to(device) # [B, pred_len, 1]

            # 计算分位数损失并记录
            loss = criterion(outputs, batch_y_target)
            total_loss.append(loss.item())

    model.train()  # 恢复为训练模式以便后续继续训练
    return np.average(total_loss)


# =============================================================================
# 自定义测试流程
# =============================================================================
def test_quantile_model(model, args, device):
    """
    测试 TimeXerQuantile 模型，保存 P50 预测、真实值和全分位数输出至文件系统。

    该函数会加载测试集进行前向推理，执行反标准化（如果启用了 Scaling），
    并将预测结果保存为 numpy 数组 (.npy)，最后计算 P50 的点预测误差指标评估。

    保存的文件名说明:
        - pred.npy / true.npy: Standardized 尺度的 P50 点预测和真实值，形状 [N, pred_len, 1]
        - pred_inv.npy / true_inv.npy: 反标准化 (Inverse-transformed) 回物理量纲的 P50 和 真实值
        - quantile_preds.npy: Standardized 尺度的全分位数预测结果，形状 [N, pred_len, n_quantiles]
        - quantile_preds_inv.npy: 反标准化回物理量纲的全分位数预测结果
        
    Returns:
        保存预测结果数据的文件夹路径
    """
    # 加载用于测试的数据集及其 DataLoader
    test_data, test_loader = data_provider(args, 'test')

    # 生成本次实验的唯一标识目录，确保不同超参数组合的输出隔离
    setting = _get_setting(args)
    folder_path = os.path.join(getattr(args, "results_root", "./results/"), setting)
    os.makedirs(folder_path, exist_ok=True)

    # 预先分配列表，用于收集各个批次的预测和真实值
    preds_p50 = []         # P50 (第 50 百分位数中位数) 预测值
    trues = []             # 真实标签
    quantile_preds_all = []# 各个分位数的完整预测输出

    model.eval()  # 设置模型为评估模式
    with torch.no_grad(): # 推理过程关闭梯度计算
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            # 数据迁移到 GPU/CPU
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # 构建解码器占位张量 (Decoder Padding)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            # 模型前向输出：涵盖全部预设的 n_quantiles (例如 7 个分位数)
            # 输出张量形状: [Batch_size, pred_len, n_quantiles]
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            # 提取测试目标受试通道
            f_dim = -1 if args.features == 'MS' else 0
            batch_y_target = batch_y[:, -args.pred_len:, f_dim:]  # [Batch_size, pred_len, 1]

            # 提取中位数 P50(如 q=0.5) 对应的通道作为点预测代表
            p50_pred = outputs[:, :, P50_IDX:P50_IDX+1]  # [Batch_size, pred_len, 1]

            # 将张量断开计算图、转移至 CPU 内存并转为 numpy 数组
            quantile_np = outputs.detach().cpu().numpy()         # [B, pred_len, 7]
            p50_np = p50_pred.detach().cpu().numpy()             # [B, pred_len, 1]
            true_np = batch_y_target.detach().cpu().numpy()      # [B, pred_len, 1]

            # 收集每个批次的 numpy 数组
            preds_p50.append(p50_np)
            trues.append(true_np)
            quantile_preds_all.append(quantile_np)

    # 将所有批次的列表数据沿第一维度(样本数量维度 N)拼接
    preds_p50 = np.concatenate(preds_p50, axis=0)                    # [总体样本数 N, pred_len, 1]
    trues = np.concatenate(trues, axis=0)                            # [总体样本数 N, pred_len, 1]
    quantile_preds_all = np.concatenate(quantile_preds_all, axis=0)  # [总体样本数 N, pred_len, 7]

    print(f'Test shape: preds={preds_p50.shape}, trues={trues.shape}, quantiles={quantile_preds_all.shape}')

    # ==================== 保存数据 (Standardized Scale) ====================
    np.save(os.path.join(folder_path, 'pred.npy'), preds_p50)
    np.save(os.path.join(folder_path, 'true.npy'), trues)
    np.save(os.path.join(folder_path, 'quantile_preds.npy'), quantile_preds_all)

    # ==================== 反标准化预测结果 ====================
    # 模型输出通常在标准化后的空间内，需要使用 Dataset 中记录的 scaler 参数还原为实际数值
    if test_data.scale:
        shape = trues.shape # [N, pred_len, 1]
        
        # 将 P50预测值和真实值 从 [N, pred_len, 1] 展平为二维 [N * pred_len, 1]，执行反标准化，再恢复原有形状
        preds_inv = test_data.inverse_transform(preds_p50.reshape(shape[0]*shape[1], -1)).reshape(shape)
        trues_inv = test_data.inverse_transform(trues.reshape(shape[0]*shape[1], -1)).reshape(shape)

        # 遍历还原所有分位数对应的张量空间
        q_shape = quantile_preds_all.shape  # [N, pred_len, 7]
        quantile_inv = np.zeros_like(quantile_preds_all) # 初始化同等大小全零结果张量
        
        for qi in range(N_QUANTILES): # 逐通道处理 7 个分位数
            # 切片提取单个分位数: [N, pred_len, 1]
            q_slice = quantile_preds_all[:, :, qi:qi+1]  
            # 反标准化该切片通道
            q_inv = test_data.inverse_transform(q_slice.reshape(q_shape[0]*q_shape[1], -1)).reshape(q_shape[0], q_shape[1], 1)
            # 填回初始化好的 quantile_inv 的对应通道中
            quantile_inv[:, :, qi] = q_inv[:, :, 0]

        # 保存反标准化后的文件
        np.save(os.path.join(folder_path, 'pred_inv.npy'), preds_inv)
        np.save(os.path.join(folder_path, 'true_inv.npy'), trues_inv)
        np.save(os.path.join(folder_path, 'quantile_preds_inv.npy'), quantile_inv)

    origin_pred = preds_inv if test_data.scale else preds_p50
    origin_true = trues_inv if test_data.scale else trues
    origin_quantiles = quantile_inv if test_data.scale else quantile_preds_all
    origin_eval_df = cal_eval(origin_true, origin_pred)
    origin_eval_df = append_probabilistic_eval(origin_eval_df, origin_true, origin_quantiles, QUANTILES)
    print("[origin Eval] metrics:")
    print(origin_eval_df)

    # ==================== 指标计算评估 ====================
    # 对 P50 点预测结果在标准化空间内执行 MSE、MAE 等经典评估指标计算并打印输出
    mae, mse, rmse, mape, mspe = metric(preds_p50, trues)
    print(f'P50 Test Metrics: MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}')

    # 返回结果包含数据的文件夹路径，便利后续的可视化模块共享取用数据
    return folder_path


# =============================================================================
# 辅助函数
# =============================================================================
def _get_setting_legacy(args, itr=0):
    """生成实验设置字符串"""
    return (
        f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_"
        f"ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_"
        f"dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_"
        f"lr{args.learning_rate}_bs{args.batch_size}_"
        f"{args.des}_{itr}"
    )


# =============================================================================
# 主函数
# =============================================================================
def main():
    """
    主函数：训练和测试 TimeXer-Quantile 概率预测模型
    """
    # ==================== 设置随机种子 ====================
    fix_seed = 2026
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # ==================== 构建实验参数 ====================
    args = argparse.Namespace(
        task_name=TASK_NAME,
        is_training=1 if TRAIN_MODE else 0,
        model_id=MODEL_ID,
        model=MODEL,

        data=DATA,
        root_path=ROOT_PATH,
        data_path=DATA_PATH,
        features=FEATURES,
        target=TARGET,
        freq="min",
        embed="timeF",
        checkpoints="./checkpoints_quantile/",  # 使用独立的检查点目录

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

        target_channel_idx=0,

        num_workers=NUM_WORKERS,
        itr=ITR,
        train_epochs=TRAIN_EPOCHS,
        batch_size=BATCH_SIZE,
        patience=PATIENCE,
        learning_rate=LEARNING_RATE,
        des=DES,
        loss="Quantile",
        lradj="cosine",
        use_amp=True,
        inverse_eval=INVERSE_EVAL,

        use_gpu=USE_GPU,
        gpu=GPU,
        use_multi_gpu=False,
        devices="0,1,2,3",

        # 分位数配置
        quantiles=QUANTILES,
        n_quantiles=N_QUANTILES,
    )

    args.checkpoints = CHECKPOINTS_DIR
    args.results_root = "./results/"
    args.load_weight_path = None

    if not args.is_training:
        if LOAD_FROM_OPTUNA:
            args = _apply_optuna_artifacts(args)

    # ==================== 配置计算设备 ====================
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU: cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # ==================== 构建模型 ====================
    model = TimeXerQuantile(args, quantiles=args.quantiles).float().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"TimeXerQuantile 模型参数量: {total_params:,}")

    setting = _get_setting(args)

    # ==================== 训练或加载模型 ====================
    if args.is_training:
        print(f"\n>>> 开始训练: {setting}")
        model = train_quantile_model(model, args, device)

        print(f"\n>>> 开始测试: {setting}")
        results_dir = test_quantile_model(model, args, device)

        shared_plot_pred_vs_true(
            results_dir,
            use_inverse=args.inverse_eval,
            quantiles=args.quantiles,
            title_prefix="TimeXer Quantile Prediction",
        )
        shared_predict_future_load_from_csv(
            model=model, args=args, device=device,
            weather_store=None,
            results_dir=results_dir, future_path=FUTURE_PATH,
            steps=PRED_LEN, use_inverse=args.inverse_eval,
            quantiles=args.quantiles,
            data_provider_fn=data_provider,
            model_label="TimeXer",
        )
    else:
        # 加载已训练模型
        ckpt_path = getattr(args, "load_weight_path", None)
        if ckpt_path is None:
            ckpt_path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
        
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"成功加载模型: {ckpt_path}")
        else:
            raise FileNotFoundError(
                f"未找到模型文件 {ckpt_path}，请先设置 TRAIN_MODE = True 训练模型。"
            )

        print(f"\n>>> 仅测试: {setting}")
        results_dir = test_quantile_model(model, args, device)

        shared_plot_pred_vs_true(
            results_dir,
            use_inverse=args.inverse_eval,
            quantiles=args.quantiles,
            title_prefix="TimeXer Quantile Prediction",
        )
        shared_predict_future_load_from_csv(
            model=model, args=args, device=device,
            weather_store=None,
            results_dir=results_dir, future_path=FUTURE_PATH,
            steps=PRED_LEN, use_inverse=args.inverse_eval,
            quantiles=args.quantiles,
            data_provider_fn=data_provider,
            model_label="TimeXer",
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==================== 程序入口 ====================
def _build_full_setting_v2(args, itr=None):
    run_itr = args.itr if itr is None else itr
    return (
        f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_"
        f"ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_"
        f"dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_"
        f"lr{args.learning_rate}_bs{args.batch_size}_"
        f"{args.des}_{run_itr}"
    )


def _sanitize_setting_part(value, max_len):
    safe = "".join(ch if str(ch).isalnum() else "_" for ch in str(value))
    safe = safe.strip("_") or "na"
    return safe[:max_len]


def _get_setting(args, itr=None):
    """生成较短且稳定的实验目录名，避免 Windows 长路径错误。"""
    full_setting = _build_full_setting_v2(args, itr=itr)
    run_itr = args.itr if itr is None else itr
    digest = hashlib.md5(full_setting.encode("utf-8")).hexdigest()[:10]
    model_tag = _sanitize_setting_part(args.model_id, 18)
    return (
        f"test1_{model_tag}_sl{args.seq_len}_pl{args.pred_len}_"
        f"dm{args.d_model}_bs{args.batch_size}_itr{run_itr}_{digest}"
    )


if __name__ == "__main__":
    main()
