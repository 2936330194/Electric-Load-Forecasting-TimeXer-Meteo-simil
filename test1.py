"""
test3.py - TimeXer 概率预测脚本 (分位数回归) - 湖南省电力负荷预测

基于 test2.py 升级，将 TimeXer 从点预测升级为概率预测：
- 使用分位数回归 (Quantile Regression) 方法
- 输出 7 个分位数: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
- P50 (中位数) 作为点预测，P10-P90 作为置信区间
- 在预测图中绘制 P10-P90 置信区间带

核心改动：
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
from utils.quantile import QuantileLoss as SharedQuantileLoss
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, cal_eval
from utils.timefeatures import time_features

CHECKPOINTS_DIR = "./checkpoints_test1/"

# ==================== 分位数配置 ====================
QUANTILES = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
N_QUANTILES = len(QUANTILES)
P50_IDX = QUANTILES.index(0.5)   # 索引 3
P10_IDX = QUANTILES.index(0.1)   # 索引 1
P90_IDX = QUANTILES.index(0.9)   # 索引 5

# ==================== 任务与模型配置 ====================
TASK_NAME = "long_term_forecast"
MODEL = "TimeXer"
MODEL_ID = "HunanLoad_uk_672_96"

# ==================== 数据集配置 ====================
DATA = "custom"
ROOT_PATH = "./data/"
DATA_PATH = "湖南省电力负荷2024.csv"
FEATURES = "S"
TARGET = "load"
FUTURE_PATH = "./data/湖南省电力负荷2024_future.csv"

# ==================== 序列长度配置 ====================
SEQ_LEN = 96 * 7     # 672 个 15min = 7 天
LABEL_LEN = 0
PRED_LEN = 96        # 96 个 15min = 1 天

# ==================== 模型架构参数 ====================
ENC_IN = 1
C_OUT = 1
D_MODEL = 256
N_HEADS = 4
E_LAYERS = 3
D_FF = 1024
FACTOR = 3
DROPOUT = 0.1
ACTIVATION = "gelu"
PATCH_LEN = 96
USE_NORM = 1

# ==================== 训练超参数 ====================
TRAIN_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
PATIENCE = 5
NUM_WORKERS = 0

# ==================== 设备配置 ====================
USE_GPU = True
GPU = 0

# ==================== 其他配置 ====================
DES = "Exp"
ITR = 1
INVERSE_EVAL = True
TRAIN_MODE = True    # 模型重训练

# ==================== 未来协变量配置 ====================
USE_FUTURE_COVARIATES = False
FUTURE_COV_DIM = 0
FUTURE_COV_DROPOUT = 0.1


# ==================== /use 导入配置 ====================
LOAD_FROM_USE = False  # 是否从 /use 导入最优参数和权重（仅测试模式有效）
USE_DIR = "./use"
USE_BEST_PARAMS_FILE = "best_params.json"
USE_BEST_CONFIG_FILE = "best_config.json"
USE_BEST_WEIGHT_FILE = "best_model.pth"
TUNABLE_PARAM_MAP = {
    "D_MODEL": "d_model",
    "N_HEADS": "n_heads",
    "E_LAYERS": "e_layers",
    "D_FF": "d_ff",
    "DROPOUT": "dropout",
    "PATCH_LEN": "patch_len",
    "BATCH_SIZE": "batch_size",
    "LEARNING_RATE": "learning_rate",
}

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
    args.n_quantiles = len(args.quantiles)
    print(f"已从 /use 导入参数与权重: {weight_path}")
    return args


# =============================================================================
# 分位数损失函数
# =============================================================================
class LegacyQuantileLoss(nn.Module):
    """
    分位数损失函数 (Pinball Loss / Quantile Loss)

    对于分位数 q:
        L_q(y, ŷ) = max(q * (y - ŷ), (q - 1) * (y - ŷ))

    直觉理解：
    - q=0.5 时，等价于 MAE（对称惩罚）
    - q=0.1 时，过高预测惩罚更大 → 模型倾向偏低预测 → 形成 P10 下界
    - q=0.9 时，过低预测惩罚更大 → 模型倾向偏高预测 → 形成 P90 上界
    """
    def __init__(self, quantiles=None):
        super(LegacyQuantileLoss, self).__init__()
        self.quantiles = quantiles if quantiles is not None else QUANTILES

    def forward(self, predictions, targets):
        """
        参数:
            predictions: [B, pred_len, n_quantiles] - 模型输出的各分位数预测
            targets:     [B, pred_len, 1] 或 [B, pred_len] - 真实值
        """
        if targets.dim() == 2:
            targets = targets.unsqueeze(-1)  # [B, pred_len, 1]

        # targets 广播到 [B, pred_len, n_quantiles]
        errors = targets - predictions  # [B, pred_len, n_quantiles]

        # 构建分位数权重张量
        quantiles_tensor = torch.tensor(
            self.quantiles, dtype=predictions.dtype, device=predictions.device
        )  # [n_quantiles]

        # Pinball Loss: max(q * e, (q-1) * e) 对每个分位数分别计算
        losses = torch.max(
            quantiles_tensor * errors,
            (quantiles_tensor - 1.0) * errors
        )

        return losses.mean()


# =============================================================================
# TimeXer 分位数包装模型
# =============================================================================
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
    自定义训练流程，使用 QuantileLoss 训练 TimeXerQuantile 模型。

    Returns:
        训练好的模型
    """
    # 加载数据
    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    # 创建检查点目录
    setting = _get_setting(args)
    path = os.path.join(args.checkpoints, setting)
    os.makedirs(path, exist_ok=True)

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = SharedQuantileLoss(args.quantiles)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    print(f"\n{'='*60}")
    print(f"开始训练 TimeXer-Quantile 模型")
    print(f"分位数: {QUANTILES}")
    print(f"{'='*60}")

    for epoch in range(args.train_epochs):
        model.train()
        train_loss = []
        epoch_time = time.time()

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            optimizer.zero_grad()

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # 构造解码器输入
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            # 前向传播: 输出 [B, pred_len, n_quantiles]
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            # 取目标维度
            f_dim = -1 if args.features == 'MS' else 0
            batch_y_target = batch_y[:, -args.pred_len:, f_dim:]  # [B, pred_len, 1]

            # 计算分位数损失
            loss = criterion(outputs, batch_y_target)
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")

        # 验证
        vali_loss = validate_quantile(model, vali_loader, criterion, args, device)
        test_loss = validate_quantile(model, test_loader, criterion, args, device)

        train_loss_avg = np.average(train_loss)
        print(f"Epoch: {epoch+1} cost time: {time.time()-epoch_time:.1f}s | "
              f"Train: {train_loss_avg:.7f} Vali: {vali_loss:.7f} Test: {test_loss:.7f}")

        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        adjust_learning_rate(optimizer, epoch + 1, args)

    # 加载最佳模型
    best_model_path = os.path.join(path, 'checkpoint.pth')
    model.load_state_dict(torch.load(best_model_path))
    print(f"已加载最佳模型: {best_model_path}")

    return model


def validate_quantile(model, data_loader, criterion, args, device):
    """验证函数"""
    model.eval()
    total_loss = []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            batch_y_target = batch_y[:, -args.pred_len:, f_dim:].to(device)

            loss = criterion(outputs, batch_y_target)
            total_loss.append(loss.item())

    model.train()
    return np.average(total_loss)


# =============================================================================
# 自定义测试流程
# =============================================================================
def test_quantile_model(model, args, device):
    """
    测试 TimeXerQuantile 模型，保存 P50 预测和全分位数输出。

    保存文件:
        - pred.npy / true.npy: P50 点预测和真实值 [N, pred_len, 1]
        - pred_inv.npy / true_inv.npy: 反标准化后的版本
        - quantile_preds.npy: 全分位数预测 [N, pred_len, n_quantiles]
        - quantile_preds_inv.npy: 反标准化后的全分位数
    """
    test_data, test_loader = data_provider(args, 'test')

    setting = _get_setting(args)
    folder_path = os.path.join('./results/', setting)
    os.makedirs(folder_path, exist_ok=True)

    preds_p50 = []
    trues = []
    quantile_preds_all = []

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            # [B, pred_len, n_quantiles]
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            batch_y_target = batch_y[:, -args.pred_len:, f_dim:]  # [B, pred_len, 1]

            # 提取 P50 作为点预测
            p50_pred = outputs[:, :, P50_IDX:P50_IDX+1]  # [B, pred_len, 1]

            quantile_np = outputs.detach().cpu().numpy()        # [B, pred_len, 7]
            p50_np = p50_pred.detach().cpu().numpy()             # [B, pred_len, 1]
            true_np = batch_y_target.detach().cpu().numpy()      # [B, pred_len, 1]

            preds_p50.append(p50_np)
            trues.append(true_np)
            quantile_preds_all.append(quantile_np)

    preds_p50 = np.concatenate(preds_p50, axis=0)           # [N, pred_len, 1]
    trues = np.concatenate(trues, axis=0)                     # [N, pred_len, 1]
    quantile_preds_all = np.concatenate(quantile_preds_all, axis=0)  # [N, pred_len, 7]

    print(f'Test shape: preds={preds_p50.shape}, trues={trues.shape}, quantiles={quantile_preds_all.shape}')

    # 保存标准化版本
    np.save(os.path.join(folder_path, 'pred.npy'), preds_p50)
    np.save(os.path.join(folder_path, 'true.npy'), trues)
    np.save(os.path.join(folder_path, 'quantile_preds.npy'), quantile_preds_all)

    # 反标准化
    if test_data.scale:
        shape = trues.shape
        # P50
        preds_inv = test_data.inverse_transform(preds_p50.reshape(shape[0]*shape[1], -1)).reshape(shape)
        trues_inv = test_data.inverse_transform(trues.reshape(shape[0]*shape[1], -1)).reshape(shape)

        # 各分位数逐个反标准化
        q_shape = quantile_preds_all.shape  # [N, pred_len, 7]
        quantile_inv = np.zeros_like(quantile_preds_all)
        for qi in range(N_QUANTILES):
            q_slice = quantile_preds_all[:, :, qi:qi+1]  # [N, pred_len, 1]
            q_inv = test_data.inverse_transform(q_slice.reshape(q_shape[0]*q_shape[1], -1)).reshape(q_shape[0], q_shape[1], 1)
            quantile_inv[:, :, qi] = q_inv[:, :, 0]

        np.save(os.path.join(folder_path, 'pred_inv.npy'), preds_inv)
        np.save(os.path.join(folder_path, 'true_inv.npy'), trues_inv)
        np.save(os.path.join(folder_path, 'quantile_preds_inv.npy'), quantile_inv)

    # 计算评估指标 (使用 P50)
    mae, mse, rmse, mape, mspe = metric(preds_p50, trues)
    print(f'P50 Test Metrics: MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}')

    return folder_path


# =============================================================================
# 可视化函数
# =============================================================================
def plot_pred_vs_true(results_dir, feat_idx=0, out_name="pred_vs_true.png", use_inverse=False):
    """
    绘制预测值与真实值对比图，包含 P10-P90 置信区间带
    """
    try:
        plt.switch_backend("TkAgg")
    except Exception:
        pass

    # 确定数据文件路径
    if use_inverse:
        pred_path = os.path.join(results_dir, "pred_inv.npy")
        true_path = os.path.join(results_dir, "true_inv.npy")
        quantile_path = os.path.join(results_dir, "quantile_preds_inv.npy")
        if not os.path.exists(pred_path):
            pred_path = os.path.join(results_dir, "pred.npy")
            true_path = os.path.join(results_dir, "true.npy")
            quantile_path = os.path.join(results_dir, "quantile_preds.npy")
    else:
        pred_path = os.path.join(results_dir, "pred.npy")
        true_path = os.path.join(results_dir, "true.npy")
        quantile_path = os.path.join(results_dir, "quantile_preds.npy")

    if not os.path.exists(pred_path) or not os.path.exists(true_path):
        print("Prediction files not found, skip plotting.")
        return

    preds = np.load(pred_path)   # [N, pred_len, 1]
    trues = np.load(true_path)   # [N, pred_len, 1]

    # 加载分位数预测
    has_quantiles = os.path.exists(quantile_path)
    if has_quantiles:
        quantile_preds = np.load(quantile_path)  # [N, pred_len, 7]

    # ==================== 数据还原 ====================
    if preds.ndim == 3:
        pred_seq = restore_sliding_window_3d(preds)
        true_seq = restore_sliding_window_3d(trues)
        if pred_seq.ndim == 2:
            feat_idx = min(feat_idx, pred_seq.shape[1] - 1)
            pred_series = pred_seq[:, feat_idx]
            true_series = true_seq[:, feat_idx]
        else:
            pred_series = pred_seq.reshape(-1)
            true_series = true_seq.reshape(-1)
    elif preds.ndim == 2:
        pred_series = restore_sliding_window_2d(preds)
        true_series = restore_sliding_window_2d(trues)
    else:
        pred_series = preds.reshape(-1)
        true_series = trues.reshape(-1)

    # 还原分位数序列
    if has_quantiles:
        q_p10_raw = quantile_preds[:, :, P10_IDX:P10_IDX+1]
        q_p90_raw = quantile_preds[:, :, P90_IDX:P90_IDX+1]

        p10_seq = restore_sliding_window_3d(q_p10_raw)
        p90_seq = restore_sliding_window_3d(q_p90_raw)

        if p10_seq.ndim == 2:
            p10_series = p10_seq[:, min(feat_idx, p10_seq.shape[1]-1)]
            p90_series = p90_seq[:, min(feat_idx, p90_seq.shape[1]-1)]
        else:
            p10_series = p10_seq.reshape(-1)
            p90_series = p90_seq.reshape(-1)

    # ==================== 计算评估指标 ====================
    eval_df = cal_eval(true_series, pred_series)
    print("[Plot Eval] metrics:")
    print(eval_df)

    # ==================== 绘图 ====================
    os.makedirs(results_dir, exist_ok=True)
    mape_val = eval_df.iloc[0]["MAPE"]

    fig, ax = plt.subplots(1, 1, figsize=(15, 5), facecolor="white")
    ax.plot(true_series, label="GroundTruth", alpha=0.8, color="tab:blue")
    ax.plot(pred_series, label="Prediction (P50)", alpha=0.7, color="tab:orange")

    # 绘制 P10-P90 置信区间
    if has_quantiles:
        x_range = range(len(p10_series))
        ax.fill_between(
            x_range, p10_series, p90_series,
            alpha=0.2, color="tab:orange",
            label="P10-P90 Confidence Interval"
        )

    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    if np.isfinite(mape_val):
        ax.set_title(f"TimeXer Quantile Prediction - MAPE: {100*mape_val:.2f}%")
    else:
        ax.set_title("TimeXer Quantile Prediction - MAPE: NaN")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Load (MW)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, out_name), dpi=600, bbox_inches="tight")
    plt.show()


def predict_future_load_from_csv(
    model,
    args,
    device,
    results_dir: str,
    future_path: str = FUTURE_PATH,
    steps: int = PRED_LEN,
    use_inverse: bool = True,
):
    """
    基于 future.csv 使用直推预测未来电力负荷（包含 P10-P90 置信区间）

    与 DLinear/PatchTST 不同的是，TimeXer 使用 time_features 生成时间标记。
    """
    print("\n" + "=" * 60)
    print(f"Future Forecast: from {future_path}")
    print("=" * 60)

    abs_future_path = os.path.abspath(future_path)
    if not os.path.exists(abs_future_path):
        print(f"Future file not found, skip: {abs_future_path}")
        return

    history_path = os.path.join(args.root_path, args.data_path)
    if not os.path.exists(history_path):
        print(f"History file not found, skip: {history_path}")
        return

    try:
        history_df = _load_ordered_dataframe(history_path, args.target)
    except Exception as e:
        print(f"Load history data failed: {e}")
        return

    # 加载未来时间戳
    future_df = pd.read_csv(abs_future_path)
    if "date" not in future_df.columns:
        print(f"Future file missing date column: {abs_future_path}")
        return
    future_df["date"] = pd.to_datetime(future_df["date"])
    future_df = future_df.sort_values("date").reset_index(drop=True)

    if len(history_df) < args.seq_len:
        print(f"History length ({len(history_df)}) < seq_len ({args.seq_len}), skip.")
        return

    predict_steps = min(int(steps), len(future_df), args.pred_len)
    if predict_steps <= 0 or predict_steps < args.pred_len:
        print(f"Future rows ({predict_steps}) < pred_len ({args.pred_len}), skip.")
        return

    # 获取 Scaler
    train_data, _ = data_provider(args, 'train')
    scaler = getattr(train_data, "scaler", None)
    has_scaler = scaler is not None and hasattr(scaler, "mean_")

    feature_cols = [args.target]
    target_idx = 0
    history_values = history_df[feature_cols].values.astype(np.float32)

    if has_scaler:
        history_values = scaler.transform(history_values)

    # 构建模型输入（TimeXer 需要 time_features 时间标记）
    enc_window = history_values[-args.seq_len:].copy()
    hist_dates = pd.to_datetime(history_df["date"].iloc[-args.seq_len:].values)
    x_mark_np = time_features(hist_dates, freq=args.freq).transpose(1, 0).astype(np.float32)

    model.eval()
    with torch.no_grad():
        batch_x = torch.as_tensor(enc_window, dtype=torch.float32, device=device).unsqueeze(0)
        batch_x_mark = torch.as_tensor(x_mark_np, dtype=torch.float32, device=device).unsqueeze(0)

        dec_len = args.label_len + args.pred_len
        dec_inp = torch.zeros((1, dec_len, batch_x.shape[-1]), dtype=torch.float32, device=device)
        batch_y_mark = torch.zeros((1, dec_len, batch_x_mark.shape[-1]), dtype=torch.float32, device=device)

        # 输出: [1, pred_len, n_quantiles]
        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        quantile_scaled = outputs[0, :args.pred_len, :].detach().cpu().numpy()  # [pred_len, 7]
        p50_scaled = quantile_scaled[:, P50_IDX]
        p10_scaled = quantile_scaled[:, P10_IDX]
        p90_scaled = quantile_scaled[:, P90_IDX]

    # 反归一化
    if has_scaler and use_inverse:
        def inv(arr):
            return arr * scaler.scale_[target_idx] + scaler.mean_[target_idx]
        preds_p50 = inv(p50_scaled)
        preds_p10 = inv(p10_scaled)
        preds_p90 = inv(p90_scaled)
        history_target = history_df[args.target].values
    else:
        preds_p50 = p50_scaled
        preds_p10 = p10_scaled
        preds_p90 = p90_scaled
        history_target = history_values[:, target_idx]

    future_dates = future_df["date"].iloc[:args.pred_len].reset_index(drop=True)
    preds_p50 = preds_p50[:len(future_dates)]
    preds_p10 = preds_p10[:len(future_dates)]
    preds_p90 = preds_p90[:len(future_dates)]
    predict_steps = len(preds_p50)

    # 保存结果（包含 P10/P90）
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "future_load_prediction.csv")
    pd.DataFrame({
        "date": future_dates,
        f"{args.target}_pred_P10": preds_p10,
        f"{args.target}_pred_P50": preds_p50,
        f"{args.target}_pred_P90": preds_p90,
    }).to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\nFuture {predict_steps}-step {args.target} predictions:")
    print(f"{'Time':<25} {'P10':<12} {'P50':<12} {'P90':<12}")
    print("-" * 65)
    for i in range(predict_steps):
        print(f"  {future_dates.iloc[i]}: {preds_p10[i]:<12.4f} {preds_p50[i]:<12.4f} {preds_p90[i]:<12.4f}")

    # ==================== 绘图（包含置信区间） ====================
    n_history = min(672, len(history_target))
    history_tail = history_target[-n_history:]
    future_x = range(n_history, n_history + predict_steps)

    plt.figure(figsize=(15, 6), facecolor="white")
    plt.plot(range(n_history), history_tail, label="Historical Load", color="blue", alpha=0.8)
    plt.plot(
        future_x, preds_p50,
        label="TimeXer P50 Prediction",
        color="orange", linewidth=2, marker="o", markersize=2,
    )
    # P10-P90 置信区间
    plt.fill_between(
        future_x, preds_p10, preds_p90,
        alpha=0.25, color="orange",
        label="P10-P90 Confidence Interval"
    )
    plt.axvline(x=n_history - 0.5, color="gray", linestyle="--", alpha=0.6, label="Prediction Start")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.title(f"TimeXer Future {predict_steps}-Step Load Prediction (Quantile)")
    plt.xlabel("Time Step (15min)")
    plt.ylabel("Load (MW)")
    plt.tight_layout()

    out_fig = os.path.join(results_dir, "future_load_prediction.png")
    plt.savefig(out_fig, dpi=600, bbox_inches="tight")
    plt.show()

    print(f"Saved future prediction csv: {out_csv}")
    print(f"Saved future prediction figure: {out_fig}")


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
        f"fcov{int(args.use_future_covariates)}_fcd{args.future_cov_dim}_"
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

        use_future_covariates=USE_FUTURE_COVARIATES,
        future_cov_dim=FUTURE_COV_DIM,
        future_cov_dropout=FUTURE_COV_DROPOUT,
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

    if not TRAIN_MODE and LOAD_FROM_USE:
        args = _apply_use_artifacts(args)

    # ==================== 配置计算设备 ====================
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU: cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # ==================== 构建模型 ====================
    model = TimeXerQuantile(args, quantiles=QUANTILES).float().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"TimeXerQuantile 模型参数量: {total_params:,}")

    setting = _get_setting(args)

    # ==================== 训练或加载模型 ====================
    if TRAIN_MODE:
        print(f"\n>>> 开始训练: {setting}")
        model = train_quantile_model(model, args, device)

        print(f"\n>>> 开始测试: {setting}")
        results_dir = test_quantile_model(model, args, device)

        plot_pred_vs_true(results_dir, use_inverse=INVERSE_EVAL)
        predict_future_load_from_csv(
            model=model, args=args, device=device,
            results_dir=results_dir, future_path=FUTURE_PATH,
            steps=PRED_LEN, use_inverse=INVERSE_EVAL,
        )
    else:
        # 加载已训练模型
        ckpt_path = getattr(args, "load_weight_path", None)
        if ckpt_path is None:
            ckpt_path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
        
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path))
            print(f"成功加载模型: {ckpt_path}")
        else:
            raise FileNotFoundError(
                f"未找到模型文件 {ckpt_path}，请先设置 TRAIN_MODE = True 训练模型。"
            )

        print(f"\n>>> 仅测试: {setting}")
        results_dir = test_quantile_model(model, args, device)

        plot_pred_vs_true(results_dir, use_inverse=INVERSE_EVAL)
        predict_future_load_from_csv(
            model=model, args=args, device=device,
            results_dir=results_dir, future_path=FUTURE_PATH,
            steps=PRED_LEN, use_inverse=INVERSE_EVAL,
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
        f"fcov{int(args.use_future_covariates)}_fcd{args.future_cov_dim}_"
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
