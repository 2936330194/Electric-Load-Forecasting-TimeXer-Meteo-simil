import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import os
import sys
import time
import json
import matplotlib.pyplot as plt

# Set project root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from models.PatchTST import Model as PatchTST
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, cal_eval, append_probabilistic_eval

QUANTILES = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
N_QUANTILES = len(QUANTILES)
P50_IDX = QUANTILES.index(0.5)
P10_IDX = QUANTILES.index(0.1)
P90_IDX = QUANTILES.index(0.9)

class QuantileLoss(nn.Module):
    def __init__(self, quantiles=None):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles if quantiles is not None else QUANTILES

    def forward(self, predictions, targets):
        if targets.dim() == 2:
            targets = targets.unsqueeze(-1)
        errors = targets - predictions
        quantiles_tensor = torch.tensor(
            self.quantiles, dtype=predictions.dtype, device=predictions.device
        )
        losses = torch.max(
            quantiles_tensor * errors,
            (quantiles_tensor - 1.0) * errors
        )
        return losses.mean()

class PatchTSTQuantile(nn.Module):
    def __init__(self, configs, quantiles=None):
        super(PatchTSTQuantile, self).__init__()
        self.quantiles = quantiles if quantiles is not None else QUANTILES
        self.n_quantiles = len(self.quantiles)
        self.patchtst = PatchTST(configs)
        self.quantile_head = nn.Linear(1, self.n_quantiles)
        with torch.no_grad():
            self.quantile_head.weight.fill_(1.0)
            self.quantile_head.bias.copy_(
                torch.tensor([q - 0.5 for q in self.quantiles]) * 0.1
            )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        point_pred = self.patchtst(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        point_pred = point_pred[:, -self.patchtst.pred_len:, :]
        quantile_pred = self.quantile_head(point_pred)
        return quantile_pred

def restore_sliding_window_2d(data_2d: np.ndarray) -> np.ndarray:
    if len(data_2d) == 0:
        return np.array([])
    restored = list(data_2d[0, :])
    for i in range(1, len(data_2d)):
        restored.append(data_2d[i, -1])
    return np.asarray(restored)

def restore_sliding_window_3d(data_3d: np.ndarray) -> np.ndarray:
    if len(data_3d) == 0:
        return np.array([])
    restored = list(data_3d[0, :, :])
    for i in range(1, len(data_3d)):
        restored.append(data_3d[i, -1, :])
    return np.asarray(restored)

def _load_ordered_dataframe(csv_path: str, target: str) -> pd.DataFrame:
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

def train_quantile_model(model, args, device):
    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    path = args.checkpoints
    os.makedirs(path, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = QuantileLoss()
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    print(f"\n{'='*60}")
    print(f"开始训练 PatchTST-Quantile 模型")
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

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            batch_y_target = batch_y[:, -args.pred_len:, f_dim:]

            loss = criterion(outputs, batch_y_target)
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f"\titers: {i+1}, epoch: {epoch+1} | loss: {loss.item():.7f}")

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

    best_model_path = os.path.join(path, 'checkpoint.pth')
    model.load_state_dict(torch.load(best_model_path))
    print(f"已加载最佳模型: {best_model_path}")

    return model

def validate_quantile(model, data_loader, criterion, args, device):
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

def test_quantile_model(model, args, device):
    test_data, test_loader = data_provider(args, 'test')
    folder_path = args.checkpoints

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

            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if args.features == 'MS' else 0
            batch_y_target = batch_y[:, -args.pred_len:, f_dim:]

            p50_pred = outputs[:, :, P50_IDX:P50_IDX+1]

            quantile_np = outputs.detach().cpu().numpy()
            p50_np = p50_pred.detach().cpu().numpy()
            true_np = batch_y_target.detach().cpu().numpy()

            preds_p50.append(p50_np)
            trues.append(true_np)
            quantile_preds_all.append(quantile_np)

    preds_p50 = np.concatenate(preds_p50, axis=0)
    trues = np.concatenate(trues, axis=0)
    quantile_preds_all = np.concatenate(quantile_preds_all, axis=0)

    print(f'Test shape: preds={preds_p50.shape}, trues={trues.shape}, quantiles={quantile_preds_all.shape}')

    np.save(os.path.join(folder_path, 'pred.npy'), preds_p50)
    np.save(os.path.join(folder_path, 'true.npy'), trues)
    np.save(os.path.join(folder_path, 'quantile_preds.npy'), quantile_preds_all)

    if test_data.scale:
        shape = trues.shape
        preds_inv = test_data.inverse_transform(preds_p50.reshape(shape[0]*shape[1], -1)).reshape(shape)
        trues_inv = test_data.inverse_transform(trues.reshape(shape[0]*shape[1], -1)).reshape(shape)

        q_shape = quantile_preds_all.shape
        quantile_inv = np.zeros_like(quantile_preds_all)
        for qi in range(N_QUANTILES):
            q_slice = quantile_preds_all[:, :, qi:qi+1]
            q_inv = test_data.inverse_transform(q_slice.reshape(q_shape[0]*q_shape[1], -1)).reshape(q_shape[0], q_shape[1], 1)
            quantile_inv[:, :, qi] = q_inv[:, :, 0]

        np.save(os.path.join(folder_path, 'pred_inv.npy'), preds_inv)
        np.save(os.path.join(folder_path, 'true_inv.npy'), trues_inv)
        np.save(os.path.join(folder_path, 'quantile_preds_inv.npy'), quantile_inv)
    else:
        preds_inv = preds_p50
        trues_inv = trues

    origin_quantiles = quantile_inv if 'quantile_inv' in locals() else quantile_preds_all
    origin_eval_df = cal_eval(trues_inv, preds_inv)
    origin_eval_df = append_probabilistic_eval(origin_eval_df, trues_inv, origin_quantiles, QUANTILES)
    print("[origin Eval] metrics:")
    print(origin_eval_df)

    return folder_path, preds_inv, trues_inv, origin_eval_df

def plot_pred_vs_true(results_dir, use_inverse=False):
    if use_inverse:
        pred_path = os.path.join(results_dir, "pred_inv.npy")
        true_path = os.path.join(results_dir, "true_inv.npy")
        quantile_path = os.path.join(results_dir, "quantile_preds_inv.npy")
    else:
        pred_path = os.path.join(results_dir, "pred.npy")
        true_path = os.path.join(results_dir, "true.npy")
        quantile_path = os.path.join(results_dir, "quantile_preds.npy")

    if not os.path.exists(pred_path) or not os.path.exists(true_path):
        print("Prediction files not found, skip plotting.")
        return

    preds = np.load(pred_path)
    trues = np.load(true_path)

    has_quantiles = os.path.exists(quantile_path)
    if has_quantiles:
        quantile_preds = np.load(quantile_path)

    pred_series = restore_sliding_window_3d(preds).squeeze()
    true_series = restore_sliding_window_3d(trues).squeeze()

    if has_quantiles:
        q_p10_raw = quantile_preds[:, :, P10_IDX:P10_IDX+1]
        q_p90_raw = quantile_preds[:, :, P90_IDX:P90_IDX+1]
        p10_series = restore_sliding_window_3d(q_p10_raw).squeeze()
        p90_series = restore_sliding_window_3d(q_p90_raw).squeeze()

    eval_df = cal_eval(true_series, pred_series)
    print("[Plot Eval] metrics:")
    print(eval_df)

    mape_val = eval_df.iloc[0]["MAPE"]

    plt.figure(figsize=(15, 5), facecolor="white")
    plt.plot(true_series, label="GroundTruth", alpha=0.8, color="tab:blue")
    plt.plot(pred_series, label="Prediction (P50)", alpha=0.7, color="tab:orange")

    if has_quantiles:
        x_range = range(len(p10_series))
        plt.fill_between(
            x_range, p10_series, p90_series,
            alpha=0.2, color="tab:orange",
            label="P10-P90 Confidence Interval"
        )

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    if np.isfinite(mape_val):
        plt.title(f"PatchTST Quantile Prediction - MAPE: {100*mape_val:.2f}%")
    else:
        plt.title("PatchTST Quantile Prediction - MAPE: NaN")
    plt.xlabel("Time Step")
    plt.ylabel("Load (MW)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "pred_vs_true.png"), dpi=600, bbox_inches="tight")
    plt.close()

def predict_future_load_from_csv(model, args, device, results_dir, future_path, steps=96, use_inverse=True):
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

    train_data, _ = data_provider(args, 'train')
    scaler = getattr(train_data, "scaler", None)
    has_scaler = scaler is not None and hasattr(scaler, "mean_")

    feature_cols = [args.target]
    target_idx = 0
    history_values = history_df[feature_cols].values.astype(np.float32)

    if has_scaler:
        history_values = scaler.transform(history_values)

    enc_window = history_values[-args.seq_len:].copy()

    model.eval()
    with torch.no_grad():
        batch_x = torch.as_tensor(enc_window, dtype=torch.float32, device=device).unsqueeze(0)
        x_mark = torch.zeros((1, args.seq_len, 1), dtype=torch.float32, device=device)
        dec_len = args.label_len + args.pred_len
        dec_inp = torch.zeros((1, dec_len, batch_x.shape[-1]), dtype=torch.float32, device=device)
        y_mark = torch.zeros((1, dec_len, 1), dtype=torch.float32, device=device)

        outputs = model(batch_x, x_mark, dec_inp, y_mark)

        quantile_scaled = outputs[0, :args.pred_len, :].detach().cpu().numpy()
        p50_scaled = quantile_scaled[:, P50_IDX]
        p10_scaled = quantile_scaled[:, P10_IDX]
        p90_scaled = quantile_scaled[:, P90_IDX]

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

    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "future_load_prediction.csv")
    pd.DataFrame({
        "date": future_dates,
        f"{args.target}_pred_P10": preds_p10,
        f"{args.target}_pred_P50": preds_p50,
        f"{args.target}_pred_P90": preds_p90,
    }).to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\nFuture {predict_steps}-step {args.target} predictions:")
    for i in range(predict_steps):
        print(f"  {future_dates.iloc[i]}: {preds_p10[i]:.4f} {preds_p50[i]:.4f} {preds_p90[i]:.4f}")

    n_history = min(672, len(history_target))
    history_tail = history_target[-n_history:]
    future_x = range(n_history, n_history + predict_steps)

    plt.figure(figsize=(15, 6), facecolor="white")
    plt.plot(range(n_history), history_tail, label="Historical Load", color="blue", alpha=0.8)
    plt.plot(
        future_x, preds_p50,
        label="PatchTST P50 Prediction",
        color="orange", linewidth=2, marker="o", markersize=2,
    )
    plt.fill_between(
        future_x, preds_p10, preds_p90,
        alpha=0.25, color="orange",
        label="P10-P90 Confidence Interval"
    )
    plt.axvline(x=n_history - 0.5, color="gray", linestyle="--", alpha=0.6, label="Prediction Start")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.title(f"PatchTST Future {predict_steps}-Step Load Prediction (Quantile)")
    plt.xlabel("Time Step (15min)")
    plt.ylabel("Load (MW)")
    plt.tight_layout()
    out_fig = os.path.join(results_dir, "future_load_prediction.png")
    plt.savefig(out_fig, dpi=600, bbox_inches="tight")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="PatchTST Baseline")
    parser.add_argument("--is_training", type=int, default=1, help="status")
    parser.add_argument("--train_epochs", type=int, default=50, help="train epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--patience", type=int, default=5, help="early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="optimizer learning rate")
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    args = parser.parse_args()

    # Fixed project settings
    args.task_name = "long_term_forecast"
    args.model_id = "HunanLoad_2024_672_96"
    args.model = "PatchTST"
    args.data = "custom"
    args.root_path = os.path.join(project_root, "data")
    args.data_path = "湖南省电力负荷2024.csv"
    args.features = "S"
    args.target = "load"
    args.freq = "t"
    args.embed = "timeF"
    args.seq_len = 672
    args.label_len = 0
    args.pred_len = 96
    args.enc_in = 1
    args.c_out = 1
    args.d_model = 512
    args.n_heads = 4
    args.e_layers = 3
    args.d_ff = 2048
    args.factor = 3
    args.dropout = 0.1
    args.activation = "gelu"
    args.patch_len = 96
    args.stride = 48
    args.use_future_covariates = False
    args.future_cov_dim = 0
    args.future_cov_dropout = 0.1
    args.num_workers = 0
    args.checkpoints = os.path.dirname(__file__)
    args.loss = "Quantile"
    args.lradj = "cosine"
    args.use_amp = True
    args.inverse_eval = True
    args.des = "Exp"
    args.itr = 1

    fix_seed = 2026
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {device}")

    # Check if checkpoint exists
    best_weight_path = os.path.join(args.checkpoints, 'checkpoint.pth')
    if os.path.exists(best_weight_path):
        print("Checkpoint found. Skipping training phase.")
        args.is_training = 0

    model = PatchTSTQuantile(args, quantiles=QUANTILES).float().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PatchTSTQuantile total params: {total_params:,}")
    print(f"PatchTSTQuantile trainable params: {trainable_params:,}")

    if args.is_training:
        model = train_quantile_model(model, args, device)
    else:
        if os.path.exists(best_weight_path):
            model.load_state_dict(torch.load(best_weight_path, map_location=device))
            print(f"Loaded model checkpoint: {best_weight_path}")
        else:
            raise FileNotFoundError(f"Model checkpoint not found at {best_weight_path}")

    results_dir, preds_inv, trues_inv, origin_eval_df = test_quantile_model(model, args, device)

    # Standard metrics
    mae, mse, rmse, mape, mspe = metric(preds_inv, trues_inv)
    target_mean = np.mean(trues_inv)
    ss_tot = np.sum((trues_inv - target_mean) ** 2)
    ss_res = np.sum((trues_inv - preds_inv) ** 2)
    r2 = 1 - ss_res / ss_tot

    # 1D contiguous timeline metrics
    pred_1d = restore_sliding_window_3d(preds_inv).squeeze()
    true_1d = restore_sliding_window_3d(trues_inv).squeeze()

    mae_1d = float(np.mean(np.abs(pred_1d - true_1d)))
    mse_1d = float(np.mean((pred_1d - true_1d) ** 2))
    rmse_1d = float(np.sqrt(mse_1d))
    mape_1d = float(np.mean(np.abs((pred_1d - true_1d) / np.maximum(true_1d, 1e-5))))
    
    target_mean_1d = np.mean(true_1d)
    ss_tot_1d = np.sum((true_1d - target_mean_1d) ** 2)
    ss_res_1d = np.sum((true_1d - pred_1d) ** 2)
    r2_1d = float(1 - ss_res_1d / ss_tot_1d)

    # Save metrics JSON
    metrics = {
        "model": "PatchTST",
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "mape": float(mape),
        "r2": float(r2),
        "mae_1d": mae_1d,
        "mse_1d": mse_1d,
        "rmse_1d": rmse_1d,
        "mape_1d": mape_1d,
        "r2_1d": r2_1d,
        "mean_pinball": float(origin_eval_df.loc["Eval", "Mean Pinball"]),
        "picp_p10_p90": float(origin_eval_df.loc["Eval", "PICP (P10-P90)"]),
        "pinaw_p10_p90": float(origin_eval_df.loc["Eval", "PINAW (P10-P90)"]),
        "shape": list(preds_inv.shape)
    }

    metrics_path = os.path.join(args.checkpoints, "patchtst_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

if __name__ == "__main__":
    main()
