import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

# Set project root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

# Set seed
fix_seed = 2026
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# Import project components
from data_provider.data_factory import data_provider
from utils.forecast_visualization import plot_pred_vs_true
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, cal_eval

class LSTMBaseline(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_len=96):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_len)
        
    def forward(self, x):
        # x: [B, seq_len, input_size]
        out, _ = self.lstm(x)
        # We take the output of the last step and project to target steps
        out = self.fc(out[:, -1, :]) # [B, output_len]
        return out.unsqueeze(-1) # [B, output_len, 1]

def main():
    import argparse
    parser = argparse.ArgumentParser(description="LSTM Baseline for Electricity Load Forecasting")
    parser.add_argument("--is_training", type=int, default=1, help="status")
    parser.add_argument("--train_epochs", type=int, default=50, help="train epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--patience", type=int, default=5, help="early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="optimizer learning rate")
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    
    args = parser.parse_args()
    
    # Overwrite default parameters to match baseline requirements
    args.task_name = "long_term_forecast"
    args.model_id = "HunanLoad_2024_672_96"
    args.model = "LSTM"
    args.data = "custom"
    args.root_path = os.path.join(project_root, "data")
    args.data_path = "湖南省电力负荷2024.csv"
    args.features = "S"
    args.target = "load"
    args.freq = "min"
    args.embed = "timeF"
    args.seq_len = 672
    args.label_len = 0
    args.pred_len = 96
    args.enc_in = 1
    args.c_out = 1
    args.num_workers = 0
    args.checkpoints = os.path.dirname(__file__)
    args.loss = "MSE"
    args.lradj = "cosine"
    args.inverse_eval = True
    args.des = "Exp"
    args.itr = 1
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {device}")
    
    # Load data loaders
    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')
    
    # Check if checkpoint already exists to skip training
    best_weight_path = os.path.join(args.checkpoints, 'checkpoint.pth')
    if os.path.exists(best_weight_path):
        print("Checkpoint found. Skipping training phase.")
        args.is_training = 0

    checkpoint_path = os.path.join(args.checkpoints, 'lstm_checkpoint.pth')
    model = LSTMBaseline(input_size=1, hidden_size=128, num_layers=2, output_len=args.pred_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    if args.is_training:
        print("Training LSTM baseline...")
        t0 = time.time()
        for epoch in range(args.train_epochs):
            model.train()
            train_loss = []
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                optimizer.zero_grad()
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                
                # LSTM is univariate S mode, so we use the load feature channel
                # Shape of batch_x: [B, seq_len, 1]
                outputs = model(batch_x)
                batch_y_target = batch_y[:, -args.pred_len:, -1:] # [B, pred_len, 1]
                
                loss = criterion(outputs, batch_y_target)
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()
                
            # Validate
            model.eval()
            vali_loss = []
            with torch.no_grad():
                for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                    batch_x = batch_x.float().to(device)
                    batch_y = batch_y.float().to(device)
                    outputs = model(batch_x)
                    batch_y_target = batch_y[:, -args.pred_len:, -1:]
                    loss = criterion(outputs, batch_y_target)
                    vali_loss.append(loss.item())
            
            train_loss_avg = np.mean(train_loss)
            vali_loss_avg = np.mean(vali_loss)
            print(f"Epoch {epoch+1} | Train Loss: {train_loss_avg:.6f} | Vali Loss: {vali_loss_avg:.6f}")
            
            early_stopping(vali_loss_avg, model, args.checkpoints)
            if early_stopping.early_stop:
                print("Early stopping")
                break
                
            adjust_learning_rate(optimizer, epoch + 1, args)
            
        print(f"Training completed in {time.time() - t0:.1f}s")
    
    # Load best weights
    loaded_weight = False
    if os.path.exists(best_weight_path):
        model.load_state_dict(torch.load(best_weight_path, map_location=device))
        print("Loaded best model.")
        loaded_weight = True
    else:
        # Fallback to checkpoint_path
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print("Loaded checkpoint model.")
            loaded_weight = True

    if not loaded_weight and not args.is_training:
        raise FileNotFoundError(
            f"Model checkpoint not found: {best_weight_path}. "
            "Set --is_training/default is_training to 1 to train LSTM first."
        )
            
    # Evaluation on Test set
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x = batch_x.float().to(device)
            outputs = model(batch_x)
            preds.append(outputs.detach().cpu().numpy())
            trues.append(batch_y[:, -args.pred_len:, -1:].detach().cpu().numpy())
            
    preds = np.concatenate(preds, axis=0) # [N, pred_len, 1]
    trues = np.concatenate(trues, axis=0) # [N, pred_len, 1]
    
    # Inverse scale to physical load scale
    if args.inverse_eval and test_data.scale:
        shape = trues.shape
        preds_inv = test_data.inverse_transform(preds.reshape(shape[0]*shape[1], -1)).reshape(shape)
        trues_inv = test_data.inverse_transform(trues.reshape(shape[0]*shape[1], -1)).reshape(shape)
    else:
        preds_inv = preds
        trues_inv = trues
        
    # Save numpy arrays
    np.save(os.path.join(args.checkpoints, 'preds_inv.npy'), preds_inv)
    np.save(os.path.join(args.checkpoints, 'trues_inv.npy'), trues_inv)
    np.save(os.path.join(args.checkpoints, 'pred_inv.npy'), preds_inv)
    np.save(os.path.join(args.checkpoints, 'true_inv.npy'), trues_inv)

    plot_pred_vs_true(
        args.checkpoints,
        use_inverse=args.inverse_eval,
        title_prefix="LSTM Prediction",
    )
        
    # Standard 2D window-average metrics
    mae, mse, rmse, mape, mspe = metric(preds_inv, trues_inv)
    target_mean = np.mean(trues_inv)
    ss_tot = np.sum((trues_inv - target_mean) ** 2)
    ss_res = np.sum((trues_inv - preds_inv) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    # 1D contiguous timeline metrics
    def restore_sliding_window_3d(data_3d):
        if len(data_3d) == 0: return np.array([])
        restored = list(data_3d[0, :, :])
        for i in range(1, len(data_3d)):
            restored.append(data_3d[i, -1, :])
        return np.asarray(restored)

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
    
    metrics = {
        "model": "LSTM",
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
        "shape": list(trues.shape)
    }
    
    print("\nLSTM Baseline Test Metrics:")
    print(json.dumps(metrics, indent=2))
    
    metrics_path = os.path.join(args.checkpoints, 'lstm_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    # Generate future forecasting predictions
    print("\nGenerating future forecast predictions...")
    future_csv_path = os.path.join(args.root_path, "湖南省电力负荷2024_future.csv")
    if os.path.exists(future_csv_path):
        future_df = pd.read_csv(future_csv_path)
        future_df["date"] = pd.to_datetime(future_df["date"])
        
        # Load historical load tail to build input window
        history_csv_path = os.path.join(args.root_path, args.data_path)
        history_df = pd.read_csv(history_csv_path)
        hist_load = history_df[args.target].iloc[-args.seq_len:].values.reshape(-1, 1)
        
        # Standardize using training scaler
        scaled_hist_load = train_data.scaler.transform(hist_load)
        
        # Make future prediction
        batch_x = torch.as_tensor(scaled_hist_load, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            outputs = model(batch_x) # [1, pred_len, 1]
            pred_scaled = outputs[0, :, 0].cpu().numpy().reshape(-1, 1)
            
        # Inverse transform prediction
        pred_phys = train_data.scaler.inverse_transform(pred_scaled).flatten()
        
        # Save to CSV
        output_df = pd.DataFrame({
            "date": future_df["date"].iloc[:args.pred_len],
            "load_pred_P50": pred_phys
        })
        output_csv_path = os.path.join(args.checkpoints, "future_load_prediction.csv")
        output_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
        print(f"Saved future predictions to {output_csv_path}")

if __name__ == "__main__":
    main()
