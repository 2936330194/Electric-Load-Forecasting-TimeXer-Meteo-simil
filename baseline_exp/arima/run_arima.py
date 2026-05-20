import os
import sys
import json
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

# Set project root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from utils.forecast_visualization import plot_pred_vs_true
from utils.metrics import cal_eval

def main():
    print("Loading data...")
    data_dir = os.path.join(project_root, "data")
    df = pd.read_csv(os.path.join(data_dir, "湖南省电力负荷2024.csv"))
    load = df['load'].values
    total_len = len(load)
    
    # 4:1:1 Split
    num_train = int(total_len * 2/3) # 23424
    num_vali = int(total_len * 1/6) # 5856
    num_test = total_len - num_train - num_vali # 5856
    
    # Standardize
    scaler = StandardScaler()
    scaler.fit(load[:num_train].reshape(-1, 1))
    scaled_load = scaler.transform(load.reshape(-1, 1)).flatten()
    
    # Fit ARIMA(2, 1, 2) on full training set
    print("Fitting ARIMA(2,1,2) on full train set...")
    t0 = time.time()
    train_data = scaled_load[:num_train]
    model = ARIMA(train_data, order=(2, 1, 2))
    res = model.fit()
    print(f"Fit time: {time.time() - t0:.2f}s")
    
    # Test sliding windows setup
    seq_len = 672
    pred_len = 96
    border1_test = num_train + num_vali - seq_len # 28608
    border2_test = total_len # 35136
    
    n_windows = border2_test - border1_test - seq_len - pred_len + 1 # 5761
    
    print(f"Running rolling forecast on {n_windows} windows...")
    t0 = time.time()
    preds_scaled = []
    trues_scaled = []
    
    # Loop over all test windows
    for i in range(n_windows):
        s_begin = border1_test + i
        s_end = s_begin + seq_len
        r_begin = s_end
        r_end = r_begin + pred_len
        
        history = scaled_load[s_begin:s_end]
        target = scaled_load[r_begin:r_end]
        
        # Apply fitted parameters on history window and forecast
        new_res = res.apply(history)
        pred = new_res.forecast(steps=pred_len)
        
        preds_scaled.append(pred)
        trues_scaled.append(target)
        
        if (i + 1) % 1000 == 0 or i == 0 or i == n_windows - 1:
            elapsed = time.time() - t0
            print(f"Window {i+1}/{n_windows} | Elapsed: {elapsed:.1f}s | Est. remaining: {elapsed/(i+1)*(n_windows-i-1):.1f}s")
    
    preds_scaled = np.array(preds_scaled) # [n_windows, pred_len]
    trues_scaled = np.array(trues_scaled) # [n_windows, pred_len]
    
    # Inverse scale back to Physical scale
    print("Inverse scaling results...")
    preds_inv = scaler.inverse_transform(preds_scaled)
    trues_inv = scaler.inverse_transform(trues_scaled)
    
    # Metrics calculation
    # 1. MAE
    mae = np.mean(np.abs(preds_inv - trues_inv))
    # 2. MSE
    mse = np.mean((preds_inv - trues_inv) ** 2)
    # 3. RMSE
    rmse = np.sqrt(mse)
    # 4. MAPE
    from utils.metrics import MAPE
    mape = MAPE(preds_inv, trues_inv)
    # 5. R2
    target_mean = np.mean(trues_inv)
    ss_tot = np.sum((trues_inv - target_mean) ** 2)
    ss_res = np.sum((trues_inv - preds_inv) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    # Save numpy arrays
    results_dir = os.path.dirname(__file__)
    np.save(os.path.join(results_dir, 'preds_inv.npy'), preds_inv)
    np.save(os.path.join(results_dir, 'trues_inv.npy'), trues_inv)
    np.save(os.path.join(results_dir, 'pred_inv.npy'), preds_inv)
    np.save(os.path.join(results_dir, 'true_inv.npy'), trues_inv)

    origin_eval_df = cal_eval(trues_inv, preds_inv)
    print("[origin Eval] metrics:")
    print(origin_eval_df)

    plot_pred_vs_true(
        results_dir,
        use_inverse=True,
        title_prefix="ARIMA Prediction",
    )
    
    # 1D contiguous timeline metrics
    def restore_sliding_window_2d(data_2d):
        if len(data_2d) == 0: return np.array([])
        restored = list(data_2d[0, :])
        for i in range(1, len(data_2d)):
            restored.append(data_2d[i, -1])
        return np.asarray(restored)

    pred_1d = restore_sliding_window_2d(preds_inv)
    true_1d = restore_sliding_window_2d(trues_inv)

    mae_1d = float(np.mean(np.abs(pred_1d - true_1d)))
    mse_1d = float(np.mean((pred_1d - true_1d) ** 2))
    rmse_1d = float(np.sqrt(mse_1d))
    mape_1d = float(np.mean(np.abs((pred_1d - true_1d) / np.maximum(true_1d, 1e-5))))
    
    target_mean_1d = np.mean(true_1d)
    ss_tot_1d = np.sum((true_1d - target_mean_1d) ** 2)
    ss_res_1d = np.sum((true_1d - pred_1d) ** 2)
    r2_1d = float(1 - ss_res_1d / ss_tot_1d)
    
    # Generate future forecasting predictions
    print("\nGenerating future forecast predictions...")
    future_csv_path = os.path.join(data_dir, "湖南省电力负荷2024_future.csv")
    if os.path.exists(future_csv_path):
        future_df = pd.read_csv(future_csv_path)
        future_df["date"] = pd.to_datetime(future_df["date"])
        
        hist_load_tail = scaled_load[-seq_len:]
        new_res = res.apply(hist_load_tail)
        pred_scaled = new_res.forecast(steps=pred_len).reshape(-1, 1)
        pred_phys = scaler.inverse_transform(pred_scaled).flatten()
        
        output_df = pd.DataFrame({
            "date": future_df["date"].iloc[:pred_len],
            "load_pred_P50": pred_phys
        })
        output_csv_path = os.path.join(os.path.dirname(__file__), "future_load_prediction.csv")
        output_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
        print(f"Saved future predictions to {output_csv_path}")

if __name__ == "__main__":
    main()
