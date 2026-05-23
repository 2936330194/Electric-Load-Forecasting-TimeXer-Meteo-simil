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

if __name__ == "__main__":
    main()
