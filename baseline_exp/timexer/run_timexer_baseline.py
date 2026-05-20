import os
import sys
import json
import subprocess
import shutil
import numpy as np

# Set project root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

# Import metrics computation helper
from utils.metrics import metric, cal_eval

def main():
    print("Running test1.py to obtain pure TimeXer baseline results...")
    
    # Run test1.py in a subprocess with the active conda environment python executable
    python_exe = sys.executable
    cmd = [python_exe, "test1.py"]
    
    print(f"Command: {' '.join(cmd)}")
    t0 = np.datetime64('now')
    res = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True, encoding='utf-8')
    
    if res.returncode != 0:
        print("Error running test1.py!")
        print("Stdout:", res.stdout)
        print("Stderr:", res.stderr)
        sys.exit(res.returncode)
        
    print("test1.py completed successfully.")
    
    # Locate the results directory of test1.py
    # Setting format from test1.py:
    # test1_HunanLoad_2024_672_sl672_pl96_dm512_bs32_itr1_f59463f003
    results_dir = os.path.join(project_root, "results", "test1_HunanLoad_2024_672_sl672_pl96_dm512_bs32_itr1_f59463f003")
    
    if not os.path.exists(results_dir):
        # Scan results directory for anything starting with test1_HunanLoad_2024_672
        cand_dir = os.path.join(project_root, "results")
        if os.path.exists(cand_dir):
            for name in os.listdir(cand_dir):
                if name.startswith("test1_HunanLoad_2024_672"):
                    results_dir = os.path.join(cand_dir, name)
                    break
                    
    print(f"Loading predictions from: {results_dir}")
    
    # Load pred_inv.npy and true_inv.npy
    pred_inv = np.load(os.path.join(results_dir, "pred_inv.npy"))
    true_inv = np.load(os.path.join(results_dir, "true_inv.npy"))
    
    # Compute metrics
    mae, mse, rmse, mape, mspe = metric(pred_inv, true_inv)
    
    # R2
    target_mean = np.mean(true_inv)
    ss_tot = np.sum((true_inv - target_mean) ** 2)
    ss_res = np.sum((true_inv - pred_inv) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    # 1D contiguous timeline metrics
    def restore_sliding_window_3d(data_3d):
        if len(data_3d) == 0: return np.array([])
        restored = list(data_3d[0, :, :])
        for i in range(1, len(data_3d)):
            restored.append(data_3d[i, -1, :])
        return np.asarray(restored)

    pred_1d = restore_sliding_window_3d(pred_inv).squeeze()
    true_1d = restore_sliding_window_3d(true_inv).squeeze()

    mae_1d = float(np.mean(np.abs(pred_1d - true_1d)))
    mse_1d = float(np.mean((pred_1d - true_1d) ** 2))
    rmse_1d = float(np.sqrt(mse_1d))
    mape_1d = float(np.mean(np.abs((pred_1d - true_1d) / np.maximum(true_1d, 1e-5))))
    
    target_mean_1d = np.mean(true_1d)
    ss_tot_1d = np.sum((true_1d - target_mean_1d) ** 2)
    ss_res_1d = np.sum((true_1d - pred_1d) ** 2)
    r2_1d = float(1 - ss_res_1d / ss_tot_1d)
    
    metrics = {
        "model": "TimeXer",
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
        "shape": list(true_inv.shape)
    }
    
    print("\nPure TimeXer Baseline Test Metrics:")
    print(json.dumps(metrics, indent=2))
    
    # Save metrics in baseline_exp/timexer/
    timexer_dir = os.path.dirname(__file__)
    metrics_path = os.path.join(timexer_dir, "timexer_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    # Copy prediction csv and figure to baseline_exp/timexer/
    future_pred_src = os.path.join(results_dir, "future_load_prediction.csv")
    future_pred_dst = os.path.join(timexer_dir, "future_load_prediction.csv")
    if os.path.exists(future_pred_src):
        shutil.copy2(future_pred_src, future_pred_dst)
        print(f"Copied future prediction to {future_pred_dst}")
        
    png_src = os.path.join(results_dir, "pred_vs_true.png")
    png_dst = os.path.join(timexer_dir, "pred_vs_true.png")
    if os.path.exists(png_src):
        shutil.copy2(png_src, png_dst)
        print(f"Copied test set plot to {png_dst}")
        
    # Copy checkpoint
    ckpt_src = os.path.join(project_root, "checkpoints_test1", "test1_HunanLoad_2024_672_sl672_pl96_dm512_bs32_itr1_f59463f003", "checkpoint.pth")
    ckpt_dst = os.path.join(timexer_dir, "checkpoint.pth")
    if os.path.exists(ckpt_src):
        shutil.copy2(ckpt_src, ckpt_dst)
        print(f"Copied checkpoint to {ckpt_dst}")

if __name__ == "__main__":
    main()
