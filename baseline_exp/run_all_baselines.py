import os
import sys
import subprocess
import json
import time
import pandas as pd

def run_script(script_path, cwd):
    print(f"\n========================================\nRunning: {script_path}\n========================================")
    t0 = time.time()
    python_exe = sys.executable
    cmd = [python_exe, script_path]
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding='utf-8')
    elapsed = time.time() - t0
    print(f"Finished in {elapsed:.1f}s")
    if res.returncode != 0:
        print(f"Error executing {script_path}:")
        print("STDOUT:", res.stdout)
        print("STDERR:", res.stderr)
    else:
        print("Output logs:")
        print(res.stdout[-1500:]) # Show the end of the log
    return res.returncode

def main():
    base_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    
    # Run ARIMA
    arima_metrics_path = os.path.join(base_dir, "arima", "arima_metrics.json")
    if not os.path.exists(arima_metrics_path):
        arima_script = os.path.join(base_dir, "arima", "run_arima.py")
        code_arima = run_script(arima_script, project_root)
    else:
        print("ARIMA metrics file found, skipping execution.")
    
    # Run LSTM
    lstm_metrics_path = os.path.join(base_dir, "lstm", "lstm_metrics.json")
    if not os.path.exists(lstm_metrics_path):
        lstm_script = os.path.join(base_dir, "lstm", "run_lstm.py")
        code_lstm = run_script(lstm_script, project_root)
    else:
        print("LSTM metrics file found, skipping execution.")
    
    # Run TimeXer (Pure)
    timexer_metrics_path = os.path.join(base_dir, "timexer", "timexer_metrics.json")
    if not os.path.exists(timexer_metrics_path):
        timexer_script = os.path.join(base_dir, "timexer", "run_timexer_baseline.py")
        code_timexer = run_script(timexer_script, project_root)
    else:
        print("TimeXer metrics file found, skipping execution.")
    
    # Read and aggregate metrics
    models = ["ARIMA", "LSTM", "TimeXer"]
    metrics_files = [
        os.path.join(base_dir, "arima", "arima_metrics.json"),
        os.path.join(base_dir, "lstm", "lstm_metrics.json"),
        os.path.join(base_dir, "timexer", "timexer_metrics.json")
    ]
    
    aggregated_2d = []
    aggregated_1d = []
    
    for model_name, file_path in zip(models, metrics_files):
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # Standard 2D
                mape_val = data["mape"]
                if mape_val < 1.0:
                    mape_val *= 100
                aggregated_2d.append({
                    "Model": data.get("model", model_name),
                    "MAE": round(data["mae"], 6),
                    "MSE": round(data["mse"], 6),
                    "RMSE": round(data["rmse"], 6),
                    "MAPE (%)": round(mape_val, 2),
                    "R2": round(data["r2"], 4)
                })
                
                # Reconstructed 1D
                mape_1d_val = data.get("mape_1d", data["mape"])
                if mape_1d_val < 1.0:
                    mape_1d_val *= 100
                aggregated_1d.append({
                    "Model": data.get("model", model_name),
                    "MAE": round(data.get("mae_1d", data["mae"]), 6),
                    "MSE": round(data.get("mse_1d", data["mse"]), 6),
                    "RMSE": round(data.get("rmse_1d", data["rmse"]), 6),
                    "MAPE (%)": round(mape_1d_val, 2),
                    "R2": round(data.get("r2_1d", data["r2"]), 4)
                })
        else:
            print(f"Warning: metrics file not found for {model_name} at {file_path}")
            
    if aggregated_2d:
        header = "| Model | MAE | MSE | RMSE | MAPE (%) | R2 |"
        separator = "| :--- | :---: | :---: | :---: | :---: | :---: |"
        
        # 2D table
        rows_2d = []
        for row in aggregated_2d:
            rows_2d.append(f"| {row['Model']} | {row['MAE']:.6f} | {row['MSE']:.6f} | {row['RMSE']:.6f} | {row['MAPE (%)']:.2f}% | {row['R2']:.4f} |")
        md_table_2d = "\n".join([header, separator] + rows_2d)
        
        # 1D table
        rows_1d = []
        for row in aggregated_1d:
            rows_1d.append(f"| {row['Model']} | {row['MAE']:.6f} | {row['MSE']:.6f} | {row['RMSE']:.6f} | {row['MAPE (%)']:.2f}% | {row['R2']:.4f} |")
        md_table_1d = "\n".join([header, separator] + rows_1d)
        
        print("\n========================================")
        print("2D Sliding Window-average Metrics:")
        print("========================================")
        print(md_table_2d)
        
        print("\n========================================")
        print("1D Contiguous Reconstructed Timeline Metrics:")
        print("========================================")
        print(md_table_1d)
        print("========================================")
        
        summary_path = os.path.join(base_dir, "comparison_summary.md")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("# Baseline Models Comparison Summary\n\n")
            f.write("Evaluation on Test set (4:1:1 split, input 672, prediction 96):\n\n")
            f.write("## 1. 2D Sliding Window-average Metrics (Overlapping)\n")
            f.write("This table computes metrics by taking the average over all sliding windows (predicting 96 steps from each point in test set).\n\n")
            f.write(md_table_2d)
            f.write("\n\n")
            f.write("## 2. 1D Contiguous Reconstructed Timeline Metrics (Non-overlapping)\n")
            f.write("This table reconstructs a single chronological 1D time series from the overlapping windows and computes metrics on this continuous series (directly comparable to `test1.py`'s printed Plot Eval metrics).\n\n")
            f.write(md_table_1d)
            f.write("\n")
        print(f"Comparison summary saved to {summary_path}")

if __name__ == "__main__":
    main()
