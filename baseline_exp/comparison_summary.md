# Baseline Models Comparison Summary

Evaluation on Test set (4:1:1 split, input 672, prediction 96):

## 1. 2D Sliding Window-average Metrics (Overlapping)
This table computes metrics by taking the average over all sliding windows (predicting 96 steps from each point in test set).

| Model | MAE | MSE | RMSE | MAPE (%) | R2 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| ARIMA | 0.125070 | 0.028472 | 0.168737 | 21.53% | -0.3221 |
| LSTM | 0.025039 | 0.001083 | 0.032914 | 4.07% | 0.9497 |
| TimeXer | 0.034718 | 0.002351 | 0.048486 | 5.29% | 0.8908 |

## 2. 1D Contiguous Reconstructed Timeline Metrics (Non-overlapping)
This table reconstructs a single chronological 1D time series from the overlapping windows and computes metrics on this continuous series (directly comparable to `test1.py`'s printed Plot Eval metrics).

| Model | MAE | MSE | RMSE | MAPE (%) | R2 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| ARIMA | 0.026502 | 0.001661 | 0.040753 | 4.38% | 0.9240 |
| LSTM | 0.023463 | 0.000964 | 0.031044 | 3.74% | 0.9559 |
| TimeXer | 0.042844 | 0.003507 | 0.059217 | 6.42% | 0.8395 |
