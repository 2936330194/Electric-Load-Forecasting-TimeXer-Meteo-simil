import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def _safe_percentage_denominator(true, eps=1e-8):
    true = np.asarray(true, dtype=np.float64)
    abs_true = np.abs(true)
    valid_mask = abs_true > float(eps)
    safe_denominator = np.where(valid_mask, abs_true, float(eps))
    return true, valid_mask, safe_denominator


def MAPE(pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true, valid_mask, safe_denominator = _safe_percentage_denominator(true)
    percentage_error = np.abs(true - pred) / safe_denominator
    if np.any(valid_mask):
        return float(np.mean(percentage_error[valid_mask]))
    return float(np.mean(percentage_error))


def MSPE(pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true, valid_mask, safe_denominator = _safe_percentage_denominator(true)
    squared_percentage_error = np.square((true - pred) / safe_denominator)
    if np.any(valid_mask):
        return float(np.mean(squared_percentage_error[valid_mask]))
    return float(np.mean(squared_percentage_error))


def R2(pred, true):
    pred = np.asarray(pred).reshape(-1)
    true = np.asarray(true).reshape(-1)
    return r2_score(true, pred)


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    return mae, mse, rmse, mape, mspe


def cal_eval(y_real: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """计算评估指标"""
    y_real = np.asarray(y_real).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    r2 = r2_score(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_real, y_pred)
    mape = MAPE(y_pred, y_real)

    return pd.DataFrame(
        {"R2": [r2], "MSE": [mse], "RMSE": [rmse], "MAE": [mae], "MAPE": [mape]},
        index=["Eval"],
    )
