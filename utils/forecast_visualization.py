"""
forecast_visualization.py - 预测结果可视化与未来预测工具模块

该模块提供了模型测试结果的可视化功能，以及加载外部未来气象数据
并使用已训练模型进行未来连续多步负荷预测的完整流程。支持分位数预测结果的展示与置信区间绘制。
"""

import os
from typing import Any, Callable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from utils.metrics import cal_eval
from utils.timefeatures import time_features


def _find_quantile_index(quantiles: Sequence[float], value: float) -> Optional[int]:
    """
    在一个分位数列表中查找目标分位数对应的索引位置。

    参数:
        quantiles (Sequence[float]): 模型预测所使用的分位数列表 (如 [0.1, 0.5, 0.9])
        value (float): 需要查找的目标分位数值 (如 0.5)

    返回:
        Optional[int]: 目标分位数在列表中的索引。如果未找到则返回 None。
    """
    for idx, quantile in enumerate(quantiles):
        # 处理浮点数精度问题，允许 1e-8 的误差
        if abs(float(quantile) - value) < 1e-8:
            return idx
    return None


def restore_sliding_window_2d(data_2d: np.ndarray) -> np.ndarray:
    """
    将基于滑窗采样的 2D 数据恢复为连续的 1D 时间序列。
    适用于没有特征维度的直接输出 (如 shape: [batch_size, pred_len])。

    在自回归或滑窗预测中，连续的 batch(预测目标) 常有重叠。
    此函数通过拼接第一个窗口的全部元素，以及后续窗口的最后一个元素来还原真实序列。

    参数:
        data_2d (np.ndarray): 形状为 [N, W] 的 2D 数组，N 为样本数，W 为窗口长度(pred_len)。

    返回:
        np.ndarray: 还原后的 1D 连续序列。
    """
    if len(data_2d) == 0:
        return np.array([])
    # 取第一个窗口的完整序列
    restored = list(data_2d[0, :])
    # 之后每个窗口只取其最后一步（因为步长为 1）
    for i in range(1, len(data_2d)):
        restored.append(data_2d[i, -1])
    return np.asarray(restored)


def restore_sliding_window_3d(data_3d: np.ndarray) -> np.ndarray:
    """
    将基于滑窗采样的 3D 数据恢复为连续的 2D 时间序列。
    适用于具有特征维度的特征预测或分位数预测 (如 shape: [batch_size, pred_len, feature_dim])。

    参数:
        data_3d (np.ndarray): 形状为 [N, W, D] 的 3D 数组。

    返回:
        np.ndarray: 形状为 [L, D] 的 2D 连续序列。
    """
    if len(data_3d) == 0:
        return np.array([])
    # 取第一个窗口的完整序列及其全部特征维度
    restored = list(data_3d[0, :, :])
    # 之后每个窗口只取其时间序列上的最后一步
    for i in range(1, len(data_3d)):
        restored.append(data_3d[i, -1, :])
    return np.asarray(restored)


def _load_ordered_dataframe(csv_path: str, target: str) -> pd.DataFrame:
    """
    从 CSV 文件读取历史负荷/气象标签数据，确保按时间递增顺序排列，
    并能够自动适配不同情况下的目标列名。

    参数:
        csv_path (str): 历史负荷数据文件的路径。
        target (str): 指定的目标变量列名 (如 "load")。

    返回:
        pd.DataFrame: 处理好的 DataFrame，按 `date` 列升序排序，且包含特征及目标列。
    """
    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        raise ValueError(f"Missing date column in {csv_path}")
    
    # 强制时间转换并按时间排序，确保时序关系不变
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    # 如果指定 target 不在列中，但有 Target 别名，则重命名
    if target not in df.columns and "Target" in df.columns:
        df = df.rename(columns={"Target": target})
    if target not in df.columns:
        raise ValueError(f"Missing target column {target} in {csv_path}")
    
    # 将 date 和其他非目标列放在前面，target 放在最后一列以符合通用格式
    other_cols = [col for col in df.columns if col not in ("date", target)]
    return df[["date"] + other_cols + [target]]


def plot_pred_vs_true(
    results_dir: str,
    feat_idx: int = 0,
    out_name: str = "pred_vs_true.png",
    use_inverse: bool = False,
    quantiles: Optional[Sequence[float]] = None,
    title_prefix: str = "Prediction",
    y_label: str = "Load (MW)",
) -> None:
    """
    绘制测试集上模型预测值与真实标签的对比曲线 (支持带 P10-P90 置信区间)。
    该函数会自动加载 `test_quantile_model` 保存在 results_dir 目录下的 npy 文件。

    参数:
        results_dir (str): 模型预测输出 npy 结果文件的存储目录
        feat_idx (int): 多变量下需要绘制的特征维索引，默认为 0（单变量即它）
        out_name (str): 保存输出图片的文件名
        use_inverse (bool): 是否使用反归一化后的数据绘制图像（单位物理量）
        quantiles (Optional[Sequence[float]]): 模型支持的分位数配置，用于提取 P10 与 P90
        title_prefix (str): 图像主标题的前缀
        y_label (str): y 轴自定义标签名字
    """
    try:
        plt.switch_backend("TkAgg")
    except Exception:
        pass

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

    preds = np.load(pred_path)
    trues = np.load(true_path)

    has_quantiles = os.path.exists(quantile_path)
    quantile_preds = np.load(quantile_path) if has_quantiles else None
    p10_idx = _find_quantile_index(quantiles or [], 0.1) if has_quantiles else None
    p90_idx = _find_quantile_index(quantiles or [], 0.9) if has_quantiles else None
    can_plot_band = has_quantiles and p10_idx is not None and p90_idx is not None

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

    if can_plot_band:
        q_p10_raw = quantile_preds[:, :, p10_idx : p10_idx + 1]
        q_p90_raw = quantile_preds[:, :, p90_idx : p90_idx + 1]

        p10_seq = restore_sliding_window_3d(q_p10_raw)
        p90_seq = restore_sliding_window_3d(q_p90_raw)

        if p10_seq.ndim == 2:
            p10_series = p10_seq[:, min(feat_idx, p10_seq.shape[1] - 1)]
            p90_series = p90_seq[:, min(feat_idx, p90_seq.shape[1] - 1)]
        else:
            p10_series = p10_seq.reshape(-1)
            p90_series = p90_seq.reshape(-1)

    eval_df = cal_eval(true_series, pred_series)
    print("[Plot Eval] metrics:")
    print(eval_df)

    os.makedirs(results_dir, exist_ok=True)
    mape_val = eval_df.iloc[0]["MAPE"]

    fig, ax = plt.subplots(1, 1, figsize=(15, 5), facecolor="white")
    ax.plot(true_series, label="GroundTruth", alpha=0.8, color="tab:blue")
    ax.plot(pred_series, label="Prediction (P50)", alpha=0.7, color="tab:orange")

    if can_plot_band:
        ax.fill_between(
            range(len(p10_series)),
            p10_series,
            p90_series,
            alpha=0.2,
            color="tab:orange",
            label="P10-P90 Confidence Interval",
        )

    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    if np.isfinite(mape_val):
        ax.set_title(f"{title_prefix} - MAPE: {100 * mape_val:.2f}%")
    else:
        ax.set_title(f"{title_prefix} - MAPE: NaN")
    ax.set_xlabel("Time Step")
    ax.set_ylabel(y_label)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, out_name), dpi=600, bbox_inches="tight")
    plt.show()


def predict_future_load_from_csv(
    model,
    args,
    device,
    weather_store: Any,
    results_dir: str,
    future_path: str,
    steps: int,
    use_inverse: bool = True,
    quantiles: Optional[Sequence[float]] = None,
    data_provider_fn: Optional[Callable[..., Any]] = None,
    model_label: str = "Forecast",
    y_label: str = "Load (MW)",
) -> None:
    """
    加载历史数据及给定的未来气象文件，执行端到端的未来多步（含扩展气象视窗）负荷预测。
    该函数用于将完全训练的模型推广到生产或未来模拟中去，绘制并生成包含置信区间的对应 CSV 与 PNG 文件。

    参数:
        model: 训练好的模型，如 FullMapConvTimeXerQuantile
        args: 初始化并运行模型使用的配置（包含 seq_len, pred_len 等属性）
        device: 模型部署所在的计算设备 (cpu 或 cuda)
        weather_store: WeatherGridStore 的实例，负责读取 4D 气象 HDF5 数据
        results_dir: 预测结果 PNG 和 CSV 文件保存路径
        future_path: 指定期望预测的时间范围的占位文件 (格式与原始数据相似，须含有未来需要的日期 date 列)
        steps: 尝试向后预测的步数最大限制
        use_inverse: 置真则利用 Dataset 的反归一化操作重构预测输出为原始负荷单位
        quantiles: 运行模型时用的分位数列表 (需要提取 P10, P50, P90)
        data_provider_fn: 返回 (dataset, dataloader) 的天气初始化回调，借此获取 Scaler
        model_label: 画图标签配置里的模型名称 (仅用来格式化图标)
        y_label: Y 轴文字名称
    """
    if quantiles is None:
        raise ValueError("quantiles must be provided.")
    if data_provider_fn is None:
        raise ValueError("data_provider_fn must be provided.")

    p10_idx = _find_quantile_index(quantiles, 0.1)
    p50_idx = _find_quantile_index(quantiles, 0.5)
    p90_idx = _find_quantile_index(quantiles, 0.9)
    if p50_idx is None:
        raise ValueError("quantiles must include 0.5 for P50 output.")

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
        future_df = pd.read_csv(abs_future_path)
    except Exception as exc:
        print(f"Load csv failed: {exc}")
        return

    if "date" not in future_df.columns:
        print(f"Future file missing date column: {abs_future_path}")
        return
    future_df["date"] = pd.to_datetime(future_df["date"])
    future_df = future_df.sort_values("date").reset_index(drop=True)

    if len(history_df) < args.seq_len:
        print(f"History length ({len(history_df)}) < seq_len ({args.seq_len}), skip.")
        return

    predict_steps = min(int(steps), len(future_df), args.pred_len)
    if predict_steps < args.pred_len:
        print(f"Future rows ({predict_steps}) < pred_len ({args.pred_len}), skip.")
        return

    ref_data, _ = data_provider_fn(args, "train", weather_store)

    hist_dates = pd.to_datetime(history_df["date"].iloc[-args.seq_len :].values)
    future_dates = pd.to_datetime(future_df["date"].iloc[: args.pred_len].values)
    hist_load = history_df[args.target].iloc[-args.seq_len :].values.astype(np.float32).reshape(-1, 1)
    hist_load_scaled = ref_data.scale_target(hist_load)

    # 扩展气象数据范围：历史 + 未来预测期
    all_dates = np.concatenate([hist_dates, future_dates])
    hist_weather = weather_store.fetch_frames_by_dates(all_dates)

    # 内生变量时间标记（仅历史期）
    x_mark_np = time_features(pd.to_datetime(hist_dates), freq=args.freq).transpose(1, 0).astype(np.float32)
    # 外生变量时间标记（历史 + 未来期）
    exo_mark_np = time_features(pd.to_datetime(all_dates), freq=args.freq).transpose(1, 0).astype(np.float32)

    # 执行模型前向预测算子，此时禁用了梯度追踪
    model.eval()
    with torch.inference_mode():
        # [B, L, 1] - 提供目标序列前截断序列的输入窗口
        batch_x = torch.as_tensor(hist_load_scaled, dtype=torch.float32, device=device).unsqueeze(0)
        # [B, L_endo, T] - 提供目标序列输入窗口对应的时间特性
        batch_x_mark = torch.as_tensor(x_mark_np, dtype=torch.float32, device=device).unsqueeze(0)
        # [B, L_exo, T] - 外源时频变量的时间标记提取（用于在 timeXer 中计算更长的交叉注意力映射）
        batch_exo_mark = torch.as_tensor(exo_mark_np, dtype=torch.float32, device=device).unsqueeze(0)
        # [B, L_exo, C, H, W] - 直接取出气象库内的环境光栅时基文件组合特征矩阵
        batch_weather_x = torch.as_tensor(hist_weather, dtype=torch.float32, device=device).unsqueeze(0)

        outputs = model(
            load_x=batch_x,
            x_mark_enc=batch_x_mark,
            x_exo_mark=batch_exo_mark,
            weather_x=batch_weather_x,
        )

    # 模型输出维度：[Batch=1, pred_len, n_quantiles]
    quantile_scaled = outputs[0, : args.pred_len, :].detach().cpu().numpy()
    p50_scaled = quantile_scaled[:, p50_idx]

    if use_inverse:
        preds_p50 = ref_data.inverse_transform_target(p50_scaled.reshape(-1, 1)).reshape(-1)
        history_target = history_df[args.target].values
        preds_p10 = (
            ref_data.inverse_transform_target(quantile_scaled[:, p10_idx].reshape(-1, 1)).reshape(-1)
            if p10_idx is not None
            else None
        )
        preds_p90 = (
            ref_data.inverse_transform_target(quantile_scaled[:, p90_idx].reshape(-1, 1)).reshape(-1)
            if p90_idx is not None
            else None
        )
    else:
        preds_p50 = p50_scaled
        history_target = hist_load_scaled.reshape(-1)
        preds_p10 = quantile_scaled[:, p10_idx] if p10_idx is not None else None
        preds_p90 = quantile_scaled[:, p90_idx] if p90_idx is not None else None

    future_dates = pd.Series(future_dates[:predict_steps])
    preds_p50 = preds_p50[:predict_steps]
    if preds_p10 is not None:
        preds_p10 = preds_p10[:predict_steps]
    if preds_p90 is not None:
        preds_p90 = preds_p90[:predict_steps]

    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "future_load_prediction.csv")
    output_payload = {
        "date": future_dates,
        f"{args.target}_pred_P50": preds_p50,
    }
    if preds_p10 is not None:
        output_payload[f"{args.target}_pred_P10"] = preds_p10
    if preds_p90 is not None:
        output_payload[f"{args.target}_pred_P90"] = preds_p90
    pd.DataFrame(output_payload).to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\nFuture {predict_steps}-step {args.target} predictions:")
    if preds_p10 is not None and preds_p90 is not None:
        print(f"{'Time':<25} {'P10':<12} {'P50':<12} {'P90':<12}")
        print("-" * 65)
        for i in range(predict_steps):
            print(
                f"  {future_dates.iloc[i]}: "
                f"{preds_p10[i]:<12.4f} {preds_p50[i]:<12.4f} {preds_p90[i]:<12.4f}"
            )
    else:
        print(f"{'Time':<25} {'P50':<12}")
        print("-" * 40)
        for i in range(predict_steps):
            print(f"  {future_dates.iloc[i]}: {preds_p50[i]:<12.4f}")

    n_history = min(args.seq_len, len(history_target))
    history_tail = history_target[-n_history:]
    future_x = range(n_history, n_history + predict_steps)

    plt.figure(figsize=(15, 6), facecolor="white")
    plt.plot(range(n_history), history_tail, label="Historical Load", color="tab:blue", alpha=0.8)
    plt.plot(
        future_x,
        preds_p50,
        label=f"{model_label} P50 Prediction",
        color="tab:orange",
        linewidth=2,
        marker="o",
        markersize=2,
    )
    if preds_p10 is not None and preds_p90 is not None:
        plt.fill_between(
            future_x,
            preds_p10,
            preds_p90,
            alpha=0.25,
            color="tab:orange",
            label="P10-P90 Confidence Interval",
        )
    plt.axvline(x=n_history - 0.5, color="gray", linestyle="--", alpha=0.6, label="Prediction Start")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.title(f"{model_label} Future {predict_steps}-Step Load Prediction")
    plt.xlabel("Time Step (15min)")
    plt.ylabel(y_label)
    plt.tight_layout()

    out_fig = os.path.join(results_dir, "future_load_prediction.png")
    plt.savefig(out_fig, dpi=600, bbox_inches="tight")
    plt.show()

    print(f"Saved future prediction csv: {out_csv}")
    print(f"Saved future prediction figure: {out_fig}")


__all__ = [
    "plot_pred_vs_true",
    "predict_future_load_from_csv",
    "restore_sliding_window_2d",
    "restore_sliding_window_3d",
]
