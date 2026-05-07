"""
forecast_visualization.py - 预测结果可视化与未来预测工具模块

该模块提供了模型测试结果的可视化功能，以及加载外部未来气象数据
并使用已训练模型进行未来连续多步负荷预测的完整流程。支持分位数预测结果的展示与置信区间绘制。
"""

import copy
import inspect
import json
import os
from typing import Any, Callable, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import torch

from utils.metrics import cal_eval
from utils.timefeatures import time_features
from utils.weather_e2e import build_weather_sequence_timestamps


def _configure_matplotlib_cjk_font() -> None:
    """
    为 Matplotlib 配置可用的中文字体回退，避免 Windows 下默认 DejaVu Sans 缺字。
    """
    preferred_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
    ]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    available = [name for name in preferred_fonts if name in installed]
    if not available:
        return

    current = list(plt.rcParams.get("font.sans-serif", []))
    merged = []
    for name in available + current:
        if name not in merged:
            merged.append(name)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = merged
    plt.rcParams["axes.unicode_minus"] = False


_configure_matplotlib_cjk_font()


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


def _build_future_similar_day_prior_tensor(
    similar_day_result: Any,
    ref_data: Any,
    pred_len: int,
    top_k: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    将未来预测时的 Top-K 相似日检索结果整理为模型可直接使用的先验张量。

    参数:
        similar_day_result (Any): 包含检索到的相似日负荷曲线及得分的对象
        ref_data (Any): 用于数据归一化的参考数据集对象（包含 Scaler）
        pred_len (int): 预测长度
        top_k (int): 取前 K 个相似日
        device (torch.device): 数据放置的目标设备 (CPU/CUDA)

    返回:
        Optional[torch.Tensor]: 形状为 [1, pred_len, top_k + 1] 的先验特征张量。
                                如果检索结果无效则返回 None。
    """
    if similar_day_result is None or int(pred_len) <= 0 or int(top_k) <= 0:
        return None

    # 从检索结果中提取负荷曲线数据，各行代表一个相似日的负荷序列
    curves = np.asarray(getattr(similar_day_result, "load_curves", []), dtype=np.float32)
    # 容错处理：确保曲线数组是二维的 [n_similar_days, time_steps]
    if curves.ndim == 1:
        curves = curves.reshape(1, -1)
    if curves.ndim != 2 or curves.size == 0:
        return None

    # 截取预测视窗长度部分的负荷曲线
    curves = curves[:, : int(pred_len)]
    # 使用参考数据集的缩放算子对相似日负荷进行归一化，使其处于模型训练时的数值空间
    curves_scaled = ref_data.scale_target(curves.reshape(-1, 1)).reshape(curves.shape[0], curves.shape[1])
    # 提取各相似日对应的检索相似度得分
    scores = np.asarray(getattr(similar_day_result, "similarity_scores", []), dtype=np.float32)

    from utils.weather_e2e import _build_similar_day_prior_features

    # 调用底层构建工具，将多条相似日曲线与得分融合为 [pred_len, top_k + 1] 维度的特征矩阵
    prior_features = _build_similar_day_prior_features(
        load_curves=curves_scaled,
        similarity_scores=scores,
        pred_len=int(pred_len),
        top_k=int(top_k),
        shift_steps=0,
    )
    return torch.as_tensor(prior_features, dtype=torch.float32, device=device).unsqueeze(0)


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


def _scale_target_generic(ref_data: Any, data: np.ndarray) -> np.ndarray:
    if hasattr(ref_data, "scale_target"):
        return ref_data.scale_target(data)
    scaler = getattr(ref_data, "scaler", None)
    if scaler is not None and hasattr(scaler, "transform"):
        return scaler.transform(data)
    return data


def _inverse_transform_target_generic(ref_data: Any, data: np.ndarray) -> np.ndarray:
    if hasattr(ref_data, "inverse_transform_target"):
        return ref_data.inverse_transform_target(data)
    if hasattr(ref_data, "inverse_transform"):
        return ref_data.inverse_transform(data)
    scaler = getattr(ref_data, "scaler", None)
    if scaler is not None and hasattr(scaler, "inverse_transform"):
        return scaler.inverse_transform(data)
    return data


def plot_pred_vs_true(
    results_dir: str,
    feat_idx: int = 0,
    out_name: str = "pred_vs_true.png",
    use_inverse: bool = False,
    quantiles: Optional[Sequence[float]] = None,
    title_prefix: str = "Prediction",
    y_label: str = "Load (MW)",
    eval_first_n_steps: Optional[int] = None,
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
    # 尝试切换后端，防止在某些环境中绘图引擎冲突
    try:
        plt.switch_backend("TkAgg")
    except Exception:
        pass

    # 根据是否需要反归一化，确定 npy 文件的数据源路径
    if use_inverse:
        pred_path = os.path.join(results_dir, "pred_inv.npy")
        true_path = os.path.join(results_dir, "true_inv.npy")
        quantile_path = os.path.join(results_dir, "quantile_preds_inv.npy")
        # 如果反归一化文件不存在，则回退到原始归一化文件
        if not os.path.exists(pred_path):
            pred_path = os.path.join(results_dir, "pred.npy")
            true_path = os.path.join(results_dir, "true.npy")
            quantile_path = os.path.join(results_dir, "quantile_preds.npy")
    else:
        pred_path = os.path.join(results_dir, "pred.npy")
        true_path = os.path.join(results_dir, "true.npy")
        quantile_path = os.path.join(results_dir, "quantile_preds.npy")

    # 检查核心文件是否存在，不存在则跳过绘图
    if not os.path.exists(pred_path) or not os.path.exists(true_path):
        print("Prediction files not found, skip plotting.")
        return

    # 加载预测值与真实值
    preds = np.load(pred_path)
    trues = np.load(true_path)

    # 检查是否存在分位数预测结果，并提取 P10 与 P90 的索引
    has_quantiles = os.path.exists(quantile_path)
    quantile_preds = np.load(quantile_path) if has_quantiles else None
    p10_idx = _find_quantile_index(quantiles or [], 0.1) if has_quantiles else None
    p90_idx = _find_quantile_index(quantiles or [], 0.9) if has_quantiles else None
    # 只有当包含分位数预测、且指定了 P10/P90 索引时，才能绘制置信区间带
    can_plot_band = has_quantiles and p10_idx is not None and p90_idx is not None

    # 将滑窗采样形状的数据还原为一维时间序列
    if preds.ndim == 3:
        # [batch_size, pred_len, d_model]
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
        # [batch_size, pred_len]
        pred_series = restore_sliding_window_2d(preds)
        true_series = restore_sliding_window_2d(trues)
    else:
        pred_series = preds.reshape(-1)
        true_series = trues.reshape(-1)

    # 如果可以绘制置信区间带，对应提取 P10 和 P90 序列并解滑窗
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

    # 计算预测指标并在控制台打印
    eval_df = cal_eval(true_series, pred_series)
    print("[Plot Eval] metrics:")
    print(eval_df)
    if eval_first_n_steps is not None:
        first_n_steps = int(eval_first_n_steps)
        if preds.ndim >= 2 and trues.ndim >= 2 and first_n_steps > 0:
            usable_steps = min(first_n_steps, preds.shape[1], trues.shape[1])
            first_eval_df = cal_eval(
                trues[:, :usable_steps, ...],
                preds[:, :usable_steps, ...],
            )
            print(f"[Plot Eval] first {usable_steps} steps metrics (overlap included):")
            print(first_eval_df)

    # 创建保存目录
    os.makedirs(results_dir, exist_ok=True)
    mape_val = eval_df.iloc[0]["MAPE"]

    # 初始化绘图画布
    fig, ax = plt.subplots(1, 1, figsize=(15, 5), facecolor="white")
    # 绘制真实值与 P50 预测值
    ax.plot(true_series, label="GroundTruth", alpha=0.8, color="tab:blue")
    ax.plot(pred_series, label="Prediction (P50)", alpha=0.7, color="tab:orange")

    # 绘制 P10-P90 置信区间阴影部分
    if can_plot_band:
        ax.fill_between(
            range(len(p10_series)),
            p10_series,
            p90_series,
            alpha=0.2,
            color="tab:orange",
            label="P10-P90 Confidence Interval",
        )

    # 图表细节配置：图例、网格、标题及标签
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    if np.isfinite(mape_val):
        ax.set_title(f"{title_prefix} - MAPE: {100 * mape_val:.2f}%")
    else:
        ax.set_title(f"{title_prefix} - MAPE: NaN")
    ax.set_xlabel("Time Step")
    ax.set_ylabel(y_label)
    
    # 自动调整布局并保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, out_name), dpi=600, bbox_inches="tight")
    plt.show()


def predict_future_load_from_csv(
    model,
    args,
    device,
    weather_store: Optional[Any],
    results_dir: str,
    future_path: str,
    steps: int,
    use_inverse: bool = True,
    quantiles: Optional[Sequence[float]] = None,
    data_provider_fn: Optional[Callable[..., Any]] = None,
    model_label: str = "Forecast",
    y_label: str = "Load (MW)",
    similar_day_result: Optional[Any] = None,
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
        similar_day_result: 可选的相似日检索结果；若提供，则叠加其 Top-K 负荷曲线到未来预测图
    """
    # 基础参数校验
    if quantiles is None:
        raise ValueError("quantiles must be provided.")
    if data_provider_fn is None:
        raise ValueError("data_provider_fn must be provided.")

    # 提取 P10、P50 (中位数) 和 P90 分位数的索引，用于结果展示和区间绘制
    p10_idx = _find_quantile_index(quantiles, 0.1)
    p50_idx = _find_quantile_index(quantiles, 0.5)
    p90_idx = _find_quantile_index(quantiles, 0.9)
    if p50_idx is None:
        raise ValueError("quantiles must include 0.5 for P50 output.")

    print("\n" + "=" * 60)
    print(f"Future Forecast: from {future_path}")
    print("=" * 60)

    # 路径合法性检查
    abs_future_path = os.path.abspath(future_path)
    if not os.path.exists(abs_future_path):
        print(f"Future file not found, skip: {abs_future_path}")
        return

    history_path = os.path.join(args.root_path, args.data_path)
    if not os.path.exists(history_path):
        print(f"History file not found, skip: {history_path}")
        return

    # 加载历史负荷数据与带有未来时间戳的气象/占位文件
    try:
        history_df = _load_ordered_dataframe(history_path, args.target)
        future_df = pd.read_csv(abs_future_path)
    except Exception as exc:
        print(f"Load csv failed: {exc}")
        return

    # 预处理未来数据的时间戳
    if "date" not in future_df.columns:
        print(f"Future file missing date column: {abs_future_path}")
        return
    future_df["date"] = pd.to_datetime(future_df["date"])
    future_df = future_df.sort_values("date").reset_index(drop=True)

    # 确保历史数据长度满足模型的 lookback 窗口要求
    if len(history_df) < args.seq_len:
        print(f"History length ({len(history_df)}) < seq_len ({args.seq_len}), skip.")
        return

    # 计算实际预测步目，受限于 steps 参数、CSV 行数和模型最大预测能力
    predict_steps = min(int(steps), len(future_df), args.pred_len)
    if predict_steps < args.pred_len:
        print(f"Future rows ({predict_steps}) < pred_len ({args.pred_len}), skip.")
        return

    # 获取训练集的数据提供器，主要为了获取归一化算子 (Scaler)
    ref_args = copy.copy(args)
    if hasattr(ref_args, "use_similar_day_prior"):
        ref_args.use_similar_day_prior = False
    if weather_store is None:
        ref_data, _ = data_provider_fn(ref_args, "train")
    else:
        ref_data, _ = data_provider_fn(ref_args, "train", weather_store)

    # 提取时间轴：seq_len 长度的历史日期 + pred_len 长度的未来日期
    hist_dates = pd.to_datetime(history_df["date"].iloc[-args.seq_len :].values)
    future_dates = pd.to_datetime(future_df["date"].iloc[: args.pred_len].values)
    
    # 提取历史负荷并进行归一化
    hist_load = history_df[args.target].iloc[-args.seq_len :].values.astype(np.float32).reshape(-1, 1)
    hist_load_scaled = _scale_target_generic(ref_data, hist_load)

    # 准备气象特征：涵盖历史窗口与未来预测窗口，共 seq_len + pred_len 步
    weather_dates = None
    hist_weather = None
    weather_mark_freq = None
    if weather_store is not None:
        weather_freq = getattr(args, "weather_step_freq", None)
        if weather_freq is None and getattr(weather_store, "native_freq", None) is not None:
            weather_freq = weather_store.native_freq
        weather_seq_len = int(getattr(args, "weather_seq_len", args.seq_len + args.pred_len))
        weather_history_len = int(getattr(args, "weather_history_len", args.seq_len))
        weather_mark_freq = (
            getattr(args, "weather_mark_freq", None)
            or getattr(args, "weather_step_freq", None)
            or args.freq
        )
        reference_weather_timestamps_ns = None
        if hasattr(weather_store, "sources"):
            candidate_timestamps = [
                np.asarray(source.get("timestamps_ns"), dtype=np.int64)
                for source in getattr(weather_store, "sources", [])
                if source.get("timestamps_ns") is not None
            ]
            if candidate_timestamps:
                reference_weather_timestamps_ns = np.concatenate(candidate_timestamps, axis=0)

        weather_dates = build_weather_sequence_timestamps(
            target_start=pd.Timestamp(future_dates[0]),
            weather_seq_len=weather_seq_len,
            weather_history_len=weather_history_len,
            weather_freq=weather_freq,
            reference_timestamps_ns=reference_weather_timestamps_ns,
        )
        hist_weather = weather_store.fetch_frames_by_dates(weather_dates)
    
    # 如果模型启用了相似日先验且提供了检索结果，则构建先验张量
    similar_day_prior_tensor = None
    if bool(getattr(model, "use_similar_day_prior", False)) and similar_day_result is not None:
        similar_day_prior_tensor = _build_future_similar_day_prior_tensor(
            similar_day_result=similar_day_result,
            ref_data=ref_data,
            pred_len=args.pred_len,
            top_k=int(getattr(model, "similar_day_top_k", getattr(args, "similar_day_top_k", 3))),
            device=device,
        )

    # 生成时间位置编码 (Time Features)
    # 内生变量标记仅针对历史窗口
    x_mark_np = time_features(pd.to_datetime(hist_dates), freq=args.freq).transpose(1, 0).astype(np.float32)
    # 外生变量标记针对全部窗口 (历史 + 未来)
    exo_mark_np = None
    if weather_dates is not None:
        exo_mark_np = time_features(
            pd.to_datetime(weather_dates.values),
            freq=weather_mark_freq,
        ).transpose(1, 0).astype(np.float32)

    # 切换模型为评估模式，并禁用梯度计算
    model.eval()
    with torch.inference_mode():
        # 封装为 Tensor 并增加 Batch 维度 (B=1)
        # batch_x: [1, seq_len, 1] 历史负荷
        batch_x = torch.as_tensor(hist_load_scaled, dtype=torch.float32, device=device).unsqueeze(0)
        # batch_x_mark: [1, seq_len, T] 历史时间编码
        batch_x_mark = torch.as_tensor(x_mark_np, dtype=torch.float32, device=device).unsqueeze(0)
        # batch_exo_mark: [1, seq_len + pred_len, T] 全局时间编码
        batch_exo_mark = torch.as_tensor(exo_mark_np, dtype=torch.float32, device=device).unsqueeze(0)
        # batch_weather_x: [1, seq_len + pred_len, C, H, W] 气象格点数据
        batch_weather_x = torch.as_tensor(hist_weather, dtype=torch.float32, device=device).unsqueeze(0)

        # 执行模型前向传播，得到分位数预测结果
        model_kwargs = {
            "load_x": batch_x,
            "x_mark_enc": batch_x_mark,
            "x_exo_mark": batch_exo_mark,
            "weather_x": batch_weather_x,
        }
        forward_params = inspect.signature(model.forward).parameters
        if (
            similar_day_prior_tensor is not None
            and "similar_day_prior" in forward_params
        ):
            model_kwargs["similar_day_prior"] = similar_day_prior_tensor

        outputs = model(**model_kwargs)

    # 处理模型输出，维度为 [1, pred_len, n_quantiles]
    quantile_scaled = outputs[0, : args.pred_len, :].detach().cpu().numpy()
    p50_scaled = quantile_scaled[:, p50_idx]

    # 根据配置决定是否执行反归一化，将预测值转回真实单位
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

    # 根据 predict_steps 截取有效预测部分
    effective_future_dates = pd.Series(future_dates[:predict_steps])
    preds_p50 = preds_p50[:predict_steps]
    if preds_p10 is not None:
        preds_p10 = preds_p10[:predict_steps]
    if preds_p90 is not None:
        preds_p90 = preds_p90[:predict_steps]

    # 保存预测结果到 CSV 文件
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "future_load_prediction.csv")
    output_payload = {
        "date": effective_future_dates,
        f"{args.target}_pred_P50": preds_p50,
    }
    if preds_p10 is not None:
        output_payload[f"{args.target}_pred_P10"] = preds_p10
    if preds_p90 is not None:
        output_payload[f"{args.target}_pred_P90"] = preds_p90
    pd.DataFrame(output_payload).to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 在控制台打印预测数值概览
    print(f"\nFuture {predict_steps}-step {args.target} predictions:")
    if preds_p10 is not None and preds_p90 is not None:
        print(f"{'Time':<25} {'P10':<12} {'P50':<12} {'P90':<12}")
        print("-" * 65)
        for i in range(predict_steps):
            print(
                f"  {effective_future_dates.iloc[i]}: "
                f"{preds_p10[i]:<12.4f} {preds_p50[i]:<12.4f} {preds_p90[i]:<12.4f}"
            )
    else:
        print(f"{'Time':<25} {'P50':<12}")
        print("-" * 40)
        for i in range(predict_steps):
            print(f"  {effective_future_dates.iloc[i]}: {preds_p50[i]:<12.4f}")

    # 开始可视化：合并历史尾部数据与未来预测数据
    n_history = min(args.seq_len, len(history_target))
    history_tail = history_target[-n_history:]
    future_x = range(n_history, n_history + predict_steps)

    plt.figure(figsize=(15, 6), facecolor="white")
    # 绘制历史曲线
    plt.plot(range(n_history), history_tail, label="Historical Load", color="tab:blue", alpha=0.8)
    # 绘制预测曲线 (P50)
    plt.plot(
        future_x,
        preds_p50,
        label=f"{model_label} P50 Prediction",
        color="tab:orange",
        linewidth=2,
        marker="o",
        markersize=2,
    )
    # 填充 P10 到 P90 的置信区间阴影
    if preds_p10 is not None and preds_p90 is not None:
        plt.fill_between(
            future_x,
            preds_p10,
            preds_p90,
            alpha=0.25,
            color="tab:orange",
            label="P10-P90 Confidence Interval",
        )
        
    # 如果提供了相似日结果，将其负荷曲线也绘制在预测图上进行参考对比
    if similar_day_result is not None:
        similar_curves = np.asarray(getattr(similar_day_result, "load_curves", []), dtype=np.float32)
        similar_times = list(getattr(similar_day_result, "historical_timestamps", []))
        similar_scores = list(getattr(similar_day_result, "similarity_scores", []))
        if similar_curves.ndim == 1:
            similar_curves = similar_curves.reshape(1, -1)
        if similar_curves.ndim == 2 and similar_curves.size > 0:
            similar_colors = ["tab:red", "tab:green", "tab:purple", "tab:brown", "tab:pink"]
            for idx, curve in enumerate(similar_curves):
                curve_steps = min(len(curve), predict_steps)
                source_time = (
                    similar_times[idx]
                    if idx < len(similar_times)
                    else f"Historical Match #{idx + 1}"
                )
                score_text = ""
                if idx < len(similar_scores):
                    score_text = f" | sim={float(similar_scores[idx]):.4f}"
                plt.plot(
                    range(n_history, n_history + curve_steps),
                    curve[:curve_steps],
                    label=f"Similar Day Top {idx + 1} | {source_time}{score_text}",
                    color=similar_colors[idx % len(similar_colors)],
                    linewidth=1.8,
                    linestyle="--",
                    alpha=0.9,
                )
    
    # 绘制预测起点分割线
    plt.axvline(x=n_history - 0.5, color="gray", linestyle="--", alpha=0.6, label="Prediction Start")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.title(f"{model_label} Future {predict_steps}-Step Load Prediction")
    plt.xlabel("Time Step (15min)")
    plt.ylabel(y_label)
    plt.tight_layout()

    # 保存并展示预测图
    out_fig = os.path.join(results_dir, "future_load_prediction.png")
    plt.savefig(out_fig, dpi=600, bbox_inches="tight")
    plt.show()

    print(f"Saved future prediction csv: {out_csv}")
    print(f"Saved future prediction figure: {out_fig}")


def plot_similar_day_curves(
    results_dir: str,
    retrieval_result: Any,
    out_name: str = "similar_day_retrieval.png",
    csv_name: str = "similar_day_retrieval.csv",
    json_name: str = "similar_day_retrieval.json",
    title_prefix: str = "相似日负荷检索",
    y_label: str = "电负荷 (MW)",
    freq: str = "15min",
) -> None:
    """
    将相似日检索返回的多条历史负荷曲线绘制在一张图中，并同步导出 CSV/JSON。

    参数:
        results_dir: 图表与导出文件保存目录
        retrieval_result: `SimilarDayRetriever` 返回的 RetrievalResult 对象或兼容对象
        out_name: 相似日曲线图文件名
        csv_name: 宽表格式曲线 CSV 文件名
        json_name: 检索元信息 JSON 文件名
        title_prefix: 图标题前缀
        y_label: Y 轴标签
        freq: 预测时间分辨率，默认 15 分钟
    """
    # 尝试切换 matplotlib 后端，防止在无显示设备的服务器上报错
    try:
        plt.switch_backend("TkAgg")
    except Exception:
        pass

    # 提取负荷曲线数据并转换为 numpy 数组
    load_curves = np.asarray(getattr(retrieval_result, "load_curves", []), dtype=np.float32)
    if load_curves.size == 0:
        print("未找到相似日检索曲线，跳过绘图。")
        return
        
    # 如果只有一条曲线，将其调整为二维数组
    if load_curves.ndim == 1:
        load_curves = load_curves.reshape(1, -1)
    if load_curves.ndim != 2:
        raise ValueError(f"相似日 load_curves 必须是二维数组，实际形状为 {load_curves.shape}")

    # 获取检索结果的元数据（查询时间、历史相似时间、相似度得分）
    query_timestamp = str(getattr(retrieval_result, "query_timestamp", ""))
    historical_timestamps = list(getattr(retrieval_result, "historical_timestamps", []))
    similarity_scores = list(getattr(retrieval_result, "similarity_scores", []))
    n_curves, n_steps = load_curves.shape

    # 生成预测步数和时间轴
    step_axis = np.arange(n_steps, dtype=np.int32)
    forecast_time = None
    x_axis = step_axis
    x_label = f"预测步数 ({freq})"
    
    # 如果查询时间非空，则根据频率生成对应的时间序列作为 X 轴
    try:
        if query_timestamp:
            forecast_time = pd.date_range(start=pd.Timestamp(query_timestamp), periods=n_steps, freq=freq)
            x_axis = forecast_time
            x_label = "预测时间"
    except Exception:
        forecast_time = None

    # 创建输出目录并生成文件路径
    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, csv_name)
    out_json = os.path.join(results_dir, json_name)
    out_fig = os.path.join(results_dir, out_name)

    # 构造待导出为 CSV 的数据字典
    csv_payload = {"step": step_axis}
    if forecast_time is not None:
        csv_payload["forecast_time"] = forecast_time
    for idx in range(n_curves):
        csv_payload[f"top_{idx + 1}_load"] = load_curves[idx]
        
    # 保存宽表数据到带有 UTF-8 BOM 签名的 CSV 文件中（防止在 Excel 中出现中文乱码）
    pd.DataFrame(csv_payload).to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 处理并保存检索元信息到 JSON 文件
    if hasattr(retrieval_result, "to_dict"):
        metadata = retrieval_result.to_dict()
    else:
        metadata = {
            "query_timestamp": query_timestamp,
            "historical_timestamps": historical_timestamps,
            "similarity_scores": similarity_scores,
            "load_curves": load_curves.tolist(),
        }
    metadata["curve_columns"] = [f"top_{idx + 1}_load" for idx in range(n_curves)]
    metadata["freq"] = freq
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # 初始化图表
    fig, ax = plt.subplots(1, 1, figsize=(15, 6), facecolor="white")
    # 定义绘图颜色循环，便于区分不同排名的相似日
    color_cycle = ["tab:red", "tab:orange", "tab:green", "tab:purple", "tab:brown"]
    
    # 逐条绘制相似日负荷曲线
    for idx in range(n_curves):
        curve = load_curves[idx]
        source_time = (
            historical_timestamps[idx]
            if idx < len(historical_timestamps)
            else f"历史匹配 #{idx + 1}"
        )
        score_text = ""
        if idx < len(similarity_scores):
            score_text = f" | 相似度={float(similarity_scores[idx]):.4f}"
            
        # 绘制折线并配置图例
        ax.plot(
            x_axis,
            curve,
            linewidth=2,
            color=color_cycle[idx % len(color_cycle)],
            label=f"排名 {idx + 1} | {source_time}{score_text}",
        )

    # 设置图表标题和坐标轴标签
    title = title_prefix
    if query_timestamp:
        title = f"{title_prefix}\n查询起点: {query_timestamp}"
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # 添加网格线辅助读取
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    
    # 如果 X 轴是时间对象，则格式化倾斜显示刻度标签以防重叠
    if forecast_time is not None:
        fig.autofmt_xdate()
        
    # 调整布局、保存并展示图表
    plt.tight_layout()
    plt.savefig(out_fig, dpi=600, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # 在控制台输出文件保存路径的提示信息
    print(f"已保存相似日检索数据 (CSV): {out_csv}")
    print(f"已保存相似日检索元信息 (JSON): {out_json}")
    print(f"已保存相似日检索图表 (Figure): {out_fig}")


def predict_future_load_from_csv(
    model,
    args,
    device,
    weather_store: Optional[Any],
    results_dir: str,
    future_path: str,
    steps: int,
    use_inverse: bool = True,
    quantiles: Optional[Sequence[float]] = None,
    data_provider_fn: Optional[Callable[..., Any]] = None,
    model_label: str = "Forecast",
    y_label: str = "Load (MW)",
    similar_day_result: Optional[Any] = None,
) -> None:
    """Unified future-forecast utility for both plain TimeXer and weather-enhanced variants."""
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

    ref_args = copy.copy(args)
    if hasattr(ref_args, "use_similar_day_prior"):
        ref_args.use_similar_day_prior = False
    if weather_store is None:
        ref_data, _ = data_provider_fn(ref_args, "train")
    else:
        ref_data, _ = data_provider_fn(ref_args, "train", weather_store)

    hist_dates = pd.to_datetime(history_df["date"].iloc[-args.seq_len :].values)
    future_dates = pd.to_datetime(future_df["date"].iloc[: args.pred_len].values)
    hist_load = history_df[args.target].iloc[-args.seq_len :].values.astype(np.float32).reshape(-1, 1)
    hist_load_scaled = _scale_target_generic(ref_data, hist_load)

    weather_dates = None
    hist_weather = None
    weather_mark_freq = None
    if weather_store is not None:
        weather_freq = getattr(args, "weather_step_freq", None)
        if weather_freq is None and getattr(weather_store, "native_freq", None) is not None:
            weather_freq = weather_store.native_freq
        weather_seq_len = int(getattr(args, "weather_seq_len", args.seq_len + args.pred_len))
        weather_history_len = int(getattr(args, "weather_history_len", args.seq_len))
        weather_mark_freq = (
            getattr(args, "weather_mark_freq", None)
            or getattr(args, "weather_step_freq", None)
            or args.freq
        )
        reference_weather_timestamps_ns = None
        if hasattr(weather_store, "sources"):
            candidate_timestamps = [
                np.asarray(source.get("timestamps_ns"), dtype=np.int64)
                for source in getattr(weather_store, "sources", [])
                if source.get("timestamps_ns") is not None
            ]
            if candidate_timestamps:
                reference_weather_timestamps_ns = np.concatenate(candidate_timestamps, axis=0)

        weather_dates = build_weather_sequence_timestamps(
            target_start=pd.Timestamp(future_dates[0]),
            weather_seq_len=weather_seq_len,
            weather_history_len=weather_history_len,
            weather_freq=weather_freq,
            reference_timestamps_ns=reference_weather_timestamps_ns,
        )
        hist_weather = weather_store.fetch_frames_by_dates(weather_dates)

    similar_day_prior_tensor = None
    if bool(getattr(model, "use_similar_day_prior", False)) and similar_day_result is not None:
        similar_day_prior_tensor = _build_future_similar_day_prior_tensor(
            similar_day_result=similar_day_result,
            ref_data=ref_data,
            pred_len=args.pred_len,
            top_k=int(getattr(model, "similar_day_top_k", getattr(args, "similar_day_top_k", 3))),
            device=device,
        )

    x_mark_np = time_features(pd.to_datetime(hist_dates), freq=args.freq).transpose(1, 0).astype(np.float32)
    exo_mark_np = None
    if weather_dates is not None:
        exo_mark_np = time_features(
            pd.to_datetime(weather_dates.values),
            freq=weather_mark_freq,
        ).transpose(1, 0).astype(np.float32)

    model.eval()
    with torch.inference_mode():
        batch_x = torch.as_tensor(hist_load_scaled, dtype=torch.float32, device=device).unsqueeze(0)
        batch_x_mark = torch.as_tensor(x_mark_np, dtype=torch.float32, device=device).unsqueeze(0)

        if weather_store is None:
            dec_len = args.label_len + args.pred_len
            dec_inp = torch.zeros((1, dec_len, batch_x.shape[-1]), dtype=torch.float32, device=device)
            batch_y_mark = torch.zeros((1, dec_len, batch_x_mark.shape[-1]), dtype=torch.float32, device=device)
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            batch_exo_mark = torch.as_tensor(exo_mark_np, dtype=torch.float32, device=device).unsqueeze(0)
            batch_weather_x = torch.as_tensor(hist_weather, dtype=torch.float32, device=device).unsqueeze(0)
            model_kwargs = {
                "load_x": batch_x,
                "x_mark_enc": batch_x_mark,
                "x_exo_mark": batch_exo_mark,
                "weather_x": batch_weather_x,
            }
            forward_params = inspect.signature(model.forward).parameters
            if (
                similar_day_prior_tensor is not None
                and "similar_day_prior" in forward_params
            ):
                model_kwargs["similar_day_prior"] = similar_day_prior_tensor
            outputs = model(**model_kwargs)

    quantile_scaled = outputs[0, : args.pred_len, :].detach().cpu().numpy()
    p50_scaled = quantile_scaled[:, p50_idx]

    if use_inverse:
        preds_p50 = _inverse_transform_target_generic(ref_data, p50_scaled.reshape(-1, 1)).reshape(-1)
        history_target = history_df[args.target].values
        preds_p10 = (
            _inverse_transform_target_generic(ref_data, quantile_scaled[:, p10_idx].reshape(-1, 1)).reshape(-1)
            if p10_idx is not None
            else None
        )
        preds_p90 = (
            _inverse_transform_target_generic(ref_data, quantile_scaled[:, p90_idx].reshape(-1, 1)).reshape(-1)
            if p90_idx is not None
            else None
        )
    else:
        preds_p50 = p50_scaled
        history_target = hist_load_scaled.reshape(-1)
        preds_p10 = quantile_scaled[:, p10_idx] if p10_idx is not None else None
        preds_p90 = quantile_scaled[:, p90_idx] if p90_idx is not None else None

    effective_future_dates = pd.Series(future_dates[:predict_steps])
    preds_p50 = preds_p50[:predict_steps]
    if preds_p10 is not None:
        preds_p10 = preds_p10[:predict_steps]
    if preds_p90 is not None:
        preds_p90 = preds_p90[:predict_steps]

    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "future_load_prediction.csv")
    output_payload = {
        "date": effective_future_dates,
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
                f"  {effective_future_dates.iloc[i]}: "
                f"{preds_p10[i]:<12.4f} {preds_p50[i]:<12.4f} {preds_p90[i]:<12.4f}"
            )
    else:
        print(f"{'Time':<25} {'P50':<12}")
        print("-" * 40)
        for i in range(predict_steps):
            print(f"  {effective_future_dates.iloc[i]}: {preds_p50[i]:<12.4f}")

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

    if similar_day_result is not None:
        similar_curves = np.asarray(getattr(similar_day_result, "load_curves", []), dtype=np.float32)
        similar_times = list(getattr(similar_day_result, "historical_timestamps", []))
        similar_scores = list(getattr(similar_day_result, "similarity_scores", []))
        if similar_curves.ndim == 1:
            similar_curves = similar_curves.reshape(1, -1)
        if similar_curves.ndim == 2 and similar_curves.size > 0:
            similar_colors = ["tab:red", "tab:green", "tab:purple", "tab:brown", "tab:pink"]
            for idx, curve in enumerate(similar_curves):
                curve_steps = min(len(curve), predict_steps)
                source_time = (
                    similar_times[idx]
                    if idx < len(similar_times)
                    else f"Historical Match #{idx + 1}"
                )
                score_text = ""
                if idx < len(similar_scores):
                    score_text = f" | sim={float(similar_scores[idx]):.4f}"
                plt.plot(
                    range(n_history, n_history + curve_steps),
                    curve[:curve_steps],
                    label=f"Similar Day Top {idx + 1} | {source_time}{score_text}",
                    color=similar_colors[idx % len(similar_colors)],
                    linewidth=1.8,
                    linestyle="--",
                    alpha=0.9,
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
    "plot_similar_day_curves",
    "predict_future_load_from_csv",
    "restore_sliding_window_2d",
    "restore_sliding_window_3d",
]
