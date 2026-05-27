from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "湖南省电力负荷2024.csv"

TRAIN_LEN = 23424
VAL_LEN = 5856
SEQ_LEN = 672
PRED_LEN = 96
BORDER1_TEST = TRAIN_LEN + VAL_LEN - SEQ_LEN
FIRST_PRED_ROW = BORDER1_TEST + SEQ_LEN

TARGET_DATES = pd.DatetimeIndex([pd.Timestamp("2024-11-24")])
QUANTILES = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
Q10_INDEX = QUANTILES.index(0.1)
Q90_INDEX = QUANTILES.index(0.9)
CM_TO_INCH = 1.0 / 2.54
FIG_WIDTH_CM = 14.0
FIG_HEIGHT_CM = 12.0
JOURNAL_LINE_WIDTH_PT = 0.5
JOURNAL_FONT_SIZE_PT = 6.0

FULL_DIR = ROOT / "results" / "sdv4_sl672_pl96_wd2_sdk3_ts1_bs64_exp000_e6704ca5"


MODEL_SPECS = OrderedDict(
    [
        (
            "ARIMA",
            {
                "path": ROOT / "baseline_exp" / "arima" / "pred_inv.npy",
                "color": "#8C510A",
                "linestyle": (0, (1.2, 1.0)),
            },
        ),
        (
            "LSTM",
            {
                "path": ROOT / "baseline_exp" / "lstm" / "pred_inv.npy",
                "quantile_path": ROOT / "baseline_exp" / "lstm" / "quantile_preds_inv.npy",
                "color": "#7209B7",
                "linestyle": (0, (4.8, 1.2)),
            },
        ),
        (
            "Informer",
            {
                "path": ROOT / "baseline_exp" / "informer" / "pred_inv.npy",
                "quantile_path": ROOT / "baseline_exp" / "informer" / "quantile_preds_inv.npy",
                "color": "#4B5563",
                "linestyle": (0, (2.4, 1.2)),
            },
        ),
        (
            "PatchTST",
            {
                "path": ROOT / "baseline_exp" / "patchtst" / "pred_inv.npy",
                "quantile_path": ROOT / "baseline_exp" / "patchtst" / "quantile_preds_inv.npy",
                "color": "#C51B7D",
                "linestyle": (0, (5.0, 1.0, 1.0, 1.0)),
            },
        ),
        (
            "iTransformer",
            {
                "path": ROOT / "baseline_exp" / "itransformer" / "pred_inv.npy",
                "quantile_path": ROOT / "baseline_exp" / "itransformer" / "quantile_preds_inv.npy",
                "color": "#009E73",
                "linestyle": (0, (3.6, 1.0, 1.0, 1.0)),
            },
        ),
        (
            "TimeXer*",
            {
                "path": ROOT
                / "results"
                / "test1_HunanLoad_2024_672_sl672_pl96_dm512_bs32_itr1_f59463f003"
                / "pred_inv.npy",
                "quantile_path": ROOT
                / "results"
                / "test1_HunanLoad_2024_672_sl672_pl96_dm512_bs32_itr1_f59463f003"
                / "quantile_preds_inv.npy",
                "color": "#023E8A",
                "linestyle": (0, (3.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
            },
        ),
        (
            "+MeteoConv",
            {
                "path": ROOT
                / "results"
                / "TimeXerE2E_sl672_pl96_wd3_wsl672_wh672_wk62x61_bs32_Exp_0_99626da6"
                / "pred_inv.npy",
                "quantile_path": ROOT
                / "results"
                / "TimeXerE2E_sl672_pl96_wd3_wsl672_wh672_wk62x61_bs32_Exp_0_99626da6"
                / "quantile_preds_inv.npy",
                "color": "#2166AC",
                "linestyle": (0, (6.0, 1.2)),
            },
        ),
        (
            "+Optuna",
            {
                "path": ROOT / "optuna_15min_7_1" / "results" / "TimeXerE2E_sl672_pl96_wd2_wsl768_wh672_wk62x61_bs64_Optuna_trial032_0_437d0f80" / "pred_inv.npy",
                "quantile_path": ROOT
                / "optuna_15min_7_1"
                / "results"
                / "TimeXerE2E_sl672_pl96_wd2_wsl768_wh672_wk62x61_bs64_Optuna_trial032_0_437d0f80"
                / "quantile_preds_inv.npy",
                "color": "#B35806",
                "linestyle": (0, (2.0, 1.0, 5.0, 1.0)),
            },
        ),
        (
            "SDR",
            {
                "path": ROOT
                / "results"
                / "test5_similar_only_similar_day_retriever_hunan_grid_2024_2025_filtered_15min_top3"
                / "pred.npy",
                "color": "#A6611A",
                "linestyle": (0, (1.0, 1.2, 4.0, 1.2)),
            },
        ),
        (
            "Full (ours)",
            {
                "path": FULL_DIR / "pred_inv.npy",
                "quantile_path": FULL_DIR / "quantile_preds_inv.npy",
                "color": "#E63946",
                "linestyle": "-",
            },
        ),
    ]
)

LEGEND_ORDER = [
    "实际值",
    "Informer",
    "SDR",
    "本文方法",
    "LSTM",
    "TimeXer*",
    "iTransformer",
    "ARIMA",
    "+MeteoConv",
    "PatchTST",
    "P10-P90预测区间",
    "+Optuna",
]

DISPLAY_LABELS = {
    "Ground Truth": "实际值",
    "Full (ours)": "本文方法",
    "P10-P90 CI": "P10-P90预测区间",
}


def load_point_prediction(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)

    arr = np.load(path)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"{path} must have shape (windows, 96) or (windows, 96, 1), got {arr.shape}")
    if arr.shape[1] != PRED_LEN:
        raise ValueError(f"{path} prediction length must be {PRED_LEN}, got {arr.shape}")
    return arr.astype(np.float64, copy=False)


def load_quantiles(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)

    arr = np.load(path)
    expected = (PRED_LEN, len(QUANTILES))
    if arr.ndim != 3 or arr.shape[1:] != expected:
        raise ValueError(f"{path} must have shape (windows, {PRED_LEN}, {len(QUANTILES)}), got {arr.shape}")
    return arr.astype(np.float64, copy=False)


def read_dates() -> pd.Series:
    df = pd.read_csv(DATA_PATH, usecols=["date"])
    dates = pd.to_datetime(df["date"])
    expected_test_start = pd.Timestamp("2024-11-01 00:15:00")
    actual_test_start = dates.iloc[FIRST_PRED_ROW]
    if actual_test_start != expected_test_start:
        raise ValueError(
            f"Test first prediction row {FIRST_PRED_ROW} is {actual_test_start}, "
            f"expected {expected_test_start}"
        )
    return dates


def window_index_for_date(dates: pd.Series, day: pd.Timestamp) -> int:
    target = day + pd.Timedelta(minutes=15)
    matches = np.flatnonzero(dates.to_numpy() == target.to_datetime64())
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one CSV row for {target}, found {len(matches)}")

    row_index = int(matches[0])
    window_index = row_index - FIRST_PRED_ROW
    if window_index < 0:
        raise ValueError(f"{target} is before the first prediction row")
    return window_index


def daily_metrics(pred: np.ndarray, true: np.ndarray) -> tuple[float, float, float, float]:
    eps = 1e-8
    err = pred - true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(true), eps)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > eps else float("nan")
    return mae, rmse, mape, r2


def daily_probabilistic_metrics(quantile_pred: np.ndarray, true: np.ndarray) -> tuple[float, float, float]:
    true_2d = true[:, None]
    errors = true_2d - quantile_pred
    losses = np.maximum(np.array(QUANTILES) * errors, (np.array(QUANTILES) - 1.0) * errors)
    mean_pinball = float(np.mean(losses))

    q10 = quantile_pred[:, Q10_INDEX]
    q90 = quantile_pred[:, Q90_INDEX]
    picp80 = float(np.mean((true >= q10) & (true <= q90)))

    true_range = float(np.max(true) - np.min(true))
    pinaw80 = float(np.mean(q90 - q10) / true_range) if true_range > 1e-8 else float("nan")
    return mean_pinball, picp80, pinaw80


def print_metrics(
    window_indices: dict[pd.Timestamp, int],
    predictions: dict[str, np.ndarray],
    true: np.ndarray,
    quantile_predictions: dict[str, np.ndarray],
) -> None:
    print(f"Verified first prediction row: {FIRST_PRED_ROW}")
    print(f"Verified first prediction timestamp: 2024-11-01 00:15:00")
    print()

    for day in TARGET_DATES:
        idx = window_indices[day]
        print(f"=== {day:%Y-%m-%d} ===")
        print(f"Window index: {idx}")
        print(
            f"{'Model':<18}{'MAE':>10}{'RMSE':>10}{'MAPE':>10}{'R2':>10}"
            f"{'MeanPinball':>14}{'PICP80':>12}{'PINAW80':>10}"
        )
        for name in MODEL_SPECS:
            mae, rmse, mape, r2 = daily_metrics(predictions[name][idx], true[idx])
            if name in quantile_predictions:
                mean_pinball, picp80, pinaw80 = daily_probabilistic_metrics(quantile_predictions[name][idx], true[idx])
                print(
                    f"{name:<18}{mae:>10.4f}{rmse:>10.4f}{mape:>10.4f}{r2:>10.4f}"
                    f"{mean_pinball:>14.4f}{picp80:>12.4f}{pinaw80:>10.4f}"
                )
            else:
                print(
                    f"{name:<18}{mae:>10.4f}{rmse:>10.4f}{mape:>10.4f}{r2:>10.4f}"
                    f"{'-':>14}{'-':>12}{'-':>10}"
                )
        print()


def display_label(name: str) -> str:
    return DISPLAY_LABELS.get(name, name)


def configure_axes(ax: plt.Axes, ylabel: str) -> None:
    x_ticks = np.arange(0, PRED_LEN + 1, 12)
    x_tick_positions = np.minimum(x_ticks, PRED_LEN - 1)
    x_tick_labels = [f"{hour:02d}:00" for hour in range(0, 25, 3)]

    ax.set_xlim(0, PRED_LEN - 1)
    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels(x_tick_labels)
    tick_labels = ax.get_xticklabels()
    if tick_labels:
        tick_labels[0].set_ha("left")
        tick_labels[-1].set_ha("right")
    ax.set_xlabel("时刻", fontsize=JOURNAL_FONT_SIZE_PT)
    ax.set_ylabel(ylabel, fontsize=JOURNAL_FONT_SIZE_PT)
    ax.tick_params(axis="both", labelsize=5.5, length=2.0, width=0.5, pad=1.5)
    ax.grid(True, linestyle=":", linewidth=0.25, alpha=0.45)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


def build_legend_handles() -> list:
    handles = {
        "实际值": Line2D(
            [0],
            [0],
            color="#000000",
            linestyle="-",
            linewidth=JOURNAL_LINE_WIDTH_PT,
            label="实际值",
        ),
        "P10-P90预测区间": Patch(facecolor="#E63946", alpha=0.13, edgecolor="none", label="P10-P90预测区间"),
    }
    for name, spec in MODEL_SPECS.items():
        label = display_label(name)
        handles[label] = Line2D(
            [0],
            [0],
            color=spec["color"],
            linestyle=spec["linestyle"],
            linewidth=JOURNAL_LINE_WIDTH_PT,
            label=label,
        )
    return [handles[name] for name in LEGEND_ORDER]


def save_figure(fig: plt.Figure, path: Path, **kwargs) -> Path:
    try:
        fig.savefig(path, **kwargs)
        return path
    except PermissionError:
        fallback = path.with_name(f"{path.stem}_updated{path.suffix}")
        fig.savefig(fallback, **kwargs)
        print(f"Warning: {path.name} is locked; saved {fallback.name} instead.")
        return fallback


def plot_case_study(
    window_indices: dict[pd.Timestamp, int],
    predictions: dict[str, np.ndarray],
    true: np.ndarray,
    quantile_predictions: dict[str, np.ndarray],
) -> list[Path]:
    plt.rcParams["font.family"] = ["SimSun", "Times New Roman"]
    plt.rcParams["font.size"] = JOURNAL_FONT_SIZE_PT
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(FIG_WIDTH_CM * CM_TO_INCH, FIG_HEIGHT_CM * CM_TO_INCH),
        dpi=600,
        sharex=True,
    )
    x = np.arange(PRED_LEN)

    prediction_plot_order = [
        "Full (ours)",
        "iTransformer",
        "PatchTST",
        "LSTM",
        "Informer",
        "ARIMA",
        "+Optuna",
        "+MeteoConv",
        "TimeXer*",
        "SDR",
    ]

    day = TARGET_DATES[0]
    idx = window_indices[day]

    ax_pred = axes[0]
    ax_err = axes[1]

    full_quantiles = quantile_predictions["Full (ours)"]
    q10 = full_quantiles[idx, :, Q10_INDEX]
    q90 = full_quantiles[idx, :, Q90_INDEX]
    ax_pred.fill_between(x, q10, q90, color="#E63946", alpha=0.13, linewidth=0)
    ax_pred.plot(
        x,
        true[idx],
        color="#000000",
        linestyle="-",
        linewidth=JOURNAL_LINE_WIDTH_PT,
        label="实际值",
        zorder=4,
    )

    for name in prediction_plot_order:
        spec = MODEL_SPECS[name]
        ax_pred.plot(
            x,
            predictions[name][idx],
            color=spec["color"],
            linestyle=spec["linestyle"],
            linewidth=JOURNAL_LINE_WIDTH_PT,
            label=display_label(name),
        )

    for name, spec in MODEL_SPECS.items():
        ax_err.plot(
            x,
            np.abs(predictions[name][idx] - true[idx]),
            color=spec["color"],
            linestyle=spec["linestyle"],
            linewidth=JOURNAL_LINE_WIDTH_PT,
            label=display_label(name),
        )

    for ax, ylabel in ((ax_pred, "归一化负荷"), (ax_err, "归一化绝对误差")):
        configure_axes(ax, ylabel)
        ax.text(
            0.015,
            0.93,
            f"2024年11月{day.day}日",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=JOURNAL_FONT_SIZE_PT,
        )
    ax_pred.tick_params(axis="x", labelbottom=True)

    ax_pred.set_title("（a）负荷预测曲线", fontsize=JOURNAL_FONT_SIZE_PT, pad=2.0)
    ax_err.set_title("（b）绝对误差曲线", fontsize=JOURNAL_FONT_SIZE_PT, pad=2.0)

    legend = fig.legend(
        handles=build_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=6,
        fontsize=5.2,
        frameon=True,
        facecolor="white",
        edgecolor="#BDBDBD",
        framealpha=1.0,
        handlelength=1.8,
        columnspacing=0.8,
        labelspacing=0.25,
        borderpad=0.25,
    )
    legend.set_zorder(10)

    fig.subplots_adjust(left=0.075, right=0.97, top=0.93, bottom=0.18, hspace=0.35)
    saved_paths = [
        save_figure(fig, ROOT / "case_study_cold_wave_nov2024.png", dpi=600),
        save_figure(fig, ROOT / "case_study_cold_wave_nov2024.pdf"),
        save_figure(fig, ROOT / "case_study_cold_wave_nov2024.svg"),
        # save_figure(fig, ROOT / "case_study_cold_wave_nov2024.eps"),
    ]
    plt.close(fig)
    return saved_paths


def main() -> None:
    dates = read_dates()
    window_indices = {day: window_index_for_date(dates, day) for day in TARGET_DATES}

    predictions = {name: load_point_prediction(spec["path"]) for name, spec in MODEL_SPECS.items()}
    true = load_point_prediction(FULL_DIR / "true_inv.npy")
    quantile_predictions = {
        name: load_quantiles(spec["quantile_path"])
        for name, spec in MODEL_SPECS.items()
        if "quantile_path" in spec and spec["quantile_path"].exists()
    }

    expected_windows = true.shape[0]
    for name, pred in predictions.items():
        if pred.shape != true.shape:
            raise ValueError(f"{name} shape {pred.shape} does not match true shape {true.shape}")
    for name, quantiles in quantile_predictions.items():
        if quantiles.shape[0] != expected_windows:
            raise ValueError(
                f"{name} quantile window count {quantiles.shape[0]} does not match true window count {expected_windows}"
            )

    print_metrics(window_indices, predictions, true, quantile_predictions)
    saved_paths = plot_case_study(window_indices, predictions, true, quantile_predictions)
    for path in saved_paths:
        print(f"Saved: {path.name}")


if __name__ == "__main__":
    main()
