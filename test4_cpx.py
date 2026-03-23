"""
test4.py - ConvNeXt-Tiny + TimeXer 端到端联合训练版本

核心改动：
1. 不再离线缓存 ConvNeXt-Tiny 气象特征。
2. DataLoader 直接返回历史/未来气象网格，模型内部完成 ConvNeXt-Tiny 前向。
3. ConvNeXt-Tiny、TimeXer、Quantile Head 统一组成一个模型并统一保存/加载权重。
4. 历史气象特征进入 TimeXer 编码器输入，未来气象特征进入 x_fut_known。

注意：
- 这是端到端版本，计算量显著大于离线特征版本。
- 为了避免一次性把 B*seq_len 帧全部送入 ConvNeXt，模型内部使用 chunk 方式编码天气序列。
- HDF5 读取建议保持 num_workers=0。
"""

import argparse
import hashlib
import os
import random
import re
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.utils.data import DataLoader, Dataset

from models.TimeXer import Model as TimeXer
from utils.metrics import cal_eval, metric
from utils.timefeatures import time_features
from utils.tools import EarlyStopping, adjust_learning_rate

warnings.filterwarnings("ignore")

try:
    import h5py
except ImportError:
    h5py = None

try:
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
except Exception:
    ConvNeXt_Tiny_Weights = None
    convnext_tiny = None


# ==================== 分位数配置 ====================
QUANTILES = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
N_QUANTILES = len(QUANTILES)
P50_IDX = QUANTILES.index(0.5)
P10_IDX = QUANTILES.index(0.1)
P90_IDX = QUANTILES.index(0.9)


# ==================== 基础任务配置 ====================
TASK_NAME = "long_term_forecast"
MODEL = "TimeXer"
MODEL_ID = "HunanLoad_672_96_ConvNeXtTiny_E2E"


# ==================== 数据配置 ====================
ROOT_PATH = "./data/"
DATA_PATH = "湖南省电力负荷_processed（25.10.25-26.2.26）.csv"
FUTURE_PATH = "./data/湖南省电力负荷_future（25.10.25-26.2.26）.csv"
TARGET = "load"
FEATURES = "MS"


# ==================== 时序长度配置 ====================
SEQ_LEN = 96 * 7
LABEL_LEN = 0
PRED_LEN = 96


# ==================== ConvNeXt-Tiny 气象配置 ====================
WEATHER_H5_SPECS: List[Tuple[str, str]] = [
    ("./data/hunan_grid_meteo_20250101_20260228.h5", "2025-01-01 00:00:00"),
]
WEATHER_IN_CHANNELS = 10
WEATHER_FEATURE_STAGE = 1
WEATHER_STAGE_TO_DIM = {1: 48, 2: 96, 3: 192, 4: 224}
WEATHER_BACKBONE_OUT_DIM = WEATHER_STAGE_TO_DIM[WEATHER_FEATURE_STAGE]
WEATHER_FEATURE_DIM = WEATHER_BACKBONE_OUT_DIM
WEATHER_PRETRAINED = True
WEATHER_TRAIN_BACKBONE = True
WEATHER_ENCODE_CHUNK_SIZE = 16
WEATHER_FILL_VALUE = 0.0


# ==================== TimeXer 模型配置 ====================
ENC_IN = WEATHER_FEATURE_DIM + 1
C_OUT = 1
D_MODEL = 512
N_HEADS = 4
E_LAYERS = 3
D_FF = 2048
FACTOR = 3
DROPOUT = 0.1
ACTIVATION = "gelu"
PATCH_LEN = 96
USE_NORM = 1


# ==================== 训练配置 ====================
TRAIN_EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
PATIENCE = 5
NUM_WORKERS = 0


# ==================== 硬件配置 ====================
USE_GPU = True
GPU = 0


# ==================== 运行配置 ====================
DES = "Exp"
ITR = 1
INVERSE_EVAL = True
TRAIN_MODE = True


# ==================== 未来协变量配置 ====================
USE_FUTURE_COVARIATES = False
FUTURE_COV_DIM = WEATHER_FEATURE_DIM
FUTURE_COV_DROPOUT = 0.1


def _require_weather_runtime() -> None:
    if h5py is None:
        raise ImportError("缺少 h5py，test4.py 需要 h5py 读取 HDF5 气象文件。")
    if convnext_tiny is None:
        raise ImportError("缺少 torchvision，test4.py 需要 torchvision.models.convnext_tiny。")


def _guess_year_start_from_path(file_path: str) -> pd.Timestamp:
    name = Path(file_path).stem
    match = re.search(r"(20\\d{2})", name)
    if match:
        return pd.Timestamp(f"{match.group(1)}-01-01 00:00:00")
    return pd.Timestamp("2025-01-01 00:00:00")


def _find_first_4d_dataset(h5_obj):
    for key in h5_obj.keys():
        item = h5_obj[key]
        if isinstance(item, h5py.Dataset) and item.ndim == 4:
            return item
        if isinstance(item, h5py.Group):
            found = _find_first_4d_dataset(item)
            if found is not None:
                return found
    return None


def _infer_weather_freq(n_steps: int, start_time: pd.Timestamp) -> pd.Timedelta:
    days_in_year = 366 if start_time.is_leap_year else 365
    steps_per_day = n_steps / float(days_in_year)
    candidates = np.array([24.0, 48.0, 96.0])
    best = candidates[np.argmin(np.abs(candidates - steps_per_day))]
    if not np.isfinite(best) or best <= 0:
        raise ValueError(f"无法从 n_steps={n_steps} 推断气象时间频率。")
    minutes = int(round(1440.0 / best))
    return pd.Timedelta(minutes=minutes)


def _adapt_first_conv_to_multichannel(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    if conv.in_channels == in_channels:
        return conv

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )

    with torch.no_grad():
        old_weight = conv.weight.data
        mean_weight = old_weight.mean(dim=1, keepdim=True)
        expanded = mean_weight.repeat(1, in_channels, 1, 1)
        expanded *= old_weight.shape[1] / float(in_channels)
        new_conv.weight.copy_(expanded)
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias.data)

    return new_conv


class ConvNeXtTinyWeatherExtractor(nn.Module):
    """ConvNeXt-Tiny 主干，可用于端到端微调。"""

    STAGE_TO_LAYER = {1: 1, 2: 3, 3: 5, 4: 7}
    STAGE_TO_DIM = {1: 96, 2: 192, 3: 384, 4: 768}

    def __init__(
        self,
        in_channels: int = WEATHER_IN_CHANNELS,
        feature_stage: int = WEATHER_FEATURE_STAGE,
        use_pretrained: bool = WEATHER_PRETRAINED,
        train_backbone: bool = WEATHER_TRAIN_BACKBONE,
    ):
        super().__init__()
        if feature_stage not in self.STAGE_TO_LAYER:
            raise ValueError(f"feature_stage 必须在 {sorted(self.STAGE_TO_LAYER)} 中，收到 {feature_stage}")

        weights = ConvNeXt_Tiny_Weights.DEFAULT if (use_pretrained and ConvNeXt_Tiny_Weights is not None) else None
        try:
            model = convnext_tiny(weights=weights)
        except Exception as exc:
            warnings.warn(f"ConvNeXt-Tiny 预训练权重加载失败，退回随机初始化：{exc}")
            model = convnext_tiny(weights=None)

        if not isinstance(model.features[0][0], nn.Conv2d):
            raise TypeError("未识别到 ConvNeXt-Tiny stem 卷积层，无法适配输入通道数。")

        model.features[0][0] = _adapt_first_conv_to_multichannel(model.features[0][0], in_channels)
        self.backbone = model.features
        self.stop_layer = self.STAGE_TO_LAYER[feature_stage]
        self.output_dim = self.STAGE_TO_DIM[feature_stage]
        self.train_backbone = bool(train_backbone)

        if not self.train_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"天气输入必须是 [B, C, H, W]，收到 {tuple(x.shape)}")

        x = x.float()
        spatial_mean = x.mean(dim=(-2, -1), keepdim=True)
        spatial_std = x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        x = (x - spatial_mean) / spatial_std

        if not self.train_backbone:
            self.backbone.eval()

        for layer_idx, layer in enumerate(self.backbone):
            x = layer(x)
            if layer_idx == self.stop_layer:
                break

        return x.mean(dim=(-2, -1))


class WeatherGridStore:
    """按时间戳对齐 HDF5 原始网格，并按需读取。"""

    def __init__(self, h5_specs: Sequence[Tuple[str, str]], fill_value: float = WEATHER_FILL_VALUE):
        _require_weather_runtime()
        self.h5_specs = [
            (
                os.path.abspath(path),
                pd.Timestamp(start) if start else _guess_year_start_from_path(path),
            )
            for path, start in h5_specs
        ]
        self.fill_value = float(fill_value)
        self.sources: List[Dict[str, object]] = []
        self.frame_shape: Optional[Tuple[int, int, int]] = None
        self._file_handles: Dict[int, Any] = {}
        self._datasets: Dict[int, Any] = {}
        self._warned_out_of_range = False
        self.prepare()

    def prepare(self) -> None:
        if self.sources:
            return

        for h5_path, start_time in self.h5_specs:
            if not os.path.exists(h5_path):
                print(f"[Weather] 文件不存在，跳过: {h5_path}")
                continue

            with h5py.File(h5_path, "r") as h5_file:
                dataset = _find_first_4d_dataset(h5_file)
                if dataset is None:
                    raise ValueError(f"在 {h5_path} 中未找到 4D dataset。")
                if dataset.shape[1] != WEATHER_IN_CHANNELS:
                    raise ValueError(
                        f"{h5_path} 的气象通道数不是 {WEATHER_IN_CHANNELS}，实际为 {dataset.shape[1]}"
                    )

                n_steps, n_channels, height, width = dataset.shape
                dataset_name = dataset.name

            freq = _infer_weather_freq(n_steps, start_time)
            timestamps = pd.date_range(start=start_time, periods=n_steps, freq=freq)
            source = {
                "path": h5_path,
                "dataset_name": dataset_name,
                "n_steps": n_steps,
                "timestamps_ns": timestamps.asi8.copy(),
                "start_ns": int(timestamps[0].value),
                "end_ns": int(timestamps[-1].value),
                "freq": freq,
            }
            self.sources.append(source)

            if self.frame_shape is None:
                self.frame_shape = (n_channels, height, width)
            elif self.frame_shape != (n_channels, height, width):
                raise ValueError(
                    f"多个 HDF5 的网格形状不一致：已有 {self.frame_shape}，新文件 {(n_channels, height, width)}"
                )

            print(f"[Weather] {Path(h5_path).name}: steps={n_steps}, freq={freq}")

        if not self.sources:
            raise FileNotFoundError("未找到任何可用的气象 HDF5 文件。请检查 WEATHER_H5_SPECS。")

        self.sources.sort(key=lambda x: x["start_ns"])
        start_ts = pd.Timestamp(min(source["start_ns"] for source in self.sources))
        end_ts = pd.Timestamp(max(source["end_ns"] for source in self.sources))
        print(f"[Weather] 时间范围: {start_ts} ~ {end_ts}")

    def _get_dataset(self, source_idx: int):
        if source_idx in self._datasets:
            return self._datasets[source_idx]

        source = self.sources[source_idx]
        h5_file = h5py.File(source["path"], "r")
        dataset = h5_file[source["dataset_name"]]
        self._file_handles[source_idx] = h5_file
        self._datasets[source_idx] = dataset
        return dataset

    def close(self) -> None:
        for file_handle in self._file_handles.values():
            try:
                file_handle.close()
            except Exception:
                pass
        self._file_handles.clear()
        self._datasets.clear()

    def __del__(self):
        self.close()

    def build_alignment(self, dates: Sequence[pd.Timestamp]) -> Dict[str, np.ndarray]:
        dates = pd.DatetimeIndex(pd.to_datetime(dates))
        request_ns = dates.asi8.astype(np.int64)
        n = len(request_ns)

        source_idx = np.full(n, -1, dtype=np.int32)
        left_idx = np.zeros(n, dtype=np.int32)
        right_idx = np.zeros(n, dtype=np.int32)
        alpha = np.zeros(n, dtype=np.float32)
        valid = np.zeros(n, dtype=bool)

        for idx, source in enumerate(self.sources):
            mask = (request_ns >= source["start_ns"]) & (request_ns <= source["end_ns"])
            if not mask.any():
                continue

            ts_ns = source["timestamps_ns"]
            req = request_ns[mask]
            pos = np.searchsorted(ts_ns, req, side="left")

            pos_clipped = np.clip(pos, 0, len(ts_ns) - 1)
            exact_mask = ts_ns[pos_clipped] == req

            current_indices = np.where(mask)[0]
            exact_indices = current_indices[exact_mask]
            source_idx[exact_indices] = idx
            left_idx[exact_indices] = pos_clipped[exact_mask]
            right_idx[exact_indices] = pos_clipped[exact_mask]
            alpha[exact_indices] = 0.0
            valid[exact_indices] = True

            non_exact_indices = current_indices[~exact_mask]
            if len(non_exact_indices) == 0:
                continue

            non_exact_pos = pos[~exact_mask]
            right = np.clip(non_exact_pos, 1, len(ts_ns) - 1)
            left = np.clip(right - 1, 0, len(ts_ns) - 1)
            left_ts = ts_ns[left].astype(np.float64)
            right_ts = ts_ns[right].astype(np.float64)
            req_ts = req[~exact_mask].astype(np.float64)
            denom = np.maximum(right_ts - left_ts, 1.0)
            alpha_values = ((req_ts - left_ts) / denom).astype(np.float32)

            source_idx[non_exact_indices] = idx
            left_idx[non_exact_indices] = left
            right_idx[non_exact_indices] = right
            alpha[non_exact_indices] = alpha_values
            valid[non_exact_indices] = True

        if (~valid).any() and not self._warned_out_of_range:
            print(
                f"[Weather] 警告: 有 {(~valid).sum()} 个时间点超出气象覆盖范围，"
                f"将使用 fill_value={self.fill_value} 填充。"
            )
            self._warned_out_of_range = True

        return {
            "source_idx": source_idx,
            "left_idx": left_idx,
            "right_idx": right_idx,
            "alpha": alpha,
            "valid": valid,
        }

    def fetch_frames_from_alignment(
        self,
        alignment: Dict[str, np.ndarray],
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> np.ndarray:
        if self.frame_shape is None:
            raise RuntimeError("frame_shape 尚未初始化。")

        sl = slice(start, end)
        source_idx = alignment["source_idx"][sl]
        left_idx = alignment["left_idx"][sl]
        right_idx = alignment["right_idx"][sl]
        alpha = alignment["alpha"][sl]
        valid = alignment["valid"][sl]

        n = len(source_idx)
        frames = np.full((n,) + self.frame_shape, self.fill_value, dtype=np.float32)
        if not valid.any():
            return frames

        for src in np.unique(source_idx[valid]):
            src_mask = valid & (source_idx == src)
            dataset = self._get_dataset(int(src))

            left_frames = np.asarray(dataset[left_idx[src_mask]], dtype=np.float32)
            alpha_src = alpha[src_mask]

            if np.allclose(alpha_src, 0.0):
                frames[src_mask] = left_frames
                continue

            right_frames = np.asarray(dataset[right_idx[src_mask]], dtype=np.float32)
            alpha_view = alpha_src.reshape(-1, 1, 1, 1)
            frames[src_mask] = (1.0 - alpha_view) * left_frames + alpha_view * right_frames

        return frames

    def fetch_frames_by_dates(self, dates: Sequence[pd.Timestamp]) -> np.ndarray:
        alignment = self.build_alignment(dates)
        return self.fetch_frames_from_alignment(alignment)


class QuantileLoss(nn.Module):
    def __init__(self, quantiles: Optional[Sequence[float]] = None):
        super().__init__()
        self.quantiles = list(quantiles) if quantiles is not None else QUANTILES

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 2:
            targets = targets.unsqueeze(-1)

        errors = targets - predictions
        quantiles_tensor = torch.tensor(
            self.quantiles, dtype=predictions.dtype, device=predictions.device
        )
        losses = torch.max(quantiles_tensor * errors, (quantiles_tensor - 1.0) * errors)
        return losses.mean()


class ConvNeXtTimeXerQuantile(nn.Module):
    """联合模型：ConvNeXt-Tiny -> weather embedding -> TimeXer -> quantiles。"""

    def __init__(self, configs, quantiles: Optional[Sequence[float]] = None):
        super().__init__()
        self.quantiles = list(quantiles) if quantiles is not None else QUANTILES
        self.n_quantiles = len(self.quantiles)
        self.weather_feature_dim = int(configs.weather_feature_dim)
        self.encode_chunk_size = int(getattr(configs, "weather_encode_chunk_size", WEATHER_ENCODE_CHUNK_SIZE))

        self.weather_backbone = ConvNeXtTinyWeatherExtractor(
            in_channels=WEATHER_IN_CHANNELS,
            feature_stage=configs.weather_feature_stage,
            use_pretrained=getattr(configs, "weather_pretrained", WEATHER_PRETRAINED),
            train_backbone=getattr(configs, "weather_train_backbone", WEATHER_TRAIN_BACKBONE),
        )
        self.weather_projector = nn.Sequential(
            nn.LayerNorm(self.weather_backbone.output_dim),
            nn.Linear(self.weather_backbone.output_dim, self.weather_feature_dim),
            nn.GELU(),
            nn.Dropout(configs.dropout),
        )

        self.timexer = TimeXer(configs)
        self.quantile_head = nn.Linear(1, self.n_quantiles)

        with torch.no_grad():
            self.quantile_head.weight.fill_(1.0)
            self.quantile_head.bias.copy_(torch.tensor([q - 0.5 for q in self.quantiles]) * 0.1)

    def _encode_weather_chunk(self, weather_chunk: torch.Tensor) -> torch.Tensor:
        features = self.weather_backbone(weather_chunk)
        return self.weather_projector(features)

    def _encode_weather_sequence(self, weather_seq: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if weather_seq is None:
            return None
        if weather_seq.ndim != 5:
            raise ValueError(f"weather_seq 必须是 [B, T, C, H, W]，收到 {tuple(weather_seq.shape)}")

        bsz, time_len, channels, height, width = weather_seq.shape
        flat = weather_seq.reshape(bsz * time_len, channels, height, width).float()
        encoded_chunks: List[torch.Tensor] = []

        for start in range(0, flat.shape[0], self.encode_chunk_size):
            end = min(start + self.encode_chunk_size, flat.shape[0])
            chunk = flat[start:end]
            encoded_chunks.append(self._encode_weather_chunk(chunk))

        encoded = torch.cat(encoded_chunks, dim=0)
        return encoded.reshape(bsz, time_len, self.weather_feature_dim)

    def forward(
        self,
        load_x: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        weather_x: torch.Tensor,
        weather_y: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hist_weather_feature = self._encode_weather_sequence(weather_x)
        x_enc = torch.cat([hist_weather_feature, load_x], dim=-1)

        x_fut_known = None
        if self.timexer.use_future_covariates and weather_y is not None:
            future_weather = weather_y[:, -self.timexer.pred_len :, :, :, :]
            x_fut_known = self._encode_weather_sequence(future_weather)

        point_pred = self.timexer(
            x_enc,
            x_mark_enc,
            x_dec,
            x_mark_dec,
            mask=mask,
            x_fut_known=x_fut_known,
        )
        point_pred = point_pred[:, -self.timexer.pred_len :, :]
        return self.quantile_head(point_pred)


class LoadWeatherEndToEndDataset(Dataset):
    """返回负荷序列、时间特征以及原始天气网格。"""

    def __init__(
        self,
        args,
        weather_store: WeatherGridStore,
        flag: str = "train",
        size: Optional[Sequence[int]] = None,
        target: str = TARGET,
        scale: bool = True,
        timeenc: int = 1,
        freq: str = "15min",
    ):
        if size is None:
            size = [SEQ_LEN, LABEL_LEN, PRED_LEN]

        self.args = args
        self.seq_len = int(size[0])
        self.label_len = int(size[1])
        self.pred_len = int(size[2])
        self.target = target
        self.scale = bool(scale)
        self.timeenc = int(timeenc)
        self.freq = freq
        self.weather_store = weather_store

        flag_map = {"train": 0, "val": 1, "test": 2}
        if flag not in flag_map:
            raise ValueError(f"flag 必须是 train/val/test，收到 {flag}")
        self.set_type = flag_map[flag]

        self.scaler: Optional[StandardScaler] = None
        self.target_mean = 0.0
        self.target_scale = 1.0
        self.data_x: Optional[np.ndarray] = None
        self.data_y: Optional[np.ndarray] = None
        self.data_stamp: Optional[np.ndarray] = None
        self.raw_dates: Optional[pd.Series] = None
        self.weather_alignment: Optional[Dict[str, np.ndarray]] = None

        self.__read_data__()

    def __read_data__(self) -> None:
        csv_path = os.path.join(self.args.root_path, self.args.data_path)
        df_raw = pd.read_csv(csv_path)
        if "date" not in df_raw.columns:
            raise ValueError(f"数据文件缺少 date 列: {csv_path}")

        df_raw["date"] = pd.to_datetime(df_raw["date"])
        df_raw = df_raw.sort_values("date").reset_index(drop=True)

        if self.target not in df_raw.columns and "Target" in df_raw.columns:
            df_raw = df_raw.rename(columns={"Target": self.target})
        if self.target not in df_raw.columns:
            raise ValueError(f"数据文件中不存在目标列 {self.target}: {csv_path}")

        total_len = len(df_raw)
        num_train = int(total_len * 0.6)
        num_test = int(total_len * 0.1)
        num_vali = total_len - num_train - num_test

        border1s = [
            0,
            max(0, num_train - self.seq_len),
            max(0, num_train + num_vali - self.seq_len),
        ]
        border2s = [num_train, num_train + num_vali, total_len]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        target_values = df_raw[[self.target]].values.astype(np.float32)
        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(target_values[: border2s[0]])
            target_values = self.scaler.transform(target_values).astype(np.float32)
            self.target_mean = float(self.scaler.mean_[0])
            self.target_scale = float(self.scaler.scale_[0]) if self.scaler.scale_[0] != 0 else 1.0

        df_stamp = df_raw[["date"]].iloc[border1:border2].copy()
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp["date"].apply(lambda row: row.month)
            df_stamp["day"] = df_stamp["date"].apply(lambda row: row.day)
            df_stamp["weekday"] = df_stamp["date"].apply(lambda row: row.weekday())
            df_stamp["hour"] = df_stamp["date"].apply(lambda row: row.hour)
            df_stamp["minute"] = df_stamp["date"].apply(lambda row: row.minute)
            data_stamp = df_stamp.drop(columns=["date"]).values.astype(np.float32)
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp["date"].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0).astype(np.float32)

        self.data_x = target_values[border1:border2]
        self.data_y = target_values[border1:border2]
        self.data_stamp = data_stamp
        self.raw_dates = df_raw["date"].iloc[border1:border2].reset_index(drop=True)
        self.weather_alignment = self.weather_store.build_alignment(self.raw_dates)

        print(
            f"[Dataset-{self.set_type}] rows={len(self.data_x)}, "
            f"load_dim={self.data_x.shape[-1]}, frame_shape={self.weather_store.frame_shape}"
        )

    def __getitem__(self, index: int):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        weather_x = self.weather_store.fetch_frames_from_alignment(self.weather_alignment, s_begin, s_end)
        weather_y = self.weather_store.fetch_frames_from_alignment(self.weather_alignment, r_begin, r_end)

        return seq_x, seq_y, seq_x_mark, seq_y_mark, weather_x, weather_y

    def __len__(self) -> int:
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def scale_target(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float32).reshape(-1, 1)
        if not self.scale or self.scaler is None:
            return data.astype(np.float32)
        return self.scaler.transform(data).astype(np.float32)

    def inverse_transform_target(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float32)
        if not self.scale:
            return data
        return data * self.target_scale + self.target_mean


def weather_data_provider(args, flag: str, weather_store: WeatherGridStore):
    timeenc = 0 if args.embed != "timeF" else 1
    shuffle_flag = flag == "train"

    dataset = LoadWeatherEndToEndDataset(
        args=args,
        weather_store=weather_store,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        target=args.target,
        scale=True,
        timeenc=timeenc,
        freq=args.freq,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=False,
    )
    return dataset, loader


def extract_target(batch_y: torch.Tensor) -> torch.Tensor:
    return batch_y[:, :, :]


def restore_sliding_window_2d(data_2d: np.ndarray) -> np.ndarray:
    if len(data_2d) == 0:
        return np.array([])
    restored = list(data_2d[0, :])
    for i in range(1, len(data_2d)):
        restored.append(data_2d[i, -1])
    return np.asarray(restored)


def restore_sliding_window_3d(data_3d: np.ndarray) -> np.ndarray:
    if len(data_3d) == 0:
        return np.array([])
    restored = list(data_3d[0, :, :])
    for i in range(1, len(data_3d)):
        restored.append(data_3d[i, -1, :])
    return np.asarray(restored)


def _load_ordered_dataframe(csv_path: str, target: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "date" not in df.columns:
        raise ValueError(f"缺少 date 列: {csv_path}")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if target not in df.columns and "Target" in df.columns:
        df = df.rename(columns={"Target": target})
    if target not in df.columns:
        raise ValueError(f"缺少目标列 {target}: {csv_path}")
    other_cols = [c for c in df.columns if c not in ("date", target)]
    return df[["date"] + other_cols + [target]]


def validate_quantile(model, data_loader, criterion, args, device):
    model.eval()
    total_loss = []

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark, batch_weather_x, batch_weather_y in data_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            batch_weather_x = batch_weather_x.float().to(device)
            batch_weather_y = batch_weather_y.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :])
            dec_inp = torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1).float().to(device)

            outputs = model(
                load_x=batch_x,
                x_mark_enc=batch_x_mark,
                x_dec=dec_inp,
                x_mark_dec=batch_y_mark,
                weather_x=batch_weather_x,
                weather_y=batch_weather_y,
            )

            batch_y_target = extract_target(batch_y[:, -args.pred_len :, :]).to(device)
            loss = criterion(outputs, batch_y_target)
            total_loss.append(loss.item())

    model.train()
    return float(np.average(total_loss)) if total_loss else np.nan


def train_quantile_model(model, args, device, weather_store: WeatherGridStore):
    _, train_loader = weather_data_provider(args, "train", weather_store)
    _, vali_loader = weather_data_provider(args, "val", weather_store)
    _, test_loader = weather_data_provider(args, "test", weather_store)

    setting = _get_setting(args)
    path = os.path.join(args.checkpoints, setting)
    os.makedirs(path, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = QuantileLoss(args.quantiles)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    print("\n" + "=" * 72)
    print("Start training ConvNeXt-Tiny + TimeXer end-to-end quantile model")
    print(f"setting: {setting}")
    print(f"quantiles: {args.quantiles}")
    print(f"weather_feature_dim: {args.weather_feature_dim}")
    print(f"weather_train_backbone: {args.weather_train_backbone}")
    print(f"batch_size: {args.batch_size}")
    print("=" * 72)

    for epoch in range(args.train_epochs):
        model.train()
        train_loss = []
        epoch_time = time.time()

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_weather_x, batch_weather_y) in enumerate(train_loader):
            optimizer.zero_grad()

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            batch_weather_x = batch_weather_x.float().to(device)
            batch_weather_y = batch_weather_y.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :])
            dec_inp = torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1).float().to(device)

            outputs = model(
                load_x=batch_x,
                x_mark_enc=batch_x_mark,
                x_dec=dec_inp,
                x_mark_dec=batch_y_mark,
                weather_x=batch_weather_x,
                weather_y=batch_weather_y,
            )

            batch_y_target = extract_target(batch_y[:, -args.pred_len :, :])
            loss = criterion(outputs, batch_y_target)
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

            if (i + 1) % 20 == 0:
                print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")

        vali_loss = validate_quantile(model, vali_loader, criterion, args, device)
        test_loss = validate_quantile(model, test_loader, criterion, args, device)
        train_loss_avg = float(np.average(train_loss)) if train_loss else np.nan
        print(
            f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.1f}s | "
            f"Train: {train_loss_avg:.7f} Vali: {vali_loss:.7f} Test: {test_loss:.7f}"
        )

        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        adjust_learning_rate(optimizer, epoch + 1, args)

    best_model_path = os.path.join(path, "checkpoint.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"Loaded best model weights: {best_model_path}")
    return model


def test_quantile_model(model, args, device, weather_store: WeatherGridStore):
    test_data, test_loader = weather_data_provider(args, "test", weather_store)

    setting = _get_setting(args)
    folder_path = os.path.join("./results/", setting)
    os.makedirs(folder_path, exist_ok=True)

    preds_p50 = []
    trues = []
    quantile_preds_all = []

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark, batch_weather_x, batch_weather_y in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            batch_weather_x = batch_weather_x.float().to(device)
            batch_weather_y = batch_weather_y.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :])
            dec_inp = torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1).float().to(device)

            outputs = model(
                load_x=batch_x,
                x_mark_enc=batch_x_mark,
                x_dec=dec_inp,
                x_mark_dec=batch_y_mark,
                weather_x=batch_weather_x,
                weather_y=batch_weather_y,
            )

            batch_y_target = extract_target(batch_y[:, -args.pred_len :, :])
            p50_pred = outputs[:, :, P50_IDX : P50_IDX + 1]

            quantile_preds_all.append(outputs.detach().cpu().numpy())
            preds_p50.append(p50_pred.detach().cpu().numpy())
            trues.append(batch_y_target.detach().cpu().numpy())

    preds_p50 = np.concatenate(preds_p50, axis=0)
    trues = np.concatenate(trues, axis=0)
    quantile_preds_all = np.concatenate(quantile_preds_all, axis=0)

    print(
        f"Test shape: preds={preds_p50.shape}, "
        f"trues={trues.shape}, quantiles={quantile_preds_all.shape}"
    )

    np.save(os.path.join(folder_path, "pred.npy"), preds_p50)
    np.save(os.path.join(folder_path, "true.npy"), trues)
    np.save(os.path.join(folder_path, "quantile_preds.npy"), quantile_preds_all)

    if test_data.scale:
        shape = trues.shape
        preds_inv = test_data.inverse_transform_target(preds_p50.reshape(shape[0] * shape[1], -1)).reshape(shape)
        trues_inv = test_data.inverse_transform_target(trues.reshape(shape[0] * shape[1], -1)).reshape(shape)

        q_shape = quantile_preds_all.shape
        quantile_inv = np.zeros_like(quantile_preds_all)
        for qi in range(N_QUANTILES):
            q_slice = quantile_preds_all[:, :, qi : qi + 1]
            q_inv = test_data.inverse_transform_target(
                q_slice.reshape(q_shape[0] * q_shape[1], -1)
            ).reshape(q_shape[0], q_shape[1], 1)
            quantile_inv[:, :, qi] = q_inv[:, :, 0]

        np.save(os.path.join(folder_path, "pred_inv.npy"), preds_inv)
        np.save(os.path.join(folder_path, "true_inv.npy"), trues_inv)
        np.save(os.path.join(folder_path, "quantile_preds_inv.npy"), quantile_inv)

    mae, mse, rmse, mape, mspe = metric(preds_p50, trues)
    print(f"P50 Test Metrics: MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}")

    return folder_path


def plot_pred_vs_true(results_dir, feat_idx=0, out_name="pred_vs_true.png", use_inverse=False):
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
    if has_quantiles:
        quantile_preds = np.load(quantile_path)

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

    if has_quantiles:
        q_p10_raw = quantile_preds[:, :, P10_IDX : P10_IDX + 1]
        q_p90_raw = quantile_preds[:, :, P90_IDX : P90_IDX + 1]

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

    if has_quantiles:
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
        ax.set_title(f"ConvNeXt-Tiny + TimeXer Prediction - MAPE: {100 * mape_val:.2f}%")
    else:
        ax.set_title("ConvNeXt-Tiny + TimeXer Prediction - MAPE: NaN")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Load (MW)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, out_name), dpi=600, bbox_inches="tight")
    plt.show()


def predict_future_load_from_csv(
    model,
    args,
    device,
    weather_store: WeatherGridStore,
    results_dir: str,
    future_path: str = FUTURE_PATH,
    steps: int = PRED_LEN,
    use_inverse: bool = True,
):
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

    ref_data, _ = weather_data_provider(args, "train", weather_store)

    hist_dates = pd.to_datetime(history_df["date"].iloc[-args.seq_len :].values)
    future_dates = pd.to_datetime(future_df["date"].iloc[: args.pred_len].values)

    hist_load = history_df[args.target].iloc[-args.seq_len :].values.astype(np.float32).reshape(-1, 1)
    hist_load_scaled = ref_data.scale_target(hist_load)

    hist_weather = weather_store.fetch_frames_by_dates(hist_dates)
    future_weather = weather_store.fetch_frames_by_dates(future_dates)

    x_mark_np = time_features(pd.to_datetime(hist_dates), freq=args.freq).transpose(1, 0).astype(np.float32)
    dec_len = args.label_len + args.pred_len
    dec_inp_np = np.zeros((dec_len, 1), dtype=np.float32)
    if args.label_len > 0:
        dec_inp_np[: args.label_len] = hist_load_scaled[-args.label_len :]

    weather_y_np = np.full(
        (dec_len,) + weather_store.frame_shape,
        WEATHER_FILL_VALUE,
        dtype=np.float32,
    )
    if args.label_len > 0:
        weather_y_np[: args.label_len] = hist_weather[-args.label_len :]
    weather_y_np[-args.pred_len :] = future_weather

    if args.label_len > 0:
        label_dates = hist_dates[-args.label_len :]
        y_dates = np.concatenate([label_dates.to_numpy(), future_dates.to_numpy()])
    else:
        y_dates = future_dates.to_numpy()
    batch_y_mark = time_features(pd.to_datetime(y_dates), freq=args.freq).transpose(1, 0).astype(np.float32)

    model.eval()
    with torch.no_grad():
        batch_x = torch.as_tensor(hist_load_scaled, dtype=torch.float32, device=device).unsqueeze(0)
        batch_x_mark = torch.as_tensor(x_mark_np, dtype=torch.float32, device=device).unsqueeze(0)
        dec_inp = torch.as_tensor(dec_inp_np, dtype=torch.float32, device=device).unsqueeze(0)
        batch_y_mark = torch.as_tensor(batch_y_mark, dtype=torch.float32, device=device).unsqueeze(0)
        batch_weather_x = torch.as_tensor(hist_weather, dtype=torch.float32, device=device).unsqueeze(0)
        batch_weather_y = torch.as_tensor(weather_y_np, dtype=torch.float32, device=device).unsqueeze(0)

        outputs = model(
            load_x=batch_x,
            x_mark_enc=batch_x_mark,
            x_dec=dec_inp,
            x_mark_dec=batch_y_mark,
            weather_x=batch_weather_x,
            weather_y=batch_weather_y,
        )

    quantile_scaled = outputs[0, : args.pred_len, :].detach().cpu().numpy()
    p10_scaled = quantile_scaled[:, P10_IDX]
    p50_scaled = quantile_scaled[:, P50_IDX]
    p90_scaled = quantile_scaled[:, P90_IDX]

    if use_inverse:
        preds_p10 = ref_data.inverse_transform_target(p10_scaled.reshape(-1, 1)).reshape(-1)
        preds_p50 = ref_data.inverse_transform_target(p50_scaled.reshape(-1, 1)).reshape(-1)
        preds_p90 = ref_data.inverse_transform_target(p90_scaled.reshape(-1, 1)).reshape(-1)
        history_target = history_df[args.target].values
    else:
        preds_p10 = p10_scaled
        preds_p50 = p50_scaled
        preds_p90 = p90_scaled
        history_target = hist_load_scaled.reshape(-1)

    future_dates = pd.Series(future_dates[:predict_steps])
    preds_p10 = preds_p10[:predict_steps]
    preds_p50 = preds_p50[:predict_steps]
    preds_p90 = preds_p90[:predict_steps]

    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "future_load_prediction.csv")
    pd.DataFrame(
        {
            "date": future_dates,
            f"{args.target}_pred_P10": preds_p10,
            f"{args.target}_pred_P50": preds_p50,
            f"{args.target}_pred_P90": preds_p90,
        }
    ).to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"\nFuture {predict_steps}-step {args.target} predictions:")
    print(f"{'Time':<25} {'P10':<12} {'P50':<12} {'P90':<12}")
    print("-" * 65)
    for i in range(predict_steps):
        print(f"  {future_dates.iloc[i]}: {preds_p10[i]:<12.4f} {preds_p50[i]:<12.4f} {preds_p90[i]:<12.4f}")

    n_history = min(args.seq_len, len(history_target))
    history_tail = history_target[-n_history:]
    future_x = range(n_history, n_history + predict_steps)

    plt.figure(figsize=(15, 6), facecolor="white")
    plt.plot(range(n_history), history_tail, label="Historical Load", color="tab:blue", alpha=0.8)
    plt.plot(
        future_x,
        preds_p50,
        label="ConvNeXt-Tiny + TimeXer P50 Prediction",
        color="tab:orange",
        linewidth=2,
        marker="o",
        markersize=2,
    )
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
    plt.title(f"ConvNeXt-Tiny + TimeXer Future {predict_steps}-Step Load Prediction")
    plt.xlabel("Time Step (15min)")
    plt.ylabel("Load (MW)")
    plt.tight_layout()

    out_fig = os.path.join(results_dir, "future_load_prediction.png")
    plt.savefig(out_fig, dpi=600, bbox_inches="tight")
    plt.show()

    print(f"Saved future prediction csv: {out_csv}")
    print(f"Saved future prediction figure: {out_fig}")


def _get_setting(args, itr=0):
    signature = (
        f"{args.task_name}_{args.model_id}_{args.model}_e2e_"
        f"sl{args.seq_len}_pl{args.pred_len}_dm{args.d_model}_"
        f"el{args.e_layers}_wd{args.weather_feature_dim}_"
        f"ws{args.weather_feature_stage}_fc{int(args.use_future_covariates)}_"
        f"wb{int(args.weather_train_backbone)}_lr{args.learning_rate}_"
        f"bs{args.batch_size}_{args.des}_{itr}"
    )
    digest = hashlib.md5(signature.encode("utf-8")).hexdigest()[:8]
    return (
        f"TimeXerE2E_sl{args.seq_len}_pl{args.pred_len}_"
        f"wd{args.weather_feature_dim}_fc{int(args.use_future_covariates)}_"
        f"wb{int(args.weather_train_backbone)}_bs{args.batch_size}_{args.des}_{itr}_{digest}"
    )


def main():
    fix_seed = 2026
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args = argparse.Namespace(
        task_name=TASK_NAME,
        is_training=1 if TRAIN_MODE else 0,
        model_id=MODEL_ID,
        model=MODEL,
        data="custom",
        root_path=ROOT_PATH,
        data_path=DATA_PATH,
        features=FEATURES,
        target=TARGET,
        freq="15min",
        embed="timeF",
        checkpoints="./checkpoints_quantile/",
        seq_len=SEQ_LEN,
        label_len=LABEL_LEN,
        pred_len=PRED_LEN,
        enc_in=ENC_IN,
        c_out=C_OUT,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        e_layers=E_LAYERS,
        d_ff=D_FF,
        factor=FACTOR,
        dropout=DROPOUT,
        activation=ACTIVATION,
        patch_len=PATCH_LEN,
        use_norm=USE_NORM,
        use_future_covariates=USE_FUTURE_COVARIATES,
        future_cov_dim=FUTURE_COV_DIM,
        future_cov_dropout=FUTURE_COV_DROPOUT,
        target_channel_idx=0,
        weather_h5_specs=WEATHER_H5_SPECS,
        weather_feature_stage=WEATHER_FEATURE_STAGE,
        weather_feature_dim=WEATHER_FEATURE_DIM,
        weather_pretrained=WEATHER_PRETRAINED,
        weather_train_backbone=WEATHER_TRAIN_BACKBONE,
        weather_encode_chunk_size=WEATHER_ENCODE_CHUNK_SIZE,
        num_workers=NUM_WORKERS,
        itr=ITR,
        train_epochs=TRAIN_EPOCHS,
        batch_size=BATCH_SIZE,
        patience=PATIENCE,
        learning_rate=LEARNING_RATE,
        des=DES,
        loss="Quantile",
        lradj="type1",
        use_amp=False,
        inverse_eval=INVERSE_EVAL,
        use_gpu=USE_GPU,
        gpu=GPU,
        use_multi_gpu=False,
        devices="0,1,2,3",
        quantiles=QUANTILES,
        n_quantiles=N_QUANTILES,
    )

    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU: cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    weather_store = WeatherGridStore(args.weather_h5_specs, fill_value=WEATHER_FILL_VALUE)
    try:
        model = ConvNeXtTimeXerQuantile(args, quantiles=QUANTILES).float().to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"ConvNeXt-Tiny + TimeXer total params: {total_params:,}")
        print(f"ConvNeXt-Tiny + TimeXer trainable params: {trainable_params:,}")

        setting = _get_setting(args)

        if TRAIN_MODE:
            print(f"\n>>> Start training {setting}")
            model = train_quantile_model(model, args, device, weather_store)

            print(f"\n>>> Start testing {setting}")
            results_dir = test_quantile_model(model, args, device, weather_store)

            plot_pred_vs_true(results_dir, use_inverse=INVERSE_EVAL)
            predict_future_load_from_csv(
                model=model,
                args=args,
                device=device,
                weather_store=weather_store,
                results_dir=results_dir,
                future_path=FUTURE_PATH,
                steps=PRED_LEN,
                use_inverse=INVERSE_EVAL,
            )
        else:
            ckpt_path = os.path.join(args.checkpoints, setting, "checkpoint.pth")
            if os.path.exists(ckpt_path):
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                print(f"Loaded model: {ckpt_path}")
            else:
                raise FileNotFoundError(
                    f"Model file not found: {ckpt_path}. Please set TRAIN_MODE = True first."
                )

            print(f"\n>>> Test only {setting}")
            results_dir = test_quantile_model(model, args, device, weather_store)

            plot_pred_vs_true(results_dir, use_inverse=INVERSE_EVAL)
            predict_future_load_from_csv(
                model=model,
                args=args,
                device=device,
                weather_store=weather_store,
                results_dir=results_dir,
                future_path=FUTURE_PATH,
                steps=PRED_LEN,
                use_inverse=INVERSE_EVAL,
            )
    finally:
        weather_store.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
