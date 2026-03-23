"""
test4.py - TimeXer + ConvNeXt-Tiny 气象外生变量概率预测脚本

基于 test3.py 改造：
1. 使用 ConvNeXt-Tiny 从湖南省网格气象 HDF5 中提取每个时间步的高维气象表征。
2. 将气象表征作为 TimeXer 的外生变量输入编码器。
3. 同时将未来气象表征作为 future covariates 传入 TimeXer 的 future_cov_head。
4. 保留 test3.py 的分位数预测流程，输出 P10 / P50 / P90。

说明：
- 这里采用“先提取、后训练”的工程方案：ConvNeXt-Tiny 在脚本启动时对 HDF5 帧做一次特征缓存，
  然后训练 TimeXer。这样可以避免把整段 7 天天气网格直接塞进 DataLoader 造成显存/内存爆炸。
- 为了兼容你后续补齐 2026 年气象数据，脚本支持多个 HDF5 文件；不存在的文件会自动跳过。
- 如果本地没有 torchvision 预训练权重缓存，且运行环境无法联网，脚本会退回随机初始化的
  ConvNeXt-Tiny 特征提取器，但代码逻辑不受影响。
"""

import argparse
import hashlib
import os
import random
import re
import time
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import h5py
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from models.TimeXer import Model as TimeXer
from utils.metrics import cal_eval, metric
from utils.timefeatures import time_features
from utils.tools import EarlyStopping, adjust_learning_rate

warnings.filterwarnings("ignore")


# ==================== 分位数配置 ====================
QUANTILES = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
N_QUANTILES = len(QUANTILES)
P50_IDX = QUANTILES.index(0.5)
P10_IDX = QUANTILES.index(0.1)
P90_IDX = QUANTILES.index(0.9)


# ==================== 基础任务配置 ====================
TASK_NAME = "long_term_forecast"
MODEL = "TimeXer"
MODEL_ID = "HunanLoad_672_96_ConvNeXtTiny"


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

WEATHER_CHANNEL_NAMES = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "dew_point_2m",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "shortwave_radiation",
    "direct_radiation",
    "precipitation",
]
WEATHER_CACHE_DIR = "./data/weather_cache"
WEATHER_IN_CHANNELS = 10
WEATHER_FEATURE_STAGE = 2
WEATHER_STAGE_TO_DIM = {1: 96, 2: 192, 3: 384, 4: 768}
WEATHER_FEATURE_DIM = WEATHER_STAGE_TO_DIM[WEATHER_FEATURE_STAGE]
WEATHER_CACHE_BATCH_SIZE = 64
WEATHER_PRETRAINED = True
WEATHER_FORCE_REBUILD = False
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
TRAIN_MODE = False


# ==================== future covariates 配置 ====================
USE_FUTURE_COVARIATES = False
FUTURE_COV_DIM = WEATHER_FEATURE_DIM
FUTURE_COV_DROPOUT = 0.1


def _require_weather_runtime() -> None:
    """运行前检查必要依赖。"""
    if h5py is None:
        raise ImportError("缺少 h5py，test4.py 需要 h5py 读取 HDF5 气象文件。")
    if convnext_tiny is None:
        raise ImportError("缺少 torchvision，test4.py 需要 torchvision.models.convnext_tiny。")


def _guess_year_start_from_path(file_path: str) -> pd.Timestamp:
    """从文件名推断年份起点。"""
    name = Path(file_path).stem
    match = re.search(r"(20\d{2})", name)
    if match:
        return pd.Timestamp(f"{match.group(1)}-01-01 00:00:00")
    return pd.Timestamp("2025-01-01 00:00:00")


def _find_first_4d_dataset(h5_obj):
    """递归查找第一个 4D dataset。"""
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
    """根据全年步数推断 HDF5 的时间分辨率。"""
    days_in_year = 366 if start_time.is_leap_year else 365
    steps_per_day = n_steps / float(days_in_year)
    candidates = np.array([24.0, 48.0, 96.0])
    best = candidates[np.argmin(np.abs(candidates - steps_per_day))]
    if not np.isfinite(best) or best <= 0:
        raise ValueError(f"无法从 n_steps={n_steps} 推断气象时间频率。")
    minutes = int(round(1440.0 / best))
    return pd.Timedelta(minutes=minutes)


def _adapt_first_conv_to_multichannel(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """把第一层卷积从 3 通道扩展到 10 通道。"""
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
    """冻结的 ConvNeXt-Tiny 气象特征提取器。"""

    STAGE_TO_LAYER = {1: 1, 2: 3, 3: 5, 4: 7}
    STAGE_TO_DIM = {1: 96, 2: 192, 3: 384, 4: 768}

    def __init__(
        self,
        in_channels: int = WEATHER_IN_CHANNELS,
        feature_stage: int = WEATHER_FEATURE_STAGE,
        use_pretrained: bool = WEATHER_PRETRAINED,
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

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 [B, C, H, W]，输出 [B, feature_dim]。"""
        if x.ndim != 4:
            raise ValueError(f"气象输入必须是 [B, C, H, W]，收到 {tuple(x.shape)}")

        x = x.float()
        spatial_mean = x.mean(dim=(-2, -1), keepdim=True)
        spatial_std = x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        x = (x - spatial_mean) / spatial_std

        for layer_idx, layer in enumerate(self.backbone):
            x = layer(x)
            if layer_idx == self.stop_layer:
                break

        return x.mean(dim=(-2, -1))


class WeatherFeatureStore:
    """从 HDF5 中提取、缓存并按时间戳对齐气象特征。"""

    def __init__(
        self,
        h5_specs: Sequence[Tuple[str, str]],
        cache_dir: str,
        feature_stage: int = WEATHER_FEATURE_STAGE,
        batch_size: int = WEATHER_CACHE_BATCH_SIZE,
        use_pretrained: bool = WEATHER_PRETRAINED,
        fill_value: float = WEATHER_FILL_VALUE,
        force_rebuild: bool = WEATHER_FORCE_REBUILD,
        device: Optional[str] = None,
    ):
        _require_weather_runtime()

        self.h5_specs = [
            (
                os.path.abspath(path),
                pd.Timestamp(start) if start else _guess_year_start_from_path(path),
            )
            for path, start in h5_specs
        ]
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.feature_stage = feature_stage
        self.feature_dim = ConvNeXtTinyWeatherExtractor.STAGE_TO_DIM[feature_stage]
        self.batch_size = int(batch_size)
        self.use_pretrained = bool(use_pretrained)
        self.fill_value = float(fill_value)
        self.force_rebuild = bool(force_rebuild)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.features: Optional[np.ndarray] = None
        self.timestamp_ns: Optional[np.ndarray] = None
        self.timestamps: Optional[pd.DatetimeIndex] = None
        self._warned_out_of_range = False

    def _cache_path_for(self, h5_path: str) -> Path:
        stem = Path(h5_path).stem
        return self.cache_dir / f"{stem}_convnext_tiny_stage{self.feature_stage}_{self.feature_dim}d.npy"

    def _extract_single_file(self, h5_path: str) -> np.ndarray:
        cache_path = self._cache_path_for(h5_path)
        if cache_path.exists() and not self.force_rebuild:
            features = np.load(cache_path, mmap_mode="r")
            print(f"[Weather] 加载缓存: {cache_path} -> {tuple(features.shape)}")
            return features

        print(f"[Weather] 开始提取 ConvNeXt-Tiny 特征: {h5_path}")
        extractor = ConvNeXtTinyWeatherExtractor(
            in_channels=WEATHER_IN_CHANNELS,
            feature_stage=self.feature_stage,
            use_pretrained=self.use_pretrained,
        ).to(self.device)
        extractor.eval()

        with h5py.File(h5_path, "r") as h5_file:
            dataset = _find_first_4d_dataset(h5_file)
            if dataset is None:
                raise ValueError(f"在 {h5_path} 中未找到 4D dataset。")
            if dataset.shape[1] != WEATHER_IN_CHANNELS:
                raise ValueError(
                    f"{h5_path} 的气象通道数不是 {WEATHER_IN_CHANNELS}，实际为 {dataset.shape[1]}"
                )

            total_steps = dataset.shape[0]
            features = np.empty((total_steps, self.feature_dim), dtype=np.float32)

            with torch.inference_mode():
                for start in range(0, total_steps, self.batch_size):
                    end = min(start + self.batch_size, total_steps)
                    batch_np = dataset[start:end]
                    batch_tensor = torch.from_numpy(batch_np).float().to(self.device)
                    batch_features = extractor(batch_tensor).cpu().numpy().astype(np.float32)
                    features[start:end] = batch_features

                    batch_idx = start // self.batch_size + 1
                    if batch_idx % 20 == 0 or end == total_steps:
                        print(f"[Weather] {Path(h5_path).name}: {end}/{total_steps}")

        np.save(cache_path, features)
        print(f"[Weather] 特征缓存已保存: {cache_path}")
        return np.load(cache_path, mmap_mode="r")

    def _build_timestamps(self, n_steps: int, start_time: pd.Timestamp) -> pd.DatetimeIndex:
        freq = _infer_weather_freq(n_steps, start_time)
        print(f"[Weather] 推断 {start_time.year} 气象时间分辨率为: {freq}")
        return pd.date_range(start=start_time, periods=n_steps, freq=freq)

    def prepare(self) -> None:
        """提取或加载全部气象特征。"""
        if self.features is not None and self.timestamp_ns is not None:
            return

        feature_parts: List[np.ndarray] = []
        timestamp_parts: List[np.ndarray] = []

        for h5_path, start_time in self.h5_specs:
            if not os.path.exists(h5_path):
                print(f"[Weather] 文件不存在，跳过: {h5_path}")
                continue

            features = self._extract_single_file(h5_path)
            timestamps = self._build_timestamps(features.shape[0], start_time)

            feature_parts.append(np.asarray(features, dtype=np.float32))
            timestamp_parts.append(timestamps.asi8.copy())

        if not feature_parts:
            raise FileNotFoundError("未找到任何可用的气象 HDF5 文件。请检查 WEATHER_H5_SPECS。")

        all_features = np.concatenate(feature_parts, axis=0)
        all_timestamp_ns = np.concatenate(timestamp_parts, axis=0).astype(np.int64)

        order = np.argsort(all_timestamp_ns)
        all_features = all_features[order]
        all_timestamp_ns = all_timestamp_ns[order]

        unique_mask = np.ones(len(all_timestamp_ns), dtype=bool)
        if len(unique_mask) > 1:
            unique_mask[1:] = all_timestamp_ns[1:] != all_timestamp_ns[:-1]

        self.features = all_features[unique_mask]
        self.timestamp_ns = all_timestamp_ns[unique_mask]
        self.timestamps = pd.to_datetime(self.timestamp_ns)

        print(f"[Weather] 合并后特征矩阵: {tuple(self.features.shape)}")
        print(f"[Weather] 时间范围: {self.timestamps.min()} ~ {self.timestamps.max()}")

    def align_to_dates(self, dates: Sequence[pd.Timestamp]) -> np.ndarray:
        """把气象特征插值对齐到负荷时间戳。"""
        self.prepare()

        date_index = pd.DatetimeIndex(pd.to_datetime(dates))
        if len(date_index) == 0:
            return np.empty((0, self.feature_dim), dtype=np.float32)

        request_ns = date_index.asi8.astype(np.float64)
        source_ns = self.timestamp_ns.astype(np.float64)

        aligned = np.full((len(date_index), self.feature_dim), self.fill_value, dtype=np.float32)
        valid_mask = (request_ns >= source_ns[0]) & (request_ns <= source_ns[-1])

        if valid_mask.any():
            valid_ns = request_ns[valid_mask]
            for feat_idx in range(self.feature_dim):
                aligned[valid_mask, feat_idx] = np.interp(
                    valid_ns,
                    source_ns,
                    np.asarray(self.features[:, feat_idx], dtype=np.float64),
                ).astype(np.float32)

        if (~valid_mask).any() and not self._warned_out_of_range:
            n_missing = int((~valid_mask).sum())
            print(
                f"[Weather] 警告: 有 {n_missing} 个时间点超出气象覆盖范围，"
                f"将使用 fill_value={self.fill_value} 填充。"
            )
            self._warned_out_of_range = True

        return aligned


class QuantileLoss(nn.Module):
    """Pinball Loss / Quantile Loss。"""

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


class TimeXerWeatherQuantile(nn.Module):
    """支持气象外生变量与 future covariates 的 TimeXer 分位数包装器。"""

    def __init__(self, configs, quantiles: Optional[Sequence[float]] = None):
        super().__init__()
        self.quantiles = list(quantiles) if quantiles is not None else QUANTILES
        self.n_quantiles = len(self.quantiles)
        self.timexer = TimeXer(configs)
        self.quantile_head = nn.Linear(1, self.n_quantiles)

        with torch.no_grad():
            self.quantile_head.weight.fill_(1.0)
            self.quantile_head.bias.copy_(torch.tensor([q - 0.5 for q in self.quantiles]) * 0.1)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        x_fut_known: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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


class LoadWeatherDataset(Dataset):
    """负荷 + ConvNeXt-Tiny 气象特征数据集。"""

    def __init__(
        self,
        args,
        weather_store: WeatherFeatureStore,
        flag: str = "train",
        size: Optional[Sequence[int]] = None,
        features: str = FEATURES,
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
        self.features = features
        self.target = target
        self.scale = bool(scale)
        self.timeenc = int(timeenc)
        self.freq = freq
        self.weather_store = weather_store
        self.weather_dim = int(args.weather_feature_dim)

        flag_map = {"train": 0, "val": 1, "test": 2}
        if flag not in flag_map:
            raise ValueError(f"flag 必须为 train/val/test，收到 {flag}")
        self.set_type = flag_map[flag]

        self.scaler: Optional[StandardScaler] = None
        self.target_mean = 0.0
        self.target_scale = 1.0
        self.data_x: Optional[np.ndarray] = None
        self.data_y: Optional[np.ndarray] = None
        self.data_stamp: Optional[np.ndarray] = None
        self.raw_dates: Optional[pd.Series] = None

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

        weather_features = self.weather_store.align_to_dates(df_raw["date"])
        if weather_features.shape[1] != self.weather_dim:
            raise ValueError(
                f"weather dim 不匹配，期望 {self.weather_dim}，实际 {weather_features.shape[1]}"
            )

        target_values = df_raw[[self.target]].values.astype(np.float32)
        all_features = np.concatenate([weather_features, target_values], axis=1).astype(np.float32)

        if self.scale:
            self.scaler = StandardScaler()
            self.scaler.fit(all_features[:border2s[0]])
            data = self.scaler.transform(all_features).astype(np.float32)
            self.target_mean = float(self.scaler.mean_[-1])
            self.target_scale = float(self.scaler.scale_[-1]) if self.scaler.scale_[-1] != 0 else 1.0
        else:
            data = all_features
            self.target_mean = 0.0
            self.target_scale = 1.0

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

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.raw_dates = df_raw["date"].iloc[border1:border2].reset_index(drop=True)

        print(
            f"[Dataset-{self.set_type}] rows={len(self.data_x)}, "
            f"features={self.data_x.shape[-1]}, weather_dim={self.weather_dim}"
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
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self) -> int:
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def scale_full_features(self, data: np.ndarray) -> np.ndarray:
        if not self.scale or self.scaler is None:
            return np.asarray(data, dtype=np.float32)
        return self.scaler.transform(np.asarray(data, dtype=np.float32)).astype(np.float32)

    def scale_weather_features(self, weather: np.ndarray) -> np.ndarray:
        weather = np.asarray(weather, dtype=np.float32)
        if not self.scale or self.scaler is None:
            return weather
        mean = self.scaler.mean_[: self.weather_dim]
        scale = np.where(self.scaler.scale_[: self.weather_dim] == 0, 1.0, self.scaler.scale_[: self.weather_dim])
        return ((weather - mean) / scale).astype(np.float32)

    def inverse_transform_target(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float32)
        if not self.scale:
            return data
        return data * self.target_scale + self.target_mean


def weather_data_provider(args, flag: str, weather_store: WeatherFeatureStore):
    """构造带气象外生变量的数据集与 DataLoader。"""
    timeenc = 0 if args.embed != "timeF" else 1
    shuffle_flag = flag == "train"

    dataset = LoadWeatherDataset(
        args=args,
        weather_store=weather_store,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
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


def extract_future_covariates(batch_y: torch.Tensor, args) -> Optional[torch.Tensor]:
    """从 batch_y 中取出未来气象特征 [B, pred_len, weather_dim]。"""
    if not getattr(args, "use_future_covariates", False):
        return None
    if getattr(args, "future_cov_dim", 0) <= 0:
        return None
    return batch_y[:, -args.pred_len :, : args.future_cov_dim]


def extract_target(batch_y: torch.Tensor, args) -> torch.Tensor:
    """目标变量固定放在最后一列。"""
    return batch_y[:, -args.pred_len :, -1:]


def restore_sliding_window_2d(data_2d: np.ndarray) -> np.ndarray:
    """将 [N, pred_len] 的滑窗预测恢复为连续序列。"""
    if len(data_2d) == 0:
        return np.array([])
    restored = list(data_2d[0, :])
    for i in range(1, len(data_2d)):
        restored.append(data_2d[i, -1])
    return np.asarray(restored)


def restore_sliding_window_3d(data_3d: np.ndarray) -> np.ndarray:
    """将 [N, pred_len, C] 的滑窗预测恢复为连续序列。"""
    if len(data_3d) == 0:
        return np.array([])
    restored = list(data_3d[0, :, :])
    for i in range(1, len(data_3d)):
        restored.append(data_3d[i, -1, :])
    return np.asarray(restored)


def _load_ordered_dataframe(csv_path: str, target: str) -> pd.DataFrame:
    """读取并按时间排序，且保证目标列存在。"""
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


def train_quantile_model(model, args, device, weather_store: WeatherFeatureStore):
    """训练带气象外生变量的 TimeXer-Quantile。"""
    train_data, train_loader = weather_data_provider(args, "train", weather_store)
    vali_data, vali_loader = weather_data_provider(args, "val", weather_store)
    test_data, test_loader = weather_data_provider(args, "test", weather_store)

    setting = _get_setting(args)
    path = os.path.join(args.checkpoints, setting)
    os.makedirs(path, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = QuantileLoss(args.quantiles)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    print("\n" + "=" * 72)
    print("开始训练 TimeXer + ConvNeXt-Tiny 气象分位数模型")
    print(f"分位数: {args.quantiles}")
    print(f"weather dim: {args.weather_feature_dim}")
    print("=" * 72)

    for epoch in range(args.train_epochs):
        model.train()
        train_loss = []
        epoch_time = time.time()

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            optimizer.zero_grad()

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float()
            dec_inp = torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1).float().to(device)
            batch_fut_known = extract_future_covariates(batch_y, args)

            outputs = model(
                batch_x,
                batch_x_mark,
                dec_inp,
                batch_y_mark,
                x_fut_known=batch_fut_known,
            )

            batch_y_target = extract_target(batch_y, args)
            loss = criterion(outputs, batch_y_target)
            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")

        vali_loss = validate_quantile(model, vali_loader, criterion, args, device)
        test_loss = validate_quantile(model, test_loader, criterion, args, device)

        train_loss_avg = float(np.average(train_loss))
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
    print(f"已加载最优模型参数: {best_model_path}")

    return model


def validate_quantile(model, data_loader, criterion, args, device) -> float:
    """验证集/测试集损失。"""
    model.eval()
    total_loss = []

    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float()
            dec_inp = torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1).float().to(device)
            batch_fut_known = extract_future_covariates(batch_y, args)

            outputs = model(
                batch_x,
                batch_x_mark,
                dec_inp,
                batch_y_mark,
                x_fut_known=batch_fut_known,
            )

            batch_y_target = extract_target(batch_y, args)
            loss = criterion(outputs, batch_y_target)
            total_loss.append(loss.item())

    model.train()
    return float(np.average(total_loss))


def test_quantile_model(model, args, device, weather_store: WeatherFeatureStore) -> str:
    """测试模型并保存预测结果。"""
    test_data, test_loader = weather_data_provider(args, "test", weather_store)

    setting = _get_setting(args)
    folder_path = os.path.join("./results/", setting)
    os.makedirs(folder_path, exist_ok=True)

    preds_p50 = []
    trues = []
    quantile_preds_all = []

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float()
            dec_inp = torch.cat([batch_y[:, : args.label_len, :], dec_inp], dim=1).float().to(device)
            batch_fut_known = extract_future_covariates(batch_y, args)

            outputs = model(
                batch_x,
                batch_x_mark,
                dec_inp,
                batch_y_mark,
                x_fut_known=batch_fut_known,
            )

            batch_y_target = extract_target(batch_y, args)
            p50_pred = outputs[:, :, P50_IDX : P50_IDX + 1]

            preds_p50.append(p50_pred.detach().cpu().numpy())
            trues.append(batch_y_target.detach().cpu().numpy())
            quantile_preds_all.append(outputs.detach().cpu().numpy())

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
        preds_inv = test_data.inverse_transform_target(preds_p50)
        trues_inv = test_data.inverse_transform_target(trues)

        quantile_inv = np.zeros_like(quantile_preds_all)
        for qi in range(N_QUANTILES):
            quantile_inv[:, :, qi] = test_data.inverse_transform_target(quantile_preds_all[:, :, qi])

        np.save(os.path.join(folder_path, "pred_inv.npy"), preds_inv)
        np.save(os.path.join(folder_path, "true_inv.npy"), trues_inv)
        np.save(os.path.join(folder_path, "quantile_preds_inv.npy"), quantile_inv)

        mae_i, mse_i, rmse_i, mape_i, mspe_i = metric(preds_inv, trues_inv)
        print(
            f"[Inverse] P50 Metrics: MSE={mse_i:.6f}, MAE={mae_i:.6f}, "
            f"RMSE={rmse_i:.6f}, MAPE={mape_i:.6f}"
        )

    mae, mse, rmse, mape, mspe = metric(preds_p50, trues)
    print(f"[Scaled] P50 Metrics: MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}")

    return folder_path


def plot_pred_vs_true(results_dir: str, feat_idx: int = 0, out_name: str = "pred_vs_true.png", use_inverse: bool = False):
    """绘制真实值、P50 预测及 P10-P90 区间。"""
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
    ax.plot(pred_series, label="Prediction (P50)", alpha=0.75, color="tab:orange")

    if has_quantiles:
        x_range = range(len(p10_series))
        ax.fill_between(
            x_range,
            p10_series,
            p90_series,
            alpha=0.22,
            color="tab:orange",
            label="P10-P90 Confidence Interval",
        )

    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    if np.isfinite(mape_val):
        ax.set_title(f"TimeXer + ConvNeXt-Tiny Weather Quantile Prediction - MAPE: {100 * mape_val:.2f}%")
    else:
        ax.set_title("TimeXer + ConvNeXt-Tiny Weather Quantile Prediction - MAPE: NaN")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Load (MW)")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, out_name), dpi=600, bbox_inches="tight")
    plt.show()


def predict_future_load_from_csv(
    model,
    args,
    device,
    results_dir: str,
    weather_store: WeatherFeatureStore,
    future_path: str = FUTURE_PATH,
    steps: int = PRED_LEN,
    use_inverse: bool = True,
):
    """基于 future.csv 的未来时间戳与未来气象特征进行预测。"""
    print("\n" + "=" * 72)
    print(f"Future Forecast: from {future_path}")
    print("=" * 72)

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
    except Exception as exc:
        print(f"Load history data failed: {exc}")
        return

    future_df = pd.read_csv(abs_future_path)
    if "date" not in future_df.columns:
        print(f"Future file missing date column: {abs_future_path}")
        return

    future_df["date"] = pd.to_datetime(future_df["date"])
    future_df = future_df.sort_values("date").reset_index(drop=True)

    if len(history_df) < args.seq_len:
        print(f"History length ({len(history_df)}) < seq_len ({args.seq_len}), skip.")
        return

    predict_steps = min(int(steps), len(future_df), args.pred_len)
    if predict_steps <= 0 or predict_steps < args.pred_len:
        print(f"Future rows ({predict_steps}) < pred_len ({args.pred_len}), skip.")
        return

    ref_data, _ = weather_data_provider(args, "train", weather_store)

    hist_dates = pd.to_datetime(history_df["date"].iloc[-args.seq_len :].values)
    future_dates = pd.to_datetime(future_df["date"].iloc[: args.pred_len].values)

    hist_weather = weather_store.align_to_dates(hist_dates)
    future_weather = weather_store.align_to_dates(future_dates)

    hist_load = history_df[args.target].iloc[-args.seq_len :].values.astype(np.float32).reshape(-1, 1)
    hist_features = np.concatenate([hist_weather, hist_load], axis=1).astype(np.float32)
    hist_features = ref_data.scale_full_features(hist_features)

    if args.use_future_covariates:
        future_weather_scaled = ref_data.scale_weather_features(future_weather)
    else:
        future_weather_scaled = None

    x_mark_np = time_features(hist_dates, freq=args.freq).transpose(1, 0).astype(np.float32)

    if args.label_len > 0:
        label_dates = hist_dates[-args.label_len :]
        dec_dates = np.concatenate([label_dates, future_dates], axis=0)
    else:
        dec_dates = future_dates
    y_mark_np = time_features(pd.to_datetime(dec_dates), freq=args.freq).transpose(1, 0).astype(np.float32)

    dec_len = args.label_len + args.pred_len
    dec_inp_np = np.zeros((dec_len, hist_features.shape[-1]), dtype=np.float32)
    if args.label_len > 0:
        dec_inp_np[: args.label_len] = hist_features[-args.label_len :]

    model.eval()
    with torch.no_grad():
        batch_x = torch.as_tensor(hist_features, dtype=torch.float32, device=device).unsqueeze(0)
        batch_x_mark = torch.as_tensor(x_mark_np, dtype=torch.float32, device=device).unsqueeze(0)
        dec_inp = torch.as_tensor(dec_inp_np, dtype=torch.float32, device=device).unsqueeze(0)
        batch_y_mark = torch.as_tensor(y_mark_np, dtype=torch.float32, device=device).unsqueeze(0)

        batch_fut_known = None
        if future_weather_scaled is not None:
            batch_fut_known = torch.as_tensor(
                future_weather_scaled, dtype=torch.float32, device=device
            ).unsqueeze(0)

        outputs = model(
            batch_x,
            batch_x_mark,
            dec_inp,
            batch_y_mark,
            x_fut_known=batch_fut_known,
        )

        quantile_scaled = outputs[0, : args.pred_len, :].detach().cpu().numpy()
        p50_scaled = quantile_scaled[:, P50_IDX]
        p10_scaled = quantile_scaled[:, P10_IDX]
        p90_scaled = quantile_scaled[:, P90_IDX]

    if ref_data.scale and use_inverse:
        preds_p50 = ref_data.inverse_transform_target(p50_scaled)
        preds_p10 = ref_data.inverse_transform_target(p10_scaled)
        preds_p90 = ref_data.inverse_transform_target(p90_scaled)
        history_target = history_df[args.target].values
    else:
        preds_p50 = p50_scaled
        preds_p10 = p10_scaled
        preds_p90 = p90_scaled
        history_target = hist_features[:, -1]

    future_dates = future_df["date"].iloc[: args.pred_len].reset_index(drop=True)
    preds_p50 = preds_p50[: len(future_dates)]
    preds_p10 = preds_p10[: len(future_dates)]
    preds_p90 = preds_p90[: len(future_dates)]
    predict_steps = len(preds_p50)

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
        print(
            f"  {future_dates.iloc[i]}: "
            f"{preds_p10[i]:<12.4f} {preds_p50[i]:<12.4f} {preds_p90[i]:<12.4f}"
        )

    n_history = min(672, len(history_target))
    history_tail = history_target[-n_history:]
    future_x = range(n_history, n_history + predict_steps)

    plt.figure(figsize=(15, 6), facecolor="white")
    plt.plot(range(n_history), history_tail, label="Historical Load", color="tab:blue", alpha=0.8)
    plt.plot(
        future_x,
        preds_p50,
        label="TimeXer + Weather P50 Prediction",
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
    plt.title(f"TimeXer + ConvNeXt-Tiny Future {predict_steps}-Step Load Prediction")
    plt.xlabel("Time Step (15min)")
    plt.ylabel("Load (MW)")
    plt.tight_layout()

    out_fig = os.path.join(results_dir, "future_load_prediction.png")
    plt.savefig(out_fig, dpi=600, bbox_inches="tight")
    plt.show()

    print(f"Saved future prediction csv: {out_csv}")
    print(f"Saved future prediction figure: {out_fig}")


def _get_setting(args, itr: int = 0) -> str:
    """生成实验 setting 字符串。"""
    signature = (
        f"{args.task_name}|{args.model_id}|{args.model}|{args.features}|"
        f"{args.seq_len}|{args.label_len}|{args.pred_len}|{args.d_model}|"
        f"{args.n_heads}|{args.e_layers}|{args.weather_feature_dim}|"
        f"{args.weather_feature_stage}|{int(args.use_future_covariates)}|"
        f"{args.learning_rate}|{args.batch_size}|{args.des}|{itr}"
    )
    short_hash = hashlib.md5(signature.encode("utf-8")).hexdigest()[:8]
    return (
        f"{args.model}_sl{args.seq_len}_pl{args.pred_len}_"
        f"wd{args.weather_feature_dim}_fc{int(args.use_future_covariates)}_"
        f"bs{args.batch_size}_{args.des}_{itr}_{short_hash}"
    )


def main():
    """主函数。"""
    fix_seed = 2026
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    torch.manual_seed(fix_seed)

    args = argparse.Namespace(
        task_name=TASK_NAME,
        is_training=1 if TRAIN_MODE else 0,
        model_id=MODEL_ID,
        model=MODEL,
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
        weather_feature_dim=WEATHER_FEATURE_DIM,
        weather_feature_stage=WEATHER_FEATURE_STAGE,
        weather_cache_dir=WEATHER_CACHE_DIR,
        weather_h5_specs=WEATHER_H5_SPECS,
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

    weather_store = WeatherFeatureStore(
        h5_specs=args.weather_h5_specs,
        cache_dir=args.weather_cache_dir,
        feature_stage=args.weather_feature_stage,
        batch_size=WEATHER_CACHE_BATCH_SIZE,
        use_pretrained=WEATHER_PRETRAINED,
        fill_value=WEATHER_FILL_VALUE,
        force_rebuild=WEATHER_FORCE_REBUILD,
    )

    model = TimeXerWeatherQuantile(args, quantiles=QUANTILES).float().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"TimeXerWeatherQuantile 参数量: {total_params:,}")

    setting = _get_setting(args)

    if TRAIN_MODE:
        print(f"\n>>> 开始训练: {setting}")
        model = train_quantile_model(model, args, device, weather_store)

        print(f"\n>>> 开始测试: {setting}")
        results_dir = test_quantile_model(model, args, device, weather_store)

        plot_pred_vs_true(results_dir, use_inverse=INVERSE_EVAL)
        predict_future_load_from_csv(
            model=model,
            args=args,
            device=device,
            results_dir=results_dir,
            weather_store=weather_store,
            future_path=FUTURE_PATH,
            steps=PRED_LEN,
            use_inverse=INVERSE_EVAL,
        )
    else:
        ckpt_path = os.path.join(args.checkpoints, setting, "checkpoint.pth")
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"已加载模型: {ckpt_path}")
        else:
            raise FileNotFoundError(
                f"未找到模型检查点: {ckpt_path}。请先把 TRAIN_MODE 设为 True 完成训练。"
            )

        print(f"\n>>> 开始测试: {setting}")
        results_dir = test_quantile_model(model, args, device, weather_store)

        plot_pred_vs_true(results_dir, use_inverse=INVERSE_EVAL)
        predict_future_load_from_csv(
            model=model,
            args=args,
            device=device,
            results_dir=results_dir,
            weather_store=weather_store,
            future_path=FUTURE_PATH,
            steps=PRED_LEN,
            use_inverse=INVERSE_EVAL,
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
