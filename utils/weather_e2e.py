import os
import hashlib
import re
import time
import h5py
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Sampler

from models.TimeXer import Model as TimeXer
from utils.timefeatures import time_features


def _build_similar_day_prior_features(
    load_curves: np.ndarray,
    similarity_scores: Sequence[float],
    pred_len: int,
    top_k: int,
    shift_steps: int = 0,
) -> np.ndarray:
    
    pred_len = int(pred_len)
    top_k = int(top_k)
    shift_steps = int(shift_steps) % max(1, pred_len)

    prior_curves = np.zeros((top_k, pred_len), dtype=np.float32)
    weighted_prior = np.zeros((pred_len,), dtype=np.float32)

    curves = np.asarray(load_curves, dtype=np.float32)
    if curves.ndim == 1:
        curves = curves.reshape(1, -1)

    if curves.ndim == 2 and curves.size > 0:
        usable = min(top_k, curves.shape[0])
        for idx in range(usable):
            curve = curves[idx]
            if curve.shape[0] < pred_len:
                padded = np.zeros((pred_len,), dtype=np.float32)
                padded[: curve.shape[0]] = curve
                curve = padded
            else:
                curve = curve[:pred_len]
            prior_curves[idx] = np.roll(curve.astype(np.float32, copy=False), -shift_steps)

        scores = np.asarray(similarity_scores[:usable], dtype=np.float32)
        if scores.size > 0:
            scores = scores - np.max(scores)
            weights = np.exp(scores).astype(np.float32, copy=False)
            weight_sum = float(np.sum(weights))
            if weight_sum > 0:
                weights = weights / weight_sum
                weighted_prior = np.sum(
                    prior_curves[:usable] * weights[:, None],
                    axis=0,
                    dtype=np.float32,
                )

    return np.concatenate(
        [weighted_prior[:, None], prior_curves.transpose(1, 0)],
        axis=1,
    ).astype(np.float32, copy=False)


def _require_weather_runtime() -> None:
    
    if h5py is None:
        raise ImportError("缺少 h5py 库。气象 HDF5 数据加载需要 h5py。")


def _guess_year_start_from_path(file_path: str) -> pd.Timestamp:
    
    
    match = re.search(r"(20\d{2})", Path(file_path).stem)
    if match:
        return pd.Timestamp(f"{match.group(1)}-01-01 00:00:00")
    return pd.Timestamp("2024-01-01 00:00:00")


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


def _find_named_1d_dataset(h5_obj, keyword: str, expected_len: Optional[int] = None):
    
    keyword = str(keyword).lower()
    for key in h5_obj.keys():
        item = h5_obj[key]
        if isinstance(item, h5py.Dataset) and item.ndim == 1:
            
            if keyword in str(key).lower() and (expected_len is None or len(item) == expected_len):
                return item
        if isinstance(item, h5py.Group):
            found = _find_named_1d_dataset(item, keyword, expected_len)
            if found is not None:
                return found
    return None


def _load_timestamp_index(timestamp_dataset) -> pd.DatetimeIndex:
    
    try:
        
        raw_values = timestamp_dataset.asstr()[...]
    except Exception:
        
        raw_values = timestamp_dataset[...]
    
    normalized = []
    
    for value in np.asarray(raw_values).reshape(-1):
        if isinstance(value, (bytes, bytearray)):
            normalized.append(value.decode("utf-8"))
        else:
            normalized.append(str(value))
            
    timestamps = pd.DatetimeIndex(pd.to_datetime(normalized))
    
    if timestamps.isna().any():
        raise ValueError(f"在数据集 {timestamp_dataset.name} 中发现无效的气象时间戳")
    return timestamps


def _infer_step_from_timestamps(timestamps: pd.DatetimeIndex) -> pd.Timedelta:
    
    if len(timestamps) < 2:
        raise ValueError("推断气象频率至少需要两个时间戳。")
        
    
    diffs = np.diff(timestamps.asi8)
    
    positive_diffs = diffs[diffs > 0]
    if len(positive_diffs) == 0:
        raise ValueError("无法从非递增的时间戳中推断气象频率。")
        
    return pd.Timedelta(int(np.min(positive_diffs)), unit="ns")


def _ensure_timedelta(freq: Any) -> pd.Timedelta:
    
    if isinstance(freq, pd.Timedelta):
        return freq
    if freq is None:
        raise ValueError("频率不能为 None。")
    return pd.Timedelta(freq)


def _normalize_reference_timestamps_ns(reference_timestamps_ns: np.ndarray) -> np.ndarray:
    """
    Normalize concatenated weather timestamps into a sorted unique 1D array.
    This keeps searchsorted-based alignment stable when multiple HDF5 files
    overlap in time.
    """
    reference_timestamps_ns = np.asarray(reference_timestamps_ns, dtype=np.int64)
    reference_timestamps_ns = reference_timestamps_ns.reshape(-1)
    if reference_timestamps_ns.size == 0:
        raise ValueError("reference_timestamps_ns must not be empty.")
    return np.unique(np.sort(reference_timestamps_ns))


def _timedelta_to_freq_str(freq: Any) -> str:
    
    freq = _ensure_timedelta(freq)
    total_seconds = int(freq.total_seconds())
    if total_seconds <= 0:
        raise ValueError(f"频率必须为正数，得到: {freq}。")
        
    
    if total_seconds % 86400 == 0:
        return f"{total_seconds // 86400}d"
    if total_seconds % 3600 == 0:
        return f"{total_seconds // 3600}h"
    if total_seconds % 60 == 0:
        return f"{total_seconds // 60}min"
    return f"{total_seconds}s"


def _take_h5_rows_in_original_order(dataset, indices: np.ndarray) -> np.ndarray:
    
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError(f"索引必须是1维的，得到形状: {indices.shape}")
    if len(indices) == 0:
        return np.empty((0,) + tuple(dataset.shape[1:]), dtype=np.float32)
        
    
    unique_indices, inverse = np.unique(indices, return_inverse=True)
    fetched_unique = np.asarray(dataset[unique_indices], dtype=np.float32)
    
    return fetched_unique[inverse]


def infer_weather_history_len(seq_len: int, load_freq: Any, weather_freq: Any) -> int:
    
    load_freq = _ensure_timedelta(load_freq)
    weather_freq = _ensure_timedelta(weather_freq)
    
    seq_len = int(seq_len)
    if seq_len < 0:
        raise ValueError(f"seq_len must be non-negative, got {seq_len}.")
    if seq_len == 0:
        return 0

    
    load_history_ns = seq_len * int(load_freq.value)
    weather_step_ns = int(weather_freq.value)
    
    history_len, remainder = divmod(load_history_ns, weather_step_ns)
    
    if load_history_ns <= 0 or weather_step_ns <= 0 or remainder != 0 or history_len <= 0:
        raise ValueError(
            "负荷历史时长无法被气象频率整除: "
            f"seq_len={seq_len}, load_freq={load_freq}, weather_freq={weather_freq}"
        )
    return int(history_len)


def build_weather_sequence_timestamps(
    target_start: pd.Timestamp,
    weather_seq_len: int,
    weather_history_len: int,
    weather_freq: Any,
    reference_timestamps_ns: Optional[np.ndarray] = None,
) -> pd.DatetimeIndex:
    
    
    weather_freq = _ensure_timedelta(weather_freq)
    weather_seq_len = int(weather_seq_len)
    weather_history_len = int(weather_history_len)
    
    
    if weather_seq_len <= 0 or weather_history_len <= 0 or weather_seq_len < weather_history_len:
        raise ValueError(
            f"无效的气象窗口配置: seq_len={weather_seq_len}, history_len={weather_history_len}"
        )

    
    
    
    step_ns = int(weather_freq.value)
    anchor_ns = (int(pd.Timestamp(target_start).value) // step_ns) * step_ns
    
    
    offsets = np.arange(weather_seq_len, dtype=np.int64)

    
    if reference_timestamps_ns is None:
        
        start_ns = anchor_ns - weather_history_len * step_ns
        
        return pd.DatetimeIndex(pd.to_datetime(start_ns + offsets * step_ns))

    
    
    reference_timestamps_ns = _normalize_reference_timestamps_ns(reference_timestamps_ns)
    if reference_timestamps_ns.size == 0:
        raise ValueError("reference_timestamps_ns 不能为空。")

    
    anchor_pos = np.searchsorted(reference_timestamps_ns, anchor_ns, side="left")
    
    
    if anchor_pos >= reference_timestamps_ns.size or reference_timestamps_ns[anchor_pos] != anchor_ns:
        raise ValueError(
            "target_start 对齐后的气象锚点未在气象时间轴中找到 (气象观测数据可能不包含此预设定点): "
            f"{pd.Timestamp(anchor_ns)}"
        )

    
    
    requested_positions = anchor_pos - weather_history_len + offsets
    
    
    in_bounds = (requested_positions >= 0) & (requested_positions < reference_timestamps_ns.size)
    requested_ns = np.empty_like(requested_positions, dtype=np.int64)
    
    
    requested_ns[in_bounds] = reference_timestamps_ns[requested_positions[in_bounds]]
    
    
    
    if (~in_bounds).any():
        before = requested_positions < 0
        after = requested_positions >= reference_timestamps_ns.size
        
        
        if before.any():
            requested_ns[before] = reference_timestamps_ns[0] + requested_positions[before] * step_ns
            
        
        if after.any():
            requested_ns[after] = (
                reference_timestamps_ns[-1]
                + (requested_positions[after] - (reference_timestamps_ns.size - 1)) * step_ns
            )
            
    
    return pd.DatetimeIndex(pd.to_datetime(requested_ns))


def _build_weather_window_schedule(
    load_dates: Sequence[pd.Timestamp],
    seq_len: int,
    pred_len: int,
    weather_seq_len: int,
    weather_history_len: int,
    weather_freq: Any,
    reference_timestamps_ns: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    
    
    load_dates = pd.DatetimeIndex(pd.to_datetime(load_dates))
    
    sample_count = len(load_dates) - int(seq_len) - int(pred_len) + 1
    if sample_count <= 0:
        raise ValueError("负荷时间戳不足，无法构建气象窗口。")
        
    weather_freq = _ensure_timedelta(weather_freq)
    step_ns = int(weather_freq.value)
    
    
    
    target_start_ns = load_dates.asi8[int(seq_len) : int(seq_len) + sample_count].astype(np.int64)
    
    anchor_ns = (target_start_ns // step_ns) * step_ns
    
    offsets = np.arange(int(weather_seq_len), dtype=np.int64)

    
    if reference_timestamps_ns is None:
        
        window_start_ns = anchor_ns - int(weather_history_len) * step_ns
        timeline_start_ns = int(window_start_ns.min())
        
        timeline_end_ns = int(window_start_ns.max() + offsets[-1] * step_ns)
        
        
        timeline_ns = np.arange(timeline_start_ns, timeline_end_ns + step_ns, step_ns, dtype=np.int64)
        
        
        # [N, 1] + [1, W_Seq] -> [N, W_Seq]
        window_positions = (
            ((window_start_ns - timeline_start_ns) // step_ns)[:, None] + offsets[None, :]
        ).astype(np.int32)
        return timeline_ns, window_positions

    
    reference_timestamps_ns = _normalize_reference_timestamps_ns(reference_timestamps_ns)
    if reference_timestamps_ns.size == 0:
        raise ValueError("reference_timestamps_ns 不能为空。")

    
    anchor_pos = np.searchsorted(reference_timestamps_ns, anchor_ns, side="left")
    in_bounds = anchor_pos < reference_timestamps_ns.size
    
    
    exact = np.zeros_like(anchor_pos, dtype=bool)
    exact[in_bounds] = reference_timestamps_ns[anchor_pos[in_bounds]] == anchor_ns[in_bounds]
    if not exact.all():
        bad_idx = int(np.flatnonzero(~exact)[0])
        raise ValueError(
            "存在负荷样本的气象锚点无法在气象时间轴中精确找到 (请检查气象 HDF5 是否覆盖了负荷数据的所有区间): "
            f"sample={bad_idx}, anchor={pd.Timestamp(anchor_ns[bad_idx])}"
        )

    
    
    requested_positions = (anchor_pos.astype(np.int64) - int(weather_history_len))[:, None] + offsets[None, :]
    
    
    
    timeline_start_pos = int(requested_positions.min())
    timeline_end_pos = int(requested_positions.max())
    timeline_positions = np.arange(timeline_start_pos, timeline_end_pos + 1, dtype=np.int64)

    
    timeline_ns = np.empty_like(timeline_positions, dtype=np.int64)
    
    valid = (timeline_positions >= 0) & (timeline_positions < reference_timestamps_ns.size)
    timeline_ns[valid] = reference_timestamps_ns[timeline_positions[valid]]
    
    
    if (~valid).any():
        before = timeline_positions < 0
        after = timeline_positions >= reference_timestamps_ns.size
        
        if before.any():
            timeline_ns[before] = reference_timestamps_ns[0] + timeline_positions[before] * step_ns
        
        if after.any():
            timeline_ns[after] = (
                reference_timestamps_ns[-1]
                + (timeline_positions[after] - (reference_timestamps_ns.size - 1)) * step_ns
            )

    
    window_positions = (requested_positions - timeline_start_pos).astype(np.int32)
    return timeline_ns, window_positions



class FullMapWeatherConvExtractor(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, kernel_height: int, kernel_width: int, dropout: float = 0.1):
        
        super().__init__()
        self.kernel_height = int(kernel_height)
        self.kernel_width = int(kernel_width)
        self.output_dim = int(out_channels)
        
        
        self.full_map_conv = nn.Conv2d(
            in_channels=int(in_channels),
            out_channels=int(out_channels),
            kernel_size=(self.kernel_height, self.kernel_width),
            bias=True,
        )
        
        self.norm = nn.LayerNorm(out_channels)
        
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        
        if x.ndim != 4:
            raise ValueError(f"气象输入数据必须是 [B, C, H, W] 格式，当前得到: {tuple(x.shape)}")
        if x.shape[-2] != self.kernel_height or x.shape[-1] != self.kernel_width:
            raise ValueError(
                f"输入的气象帧尺寸必须是 ({self.kernel_height}, {self.kernel_width})，"
                f"当前得到的是 ({x.shape[-2]}, {x.shape[-1]})"
            )
            
        
        
        x = self.full_map_conv(x.float()).flatten(1)
        
        
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x



class WeatherGridStore:
    
    def __init__(
        self,
        h5_specs: Sequence[Tuple],
        expected_in_channels: int,
        fill_value: float = 0.0,
        use_channel_normalization: bool = False,
        log1p_channels: Optional[Sequence[int]] = None,
        normalization_eps: float = 1e-6,
    ):
        
        _require_weather_runtime()
        self.expected_in_channels = int(expected_in_channels)
        self.h5_specs = []
        
        for spec in h5_specs:
            path = os.path.abspath(spec[0])
            
            start = pd.Timestamp(spec[1]) if spec[1] else _guess_year_start_from_path(spec[0])
            freq = spec[2] if len(spec) > 2 and spec[2] else None
            self.h5_specs.append((path, start, freq))
            
        self.fill_value = float(fill_value)
        self.use_channel_normalization = bool(use_channel_normalization)
        self.normalization_eps = float(normalization_eps)
        
        
        raw_log1p_channels = [] if log1p_channels is None else [int(ch) for ch in log1p_channels]
        invalid_channels = [ch for ch in raw_log1p_channels if ch < 0 or ch >= self.expected_in_channels]
        if invalid_channels:
            raise ValueError(
                f"log1p_channels 超出通道范围 (expected={self.expected_in_channels}): {invalid_channels}"
            )
        self.log1p_channels = tuple(sorted(set(raw_log1p_channels)))
        
        
        self.sources: List[Dict[str, object]] = []
        self.frame_shape: Optional[Tuple[int, int, int]] = None
        self._file_handles: Dict[int, Any] = {}
        self._datasets: Dict[int, Any] = {}
        self._warned_out_of_range = False
        
        
        self.native_freq: Optional[pd.Timedelta] = None
        self.native_freq_str: Optional[str] = None
        self.channel_mean: Optional[np.ndarray] = None
        self.channel_std: Optional[np.ndarray] = None
        self._normalization_sample_count = 0
        
        
        self.prepare()

    def prepare(self) -> None:
        
        if self.sources:
            return
            
        for h5_path, start_time, explicit_freq in self.h5_specs:
            if not os.path.exists(h5_path):
                print(f"[weather] 找不到文件: {h5_path}")
                continue
                
            with h5py.File(h5_path, "r") as h5_file:
                
                dataset = _find_first_4d_dataset(h5_file)
                if dataset is None:
                    raise ValueError(f"在 {h5_path} 中未找到 4D 数据集")
                if dataset.shape[1] != self.expected_in_channels:
                    raise ValueError(
                        f"{h5_path} 通道数不匹配: 预期 {self.expected_in_channels}, 实际 {dataset.shape[1]}"
                    )
                
                n_steps, n_channels, height, width = dataset.shape
                dataset_name = dataset.name
                
                
                timestamp_dataset = _find_named_1d_dataset(h5_file, "timestamp", expected_len=n_steps)
                if timestamp_dataset is not None:
                    timestamps = _load_timestamp_index(timestamp_dataset)
                    freq = _infer_step_from_timestamps(timestamps)
                else:
                    
                    if not explicit_freq:
                        raise ValueError(
                            f"{h5_path} 不包含时间戳数据集，必须在配置中显式指定 weather_h5_specs 的频率。"
                        )
                    freq = pd.Timedelta(explicit_freq)
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
                raise ValueError(f"气象帧尺寸不一致: {self.frame_shape} vs {(n_channels, height, width)}")
            
            print(f"[weather] 已加载 {Path(h5_path).name}: steps={n_steps}, freq={freq}")
            
        if not self.sources:
            raise FileNotFoundError("未找到任何有效的气象 HDF5 文件。")
            
        
        self.sources.sort(key=lambda x: x["start_ns"])
        self.native_freq = sorted({source["freq"] for source in self.sources}, key=lambda value: int(value.value))[0]
        self.native_freq_str = _timedelta_to_freq_str(self.native_freq)
        
        start_ts = pd.Timestamp(min(source["start_ns"] for source in self.sources))
        end_ts = pd.Timestamp(max(source["end_ns"] for source in self.sources))
        print(f"[weather] 数据覆盖范围: {start_ts} -> {end_ts}")

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

    def has_fitted_channel_normalization(self) -> bool:
        
        if not self.use_channel_normalization:
            return True
        return self.channel_mean is not None and self.channel_std is not None

    def _apply_log1p_transform_inplace(self, frames: np.ndarray) -> np.ndarray:
        
        if not self.log1p_channels:
            return frames
        for channel in self.log1p_channels:
            
            frames[:, channel, :, :] = np.log1p(np.clip(frames[:, channel, :, :], a_min=0.0, a_max=None))
        return frames

    def _apply_channel_normalization_inplace(self, frames: np.ndarray) -> np.ndarray:
        
        if not self.use_channel_normalization:
            return frames
        if self.channel_mean is None or self.channel_std is None:
            raise RuntimeError(
                "气象通道归一化统计量尚未拟合。请先调用 fit_channel_normalization_xxx。"
            )
        
        frames -= self.channel_mean.reshape(1, -1, 1, 1)
        frames /= self.channel_std.reshape(1, -1, 1, 1)
        return frames

    def preprocess_weather_frames(self, frames: np.ndarray, apply_normalization: bool = True) -> np.ndarray:
        
        frames = np.asarray(frames, dtype=np.float32)
        if frames.ndim != 4:
            raise ValueError(f"气象帧必须是 4D [N, C, H, W]，得到形状: {tuple(frames.shape)}")
            
        
        self._apply_log1p_transform_inplace(frames)
        
        if apply_normalization:
            self._apply_channel_normalization_inplace(frames)
        return frames

    def _accumulate_channel_stats(self, frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        
        frames = np.asarray(frames, dtype=np.float32)
        if frames.ndim != 4:
            raise ValueError(f"气象帧必须是 4D [N, C, H, W]，得到形状: {tuple(frames.shape)}")
            
        if frames.shape[0] == 0:
            zeros = np.zeros((self.expected_in_channels,), dtype=np.float64)
            return zeros, zeros.copy(), 0
            
        
        self._apply_log1p_transform_inplace(frames)
        
        
        channel_sum = frames.sum(axis=(0, 2, 3), dtype=np.float64)
        frames64 = frames.astype(np.float64, copy=False)
        channel_sq_sum = np.sum(frames64 * frames64, axis=(0, 2, 3), dtype=np.float64)
        
        
        element_count = int(frames.shape[0] * frames.shape[2] * frames.shape[3])
        return channel_sum, channel_sq_sum, element_count

    def _finalize_channel_stats(
        self,
        channel_sum: np.ndarray,
        channel_sq_sum: np.ndarray,
        element_count: int,
        sample_count: int,
        stage_name: str,
        elapsed_sec: float,
    ) -> None:
        
        if element_count <= 0:
            raise RuntimeError("拟合失败：未发现有效的数据点。")
            
        mean64 = channel_sum / float(element_count)
        
        var64 = np.maximum(
            channel_sq_sum / float(element_count) - mean64 ** 2,
            self.normalization_eps ** 2,
        )
        std64 = np.sqrt(var64)
        
        
        self.channel_mean = mean64.astype(np.float32)
        self.channel_std = np.maximum(std64, self.normalization_eps).astype(np.float32)
        self._normalization_sample_count = int(sample_count)
        
        print(
            f"[weather] 归一化统计量拟合完成 ({stage_name}): "
            f"样本数={sample_count}, log1p通道={list(self.log1p_channels)}, 耗时={elapsed_sec:.1f}s"
        )

    def fit_channel_normalization_from_dates(
        self,
        dates: Sequence[pd.Timestamp],
        chunk_size: int = 512,
        stage_name: str = "train dates",
    ) -> None:
        
        if not self.use_channel_normalization or self.has_fitted_channel_normalization():
            return
            
        if self.frame_shape is None:
            raise RuntimeError("frame_shape 未初始化，无法拟合。")
            
        
        alignment = self.build_alignment(dates)
        valid = np.asarray(alignment["valid"], dtype=bool)
        valid_count = int(valid.sum())
        
        if valid_count <= 0:
            raise RuntimeError("在气象文件中未找到所请求日期的有效数据。")
            
        channel_sum = np.zeros((self.expected_in_channels,), dtype=np.float64)
        channel_sq_sum = np.zeros((self.expected_in_channels,), dtype=np.float64)
        element_count = 0
        chunk_size = max(1, int(chunk_size))
        
        print(
            f"[weather] 正在从文件拟合归一化参数 ({stage_name}): "
            f"目标点数={len(valid)}, 有效点数={valid_count}, 块大小={chunk_size}"
        )
        
        t0 = time.time()
        
        for start in range(0, len(valid), chunk_size):
            end = min(start + chunk_size, len(valid))
            raw_chunk = self.fetch_raw_frames_from_alignment(alignment, start, end)
            
            
            if not valid[start:end].all():
                raw_chunk = raw_chunk[valid[start:end]]
                
            if raw_chunk.size == 0:
                continue
                
            chunk_sum, chunk_sq_sum, chunk_count = self._accumulate_channel_stats(raw_chunk)
            channel_sum += chunk_sum
            channel_sq_sum += chunk_sq_sum
            element_count += chunk_count
            
        
        self._finalize_channel_stats(
            channel_sum=channel_sum,
            channel_sq_sum=channel_sq_sum,
            element_count=element_count,
            sample_count=valid_count,
            stage_name=stage_name,
            elapsed_sec=time.time() - t0,
        )

    def fit_channel_normalization_from_frames(
        self,
        raw_frames: np.ndarray,
        chunk_size: int = 512,
        stage_name: str = "train cache",
    ) -> None:
        
        if not self.use_channel_normalization or self.has_fitted_channel_normalization():
            return
            
        raw_frames = np.asarray(raw_frames, dtype=np.float32)
        if raw_frames.ndim != 4:
            raise ValueError(f"气象帧必须是 4D [N, C, H, W]，得到形状: {tuple(raw_frames.shape)}")
            
        channel_sum = np.zeros((self.expected_in_channels,), dtype=np.float64)
        channel_sq_sum = np.zeros((self.expected_in_channels,), dtype=np.float64)
        element_count = 0
        chunk_size = max(1, int(chunk_size))
        
        print(
            f"[weather] 正在从内存数据拟合归一化参数 ({stage_name}): "
            f"帧数={raw_frames.shape[0]}, 块大小={chunk_size}"
        )
        
        t0 = time.time()
        for start in range(0, raw_frames.shape[0], chunk_size):
            end = min(start + chunk_size, raw_frames.shape[0])
            
            chunk = np.array(raw_frames[start:end], dtype=np.float32, copy=True)
            chunk_sum, chunk_sq_sum, chunk_count = self._accumulate_channel_stats(chunk)
            channel_sum += chunk_sum
            channel_sq_sum += chunk_sq_sum
            element_count += chunk_count
            
        self._finalize_channel_stats(
            channel_sum=channel_sum,
            channel_sq_sum=channel_sq_sum,
            element_count=element_count,
            sample_count=int(raw_frames.shape[0]),
            stage_name=stage_name,
            elapsed_sec=time.time() - t0,
        )

    def build_alignment(self, dates: Sequence[pd.Timestamp]) -> Dict[str, np.ndarray]:
        
        dates = pd.DatetimeIndex(pd.to_datetime(dates))
        request_ns = dates.asi8.astype(np.int64)
        
        source_idx = np.full(len(request_ns), -1, dtype=np.int32)
        row_idx = np.zeros(len(request_ns), dtype=np.int32)
        valid = np.zeros(len(request_ns), dtype=bool)
        
        for idx, source in enumerate(self.sources):
            
            mask = (~valid) & (request_ns >= source["start_ns"]) & (request_ns <= source["end_ns"])
            if not mask.any():
                continue
                
            ts_ns = source["timestamps_ns"]
            req = request_ns[mask]
            
            pos = np.searchsorted(ts_ns, req, side="left")
            
            in_bounds = pos < len(ts_ns)
            exact = np.zeros_like(pos, dtype=bool)
            
            exact[in_bounds] = ts_ns[pos[in_bounds]] == req[in_bounds]
            
            if not exact.any():
                continue
                
            matched_indices = np.where(mask)[0][exact]
            source_idx[matched_indices] = idx
            row_idx[matched_indices] = pos[exact]
            valid[matched_indices] = True
            
        if (~valid).any() and not self._warned_out_of_range:
            print(
                f"[weather] 警告: 有 {(~valid).sum()} 个请求的气象时间戳未找到，"
                f"将填入默认值: fill_value={self.fill_value}"
            )
            self._warned_out_of_range = True
            
        return {"source_idx": source_idx, "row_idx": row_idx, "valid": valid}

    def fetch_raw_frames_from_alignment(
        self,
        alignment: Dict[str, np.ndarray],
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> np.ndarray:
        
        if self.frame_shape is None:
            raise RuntimeError("frame_shape 未初始化。")
            
        sl = slice(start, end)
        source_idx = alignment["source_idx"][sl]
        row_idx = alignment["row_idx"][sl]
        valid = alignment["valid"][sl]
        
        
        frames = np.full((len(source_idx),) + self.frame_shape, self.fill_value, dtype=np.float32)
        
        if not valid.any():
            return frames
            
        
        for src in np.unique(source_idx[valid]):
            src_mask = valid & (source_idx == src)
            dataset = self._get_dataset(int(src))
            
            frames[src_mask] = _take_h5_rows_in_original_order(dataset, row_idx[src_mask])
            
        return frames

    def fetch_frames_from_alignment(
        self,
        alignment: Dict[str, np.ndarray],
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> np.ndarray:
        
        frames = self.fetch_raw_frames_from_alignment(alignment, start, end)
        return self.preprocess_weather_frames(frames, apply_normalization=True)

    def fetch_raw_frames_by_dates(self, dates: Sequence[pd.Timestamp]) -> np.ndarray:
                return self.fetch_raw_frames_from_alignment(self.build_alignment(dates))

    def fetch_frames_by_dates(self, dates: Sequence[pd.Timestamp]) -> np.ndarray:
                return self.fetch_frames_from_alignment(self.build_alignment(dates))



class FullMapConvTimeXerQuantile(nn.Module):
    
    def __init__(self, configs, quantiles: Sequence[float]):
        
        super().__init__()
        self.quantiles = list(quantiles)
        self.n_quantiles = len(self.quantiles)
        self.weather_feature_dim = int(configs.weather_feature_dim)
        
        self.encode_chunk_size = int(getattr(configs, "weather_encode_chunk_size", 512))
        self.use_similar_day_prior = bool(getattr(configs, "use_similar_day_prior", False))
        self.similar_day_top_k = int(getattr(configs, "similar_day_top_k", 3))
        self.similar_day_prior_dim = self.similar_day_top_k + 1 if self.use_similar_day_prior else 0
        
        
        self.weather_backbone = FullMapWeatherConvExtractor(
            in_channels=int(getattr(configs, "weather_in_channels")),
            out_channels=self.weather_feature_dim,
            kernel_height=int(getattr(configs, "weather_kernel_height")),
            kernel_width=int(getattr(configs, "weather_kernel_width")),
            dropout=float(getattr(configs, "dropout", 0.1)),
        )
        
        self.weather_seq_len = int(getattr(configs, "weather_seq_len", configs.seq_len))
        
        configs.exo_seq_len = self.weather_seq_len
        configs.enc_in = 1 
        
        
        self.timexer = TimeXer(configs)

        if self.use_similar_day_prior:
            fusion_hidden_dim = int(
                getattr(
                    configs,
                    "similar_day_fusion_hidden_dim",
                    max(16, int(getattr(configs, "d_model", 128)) // 4),
                )
            )
            self.similar_day_fusion_head = nn.Sequential(
                nn.Linear(1 + self.similar_day_prior_dim, fusion_hidden_dim),
                nn.GELU(),
                nn.Dropout(float(getattr(configs, "dropout", 0.1))),
                nn.Linear(fusion_hidden_dim, 1),
            )
            with torch.no_grad():
                nn.init.zeros_(self.similar_day_fusion_head[-1].weight)
                nn.init.zeros_(self.similar_day_fusion_head[-1].bias)
        else:
            self.similar_day_fusion_head = None
        
        
        self.quantile_head = nn.Linear(1, self.n_quantiles)
        
        
        with torch.no_grad():
            self.quantile_head.weight.fill_(1.0)
            self.quantile_head.bias.copy_(torch.tensor([q - 0.5 for q in self.quantiles]) * 0.1)

    def _encode_weather_frames(self, weather_frames: torch.Tensor) -> torch.Tensor:
        
        if weather_frames.ndim != 4:
            raise ValueError(f"气象帧维度错误，预期 [N, C, H, W]，实际: {tuple(weather_frames.shape)}")
            
        encoded_chunks: List[torch.Tensor] = []
        for start in range(0, weather_frames.shape[0], self.encode_chunk_size):
            end = min(start + self.encode_chunk_size, weather_frames.shape[0])
            
            encoded_chunks.append(self.weather_backbone(weather_frames[start:end].float()))
        return torch.cat(encoded_chunks, dim=0)

    def _encode_weather_sequence(
        self,
        weather_seq: Optional[torch.Tensor],
        weather_index: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        
        if weather_seq is None:
            return None
            
        
        if weather_index is not None:
            if weather_seq.ndim != 4 or weather_index.ndim != 2:
                raise ValueError("索引模式下，气象输入需为 [U,C,H,W] 且索引需为 [B,T]。")
            batch_size, time_len = weather_index.shape
            
            encoded_frames = self._encode_weather_frames(weather_seq)
            
            gathered = encoded_frames.index_select(0, weather_index.reshape(-1))
            return gathered.reshape(batch_size, time_len, self.weather_feature_dim)
            
        
        if weather_seq.ndim != 5:
            raise ValueError(f"序列模式下，气象输入需为 [B, T, C, H, W]，实际: {tuple(weather_seq.shape)}")
        batch_size, time_len, channels, height, width = weather_seq.shape
        
        flat = weather_seq.reshape(batch_size * time_len, channels, height, width)
        encoded = self._encode_weather_frames(flat)
        return encoded.reshape(batch_size, time_len, self.weather_feature_dim)

    def forward(
        self,
        load_x: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_exo_mark: torch.Tensor,
        weather_x: torch.Tensor,
        weather_x_index: Optional[torch.Tensor] = None,
        similar_day_prior: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        
        weather_feature = self._encode_weather_sequence(weather_x, weather_x_index)
        
        
        point_pred = self.timexer(
            load_x,
            x_mark_enc,
            None,
            None,
            mask=mask,
            x_exo=weather_feature,
            x_exo_mark=x_exo_mark,
        )
        
        
        point_pred = point_pred[:, -self.timexer.pred_len :, :]
        if self.use_similar_day_prior and similar_day_prior is not None:
            if similar_day_prior.ndim != 3:
                raise ValueError(
                    f"similar_day_prior 期望形状为 [B, pred_len, {self.similar_day_prior_dim}]，"
                    f"实际得到 {tuple(similar_day_prior.shape)}"
                )
            if similar_day_prior.shape[1] != self.timexer.pred_len:
                raise ValueError(
                    f"similar_day_prior 的时间长度与 pred_len 不一致: "
                    f"{similar_day_prior.shape[1]} vs {self.timexer.pred_len}"
                )
            if similar_day_prior.shape[2] != self.similar_day_prior_dim:
                raise ValueError(
                    f"similar_day_prior 的特征维度不一致: "
                    f"{similar_day_prior.shape[2]} vs {self.similar_day_prior_dim}"
                )
            fusion_input = torch.cat([point_pred, similar_day_prior.float()], dim=-1)
            point_pred = point_pred + self.similar_day_fusion_head(fusion_input)
        return self.quantile_head(point_pred)



class LoadWeatherEndToEndDataset(Dataset):
    
    def __init__(
        self,
        args,
        weather_store: WeatherGridStore,
        flag: str = "train",
        size: Optional[Sequence[int]] = None,
        target: Optional[str] = None,
        scale: bool = True,
        timeenc: int = 1,
        freq: Optional[str] = None,
    ):
        
        if size is None:
            size = [args.seq_len, args.label_len, args.pred_len]
        self.args = args
        self.seq_len = int(size[0])    
        self.label_len = int(size[1])  
        self.pred_len = int(size[2])   
        self.target = target or args.target
        self.scale = bool(scale)
        self.timeenc = int(timeenc)
        self.freq = freq or args.freq
        self.weather_store = weather_store
        
        self.load_freq = _ensure_timedelta(self.freq)
        self.use_similar_day_prior = bool(getattr(args, "use_similar_day_prior", False))
        self.similar_day_top_k = int(getattr(args, "similar_day_top_k", 3))
        similar_day_artifact_dir = getattr(args, "similar_day_artifact_dir", None)
        self.similar_day_artifact_dir = (
            None if similar_day_artifact_dir in (None, "") else os.path.abspath(str(similar_day_artifact_dir))
        )
        self._resolved_similar_day_artifact_dir: Optional[str] = None
        self.similar_day_prior_cache: Optional[np.ndarray] = None

        
        self.seq_offsets = np.arange(self.seq_len, dtype=np.int64)
        self.target_offsets = (
            np.arange(self.label_len + self.pred_len, dtype=np.int64) + (self.seq_len - self.label_len)
        )

        
        self.weather_seq_len = int(getattr(args, "weather_seq_len", self.seq_len))
        weather_step_override = getattr(args, "weather_step_freq", None)
        if weather_step_override is not None:
            self.weather_freq = _ensure_timedelta(weather_step_override)
            self.weather_freq_str = (
                weather_step_override
                if isinstance(weather_step_override, str)
                else _timedelta_to_freq_str(self.weather_freq)
            )
        elif self.weather_store.native_freq is not None:
            self.weather_freq = self.weather_store.native_freq
            self.weather_freq_str = self.weather_store.native_freq_str or _timedelta_to_freq_str(
                self.weather_freq
            )
        else:
            raise RuntimeError("无法获取气象数据的原始频率。")

        
        default_weather_history_len = infer_weather_history_len(
            seq_len=self.seq_len,
            load_freq=self.load_freq,
            weather_freq=self.weather_freq,
        )
        self.weather_history_len = int(getattr(args, "weather_history_len", default_weather_history_len))
        if self.weather_seq_len <= 0 or self.weather_history_len <= 0 or self.weather_seq_len < self.weather_history_len:
            raise ValueError(
                f"无效的气象窗口配置: seq_len={self.weather_seq_len}, history_len={self.weather_history_len}"
            )
        self.weather_future_len = self.weather_seq_len - self.weather_history_len

        
        weather_mark_override = getattr(args, "weather_mark_freq", None)
        if weather_mark_override is not None:
            self.weather_mark_freq = (
                weather_mark_override
                if isinstance(weather_mark_override, str)
                else _timedelta_to_freq_str(weather_mark_override)
            )
        else:
            self.weather_mark_freq = self.weather_freq_str

        self.use_weather_normalization = bool(
            getattr(args, "use_weather_normalization", self.weather_store.use_channel_normalization)
        ) or self.weather_store.use_channel_normalization
        self.weather_norm_fit_chunk_size = int(getattr(args, "weather_norm_fit_chunk_size", 2048))

        
        flag_map = {"train": 0, "val": 1, "test": 2}
        if flag not in flag_map:
            raise ValueError(f"flag 必须是 train/val/test, 得到: {flag}")
        self.set_type = flag_map[flag]

        
        self.scaler: Optional[StandardScaler] = None
        self.target_mean = 0.0
        self.target_scale = 1.0
        self.data_x: Optional[np.ndarray] = None
        self.data_y: Optional[np.ndarray] = None
        self.data_stamp: Optional[np.ndarray] = None
        self.raw_dates: Optional[pd.Series] = None
        self.weather_timestamps: Optional[pd.DatetimeIndex] = None
        self.weather_stamp: Optional[np.ndarray] = None
        self.weather_window_positions: Optional[np.ndarray] = None
        self.weather_lookup: Optional[Dict[str, np.ndarray]] = None
        self.weather_cache: Optional[np.ndarray] = None

        self.__read_data__()
        if self.use_similar_day_prior:
            self._build_similar_day_prior_cache()

    def __read_data__(self) -> None:
        
        csv_path = os.path.join(self.args.root_path, self.args.data_path)
        df_raw = pd.read_csv(csv_path)
        if "date" not in df_raw.columns:
            raise ValueError(f"CSV 文件 {csv_path} 缺失 date 列")
        df_raw["date"] = pd.to_datetime(df_raw["date"])
        df_raw = df_raw.sort_values("date").reset_index(drop=True)
        
        if self.target not in df_raw.columns and "Target" in df_raw.columns:
            df_raw = df_raw.rename(columns={"Target": self.target})
        if self.target not in df_raw.columns:
            raise ValueError(f"CSV 缺失目标列 {self.target}")

        
        total_len = len(df_raw)
        num_train = int(total_len * 2 / 3)
        num_test = int(total_len * 1 / 6)
        num_vali = total_len - num_train - num_test
        border1s = [0, max(0, num_train - self.seq_len), max(0, num_train + num_vali - self.seq_len)]
        border2s = [num_train, num_train + num_vali, total_len]
        train_dates = pd.DatetimeIndex(df_raw["date"].iloc[: border2s[0]].to_numpy())
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

        
        reference_weather_timestamps_ns = np.concatenate(
            [source["timestamps_ns"] for source in self.weather_store.sources],
            axis=0,
        )
        weather_timeline_ns, weather_window_positions = _build_weather_window_schedule(
            load_dates=self.raw_dates,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            weather_seq_len=self.weather_seq_len,
            weather_history_len=self.weather_history_len,
            weather_freq=self.weather_freq,
            reference_timestamps_ns=reference_weather_timestamps_ns,
        )
        self.weather_timestamps = pd.DatetimeIndex(pd.to_datetime(weather_timeline_ns))
        self.weather_window_positions = weather_window_positions
        
        self.weather_lookup = self.weather_store.build_alignment(self.weather_timestamps)
        
        self.weather_stamp = time_features(
            pd.to_datetime(self.weather_timestamps.values),
            freq=self.weather_mark_freq,
        ).transpose(1, 0).astype(np.float32)

        
        print(f"[dataset-{self.set_type}] 正在预加载气象网格...")
        t0 = time.time()
        if self.use_weather_normalization and not self.weather_store.has_fitted_channel_normalization():
            
            if self.set_type == 0:
                raw_weather_cache = self.weather_store.fetch_raw_frames_from_alignment(
                    self.weather_lookup, 0, len(self.weather_timestamps)
                )
                self.weather_store.fit_channel_normalization_from_frames(
                    raw_weather_cache,
                    chunk_size=self.weather_norm_fit_chunk_size,
                    stage_name="train cache",
                )
                self.weather_cache = self.weather_store.preprocess_weather_frames(
                    raw_weather_cache,
                    apply_normalization=True,
                )
            else:
                
                train_weather_timeline_ns, _ = _build_weather_window_schedule(
                    load_dates=train_dates,
                    seq_len=self.seq_len,
                    pred_len=self.pred_len,
                    weather_seq_len=self.weather_seq_len,
                    weather_history_len=self.weather_history_len,
                    weather_freq=self.weather_freq,
                    reference_timestamps_ns=reference_weather_timestamps_ns,
                )
                self.weather_store.fit_channel_normalization_from_dates(
                    pd.DatetimeIndex(pd.to_datetime(train_weather_timeline_ns)),
                    chunk_size=self.weather_norm_fit_chunk_size,
                    stage_name="train dates",
                )
                self.weather_cache = self.weather_store.fetch_frames_from_alignment(
                    self.weather_lookup, 0, len(self.weather_timestamps)
                )
        else:
            
            self.weather_cache = self.weather_store.fetch_frames_from_alignment(
                self.weather_lookup, 0, len(self.weather_timestamps)
            )

        
        mem_gb = self.weather_cache.nbytes / (1024 ** 3)
        print(
            f"[dataset-{self.set_type}] 气象缓存就绪: 形状={self.weather_cache.shape}, "
            f"占用显存={mem_gb:.2f} GB, 耗时={time.time() - t0:.1f}s"
        )
        print(
            f"[dataset-{self.set_type}] 窗口配置: step={self.weather_freq_str}, "
            f"seq_len={self.weather_seq_len}, 历史={self.weather_history_len}, "
            f"未来={self.weather_future_len}"
        )

    def _resolve_primary_weather_h5_path(self) -> str:
        weather_h5_specs = getattr(self.args, "weather_h5_specs", None)
        if weather_h5_specs:
            return os.path.abspath(str(weather_h5_specs[0][0]))
        if getattr(self.weather_store, "sources", None):
            return os.path.abspath(str(self.weather_store.sources[0]["path"]))
        raise ValueError("无法解析当前数据集绑定的天气 H5 路径。")

    def _resolve_similar_day_artifact_dir(self) -> str:
        if self._resolved_similar_day_artifact_dir is not None:
            return self._resolved_similar_day_artifact_dir
        if self.similar_day_artifact_dir is not None:
            artifact_dir = self.similar_day_artifact_dir
        else:
            try:
                from similar_day_retriever import resolve_retriever_runtime_paths
            except Exception as exc:
                raise ImportError(f"导入 similar_day_retriever 失败，无法自动解析相似日特征库: {exc}") from exc
            weather_h5_path = self._resolve_primary_weather_h5_path()
            _, resolved_artifact_dir, _, _ = resolve_retriever_runtime_paths(weather_h5_path=weather_h5_path)
            artifact_dir = os.path.abspath(str(resolved_artifact_dir))
        self._resolved_similar_day_artifact_dir = artifact_dir
        return artifact_dir

    def _resolve_similar_day_weather_h5_path(
        self,
        saved_weather_h5_path: Optional[str],
    ) -> str:
        runtime_weather_h5 = self._resolve_primary_weather_h5_path()
        if saved_weather_h5_path in (None, ""):
            print(
                f"[dataset-{self.set_type}] similar-day retriever missing weather_h5_path; "
                f"fallback to runtime weather H5: {runtime_weather_h5}"
            )
            return runtime_weather_h5

        saved_weather_h5 = os.path.abspath(str(saved_weather_h5_path))
        if os.path.exists(saved_weather_h5):
            if os.path.normcase(saved_weather_h5) != os.path.normcase(runtime_weather_h5):
                print(
                    f"[dataset-{self.set_type}] similar-day retriever weather_h5 differs from "
                    f"runtime config; keep artifact weather H5: {saved_weather_h5}"
                )
            return saved_weather_h5

        if not os.path.exists(runtime_weather_h5):
            raise FileNotFoundError(
                "similar-day retriever saved weather_h5_path is missing on this device, "
                f"and runtime weather H5 is also missing: saved={saved_weather_h5}, "
                f"runtime={runtime_weather_h5}"
            )

        print(
            f"[dataset-{self.set_type}] similar-day retriever weather_h5 not found locally; "
            f"fallback to runtime weather H5: saved={saved_weather_h5}, "
            f"runtime={runtime_weather_h5}"
        )
        return runtime_weather_h5

    def _get_similar_day_prior_cache_path(self, artifact_dir: str) -> str:
        split_name = {0: "train", 1: "val", 2: "test"}.get(self.set_type, str(self.set_type))
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(repo_root, "cache", "similar_day_prior")
        os.makedirs(cache_dir, exist_ok=True)

        artifact_hash = hashlib.md5(os.path.normcase(artifact_dir).encode("utf-8")).hexdigest()[:12]
        signature = "|".join(
            [
                "exact_query_ts_v1",
                split_name,
                str(self.pred_len),
                str(self.similar_day_top_k),
                artifact_hash,
                str(self.seq_len),
                str(self.label_len),
                str(self.weather_seq_len),
                str(self.weather_history_len),
                str(getattr(self.args, "data_path", "")),
                str(len(self)),
            ]
        )
        cache_key = hashlib.md5(signature.encode("utf-8")).hexdigest()[:16]
        return os.path.join(
            cache_dir,
            f"sd_prior_{split_name}_pl{self.pred_len}_topk{self.similar_day_top_k}_{artifact_hash}_{cache_key}.npy",
        )

    def _build_similar_day_prior_cache(self) -> None:
        sample_count = len(self)
        feature_dim = self.similar_day_top_k + 1
        self.similar_day_prior_cache = np.zeros(
            (sample_count, self.pred_len, feature_dim),
            dtype=np.float32,
        )
        if sample_count <= 0:
            return
        if self.raw_dates is None:
            raise RuntimeError("raw_dates unavailable; cannot build similar-day prior cache.")

        artifact_dir = self._resolve_similar_day_artifact_dir()
        if not os.path.isdir(artifact_dir):
            raise FileNotFoundError(f"similar-day artifact directory does not exist: {artifact_dir}")

        cache_path = self._get_similar_day_prior_cache_path(artifact_dir)
        if os.path.exists(cache_path):
            loaded_cache = np.load(cache_path, allow_pickle=False)
            expected_shape = (sample_count, self.pred_len, feature_dim)
            if loaded_cache.shape == expected_shape:
                self.similar_day_prior_cache = loaded_cache.astype(np.float32, copy=False)
                mem_mb = self.similar_day_prior_cache.nbytes / (1024 ** 2)
                print(
                    f"[dataset-{self.set_type}] loaded similar-day prior cache: {cache_path} | "
                    f"shape={self.similar_day_prior_cache.shape}, mem={mem_mb:.1f} MB"
                )
                return
            print(
                f"[dataset-{self.set_type}] ignore stale similar-day prior cache: {cache_path} | "
                f"expected={expected_shape}, got={loaded_cache.shape}"
            )

        try:
            from similar_day_retriever import HDF5WeatherSequenceStore, SimilarDayRetriever
        except Exception as exc:
            raise ImportError(f"failed to import similar_day_retriever: {exc}") from exc

        retriever = SimilarDayRetriever.load(artifact_dir)
        retriever.weather_h5_path = self._resolve_similar_day_weather_h5_path(retriever.weather_h5_path)
        if retriever.weather_h5_path is None:
            raise RuntimeError("similar-day retriever weather_h5_path is unavailable.")

        query_positions = np.arange(sample_count, dtype=np.int64) + self.seq_len
        query_timestamps = pd.DatetimeIndex(self.raw_dates.iloc[query_positions].to_numpy())
        unique_query_timestamps = pd.DatetimeIndex(pd.unique(query_timestamps)).sort_values()

        print(
            f"[dataset-{self.set_type}] building similar-day priors with exact query timestamps: "
            f"samples={sample_count}, unique_queries={len(unique_query_timestamps)}, top_k={self.similar_day_top_k}"
        )
        print(
            f"[dataset-{self.set_type}] similar-day artifact: {artifact_dir} | "
            f"weather_h5={retriever.weather_h5_path}"
        )
        t0 = time.time()
        retrieval_cache: Dict[int, Dict[str, object]] = {}
        weather_store = HDF5WeatherSequenceStore(retriever.weather_h5_path)
        try:
            total_queries = len(unique_query_timestamps)
            for query_idx, query_ts in enumerate(unique_query_timestamps, start=1):
                result = retriever.search_by_timestamp(
                    query_timestamp=query_ts,
                    top_k=self.similar_day_top_k,
                    weather_store=weather_store,
                    history_end_timestamp_exclusive=query_ts,
                )
                curves = np.asarray(result.load_curves, dtype=np.float32)
                if curves.ndim == 1:
                    curves = curves.reshape(1, -1)
                if curves.ndim == 2 and curves.size > 0:
                    curves = curves[:, : self.pred_len]
                    curves = self.scale_target(curves.reshape(-1, 1)).reshape(curves.shape[0], curves.shape[1])
                else:
                    curves = np.empty((0, self.pred_len), dtype=np.float32)

                retrieval_cache[int(pd.Timestamp(query_ts).value)] = {
                    "curves": curves.astype(np.float32, copy=False),
                    "scores": np.asarray(result.similarity_scores, dtype=np.float32),
                }

                if (
                    query_idx == 1
                    or query_idx == total_queries
                    or query_idx % max(1, total_queries // 8) == 0
                ):
                    print(
                        f"[dataset-{self.set_type}] similar-day retrieval progress "
                        f"{query_idx}/{total_queries}: {pd.Timestamp(query_ts)}"
                    )
        finally:
            weather_store.close()

        for sample_idx, query_ts in enumerate(query_timestamps, start=0):
            cache_item = retrieval_cache.get(int(pd.Timestamp(query_ts).value))
            if cache_item is None:
                continue
            self.similar_day_prior_cache[sample_idx] = _build_similar_day_prior_features(
                load_curves=np.asarray(cache_item["curves"], dtype=np.float32),
                similarity_scores=np.asarray(cache_item["scores"], dtype=np.float32),
                pred_len=self.pred_len,
                top_k=self.similar_day_top_k,
                shift_steps=0,
            )

        np.save(cache_path, self.similar_day_prior_cache)

        mem_mb = self.similar_day_prior_cache.nbytes / (1024 ** 2)
        print(
            f"[dataset-{self.set_type}] similar-day prior cache ready | "
            f"shape={self.similar_day_prior_cache.shape}, mem={mem_mb:.1f} MB, "
            f"time={time.time() - t0:.1f}s, path={cache_path}"
        )

    def __getitem__(self, index: int):
        
        return int(index)

    def __len__(self) -> int:
                return len(self.data_x) - self.seq_len - self.pred_len + 1

    def build_overlap_batch(
        self,
        batch_indices: Sequence[int],
    ) -> Tuple[torch.Tensor, ...]:
        
        if (
            self.data_x is None
            or self.data_y is None
            or self.data_stamp is None
            or self.weather_cache is None
            or self.weather_stamp is None
            or self.weather_window_positions is None
        ):
            raise RuntimeError("数据集尚未正确初始化。")
            
        indices = np.asarray(batch_indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError(f"batch_indices 必须是非空的一维数组，当前形状: {indices.shape}")
            
        
        seq_positions = indices[:, None] + self.seq_offsets[None, :]
        target_positions = indices[:, None] + self.target_offsets[None, :]
        weather_positions = self.weather_window_positions[indices]
        
        batch_x = torch.from_numpy(np.ascontiguousarray(self.data_x[seq_positions]))
        batch_y = torch.from_numpy(np.ascontiguousarray(self.data_y[target_positions]))
        batch_x_mark = torch.from_numpy(np.ascontiguousarray(self.data_stamp[seq_positions]))
        
        
        batch_exo_mark = torch.from_numpy(np.ascontiguousarray(self.weather_stamp[weather_positions]))
        similar_day_prior = None
        if self.use_similar_day_prior:
            if self.similar_day_prior_cache is None:
                raise RuntimeError("similar_day_prior_cache 尚未构建。")
            similar_day_prior = torch.from_numpy(
                np.ascontiguousarray(self.similar_day_prior_cache[indices])
            )
        
        
        
        unique_weather_idx, inverse = np.unique(weather_positions.reshape(-1), return_inverse=True)
        
        weather_frames = torch.from_numpy(np.ascontiguousarray(self.weather_cache[unique_weather_idx]))
        
        weather_index = torch.from_numpy(
            np.ascontiguousarray(inverse.reshape(len(indices), self.weather_seq_len))
        )
        
        if similar_day_prior is not None:
            return (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_exo_mark,
                weather_frames,
                weather_index,
                similar_day_prior,
            )
        return batch_x, batch_y, batch_x_mark, batch_exo_mark, weather_frames, weather_index

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



class ContiguousWindowBatchSampler(Sampler[List[int]]):
    
    def __init__(self, dataset_len: int, batch_size: int, drop_last: bool = False):
        if dataset_len <= 0 or batch_size <= 0:
            raise ValueError(f"无效的采样器参数: dataset_len={dataset_len}, batch_size={batch_size}")
        self.dataset_len = int(dataset_len)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)

    def __iter__(self) -> Iterator[List[int]]:
        
        block_starts = np.arange(0, self.dataset_len, self.batch_size, dtype=np.int64)
        
        np.random.shuffle(block_starts)
        
        for start in block_starts.tolist():
            end = start + self.batch_size
            if end > self.dataset_len:
                if self.drop_last:
                    continue
                end = self.dataset_len
            
            yield list(range(start, end))

    def __len__(self) -> int:
        if self.drop_last:
            return self.dataset_len // self.batch_size
        return (self.dataset_len + self.batch_size - 1) // self.batch_size


class OverlapAwareBatchCollator:
    
    def __init__(self, dataset: LoadWeatherEndToEndDataset):
        self.dataset = dataset

    def __call__(
        self,
        batch: Sequence[int],
    ) -> Tuple[torch.Tensor, ...]:
        
        return self.dataset.build_overlap_batch(batch)



def weather_data_provider(args, flag: str, weather_store: WeatherGridStore):
    
    timeenc = 0 if args.embed != "timeF" else 1
    shuffle_flag = flag == "train" 
    
    
    use_contiguous_train_batches = flag == "train" and bool(getattr(args, "contiguous_train_batches", False))
    
    use_pin_memory = bool(getattr(args, "pin_memory", False)) and torch.cuda.is_available() and bool(getattr(args, "use_gpu", False))
    
    
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
    
    
    collate_fn = OverlapAwareBatchCollator(dataset)
    
    
    if use_contiguous_train_batches:
        
        loader = DataLoader(
            dataset,
            batch_sampler=ContiguousWindowBatchSampler(
                dataset_len=len(dataset),
                batch_size=args.batch_size,
                drop_last=False,
            ),
            num_workers=args.num_workers,
            pin_memory=use_pin_memory,
            collate_fn=collate_fn,
        )
    else:
        
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            pin_memory=use_pin_memory,
            collate_fn=collate_fn,
            drop_last=False,
        )
    return dataset, loader



__all__ = [
    "ContiguousWindowBatchSampler",
    "FullMapConvTimeXerQuantile",
    "FullMapWeatherConvExtractor",
    "LoadWeatherEndToEndDataset",
    "OverlapAwareBatchCollator",
    "WeatherGridStore",
    "_build_similar_day_prior_features",
    "build_weather_sequence_timestamps",
    "infer_weather_history_len",
    "weather_data_provider",
]
