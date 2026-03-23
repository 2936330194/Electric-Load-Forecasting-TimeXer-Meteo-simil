"""
weather_e2e.py - 气象特征与负荷端到端预测的数据处理与模型封装模块

该模块负责处理高维 4D 气象网格数据 (时间, 通道, 高度, 宽度)，将其与历史负荷数据对齐，
并提供模型（FullMapConvTimeXerQuantile）将气象图像特征提取和 TimeXer 时间序列预测进行结合，
实现了端到端的概率负荷预测。同时也包含用于大幅优化气象数据内存传输的去重对齐批处理工具。
"""

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Sampler

try:
    import h5py
except ImportError:
    h5py = None

from models.TimeXer import Model as TimeXer
from utils.timefeatures import time_features


def _require_weather_runtime() -> None:
    # 将 h5py 的硬依赖检查延后到运行期，避免仅导入本模块时就失败。
    if h5py is None:
        raise ImportError("Missing h5py. Weather HDF5 loading requires h5py.")


def _guess_year_start_from_path(file_path: str) -> pd.Timestamp:
    # 如果用户没有显式提供起始时间，则尝试从文件名中提取年份作为兜底。
    name = Path(file_path).stem
    match = re.search(r"(20\d{2})", name)
    if match:
        return pd.Timestamp(f"{match.group(1)}-01-01 00:00:00")
    return pd.Timestamp("2025-01-01 00:00:00")


def _find_first_4d_dataset(h5_obj):
    # 在 HDF5 树结构里递归寻找第一个 4D 数据集，
    # 约定其组织形式为 [time, channel, height, width]。
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
    """
    推断气象数据的时间频率。
    通过尝试常见的分钟级别（60, 30, 15 分钟），检查数据总步数是否构成了完整的整数天。

    参数:
        n_steps (int): 数据集的总时间步数
        start_time (pd.Timestamp): 数据集的起始时间（可在此保留作扩展使用）

    返回:
        pd.Timedelta: 推断出的时间增量（频率）
    """
    del start_time
    # 当前只尝试常见的小时级/半小时级/15 分钟级频率。
    candidates_minutes = [60, 30, 15]
    for minutes in candidates_minutes:
        steps_per_day = 1440 // minutes
        n_days = n_steps / steps_per_day
        if abs(n_days - round(n_days)) < 0.01 and n_days >= 1:
            return pd.Timedelta(minutes=minutes)
    raise ValueError(
        f"Unable to infer weather frequency from n_steps={n_steps}. "
        "Provide an explicit frequency in weather_h5_specs."
    )


class FullMapWeatherConvExtractor(nn.Module):
    """
    气象栅格数据的空间特征提取器。
    通过一个覆盖全图的 2D 卷积层（类似于不受限的全局 Patch Embedding），
    将二维物理气象场图像直接降维映射为一维特征向量。
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_height: int,
        kernel_width: int,
        dropout: float = 0.1,
    ):
        """
        初始化全景卷积特征提取器。

        参数:
            in_channels (int): 输入气象特征通道数 (例如: 10)
            out_channels (int): 输出特征的维度 (例如: 3)
            kernel_height (int): 卷积核高度，应严格等于气象网格的垂直分辨率
            kernel_width (int): 卷积核宽度，应严格等于气象网格的水平分辨率
            dropout (float): Dropout 丢弃率
        """
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
        """
        前向传播操作。
        
        参数:
            x (torch.Tensor): 气象输入张量，形状必定为 [Batch, Channels, Height, Width]
            
        返回:
            torch.Tensor: 展平且经过层归一化和激活后的特征，形状为 [Batch, out_channels]
        """
        if x.ndim != 4:
            raise ValueError(f"weather input must be [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[-2] != self.kernel_height or x.shape[-1] != self.kernel_width:
            raise ValueError(
                f"weather frame shape must be ({self.kernel_height}, {self.kernel_width}), "
                f"got ({x.shape[-2]}, {x.shape[-1]})"
            )

        # 整图卷积后空间维会收缩为 1x1，再通过 flatten 得到每个时刻的气象向量表示。
        x = self.full_map_conv(x.float()).flatten(1)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class WeatherGridStore:
    """
    用于高效读取和对齐大规模 4D 气象 HDF5 数据的存储管理器。
    支持管理多个分断的 HDF5 文件源，并能够根据请求的目标时间戳序列，
    通过线性插值处理时间未完全对齐的情况，结合缓存复用实现高吞吐量查询。
    """
    def __init__(
        self,
        h5_specs: Sequence[Tuple],
        expected_in_channels: int,
        fill_value: float = 0.0,
    ):
        """
        初始化 WeatherGridStore。

        参数:
            h5_specs (Sequence[Tuple]): HDF5 气象文件元信息列表。其中每个 Tuple 的格式为
                                        (文件相对/绝对路径, 起始时间字符串, 频率字符串)
            expected_in_channels (int): 验证各文件必须拥有的固定通道个数
            fill_value (float): 若查询的时间戳完全超出已知气象数据范围时将填充的默认空值
        """
        _require_weather_runtime()
        self.expected_in_channels = int(expected_in_channels)
        self.h5_specs = []
        for spec in h5_specs:
            path = os.path.abspath(spec[0])
            start = pd.Timestamp(spec[1]) if spec[1] else _guess_year_start_from_path(spec[0])
            freq = spec[2] if len(spec) > 2 and spec[2] else None
            self.h5_specs.append((path, start, freq))

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

        for h5_path, start_time, explicit_freq in self.h5_specs:
            if not os.path.exists(h5_path):
                print(f"[气象] 未找到文件: {h5_path}")
                continue

            with h5py.File(h5_path, "r") as h5_file:
                dataset = _find_first_4d_dataset(h5_file)
                if dataset is None:
                    raise ValueError(f"No 4D dataset found in {h5_path}")
                if dataset.shape[1] != self.expected_in_channels:
                    raise ValueError(
                        f"{h5_path} channel count mismatch: expected {self.expected_in_channels}, "
                        f"got {dataset.shape[1]}"
                    )

                n_steps, n_channels, height, width = dataset.shape
                dataset_name = dataset.name

            if explicit_freq:
                freq = pd.Timedelta(explicit_freq)
            else:
                freq = _infer_weather_freq(n_steps, start_time)
            # 预先缓存每个源文件的时间轴，后续对齐时直接做 numpy 检索即可。
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
                    f"Inconsistent weather frame shape: {self.frame_shape} vs {(n_channels, height, width)}"
                )

            print(f"[气象] {Path(h5_path).name}: 时间步数={n_steps}, 频率={freq}")

        if not self.sources:
            raise FileNotFoundError("No available weather HDF5 file was found.")

        self.sources.sort(key=lambda x: x["start_ns"])
        start_ts = pd.Timestamp(min(source["start_ns"] for source in self.sources))
        end_ts = pd.Timestamp(max(source["end_ns"] for source in self.sources))
        print(f"[气象] 数据覆盖范围: {start_ts} ~ {end_ts}")

    def _get_dataset(self, source_idx: int):
        if source_idx in self._datasets:
            return self._datasets[source_idx]

        # 延迟打开 HDF5 文件，只在真正需要读取该 source 时建立句柄。
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
        """
        构建目标时间戳序列与底层 HDF5 文件数据时间轴的映射及插值系数对照表。
        此步骤不产生 I/O 读取开销，仅在内存中针对时间轴构建关联。

        参数:
            dates (Sequence[pd.Timestamp]): 模型或数据集需要查询的多条时间戳

        返回:
            Dict[str, np.ndarray]: 字典含有五个 Numpy 1D 对齐数组：
                - source_idx: 时间戳命中的具体源文件序号
                - left_idx: 用于插值的左侧相邻帧的实际文件内部索引
                - right_idx: 用于插值的右侧相邻帧的实际文件内部索引
                - alpha: 偏向右侧的插值权重系数 (完全对齐则为 0.0)
                - valid: 标记该查询时间是否处于数据范围涵盖之内
        """
        dates = pd.DatetimeIndex(pd.to_datetime(dates))
        request_ns = dates.asi8.astype(np.int64)
        n = len(request_ns)

        # 这些数组共同描述一次时间对齐的“执行计划”：
        # 对每个请求时刻，记录它来自哪个 source、左右参考帧是谁、是否有效，以及插值权重。
        source_idx = np.full(n, -1, dtype=np.int32)
        left_idx = np.zeros(n, dtype=np.int32)
        right_idx = np.zeros(n, dtype=np.int32)
        alpha = np.zeros(n, dtype=np.float32)
        valid = np.zeros(n, dtype=bool)

        for idx, source in enumerate(self.sources):
            mask = (request_ns >= source["start_ns"]) & (request_ns <= source["end_ns"])
            if not mask.any():
                continue

            # 在有序时间轴上定位请求时刻对应的位置，用于判断是否精确命中。
            ts_ns = source["timestamps_ns"]
            req = request_ns[mask]
            pos = np.searchsorted(ts_ns, req, side="left")

            # 处理完美对齐的情况（请求时间刚好等于气象帧的时间点）
            pos_clipped = np.clip(pos, 0, len(ts_ns) - 1)
            exact_mask = ts_ns[pos_clipped] == req

            current_indices = np.where(mask)[0]
            exact_indices = current_indices[exact_mask]
            source_idx[exact_indices] = idx
            left_idx[exact_indices] = pos_clipped[exact_mask]
            right_idx[exact_indices] = pos_clipped[exact_mask]
            alpha[exact_indices] = 0.0 # 无需插值
            valid[exact_indices] = True

            # 对未命中的时刻，采用左右两帧之间的时间维线性插值。
            non_exact_indices = current_indices[~exact_mask]
            if len(non_exact_indices) == 0:
                continue

            non_exact_pos = pos[~exact_mask]
            right = np.clip(non_exact_pos, 1, len(ts_ns) - 1)
            left = np.clip(right - 1, 0, len(ts_ns) - 1)
            left_ts = ts_ns[left].astype(np.float64)
            right_ts = ts_ns[right].astype(np.float64)
            req_ts = req[~exact_mask].astype(np.float64)
            # alpha 越接近 1，表示请求时刻越靠近右侧参考帧。
            denom = np.maximum(right_ts - left_ts, 1.0)
            alpha_values = ((req_ts - left_ts) / denom).astype(np.float32)

            source_idx[non_exact_indices] = idx
            left_idx[non_exact_indices] = left
            right_idx[non_exact_indices] = right
            alpha[non_exact_indices] = alpha_values
            valid[non_exact_indices] = True

        if (~valid).any() and not self._warned_out_of_range:
            print(
                f"[气象] 警告: {(~valid).sum()} 个时间戳超出气象数据范围; "
                f"将使用 fill_value={self.fill_value} 进行填充。"
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
        """
        利用已经计算好的配置结构表从磁盘中实际提取数据，并在时间没有完美对齐时应用时间尺度的线性混合插值。
        为了内存友好，可以接受数组切片的起始终止参数。

        参数:
            alignment (Dict): build_alignment 的返回值字典
            start (Optional[int]): batch 或 slice 请求的起始偏移行
            end (Optional[int]): batch 或 slice 请求的截止偏移行

        返回:
            np.ndarray: 对齐后提取出的精确张量，形状为 [N, Channels, Height, Width]
        """
        if self.frame_shape is None:
            raise RuntimeError("frame_shape is not initialized.")

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

        # 按源文件分桶读取，尽量减少 HDF5 的随机跳转和句柄切换。
        for src in np.unique(source_idx[valid]):
            src_mask = valid & (source_idx == src)
            dataset = self._get_dataset(int(src))

            # 先读取左参考帧；如果这批样本都精确命中，就不再读取右帧。
            left_frames = np.asarray(dataset[left_idx[src_mask]], dtype=np.float32)
            alpha_src = alpha[src_mask]

            # allclose 用来兼容浮点权重中的微小误差。
            if np.allclose(alpha_src, 0.0):
                frames[src_mask] = left_frames
                continue

            # 只有确实存在非精确对齐样本时才读取右参考帧。
            right_frames = np.asarray(dataset[right_idx[src_mask]], dtype=np.float32)
            alpha_view = alpha_src.reshape(-1, 1, 1, 1)
            frames[src_mask] = (1.0 - alpha_view) * left_frames + alpha_view * right_frames

        return frames

    def fetch_frames_by_dates(self, dates: Sequence[pd.Timestamp]) -> np.ndarray:
        alignment = self.build_alignment(dates)
        return self.fetch_frames_from_alignment(alignment)


class FullMapConvTimeXerQuantile(nn.Module):
    """
    集成了全图气象卷积特征提取器（FullMapWeatherConvExtractor）与
    TimeXer 时序预测骨干模型的端到端复合模型。

    在分离模式设计下，此模型能够同时消化两个不同序列长度的输入域：
    - 主体内生变量（例如负荷数据）：长度较短 (通常只用历史观测窗口 seq_len)
    - 外部外生变量（例如引入预报的气象）：扩展长度包含了历史加上未来的预测期 (seq_len + pred_len)
    最终向用户输出定义好分位数的预测概率分布序列。
    """
    def __init__(self, configs, quantiles: Sequence[float]):
        """
        初始化端到端模型。

        参数:
            configs: 超参数配置字典类，通常源于 argparse Namespace。需包含时间、卷积和 Transformer等各类参数。
            quantiles (Sequence[float]): 需要预测的分位数列表 (例如 [0.1, 0.5, 0.9])
        """
        super().__init__()
        self.quantiles = list(quantiles)
        self.n_quantiles = len(self.quantiles)
        self.weather_feature_dim = int(configs.weather_feature_dim)
        self.encode_chunk_size = int(getattr(configs, "weather_encode_chunk_size", 512))

        self.weather_backbone = FullMapWeatherConvExtractor(
            in_channels=int(getattr(configs, "weather_in_channels")),
            out_channels=self.weather_feature_dim,
            kernel_height=int(getattr(configs, "weather_kernel_height")),
            kernel_width=int(getattr(configs, "weather_kernel_width")),
            dropout=float(getattr(configs, "dropout", 0.1)),
        )

        # 使用 weather_seq_len（含未来气象）作为外生变量序列长度
        self.weather_seq_len = int(getattr(configs, "weather_seq_len", configs.seq_len))
        # 为 TimeXer 注入 exo_seq_len 以适配扩展的外生变量长度
        configs.exo_seq_len = self.weather_seq_len
        # enc_in 在分离模式下仅表示内生变量（load）数量
        configs.enc_in = 1

        self.timexer = TimeXer(configs)
        self.quantile_head = nn.Linear(1, self.n_quantiles)

        with torch.no_grad():
            # 将分位数头初始化为“在点预测附近做轻微平移”，
            # 这样训练初期各分位数输出不会无约束地发散。
            self.quantile_head.weight.fill_(1.0)
            self.quantile_head.bias.copy_(torch.tensor([q - 0.5 for q in self.quantiles]) * 0.1)

    def _encode_weather_chunk(self, weather_chunk: torch.Tensor) -> torch.Tensor:
        # 单独封装一层，便于后续更换气象编码器实现。
        return self.weather_backbone(weather_chunk)

    def _encode_weather_frames(self, weather_frames: torch.Tensor) -> torch.Tensor:
        if weather_frames.ndim != 4:
            raise ValueError(f"weather_frames must be [N, C, H, W], got {tuple(weather_frames.shape)}")

        flat = weather_frames.float()
        encoded_chunks: List[torch.Tensor] = []
        # 分块过模型以防止大规模气象序列导致显存溢出 (OOM)
        for start in range(0, flat.shape[0], self.encode_chunk_size):
            end = min(start + self.encode_chunk_size, flat.shape[0])
            encoded_chunks.append(self._encode_weather_chunk(flat[start:end]))
        return torch.cat(encoded_chunks, dim=0)

    def _encode_weather_sequence(
        self,
        weather_seq: Optional[torch.Tensor],
        weather_index: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        利用已实例化的全感野卷积层转换大规模（时间长、Batch 大）的序列气象数据。
        提供普通推断法与通过 `weather_index` 处理去重复缓存的方式。

        参数:
            weather_seq: 当使用索引重构时，此张量代表去重后的唯一性张量 [U, C, H, W]（其中 U 是 Unqiue数）。
                         否则的话其应为 [Batch, Time_Len, C, H, W] 的高维时序张矩阵。
            weather_index: 长度为时间步长的坐标轴切片，形状 [Batch, Time_Len]，用来恢复去重前的重复样本映射。

        返回:
            Optional[torch.Tensor]: 编码完毕的气象特质数组维度：[Batch, Time_Len, Weather_Feature_Dim]
        """
        if weather_seq is None:
            return None

        if weather_index is not None:
            if weather_seq.ndim != 4:
                raise ValueError(
                    f"Indexed weather_seq must be [U, C, H, W], got {tuple(weather_seq.shape)}"
                )
            if weather_index.ndim != 2:
                raise ValueError(f"weather_index must be [B, T], got {tuple(weather_index.shape)}")

            batch_size, time_len = weather_index.shape
            # weather_seq 这里只包含去重后的唯一气象帧 [U, C, H, W]。
            # 先编码成 [U, D]，再用 weather_index 恢复到 [B, T, D]。
            encoded_frames = self._encode_weather_frames(weather_seq)
            gathered = encoded_frames.index_select(0, weather_index.reshape(-1))
            return gathered.reshape(batch_size, time_len, self.weather_feature_dim)

        if weather_seq.ndim != 5:
            raise ValueError(f"weather_seq must be [B, T, C, H, W], got {tuple(weather_seq.shape)}")

        bsz, time_len, channels, height, width = weather_seq.shape
        # 普通路径下直接把 batch 和时间维展平，再统一送入卷积编码器。
        flat = weather_seq.reshape(bsz * time_len, channels, height, width)
        encoded = self._encode_weather_frames(flat)
        return encoded.reshape(bsz, time_len, self.weather_feature_dim)

    def forward(
        self,
        load_x: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_exo_mark: torch.Tensor,
        weather_x: torch.Tensor,
        weather_x_index: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播（分离模式）

        将气象特征（768步）作为独立外生变量传入 TimeXer，
        负荷数据（672步）作为内生变量传入。

        参数:
            load_x: [B, seq_len, 1] 负荷序列
            x_mark_enc: [B, seq_len, T] 内生变量时间标记
            x_exo_mark: [B, weather_seq_len, T] 外生变量时间标记
            weather_x: [U, C, H, W] 或 [B, weather_seq_len, C, H, W] 气象帧
            weather_x_index: [B, weather_seq_len] 索引到 weather_x
            mask: 掩码（可选）
        """
        # 先把原始气象场编码成时间序列特征，再交给 TimeXer 作为外生变量。
        weather_feature = self._encode_weather_sequence(weather_x, weather_x_index)

        # 分离模式下，负荷历史与气象外生驱动分别走各自输入通道。
        point_pred = self.timexer(
            load_x,       # [B, seq_len, 1] — 内生变量
            x_mark_enc,   # [B, seq_len, T] — 内生时间标记
            None,
            None,
            mask=mask,
            x_exo=weather_feature,  # [B, weather_seq_len, weather_feature_dim] — 外生变量
            x_exo_mark=x_exo_mark,  # [B, weather_seq_len, T] — 外生时间标记
        )
        point_pred = point_pred[:, -self.timexer.pred_len:, :]
        return self.quantile_head(point_pred)


class LoadWeatherEndToEndDataset(Dataset):
    """
    同时管理时间序列负荷数据及相关天气的混合端到端数据集。
    会自动在内存中截取相应时间范围的数据，并在训练/验证/测试阶段预缓存与之对应时间的全部气象帧数据。
    支持将历史内生负荷序列 (seq_len) 与由独立索引支持的长周期气象变量序列 (weather_seq_len) 对齐生成批次。
    """
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
        """
        初始化端到端负荷与气象混合的数据集。

        参数:
            args: 配置命名空间
            weather_store (WeatherGridStore): 已读取好 HDF5 气象对齐信息的网格管理器，以供抽样天气切片片段
            flag (str): 数据集用途枚举，必须是 "train", "val", "test" 其中之一
            size (Optional[Sequence[int]]): 切片信息，包含 [seq_len, label_len, pred_len]
            target (Optional[str]): CSV里的需要去预测处理的目标列名 (默认依据 args.target)
            scale (bool): 是否对原始负荷数据进行标准化缩放变换 (StandardScaler)
            timeenc (int): 时间戳的特性编码方式（0 - 分散的月份天数类别变量，1 - 连续时间特性的 TimeFeature）
            freq (Optional[str]): pandas 中的时间频率占位单位 (如 '15min')
        """
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
        self.seq_offsets = np.arange(self.seq_len, dtype=np.int64)
        self.target_offsets = (
            np.arange(self.label_len + self.pred_len, dtype=np.int64) + (self.seq_len - self.label_len)
        )

        # weather_seq_len 允许独立配置，必要时可长于负荷历史窗。
        self.weather_seq_len = int(getattr(args, 'weather_seq_len', self.seq_len))
        self.weather_seq_offsets = np.arange(self.weather_seq_len, dtype=np.int64)

        flag_map = {"train": 0, "val": 1, "test": 2}
        if flag not in flag_map:
            raise ValueError(f"flag must be train/val/test, got {flag}")
        self.set_type = flag_map[flag]

        self.scaler: Optional[StandardScaler] = None
        self.target_mean = 0.0
        self.target_scale = 1.0
        self.data_x: Optional[np.ndarray] = None
        self.data_y: Optional[np.ndarray] = None
        self.data_stamp: Optional[np.ndarray] = None
        self.raw_dates: Optional[pd.Series] = None
        self.weather_alignment: Optional[Dict[str, np.ndarray]] = None
        self.weather_cache: Optional[np.ndarray] = None

        self.__read_data__()

    def __read_data__(self) -> None:
        csv_path = os.path.join(self.args.root_path, self.args.data_path)
        df_raw = pd.read_csv(csv_path)
        if "date" not in df_raw.columns:
            raise ValueError(f"Missing date column in {csv_path}")

        df_raw["date"] = pd.to_datetime(df_raw["date"])
        df_raw = df_raw.sort_values("date").reset_index(drop=True)

        if self.target not in df_raw.columns and "Target" in df_raw.columns:
            df_raw = df_raw.rename(columns={"Target": self.target})
        if self.target not in df_raw.columns:
            raise ValueError(f"Missing target column {self.target} in {csv_path}")

        total_len = len(df_raw)
        num_train = int(total_len * 0.6)  # 60% 训练集
        num_test = int(total_len * 0.2)   # 20% 测试集
        num_vali = total_len - num_train - num_test # 20% 验证集

        # 这里不是简单切分行区间，而是额外向前预留 seq_len，
        # 以保证验证/测试集构造滑窗时仍有足够历史长度。
        border1s = [
            0,
            max(0, num_train - self.seq_len),
            max(0, num_train + num_vali - self.seq_len),
        ]
        border2s = [num_train, num_train + num_vali, total_len]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 目前只对负荷目标列做缩放；气象数据保持其原始物理量尺度。
        target_values = df_raw[[self.target]].values.astype(np.float32)
        if self.scale:
            self.scaler = StandardScaler()
            # 严格只在训练段上拟合 scaler，避免未来信息泄漏。
            self.scaler.fit(target_values[: border2s[0]]) # 仅在训练集上 fit
            target_values = self.scaler.transform(target_values).astype(np.float32)
            self.target_mean = float(self.scaler.mean_[0])
            self.target_scale = float(self.scaler.scale_[0]) if self.scaler.scale_[0] != 0 else 1.0

        # 时间特性提取 (Month, Day, Weekday, Hour, Minute)
        df_stamp = df_raw[["date"]].iloc[border1:border2].copy()
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp["date"].apply(lambda row: row.month)
            df_stamp["day"] = df_stamp["date"].apply(lambda row: row.day)
            df_stamp["weekday"] = df_stamp["date"].apply(lambda row: row.weekday())
            df_stamp["hour"] = df_stamp["date"].apply(lambda row: row.hour)
            df_stamp["minute"] = df_stamp["date"].apply(lambda row: row.minute)
            data_stamp = df_stamp.drop(columns=["date"]).values.astype(np.float32)
        else:
            # 使用官方 DataEmbedding 风格的时间特征
            data_stamp = time_features(pd.to_datetime(df_stamp["date"].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0).astype(np.float32)

        self.data_x = target_values[border1:border2]
        self.data_y = target_values[border1:border2]
        self.data_stamp = data_stamp
        self.raw_dates = df_raw["date"].iloc[border1:border2].reset_index(drop=True)
        # 先构造时间对齐计划，再一次性把该 split 对应的气象帧预取到内存。
        self.weather_alignment = self.weather_store.build_alignment(self.raw_dates)

        print(f"[数据集-{self.set_type}] 正在预加载气象帧...")
        t0 = time.time()
        self.weather_cache = self.weather_store.fetch_frames_from_alignment(
            self.weather_alignment, 0, len(self.data_x)
        )
        mem_gb = self.weather_cache.nbytes / (1024 ** 3)
        print(
            f"[数据集-{self.set_type}] 气象缓存已就绪: shape={self.weather_cache.shape}, "
            f"内存={mem_gb:.2f} GB, 耗时={time.time() - t0:.1f}s"
        )
        print(
            f"[数据集-{self.set_type}] 样本行数={len(self.data_x)}, "
            f"负荷维度={self.data_x.shape[-1]}, 气象帧形状={self.weather_store.frame_shape}"
        )

    def __getitem__(self, index: int):
        return int(index)

    def __len__(self) -> int:
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def build_overlap_batch(
        self,
        batch_indices: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """构建批次数据（含扩展气象窗口和外生时间标记）

        返回:
            batch_x: [B, seq_len, 1] 负荷序列
            batch_y: [B, label_len+pred_len, 1] 目标序列
            batch_x_mark: [B, seq_len, T] 内生变量时间标记
            batch_exo_mark: [B, weather_seq_len, T] 外生变量时间标记
            weather_frames: [U, C, H, W] 去重后的气象帧
            weather_index: [B, weather_seq_len] 索引到 weather_frames
        """
        if self.data_x is None or self.data_y is None or self.data_stamp is None or self.weather_cache is None:
            raise RuntimeError("Dataset is not initialized.")

        indices = np.asarray(batch_indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError(f"batch_indices must be a non-empty 1D array, got shape={indices.shape}")

        # 三套位置矩阵分别对应负荷输入窗、监督目标窗、气象外生窗。
        seq_positions = indices[:, None] + self.seq_offsets[None, :]   # 内生序列位置
        target_positions = indices[:, None] + self.target_offsets[None, :]  # 预测目标位置
        weather_positions = indices[:, None] + self.weather_seq_offsets[None, :]  # 气象序列位置

        # 使用高级索引一次性提取所有样本（效率远高于循环提取）
        batch_x = torch.from_numpy(np.ascontiguousarray(self.data_x[seq_positions]))
        batch_y = torch.from_numpy(np.ascontiguousarray(self.data_y[target_positions]))
        batch_x_mark = torch.from_numpy(np.ascontiguousarray(self.data_stamp[seq_positions]))
        # 外生时间标记要与 weather_seq_len 严格对齐，因此单独构造。
        batch_exo_mark = torch.from_numpy(np.ascontiguousarray(self.data_stamp[weather_positions]))

        # --- 气象帧提取优化策略 ---
        # 如果 batch 内索引是连续的，则其气象窗口也高度重叠。
        # 这时直接切一整段连续缓存，再配合相对索引恢复即可，代价最低。
        if len(indices) == 1 or np.all(indices[1:] == indices[:-1] + 1):
            first_index = int(indices[0])
            last_index = int(indices[-1])
            weather_frames = torch.from_numpy(
                np.ascontiguousarray(
                    self.weather_cache[first_index : last_index + self.weather_seq_len]
                )
            )
            # 构建相对索引，让模型知道每个样本/时间步该取连续块中的哪一帧。
            weather_index = torch.from_numpy(
                np.ascontiguousarray(
                    (indices - first_index)[:, None] + self.weather_seq_offsets[None, :]
                )
            )
            return batch_x, batch_y, batch_x_mark, batch_exo_mark, weather_frames, weather_index

        # 非连续 batch 无法整段切片，但仍可通过 unique 去除重复气象帧。
        unique_weather_idx, inverse = np.unique(weather_positions.reshape(-1), return_inverse=True)
        weather_frames = torch.from_numpy(np.ascontiguousarray(self.weather_cache[unique_weather_idx]))
        weather_index = torch.from_numpy(
            np.ascontiguousarray(inverse.reshape(len(indices), self.weather_seq_len))
        )
        return batch_x, batch_y, batch_x_mark, batch_exo_mark, weather_frames, weather_index

    def scale_target(self, data: np.ndarray) -> np.ndarray:
        """根据当前数据集内部维护的 scaler 均值和防差，将输入的数据正向标准化归一。"""
        data = np.asarray(data, dtype=np.float32).reshape(-1, 1)
        if not self.scale or self.scaler is None:
            return data.astype(np.float32)
        return self.scaler.transform(data).astype(np.float32)

    def inverse_transform_target(self, data: np.ndarray) -> np.ndarray:
        """从标准差域内退回至其物理世界实际的单位值，常在验证和图表输出时调用。"""
        data = np.asarray(data, dtype=np.float32)
        if not self.scale:
            return data
        return data * self.target_scale + self.target_mean


class ContiguousWindowBatchSampler(Sampler[List[int]]):
    """
    连续时间窗口批次采样器。
    
    为了优化多重叠的滑窗时间序列提取所产生的冗余大体积气象矩阵重复提取及解码瓶颈，
    强制将批次内部采样的索引限制在严格相连（连续步长）的片段里。
    这样缓存层和模型在合并特征时都可以大幅压缩显存和 IO 开支。
    """
    def __init__(self, dataset_len: int, batch_size: int, drop_last: bool = False):
        if dataset_len <= 0:
            raise ValueError(f"dataset_len must be positive, got {dataset_len}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.dataset_len = int(dataset_len)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)

    def __iter__(self) -> Iterator[List[int]]:
        # 先划分连续块，再打乱块顺序，从而同时满足
        # “batch 内连续”与“epoch 间随机化”这两个目标。
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
    """
    重叠感知批次整理器 (Collate Fn)。
    
    与传统的利用 PyTorch 缺省 stack 方法相比，该配合类能够识别由连续窗口采样器引发的具有
    极度时间重叠性质的气象数据切片请求。并会在进入模型批次计算前，将这些重叠的气象帧执行去重归一提取，
    并把唯一的索引图表返回给模型组合，以此避免同一张气象图像因多个同批次时间片段重复输入 GPU。
    """
    def __init__(self, dataset: LoadWeatherEndToEndDataset):
        self.dataset = dataset

    def __call__(
        self,
        batch: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.dataset.build_overlap_batch(batch)


def weather_data_provider(args, flag: str, weather_store: WeatherGridStore):
    """
    负责生产与分配包含外部长效天气信息端到端的混合数据加载适配器 (Data Provider/Loader)。
    能够自动检测配置属性决定是否开启连续性快速对齐及批重叠去除的并行组合方式 (Overlap/Contiguous Smart Batching)。

    参数:
        args: 通用全局参数
        flag (str): 指定产出目标集种类：'train', 'val' 或是 'test'
        weather_store (WeatherGridStore): 掌管 4D HD5 时空切片的存储池实例的引用

    返回:
        Tuple[LoadWeatherEndToEndDataset, DataLoader]
        由它产出的专属数据集 Dataset 实例以及被安全装载进 DataLoader 打包池里的数据管道处理器。
    """
    timeenc = 0 if args.embed != "timeF" else 1
    shuffle_flag = flag == "train"
    use_contiguous_train_batches = flag == "train" and bool(
        getattr(args, "contiguous_train_batches", False)
    )
    # pin_memory 只在 CUDA 训练时有实际收益。
    use_pin_memory = bool(getattr(args, "pin_memory", False)) and torch.cuda.is_available() and args.use_gpu

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
        # 训练时优先启用连续窗口批采样，以减少气象帧重复搬运。
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
        # 验证/测试保持普通 DataLoader 语义，行为更直接。
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
    "weather_data_provider",
]
