"""
湖南省空间网格历史气象数据下载脚本 (CNN 版)
=============================================
从 Open-Meteo Historical Forecast API 下载湖南省指定时间范围的
5 个气象要素 (ECMWF IFS HRES 9km, 小时分辨率)。

输出格式: HDF5 张量 (T, C, H, W)
  T : 时间步 (1小时分辨率，全年约 8,760 步)
  C : 5 个气象变量通道
  H : 69  (纬度格点, 从北到南降序, H=0 对应最北端 ~30.13°N)
  W : 68  (经度格点, 从西到东升序, W=0 对应最西端 ~108.78°E)
"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime
from calendar import monthrange
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import h5py
from scipy.spatial import KDTree

import openmeteo_requests
import requests_cache
from retry_requests import retry

# =============================================================================
# 日志配置
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# 核心配置
# =============================================================================
API_URL        = "https://historical-forecast-api.open-meteo.com/v1/forecast"
MODEL          = "ecmwf_ifs"          # ECMWF IFS HRES 9 km
TIMEZONE       = "Asia/Shanghai"
WIND_SPEED_UNIT = "ms"                # 风速单位: m/s
REQUEST_INTERVAL = 3.0                # 成功请求后的等待间隔 (秒)

# 5 个气象要素 (顺序 = 通道索引)
HOURLY_VARIABLES = [
    "relative_humidity_2m",      # 1.  2m 相对湿度
    "apparent_temperature",      # 2.  体感温度
    "surface_pressure",          # 3.  地面气压
    "wind_speed_10m",            # 4.  10m 风速
    "direct_radiation",          # 5.  直接辐射
]
C = len(HOURLY_VARIABLES)

# 湖南省全域: [south, west, north, east]
# 自由版 API 限制单次请求 <= 1,000 个格点
# 全域 ~4,692 格点 (69×68) 超出限制，因此按纬度切分为 5 条带
# 每条带约 14×68 = 952 格点，在限制内
_S, _W, _N, _E = 24.63, 108.78, 30.13, 114.25
_N_STRIPS = 5
_strip_h  = (_N - _S) / _N_STRIPS  # 每条带纬度跨度 ≈ 1.1°
HUNAN_BBOX_STRIPS = [
    [round(_S + i * _strip_h, 4), _W,
     round(_S + (i + 1) * _strip_h, 4), _E]
    for i in range(_N_STRIPS)
]  # 5 × [south, west, north, east]
HUNAN_BBOX = [_S, _W, _N, _E]  # 仅用于元数据记录

# =============================================================================
# 目标规则网格定义 (用于将 ECMWF 不规则网格重模到规则格局)
#
# ECMWF IFS 实际返回“减少高斯网格”，不同纬度网格点分布不同，不可直接拼连为矩形数组。
# 解决方案：将不规则源点通过 KDTree 最近邓插射到如下定义的规则 0.09° 目标网格。
# 目标网格: H=62, W=62 (共整合湖南省范围)
# =============================================================================
TARGET_LATS = np.arange(_N, _S - 1e-6, -0.09, dtype=np.float32)  # 降序 (N→S), ~62点
TARGET_LONS = np.arange(_W, _E + 1e-6,  0.09, dtype=np.float32)  # 升序 (W→E), ~62点

# 目标年份
YEARS = [2025]

# 输出路径
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))
MONTHLY_DIR  = os.path.join(OUTPUT_DIR, "hunan_grid_monthly")
PROGRESS_FILE = os.path.join(MONTHLY_DIR, "_progress.json")

# =============================================================================
# API 客户端 (带缓存和重试)
# =============================================================================
cache_session  = requests_cache.CachedSession(
    os.path.join(OUTPUT_DIR, ".cache_grid"),
    expire_after=-1,   # 历史数据永不过期
)
retry_session  = retry(cache_session, retries=2, backoff_factor=0.5)
openmeteo      = openmeteo_requests.Client(session=retry_session)


# =============================================================================
# 进度管理 (断点续传)
# =============================================================================

def load_progress() -> Dict:
    """加载下载进度记录。"""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "completed_months": [],
        "grid_shape": None,    # [H, W]，首次成功后保存
        "latitudes":   None,   # list[float]，降序
        "longitudes":  None,   # list[float]，升序
    }


def save_progress(progress: Dict) -> None:
    """保存下载进度记录。"""
    os.makedirs(MONTHLY_DIR, exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


# =============================================================================
# 工具函数
# =============================================================================

def generate_monthly_ranges(years: List[int]) -> List[Tuple[str, str]]:
    """生成各月起止日期列表。"""
    ranges = []
    for year in years:
        for month in range(1, 13):
            start = f"{year}-{month:02d}-01"
            _, last_day = monthrange(year, month)
            end   = f"{year}-{month:02d}-{last_day:02d}"
            ranges.append((start, end))
    return ranges


def handle_rate_limit_prompt(attempt: int) -> None:
    """
    触发速率限制时，暂停程序并提示用户切换 VPN 节点。
    用户按 Enter 后继续。
    """
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  ⚠️  API 速率限制已触发！(第 {attempt + 1} 次)")
    print(sep)
    print("  请执行以下步骤后再继续：")
    print("  1. 切换到一个新的 VPN 节点（建议选亚洲节点，延迟低）")
    print("  2. 等待 5~10 秒，确认 IP 已切换")
    print("     （可在浏览器访问 https://api.ipify.org 验证新 IP）")
    print("  3. 按 Enter 键告知程序继续下载")
    print(sep)
    input("  >>> 已切换 VPN，按 Enter 键继续... ")
    # 清除缓存中的失败记录，确保以新 IP 重新发起请求
    try:
        cache_session.cache.clear()
        logger.info("  已清除请求缓存，将以新 IP 重新发起请求。")
    except Exception:
        pass
    print()


def call_api_with_retry(params: dict) -> Optional[list]:
    """
    调用 Open-Meteo API。
    触发速率限制 (429) 时，提示用户切换 VPN 后重试，最多 15 次。
    其他错误直接返回 None。
    """
    for attempt in range(15):
        try:
            responses = openmeteo.weather_api(API_URL, params=params)
            return responses
        except Exception as e:
            msg = str(e).lower()
            if "limit exceeded" in msg or "too many" in msg or "429" in msg:
                handle_rate_limit_prompt(attempt)
                continue
            else:
                logger.error(f"  API 调用失败 — {e}")
                return None
    logger.error("  多次切换 VPN 重试后仍失败, 放弃本次请求。")
    return None


def fetch_all_strips_parsed(
    start_date:  str,
    end_date:    str,
    base_params: dict,
) -> tuple:
    """
    对 5 个纬度条带分别发起 API 请求。

    重要: 每个条带的 response 对象在下一次 API 调用前必须立即解析成
    Python 原生数据。这是因为 openmeteo_requests 底层的 flatbuffers
    缓冲区会被后续调用覆盖，导致留存的 response 对象指向错误内存。

    返回:
        point_data     : {(lat, lon): ndarray (T_h, C), float32}
        ref_timestamps : ndarray (T_h,), str
        任一条带失败则返回 (None, None)
    """
    all_point_data: dict = {}
    ref_timestamps = None

    for strip_idx, bbox in enumerate(HUNAN_BBOX_STRIPS):
        logger.info(
            f"  条带 {strip_idx + 1}/{len(HUNAN_BBOX_STRIPS)}: "
            f"lat [{bbox[0]}, {bbox[2]}]  lon [{bbox[1]}, {bbox[3]}]"
        )
        params    = {**base_params, "bounding_box": bbox}
        responses = call_api_with_retry(params)
        if responses is None:
            logger.error(f"  条带 {strip_idx + 1} 请求失败，中止本月下载！")
            return None, None

        # === 立即解析，不要留存原始 response 对象 ===
        parsed_count = 0
        for resp in responses:
            try:
                lat = round(float(resp.Latitude()),  2)  # 2 位小数足够区分 ~9km 格点
                lon = round(float(resp.Longitude()), 2)
                hourly = resp.Hourly()

                # 构造时间轴（只需记录一次）
                if ref_timestamps is None:
                    dates = pd.date_range(
                        start=pd.to_datetime(
                            hourly.Time(), unit="s", utc=True
                        ).tz_convert(TIMEZONE),
                        end=pd.to_datetime(
                            hourly.TimeEnd(), unit="s", utc=True
                        ).tz_convert(TIMEZONE),
                        freq=pd.Timedelta(seconds=hourly.Interval()),
                        inclusive="left",
                    )
                    ref_timestamps = (
                        dates.tz_localize(None)
                             .strftime("%Y-%m-%d %H:%M:%S")
                             .values
                    )

                # 提取 C 个变量，锻造为 (T_h, C)
                arr = np.stack(
                    [hourly.Variables(i).ValuesAsNumpy().astype(np.float32)
                     for i in range(C)],
                    axis=0,
                ).T  # (T_h, C)

                all_point_data[(lat, lon)] = arr
                parsed_count += 1
            except Exception as ex:
                logger.warning(f"    格点解析失败: {ex}")

        logger.info(f"    → 成功解析 {parsed_count}/{len(responses)} 个格点")

        # 条带间短暂等待
        if strip_idx < len(HUNAN_BBOX_STRIPS) - 1:
            time.sleep(1.5)

    logger.info(f"  5 个条带合计解析 {len(all_point_data)} 个唯一格点")
    return all_point_data, ref_timestamps


# =============================================================================
# 网格重建
# =============================================================================

def build_grid_from_parsed(
    point_data:     dict,
    ref_timestamps: np.ndarray,
) -> tuple:
    """
    将不规则源格点字典重模到 TARGET_LATS × TARGET_LONS 规则网格。

    ECMWF 减少高斯网格在不同纬度上经度格点不同，会导致
    直接拼挺盘内大量 NaN。本函数用 KDTree 最近邓将任意
    不规则源格点映射到各目标格点，输出干净的 (T_h, C, H, W)。

    返回:
        grid : float32 (T_h, C, H, W)
        lats : float32 (H,) = TARGET_LATS
        lons : float32 (W,) = TARGET_LONS
    """
    T_h = len(ref_timestamps)
    H   = len(TARGET_LATS)
    W   = len(TARGET_LONS)
    logger.info(f"  源格点: {len(point_data)} 个（不规则）")
    logger.info(f"  目标网格: H={H} × W={W}（规则 0.09°，N→S × W→E）")

    # 源格点坐标矩阵 (N_src, 2)
    src_keys  = list(point_data.keys())
    src_coord = np.array(src_keys, dtype=np.float32)          # (N_src, 2): [lat, lon]
    # 为 KDTree 采用经度坐标差异补偿 (先不考虑纳米度转换，省纬度相过即可)
    tree = KDTree(src_coord)

    # 构建目标网格所有 (lat, lon) 对
    tgt_la, tgt_lo = np.meshgrid(TARGET_LATS, TARGET_LONS, indexing="ij")  # (H, W) each
    tgt_coord = np.column_stack([tgt_la.ravel(), tgt_lo.ravel()])          # (H*W, 2)

    _, nn_idx = tree.query(tgt_coord, k=1)   # nearest-neighbor index per target cell

    # 将源数据对正为 (N_src, T_h, C)
    src_data = np.stack(
        [point_data[k] for k in src_keys], axis=0
    )  # (N_src, T_h, C)

    # 射出目标网格 (H*W, T_h, C) 再 reshape
    tgt_data = src_data[nn_idx]              # (H*W, T_h, C)
    tgt_data = tgt_data.reshape(H, W, T_h, C)  # (H, W, T_h, C)
    grid = tgt_data.transpose(2, 3, 0, 1).astype(np.float32)  # (T_h, C, H, W)

    logger.info(f"  KDTree 最近邻重建完成，无 NaN")
    return grid, TARGET_LATS.copy(), TARGET_LONS.copy()


# =============================================================================
# HDF5 读写
# =============================================================================

def save_monthly_h5(
    grid:   np.ndarray,
    ts:     np.ndarray,
    lats:   np.ndarray,
    lons:   np.ndarray,
    filepath: str,
) -> None:
    """
    保存月度小时网格到 HDF5。
    结构:
        /data        float32 (T_h, C, H, W)  gzip 压缩
        /timestamps  bytes   (T_h,)           ISO 8601 字符串
        /latitudes   float32 (H,)             降序
        /longitudes  float32 (W,)             升序
        /variables   bytes   (C,)             通道名称
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    T_h, _, H, W = grid.shape

    with h5py.File(filepath, "w") as f:
        ds = f.create_dataset(
            "data", data=grid, dtype="float32",
            chunks=(min(24, T_h), C, H, W),
            compression="gzip", compression_opts=4,
        )
        ds.attrs["description"] = "(T_hourly, C, H, W): 小时分辨率气象网格"
        ds.attrs["axes"]        = "time | channel | lat(N→S) | lon(W→E)"

        f.create_dataset("timestamps", data=ts.astype("S19"))
        f.create_dataset("latitudes",  data=lats)
        f.create_dataset("longitudes", data=lons)
        f.create_dataset("variables",  data=np.array(HOURLY_VARIABLES, dtype="S64"))

        f.attrs["bounding_box"] = HUNAN_BBOX
        f.attrs["model"]        = MODEL
        f.attrs["created_at"]   = datetime.now().isoformat()

    size_mb = os.path.getsize(filepath) / 1024 / 1024
    logger.info(f"  ✅ 已保存: {os.path.basename(filepath)}  "
                f"({T_h} 小时, {H}×{W} 格点, {size_mb:.1f} MB)")


def validate_monthly_h5(filepath: str, expected_T_h: int) -> bool:
    """
    验证月度 HDF5 文件是否完整有效。
    检查项: 文件存在、可打开、时间步正确、无全 NaN 帧。
    """
    if not os.path.exists(filepath):
        return False
    try:
        with h5py.File(filepath, "r") as f:
            if "data" not in f:
                return False
            shape = f["data"].shape
            if len(shape) != 4 or shape[0] != expected_T_h or shape[1] != C:
                logger.warning(f"  文件形状 {shape} 与期望 ({expected_T_h},{C},H,W) 不符")
                return False
            # 抽检中间一帧是否全为 NaN
            mid = shape[0] // 2
            if np.isnan(f["data"][mid]).all():
                return False
        return True
    except Exception as e:
        logger.warning(f"  验证失败 ({os.path.basename(filepath)}): {e}")
        return False


def merge_monthly_to_yearly(
    year:        int,
    monthly_dir: str,
    output_dir:  str,
) -> None:
    """
    将该年所有月份的小时 HDF5 文件合并为一个年度 HDF5 文件。

    输出:
        hunan_grid_{year}_hourly.h5   (T_h_all, C, H, W)
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  合并 {year} 年所有月份 → 年度 HDF5")
    logger.info(f"{'=' * 60}")

    all_grids  = []
    all_ts     = []
    lats = lons = None

    for month in range(1, 13):
        fp = os.path.join(monthly_dir, f"hunan_grid_{year}-{month:02d}.h5")
        if not os.path.exists(fp):
            logger.warning(f"  ❌ {year}-{month:02d}: 文件不存在，跳过")
            continue
        with h5py.File(fp, "r") as f:
            all_grids.append(f["data"][:])
            all_ts.append(np.array([t.decode() for t in f["timestamps"][:]]))
            if lats is None:
                lats = f["latitudes"][:]
                lons = f["longitudes"][:]
        logger.info(f"  ✅ {year}-{month:02d}: {all_grids[-1].shape}")

    if not all_grids:
        logger.error(f"  {year} 年无有效数据，跳过合并！")
        return

    # 拼接并去重
    grid_all = np.concatenate(all_grids, axis=0)    # (T_h_all, C, H, W)
    ts_all   = np.concatenate(all_ts,    axis=0)
    del all_grids

    _, uniq_idx = np.unique(ts_all, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    grid_all = grid_all[uniq_idx]
    ts_all   = ts_all[uniq_idx]
    T_h, _, H, W = grid_all.shape
    logger.info(f"  拼接完成，去重后: {T_h} 小时, {H}×{W} 格点")

    # ---- 保存年度小时 HDF5 ----
    hourly_path = os.path.join(output_dir, f"hunan_grid_{year}_hourly.h5")
    with h5py.File(hourly_path, "w") as f:
        ds = f.create_dataset(
            "data", data=grid_all, dtype="float32",
            chunks=(24, C, H, W), compression="gzip", compression_opts=4,
        )
        ds.attrs["description"] = "(T_h, C, H, W): 小时分辨率省级网格"
        ds.attrs["axes"]        = "time | channel | lat(N→S) | lon(W→E)"
        f.create_dataset("timestamps", data=ts_all.astype("S19"))
        f.create_dataset("latitudes",  data=lats)
        f.create_dataset("longitudes", data=lons)
        f.create_dataset("variables",  data=np.array(HOURLY_VARIABLES, dtype="S64"))
    size_mb = os.path.getsize(hourly_path) / 1024 / 1024
    logger.info(f"  ✅ 年度小时 HDF5: {os.path.basename(hourly_path)}  ({size_mb:.1f} MB)")


# =============================================================================
# 主流程
# =============================================================================

def run(test_mode: bool = False) -> None:
    """主运行入口。"""
    logger.info("=" * 70)
    logger.info("湖南省空间网格气象数据下载程序 (CNN 版 — HDF5 输出)")
    logger.info(f"模型    : {MODEL} (ECMWF IFS HRES 9km)")
    logger.info(f"变量    : {C} 个 | bounding_box 模式")
    logger.info(f"输出格式: HDF5 张量 (T, {C}, H, W)")
    logger.info(f"目标年份: {YEARS}")
    logger.info("=" * 70)
    print()
    print("  ⚠️  注意: bounding_box 模式每月约消耗 ~10,370 次 API 配额")
    print("  触发速率限制时，程序将暂停并提示您切换 VPN 节点后继续。")
    print()

    # 生成月份列表
    if test_mode:
        years          = YEARS[:1]
        first_year     = years[0]
        monthly_ranges = [(f"{first_year}-01-01", f"{first_year}-01-31")]
        logger.info(f">>> 测试模式: 仅下载 {first_year} 年 1 月")
    else:
        years          = YEARS
        # monthly_ranges = generate_monthly_ranges(years)
        # 手动指定起止日期（只要是 API 支持的格式即可）
        monthly_ranges = [("2025-01-01", "2025-01-31")]

    # 加载进度
    progress  = load_progress()
    completed = set(progress["completed_months"])

    total = len(monthly_ranges)
    done  = sum(1 for s, _ in monthly_ranges if s[:7] in completed)
    logger.info(f"总月份数: {total} | 已完成: {done} | 剩余: {total - done}")
    logger.info("-" * 70)

    # -------- 逐月下载 --------
    for idx, (start_date, end_date) in enumerate(monthly_ranges):
        month_key    = start_date[:7]
        h5_path      = os.path.join(MONTHLY_DIR, f"hunan_grid_{month_key}.h5")
        year, month  = int(month_key[:4]), int(month_key[5:])
        _, last_day  = monthrange(year, month)
        expected_T_h = last_day * 24

        # 断点续传
        if month_key in completed:
            if validate_monthly_h5(h5_path, expected_T_h):
                logger.info(f"\n[{idx+1}/{total}] {month_key} — ✅ 已完成且文件有效，跳过")
                continue
            else:
                logger.warning(
                    f"\n[{idx+1}/{total}] {month_key} — "
                    f"⚠️ 进度已记录但文件缺失/损坏，重新下载"
                )
                progress["completed_months"].remove(month_key)
                completed.discard(month_key)

        logger.info(
            f"\n[{idx+1}/{total}] {month_key}  "
            f"({start_date} ~ {end_date}, 期望 {expected_T_h} 小时)"
        )

        # 调用 API (分 5 个纬度条带，每条带 ~952 格点, 立即解析避免 buffer 复用)
        base_params = {
            "hourly":          HOURLY_VARIABLES,
            "models":          MODEL,
            "timezone":        TIMEZONE,
            "wind_speed_unit": WIND_SPEED_UNIT,
            "start_date":      start_date,
            "end_date":        end_date,
        }
        point_data, ref_timestamps = fetch_all_strips_parsed(
            start_date, end_date, base_params
        )
        if point_data is None:
            logger.error(f"  {month_key}: 下载失败，跳过！下次运行时将重试。")
            time.sleep(REQUEST_INTERVAL)
            continue

        logger.info(f"  开始网格重建 ({len(point_data)} 个格点)...")
        grid, lats, lons = build_grid_from_parsed(point_data, ref_timestamps)

        # 时间步校验（允许 ±1，应对夏令时等边界情况）
        if abs(grid.shape[0] - expected_T_h) > 1:
            logger.warning(
                f"  时间步数 {grid.shape[0]} 与期望 {expected_T_h} 差距较大，"
                f"数据可能不完整，仍保存但请注意核查"
            )

        # 保存月度 HDF5
        save_monthly_h5(grid, ref_timestamps, lats, lons, h5_path)

        # 首次保存网格元数据
        if progress["grid_shape"] is None:
            progress["grid_shape"]  = list(grid.shape[2:])
            progress["latitudes"]   = lats.tolist()
            progress["longitudes"]  = lons.tolist()
            H, W = grid.shape[2], grid.shape[3]
            logger.info(f"  记录网格元数据: H={H}, W={W}")

        # 更新进度
        progress["completed_months"].append(month_key)
        save_progress(progress)
        logger.info(f"  ✅ {month_key} 完成并记录进度")

        time.sleep(REQUEST_INTERVAL)

    # -------- 合并年度 HDF5 --------
    logger.info("\n" + "=" * 70)
    logger.info("月度下载阶段结束，开始合并年度数据...")

    for year in years:
        if test_mode:
            year_months = [f"{year}-01"]
        else:
            year_months = [f"{year}-{m:02d}" for m in range(1, 13)]

        done_months = [m for m in year_months if m in progress["completed_months"]]

        if len(done_months) == len(year_months):
            merge_monthly_to_yearly(year, MONTHLY_DIR, OUTPUT_DIR)
        else:
            logger.warning(
                f"  {year} 年: 仅完成 {len(done_months)}/{len(year_months)} 个月，"
                f"暂不合并。请继续运行本脚本以下载剩余月份。"
            )

    # -------- 完成统计 --------
    logger.info("\n" + "=" * 70)
    total_done = len(progress["completed_months"])
    if total_done >= total:
        logger.info("🎉 全部下载完成！")
    else:
        logger.info(f"本次运行完成 {total_done}/{total} 个月")
        logger.info("请再次运行本脚本继续下载剩余月份。")
    logger.info("=" * 70)


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="湖南省空间网格气象数据下载 (CNN 版 — HDF5 输出)"
    )
    parser.add_argument(
        "--test", action="store_true",
        help="测试模式：仅下载第一个月（快速验证）"
    )
    args = parser.parse_args()
    run(test_mode=args.test)
