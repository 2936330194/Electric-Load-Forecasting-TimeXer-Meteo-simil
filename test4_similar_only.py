"""
test5_similar_only.py

纯相似日检索基准程序 (Pure similar-day retrieval baseline)
本程序的主要作用是评估“如果完全只依靠相似日检索，不使用任何预测模型（如 TimeXer），准确率能达到多少”。
具体步骤如下：
- 从 ./artifacts/... 目录加载事先构建并保存好的相似日检索库（retriever artifacts）。
- 按照与 test4_smp.py / test5_smp.py 完全相同的数据划分规则，切分出测试集（1/6的数据量）。
- 对于测试集中的每一个预测样本窗口，利用未来一天的气象数据检索出 Top-K（如 3个）最相似的日期。
- 取这 K 个相似日对应的历史负荷历史曲线，通过基于相似度分数的 Softmax 运算生成加权平均曲线，
  并把这条“加权相似日先验曲线”直接作为最终的预测结果。
- 直接在 CSV 存储的原始尺度（native scale）上对预测结果进行 MSE、MAE、R2 等各项指标评估。
"""

import argparse
import os
import random
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import test4_base as base
from similar_day_retriever import HDF5WeatherSequenceStore, SimilarDayRetriever
from utils.forecast_visualization import plot_pred_vs_true
from utils.metrics import cal_eval

# 预设的相似日检索库保存路径（包含之前保存的 PCA 模型、FAISS 索引库、天气降维特征等）
ARTIFACT_DIR = "./artifacts/similar_day_retriever_hunan_grid_2024_2025_filtered_15min"
# 查询 Top-K 数量（检索最相似的3天）
TOP_K = 3
# 预测结果和评测指标的默认保存根目录
RESULTS_ROOT = "./results/"


def _parse_args() -> argparse.Namespace:
    """
    解析命令行参数。可以通过命令行运行 python test5_similar_only.py --top-k 5 来改变默认行为。
    """
    parser = argparse.ArgumentParser(
        description="Evaluate pure weighted similar-day prior on the test split. (在测试集上评估纯相似日加权先验预测的效果)"
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=ARTIFACT_DIR,
        help="Path to the saved similar-day retriever artifact directory. (相似日检索库所在目录的路径)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Top-K retrieved similar days used for the weighted prior. (用于加权平均的前 K 个相似日数量)",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default=RESULTS_ROOT,
        help="Root directory for saving evaluation outputs. (保存评测结果的根目录)",
    )
    return parser.parse_args()


def _load_test_split(
    root_path: str,
    data_path: str,
    target: str,
    seq_len: int,
    pred_len: int,
) -> Tuple[pd.DatetimeIndex, np.ndarray, str]:
    """
    加载负荷数据 CSV 文件，并按照与预测模型相同的规则切分出测试集（Test Split）。
    
    参数:
        root_path: 数据集所在的根目录
        data_path: 数据集 CSV 文件名
        target: 要预测的目标列名 (通常是 'load')
        seq_len: 模型历史回溯长度 (预测模型用，这里仅为了保持对齐切分边界)
        pred_len: 未来预测长度 (通常是 96)
        
    返回:
        split_dates: 测试集包含的具体时间戳索引对象
        split_target: 测试集的真实负荷数据二维数组 (shape: [测试集总长度, 1])
        scale_note: 对数据集数值尺度的简单说明（提示是否好像已被归一化到 0-1 之间）
    """
    csv_path = os.path.join(root_path, data_path)
    df_raw = pd.read_csv(csv_path)
    
    # 基本的数据格式检验和处理
    if "date" not in df_raw.columns:
        raise ValueError(f"CSV file missing 'date' column: {csv_path} (CSV文件中缺少日期时间 'date' 列)")
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    # 按照时间从小到大进行排序，保证时序正确
    df_raw = df_raw.sort_values("date").reset_index(drop=True)

    # 处理如果列名叫 'Target' 而不是实际 target 的情况
    if target not in df_raw.columns and "Target" in df_raw.columns:
        df_raw = df_raw.rename(columns={"Target": target})
    if target not in df_raw.columns:
        raise ValueError(f"CSV file missing target column '{target}': {csv_path} (CSV 文件中缺少需要预测的数据列)")

    # ---------- 核心测试集边界切分逻辑（必须与端到端训练的 Dataset 切分规则100%一致）----------
    total_len = len(df_raw)
    num_train = int(total_len * 2 / 3)      # 前 2/3 用作训练集
    num_test = int(total_len * 1 / 6)       # 后 1/6 用作测试集
    num_vali = total_len - num_train - num_test  # 中间 1/6 用作验证集
    
    # 这里的边界定义与 TimesNet/TimeXer 框架的标准 Dataset 定义方法对齐
    border1s = [0, max(0, num_train - seq_len), max(0, num_train + num_vali - seq_len)]
    border2s = [num_train, num_train + num_vali, total_len]
    
    # 第 2 组 (索引2) 即为测试集边界
    border1 = border1s[2]
    border2 = border2s[2]

    # 根据边界提取测试集上的时间戳和负荷真值
    split_dates = pd.DatetimeIndex(df_raw["date"].iloc[border1:border2].to_numpy())
    split_target = df_raw[[target]].values.astype(np.float32)[border1:border2]
    
    # 对当前数据的取值范围做个简单扫描，用于在日志中进行警示
    # 以防用户用被归一化到 0-1 之间的数据去强行做评估（实际生产通常用 MW 为单位量纲的数据）
    source_min = float(df_raw[target].min())
    source_max = float(df_raw[target].max())
    scale_note = "csv_native"
    if source_min >= -1e-6 and source_max <= 1.000001:
        scale_note = "csv_native_likely_normalized_0_1"

    # 计算在切分后的这部分测试集序列中，究竟能提取出多少个完整的“输入->预测”滑动窗口样本
    sample_count = len(split_target) - int(seq_len) - int(pred_len) + 1
    if sample_count <= 0:
        raise ValueError(
            f"Test split is too short for seq_len={seq_len}, pred_len={pred_len}: "
            f"len(split)={len(split_target)} (测试集数据量太短，无法涵盖哪怕一个完整的“历史+未来”预测窗口)"
        )
    return split_dates, split_target, scale_note


def _build_weighted_prior_curve(
    load_curves: np.ndarray,
    similarity_scores: np.ndarray,
    pred_len: int,
    top_k: int,
    shift_steps: int,
) -> np.ndarray:
    """
    根据给定的多条检索出的相似日负荷曲线，基于相似度分值进行 Softmax 权重加权，构建出一条加权平均先验曲线。
    
    参数:
        load_curves: 查询得到的 top_k 条相似日负荷原始曲线矩阵
        similarity_scores: 对应这 top_k 条负荷曲线的（非归一化）检索相似度得分
        pred_len: 需要预测的点数（例如预测未来 24 小时，15 分钟间隔则是 96）
        top_k: 指定想要合并的曲线数量上限
        shift_steps: 时不变性校正平移步数（例如，当前检索点是 14:00，但相似日的锚点是当天 00:00，这就需要将曲线做步数偏移）
        
    返回:
        np.ndarray: [pred_len] 形状的一维浮点数组，即融合完毕并对齐的先验负荷基准预测
    """
    pred_len = int(pred_len)
    top_k = int(top_k)
    # 取预定的平移步数除以 pred_len 取余，保证循环移位落在合法长度内
    shift_steps = int(shift_steps) % max(1, pred_len)

    curves = np.asarray(load_curves, dtype=np.float32)
    # 特殊容错：如果输入变成了一维对象，将其转为二维 [1, N]
    if curves.ndim == 1:
        curves = curves.reshape(1, -1)

    # 初始化盛放对齐好并且切裁成一致长度尺寸后的历史负荷曲线容器 [Top-K, 预测长度]
    aligned_curves = np.zeros((top_k, pred_len), dtype=np.float32)
    if curves.ndim == 2 and curves.size > 0:
        # 获取实际能用的相似日数量（有些时候检索结果可能少于期望的 top_k）
        usable = min(top_k, curves.shape[0])
        for idx in range(usable):
            curve = curves[idx]
            # 若取得的该条曲线长度竟然比需求还短，则用全0填充垫后
            if curve.shape[0] < pred_len:
                padded = np.zeros((pred_len,), dtype=np.float32)
                padded[: curve.shape[0]] = curve
                curve = padded
            else:
                # 够长就截断获取前面的 pred_len 长度
                curve = curve[:pred_len]
            # 对按日提取处的曲线使用 np.roll 执行平移操作以确保日内时间的绝对位置精确对齐
            aligned_curves[idx] = np.roll(curve.astype(np.float32, copy=False), -shift_steps)

        # -----------------------------
        # 计算每条不同相似度负荷曲线融合时的重要性权重（Softmax）
        # -----------------------------
        scores = np.asarray(similarity_scores[:usable], dtype=np.float32)
        if scores.size > 0:
            # 减掉最大的 score，保证 exp() 指数运算不会数值上溢报 NaN
            scores = scores - np.max(scores)
            weights = np.exp(scores).astype(np.float32, copy=False)
            weight_sum = float(np.sum(weights))
            if weight_sum > 0:
                # 根据 Softmax 公式归一化得到各曲线上乘的比重系数
                weights = weights / weight_sum
                # 通过 weights 进行点乘再累加，算出各点融合的最终一条合成数值线
                return np.sum(
                    aligned_curves[:usable] * weights[:, None],
                    axis=0,
                    dtype=np.float32,
                )

    # 兜底情况（极其罕见）：未能检索到任何有意义的曲线时提供全部为 0 点的无效预测基准
    return np.zeros((pred_len,), dtype=np.float32)


def _evaluate_weighted_prior(
    split_dates: pd.DatetimeIndex,
    split_target: np.ndarray,
    retriever: SimilarDayRetriever,
    top_k: int,
    seq_len: int,
    pred_len: int,
    load_freq: str,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    遍历测试集中的全部滑动窗口进行纯相似日的检索和预测，以完全复现数据集中进行批次评测的环境体验与流程。
    
    这个函数设计上：
    先使用批量的去重思想，按“天”去收集每个测试样本所需的检索库请求减缓大量重复开通 HDF5 所引发的 IO 瘫痪。
    接着利用事先缓存下来的信息对预测任务中特定的滑动起点做进一步偏移动态微调对齐。
    最后，返回所有时间步上的所有组预测和真值以及基于这些对的测评结果。
    """
    sample_count = len(split_target) - int(seq_len) - int(pred_len) + 1
    # 模拟在训练代码中由 DataLoader 生成的实际访问位置编号
    query_positions = np.arange(sample_count, dtype=np.int64) + int(seq_len)
    # 取出每个滑窗样本需要执行“预测”命令的时间戳点集（即当前正站在哪个时刻做规划）
    query_timestamps = pd.DatetimeIndex(split_dates[query_positions])
    # 由于检索系统是按整天 (Anchor Day) 作为单位在底库中索引，这里去时去分归一化获得这一天0点的日期戳
    query_anchor_days = query_timestamps.normalize()

    # 准备承载预测输出和真相数据的空张量
    preds = np.zeros((sample_count, pred_len, 1), dtype=np.float32)
    trues = np.zeros((sample_count, pred_len, 1), dtype=np.float32)
    # 获得数据步长的纳秒尺寸（用于时间计算偏差补齐）
    freq_ns = int(pd.Timedelta(load_freq).value)

    # 去重缓存设计：把同一天之内的连续查询窗口归拢到一天进行一次大查询，能节省90%以上检索开支
    anchor_cache: Dict[int, Dict[str, object]] = {}
    unique_anchor_days = pd.DatetimeIndex(pd.unique(query_anchor_days)).sort_values()
    print(
        f"[similar-only] test samples={sample_count}, unique anchor days={len(unique_anchor_days)}, top_k={top_k}"
        f"\n（将共有 {sample_count} 个按滑动步进构成的预测需求汇总合并成去重的 {len(unique_anchor_days)} 次整日基座气象索引搜寻）"
    )

    # --------- 第一阶段：执行全量底层气象检索并填充分天缓存表 -------------
    # 初始化获取气象数据的读取管道句柄
    weather_store = HDF5WeatherSequenceStore(retriever.weather_h5_path)
    try:
        total_anchors = len(unique_anchor_days)
        t0 = time.time()
        for anchor_idx, anchor_day in enumerate(unique_anchor_days, start=1):
            # 将归零时刻的天作为待规划气象特征提取基点，在检索树中利用 FAISS 完成在线查询最近邻的历史天气
            result = retriever.search_by_timestamp(
                query_timestamp=anchor_day,
                top_k=int(top_k),
                weather_store=weather_store,
                # 排除自身日期防止检索到作弊解（信息泄漏），从而满足回环检测上的理论完全隔离原则
                history_end_timestamp_exclusive=anchor_day,
            )
            # 储进内存中用于之后的批量使用
            anchor_cache[int(anchor_day.value)] = {
                "query_timestamp": pd.Timestamp(result.query_timestamp),  # 一般是传入进来的按天锚点
                "load_curves": np.asarray(result.load_curves, dtype=np.float32), 
                "scores": np.asarray(result.similarity_scores, dtype=np.float32),
            }
            # 过程日志轮循播报
            if (
                anchor_idx == 1
                or anchor_idx == total_anchors
                or anchor_idx % max(1, total_anchors // 8) == 0
            ):
                print(
                    f"[similar-only] retrieval progress {anchor_idx}/{total_anchors}: "
                    f"{pd.Timestamp(anchor_day)} (相似日在线特征提取进程持续进行中...)"
                )
        print(f"[similar-only] retrieval cache ready in {time.time() - t0:.1f}s (共耗时完成缓存构建)")
    finally:
        # 主动关闭 HDF5 解指免除长期独占进程系统内核导致的 IO 污染
        weather_store.close()

    # --------- 第二阶段：提取真值并为各样本计算对齐处理后的最终先验 ----------
    meta_rows = []
    # 按照滑动产生的每一个任务，依次组装预测解与真知用于打分对比
    for sample_idx, (query_ts, anchor_day) in enumerate(zip(query_timestamps, query_anchor_days)):
        # 寻找这一批该天共用的底层寻回信息
        cache_item = anchor_cache.get(int(anchor_day.value))
        if cache_item is None:
            raise RuntimeError(f"Missing retrieval cache for anchor day: {anchor_day} (缓存发生非正常丢失)")

        anchor_start_ts = pd.Timestamp(cache_item["query_timestamp"])
        
        # 精密计算由于查询时间和搜索锚定天的差距，产生的数据切片起始偏差步长
        shift_steps = int((pd.Timestamp(query_ts).value - anchor_start_ts.value) // freq_ns) % int(pred_len)
        
        # 将上面取得的一束负荷曲线调用统一接口软性打分拼接并裁剪和转圈平移
        weighted_prior = _build_weighted_prior_curve(
            load_curves=np.asarray(cache_item["load_curves"], dtype=np.float32),
            similarity_scores=np.asarray(cache_item["scores"], dtype=np.float32),
            pred_len=pred_len,
            top_k=top_k,
            shift_steps=shift_steps,
        )
        # 从验证集真实池中剥离出属于该未来片段的正解序列
        true_curve = split_target[sample_idx + seq_len : sample_idx + seq_len + pred_len, 0]

        preds[sample_idx, :, 0] = weighted_prior
        trues[sample_idx, :, 0] = true_curve.astype(np.float32, copy=False)
        # 用作之后分析对各批详细特性的留存记录
        meta_rows.append(
            {
                "sample_idx": sample_idx,
                "query_timestamp": pd.Timestamp(query_ts),
                "anchor_day": pd.Timestamp(anchor_day),
                "shift_steps": shift_steps,
            }
        )

    # 聚合得出所有评测指标对象 MSE, MAE, R2, CORR 及 MAPE
    metrics_df = cal_eval(y_real=trues.reshape(-1), y_pred=preds.reshape(-1))
    print("[origin Eval] metrics:")
    print(metrics_df)
    # 将记录字典转换为可方便分析的结构化 Pandas 表格
    meta_df = pd.DataFrame(meta_rows)
    return preds, trues, metrics_df, meta_df


def main() -> None:
    # 强制固定整个程序伪随机数流的执行种子，确保科学化分析下一切比较验证测试具备高度再现能力
    fix_seed = 2026
    random.seed(fix_seed)
    np.random.seed(fix_seed)

    args = _parse_args()
    artifact_dir = os.path.abspath(args.artifact_dir)
    # 检测检索根目录正确无误
    if not os.path.isdir(artifact_dir):
        raise FileNotFoundError(f"Artifact directory not found: {artifact_dir} (无法找到构建出的底模型缓存体)")

    # 执行统一的数据裁剪规则调用，这里与训练程序完全保证了引用基础参数同构
    split_dates, split_target, scale_note = _load_test_split(
        root_path=base.ROOT_PATH,
        data_path=base.DATA_PATH,
        target=base.TARGET,
        seq_len=base.SEQ_LEN,
        pred_len=base.PRED_LEN,
    )
    
    # 实例化检索主操控器对象，加载各种包括均值、降维编码器、查询引擎等持久化好的模块
    retriever = SimilarDayRetriever.load(artifact_dir)
    if retriever.weather_h5_path is None:
        raise RuntimeError("Retriever artifact missing weather_h5_path. (不完整的检索体系文件没有正确指明绑定的高维HDF5路径)")

    # 核心测试及评价流线运作环节
    preds, trues, metrics_df, meta_df = _evaluate_weighted_prior(
        split_dates=split_dates,
        split_target=split_target,
        retriever=retriever,
        top_k=args.top_k,
        seq_len=base.SEQ_LEN,
        pred_len=base.PRED_LEN,
        load_freq=base.LOAD_FREQ,
    )

    # 把这次计算和构建的图表，以专有的名字放在 results 工作分析台上
    artifact_name = os.path.basename(os.path.normpath(artifact_dir))
    results_dir = os.path.join(args.results_root, f"test5_similar_only_{artifact_name}_top{int(args.top_k)}")
    os.makedirs(results_dir, exist_ok=True)

    # 持久化输出测试的所有数值和属性以方便外部独立读取，如用来分析与 Timexer 模型互作叠加关系时绘制各种图表
    np.save(os.path.join(results_dir, "pred.npy"), preds)
    np.save(os.path.join(results_dir, "true.npy"), trues)
    metrics_df.to_csv(os.path.join(results_dir, "metrics.csv"), encoding="utf-8-sig")
    meta_df.to_csv(os.path.join(results_dir, "sample_meta.csv"), index=False, encoding="utf-8-sig")
    plot_pred_vs_true(
        results_dir=results_dir,
        use_inverse=False,
        quantiles=None,
        title_prefix="Similar-Day Weighted Prior Prediction",
        y_label="Load",
        out_name="pred_vs_true.png",
    )

    print("\n" + "=" * 72)
    print("Pure weighted-prior evaluation finished (测试评估整体运作全部结束)")
    print(f"artifact_dir: {artifact_dir}")
    print(f"results_dir:  {os.path.abspath(results_dir)}")
    print(f"target_scale: {scale_note}")
    # 分页或排版打印所汇总评估表格的全面成果，向研究人员出示结果概览报告
    print(metrics_df.to_string())
    print("=" * 72)


if __name__ == "__main__":
    main()
