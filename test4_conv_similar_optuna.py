from __future__ import annotations

import argparse
import gc
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

try:
    import optuna
    from optuna.exceptions import TrialPruned
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "Optuna is not installed. Install it first, for example: pip install optuna"
    ) from exc

import test4_base as base
import test4_conv_similar as exp
from utils.quantile import QuantileLoss
from utils.weather_e2e import WeatherGridStore, weather_data_provider


DEFAULT_STUDY_DIR = "optuna_test4_conv_similar_prior_gate"
DEFAULT_STUDY_NAME = "test4_conv_similar_prior_gate_two_stage"
FIX_SEED = 2026


def _save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return value


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _apply_overrides(args: argparse.Namespace, overrides: Optional[Dict[str, Any]]) -> argparse.Namespace:
    for key, value in (overrides or {}).items():
        setattr(args, key, value)
    return args


def build_args(
    *,
    train_mode: bool,
    weather_source: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> argparse.Namespace:
    weather_h5_specs = base._resolve_weather_h5_specs(weather_source)
    args = argparse.Namespace(
        task_name=base.TASK_NAME,
        is_training=1 if train_mode else 0,
        model_id=f"{base.MODEL_ID_PREFIX}_sdv4",
        model=base.MODEL,
        des=base.DES,
        itr=base.ITR,
        data="custom",
        root_path=base.ROOT_PATH,
        data_path=base.DATA_PATH,
        future_path=base.FUTURE_PATH,
        features=base.FEATURES,
        target=base.TARGET,
        target_channel_idx=0,
        freq=base.LOAD_FREQ,
        embed="timeF",
        checkpoints="./checkpoints_test4/",
        seq_len=base.SEQ_LEN,
        label_len=base.LABEL_LEN,
        pred_len=base.PRED_LEN,
        enc_in=base.ENC_IN,
        c_out=base.C_OUT,
        d_model=base.D_MODEL,
        n_heads=base.N_HEADS,
        e_layers=base.E_LAYERS,
        d_ff=base.D_FF,
        factor=base.FACTOR,
        dropout=base.DROPOUT,
        activation=base.ACTIVATION,
        patch_len=base.PATCH_LEN,
        use_norm=base.USE_NORM,
        weather_source=weather_source,
        weather_h5_specs=weather_h5_specs,
        weather_in_channels=base.WEATHER_IN_CHANNELS,
        weather_feature_dim=base.WEATHER_FEATURE_DIM,
        weather_grid_height=base.WEATHER_GRID_HEIGHT,
        weather_grid_width=base.WEATHER_GRID_WIDTH,
        weather_kernel_height=base.WEATHER_KERNEL_HEIGHT,
        weather_kernel_width=base.WEATHER_KERNEL_WIDTH,
        weather_encode_chunk_size=base.WEATHER_ENCODE_CHUNK_SIZE,
        use_weather_normalization=True,
        num_workers=base.NUM_WORKERS,
        pin_memory=base.PIN_MEMORY,
        contiguous_train_batches=base.CONTIGUOUS_TRAIN_BATCHES,
        train_epochs=base.TRAIN_EPOCHS,
        batch_size=base.BATCH_SIZE,
        patience=base.PATIENCE,
        learning_rate=base.LEARNING_RATE,
        loss="Quantile",
        lradj="cosine",
        use_amp=True,
        inverse_eval=base.INVERSE_EVAL,
        use_gpu=base.USE_GPU,
        gpu=base.GPU,
        use_multi_gpu=False,
        devices="0,1,2,3",
        quantiles=base.QUANTILES,
        n_quantiles=base.N_QUANTILES,
        use_similar_day_prior=True,
        similar_day_top_k=exp.SIMILAR_DAY_TOP_K,
        similar_day_artifact_dir=exp.SIMILAR_DAY_ARTIFACT_DIR,
        similar_day_gate_hidden_dim=exp.SIMILAR_DAY_GATE_HIDDEN_DIM,
        similar_day_gate_init_beta=exp.SIMILAR_DAY_GATE_INIT_BETA,
        enable_two_stage_finetune=True,
        stage1_epochs=exp.STAGE1_EPOCHS,
        stage1_patience=exp.STAGE1_PATIENCE,
        stage1_gate_lr=exp.STAGE1_GATE_LR,
        stage1_head_lr=exp.STAGE1_HEAD_LR,
        stage2_epochs=exp.STAGE2_EPOCHS,
        stage2_patience=exp.STAGE2_PATIENCE,
        stage2_backbone_lr=exp.STAGE2_BACKBONE_LR,
        stage2_gate_lr_scale=exp.STAGE2_GATE_LR_SCALE,
        stage2_use_cosine_lr=exp.STAGE2_USE_COSINE_LR,
    )
    args.results_root = "./results/"
    args.load_weight_path = None
    args.optuna_backbone_weight_path = None

    if exp.LOAD_FROM_OPTUNA:
        args = exp._apply_optuna_backbone_config(args)

    # Runtime and trial settings must win over the imported Optuna backbone config.
    return _apply_overrides(args, overrides)


def sample_trial_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    精细化搜索空间设计：以 test4_conv_similar.py 当前最优参数为锚点，
    在其邻域内做集中探索（exploitation-focused）。

    当前最优锚点参考值：
        SIMILAR_DAY_TOP_K = 3 (固定，复用缓存)
        SIMILAR_DAY_GATE_HIDDEN_DIM = 64
        SIMILAR_DAY_GATE_INIT_BETA = 0.10
        STAGE1_EPOCHS = 15,  STAGE1_PATIENCE = 4
        STAGE1_GATE_LR = 7e-4,  STAGE1_HEAD_LR = 1e-4
        STAGE2_EPOCHS = 20,  STAGE2_PATIENCE = 6
        STAGE2_BACKBONE_LR = 1e-5,  STAGE2_GATE_LR_SCALE = 10.0
        STAGE2_USE_COSINE_LR = True
    """
    return {
        # ---- 固定参数 (复用已有相似日 prior 缓存) ----
        "similar_day_top_k": 3,

        # ---- 门控架构参数 ----
        # 当前最优 64；探索相邻容量，48 偏轻量 / 80-96 偏富余
        "similar_day_gate_hidden_dim": trial.suggest_categorical(
            "SIMILAR_DAY_GATE_HIDDEN_DIM", [48, 64, 80, 96]
        ),
        # 当前最优 0.10；收窄至 [0.06, 0.15] 精细搜索初始先验采纳比例
        "similar_day_gate_init_beta": trial.suggest_float(
            "SIMILAR_DAY_GATE_INIT_BETA", 0.06, 0.15
        ),

        # ---- Stage 1: 门控预热 ----
        # 当前最优 15；探索 [12, 18]，过短欠拟合 / 过长浪费
        "stage1_epochs": trial.suggest_int("STAGE1_EPOCHS", 12, 18),
        # 当前最优 4
        "stage1_patience": trial.suggest_categorical("STAGE1_PATIENCE", [3, 4, 5]),
        # 当前最优 7e-4；收窄至约 ±0.5 个数量级
        "stage1_gate_lr": trial.suggest_float("STAGE1_GATE_LR", 4e-4, 1.2e-3, log=True),
        # 当前最优 1e-4；收窄至约 ±0.5 个数量级
        "stage1_head_lr": trial.suggest_float("STAGE1_HEAD_LR", 6e-5, 2e-4, log=True),

        # ---- Stage 2: 全量联合微调 ----
        # 当前最优 20；上限延伸至 28 允许更充分收敛
        "stage2_epochs": trial.suggest_int("STAGE2_EPOCHS", 16, 28),
        # 当前最优 6
        "stage2_patience": trial.suggest_categorical("STAGE2_PATIENCE", [5, 6, 7]),
        # 当前最优 1e-5；收窄至 [6e-6, 2e-5]，避免骨干过度扰动
        "stage2_backbone_lr": trial.suggest_float("STAGE2_BACKBONE_LR", 6e-6, 2e-5, log=True),
        # 当前最优 10.0；收窄至 [7.0, 14.0] 做门控/头部 lr 倍率精调
        "stage2_gate_lr_scale": trial.suggest_float("STAGE2_GATE_LR_SCALE", 7.0, 14.0),
        # 已验证余弦退火优于固定 lr，锁定为 True
        "stage2_use_cosine_lr": True,
    }


def _build_weather_store(args: argparse.Namespace) -> WeatherGridStore:
    return WeatherGridStore(
        args.weather_h5_specs,
        expected_in_channels=args.weather_in_channels,
        fill_value=base.WEATHER_FILL_VALUE,
        use_channel_normalization=True,
    )


def _finalize_runtime_weather_args(
    args: argparse.Namespace,
    weather_store: WeatherGridStore,
) -> argparse.Namespace:
    args = exp._configure_runtime_weather_args(args, weather_store, args.weather_source)
    if weather_store.frame_shape is None:
        raise RuntimeError("weather_store.frame_shape is not initialized.")
    _, frame_height, frame_width = weather_store.frame_shape
    if (frame_height, frame_width) != (args.weather_kernel_height, args.weather_kernel_width):
        raise ValueError(
            "Weather frame size does not match full-map kernel size: "
            f"frame=({frame_height}, {frame_width}), "
            f"kernel=({args.weather_kernel_height}, {args.weather_kernel_width})"
        )
    return args


def run_trial(args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    weather_store = _build_weather_store(args)
    try:
        args = _finalize_runtime_weather_args(args, weather_store)
        model = exp.FullMapConvTimeXerPriorCorrectionGateQuantile(
            args, quantiles=args.quantiles
        ).float().to(device)
        exp._load_backbone_from_optuna(model, args, device)
        model = exp.train_two_stage_model(model, args, device, weather_store)

        criterion = QuantileLoss(args.quantiles).to(device)
        _, vali_loader = weather_data_provider(args, "val", weather_store)
        _, test_loader = weather_data_provider(args, "test", weather_store)
        use_amp = bool(getattr(args, "use_amp", False)) and device.type == "cuda"
        vali_loss = exp.validate_quantile(model, vali_loader, criterion, args, device, use_amp=use_amp)
        test_loss = exp.validate_quantile(model, test_loader, criterion, args, device, use_amp=use_amp)
        setting = exp._get_setting(args)
        checkpoint_path = os.path.join(args.checkpoints, setting, "checkpoint.pth")
        return {
            "setting": setting,
            "checkpoint_path": checkpoint_path,
            "vali_loss": float(vali_loss),
            "test_loss": float(test_loss),
        }
    finally:
        weather_store.close()


def _best_trial_payload(study: optuna.Study) -> Dict[str, Any]:
    best_trial = study.best_trial
    return {
        "study_name": study.study_name,
        "best_trial": {
            "number": best_trial.number,
            "value": float(best_trial.value),
            "params": _to_builtin(best_trial.params),
            "user_attrs": _to_builtin(best_trial.user_attrs),
            "state": best_trial.state.name,
        },
    }


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optuna search for test4_conv_similar similar-day gate and two-stage fine-tuning."
    )
    parser.add_argument("--study-dir", type=str, default=DEFAULT_STUDY_DIR)
    parser.add_argument("--study-name", type=str, default=DEFAULT_STUDY_NAME)
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--n-trials", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=0, help="Seconds. 0 means no timeout.")
    parser.add_argument("--seed", type=int, default=FIX_SEED)
    parser.add_argument(
        "--weather-source",
        type=str,
        choices=sorted(base.WEATHER_SOURCE_CONFIGS.keys()),
        default=base.DEFAULT_WEATHER_SOURCE,
    )
    parser.add_argument("--gpu", type=int, default=base.GPU)
    parser.add_argument("--use-cpu", action="store_true", default=False)
    parser.add_argument("--no-amp", action="store_true", default=False)
    parser.add_argument("--checkpoints-dir", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    return parser


def main() -> Dict[str, Any]:
    cli_args = build_cli_parser().parse_args()
    study_dir = Path(cli_args.study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)

    storage = cli_args.storage or f"sqlite:///{(study_dir / 'study.db').as_posix()}"
    sampler = optuna.samplers.TPESampler(seed=int(cli_args.seed), multivariate=True)
    study = optuna.create_study(
        study_name=cli_args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=sampler,
    )

    use_gpu = (not bool(cli_args.use_cpu)) and torch.cuda.is_available()
    device = torch.device(f"cuda:{int(cli_args.gpu)}" if use_gpu else "cpu")
    runtime_overrides = {
        "checkpoints": cli_args.checkpoints_dir or str(study_dir / "checkpoints"),
        "results_root": cli_args.results_dir or str(study_dir / "results"),
        "use_gpu": bool(use_gpu),
        "gpu": int(cli_args.gpu),
        "use_amp": (not bool(cli_args.no_amp)) and bool(use_gpu),
    }

    def objective(trial: optuna.Trial) -> float:
        _set_seed(int(cli_args.seed) + int(trial.number))
        params = sample_trial_params(trial)
        overrides = dict(runtime_overrides)
        overrides.update(params)
        overrides["des"] = f"OptunaGate_trial{trial.number:03d}"
        overrides["itr"] = int(trial.number)

        try:
            args = build_args(
                train_mode=True,
                weather_source=str(cli_args.weather_source),
                overrides=overrides,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise TrialPruned(str(exc)) from exc

        trial.set_user_attr("sampled_overrides", _to_builtin(params))
        try:
            result = run_trial(args, device)
        except RuntimeError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for key, value in result.items():
            trial.set_user_attr(key, _to_builtin(value))
        return float(result["vali_loss"])

    print("\n" + "=" * 72)
    print("Optuna search: similar-day gate + two-stage fine-tuning")
    print(f"study_name: {cli_args.study_name}")
    print(f"storage: {storage}")
    print(f"study_dir: {study_dir}")
    print(f"weather_source: {cli_args.weather_source}")
    print(f"device: {device}")
    print(f"n_trials: {cli_args.n_trials}")
    print("=" * 72)

    study.optimize(
        objective,
        n_trials=int(cli_args.n_trials),
        timeout=None if int(cli_args.timeout) <= 0 else int(cli_args.timeout),
        gc_after_trial=True,
        show_progress_bar=False,
    )

    best_trial = study.best_trial
    best_params_path = study_dir / "best_params_prior_gate.json"
    best_overrides_path = study_dir / "best_overrides_prior_gate.json"
    best_trial_path = study_dir / "best_trial_result_prior_gate.json"
    _save_json(best_params_path, best_trial.params)
    _save_json(best_overrides_path, best_trial.user_attrs.get("sampled_overrides", {}))
    _save_json(best_trial_path, _best_trial_payload(study))

    try:
        trials_df = study.trials_dataframe()
        trials_df.to_csv(study_dir / "trials.csv", index=False, encoding="utf-8-sig")
    except Exception as exc:
        print(f"Skip saving trials.csv: {exc}")

    print("\nBest trial summary:")
    print(f"trial_number: {best_trial.number}")
    print(f"best_vali_loss: {float(best_trial.value):.7f}")
    print(f"best_params: {best_trial.params}")
    print(f"checkpoint_path: {best_trial.user_attrs.get('checkpoint_path')}")
    print(f"best_params_path: {best_params_path}")
    print(f"best_overrides_path: {best_overrides_path}")
    print(f"best_trial_path: {best_trial_path}")

    return {
        "best_value": float(best_trial.value),
        "best_params": dict(best_trial.params),
        "best_params_path": str(best_params_path),
        "best_overrides_path": str(best_overrides_path),
        "best_trial_path": str(best_trial_path),
    }


if __name__ == "__main__":
    main()
