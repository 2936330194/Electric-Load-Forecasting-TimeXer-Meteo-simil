"""
Optuna hyperparameter tuning for test3.py.
"""

import argparse
import gc
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import optim

import test3 as base

try:
    import optuna
    from optuna.exceptions import TrialPruned
except ImportError:
    optuna = None
    TrialPruned = None


FIX_SEED = 2026
DEFAULT_CHECKPOINTS_ROOT = "./checkpoints_test3_optuna/"
DEFAULT_RESULTS_ROOT = "./results_test3_optuna/"
DEFAULT_OPTUNA_OUTPUT_ROOT = "./optuna_runs/"
DEFAULT_OPTUNA_STUDY_NAME = "test3_optuna_v1"
DEFAULT_OPTUNA_N_TRIALS = 128
DEFAULT_OPTUNA_TIMEOUT = 0
DEFAULT_OPTUNA_DIRECTION = "minimize"

DEFAULT_USE_DIR = "./optuna"
USE_BEST_PARAMS_FILE = "best_params3.json"
USE_BEST_CONFIG_FILE = "best_config3.json"
USE_BEST_WEIGHT_FILE = "best_model3.pth"
USE_BEST_TRIAL_FILE = "best_trial_result3.json"

DEFAULT_OPTUNA_SEARCH_SPACE: Dict[str, Any] = {
    "WEATHER_FEATURE_DIM": {"type": "categorical", "choices": [2, 3, 4, 5, 6, 7, 8]},
    "D_MODEL": {"type": "categorical", "choices": [128, 256, 384, 512, 768, 1024]},
    "N_HEADS": {"type": "categorical", "choices": [2, 4, 8, 12, 16]},
    "E_LAYERS": {"type": "int", "low": 1, "high": 3, "step": 1},
    "D_FF": {"type": "categorical", "choices": [256, 512, 1024, 1536, 2048, 3072, 4096]},
    "DROPOUT": {"type": "float", "low": 0.05, "high": 0.20, "step": 0.05},
    "PATCH_LEN": {"type": "categorical", "choices": [48, 72, 96]},
    "BATCH_SIZE": {"type": "categorical", "choices": [32, 48, 64, 72]},
    "LEARNING_RATE": {"type": "float", "low": 1e-5, "high": 5e-3, "log": True},
}

TUNABLE_PARAM_MAP = dict(base.TUNABLE_PARAM_MAP)


def _ensure_optuna_available() -> None:
    if optuna is None:
        raise ImportError("optuna is not installed. Please run: pip install optuna")


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_json_file(json_path: str) -> Any:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json_file(json_path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_to_builtin(payload), f, ensure_ascii=False, indent=2)


def _normalize_param_overrides(overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not overrides:
        return {}

    normalized: Dict[str, Any] = {}
    valid_arg_names = set(TUNABLE_PARAM_MAP.values())
    for raw_key, value in overrides.items():
        key = str(raw_key)
        if key in TUNABLE_PARAM_MAP:
            normalized[TUNABLE_PARAM_MAP[key]] = value
        elif key in valid_arg_names:
            normalized[key] = value
        elif key.lower() in valid_arg_names:
            normalized[key.lower()] = value
        else:
            normalized[key] = value
    return normalized


def _base_args_dict(weather_source: Optional[str] = None) -> Dict[str, Any]:
    selected_weather_source = weather_source or base.DEFAULT_WEATHER_SOURCE
    selected_weather_h5_specs = base._resolve_weather_h5_specs(selected_weather_source)
    return {
        "task_name": base.TASK_NAME,
        "is_training": 1,
        "model_id": base.MODEL_ID_PREFIX,
        "model": base.MODEL,
        "des": base.DES,
        "itr": base.ITR,
        "data": "custom",
        "root_path": base.ROOT_PATH,
        "data_path": base.DATA_PATH,
        "future_path": base.FUTURE_PATH,
        "features": base.FEATURES,
        "target": base.TARGET,
        "target_channel_idx": 0,
        "freq": base.LOAD_FREQ,
        "embed": "timeF",
        "checkpoints": DEFAULT_CHECKPOINTS_ROOT,
        "results_root": DEFAULT_RESULTS_ROOT,
        "seq_len": base.SEQ_LEN,
        "label_len": base.LABEL_LEN,
        "pred_len": base.PRED_LEN,
        "enc_in": base.ENC_IN,
        "c_out": base.C_OUT,
        "d_model": base.D_MODEL,
        "n_heads": base.N_HEADS,
        "e_layers": base.E_LAYERS,
        "d_ff": base.D_FF,
        "factor": base.FACTOR,
        "dropout": base.DROPOUT,
        "activation": base.ACTIVATION,
        "patch_len": base.PATCH_LEN,
        "use_norm": base.USE_NORM,
        "weather_source": selected_weather_source,
        "weather_h5_specs": selected_weather_h5_specs,
        "weather_in_channels": base.WEATHER_IN_CHANNELS,
        "weather_feature_dim": base.WEATHER_FEATURE_DIM,
        "weather_grid_height": base.WEATHER_GRID_HEIGHT,
        "weather_grid_width": base.WEATHER_GRID_WIDTH,
        "weather_kernel_height": base.WEATHER_KERNEL_HEIGHT,
        "weather_kernel_width": base.WEATHER_KERNEL_WIDTH,
        "weather_encode_chunk_size": base.WEATHER_ENCODE_CHUNK_SIZE,
        "use_weather_normalization": True,
        "num_workers": base.NUM_WORKERS,
        "pin_memory": base.PIN_MEMORY,
        "contiguous_train_batches": base.CONTIGUOUS_TRAIN_BATCHES,
        "train_epochs": base.TRAIN_EPOCHS,
        "batch_size": base.BATCH_SIZE,
        "patience": base.PATIENCE,
        "learning_rate": base.LEARNING_RATE,
        "loss": "Quantile",
        "lradj": "cosine",
        "use_amp": base.USE_AMP,
        "inverse_eval": base.INVERSE_EVAL,
        "use_gpu": base.USE_GPU,
        "gpu": base.GPU,
        "use_multi_gpu": False,
        "devices": "0,1,2,3",
        "quantiles": list(base.QUANTILES),
        "n_quantiles": base.N_QUANTILES,
        "load_weight_path": None,
    }


def _build_args(
    train_mode: bool = True,
    overrides: Optional[Dict[str, Any]] = None,
    *,
    weather_source: Optional[str] = None,
) -> argparse.Namespace:
    config = _base_args_dict(weather_source=weather_source)
    config["is_training"] = 1 if train_mode else 0
    config.update(_normalize_param_overrides(overrides))

    int_fields = [
        "is_training",
        "target_channel_idx",
        "seq_len",
        "label_len",
        "pred_len",
        "enc_in",
        "c_out",
        "d_model",
        "n_heads",
        "e_layers",
        "d_ff",
        "factor",
        "patch_len",
        "use_norm",
        "weather_in_channels",
        "weather_feature_dim",
        "weather_grid_height",
        "weather_grid_width",
        "weather_kernel_height",
        "weather_kernel_width",
        "weather_encode_chunk_size",
        "itr",
        "train_epochs",
        "batch_size",
        "patience",
        "gpu",
        "n_quantiles",
    ]
    for field in int_fields:
        config[field] = int(config[field])

    for field in ["dropout", "learning_rate"]:
        config[field] = float(config[field])

    for field in [
        "use_weather_normalization",
        "pin_memory",
        "contiguous_train_batches",
        "use_amp",
        "inverse_eval",
        "use_gpu",
        "use_multi_gpu",
    ]:
        config[field] = bool(config[field])

    config["quantiles"] = list(config["quantiles"])
    config["n_quantiles"] = len(config["quantiles"])

    if config["d_model"] % config["n_heads"] != 0:
        raise ValueError(f"d_model must be divisible by n_heads: {config['d_model']} vs {config['n_heads']}")
    if config["patch_len"] <= 0 or config["patch_len"] > config["seq_len"]:
        raise ValueError(f"patch_len must be in (0, seq_len], got {config['patch_len']}")

    return argparse.Namespace(**config)


def _select_device(args) -> torch.device:
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU: cuda:{args.gpu}")
        return device
    print("Using CPU")
    return torch.device("cpu")


def _normalize_search_spec(spec: Any) -> Dict[str, Any]:
    if isinstance(spec, list):
        return {"type": "categorical", "choices": spec}
    if not isinstance(spec, dict):
        raise ValueError(f"Invalid search-space spec: {spec}")
    if "choices" in spec and "type" not in spec:
        spec = dict(spec)
        spec["type"] = "categorical"
    return dict(spec)


def _suggest_from_spec(trial, name: str, spec: Any) -> Any:
    normalized = _normalize_search_spec(spec)
    spec_type = normalized.get("type")
    if spec_type == "categorical":
        return trial.suggest_categorical(name, normalized["choices"])
    if spec_type == "int":
        return trial.suggest_int(
            name,
            int(normalized["low"]),
            int(normalized["high"]),
            step=int(normalized.get("step", 1)),
            log=bool(normalized.get("log", False)),
        )
    if spec_type == "float":
        low = float(normalized["low"])
        high = float(normalized["high"])
        log = bool(normalized.get("log", False))
        step = normalized.get("step")
        if step is None:
            return trial.suggest_float(name, low, high, log=log)
        if log:
            raise ValueError(f"{name} cannot set both step and log search")
        return trial.suggest_float(name, low, high, step=float(step))
    raise ValueError(f"Unsupported search-space type: {name} -> {spec_type}")


def _load_search_space(
    search_space_json: Optional[str],
    search_space: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if search_space is not None:
        return json.loads(json.dumps(search_space))
    if search_space_json is None:
        return json.loads(json.dumps(DEFAULT_OPTUNA_SEARCH_SPACE))
    payload = _load_json_file(search_space_json)
    if not isinstance(payload, dict):
        raise ValueError("search_space_json must contain a JSON object")
    return payload


def _sample_trial_params(trial, search_space: Dict[str, Any], base_args: argparse.Namespace) -> Dict[str, Any]:
    unknown_keys = [key for key in search_space if key not in TUNABLE_PARAM_MAP]
    if unknown_keys:
        raise ValueError(f"Unknown search-space keys: {unknown_keys}")

    sampled: Dict[str, Any] = {}
    discrete_int_keys = {
        "WEATHER_FEATURE_DIM",
        "D_MODEL",
        "N_HEADS",
        "E_LAYERS",
        "D_FF",
        "PATCH_LEN",
        "BATCH_SIZE",
    }

    if "D_MODEL" in search_space:
        sampled["D_MODEL"] = int(_suggest_from_spec(trial, "D_MODEL", search_space["D_MODEL"]))
    if "N_HEADS" in search_space:
        sampled["N_HEADS"] = int(_suggest_from_spec(trial, "N_HEADS", search_space["N_HEADS"]))

    for name, spec in search_space.items():
        if name in sampled:
            continue
        value = _suggest_from_spec(trial, name, spec)
        if name in discrete_int_keys:
            value = int(value)
        elif name in {"DROPOUT", "LEARNING_RATE"}:
            value = float(value)
        sampled[name] = value

    current_d_model = int(sampled.get("D_MODEL", base_args.d_model))
    current_n_heads = int(sampled.get("N_HEADS", base_args.n_heads))
    if current_d_model % current_n_heads != 0:
        raise TrialPruned(f"Invalid head split: d_model={current_d_model}, n_heads={current_n_heads}")

    return sampled


def train_quantile_model(model, args, device, weather_store: base.WeatherGridStore, trial=None):
    _, train_loader = base.weather_data_provider(args, "train", weather_store)
    _, vali_loader = base.weather_data_provider(args, "val", weather_store)

    setting = base._get_setting(args)
    path = os.path.join(args.checkpoints, setting)
    os.makedirs(path, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = base.QuantileLoss(args.quantiles).to(device)
    early_stopping = base.EarlyStopping(patience=args.patience, verbose=True)

    use_amp = bool(getattr(args, "use_amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    use_non_blocking = base._use_non_blocking_transfer(args, device)

    print("\n" + "=" * 72)
    print("Start Optuna trial training: Full-Map Conv + TimeXer")
    print(f"setting: {setting}")
    print(f"quantiles: {args.quantiles}")
    print(f"weather_feature_dim: {args.weather_feature_dim}")
    print(
        f"weather_seq_len: {args.weather_seq_len} "
        f"(history={args.weather_history_len}, future={args.weather_seq_len - args.weather_history_len}, "
        f"step={getattr(args, 'weather_step_freq', 'native')})"
    )
    print(f"batch_size: {args.batch_size}")
    print(f"use_amp: {use_amp}")
    print("=" * 72)

    best_vali_loss = np.inf
    for epoch in range(args.train_epochs):
        model.train()
        train_loss = []
        epoch_time = time.time()

        for i, (
            batch_x,
            batch_y,
            batch_x_mark,
            batch_exo_mark,
            batch_weather_frames,
            batch_weather_index,
        ) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)

            batch_x = base._to_float_device(batch_x, device, non_blocking=use_non_blocking)
            batch_y = base._to_float_device(batch_y, device, non_blocking=use_non_blocking)
            batch_x_mark = base._to_float_device(batch_x_mark, device, non_blocking=use_non_blocking)
            batch_exo_mark = base._to_float_device(batch_exo_mark, device, non_blocking=use_non_blocking)
            batch_weather_frames = base._to_float_device(batch_weather_frames, device, non_blocking=use_non_blocking)
            batch_weather_index = base._to_long_device(batch_weather_index, device, non_blocking=use_non_blocking)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(
                    load_x=batch_x,
                    x_mark_enc=batch_x_mark,
                    x_exo_mark=batch_exo_mark,
                    weather_x=batch_weather_frames,
                    weather_x_index=batch_weather_index,
                )
                batch_y_target = base.extract_target(batch_y[:, -args.pred_len :, :])
                loss = criterion(outputs, batch_y_target)

            if not torch.isfinite(outputs).all():
                message = f"Non-finite model output at epoch={epoch + 1}, iter={i + 1}"
                if trial is not None:
                    raise TrialPruned(message)
                raise RuntimeError(message)
            if not torch.isfinite(loss):
                message = f"Non-finite training loss at epoch={epoch + 1}, iter={i + 1}"
                if trial is not None:
                    raise TrialPruned(message)
                raise RuntimeError(message)

            train_loss.append(float(loss.detach().item()))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (i + 1) % 100 == 0:
                print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")

        vali_loss = float(base.validate_quantile(model, vali_loader, criterion, args, device, use_amp=use_amp))
        train_loss_avg = float(np.average(train_loss)) if train_loss else np.nan
        print(
            f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.1f}s | "
            f"Train: {train_loss_avg:.7f} Vali: {vali_loss:.7f}"
        )

        if np.isfinite(vali_loss):
            best_vali_loss = min(best_vali_loss, vali_loss)
        if trial is not None:
            if not np.isfinite(vali_loss):
                raise TrialPruned("Validation loss is not finite")
            trial.report(vali_loss, step=epoch + 1)
            if trial.should_prune():
                raise TrialPruned(f"Trial pruned at epoch {epoch + 1}, vali_loss={vali_loss:.7f}")

        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        base.adjust_learning_rate(optimizer, epoch + 1, args)

    best_model_path = os.path.join(path, "checkpoint.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best checkpoint not found: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"Loaded best model weights: {best_model_path}")
    return model, float(best_vali_loss)


def test_quantile_model(model, args, device, weather_store: base.WeatherGridStore) -> Dict[str, Any]:
    test_data, test_loader = base.weather_data_provider(args, "test", weather_store)

    setting = base._get_setting(args)
    folder_path = os.path.join(getattr(args, "results_root", DEFAULT_RESULTS_ROOT), setting)
    os.makedirs(folder_path, exist_ok=True)

    preds_p50 = []
    trues = []
    quantile_preds_all = []
    use_amp = bool(getattr(args, "use_amp", False)) and device.type == "cuda"
    use_non_blocking = base._use_non_blocking_transfer(args, device)

    model.eval()
    with torch.inference_mode():
        for (
            batch_x,
            batch_y,
            batch_x_mark,
            batch_exo_mark,
            batch_weather_frames,
            batch_weather_index,
        ) in test_loader:
            batch_x = base._to_float_device(batch_x, device, non_blocking=use_non_blocking)
            batch_y = base._to_float_device(batch_y, device, non_blocking=use_non_blocking)
            batch_x_mark = base._to_float_device(batch_x_mark, device, non_blocking=use_non_blocking)
            batch_exo_mark = base._to_float_device(batch_exo_mark, device, non_blocking=use_non_blocking)
            batch_weather_frames = base._to_float_device(batch_weather_frames, device, non_blocking=use_non_blocking)
            batch_weather_index = base._to_long_device(batch_weather_index, device, non_blocking=use_non_blocking)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(
                    load_x=batch_x,
                    x_mark_enc=batch_x_mark,
                    x_exo_mark=batch_exo_mark,
                    weather_x=batch_weather_frames,
                    weather_x_index=batch_weather_index,
                )

            batch_y_target = base.extract_target(batch_y[:, -args.pred_len :, :])
            p50_pred = outputs.float()[:, :, base.P50_IDX : base.P50_IDX + 1]
            quantile_preds_all.append(outputs.float().detach().cpu().numpy())
            preds_p50.append(p50_pred.detach().cpu().numpy())
            trues.append(batch_y_target.detach().cpu().numpy())

    preds_p50 = np.concatenate(preds_p50, axis=0)
    trues = np.concatenate(trues, axis=0)
    quantile_preds_all = np.concatenate(quantile_preds_all, axis=0)

    np.save(os.path.join(folder_path, "pred.npy"), preds_p50)
    np.save(os.path.join(folder_path, "true.npy"), trues)
    np.save(os.path.join(folder_path, "quantile_preds.npy"), quantile_preds_all)

    preds_inv = None
    trues_inv = None
    if test_data.scale:
        shape = trues.shape
        preds_inv = test_data.inverse_transform_target(preds_p50.reshape(shape[0] * shape[1], -1)).reshape(shape)
        trues_inv = test_data.inverse_transform_target(trues.reshape(shape[0] * shape[1], -1)).reshape(shape)

        q_shape = quantile_preds_all.shape
        quantile_inv = np.zeros_like(quantile_preds_all)
        for qi in range(args.n_quantiles):
            q_slice = quantile_preds_all[:, :, qi : qi + 1]
            q_inv = test_data.inverse_transform_target(q_slice.reshape(q_shape[0] * q_shape[1], -1)).reshape(
                q_shape[0], q_shape[1], 1
            )
            quantile_inv[:, :, qi] = q_inv[:, :, 0]

        np.save(os.path.join(folder_path, "pred_inv.npy"), preds_inv)
        np.save(os.path.join(folder_path, "true_inv.npy"), trues_inv)
        np.save(os.path.join(folder_path, "quantile_preds_inv.npy"), quantile_inv)

    if test_data.scale and getattr(args, "inverse_eval", False):
        mae, mse, rmse, mape, mspe = base.metric(preds_inv, trues_inv)
        print(f"P50 Test Metrics (Inverse): MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}")
    else:
        mae, mse, rmse, mape, mspe = base.metric(preds_p50, trues)
        print(f"P50 Test Metrics (Normalized): MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}")

    return {
        "results_dir": folder_path,
        "p50_mae": float(mae),
        "p50_mse": float(mse),
        "p50_rmse": float(rmse),
        "p50_mape": float(mape),
        "p50_mspe": float(mspe),
    }


def run_experiment(
    args,
    *,
    run_test: bool,
    plot_results: bool,
    predict_future: bool,
    trial=None,
) -> Dict[str, Any]:
    device = _select_device(args)
    selected_weather_source = getattr(args, "weather_source", base.DEFAULT_WEATHER_SOURCE)
    result: Dict[str, Any] = {}

    weather_store = base.WeatherGridStore(
        args.weather_h5_specs,
        expected_in_channels=args.weather_in_channels,
        fill_value=base.WEATHER_FILL_VALUE,
        use_channel_normalization=True,
    )

    try:
        args = base._configure_runtime_weather_args(args, weather_store, selected_weather_source)
        setting = base._get_setting(args)
        result["setting"] = setting

        if weather_store.frame_shape is None:
            raise RuntimeError("weather_store.frame_shape is not initialized.")
        _, frame_height, frame_width = weather_store.frame_shape
        if (frame_height, frame_width) != (args.weather_kernel_height, args.weather_kernel_width):
            raise ValueError(
                "Weather frame size does not match full-map kernel size: "
                f"frame=({frame_height}, {frame_width}), "
                f"kernel=({args.weather_kernel_height}, {args.weather_kernel_width})"
            )

        model = base.ExogenousFullMapConvTimeXerQuantile(args, quantiles=args.quantiles).float().to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Full-Map Conv + TimeXer total params: {total_params:,}")
        print(f"Full-Map Conv + TimeXer trainable params: {trainable_params:,}")

        checkpoint_path = getattr(args, "load_weight_path", None)
        if checkpoint_path is None:
            checkpoint_path = os.path.join(args.checkpoints, setting, "checkpoint.pth")

        if args.is_training:
            print(f"\n>>> Start training {setting}")
            model, best_vali_loss = train_quantile_model(model, args, device, weather_store, trial=trial)
            result["best_vali_loss"] = float(best_vali_loss)
            checkpoint_path = os.path.join(args.checkpoints, setting, "checkpoint.pth")
        else:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Model file not found: {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"Loaded model: {checkpoint_path}")

        result["checkpoint_path"] = checkpoint_path

        if run_test:
            print(f"\n>>> Start testing {setting}")
            eval_result = test_quantile_model(model, args, device, weather_store)
            result.update(eval_result)
            results_dir = eval_result["results_dir"]

            if plot_results:
                base.plot_pred_vs_true(
                    results_dir,
                    use_inverse=args.inverse_eval,
                    quantiles=args.quantiles,
                    title_prefix="Full-Map Conv + TimeXer Prediction",
                    y_label="Load (MW)",
                )
            if predict_future:
                base.predict_future_load_from_csv(
                    model=model,
                    args=args,
                    device=device,
                    weather_store=weather_store,
                    results_dir=results_dir,
                    future_path=getattr(args, "future_path", base.FUTURE_PATH),
                    steps=args.pred_len,
                    use_inverse=args.inverse_eval,
                    quantiles=args.quantiles,
                    data_provider_fn=base.weather_data_provider,
                    model_label="Full-Map Conv + TimeXer",
                    y_label="Load (MW)",
                )

        return result
    finally:
        weather_store.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def _default_storage_uri(study_dir: Path) -> str:
    db_path = (study_dir / "optuna_study.db").resolve()
    return f"sqlite:///{db_path.as_posix()}"


def _build_best_trial_payload(study, best_eval: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    best_trial = study.best_trial
    completed_trials = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    pruned_trials = [trial for trial in study.trials if trial.state.name == "PRUNED"]
    failed_trials = [trial for trial in study.trials if trial.state.name == "FAIL"]
    payload: Dict[str, Any] = {
        "study_name": study.study_name,
        "direction": [direction.name.lower() for direction in study.directions],
        "n_trials_total": len(study.trials),
        "n_trials_completed": len(completed_trials),
        "n_trials_pruned": len(pruned_trials),
        "n_trials_failed": len(failed_trials),
        "best_trial": {
            "number": best_trial.number,
            "value": float(best_trial.value),
            "params": _to_builtin(best_trial.params),
            "user_attrs": _to_builtin(best_trial.user_attrs),
            "state": best_trial.state.name,
        },
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if best_eval is not None:
        payload["best_eval"] = _to_builtin(best_eval)
    return payload


def _export_best_trial_to_use(
    use_dir: str,
    best_trial,
    best_args,
    best_payload: Dict[str, Any],
) -> Dict[str, str]:
    use_dir = os.path.abspath(use_dir)
    os.makedirs(use_dir, exist_ok=True)

    src_weight_path = getattr(best_args, "load_weight_path", None) or best_trial.user_attrs.get("checkpoint_path")
    if not src_weight_path or not os.path.exists(src_weight_path):
        raise FileNotFoundError(f"Best checkpoint not found: {src_weight_path}")

    dst_weight_path = os.path.join(use_dir, USE_BEST_WEIGHT_FILE)
    shutil.copy2(src_weight_path, dst_weight_path)

    best_params_path = os.path.join(use_dir, USE_BEST_PARAMS_FILE)
    _save_json_file(best_params_path, best_trial.params)

    best_config = dict(vars(best_args))
    best_config["is_training"] = 0
    best_config["load_weight_path"] = dst_weight_path
    best_config["results_root"] = os.path.join(use_dir, "results")
    best_config_path = os.path.join(use_dir, USE_BEST_CONFIG_FILE)
    _save_json_file(best_config_path, best_config)

    payload_with_export = dict(best_payload)
    payload_with_export["use_export"] = {
        "use_dir": use_dir,
        "best_weight_path": dst_weight_path,
        "best_params_path": best_params_path,
        "best_config_path": best_config_path,
    }
    best_trial_result_path = os.path.join(use_dir, USE_BEST_TRIAL_FILE)
    _save_json_file(best_trial_result_path, payload_with_export)

    print("\n" + "=" * 72)
    print(f"Best trial exported to {use_dir}")
    print(f"best_params: {best_params_path}")
    print(f"best_config: {best_config_path}")
    print(f"best_model: {dst_weight_path}")
    print("=" * 72)
    return {
        "best_params_path": best_params_path,
        "best_config_path": best_config_path,
        "best_weight_path": dst_weight_path,
        "best_trial_result_path": best_trial_result_path,
    }


def run_optuna_study(cli_args) -> Dict[str, Any]:
    _ensure_optuna_available()
    _set_random_seed(int(cli_args.seed))

    output_root = Path(cli_args.output_dir).expanduser().resolve()
    study_dir = output_root / cli_args.study_name
    study_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_root = study_dir / "checkpoints"
    results_root = study_dir / "results"

    search_space = _load_search_space(cli_args.search_space_json, getattr(cli_args, "search_space", None))
    search_space_path = study_dir / "search_space.json"
    _save_json_file(str(search_space_path), search_space)

    runtime_overrides = {
        "checkpoints": str(checkpoints_root),
        "results_root": str(results_root),
        "train_epochs": int(cli_args.train_epochs),
        "patience": int(cli_args.patience),
        "use_gpu": not bool(cli_args.use_cpu),
        "gpu": int(cli_args.gpu),
        "use_amp": not bool(cli_args.no_amp),
    }
    base_args = _build_args(train_mode=True, overrides=runtime_overrides, weather_source=cli_args.weather_source)

    storage = cli_args.storage or _default_storage_uri(study_dir)
    sampler = optuna.samplers.TPESampler(seed=int(cli_args.seed))
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=1,
        max_resource=int(cli_args.train_epochs),
        reduction_factor=3,
    )
    study = optuna.create_study(
        study_name=cli_args.study_name,
        storage=storage,
        load_if_exists=True,
        direction=cli_args.direction,
        sampler=sampler,
        pruner=pruner,
    )

    def objective(trial) -> float:
        sampled_params = _sample_trial_params(trial, search_space, base_args)
        trial_overrides = dict(runtime_overrides)
        trial_overrides.update(_normalize_param_overrides(sampled_params))
        trial_overrides["des"] = f"Optuna_trial{trial.number:03d}"
        trial_overrides["itr"] = int(trial.number)

        try:
            args = _build_args(
                train_mode=True,
                overrides=trial_overrides,
                weather_source=cli_args.weather_source,
            )
        except ValueError as exc:
            raise TrialPruned(str(exc)) from exc

        experiment_result = run_experiment(
            args,
            run_test=False,
            plot_results=False,
            predict_future=False,
            trial=trial,
        )
        trial.set_user_attr("setting", experiment_result["setting"])
        trial.set_user_attr("checkpoint_path", experiment_result["checkpoint_path"])
        best_vali_loss = float(experiment_result["best_vali_loss"])
        trial.set_user_attr("best_vali_loss", best_vali_loss)
        return best_vali_loss

    print("\n" + "=" * 72)
    print(f"Optuna study: {cli_args.study_name}")
    print(f"storage: {storage}")
    print(f"n_trials: {cli_args.n_trials}")
    print(f"timeout: {cli_args.timeout}")
    print(f"output_dir: {study_dir}")
    print(f"weather_source: {cli_args.weather_source}")
    print("=" * 72)
    study.optimize(
        objective,
        n_trials=int(cli_args.n_trials),
        timeout=None if int(cli_args.timeout) <= 0 else int(cli_args.timeout),
        gc_after_trial=True,
        show_progress_bar=False,
    )

    try:
        best_trial = study.best_trial
    except ValueError as exc:
        raise RuntimeError("No completed Optuna trial is available.") from exc

    best_params_path = cli_args.best_params_path or str(study_dir / USE_BEST_PARAMS_FILE)
    _save_json_file(best_params_path, best_trial.params)

    best_overrides = dict(runtime_overrides)
    best_overrides.update(_normalize_param_overrides(best_trial.params))
    best_overrides["des"] = f"Optuna_trial{best_trial.number:03d}"
    best_overrides["itr"] = int(best_trial.number)
    best_overrides["load_weight_path"] = best_trial.user_attrs.get("checkpoint_path")
    best_args = _build_args(
        train_mode=False,
        overrides=best_overrides,
        weather_source=cli_args.weather_source,
    )

    best_eval = None
    if not cli_args.skip_best_eval:
        best_eval = run_experiment(
            best_args,
            run_test=True,
            plot_results=True,
            predict_future=True,
            trial=None,
        )

    best_payload = _build_best_trial_payload(study, best_eval=best_eval)
    best_trial_path = cli_args.best_trial_path or str(study_dir / USE_BEST_TRIAL_FILE)
    _save_json_file(best_trial_path, best_payload)
    _export_best_trial_to_use(cli_args.use_dir, best_trial, best_args, best_payload)

    trials_csv_path = study_dir / "trials.csv"
    try:
        trials_df = study.trials_dataframe()
        trials_df.to_csv(trials_csv_path, index=False, encoding="utf-8-sig")
    except Exception as exc:
        print(f"Skip saving trials.csv: {exc}")

    print("\nBest trial summary:")
    print(f"trial_number: {best_trial.number}")
    print(f"best_value: {float(best_trial.value):.7f}")
    print(f"best_params: {best_trial.params}")

    return {
        "study_dir": str(study_dir),
        "best_params_path": best_params_path,
        "best_trial_path": best_trial_path,
        "best_value": float(best_trial.value),
        "best_params": dict(best_trial.params),
    }


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning for test3.py")
    parser.add_argument(
        "--weather-source",
        type=str,
        choices=sorted(base.WEATHER_SOURCE_CONFIGS.keys()),
        default=base.DEFAULT_WEATHER_SOURCE,
    )
    parser.add_argument("--search-space-json", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OPTUNA_OUTPUT_ROOT)
    parser.add_argument("--study-name", type=str, default=DEFAULT_OPTUNA_STUDY_NAME)
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--n-trials", type=int, default=DEFAULT_OPTUNA_N_TRIALS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_OPTUNA_TIMEOUT)
    parser.add_argument("--direction", type=str, default=DEFAULT_OPTUNA_DIRECTION)
    parser.add_argument("--seed", type=int, default=FIX_SEED)
    parser.add_argument("--train-epochs", type=int, default=base.TRAIN_EPOCHS)
    parser.add_argument("--patience", type=int, default=base.PATIENCE)
    parser.add_argument("--gpu", type=int, default=base.GPU)
    parser.add_argument("--use-cpu", action="store_true", default=False)
    parser.add_argument("--no-amp", action="store_true", default=False)
    parser.add_argument("--skip-best-eval", action="store_true", default=False)
    parser.add_argument("--best-params-path", type=str, default=None)
    parser.add_argument("--best-trial-path", type=str, default=None)
    parser.add_argument("--use-dir", type=str, default=DEFAULT_USE_DIR)
    return parser


def _build_runtime_namespace(config: Dict[str, Any]) -> argparse.Namespace:
    defaults = {
        "weather_source": base.DEFAULT_WEATHER_SOURCE,
        "search_space": json.loads(json.dumps(DEFAULT_OPTUNA_SEARCH_SPACE)),
        "search_space_json": None,
        "output_dir": DEFAULT_OPTUNA_OUTPUT_ROOT,
        "study_name": DEFAULT_OPTUNA_STUDY_NAME,
        "storage": None,
        "n_trials": DEFAULT_OPTUNA_N_TRIALS,
        "timeout": DEFAULT_OPTUNA_TIMEOUT,
        "direction": DEFAULT_OPTUNA_DIRECTION,
        "seed": FIX_SEED,
        "train_epochs": base.TRAIN_EPOCHS,
        "patience": base.PATIENCE,
        "gpu": base.GPU,
        "use_cpu": False,
        "no_amp": False,
        "skip_best_eval": False,
        "best_params_path": None,
        "best_trial_path": None,
        "use_dir": DEFAULT_USE_DIR,
    }
    defaults.update(config)
    return argparse.Namespace(**defaults)


def main() -> None:
    parser = build_cli_parser()
    cli_args = parser.parse_args()
    runtime_args = _build_runtime_namespace(vars(cli_args))
    run_optuna_study(runtime_args)


if __name__ == "__main__":
    main()
