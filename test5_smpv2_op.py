"""
Optuna hyperparameter tuning for test5_smpv2.py.
"""

import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import optim

import test5_smpv2 as task
import test5_smpv3_op as shared
from utils.weather_e2e import LoadWeatherEndToEndDataset

core = task.base

try:
    import optuna
    from optuna.exceptions import TrialPruned
except ImportError:
    optuna = None
    TrialPruned = None


FIX_SEED = 2026
DEFAULT_CHECKPOINTS_ROOT = "./ck52/"
DEFAULT_RESULTS_ROOT = "./rs52/"
DEFAULT_OPTUNA_OUTPUT_ROOT = "./op52/"
DEFAULT_OPTUNA_STUDY_NAME = "t5v2"
DEFAULT_OPTUNA_N_TRIALS = 100
DEFAULT_OPTUNA_TIMEOUT = 0
DEFAULT_OPTUNA_DIRECTION = "minimize"
DEFAULT_MAX_CONSECUTIVE_NONFINITE_VALI_EPOCHS = 2

DEFAULT_USE_DIR = "./optuna"
USE_BEST_PARAMS_FILE = "best_params5.json"
USE_BEST_CONFIG_FILE = "best_config5.json"
USE_BEST_WEIGHT_FILE = "best_model5.pth"
USE_BEST_TRIAL_FILE = "best_trial_result5.json"

DEFAULT_OPTUNA_SEARCH_SPACE: Dict[str, Any] = {
    "SIMILAR_DAY_TOP_K": {"type": "int", "low": 1, "high": 4, "step": 1},
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

TUNABLE_PARAM_MAP = dict(task.TUNABLE_PARAM_MAP)

_ensure_optuna_available = shared._ensure_optuna_available
_set_random_seed = shared._set_random_seed
_to_builtin = shared._to_builtin
_load_json_file = shared._load_json_file
_save_json_file = shared._save_json_file
_normalize_search_spec = shared._normalize_search_spec
_suggest_from_spec = shared._suggest_from_spec
_load_search_space = shared._load_search_space
_default_storage_uri = shared._default_storage_uri
_build_best_trial_payload = shared._build_best_trial_payload

_ORIGINAL_BUILD_SIMILAR_DAY_PRIOR_CACHE = LoadWeatherEndToEndDataset._build_similar_day_prior_cache


def _cached_build_similar_day_prior_cache(self) -> None:
    artifact_dir = self._resolve_similar_day_artifact_dir()
    cache_path = Path(self._get_similar_day_prior_cache_path(artifact_dir))
    expected_shape = (len(self), self.pred_len, self.similar_day_top_k + 1)

    if cache_path.exists():
        try:
            cached_prior = np.load(cache_path, allow_pickle=False)
            cached_prior = np.asarray(cached_prior, dtype=np.float32)
            if cached_prior.shape == expected_shape:
                self.similar_day_prior_cache = np.ascontiguousarray(cached_prior)
                mem_mb = self.similar_day_prior_cache.nbytes / (1024 ** 2)
                print(
                    f"[dataset-{self.set_type}] loaded similar-day prior cache from disk: "
                    f"{cache_path} | shape={self.similar_day_prior_cache.shape}, mem={mem_mb:.1f} MB"
                )
                return
            print(
                f"[dataset-{self.set_type}] ignore stale similar-day prior cache: "
                f"{cache_path} | cached_shape={cached_prior.shape}, expected={expected_shape}"
            )
        except Exception as exc:
            print(f"[dataset-{self.set_type}] failed to load similar-day prior cache {cache_path}: {exc}")

    _ORIGINAL_BUILD_SIMILAR_DAY_PRIOR_CACHE(self)

    if self.similar_day_prior_cache is None:
        return
    try:
        np.save(cache_path, np.ascontiguousarray(self.similar_day_prior_cache), allow_pickle=False)
        print(f"[dataset-{self.set_type}] saved similar-day prior cache to disk: {cache_path}")
    except Exception as exc:
        print(f"[dataset-{self.set_type}] failed to save similar-day prior cache {cache_path}: {exc}")


if not getattr(LoadWeatherEndToEndDataset._build_similar_day_prior_cache, "_op_cache_patched", False):
    _cached_build_similar_day_prior_cache._op_cache_patched = True
    LoadWeatherEndToEndDataset._build_similar_day_prior_cache = _cached_build_similar_day_prior_cache


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
    selected_weather_source = weather_source or core.DEFAULT_WEATHER_SOURCE
    selected_weather_h5_specs = task._resolve_weather_h5_specs(selected_weather_source)
    return {
        "task_name": core.TASK_NAME,
        "is_training": 1,
        "model_id": f"{core.MODEL_ID_PREFIX}_sdv2",
        "model": core.MODEL,
        "des": core.DES,
        "itr": core.ITR,
        "data": "custom",
        "root_path": core.ROOT_PATH,
        "data_path": core.DATA_PATH,
        "future_path": core.FUTURE_PATH,
        "features": core.FEATURES,
        "target": core.TARGET,
        "target_channel_idx": 0,
        "freq": core.LOAD_FREQ,
        "embed": "timeF",
        "checkpoints": DEFAULT_CHECKPOINTS_ROOT,
        "results_root": DEFAULT_RESULTS_ROOT,
        "seq_len": core.SEQ_LEN,
        "label_len": core.LABEL_LEN,
        "pred_len": core.PRED_LEN,
        "enc_in": core.ENC_IN,
        "c_out": core.C_OUT,
        "d_model": core.D_MODEL,
        "n_heads": core.N_HEADS,
        "e_layers": core.E_LAYERS,
        "d_ff": core.D_FF,
        "factor": core.FACTOR,
        "dropout": core.DROPOUT,
        "activation": core.ACTIVATION,
        "patch_len": core.PATCH_LEN,
        "use_norm": core.USE_NORM,
        "weather_source": selected_weather_source,
        "weather_h5_specs": selected_weather_h5_specs,
        "weather_in_channels": core.WEATHER_IN_CHANNELS,
        "weather_feature_dim": core.WEATHER_FEATURE_DIM,
        "weather_grid_height": core.WEATHER_GRID_HEIGHT,
        "weather_grid_width": core.WEATHER_GRID_WIDTH,
        "weather_kernel_height": core.WEATHER_KERNEL_HEIGHT,
        "weather_kernel_width": core.WEATHER_KERNEL_WIDTH,
        "weather_encode_chunk_size": core.WEATHER_ENCODE_CHUNK_SIZE,
        "use_weather_normalization": True,
        "num_workers": core.NUM_WORKERS,
        "pin_memory": core.PIN_MEMORY,
        "contiguous_train_batches": core.CONTIGUOUS_TRAIN_BATCHES,
        "train_epochs": core.TRAIN_EPOCHS,
        "batch_size": core.BATCH_SIZE,
        "patience": core.PATIENCE,
        "max_consecutive_nonfinite_vali_epochs": DEFAULT_MAX_CONSECUTIVE_NONFINITE_VALI_EPOCHS,
        "learning_rate": core.LEARNING_RATE,
        "loss": "Quantile",
        "lradj": "cosine",
        "use_amp": getattr(core, "USE_AMP", True),
        "inverse_eval": core.INVERSE_EVAL,
        "use_gpu": core.USE_GPU,
        "gpu": core.GPU,
        "use_multi_gpu": False,
        "devices": "0,1,2,3",
        "quantiles": list(core.QUANTILES),
        "n_quantiles": core.N_QUANTILES,
        "load_weight_path": None,
        "use_similar_day_prior": task.USE_SIMILAR_DAY_PRIOR,
        "similar_day_top_k": task.SIMILAR_DAY_TOP_K,
        "similar_day_artifact_dir": task.SIMILAR_DAY_ARTIFACT_DIR,
        "similar_day_gate_hidden_dim": task.SIMILAR_DAY_GATE_HIDDEN_DIM,
        "similar_day_gate_init_beta": task.SIMILAR_DAY_GATE_INIT_BETA,
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
        "max_consecutive_nonfinite_vali_epochs",
        "gpu",
        "n_quantiles",
        "similar_day_top_k",
        "similar_day_gate_hidden_dim",
    ]
    for field in int_fields:
        config[field] = int(config[field])

    for field in ["dropout", "learning_rate", "similar_day_gate_init_beta"]:
        config[field] = float(config[field])

    for field in [
        "use_weather_normalization",
        "pin_memory",
        "contiguous_train_batches",
        "use_amp",
        "inverse_eval",
        "use_gpu",
        "use_multi_gpu",
        "use_similar_day_prior",
    ]:
        config[field] = bool(config[field])

    config["quantiles"] = list(config["quantiles"])
    config["n_quantiles"] = len(config["quantiles"])

    if config["d_model"] % config["n_heads"] != 0:
        raise ValueError(f"d_model must be divisible by n_heads: {config['d_model']} vs {config['n_heads']}")
    if config["patch_len"] <= 0 or config["patch_len"] > config["seq_len"]:
        raise ValueError(f"patch_len must be in (0, seq_len], got {config['patch_len']}")
    if config["use_similar_day_prior"] and config["similar_day_top_k"] <= 0:
        raise ValueError(f"similar_day_top_k must be positive, got {config['similar_day_top_k']}")
    if config["similar_day_gate_hidden_dim"] <= 0:
        raise ValueError(
            f"similar_day_gate_hidden_dim must be positive, got {config['similar_day_gate_hidden_dim']}"
        )

    return argparse.Namespace(**config)


def _select_device(args) -> torch.device:
    if torch.cuda.is_available() and args.use_gpu:
        torch.backends.cudnn.benchmark = True
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU: cuda:{args.gpu}")
        print("cuDNN benchmark: on")
        return device
    print("Using CPU")
    return torch.device("cpu")


def _slice_batch_y_target_cpu(batch_y: torch.Tensor, pred_len: int) -> torch.Tensor:
    return task.extract_target(batch_y[:, -pred_len:, :]).contiguous()


def validate_quantile(model, data_loader, criterion, args, device, use_amp: bool = False) -> float:
    model.eval()
    total_loss = []
    use_non_blocking = task._use_non_blocking_transfer(args, device)

    with torch.inference_mode():
        for batch in data_loader:
            (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_exo_mark,
                batch_weather_frames,
                batch_weather_index,
                similar_day_prior,
            ) = task._unpack_weather_batch(batch)

            batch_y_target = task._to_float_device(
                _slice_batch_y_target_cpu(batch_y, args.pred_len),
                device,
                non_blocking=use_non_blocking,
            )
            batch_x = task._to_float_device(batch_x, device, non_blocking=use_non_blocking)
            batch_x_mark = task._to_float_device(batch_x_mark, device, non_blocking=use_non_blocking)
            batch_exo_mark = task._to_float_device(batch_exo_mark, device, non_blocking=use_non_blocking)
            batch_weather_frames = task._to_float_device(batch_weather_frames, device, non_blocking=use_non_blocking)
            batch_weather_index = task._to_long_device(batch_weather_index, device, non_blocking=use_non_blocking)
            if similar_day_prior is not None:
                similar_day_prior = task._to_float_device(similar_day_prior, device, non_blocking=use_non_blocking)

            with torch.amp.autocast("cuda", enabled=use_amp):
                model_kwargs = {
                    "load_x": batch_x,
                    "x_mark_enc": batch_x_mark,
                    "x_exo_mark": batch_exo_mark,
                    "weather_x": batch_weather_frames,
                    "weather_x_index": batch_weather_index,
                }
                if similar_day_prior is not None:
                    model_kwargs["similar_day_prior"] = similar_day_prior
                outputs = model(**model_kwargs)
                loss = criterion(outputs, batch_y_target)
            total_loss.append(loss.item())

    model.train()
    return float(np.average(total_loss)) if total_loss else np.nan


def _sample_trial_params(trial, search_space: Dict[str, Any], base_args: argparse.Namespace) -> Dict[str, Any]:
    unknown_keys = [key for key in search_space if key not in TUNABLE_PARAM_MAP]
    if unknown_keys:
        raise ValueError(f"Unknown search-space keys: {unknown_keys}")

    sampled: Dict[str, Any] = {}
    discrete_int_keys = {
        "SIMILAR_DAY_TOP_K",
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

    current_top_k = int(sampled.get("SIMILAR_DAY_TOP_K", base_args.similar_day_top_k))
    if current_top_k <= 0:
        raise TrialPruned(f"Invalid similar_day_top_k: {current_top_k}")

    return sampled


def train_quantile_model(model, args, device, weather_store: task.WeatherGridStore, trial=None):
    _, train_loader = task.weather_data_provider(args, "train", weather_store)
    _, vali_loader = task.weather_data_provider(args, "val", weather_store)

    setting = task._get_setting(args)
    path = os.path.join(args.checkpoints, setting)
    os.makedirs(path, exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = task.QuantileLoss(args.quantiles).to(device)
    early_stopping = task.EarlyStopping(patience=args.patience, verbose=True)

    use_amp = bool(getattr(args, "use_amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    use_non_blocking = task._use_non_blocking_transfer(args, device)
    best_vali_loss = float("inf")
    consecutive_nonfinite_vali_epochs = 0
    max_consecutive_nonfinite_vali_epochs = max(
        1,
        int(getattr(args, "max_consecutive_nonfinite_vali_epochs", DEFAULT_MAX_CONSECUTIVE_NONFINITE_VALI_EPOCHS)),
    )

    print("\n" + "=" * 72)
    print("Start Optuna trial training: TimeXer-primary + similar-day prior-correction")
    print(f"setting: {setting}")
    print(f"quantiles: {args.quantiles}")
    print(f"weather_feature_dim: {args.weather_feature_dim}")
    print(f"weather_kernel_size: ({args.weather_kernel_height}, {args.weather_kernel_width})")
    print(
        f"weather_seq_len: {args.weather_seq_len} "
        f"(history={args.weather_history_len}, future={args.weather_seq_len - args.weather_history_len}, "
        f"step={getattr(args, 'weather_step_freq', 'native')})"
    )
    print(f"use_similar_day_prior: {bool(getattr(args, 'use_similar_day_prior', False))}")
    if bool(getattr(args, "use_similar_day_prior", False)):
        print(
            "similar_day_gate_config: "
            f"top_k={getattr(args, 'similar_day_top_k', 0)}, "
            f"gate_hidden_dim={getattr(args, 'similar_day_gate_hidden_dim', 0)}, "
            f"gate_init_beta={float(getattr(args, 'similar_day_gate_init_beta', 0.0)):.3f}, "
            f"artifact_dir={getattr(args, 'similar_day_artifact_dir', None)}"
        )
    print(f"batch_size: {args.batch_size}")
    print(f"use_amp: {use_amp}")
    print(f"max_consecutive_nonfinite_vali_epochs: {max_consecutive_nonfinite_vali_epochs}")
    print("=" * 72)

    for epoch in range(args.train_epochs):
        model.train()
        train_loss = []
        epoch_time = time.time()

        for i, batch in enumerate(train_loader):
            (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_exo_mark,
                batch_weather_frames,
                batch_weather_index,
                similar_day_prior,
            ) = task._unpack_weather_batch(batch)

            optimizer.zero_grad(set_to_none=True)
            batch_y_target = task._to_float_device(
                _slice_batch_y_target_cpu(batch_y, args.pred_len),
                device,
                non_blocking=use_non_blocking,
            )
            batch_x = task._to_float_device(batch_x, device, non_blocking=use_non_blocking)
            batch_x_mark = task._to_float_device(batch_x_mark, device, non_blocking=use_non_blocking)
            batch_exo_mark = task._to_float_device(batch_exo_mark, device, non_blocking=use_non_blocking)
            batch_weather_frames = task._to_float_device(batch_weather_frames, device, non_blocking=use_non_blocking)
            batch_weather_index = task._to_long_device(batch_weather_index, device, non_blocking=use_non_blocking)
            if similar_day_prior is not None:
                similar_day_prior = task._to_float_device(similar_day_prior, device, non_blocking=use_non_blocking)

            with torch.amp.autocast("cuda", enabled=use_amp):
                model_kwargs = {
                    "load_x": batch_x,
                    "x_mark_enc": batch_x_mark,
                    "x_exo_mark": batch_exo_mark,
                    "weather_x": batch_weather_frames,
                    "weather_x_index": batch_weather_index,
                }
                if similar_day_prior is not None:
                    model_kwargs["similar_day_prior"] = similar_day_prior
                outputs = model(**model_kwargs)
                loss = criterion(outputs, batch_y_target)

            train_loss.append(loss.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (i + 1) % 50 == 0:
                print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")

        vali_loss = validate_quantile(model, vali_loader, criterion, args, device, use_amp=use_amp)
        train_loss_avg = float(np.average(train_loss)) if train_loss else np.nan
        print(
            f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.1f}s | "
            f"Train: {train_loss_avg:.7f} Vali: {vali_loss:.7f}"
        )

        if np.isfinite(vali_loss):
            consecutive_nonfinite_vali_epochs = 0
            best_vali_loss = min(best_vali_loss, float(vali_loss))
        else:
            consecutive_nonfinite_vali_epochs += 1
            print(
                f"Non-finite vali_loss detected at epoch {epoch + 1}: {vali_loss} "
                f"(consecutive={consecutive_nonfinite_vali_epochs}/"
                f"{max_consecutive_nonfinite_vali_epochs})"
            )
            if consecutive_nonfinite_vali_epochs >= max_consecutive_nonfinite_vali_epochs:
                message = (
                    f"Pruned trial due to {consecutive_nonfinite_vali_epochs} consecutive "
                    f"non-finite vali_loss values; latest={vali_loss}"
                )
                if trial is not None:
                    raise TrialPruned(message)
                raise RuntimeError(message)

        if trial is not None and np.isfinite(vali_loss):
            trial.report(float(vali_loss), step=epoch)
            if trial.should_prune():
                raise TrialPruned(f"Trial pruned at epoch {epoch + 1} with vali_loss={float(vali_loss):.7f}")

        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        task.adjust_learning_rate(optimizer, epoch + 1, args)

    best_model_path = os.path.join(path, "checkpoint.pth")
    if not os.path.exists(best_model_path):
        message = (
            f"No checkpoint was saved for trial setting={setting}. "
            "This usually means validation loss stayed non-finite for the whole trial."
        )
        if trial is not None:
            raise TrialPruned(message)
        raise FileNotFoundError(message)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"Loaded best model weights: {best_model_path}")
    return model, best_vali_loss


def test_quantile_model(model, args, device, test_data, test_loader) -> Dict[str, Any]:
    setting = task._get_setting(args)
    folder_path = os.path.join(getattr(args, "results_root", DEFAULT_RESULTS_ROOT), setting)
    os.makedirs(folder_path, exist_ok=True)

    preds_p50 = []
    trues = []
    quantile_preds_all = []

    use_amp = bool(getattr(args, "use_amp", False)) and device.type == "cuda"
    use_non_blocking = task._use_non_blocking_transfer(args, device)

    model.eval()
    with torch.inference_mode():
        for batch in test_loader:
            (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_exo_mark,
                batch_weather_frames,
                batch_weather_index,
                similar_day_prior,
            ) = task._unpack_weather_batch(batch)

            batch_y_target = _slice_batch_y_target_cpu(batch_y, args.pred_len)
            batch_x = task._to_float_device(batch_x, device, non_blocking=use_non_blocking)
            batch_x_mark = task._to_float_device(batch_x_mark, device, non_blocking=use_non_blocking)
            batch_exo_mark = task._to_float_device(batch_exo_mark, device, non_blocking=use_non_blocking)
            batch_weather_frames = task._to_float_device(batch_weather_frames, device, non_blocking=use_non_blocking)
            batch_weather_index = task._to_long_device(batch_weather_index, device, non_blocking=use_non_blocking)
            if similar_day_prior is not None:
                similar_day_prior = task._to_float_device(similar_day_prior, device, non_blocking=use_non_blocking)

            with torch.amp.autocast("cuda", enabled=use_amp):
                model_kwargs = {
                    "load_x": batch_x,
                    "x_mark_enc": batch_x_mark,
                    "x_exo_mark": batch_exo_mark,
                    "weather_x": batch_weather_frames,
                    "weather_x_index": batch_weather_index,
                }
                if similar_day_prior is not None:
                    model_kwargs["similar_day_prior"] = similar_day_prior
                outputs = model(**model_kwargs)

            outputs_fp32 = outputs.float()
            p50_pred = outputs_fp32[:, :, core.P50_IDX : core.P50_IDX + 1]

            quantile_preds_all.append(outputs_fp32.detach().cpu().numpy())
            preds_p50.append(p50_pred.detach().cpu().numpy())
            trues.append(batch_y_target.numpy())

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
        for qi in range(args.n_quantiles):
            q_slice = quantile_preds_all[:, :, qi : qi + 1]
            q_inv = test_data.inverse_transform_target(
                q_slice.reshape(q_shape[0] * q_shape[1], -1)
            ).reshape(q_shape[0], q_shape[1], 1)
            quantile_inv[:, :, qi] = q_inv[:, :, 0]

        np.save(os.path.join(folder_path, "pred_inv.npy"), preds_inv)
        np.save(os.path.join(folder_path, "true_inv.npy"), trues_inv)
        np.save(os.path.join(folder_path, "quantile_preds_inv.npy"), quantile_inv)

    if test_data.scale and getattr(args, "inverse_eval", False):
        mae, mse, rmse, mape, mspe = task.metric(preds_inv, trues_inv)
        print(f"P50 Test Metrics (Inverse): MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}")
    else:
        mae, mse, rmse, mape, mspe = task.metric(preds_p50, trues)
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
    weather_store: Optional[task.WeatherGridStore] = None,
    trial=None,
) -> Dict[str, Any]:
    device = _select_device(args)
    selected_weather_source = getattr(args, "weather_source", core.DEFAULT_WEATHER_SOURCE)
    result: Dict[str, Any] = {}
    owns_weather_store = weather_store is None
    if weather_store is None:
        weather_store = task.WeatherGridStore(
            args.weather_h5_specs,
            expected_in_channels=args.weather_in_channels,
            fill_value=core.WEATHER_FILL_VALUE,
            use_channel_normalization=True,
        )

    try:
        args = task._configure_runtime_weather_args(args, weather_store, selected_weather_source)
        setting = task._get_setting(args)
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

        model = task.FullMapConvTimeXerPriorCorrectionGateQuantile(
            args, quantiles=args.quantiles
        ).float().to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"TimeXer-primary + prior-correction total params: {total_params:,}")
        print(f"TimeXer-primary + prior-correction trainable params: {trainable_params:,}")

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
            test_data, test_loader = task.weather_data_provider(args, "test", weather_store)
            print(f"\n>>> Start testing {setting}")
            eval_result = test_quantile_model(model, args, device, test_data, test_loader)
            result.update(eval_result)
            results_dir = eval_result["results_dir"]

            if plot_results:
                task.plot_pred_vs_true(
                    results_dir,
                    use_inverse=args.inverse_eval,
                    quantiles=args.quantiles,
                    title_prefix="TimeXer-Primary + Similar-Day Prior-Correction Prediction",
                    y_label="Load (MW)",
                )

            if predict_future:
                similar_day_result = task.export_similar_day_baseline(
                    results_dir=results_dir,
                    future_path=getattr(args, "future_path", core.FUTURE_PATH),
                    args=args,
                    artifact_dir=getattr(args, "similar_day_artifact_dir", task.SIMILAR_DAY_ARTIFACT_DIR),
                    top_k=int(getattr(args, "similar_day_top_k", task.SIMILAR_DAY_TOP_K)),
                )
                task.predict_future_load_from_csv(
                    model=model,
                    args=args,
                    device=device,
                    weather_store=weather_store,
                    results_dir=results_dir,
                    future_path=getattr(args, "future_path", core.FUTURE_PATH),
                    steps=args.pred_len,
                    use_inverse=args.inverse_eval,
                    quantiles=args.quantiles,
                    data_provider_fn=task.weather_data_provider,
                    model_label="TimeXer-Primary + Similar-Day Prior-Correction",
                    y_label="Load (MW)",
                    similar_day_result=similar_day_result,
                )

        return result
    finally:
        if owns_weather_store:
            weather_store.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


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
    checkpoints_root = study_dir / "ck"
    results_root = study_dir / "rs"

    search_space = _load_search_space(cli_args.search_space_json, getattr(cli_args, "search_space", None))
    search_space_path = study_dir / "search_space.json"
    _save_json_file(str(search_space_path), search_space)

    runtime_overrides = {
        "checkpoints": str(checkpoints_root),
        "results_root": str(results_root),
        "train_epochs": int(cli_args.train_epochs),
        "patience": int(cli_args.patience),
        "max_consecutive_nonfinite_vali_epochs": int(cli_args.max_consecutive_nonfinite_vali_epochs),
        "use_gpu": not bool(cli_args.use_cpu),
        "gpu": int(cli_args.gpu),
        "use_amp": not bool(cli_args.no_amp),
    }
    base_args = _build_args(train_mode=True, overrides=runtime_overrides, weather_source=cli_args.weather_source)
    global_weather_store = task.WeatherGridStore(
        base_args.weather_h5_specs,
        expected_in_channels=base_args.weather_in_channels,
        fill_value=core.WEATHER_FILL_VALUE,
        use_channel_normalization=True,
    )

    try:
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
            trial_overrides["des"] = f"op{trial.number:03d}"
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
                weather_store=global_weather_store,
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
        best_overrides["des"] = f"op{best_trial.number:03d}"
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
                weather_store=global_weather_store,
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
    finally:
        global_weather_store.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning for test5_smpv2.py")
    parser.add_argument(
        "--weather-source",
        type=str,
        choices=sorted(core.WEATHER_SOURCE_CONFIGS.keys()),
        default=core.DEFAULT_WEATHER_SOURCE,
    )
    parser.add_argument("--search-space-json", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OPTUNA_OUTPUT_ROOT)
    parser.add_argument("--study-name", type=str, default=DEFAULT_OPTUNA_STUDY_NAME)
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--n-trials", type=int, default=DEFAULT_OPTUNA_N_TRIALS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_OPTUNA_TIMEOUT)
    parser.add_argument("--direction", type=str, default=DEFAULT_OPTUNA_DIRECTION)
    parser.add_argument("--seed", type=int, default=FIX_SEED)
    parser.add_argument("--train-epochs", type=int, default=core.TRAIN_EPOCHS)
    parser.add_argument("--patience", type=int, default=core.PATIENCE)
    parser.add_argument(
        "--max-consecutive-nonfinite-vali-epochs",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_NONFINITE_VALI_EPOCHS,
    )
    parser.add_argument("--gpu", type=int, default=core.GPU)
    parser.add_argument("--use-cpu", action="store_true", default=False)
    parser.add_argument("--no-amp", action="store_true", default=False)
    parser.add_argument("--skip-best-eval", action="store_true", default=False)
    parser.add_argument("--best-params-path", type=str, default=None)
    parser.add_argument("--best-trial-path", type=str, default=None)
    parser.add_argument("--use-dir", type=str, default=DEFAULT_USE_DIR)
    return parser


def _build_runtime_namespace(config: Dict[str, Any]) -> argparse.Namespace:
    defaults = {
        "weather_source": core.DEFAULT_WEATHER_SOURCE,
        "search_space": json.loads(json.dumps(DEFAULT_OPTUNA_SEARCH_SPACE)),
        "search_space_json": None,
        "output_dir": DEFAULT_OPTUNA_OUTPUT_ROOT,
        "study_name": DEFAULT_OPTUNA_STUDY_NAME,
        "storage": None,
        "n_trials": DEFAULT_OPTUNA_N_TRIALS,
        "timeout": DEFAULT_OPTUNA_TIMEOUT,
        "direction": DEFAULT_OPTUNA_DIRECTION,
        "seed": FIX_SEED,
        "train_epochs": core.TRAIN_EPOCHS,
        "patience": core.PATIENCE,
        "max_consecutive_nonfinite_vali_epochs": DEFAULT_MAX_CONSECUTIVE_NONFINITE_VALI_EPOCHS,
        "gpu": core.GPU,
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
