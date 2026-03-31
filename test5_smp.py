"""
test5_smp.py - Retrieval-augmented Full-Map Conv + TimeXer quantile forecast

基于新版 test4_smp.py 的 1h/15min 自适应天气流程，增加“相似日先验残差修正”：
1. 训练/验证/测试阶段由数据集自动检索相似日负荷曲线，并整理为 similar_day_prior。
2. 模型先输出常规 point_pred，再与 similar_day_prior 拼接后经小型 MLP 做残差修正。
3. 相似日检索库会按当前天气 H5 自动切换到对应的 1h / 15min artifact。
"""

import argparse
import hashlib
import os
import random
import time
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import optim

import test4_smp as base
from utils.forecast_visualization import plot_pred_vs_true, predict_future_load_from_csv
from utils.metrics import metric
from utils.quantile import QuantileLoss
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.weather_e2e import FullMapConvTimeXerQuantile, WeatherGridStore, weather_data_provider


SIMILAR_DAY_ARTIFACT_DIR: Optional[str] = None
SIMILAR_DAY_TOP_K = 3
USE_SIMILAR_DAY_PRIOR = True
SIMILAR_DAY_FUSION_HIDDEN_DIM = 128


_use_non_blocking_transfer = base._use_non_blocking_transfer
_to_float_device = base._to_float_device
_to_long_device = base._to_long_device
extract_target = base.extract_target
_parse_cli_args = base._parse_cli_args
_resolve_weather_h5_specs = base._resolve_weather_h5_specs
_configure_runtime_weather_args = base._configure_runtime_weather_args
export_similar_day_baseline = base.export_similar_day_baseline


def _unpack_weather_batch(
    batch: Sequence[torch.Tensor],
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
]:
    if len(batch) == 6:
        batch_x, batch_y, batch_x_mark, batch_exo_mark, batch_weather_frames, batch_weather_index = batch
        return (
            batch_x,
            batch_y,
            batch_x_mark,
            batch_exo_mark,
            batch_weather_frames,
            batch_weather_index,
            None,
        )
    if len(batch) == 7:
        (
            batch_x,
            batch_y,
            batch_x_mark,
            batch_exo_mark,
            batch_weather_frames,
            batch_weather_index,
            similar_day_prior,
        ) = batch
        return (
            batch_x,
            batch_y,
            batch_x_mark,
            batch_exo_mark,
            batch_weather_frames,
            batch_weather_index,
            similar_day_prior,
        )
    raise ValueError(f"Unexpected batch size: expected 6 or 7 tensors, got {len(batch)}")


def validate_quantile(model, data_loader, criterion, args, device, use_amp: bool = False) -> float:
    model.eval()
    total_loss = []
    use_non_blocking = _use_non_blocking_transfer(args, device)

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
            ) = _unpack_weather_batch(batch)

            batch_x = _to_float_device(batch_x, device, non_blocking=use_non_blocking)
            batch_y = _to_float_device(batch_y, device, non_blocking=use_non_blocking)
            batch_x_mark = _to_float_device(batch_x_mark, device, non_blocking=use_non_blocking)
            batch_exo_mark = _to_float_device(batch_exo_mark, device, non_blocking=use_non_blocking)
            batch_weather_frames = _to_float_device(batch_weather_frames, device, non_blocking=use_non_blocking)
            batch_weather_index = _to_long_device(batch_weather_index, device, non_blocking=use_non_blocking)
            if similar_day_prior is not None:
                similar_day_prior = _to_float_device(similar_day_prior, device, non_blocking=use_non_blocking)

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
                batch_y_target = extract_target(batch_y[:, -args.pred_len :, :])
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
    criterion = QuantileLoss(args.quantiles).to(device)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    use_amp = bool(getattr(args, "use_amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    use_non_blocking = _use_non_blocking_transfer(args, device)

    print("\n" + "=" * 72)
    print("Start training Retrieval-Augmented Full-Map Conv + TimeXer quantile model")
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
            "similar_day_prior_config: "
            f"top_k={getattr(args, 'similar_day_top_k', 0)}, "
            f"fusion_hidden_dim={getattr(args, 'similar_day_fusion_hidden_dim', 0)}, "
            f"artifact_dir={getattr(args, 'similar_day_artifact_dir', None)}"
        )
    print(f"batch_size: {args.batch_size}")
    print(f"use_amp: {use_amp}")
    if bool(getattr(args, "contiguous_train_batches", False)):
        dense_weather_frames = args.batch_size * args.weather_seq_len
        print(f"overlap-aware weather batching: on (dense {dense_weather_frames} exogenous frames/batch)")
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
            ) = _unpack_weather_batch(batch)

            optimizer.zero_grad(set_to_none=True)
            batch_x = _to_float_device(batch_x, device, non_blocking=use_non_blocking)
            batch_y = _to_float_device(batch_y, device, non_blocking=use_non_blocking)
            batch_x_mark = _to_float_device(batch_x_mark, device, non_blocking=use_non_blocking)
            batch_exo_mark = _to_float_device(batch_exo_mark, device, non_blocking=use_non_blocking)
            batch_weather_frames = _to_float_device(batch_weather_frames, device, non_blocking=use_non_blocking)
            batch_weather_index = _to_long_device(batch_weather_index, device, non_blocking=use_non_blocking)
            if similar_day_prior is not None:
                similar_day_prior = _to_float_device(similar_day_prior, device, non_blocking=use_non_blocking)

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
                batch_y_target = extract_target(batch_y[:, -args.pred_len :, :])
                loss = criterion(outputs, batch_y_target)

            train_loss.append(loss.item())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (i + 1) % 50 == 0:
                print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")

        vali_loss = validate_quantile(model, vali_loader, criterion, args, device, use_amp=use_amp)
        test_loss = validate_quantile(model, test_loader, criterion, args, device, use_amp=use_amp)
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


def test_quantile_model(model, args, device, weather_store: WeatherGridStore) -> str:
    test_data, test_loader = weather_data_provider(args, "test", weather_store)

    setting = _get_setting(args)
    folder_path = os.path.join("./results/", setting)
    os.makedirs(folder_path, exist_ok=True)

    preds_p50 = []
    trues = []
    quantile_preds_all = []

    use_amp = bool(getattr(args, "use_amp", False)) and device.type == "cuda"
    use_non_blocking = _use_non_blocking_transfer(args, device)

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
            ) = _unpack_weather_batch(batch)

            batch_x = _to_float_device(batch_x, device, non_blocking=use_non_blocking)
            batch_y = _to_float_device(batch_y, device, non_blocking=use_non_blocking)
            batch_x_mark = _to_float_device(batch_x_mark, device, non_blocking=use_non_blocking)
            batch_exo_mark = _to_float_device(batch_exo_mark, device, non_blocking=use_non_blocking)
            batch_weather_frames = _to_float_device(batch_weather_frames, device, non_blocking=use_non_blocking)
            batch_weather_index = _to_long_device(batch_weather_index, device, non_blocking=use_non_blocking)
            if similar_day_prior is not None:
                similar_day_prior = _to_float_device(similar_day_prior, device, non_blocking=use_non_blocking)

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

            batch_y_target = extract_target(batch_y[:, -args.pred_len :, :])
            p50_pred = outputs.float()[:, :, base.P50_IDX : base.P50_IDX + 1]

            quantile_preds_all.append(outputs.float().detach().cpu().numpy())
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
        for qi in range(base.N_QUANTILES):
            q_slice = quantile_preds_all[:, :, qi : qi + 1]
            q_inv = test_data.inverse_transform_target(
                q_slice.reshape(q_shape[0] * q_shape[1], -1)
            ).reshape(q_shape[0], q_shape[1], 1)
            quantile_inv[:, :, qi] = q_inv[:, :, 0]

        np.save(os.path.join(folder_path, "pred_inv.npy"), preds_inv)
        np.save(os.path.join(folder_path, "true_inv.npy"), trues_inv)
        np.save(os.path.join(folder_path, "quantile_preds_inv.npy"), quantile_inv)

    if test_data.scale and getattr(args, "inverse_eval", False):
        mae, mse, rmse, mape, mspe = metric(preds_inv, trues_inv)
        print(f"P50 Test Metrics (Inverse): MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}")
    else:
        mae, mse, rmse, mape, mspe = metric(preds_p50, trues)
        print(f"P50 Test Metrics (Normalized): MSE={mse:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}")

    return folder_path


def _get_setting(args, itr: int = 0) -> str:
    signature = (
        f"{args.task_name}_{args.model_id}_{args.model}_e2e_"
        f"sl{args.seq_len}_pl{args.pred_len}_dm{args.d_model}_"
        f"el{args.e_layers}_wd{args.weather_feature_dim}_"
        f"wsl{args.weather_seq_len}_wh{args.weather_history_len}_"
        f"wk{args.weather_kernel_height}x{args.weather_kernel_width}_"
        f"sdp{int(bool(getattr(args, 'use_similar_day_prior', False)))}_"
        f"sdk{int(getattr(args, 'similar_day_top_k', 0))}_"
        f"sdfh{int(getattr(args, 'similar_day_fusion_hidden_dim', 0))}_"
        f"lr{args.learning_rate}_bs{args.batch_size}_{args.des}_{itr}"
    )
    digest = hashlib.md5(signature.encode("utf-8")).hexdigest()[:8]
    return (
        f"TimeXerE2E_sl{args.seq_len}_pl{args.pred_len}_"
        f"wd{args.weather_feature_dim}_"
        f"wsl{args.weather_seq_len}_wh{args.weather_history_len}_"
        f"sdp{int(bool(getattr(args, 'use_similar_day_prior', False)))}_"
        f"sdk{int(getattr(args, 'similar_day_top_k', 0))}_"
        f"wk{args.weather_kernel_height}x{args.weather_kernel_width}_"
        f"bs{args.batch_size}_{args.des}_{itr}_{digest}"
    )


def main() -> None:
    fix_seed = 2026
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    cli_args = _parse_cli_args()
    selected_weather_source = cli_args.weather_source
    selected_weather_h5_specs = _resolve_weather_h5_specs(selected_weather_source)

    args = argparse.Namespace(
        task_name=base.TASK_NAME,
        is_training=1 if base.TRAIN_MODE else 0,
        model_id=base.MODEL_ID_PREFIX,
        model=base.MODEL,
        des=base.DES,
        itr=base.ITR,
        data="custom",
        root_path=base.ROOT_PATH,
        data_path=base.DATA_PATH,
        features=base.FEATURES,
        target=base.TARGET,
        target_channel_idx=0,
        freq=base.LOAD_FREQ,
        embed="timeF",
        checkpoints="./checkpoints_test5/",
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
        weather_source=selected_weather_source,
        weather_h5_specs=selected_weather_h5_specs,
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
        use_similar_day_prior=USE_SIMILAR_DAY_PRIOR,
        similar_day_top_k=SIMILAR_DAY_TOP_K,
        similar_day_artifact_dir=SIMILAR_DAY_ARTIFACT_DIR,
        similar_day_fusion_hidden_dim=SIMILAR_DAY_FUSION_HIDDEN_DIM,
    )

    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Using GPU: cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    weather_store = WeatherGridStore(
        args.weather_h5_specs,
        expected_in_channels=args.weather_in_channels,
        fill_value=base.WEATHER_FILL_VALUE,
        use_channel_normalization=True,
    )
    try:
        args = _configure_runtime_weather_args(args, weather_store, selected_weather_source)

        if weather_store.frame_shape is None:
            raise RuntimeError("weather_store.frame_shape is not initialized.")
        _, frame_height, frame_width = weather_store.frame_shape
        if (frame_height, frame_width) != (args.weather_kernel_height, args.weather_kernel_width):
            raise ValueError(
                "Weather frame size does not match full-map kernel size: "
                f"frame=({frame_height}, {frame_width}), "
                f"kernel=({args.weather_kernel_height}, {args.weather_kernel_width})"
            )

        model = FullMapConvTimeXerQuantile(args, quantiles=base.QUANTILES).float().to(device)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Retrieval-Augmented Full-Map Conv + TimeXer total params: {total_params:,}")
        print(f"Retrieval-Augmented Full-Map Conv + TimeXer trainable params: {trainable_params:,}")

        setting = _get_setting(args)
        if base.TRAIN_MODE:
            print(f"\n>>> Start training {setting}")
            model = train_quantile_model(model, args, device, weather_store)

            print(f"\n>>> Start testing {setting}")
            results_dir = test_quantile_model(model, args, device, weather_store)
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

        plot_pred_vs_true(
            results_dir,
            use_inverse=base.INVERSE_EVAL,
            quantiles=args.quantiles,
            title_prefix="Retrieval-Augmented Full-Map Conv + TimeXer Prediction",
            y_label="Load (MW)",
        )

        similar_day_result = export_similar_day_baseline(
            results_dir=results_dir,
            future_path=base.FUTURE_PATH,
            args=args,
            artifact_dir=SIMILAR_DAY_ARTIFACT_DIR,
            top_k=SIMILAR_DAY_TOP_K,
        )
        predict_future_load_from_csv(
            model=model,
            args=args,
            device=device,
            weather_store=weather_store,
            results_dir=results_dir,
            future_path=base.FUTURE_PATH,
            steps=base.PRED_LEN,
            use_inverse=base.INVERSE_EVAL,
            quantiles=args.quantiles,
            data_provider_fn=weather_data_provider,
            model_label="Retrieval-Augmented Full-Map Conv + TimeXer",
            y_label="Load (MW)",
            similar_day_result=similar_day_result,
        )
    finally:
        weather_store.close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
