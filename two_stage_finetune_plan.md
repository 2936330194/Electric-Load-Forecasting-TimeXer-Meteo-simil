# 两阶段微调训练策略：冻结主干 → 训练 Gate → 联合微调

## 策略评价

**这个策略非常合理**，原因如下：

1. **`best_model3.pth` 是纯 FullMapConv + TimeXer 模型**（来自 test3 的 Optuna 调优，MAPE=3.43%），它**不包含** `similar_day_gate` 和 `quantile_head` 模块的权重。
2. 如果直接端到端训练整个 `test5_smpv2` 模型，随机初始化的 gate 网络会向 TimeXer 主干回传 **竞争性梯度**，破坏已优化好的主干特征提取能力。
3. 两阶段策略让 gate 先在 **稳定的特征空间** 上学习"纠偏策略"，之后再用小学习率做联合微调实现全局优化。

> [!TIP]
> 这本质上是 **"冻结-解冻"迁移学习** 范式在时序预测中的应用，与 CV 中常用的 backbone frozen fine-tuning 思路一致。

---

## 架构分析

### 当前模型结构（test5_smpv2.py）

```
FullMapConvTimeXerPriorCorrectionGateQuantile
├── weather_backbone (FullMapWeatherConvExtractor)  ← 来自 best_model3.pth ✓
├── timexer (TimeXer)                                ← 来自 best_model3.pth ✓
├── similar_day_gate (nn.Sequential)                 ← 新增模块，随机初始化 ✗
└── quantile_head (nn.Linear)                        ← 来自 best_model3.pth ✓
```

### best_model3.pth 的 state_dict 键映射关系

`best_model3.pth` 来自 `FullMapConvTimeXerQuantile`（test3 模型），其键名与 `FullMapConvTimeXerPriorCorrectionGateQuantile`（test5_smpv2 模型）在共享部分**完全一致**：

| test3 模型 (source) | test5_smpv2 模型 (target) | 说明 |
|---|---|---|
| `weather_backbone.*` | `weather_backbone.*` | ✅ 完全匹配 |
| `timexer.*` | `timexer.*` | ✅ 完全匹配 |
| `quantile_head.*` | `quantile_head.*` | ✅ 完全匹配 |
| ❌ 不存在 | `similar_day_gate.*` | 🆕 新增，随机初始化 |

> [!IMPORTANT]
> 需要使用 `strict=False` 加载权重，让 `similar_day_gate` 的 key 被跳过，其余共享权重完美加载。

---

## 具体实施方案

### 修改 1：新增训练阶段控制常量

在文件头部 `LOAD_FROM_OPTUNA` 附近新增以下配置：

```python
# ================= 两阶段微调配置 =================
# 是否启用从 Optuna 预训练主干加载 + 两阶段微调训练策略
ENABLE_TWO_STAGE_FINETUNE = True

# Stage 1: 冻结主干，只训练 gate + quantile_head
STAGE1_EPOCHS = 15              # 门控网络预热轮数
STAGE1_LEARNING_RATE = 1e-3     # 门控网络因为是从零开始，用较大的学习率

# Stage 2: 解冻全部参数，联合微调
STAGE2_EPOCHS = 30              # 联合微调轮数
STAGE2_LEARNING_RATE = 2e-5     # 远小于原始 Optuna LR (4.15e-4)，防止破坏主干

# 加载 Optuna test3 主干的配置
OPTUNA_BACKBONE_DIR = "./optuna"
OPTUNA_BACKBONE_PARAMS_FILE = "best_params3.json"
OPTUNA_BACKBONE_CONFIG_FILE = "best_config3.json"
OPTUNA_BACKBONE_WEIGHT_FILE = "best_model3.pth"
```

> [!NOTE]
> **学习率设计依据**：
> - Stage 1 用 `1e-3`：gate 只有 3→32→1 约 **161 个参数**，需要足够大的 LR 才能快速收敛
> - Stage 2 用 `2e-5`：约为原始 Optuna 最优 LR (`4.15e-4`) 的 **1/20**，对主干参数仅做微量调整
> - 可选方案：Stage 2 使用分组学习率，对 gate 保留 `1e-4`，主干用 `2e-5`

### 修改 2：权重加载函数

新增一个专用的主干权重加载函数：

```python
def _load_backbone_from_optuna(
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """
    从 Optuna 调优的 test3 (FullMapConvTimeXerQuantile) 模型加载
    weather_backbone + timexer + quantile_head 的权重到 test5_smpv2 模型中。
    similar_day_gate 的权重保持随机初始化。
    """
    optuna_dir = os.path.abspath(OPTUNA_BACKBONE_DIR)
    weight_path = os.path.join(optuna_dir, OPTUNA_BACKBONE_WEIGHT_FILE)
    
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Optuna backbone weight not found: {weight_path}")
    
    # 加载 test3 模型的 state_dict
    source_state = torch.load(weight_path, map_location=device)
    
    # 过滤掉 test5_smpv2 模型中不存在于 source 的 key（即 similar_day_gate.*）
    target_state = model.state_dict()
    matched_keys = []
    skipped_keys = []
    
    for key in target_state:
        if key in source_state:
            target_state[key] = source_state[key]
            matched_keys.append(key)
        else:
            skipped_keys.append(key)
    
    model.load_state_dict(target_state, strict=True)
    
    print(f"[two-stage] Loaded backbone from: {weight_path}")
    print(f"[two-stage]   Matched keys: {len(matched_keys)}")
    print(f"[two-stage]   Skipped keys (new modules): {len(skipped_keys)}")
    for k in skipped_keys:
        print(f"[two-stage]     - {k}")
```

### 修改 3：冻结/解冻工具函数

```python
def _freeze_backbone(model: nn.Module) -> None:
    """冻结 weather_backbone + timexer 的所有参数，只保留 gate + quantile_head 可训练。"""
    frozen_count = 0
    for name, param in model.named_parameters():
        if name.startswith(("weather_backbone.", "timexer.")):
            param.requires_grad = False
            frozen_count += 1
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[two-stage] STAGE 1: Frozen {frozen_count} param tensors")
    print(f"[two-stage]   Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


def _unfreeze_all(model: nn.Module) -> None:
    """解冻模型全部参数，准备联合微调。"""
    for param in model.parameters():
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[two-stage] STAGE 2: All {trainable:,} params unfrozen for joint fine-tuning")
```

### 修改 4：两阶段训练主函数

新增 `train_two_stage` 函数，替代直接调用 `train_quantile_model`：

```python
def train_two_stage(model, args, device, weather_store: WeatherGridStore):
    """
    两阶段训练策略：
    Stage 1 — 冻结 Optuna 预训练主干，仅训练 similar_day_gate + quantile_head
    Stage 2 — 解冻全部参数，用小学习率联合微调
    """
    _, train_loader = weather_data_provider(args, "train", weather_store)
    _, vali_loader = weather_data_provider(args, "val", weather_store)
    _, test_loader = weather_data_provider(args, "test", weather_store)
    
    setting = _get_setting(args)
    path = os.path.join(args.checkpoints, setting)
    os.makedirs(path, exist_ok=True)
    
    criterion = QuantileLoss(args.quantiles).to(device)
    use_amp = bool(getattr(args, "use_amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    use_non_blocking = _use_non_blocking_transfer(args, device)
    best_vali_loss = np.inf
    
    # ==================== STAGE 1: Gate-only Training ====================
    print("\n" + "=" * 72)
    print("STAGE 1: Freeze backbone, train gate + quantile_head only")
    print("=" * 72)
    
    _freeze_backbone(model)
    
    # 只收集可训练的参数（gate + quantile_head）
    stage1_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(stage1_params, lr=STAGE1_LEARNING_RATE)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    for epoch in range(STAGE1_EPOCHS):
        model.train()
        train_loss = []
        epoch_time = time.time()
        
        for i, batch in enumerate(train_loader):
            # [标准训练循环，与现有 train_quantile_model 相同]
            ...
        
        vali_loss = validate_quantile(model, vali_loader, criterion, args, device, use_amp=use_amp)
        test_loss = validate_quantile(model, test_loader, criterion, args, device, use_amp=use_amp)
        train_loss_avg = float(np.average(train_loss)) if train_loss else np.nan
        
        print(
            f"[Stage1] Epoch: {epoch + 1} cost: {time.time() - epoch_time:.1f}s | "
            f"Train: {train_loss_avg:.7f} Vali: {vali_loss:.7f} Test: {test_loss:.7f}"
        )
        
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("[Stage1] Early stopping")
            break
        
        # Stage 1 使用固定学习率，不衰减（参数少，收敛快）
    
    # 加载 Stage 1 最佳权重
    best_s1_path = os.path.join(path, "checkpoint.pth")
    if os.path.exists(best_s1_path):
        model.load_state_dict(torch.load(best_s1_path, map_location=device))
        best_vali_loss = early_stopping.val_loss_min
        print(f"[Stage1] Best vali loss: {best_vali_loss:.7f}")
    
    # ==================== STAGE 2: Joint Fine-tuning ====================
    print("\n" + "=" * 72)
    print("STAGE 2: Unfreeze all, joint fine-tuning with reduced LR")
    print("=" * 72)
    
    _unfreeze_all(model)
    
    # 分组学习率：主干用更小的 LR，gate 保留较高 LR
    optimizer = optim.Adam([
        {"params": [p for n, p in model.named_parameters() 
                    if n.startswith(("weather_backbone.", "timexer."))],
         "lr": STAGE2_LEARNING_RATE},
        {"params": [p for n, p in model.named_parameters() 
                    if n.startswith(("similar_day_gate.", "quantile_head."))],
         "lr": STAGE2_LEARNING_RATE * 5},  # gate 用 5x 主干学习率
    ])
    
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    # 用 Stage 1 的最佳 loss 作为起点，避免 Stage 2 保存更差的 checkpoint
    early_stopping.best_score = -best_vali_loss
    early_stopping.val_loss_min = best_vali_loss
    
    for epoch in range(STAGE2_EPOCHS):
        model.train()
        train_loss = []
        epoch_time = time.time()
        
        for i, batch in enumerate(train_loader):
            # [标准训练循环，与现有 train_quantile_model 相同]
            ...
        
        vali_loss = validate_quantile(model, vali_loader, criterion, args, device, use_amp=use_amp)
        test_loss = validate_quantile(model, test_loader, criterion, args, device, use_amp=use_amp)
        train_loss_avg = float(np.average(train_loss)) if train_loss else np.nan
        
        print(
            f"[Stage2] Epoch: {epoch + 1} cost: {time.time() - epoch_time:.1f}s | "
            f"Train: {train_loss_avg:.7f} Vali: {vali_loss:.7f} Test: {test_loss:.7f}"
        )
        
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("[Stage2] Early stopping")
            break
        
        # Stage 2 使用 cosine 学习率衰减
        # 需要将 args.train_epochs 临时设为 STAGE2_EPOCHS 以正确计算 cosine 周期
        _stage2_args = argparse.Namespace(**vars(args))
        _stage2_args.train_epochs = STAGE2_EPOCHS
        _stage2_args.learning_rate = STAGE2_LEARNING_RATE
        adjust_learning_rate(optimizer, epoch + 1, _stage2_args)
    
    # 加载全局最佳权重
    best_model_path = os.path.join(path, "checkpoint.pth")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"[two-stage] Loaded best model: {best_model_path}")
    return model
```

### 修改 5：main() 函数中的调用逻辑

修改 `main()` 中的训练分支（约第 817-819 行）：

```python
if args.is_training:
    if ENABLE_TWO_STAGE_FINETUNE:
        print("\n>>> Two-stage fine-tuning mode enabled")
        _load_backbone_from_optuna(model, args, device)
        model = train_two_stage(model, args, device, weather_store)
    else:
        print(f"\n>>> Start training {setting}")
        model = train_quantile_model(model, args, device, weather_store)
    
    print(f"\n>>> Start testing {setting}")
    results_dir = test_quantile_model(model, args, device, weather_store)
```

---

## 关键技术要点

### 1. quantile_head 的处理策略

> [!WARNING]
> `best_model3.pth` 中的 `quantile_head` 是在 **没有相似日纠偏** 的条件下训练的。加载后，其权重对 gate 输出的校正信号并不敏感。

**推荐做法**：Stage 1 同时训练 `quantile_head` + `similar_day_gate`。
- `quantile_head` 有预训练权重作为初始值（映射关系 `y → quantiles` 基本稳定）
- 只需微调适应 gate 引入的微小偏移

### 2. adjust_learning_rate 的兼容性

当前 `adjust_learning_rate` 对**所有 param_group 统一调度**。在 Stage 2 使用分组 LR 时，cosine 衰减会按比例缩减所有组的 LR（因为它用乘法因子），这正好是我们想要的效果。

### 3. 训练数据一致性

`best_model3.pth` 是在不含 `similar_day_prior` 的 Dataset 上训练的，而 `test5_smpv2` 的 Dataset 会额外返回第 7 个元素。这**不影响权重加载**，因为 `similar_day_prior` 是 forward 时的输入，不是 state_dict 的一部分。

### 4. Gate 参数量统计

```
similar_day_gate:
  Linear(3, 32):  3×32 + 32 = 128 params
  GELU:           0
  Dropout:        0
  Linear(32, 1):  32×1 + 1  = 33 params
  Sigmoid:        0
  ─────────────────────────────────
  Total gate:     161 params

quantile_head:
  Linear(1, 7):   1×7 + 7 = 14 params

Stage 1 可训练参数: ~175 params (相对于模型总量约 500K~800K)
```

---

## 预期效果

| 指标 | test3 基线 (Optuna) | 预期 test5_smpv2 |
|------|---------------------|------------------|
| P50 MAPE | 3.43% | **3.0%~3.2%** |
| 训练稳定性 | ✅ 稳定 | ✅ 两阶段避免梯度冲突 |
| 训练时间 | ~50 epochs | Stage1: ~10-15 epochs + Stage2: ~10-20 epochs |

> [!CAUTION]
> 如果 Stage 2 联合微调后性能反而下降（overfitting），建议：
> 1. 减小 `STAGE2_LEARNING_RATE` 至 `1e-5`
> 2. 增大 Stage 2 的 `patience` 至 8-10
> 3. 或者完全跳过 Stage 2，仅使用 Stage 1 的结果

---

## 需要您确认的决策

1. **Stage 1 是否也训练 `quantile_head`？**（推荐：是，因为 gate 改变了 point_pred 分布）
2. **Stage 2 是否使用分组学习率？**（推荐：是，gate 用 `5x` 主干 LR）
3. **是否需要保留原有的 `train_quantile_model` 作为 fallback？**（推荐：是，通过 `ENABLE_TWO_STAGE_FINETUNE` 开关控制）
4. **超参数微调**：Stage 1/2 的 epochs 和 LR 是否需要调整？

请确认后我将立即实施代码修改。
