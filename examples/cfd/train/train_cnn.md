# CNN 训练脚本 (DeepCFD / CFDBench)

适用于基于卷积神经网络的 CFD 训练脚本，包括 DeepCFD、CFDBench。

---

## 1. 核心特征

CNN 训练脚本的特点：
- 配置整体加载：`YParams(path, "root")`
- 数据为标准 tensor batch (`batch["x"]`, `batch["y"]`)
- 损失函数可能自定义（如 DeepCFD 的通道加权损失）
- 早停通过分布式 `broadcast` 同步 stop_flag
- 结构最简洁，最接近标准 PyTorch 训练流程

---

## 2. 配置加载

```python
config_path = "conf/deepcfd.yaml"
cfg = YParams(config_path, "root")

# 直接访问
# cfg.datapipe.source.data_dir
# cfg.model.name
# cfg.training.num_epochs
```

---

## 3. 模型初始化

### DeepCFD — 简单工厂函数

```python
def init_model(cfg):
    model_name = cfg.model.name
    if model_name == "UNet":
        from onescience.models.deepcfd.UNet import UNet
        net_class = UNet
    elif model_name == "UNetEx":
        from onescience.models.deepcfd.UNetEx import UNetEx
        net_class = UNetEx
    else:
        raise ValueError(f"Unknown network: {model_name}")

    return net_class(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        base_channels=cfg.model.base_channels,
        num_stages=cfg.model.num_stages,
        bilinear=cfg.model.bilinear,
        normtype=cfg.model.normtype,
        kernel_size=cfg.model.kernel_size,
    )

model = init_model(cfg).to(device)
```

### CFDBench — 多模型选择

CFDBench 在代码中根据 `cfg.model.name` 动态导入并实例化不同模型（DeepONet, FFN, FNO, UNet, ResNet 等）。

---

## 4. 自定义损失函数 (DeepCFD)

DeepCFD 使用通道加权损失：

```python
def loss_func(output, target, weights):
    """
    通道加权 MSE + Abs Error
    output/target shape: (B, 3, H, W)  — Channel 0: Ux, 1: Uy, 2: p
    weights shape: (1, 3, 1, 1)
    """
    lossu = ((output[:, 0] - target[:, 0]) ** 2)
    lossv = ((output[:, 1] - target[:, 1]) ** 2)
    lossp = torch.abs(output[:, 2] - target[:, 2])

    loss_stack = torch.stack([lossu, lossv, lossp], dim=1)
    return torch.sum(loss_stack / weights)

# 权重从 Datapipe 获取
loss_weights = datapipe.get_loss_weights().to(device)
```

---

## 5. 训练循环

```python
best_val_loss = float("inf")
patience_counter = 0

for epoch in range(cfg.training.num_epochs):
    if train_sampler:
        train_sampler.set_epoch(epoch)

    model.train()
    train_loss = 0.0
    iterator = tqdm(train_loader, desc=f"Epoch {epoch}", disable=(dist.rank != 0))

    for batch in iterator:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = loss_func(output, y, loss_weights)     # 或 loss_fn(output, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if dist.rank == 0:
            iterator.set_postfix({"loss": f"{loss.item():.4e}"})

    avg_train_loss = train_loss / len(train_loader)

    # 验证 (按 eval_interval)
    if (epoch + 1) % cfg.training.eval_interval == 0:
        val_loss = evaluate(model, test_loader, device, loss_weights, dist)

        if dist.rank == 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存
                model_to_save = model.module if hasattr(model, "module") else model
                torch.save({
                    "model_state": model_to_save.state_dict(),
                    "config": cfg.model.to_dict(),
                    "epoch": epoch,
                }, output_dir / "best_model.pt")
            else:
                patience_counter += 1

        # 分布式安全早停
        stop_flag = torch.tensor([0], device=device)
        if dist.rank == 0 and patience_counter >= cfg.training.patience:
            stop_flag += 1
        if dist.world_size > 1:
            torch.distributed.broadcast(stop_flag, src=0)
        if stop_flag.item() > 0:
            break
```

---

## 6. 验证函数

```python
def evaluate(model, loader, device, weights, dist):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        iterator = tqdm(loader, desc="Evaluating", disable=(dist.rank != 0))
        for batch in iterator:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            output = model(x)
            loss = loss_func(output, y, weights)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches
```

---

## 7. CFDBench 特有：训练模式切换

CFDBench 有两个训练脚本：
- `train.py` — 静态模式（稳态预测）
- `train_auto.py` — 自回归模式（瞬态预测）

两者结构相同，区别在于数据格式和模型调用方式：

```python
# 静态模式: 直接预测
output = model(x)                           # x: (B, C_in, H, W)
loss = outputs["loss"][cfg_train.loss_name]  # 模型内部计算多种损失

# 自回归模式: 时间步迭代
for t in range(num_steps):
    output = model(x_t)
    x_t = output                             # 预测作为下一步输入
```

---

## 8. DeepCFD vs CFDBench 差异速查

| | DeepCFD | CFDBench |
|---|---------|----------|
| 根键 | `root` | `root` |
| 模型创建 | `init_model(cfg)` | 代码内 `if-elif` |
| 数据格式 | `batch["x"], batch["y"]` | `batch` (dict) |
| 损失函数 | 自定义 `loss_func` (通道加权) | 模型返回 `outputs["loss"]` |
| 优化器 | `AdamW` | `Adam` |
| 调度器 | 无 | `StepLR` |
| 早停 | `patience` + `broadcast` | 无 |
| 验证频率 | `eval_interval` (epoch) | `eval_interval` (epoch) |
| 运行模式 | 仅训练 | `train` / `test` / `train_test` |
| 训练后测试 | 无 | 有（加载 best 模型测试） |
