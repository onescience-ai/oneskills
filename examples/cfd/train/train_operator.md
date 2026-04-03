# PDE Operator 训练脚本 (FNO / UNO / PINO / DeepONet / UNet)

适用于 PDENNEval 下的 PDE 求解器系列训练脚本。这些脚本共享高度统一的结构。

---

## 1. 核心特征

PDENNEval 训练脚本的独特之处：
- 使用 `argparse` 接收配置文件路径（而非硬编码）
- `get_model(spatial_dim, cfg)` 根据空间维度动态选择 1D/2D/3D 模型
- 支持 `autoregressive` 和 `single` 两种训练类型
- 自回归模式下有时间步循环嵌套在 batch 循环内
- 优化器/调度器通过 `getattr` 从配置动态创建

---

## 2. 配置加载

```python
parser = argparse.ArgumentParser()
parser.add_argument("config", type=str, help="Path to config file")
args = parser.parse_args()
cfg = YParams(args.config, "fno_config")  # 按模型名替换: uno_config, deeponet_config, ...
```

---

## 3. 模型初始化 — get_model 模式

每个模型有自己的 `get_model()` 工厂函数，根据 `spatial_dim` 选择变体：

```python
def get_model(spatial_dim, cfg):
    model_args = cfg.model
    initial_step = cfg.datapipe.data.initial_step

    if spatial_dim == 1:
        model = FNO1d(
            num_channels=model_args.num_channels,
            width=model_args.width,
            modes=model_args.modes,
            initial_step=initial_step
        )
    elif spatial_dim == 2:
        model = FNO2d(
            num_channels=model_args.num_channels,
            width=model_args.width,
            modes1=model_args.modes,
            modes2=model_args.modes,
            initial_step=initial_step
        )
    elif spatial_dim == 3:
        model = FNO3d(...)
    return model

# 使用：
spatial_dim = datapipe.spatial_dim     # 由 Datapipe 提供
model = get_model(spatial_dim, cfg).to(device)
```

各模型的 `get_model` 参数对照：

| 模型 | 关键参数 |
|------|---------|
| FNO | `num_channels, width, modes, initial_step` |
| UNO | `num_channels, width, initial_step` |
| PINO | `width, modes1, modes2, fc_dim, act, in_channels, out_channels` |
| DeepONet | `base_model, input_size, act, in_channels, out_channels, query_dim` |
| UNet | `in_channels, out_channels, init_features` |

---

## 4. 优化器/调度器动态创建

```python
# 从配置动态创建（所有 PDENNEval 模型统一）
optim_cfg = cfg.training.optimizer
optimizer_cls = getattr(torch.optim, optim_cfg.name)        # "Adam" → torch.optim.Adam
optimizer = optimizer_cls(model.parameters(), lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay)

sched_cfg = cfg.training.scheduler
scheduler_cls = getattr(torch.optim.lr_scheduler, sched_cfg.name)  # "StepLR" → ...
scheduler = scheduler_cls(optimizer, step_size=sched_cfg.step_size, gamma=sched_cfg.gamma)
```

---

## 5. 训练循环 — 自回归 vs 单步

### 数据格式

PDENNEval 的 DataLoader 返回 `(x, y, grid)` 三元组：
- `x`: 输入场 `(B, ..., T_in, C)` — 前 `initial_step` 个时间步
- `y`: 目标场 `(B, ..., T_total, C)` — 所有时间步
- `grid`: 空间坐标 `(B, ..., D)` — 网格坐标

### 自回归训练 (training_type == "autoregressive")

```python
def train_loop(model, train_loader, optimizer, loss_fn, scheduler, device, cfg, rank):
    model.train()
    train_l2 = 0.0
    initial_step = cfg.datapipe.data.initial_step
    t_train = cfg.training.t_train          # 自回归推进的目标时间步

    for x, y, grid in train_loader:
        x, y, grid = x.to(device), y.to(device), grid.to(device)
        loss = 0
        pred = y[..., :initial_step, :]     # 初始条件
        input_shape = list(x.shape)[:-2] + [-1]

        # 时间步循环：逐步预测
        for t in range(initial_step, t_train):
            model_input = x.reshape(input_shape)
            target = y[..., t:t+1, :]
            model_output = model(model_input, grid)

            loss += loss_fn(
                model_output.reshape(model_output.size(0), -1),
                target.reshape(target.size(0), -1)
            )
            # 滑动窗口：用预测替换最早的输入
            pred = torch.cat((pred, model_output), -2)
            x = torch.cat((x[..., 1:, :], model_output), dim=-2)

        train_l2 += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()
    return train_l2
```

### 单步训练 (training_type == "single")

```python
        # 单步：直接预测最终时间步
        model_input = x.reshape(input_shape)
        target = y[..., t_train-1:t_train, :]
        pred = model(model_input, grid)
        loss = loss_fn(
            pred.reshape(pred.size(0), -1),
            target.reshape(target.size(0), -1)
        )
```

---

## 6. PINO 特有：PDE 约束损失

PINO 在数据损失之外增加物理方程残差损失：

```python
# 训练循环中
loss_data = loss_fn(pred, target)                    # 数据拟合损失
loss_ic = loss_fn(pred_ic, target_ic)                # 初始条件损失
loss_pde = pde_residual_fn(pred, grid, aux_params)   # PDE 残差损失

loss = (train_args.xy_loss * loss_data
      + train_args.ic_loss * loss_ic
      + train_args.f_loss * loss_pde)
```

---

## 7. 验证循环

```python
def val_loop(val_loader, model, loss_fn, device, cfg):
    model.eval()
    val_l2 = 0.0
    initial_step = cfg.datapipe.data.initial_step
    t_train = cfg.training.t_train

    with torch.no_grad():
        for x, y, grid in val_loader:
            x, y, grid = x.to(device), y.to(device), grid.to(device)
            input_shape = list(x.shape)[:-2] + [-1]

            if cfg.training.training_type == "autoregressive":
                pred = y[..., :initial_step, :]
                for t in range(initial_step, y.shape[-2]):
                    model_input = x.reshape(input_shape)
                    model_output = model(model_input, grid)
                    pred = torch.cat((pred, model_output), -2)
                    x = torch.cat((x[..., 1:, :], model_output), dim=-2)

                _pred = pred[..., initial_step:t_train, :]
                _y = y[..., initial_step:t_train, :]
                val_l2 += loss_fn(_pred.reshape(_y.size(0), -1),
                                  _y.reshape(_y.size(0), -1)).item()

            elif cfg.training.training_type == "single":
                model_input = x.reshape(input_shape)
                target = y[..., t_train-1:t_train, :]
                pred = model(model_input, grid)
                val_l2 += loss_fn(pred.reshape(target.size(0), -1),
                                  target.reshape(target.size(0), -1)).item()
    return val_l2
```

---

## 8. 主循环 + Checkpoint

```python
for epoch in range(cfg.training.epochs):
    if train_sampler:
        train_sampler.set_epoch(epoch)

    train_l2, elapsed = train_loop(model, train_loader, optimizer, loss_fn,
                                    scheduler, device, cfg, dist.rank)

    if dist.rank == 0:
        print(f"[Epoch {epoch}] Train L2: {train_l2:.5f}, Time: {elapsed:.2f}s")

        if (epoch + 1) % cfg.training.save_period == 0:
            val_l2 = val_loop(val_loader, model, loss_fn, device, cfg)
            print(f"[Epoch {epoch}] Val L2: {val_l2:.5f}")

            model_to_save = model.module if hasattr(model, "module") else model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": val_l2,
            }, output_dir / f"model_epoch_{epoch}.pt")

dist.cleanup()
```

---

## 9. 各模型差异速查

| | FNO | UNO | PINO | DeepONet | UNet | MPNN |
|---|-----|-----|------|----------|------|------|
| 训练类型 | auto/single | auto/single | auto/single | auto/single | auto/single + pushforward | auto + unrolling |
| 数据格式 | `(x,y,grid)` | `(x,y,grid)` | `(x,y,grid)` | `(x,y,grid)` | `(x,y,grid)` | `(x,y,grid)` |
| PDE 损失 | 无 | 无 | 有 (ic+f+xy) | 无 | 无 | 无 |
| scheduler.step | epoch 级 | epoch 级 | epoch 级 | epoch 级 | epoch 级 | epoch 级 |
| TensorBoard | 无 | 无 | 无 | 无 | 可选 | 可选 |
