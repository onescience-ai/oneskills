# 训练脚本通用骨架 (Common Train.py Structure)

所有 CFD 训练脚本共享的 7 步通用流程。无论使用什么模型，都应遵循此骨架。

---

## 1. 完整骨架模板

```python
import os
import sys
import logging
import time
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist_torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from onescience.distributed.manager import DistributedManager
from onescience.utils.YParams import YParams
from onescience.datapipes.cfd import YourDatapipe          # 按实际替换
# from onescience.models.xxx import YourModel              # 按实际替换


def setup_logging(rank):
    """设置日志：仅 rank 0 输出 INFO"""
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logging.getLogger().setLevel(level)
    return logging.getLogger()


def main():
    # =========================================================
    # Step 1: 分布式初始化 (所有脚本 100% 统一)
    # =========================================================
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    logger = setup_logging(dist.rank)

    # =========================================================
    # Step 2: 配置加载 (两种模式，见下方说明)
    # =========================================================
    config_path = "conf/your_config.yaml"
    cfg = YParams(config_path, "root")    # 或按 section 分别加载

    output_dir = cfg.training.output_dir
    if dist.rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    # =========================================================
    # Step 3: 数据加载 (Datapipe 模式，95% 统一)
    # =========================================================
    datapipe = YourDatapipe(cfg.datapipe, distributed=(dist.world_size > 1))
    train_loader, train_sampler = datapipe.train_dataloader()
    val_loader, val_sampler = datapipe.val_dataloader()

    # =========================================================
    # Step 4: 模型初始化 + DDP (按模型分类，见对应 skill)
    # =========================================================
    model = YourModel(**model_args).to(device)

    if dist.world_size > 1:
        model = DDP(
            model,
            device_ids=[dist.local_rank],
            output_device=dist.local_rank,
            # find_unused_parameters=True  # 仅图/Transformer 模型需要
        )

    if dist.rank == 0:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model: {type(model).__name__}, Params: {total_params / 1e6:.2f}M")

    # =========================================================
    # Step 5: 优化器 / 调度器 / 损失函数
    # =========================================================
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=100, gamma=0.5
    )
    loss_fn = nn.MSELoss()

    # =========================================================
    # Step 6: 训练循环
    # =========================================================
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(cfg.training.epochs):
        epoch_start = time.time()

        # --- 6.1 设置分布式 sampler epoch ---
        if train_sampler:
            train_sampler.set_epoch(epoch)

        # --- 6.2 训练阶段 ---
        model.train()
        train_loss = 0.0
        iterator = tqdm(train_loader, desc=f"Epoch {epoch}", disable=(dist.rank != 0))

        for batch in iterator:
            # 数据搬运到 GPU (格式因模型而异)
            x, y = batch["x"].to(device), batch["y"].to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if dist.rank == 0:
                iterator.set_postfix({"loss": f"{loss.item():.4e}"})

        scheduler.step()
        train_loss /= len(train_loader)

        # --- 6.3 验证阶段 ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch["x"].to(device), batch["y"].to(device)
                out = model(x)
                loss = loss_fn(out, y)

                # 多卡同步验证 loss
                if dist.world_size > 1:
                    dist_torch.all_reduce(loss, op=dist_torch.ReduceOp.AVG)

                val_loss += loss.item()
        val_loss /= len(val_loader)

        # --- 6.4 日志 + Checkpoint ---
        if dist.rank == 0:
            logger.info(
                f"Epoch [{epoch + 1}/{cfg.training.epochs}] | "
                f"Time: {time.time() - epoch_start:.2f}s | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                model_to_save = model.module if hasattr(model, "module") else model
                torch.save({
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "loss": val_loss,
                }, os.path.join(output_dir, "best_model.pt"))
                logger.info("  -> Saved best model.")

        # --- 6.5 早停 (可选) ---
        if hasattr(cfg.training, "patience"):
            stop_flag = torch.tensor([0], device=device)
            if dist.rank == 0 and (epoch - best_epoch) > cfg.training.patience:
                logger.warning("Early stopping triggered.")
                stop_flag += 1
            if dist.world_size > 1:
                dist_torch.broadcast(stop_flag, src=0)
            if stop_flag.item() > 0:
                break

    # =========================================================
    # Step 7: 清理
    # =========================================================
    if dist.rank == 0:
        logger.info("Training finished.")
    dist.cleanup()


if __name__ == "__main__":
    main()
```

---

## 2. Step 2: 配置加载 — 两种模式

### 模式 A: 整体加载 (PDENNEval / DeepCFD / BENO / CFDBench)

```python
cfg = YParams(config_path, "fno_config")   # 按根包装键加载
# 访问: cfg.datapipe.source.data_dir, cfg.model.name, cfg.training.epochs
```

适用于配置文件有单一根包装键（`fno_config`, `root`, `beno_config` 等）的情况。

### 模式 B: 分段加载 (Transolver / Vortex_shedding / Eagle)

```python
cfg_model = YParams(config_path, "model")
cfg_data  = YParams(config_path, "datapipe")
cfg_train = YParams(config_path, "training")
# 访问: cfg_model.name, cfg_data.source.data_dir, cfg_train.max_epoch
```

适用于配置文件无根包装键、顶层直接展开的情况。

### 模式 C: Hydra (Lagrangian_MGN)

```python
@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # cfg 已自动合并所有子配置
    model = hydra.utils.instantiate(cfg.model)
    optimizer = hydra.utils.instantiate(cfg.optimizer, model.parameters())
```

---

## 3. Step 3: Datapipe 使用模式

所有 Datapipe 遵循统一接口：

```python
# 创建
datapipe = SomeDatapipe(params, distributed=(dist.world_size > 1))

# 获取 DataLoader + Sampler
train_loader, train_sampler = datapipe.train_dataloader()
val_loader, val_sampler = datapipe.val_dataloader()
test_loader, _ = datapipe.test_dataloader()       # 可选

# 训练循环中设置 sampler epoch (DDP 必须)
if train_sampler:
    train_sampler.set_epoch(epoch)
```

不同 Datapipe 的额外属性：

| Datapipe | 额外属性 |
|----------|---------|
| `DeepCFDDatapipe` | `get_loss_weights()` → 通道加权 |
| `DeepMind_CylinderFlowDatapipe` | `.stats` → 归一化统计量 |
| `AirfRANSDatapipe` | `.coef_norm` → 归一化系数 |
| `PDEBenchFNODatapipe` | `.spatial_dim` → 空间维度 |
| `DeepMindLagrangianDatapipe` | `.train_dataset.dt`, `.dim` |

---

## 4. Step 4: DDP 包装注意事项

```python
if dist.world_size > 1:
    model = DDP(
        model,
        device_ids=[dist.local_rank],
        output_device=dist.local_rank,
        find_unused_parameters=True,     # 按需设置
    )
```

| 模型类型 | `find_unused_parameters` | 原因 |
|---------|------------------------|------|
| CNN (DeepCFD, CFDBench) | `False` (默认) | 所有参数每步都使用 |
| PDE Operator (FNO, UNO) | `False` (默认) | 同上 |
| Graph (MGN, BENO) | `True` | 图结构变化导致部分参数可能未使用 |
| Transformer (Transolver) | `True` | 多模型切换，部分分支可能未使用 |

---

## 5. Step 5: 优化器/调度器常用组合

| 示例 | 优化器 | 调度器 | 特点 |
|------|--------|--------|------|
| PDENNEval (全部) | `Adam(lr=1e-3, wd=1e-4)` | `StepLR(step=100, gamma=0.5)` | 最通用 |
| DeepCFD | `AdamW(lr=1e-3, wd=5e-3)` | 无 | 简洁 |
| BENO | `Adam(lr=1e-5, wd=5e-4)` | `CosineAnnealingWarmRestarts(T0=16)` | 图模型 |
| Vortex_shedding | `Adam(lr=1e-4)` | `LambdaLR(rate=0.9999991)` | 每 step 衰减 |
| Transolver | `Adam(lr=1e-3)` | `OneCycleLR(max_lr, total_steps)` | Transformer |
| Lagrangian_MGN | `hydra.utils.instantiate(cfg.optimizer)` | `CosineAnnealingLR` | Hydra 动态 |

**动态创建优化器/调度器**（PDENNEval 推荐模式）：

```python
optim_cfg = cfg.training.optimizer
optimizer_cls = getattr(torch.optim, optim_cfg.name)
optimizer = optimizer_cls(model.parameters(), lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay)

sched_cfg = cfg.training.scheduler
scheduler_cls = getattr(torch.optim.lr_scheduler, sched_cfg.name)
scheduler = scheduler_cls(optimizer, step_size=sched_cfg.step_size, gamma=sched_cfg.gamma)
```

---

## 6. Step 6: Checkpoint 保存/加载

### 保存 (两种模式)

**模式 A: 手动保存** (大多数脚本)

```python
model_to_save = model.module if hasattr(model, "module") else model
torch.save({
    "model_state_dict": model_to_save.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "epoch": epoch,
    "loss": val_loss,
}, f"{output_dir}/best_model.pt")
```

**模式 B: 工具函数** (Vortex_shedding, Lagrangian_MGN)

```python
from onescience.launch.utils import load_checkpoint, save_checkpoint

save_checkpoint(
    checkpoint_dir,
    models=model,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,            # AMP GradScaler
    epoch=epoch,
)

epoch_init = load_checkpoint(
    checkpoint_dir,
    models=model,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,
    device=device,
)
```

### 加载最佳模型 (测试阶段)

```python
checkpoint = torch.load(best_model_path, map_location=device)
model_to_test = model.module if hasattr(model, "module") else model
model_to_test.load_state_dict(checkpoint["model_state_dict"])
```

---

## 7. 早停机制 (分布式安全)

```python
# rank 0 决定是否停止
stop_flag = torch.tensor([0], device=device)
if dist.rank == 0 and patience_counter >= patience:
    stop_flag += 1

# 广播给所有 rank
if dist.world_size > 1:
    dist_torch.broadcast(stop_flag, src=0)

if stop_flag.item() > 0:
    break
```

---

## 8. 数据格式与模型调用对照

训练循环内部的数据处理因模型类型而异：

| 类型 | 数据格式 | 模型调用 | 目标提取 |
|------|---------|---------|---------|
| CNN | `batch["x"], batch["y"]` | `model(x)` | `y` |
| Operator | `x, y, grid` (tuple) | `model(x, grid)` | `y[..., t:t+1, :]` |
| Graph (DGL) | `graph` (DGLGraph) | `model(graph.ndata["x"], graph.edata["x"], graph)` | `graph.ndata["y"]` |
| Graph (PyG) | `data` (PyG Data) | `model(data)` | `data.y` |
| Transformer | `data` (PyG Data) | `model(data)` | `data.y` |

---

## 9. Checklist

新建训练脚本时请确认：

- [ ] 使用 `DistributedManager.initialize()` + `DistributedManager()` 初始化
- [ ] 日志仅在 `rank == 0` 输出
- [ ] Datapipe 传入 `distributed=(dist.world_size > 1)`
- [ ] 训练循环中调用 `train_sampler.set_epoch(epoch)`
- [ ] 多卡时使用 DDP 包装模型
- [ ] 验证 loss 使用 `all_reduce` 同步
- [ ] Checkpoint 保存时提取 `model.module` (处理 DDP 包装)
- [ ] 早停使用 `broadcast` 同步 stop_flag
- [ ] 训练结束调用 `dist.cleanup()`
- [ ] tqdm 设置 `disable=(dist.rank != 0)`
