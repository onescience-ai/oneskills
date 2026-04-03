# Graph 模型训练脚本 (MeshGraphNet / HeteroGNS / Lagrangian_MGN)

适用于基于图神经网络的 CFD 训练脚本，包括 Vortex_shedding_mgn、BENO、Lagrangian_MGN。

---

## 1. 核心特征

Graph 模型训练脚本的独特之处：
- 配置分段加载：`YParams(path, "model")`, `YParams(path, "datapipe")`, `YParams(path, "training")`
- 使用 `specific_params` 字典提取模型参数
- DGL 图数据通过 `graph.ndata["x"]` / `graph.edata["x"]` 访问
- 支持 AMP 混合精度 (`torch.amp.autocast` + `GradScaler`)
- 使用 `onescience.launch.utils` 的 `load_checkpoint` / `save_checkpoint` 工具
- `scheduler.step()` 在每个 **batch** 级别调用（非 epoch 级别）
- `find_unused_parameters=True` 用于 DDP

---

## 2. 配置加载 (分段模式)

```python
config_path = "conf/mgn_cylinderflow.yaml"
cfg = YParams(config_path, "model")
cfg_data = YParams(config_path, "datapipe")
cfg_train = YParams(config_path, "training")

# 从 specific_params 提取模型参数
model_name = cfg.name
model_params = cfg.specific_params[model_name]
```

---

## 3. 模型初始化

### MeshGraphNet (Vortex_shedding)

```python
from onescience.models.meshgraphnet import MeshGraphNet

mlp_act = "silu" if model_params.recompute_activation else "relu"
model = MeshGraphNet(
    input_dim_nodes=model_params.num_input_features,
    input_dim_edges=model_params.num_edge_features,
    output_dim=model_params.num_output_features,
    mlp_activation_fn=mlp_act,
    do_concat_trick=model_params.do_concat_trick,
    num_processor_checkpoint_segments=model_params.num_processor_checkpoint_segments,
    recompute_activation=model_params.recompute_activation,
).to(device)
```

### Lagrangian_MGN (Hydra 动态实例化)

```python
model = hydra.utils.instantiate(cfg.model)
if cfg.compile.enabled:
    model = torch.compile(model, **cfg.compile.args).to(device)
else:
    model = model.to(device)
```

---

## 4. DDP 包装

Graph 模型需要 `find_unused_parameters=True`：

```python
if manager.world_size > 1:
    model = DistributedDataParallel(
        model,
        device_ids=[manager.local_rank],
        output_device=manager.local_rank,
        find_unused_parameters=True,      # 图结构变化导致部分参数可能未使用
    )
```

---

## 5. AMP 混合精度支持

```python
from torch.amp import GradScaler, autocast

scaler = GradScaler() if cfg_train.amp else None

# 训练步内
with autocast(device_type=device.type, enabled=cfg_train.amp):
    out = model(graph.ndata["x"], graph.edata["x"], graph)
    loss = loss_criterion(out, targets)

if cfg_train.amp:
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
else:
    loss.backward()
    optimizer.step()
```

---

## 6. 训练循环

### Vortex_shedding 风格 (标准 DGL)

```python
for epoch in range(epoch_init, cfg_train.max_epoch):
    if manager.world_size > 1:
        train_sampler.set_epoch(epoch)

    model.train()
    train_loss_sum = 0.0

    for idx, data in enumerate(train_dataloader):
        data = data.to(device)
        optimizer.zero_grad()

        with autocast(device_type=device.type, enabled=cfg_train.amp):
            out = model(data.ndata["x"], data.edata["x"], data)
            targets = data.ndata["y"]
            loss = loss_criterion(out, targets)

        # AMP 或标准反向传播
        if cfg_train.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        scheduler.step()                        # 每 batch 衰减
        train_loss_sum += loss.item()

        # 日志 (按 log_interval 打印)
        if manager.rank == 0 and (idx + 1) % log_interval == 0:
            logger.info(f"Epoch [{epoch+1}] Batch [{idx+1}] Loss: {loss.item():.6f}")
```

### Lagrangian_MGN 风格 (Trainer 类)

```python
class MGNTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dist = DistributedManager()
        self.datapipe = DeepMindLagrangianDatapipe(cfg, distributed=self.dist.distributed)
        self.dataloader = self.datapipe.train_dataloader()
        self.dataset = self.datapipe.train_dataset

        # Hydra 动态实例化
        self.model = hydra.utils.instantiate(cfg.model).to(self.dist.device)
        self.criterion = hydra.utils.instantiate(cfg.loss)
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, self.model.parameters())
        self.scheduler = hydra.utils.instantiate(cfg.lr_scheduler, self.optimizer)
        self.scaler = GradScaler(enabled=cfg.amp.enabled)

    def train_step(self, graph):
        graph = graph.to(self.dist.device)
        self.optimizer.zero_grad()

        with autocast(device_type="cuda", enabled=self.amp):
            gt_pos, gt_vel, gt_acc = self.dataset.unpack_targets(graph)
            pred_acc = self.model(graph.ndata["x"], graph.edata["x"], graph)

            mask = graph.ndata["mask"].unsqueeze(-1)
            num_nz = mask.sum() * self.dim
            loss = (mask * self.criterion(pred_acc, gt_acc)).sum() / num_nz

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        return {"loss": loss.item()}

    def run(self):
        for epoch in range(self.epoch_init, self.cfg.train.epochs + 1):
            if self.dist.distributed:
                self.dataloader.sampler.set_epoch(epoch)
            for graph in self.dataloader:
                losses = self.train_step(graph)
            # Checkpoint
            if self.dist.rank == 0 and epoch % self.cfg.train.checkpoint_save_freq == 0:
                save_checkpoint(checkpoint_dir, models=self.model, ...)
```

---

## 7. 验证循环

```python
model.eval()
valid_loss = 0.0
with torch.no_grad():
    for batch_data in val_dataloader:
        graph = batch_data[0] if isinstance(batch_data, (list, tuple)) else batch_data
        graph = graph.to(device)

        with autocast(device_type=device.type, enabled=cfg_train.amp):
            out = model(graph.ndata["x"], graph.edata["x"], graph)
            targets = graph.ndata["y"]
            loss = loss_criterion(out, targets)

        if manager.world_size > 1:
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)

        valid_loss += loss.item()

valid_loss /= len(val_dataloader)
```

---

## 8. Checkpoint (工具函数)

```python
from onescience.launch.utils import load_checkpoint, save_checkpoint

# 保存
save_checkpoint(
    checkpoint_dir,
    models=model,           # 自动处理 DDP model.module
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,          # AMP GradScaler (可为 None)
    epoch=epoch,
)

# 加载
epoch_init = load_checkpoint(
    checkpoint_dir,
    models=model,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,
    device=device,
)
```

---

## 9. 早停 + 分布式同步

```python
if manager.rank == 0:
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        best_loss_epoch = epoch
        save_checkpoint(...)

    if (epoch - best_loss_epoch) > cfg_train.patience:
        logger.warning(f"Early stopping at epoch {epoch}")
        break

if manager.world_size > 1:
    torch.distributed.barrier()
```

---

## 10. Graph vs Lagrangian 差异速查

| | Vortex_shedding | BENO | Lagrangian_MGN |
|---|----------------|------|---------------|
| 配置系统 | YParams 分段 | YParams 整体 | Hydra |
| 模型创建 | 手动 `MeshGraphNet(...)` | 手动 `HeteroGNS(...)` | `hydra.utils.instantiate` |
| 图框架 | DGL | PyG | DGL |
| AMP | 支持 | 不支持 | 支持 |
| torch.compile | 不支持 | 不支持 | 支持 |
| scheduler.step | 每 batch | 每 epoch | 每 batch |
| Checkpoint | `launch.utils` | 手动 `torch.save` | `launch.utils` |
| Trainer 类 | 无 (函数式) | 无 (函数式) | `MGNTrainer` 类 |
| 测试阶段 | 无 | 无 | 无 |
