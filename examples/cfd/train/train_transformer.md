# Transformer 训练脚本 (Transolver / GraphViT)

适用于基于 Transformer/注意力机制的 CFD 训练脚本，包括 Transolver-Airfoil-Design、Transolver-Car-Design、EagleMeshTransformer。

---

## 1. 核心特征

Transformer 训练脚本的独特之处：
- 配置分段加载：`model`, `datapipe`, `training` 三个 section
- 使用 `specific_params` + 模型名称动态选择模型类
- 支持 **表面/体积加权损失** (`loss_surf` + `loss_vol`)
- 训练完成后有 **测试阶段**，加载最佳 checkpoint 评估
- 使用 `OneCycleLR` 学习率调度器
- 需要将模型参数注入 Datapipe (`cfg_data.model_hparams = model_params`)

---

## 2. 配置加载

```python
config_path = "conf/transolver_airfrans.yaml"
cfg = YParams(config_path, "model")
cfg_data = YParams(config_path, "datapipe")
cfg_train = YParams(config_path, "training")

model_name = cfg.name                              # "Transolver", "GraphSAGE", ...
model_params = cfg.specific_params[model_name]

# 注入模型参数到 datapipe（Transolver 特有）
cfg_data.model_hparams = model_params
```

---

## 3. 模型初始化 — 多模型动态选择

```python
from onescience.models.transolver import Transolver2D, Transolver2D_plus
from onescience.models.transolver.MLP import MLP
from onescience.models.transolver.GraphSAGE import GraphSAGE
from onescience.models.transolver.PointNet import PointNet
from onescience.models.transolver.NN import NN
from onescience.models.transolver.GUNet import GUNet

if model_name in ["Transolver", "Transolver_plus"]:
    ModelClass = Transolver2D if model_name == "Transolver" else Transolver2D_plus
    model = ModelClass(
        n_hidden=model_params.n_hidden,
        n_layers=model_params.n_layers,
        space_dim=model_params.space_dim,
        fun_dim=model_params.fun_dim,
        n_head=model_params.n_head,
        mlp_ratio=model_params.mlp_ratio,
        out_dim=model_params.out_dim,
        slice_num=model_params.slice_num,
        unified_pos=model_params.unified_pos,
    ).to(device)
else:
    # 基线模型（GraphSAGE, PointNet, MLP, GUNet）使用 encoder/decoder
    encoder = MLP(list(model_params.encoder), batch_norm=False)
    decoder = MLP(list(model_params.decoder), batch_norm=False)

    if model_name == "GraphSAGE":
        model = GraphSAGE(model_params.to_dict(), encoder, decoder).to(device)
    elif model_name == "PointNet":
        model = PointNet(model_params.to_dict(), encoder, decoder).to(device)
    elif model_name == "MLP":
        model = NN(model_params.to_dict(), encoder, decoder).to(device)
    elif model_name == "GUNet":
        model = GUNet(model_params.to_dict(), encoder, decoder).to(device)
```

---

## 4. 损失函数 — 表面/体积加权

Transolver 区分 **表面节点** 和 **体积节点** 的损失，对表面施加更高权重：

```python
# 损失函数（使用 reduction='none' 以便分区域加权）
if cfg_train.loss_criterion in ["MSE", "MSE_weighted"]:
    loss_criterion = nn.MSELoss(reduction="none")
elif cfg_train.loss_criterion == "MAE":
    loss_criterion = nn.L1Loss(reduction="none")

loss_weight = cfg_train.loss_weight             # 表面损失权重
use_weighted_loss = (cfg_train.loss_criterion == "MSE_weighted")

# 训练步内
out = model(data)
targets = data.y

loss_surf = loss_criterion(out[data.surf], targets[data.surf]).mean()   # 表面
loss_vol = loss_criterion(out[~data.surf], targets[~data.surf]).mean()  # 体积

loss = (loss_vol + loss_weight * loss_surf) if use_weighted_loss else loss_criterion(out, targets).mean()
```

---

## 5. 优化器与调度器

```python
optimizer = torch.optim.Adam(model.parameters(), lr=cfg_train.lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=cfg_train.lr,
    total_steps=len(train_dataloader) * cfg_train.max_epoch,
)
```

注意：`scheduler.step()` 在每个 **batch** 后调用（不是 epoch）。

---

## 6. 训练循环

```python
for epoch in range(cfg_train.max_epoch):
    if manager.world_size > 1:
        train_sampler.set_epoch(epoch)
        if val_sampler:
            val_sampler.set_epoch(epoch)

    model.train()
    train_loss = train_loss_surf = train_loss_vol = 0.0

    iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{cfg_train.max_epoch}",
                    leave=False, disable=(manager.rank != 0))

    for data in iterator:
        data = data.to(device)                  # PyG Data 对象
        optimizer.zero_grad()
        out = model(data)
        targets = data.y

        loss_surf = loss_criterion(out[data.surf], targets[data.surf]).mean()
        loss_vol = loss_criterion(out[~data.surf], targets[~data.surf]).mean()
        loss = loss_vol + loss_weight * loss_surf if use_weighted_loss else loss_criterion(out, targets).mean()

        loss.backward()
        optimizer.step()
        scheduler.step()                        # 每 batch 调度

        train_loss += loss.item()
        if manager.rank == 0:
            iterator.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

    # 验证
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for data in val_dataloader:
            data = data.to(device)
            out = model(data)
            loss = ...  # 同上
            if manager.world_size > 1:
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            valid_loss += loss.item()

    valid_loss /= len(val_dataloader)

    # Checkpoint + 早停
    if manager.rank == 0:
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_loss_epoch = epoch
            save_checkpoint(model, optimizer, scheduler, epoch, valid_loss, checkpoint_dir, model_name)

        if epoch - best_loss_epoch > cfg_train.patience:
            break
```

---

## 7. 训练后测试

Transolver 在训练结束后加载最佳模型进行测试（仅 rank 0）：

```python
if manager.rank == 0:
    logger.info("Training finished. Starting testing...")

    # 加载最佳 checkpoint
    checkpoint = torch.load(f"{checkpoint_dir}/{model_name}.pth", map_location=device)
    model_to_test = model.module if hasattr(model, "module") else model
    model_to_test.load_state_dict(checkpoint["model_state_dict"])

    # 调用 metrics 工具
    import onescience.utils.transolver.metrics as metrics
    coefs = metrics.Results_test(
        device,
        [model_to_test],
        [model_params.to_dict()],
        datapipe.coef_norm,
        cfg_data.source.data_dir,
        checkpoint_dir,
        cfg_train.n_test,
        criterion=cfg_train.loss_criterion,
        s=cfg_data.data.splits.test_name,
    )
    np.save(os.path.join(checkpoint_dir, "true_coefs"), coefs[0])
    np.save(os.path.join(checkpoint_dir, "pred_coefs_mean"), coefs[1])
```

---

## 8. Checkpoint 保存函数

```python
def save_checkpoint(model, optimizer, scheduler, epoch, loss, ckp_dir, model_name):
    os.makedirs(ckp_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save({
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }, f"{ckp_dir}/{model_name}.pth")
```

---

## 9. Airfoil vs Car vs Eagle 差异速查

| | Airfoil | Car | Eagle |
|---|---------|-----|-------|
| Datapipe | `AirfRANSDatapipe` | `ShapeNetCarDatapipe` | `EagleDatapipe` |
| 数据格式 | PyG Data | PyG Data | 自定义 dict |
| 模型调用 | `model(data)` | `model(data)` | `model(x, graph)` |
| 表面加权损失 | 有 (`data.surf`) | 有 | 无（用 `loss_alpha`） |
| 归一化系数 | `datapipe.coef_norm` | `datapipe.coef_norm` | 内置 |
| 测试阶段 | `metrics.Results_test` | `metrics.Results_test` | 内置 rollout |
| scheduler | `OneCycleLR` | `OneCycleLR` | `OneCycleLR` |
| 参数注入 | `cfg_data.model_hparams` | `cfg_data.model_hparams` | 无 |
