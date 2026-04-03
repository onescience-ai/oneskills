# Skill: Device Adaptation

## 硬件环境适配

### 1. 单机单卡
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)
```

### 2. 单机多卡 (DataParallel)
```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.cuda()
```

### 3. 分布式训练 (DDP)
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])
```

### 4. DataLoader 配置
```python
# 单卡
train_loader = DataLoader(dataset, batch_size=32, num_workers=4)

# 多卡 DDP
sampler = DistributedSampler(dataset)
train_loader = DataLoader(dataset, batch_size=32, sampler=sampler, num_workers=4)
```

### 5. Batch Size 调整
```python
# 根据 GPU 数量自动调整
world_size = torch.cuda.device_count()
batch_size = 32 * world_size
```