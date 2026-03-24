# Skill: OneScience Module Reuse

## 复用现有模块的方法

### 1. 直接导入使用
```python
from onescience.modules.attention import MultiHeadAttention
from onescience.modules.embedding import PatchEmbedding
from onescience.modules.encoder import TransformerEncoder

# 直接实例化
attn = MultiHeadAttention(embed_dim=256, num_heads=8)
```

### 2. 通过 Registry 动态构建
```python
from onescience.models import build_model

# 适合配置文件驱动的场景
model = build_model(dict(
    type='UNet',
    in_channels=3,
    out_channels=1
))
```

### 3. 组合多个模块
```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = PatchEmbedding(3, 768, 16)
        self.encoder = TransformerEncoder(768, 12, 8)
        self.head = nn.Linear(768, 1000)
```

### 4. 常用模块速查
```python
# Attention
from onescience.modules.attention import MultiHeadAttention

# Embedding
from onescience.modules.embedding import PatchEmbedding

# Loss
from onescience.modules.loss import MSELoss, L1Loss

# Encoder/Decoder
from onescience.modules.encoder import TransformerEncoder
from onescience.modules.decoder import TransformerDecoder
```

### 5. 参考现有示例
查看 `examples/` 目录了解最佳实践：
- `examples/earth/fourcastnet/` - AFNO 模块使用
- `examples/biosciences/alphafold3/` - Attention 模块使用
- `examples/cfd/meshgraphnet/` - GNN 模块使用