# Skill: Code Bug Detection

## 常见错误检测

### 1. Import 路径错误
```python
# ❌ 错误: model vs models
from onescience.model.unet import UNet

# ✅ 正确
from onescience.models.unet import UNet
```

### 2. 忘记注册装饰器
```python
# ❌ 错误: 直接使用未注册的类
class MyModel(nn.Module):
    pass
model = MyModel()

# ✅ 正确: 先注册再使用
from onescience.registry import MODELS

@MODELS.register_module()
class MyModel(nn.Module):
    pass
```

### 3. 参数名拼写错误
```python
# ❌ 错误: input_channels 不存在
model = UNet(input_channels=3)

# ✅ 正确: 应该是 in_channels
model = UNet(in_channels=3)
```

### 4. Tensor 维度顺序错误
```python
# ❌ 错误: [B,H,W,C] 格式
data = torch.randn(32, 256, 256, 3)
output = model(data)  # 报错

# ✅ 正确: 转换为 [B,C,H,W]
data = data.permute(0, 3, 1, 2)
```

### 5. Device 不一致
```python
# ❌ 错误: model 在 GPU, data 在 CPU
model = model.cuda()
data = torch.randn(1, 3, 224, 224)
output = model(data)  # 报错

# ✅ 正确: 保持一致
data = data.cuda()
```

### 6. 缺少必要的依赖导入
```python
# ❌ 错误: 使用了 nn.Module 但没导入
class MyModel(nn.Module):
    pass

# ✅ 正确
import torch.nn as nn

class MyModel(nn.Module):
    pass
```

### 7. YParams 配置访问路径错误
YParams(yaml_file, config_name) 只加载指定 section，嵌套字段必须通过正确的层级路径访问。
必须阅读 config YAML 的结构，确认每个属性在哪个 section、哪一层级。

```python
# ❌ 错误: static_dir 在 datapipe.dataset 下，不在 model 下
cfg = YParams("config.yaml", "model")
cfg.static_dir  # AttributeError

# ✅ 正确: 分别加载需要的 section，按层级访问
model_cfg = YParams("config.yaml", "model")
data_cfg = YParams("config.yaml", "datapipe")
model_cfg.embed_dim            # model section 的顶层字段
data_cfg.dataset.static_dir    # datapipe.dataset 下的嵌套字段
data_cfg.dataloader.batch_size # datapipe.dataloader 下的嵌套字段
```

**检查要点**：
- 确认代码中每个 `cfg.xxx` 访问的属性确实存在于所加载的 YAML section 中
- 嵌套字段（如 `datapipe.dataset.img_size`）不会被自动展平，必须逐层访问
- 当多个模块（如 ERA5Datapipe）需要特定的 params 结构时，需确认传入的对象包含所需的嵌套属性