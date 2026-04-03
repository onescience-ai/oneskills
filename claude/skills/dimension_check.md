# Check: Tensor Dimension

## 检查要点

1. **Dataset 输出维度**
   - 检查 `__getitem__` 返回的 tensor shape
   - 验证 batch 后的维度: `[B, C, H, W]` 或 `[B, C, T, H, W]`

2. **Model 输入维度**
   - 检查 `forward()` 方法的输入要求
   - 注意空间维度 (H, W) 和时间维度 (T)

3. **Loss 输入维度**
   - 确认 loss 函数期望的 pred 和 target shape
   - 常见格式: `[B, C, ...]` 或 `[B, ...]`

## 常见错误

**错误 1**: Dataset 输出 `[H, W, C]` 但模型期望 `[C, H, W]`
```python
# 修复: 添加 transpose
data = data.permute(2, 0, 1)  # [H,W,C] -> [C,H,W]
```

**错误 2**: 模型输出 `[B, C, H, W]` 但 loss 期望 `[B, H, W]`
```python
# 修复: squeeze channel 维度
output = output.squeeze(1)  # [B,1,H,W] -> [B,H,W]
```

**错误 3**: 时间序列维度不匹配
```python
# 检查: 打印实际维度
print(f"Data: {data.shape}, Model expects: [B, T, C, H, W]")
```