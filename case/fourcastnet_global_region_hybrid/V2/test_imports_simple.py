import os
import sys

# 测试导入路径
print("Current working directory:", os.getcwd())
print("Adding case directory to path...")
case_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(case_dir)
print(f"Case directory: {case_dir}")

# 测试路径是否正确
print("\nChecking directory structure...")

# 检查组件目录
components_dir = os.path.join(case_dir, "components")
print(f"Components directory exists: {os.path.exists(components_dir)}")
if os.path.exists(components_dir):
    print(f"Files in components: {os.listdir(components_dir)}")

# 检查模型目录
models_dir = os.path.join(case_dir, "models")
print(f"Models directory exists: {os.path.exists(models_dir)}")
if os.path.exists(models_dir):
    print(f"Files in models: {os.listdir(models_dir)}")

# 检查工具目录
utils_dir = os.path.join(case_dir, "utils")
print(f"Utils directory exists: {os.path.exists(utils_dir)}")
if os.path.exists(utils_dir):
    print(f"Files in utils: {os.listdir(utils_dir)}")

# 测试导入 (不依赖 torch)
print("\nTesting imports without torch...")

# 临时替换 sys.modules 来模拟 torch
try:
    import torch
    print("✓ torch is available")
except ImportError:
    print("✗ torch is not available, but we'll test path resolution anyway")
    # 模拟 torch 模块
    import types
    sys.modules['torch'] = types.ModuleType('torch')
    sys.modules['torch.nn'] = types.ModuleType('torch.nn')
    sys.modules['torch.nn.functional'] = types.ModuleType('torch.nn.functional')
    sys.modules['einops'] = types.ModuleType('einops')
    sys.modules['einops'].rearrange = lambda x, *args, **kwargs: x

# 测试导入
try:
    from components.global_region_fusion import GlobalRegionFusionBlock
    print("✓ Successfully imported GlobalRegionFusionBlock")
except Exception as e:
    print(f"✗ Failed to import GlobalRegionFusionBlock: {e}")
    import traceback
    traceback.print_exc()

try:
    from models.fourcastnet_global_region_hybrid import FourCastNetBase
    print("✓ Successfully imported FourCastNetBase")
except Exception as e:
    print(f"✗ Failed to import FourCastNetBase: {e}")
    import traceback
    traceback.print_exc()

try:
    from utils.synthetic_data import generate_synthetic_weather_data
    print("✓ Successfully imported generate_synthetic_weather_data")
except Exception as e:
    print(f"✗ Failed to import generate_synthetic_weather_data: {e}")
    import traceback
    traceback.print_exc()

print("\nImport test completed.")
