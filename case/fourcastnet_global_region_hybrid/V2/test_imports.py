import os
import sys

# 测试导入路径
print("Current working directory:", os.getcwd())
print("Adding case directory to path...")
case_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(case_dir)
print(f"Case directory: {case_dir}")

# 测试导入
try:
    from components.global_region_fusion import GlobalRegionFusionBlock
    print("✓ Successfully imported GlobalRegionFusionBlock")
except Exception as e:
    print(f"✗ Failed to import GlobalRegionFusionBlock: {e}")

try:
    from models.fourcastnet_global_region_hybrid import FourCastNetBase
    print("✓ Successfully imported FourCastNetBase")
except Exception as e:
    print(f"✗ Failed to import FourCastNetBase: {e}")

try:
    from utils.synthetic_data import generate_synthetic_weather_data
    print("✓ Successfully imported generate_synthetic_weather_data")
except Exception as e:
    print(f"✗ Failed to import generate_synthetic_weather_data: {e}")

print("\nImport test completed.")
