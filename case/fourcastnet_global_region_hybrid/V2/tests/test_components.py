import sys
import torch
import os

# 添加当前 case 目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from components.global_region_fusion import (
    GlobalRegionFusionBlock,
    GlobalFeatureAligner,
)


def test_fusion_block():
    """
    测试融合模块。
    """
    print("\n" + "=" * 60)
    print("Testing Fusion Block")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    fusion_block = GlobalRegionFusionBlock(
        dim_region=768,
        dim_global=768,
        dim_out=768,
        fusion_mode="concat_conv",
        dropout=0.0,
    ).to(device)

    region_feat = torch.randn(2, 45, 90, 768).to(device)
    global_feat = torch.randn(2, 45, 90, 768).to(device)

    print(f"\nRegion feature shape:  {region_feat.shape}")
    print(f"Global feature shape:  {global_feat.shape}")

    fused_feat = fusion_block(region_feat, global_feat)
    print(f"Fused feature shape:   {fused_feat.shape}")

    assert fused_feat.shape == region_feat.shape, "Output shape mismatch!"

    print("\n✓ Fusion block test passed!")


def test_fusion_block_modes():
    """
    测试融合模块的不同模式。
    """
    print("\n" + "=" * 60)
    print("Testing Fusion Block Modes")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    fusion_modes = ["concat_mlp", "concat_conv", "gate"]

    for mode in fusion_modes:
        print(f"\nTesting mode: {mode}")

        fusion_block = GlobalRegionFusionBlock(
            dim_region=768,
            dim_global=768,
            dim_out=768,
            fusion_mode=mode,
            dropout=0.0,
        ).to(device)

        region_feat = torch.randn(2, 45, 90, 768).to(device)
        global_feat = torch.randn(2, 45, 90, 768).to(device)

        fused_feat = fusion_block(region_feat, global_feat)

        assert fused_feat.shape == region_feat.shape, f"Output shape mismatch for mode {mode}!"
        print(f"  ✓ Mode {mode} passed")


def test_feature_aligner():
    """
    测试特征对齐模块。
    """
    print("\n" + "=" * 60)
    print("Testing Feature Aligner")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    aligner = GlobalFeatureAligner(
        global_grid_size=(90, 180),
        global_img_size=(720, 1440),
        align_mode="crop",
    ).to(device)

    global_feat = torch.randn(2, 90, 180, 768).to(device)
    region_lat_range = (-5, 5)
    region_lon_range = (-5, 5)
    region_grid_size = (45, 90)

    print(f"\nGlobal feature shape:       {global_feat.shape}")
    print(f"Region lat range:          {region_lat_range}")
    print(f"Region lon range:          {region_lon_range}")
    print(f"Region grid size:          {region_grid_size}")

    aligned_feat = aligner(
        global_feat,
        region_lat_range,
        region_lon_range,
        region_grid_size,
    )
    print(f"Aligned feature shape:     {aligned_feat.shape}")

    assert aligned_feat.shape == (2, 45, 90, 768), "Output shape mismatch!"

    print("\n✓ Feature aligner test passed!")


def test_feature_aligner_modes():
    """
    测试特征对齐模块的不同模式。
    """
    print("\n" + "=" * 60)
    print("Testing Feature Aligner Modes")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    align_modes = ["crop", "interpolate"]

    for mode in align_modes:
        print(f"\nTesting mode: {mode}")

        aligner = GlobalFeatureAligner(
            global_grid_size=(90, 180),
            global_img_size=(720, 1440),
            align_mode=mode,
        ).to(device)

        global_feat = torch.randn(2, 90, 180, 768).to(device)
        region_lat_range = (-5, 5)
        region_lon_range = (-5, 5)
        region_grid_size = (45, 90)

        aligned_feat = aligner(
            global_feat,
            region_lat_range,
            region_lon_range,
            region_grid_size,
        )

        assert aligned_feat.shape == (2, 45, 90, 768), f"Output shape mismatch for mode {mode}!"
        print(f"  ✓ Mode {mode} passed")


if __name__ == "__main__":
    try:
        test_fusion_block()
        test_fusion_block_modes()
        test_feature_aligner()
        test_feature_aligner_modes()
        print("\n" + "=" * 60)
        print("All Component Layer Tests Passed!")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"\n✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
