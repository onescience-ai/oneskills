import sys
import torch
import os

# 添加当前 case 目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.fourcastnet_global_region_hybrid import (
    FourCastNetGlobalRegionHybrid,
    FourCastNetBase,
)
from utils.checkpoint_utils import print_parameter_stats


def test_base_model():
    """
    测试基座模型。
    """
    print("\n" + "=" * 60)
    print("Testing Base Model")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = FourCastNetBase(
        img_size=(720, 1440),
        patch_size=(8, 8),
        in_chans=19,
        out_chans=19,
        embed_dim=768,
        depth=12,
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        num_blocks=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
    ).to(device)

    print_parameter_stats(model)

    x = torch.randn(2, 19, 720, 1440).to(device)
    print(f"\nInput shape:  {x.shape}")

    with torch.no_grad():
        output = model(x)

    print(f"Output shape: {output.shape}")

    assert output.shape == (2, 19, 720, 1440), "Output shape mismatch!"

    print("\n✓ Base model test passed!")


def test_hybrid_model():
    """
    测试混合模型。
    """
    print("\n" + "=" * 60)
    print("Testing Hybrid Model")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = FourCastNetGlobalRegionHybrid(
        global_img_size=(720, 1440),
        global_patch_size=(8, 8),
        global_embed_dim=768,
        global_depth=12,
        global_checkpoint_path=None,

        region_img_size=(1000, 1000),
        region_patch_size=(8, 8),
        region_lat_range=(-5, 5),
        region_lon_range=(-5, 5),
        region_embed_dim=768,
        region_depth=6,

        in_chans=19,
        out_chans=19,
        fusion_layer_idx=3,
        fusion_mode="concat_conv",
        align_mode="crop",

        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        num_blocks=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
    ).to(device)

    print_parameter_stats(model)

    x_region = torch.randn(2, 19, 1000, 1000).to(device)
    x_global = torch.randn(2, 19, 720, 1440).to(device)

    print(f"\nRegion input shape:  {x_region.shape}")
    print(f"Global input shape:  {x_global.shape}")

    with torch.no_grad():
        output = model(x_region, x_global)

    print(f"Output shape:        {output.shape}")

    assert output.shape == (2, 19, 1000, 1000), "Output shape mismatch!"

    print("\n✓ Hybrid model test passed!")


def test_hybrid_model_fusion_modes():
    """
    测试混合模型的不同融合模式。
    """
    print("\n" + "=" * 60)
    print("Testing Hybrid Model Fusion Modes")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    fusion_modes = ["concat_mlp", "concat_conv", "gate"]

    for mode in fusion_modes:
        print(f"\nTesting fusion mode: {mode}")

        model = FourCastNetGlobalRegionHybrid(
            global_img_size=(720, 1440),
            global_patch_size=(8, 8),
            global_embed_dim=768,
            global_depth=12,
            region_img_size=(1000, 1000),
            region_patch_size=(8, 8),
            region_lat_range=(-5, 5),
            region_lon_range=(-5, 5),
            region_embed_dim=768,
            region_depth=6,
            in_chans=19,
            out_chans=19,
            fusion_layer_idx=3,
            fusion_mode=mode,
            align_mode="crop",
        ).to(device)

        x_region = torch.randn(1, 19, 1000, 1000).to(device)
        x_global = torch.randn(1, 19, 720, 1440).to(device)

        with torch.no_grad():
            output = model(x_region, x_global)

        assert output.shape == (1, 19, 1000, 1000), f"Output shape mismatch for mode {mode}!"
        print(f"  ✓ Fusion mode {mode} passed")


def test_hybrid_model_align_modes():
    """
    测试混合模型的不同对齐模式。
    """
    print("\n" + "=" * 60)
    print("Testing Hybrid Model Align Modes")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    align_modes = ["crop", "interpolate"]

    for mode in align_modes:
        print(f"\nTesting align mode: {mode}")

        model = FourCastNetGlobalRegionHybrid(
            global_img_size=(720, 1440),
            global_patch_size=(8, 8),
            global_embed_dim=768,
            global_depth=12,
            region_img_size=(1000, 1000),
            region_patch_size=(8, 8),
            region_lat_range=(-5, 5),
            region_lon_range=(-5, 5),
            region_embed_dim=768,
            region_depth=6,
            in_chans=19,
            out_chans=19,
            fusion_layer_idx=3,
            fusion_mode="concat_conv",
            align_mode=mode,
        ).to(device)

        x_region = torch.randn(1, 19, 1000, 1000).to(device)
        x_global = torch.randn(1, 19, 720, 1440).to(device)

        with torch.no_grad():
            output = model(x_region, x_global)

        assert output.shape == (1, 19, 1000, 1000), f"Output shape mismatch for mode {mode}!"
        print(f"  ✓ Align mode {mode} passed")


def test_hybrid_model_without_global_input():
    """
    测试混合模型在没有全球输入时的行为。
    """
    print("\n" + "=" * 60)
    print("Testing Hybrid Model Without Global Input")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = FourCastNetGlobalRegionHybrid(
        global_img_size=(720, 1440),
        global_patch_size=(8, 8),
        global_embed_dim=768,
        global_depth=12,
        region_img_size=(1000, 1000),
        region_patch_size=(8, 8),
        region_lat_range=(-5, 5),
        region_lon_range=(-5, 5),
        region_embed_dim=768,
        region_depth=6,
        in_chans=19,
        out_chans=19,
        fusion_layer_idx=3,
    ).to(device)

    x_region = torch.randn(1, 19, 1000, 1000).to(device)

    print(f"\nRegion input shape:  {x_region.shape}")
    print("Global input: None (will use zeros)")

    with torch.no_grad():
        output = model(x_region, x_global=None)

    print(f"Output shape:        {output.shape}")

    assert output.shape == (1, 19, 1000, 1000), "Output shape mismatch!"

    print("\n✓ Hybrid model without global input test passed!")


if __name__ == "__main__":
    try:
        test_base_model()
        test_hybrid_model()
        test_hybrid_model_fusion_modes()
        test_hybrid_model_align_modes()
        test_hybrid_model_without_global_input()
        print("\n" + "=" * 60)
        print("All Model Layer Tests Passed!")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"\n✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
