import os
import sys
import torch

# 添加当前 case 目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.fourcastnet_global_region_hybrid import FourCastNetBase, FourCastNetGlobalRegionHybrid
from utils.checkpoint_utils import (
    save_checkpoint,
    load_checkpoint,
    save_checkpoint_early,
    load_global_pretrained,
    freeze_parameters,
    unfreeze_parameters,
    get_parameter_count,
    print_parameter_stats,
)


def test_checkpoint_save_load():
    """
    测试 checkpoint 保存和加载。
    """
    print("\n" + "=" * 60)
    print("Testing Checkpoint Save/Load")
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
    ).to(device)

    checkpoint_dir = "./test_checkpoints"
    checkpoint_path = os.path.join(checkpoint_dir, "test_checkpoint.pth")

    save_checkpoint(
        model=model,
        optimizer=None,
        epoch=1,
        loss=0.123,
        checkpoint_path=checkpoint_path,
        is_best=True,
    )

    print(f"\nCheckpoint saved to: {checkpoint_path}")

    loaded_model = FourCastNetBase(
        img_size=(720, 1440),
        patch_size=(8, 8),
        in_chans=19,
        out_chans=19,
        embed_dim=768,
        depth=12,
    ).to(device)

    load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=loaded_model,
        optimizer=None,
        device=device,
    )

    x = torch.randn(1, 19, 720, 1440).to(device)
    with torch.no_grad():
        output1 = model(x)
        output2 = loaded_model(x)

    assert torch.allclose(output1, output2, atol=1e-6), "Model outputs don't match after loading!"

    print("\n✓ Checkpoint save/load test passed!")


def test_checkpoint_early_save():
    """
    测试快速保存 checkpoint。
    """
    print("\n" + "=" * 60)
    print("Testing Early Checkpoint Save")
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
    ).to(device)

    checkpoint_dir = "./test_checkpoints"

    save_checkpoint_early(
        model=model,
        optimizer=None,
        epoch=5,
        loss=0.456,
        checkpoint_dir=checkpoint_dir,
        filename="checkpoint_early.pth",
    )

    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_early.pth")
    print(f"\nEarly checkpoint saved to: {checkpoint_path}")

    assert os.path.exists(checkpoint_path), "Checkpoint file not found!"

    loaded_model = FourCastNetBase(
        img_size=(720, 1440),
        patch_size=(8, 8),
        in_chans=19,
        out_chans=19,
        embed_dim=768,
        depth=12,
    ).to(device)

    load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=loaded_model,
        optimizer=None,
        device=device,
    )

    x = torch.randn(1, 19, 720, 1440).to(device)
    with torch.no_grad():
        output1 = model(x)
        output2 = loaded_model(x)

    assert torch.allclose(output1, output2, atol=1e-6), "Model outputs don't match after loading!"

    print("\n✓ Early checkpoint save test passed!")


def test_freeze_unfreeze():
    """
    测试参数冻结和解冻。
    """
    print("\n" + "=" * 60)
    print("Testing Freeze/Unfreeze")
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

    print("\nBefore freezing:")
    stats_before = get_parameter_count(model)
    print(f"  Total:     {stats_before['total']:,}")
    print(f"  Trainable: {stats_before['trainable']:,}")
    print(f"  Frozen:    {stats_before['frozen']:,}")

    freeze_parameters(model, prefix="global_")

    print("\nAfter freezing global branch:")
    stats_after = get_parameter_count(model)
    print(f"  Total:     {stats_after['total']:,}")
    print(f"  Trainable: {stats_after['trainable']:,}")
    print(f"  Frozen:    {stats_after['frozen']:,}")

    assert stats_after['frozen'] > stats_before['frozen'], "No parameters were frozen!"

    unfreeze_parameters(model, prefix="global_")

    print("\nAfter unfreezing global branch:")
    stats_final = get_parameter_count(model)
    print(f"  Total:     {stats_final['total']:,}")
    print(f"  Trainable: {stats_final['trainable']:,}")
    print(f"  Frozen:    {stats_final['frozen']:,}")

    assert stats_final['trainable'] == stats_before['trainable'], "Parameters not properly unfrozen!"

    print("\n✓ Freeze/unfreeze test passed!")


def test_model_freeze_methods():
    """
    测试模型的冻结和解冻方法。
    """
    print("\n" + "=" * 60)
    print("Testing Model Freeze/Unfreeze Methods")
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

    print("\nBefore freezing:")
    stats_before = get_parameter_count(model)
    print(f"  Total:     {stats_before['total']:,}")
    print(f"  Trainable: {stats_before['trainable']:,}")

    model.freeze_global_branch()

    print("\nAfter freezing global branch:")
    stats_after = get_parameter_count(model)
    print(f"  Total:     {stats_after['total']:,}")
    print(f"  Trainable: {stats_after['trainable']:,}")

    assert stats_after['trainable'] < stats_before['trainable'], "Global branch not frozen!"

    model.unfreeze_global_branch()

    print("\nAfter unfreezing global branch:")
    stats_final = get_parameter_count(model)
    print(f"  Total:     {stats_final['total']:,}")
    print(f"  Trainable: {stats_final['trainable']:,}")

    assert stats_final['trainable'] == stats_before['trainable'], "Global branch not unfrozen!"

    print("\n✓ Model freeze/unfreeze methods test passed!")


def test_parameter_count():
    """
    测试参数计数功能。
    """
    print("\n" + "=" * 60)
    print("Testing Parameter Count")
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
    ).to(device)

    stats = get_parameter_count(model)

    print(f"\nTotal parameters:     {stats['total']:,}")
    print(f"Trainable parameters: {stats['trainable']:,}")
    print(f"Frozen parameters:    {stats['frozen']:,}")

    assert stats['total'] > 0, "Total parameters should be positive!"
    assert stats['trainable'] == stats['total'], "All parameters should be trainable initially!"
    assert stats['frozen'] == 0, "No parameters should be frozen initially!"

    print("\n✓ Parameter count test passed!")


if __name__ == "__main__":
    try:
        test_checkpoint_save_load()
        test_checkpoint_early_save()
        test_freeze_unfreeze()
        test_model_freeze_methods()
        test_parameter_count()
        print("\n" + "=" * 60)
        print("All Utility Layer Tests Passed!")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"\n✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
