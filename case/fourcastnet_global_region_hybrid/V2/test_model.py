import os
import sys
import torch
import torch.nn as nn

sys.path.append("/Users/zhao/Desktop/OneScience/dev-earth-function+commit")

from models.fourcastnet_global_region_hybrid import (
    FourCastNetGlobalRegionHybrid,
    FourCastNetBase,
)
from components.global_region_fusion import (
    GlobalRegionFusionBlock,
    GlobalFeatureAligner,
)
from utils.synthetic_data import (
    generate_synthetic_weather_data,
    SyntheticWeatherDataset,
    create_dataloaders,
)
from utils.checkpoint_utils import (
    save_checkpoint,
    load_checkpoint,
    print_parameter_stats,
)


def test_synthetic_data():
    """
    测试虚拟数据生成。
    """
    print("\n" + "=" * 60)
    print("Testing Synthetic Data Generation")
    print("=" * 60)

    data = generate_synthetic_weather_data(
        batch_size=2,
        in_chans=19,
        out_chans=19,
        global_img_size=(720, 1440),
        region_img_size=(1000, 1000),
        region_lat_range=(-5, 5),
        region_lon_range=(-5, 5),
        seed=42,
    )

    print(f"\nData keys: {list(data.keys())}")
    print(f"Global input shape:  {data['global_input'].shape}")
    print(f"Global target shape:  {data['global_target'].shape}")
    print(f"Region input shape:  {data['region_input'].shape}")
    print(f"Region target shape:  {data['region_target'].shape}")
    print(f"Region lat range:    {data['region_lat_range']}")
    print(f"Region lon range:    {data['region_lon_range']}")

    print("\n✓ Synthetic data generation test passed!")


def test_dataset_and_dataloader():
    """
    测试数据集和数据加载器。
    """
    print("\n" + "=" * 60)
    print("Testing Dataset and DataLoader")
    print("=" * 60)

    train_loader, val_loader = create_dataloaders(
        batch_size=2,
        num_train_samples=10,
        num_val_samples=5,
        in_chans=19,
        out_chans=19,
        global_img_size=(720, 1440),
        region_img_size=(1000, 1000),
        region_lat_range=(-5, 5),
        region_lon_range=(-5, 5),
        num_workers=0,
        seed=42,
    )

    print(f"\nTrain dataset size: {len(train_loader.dataset)}")
    print(f"Val dataset size:   {len(val_loader.dataset)}")
    print(f"Train batches:      {len(train_loader)}")
    print(f"Val batches:        {len(val_loader)}")

    batch = next(iter(train_loader))
    print(f"\nBatch keys: {list(batch.keys())}")
    print(f"Global input shape:  {batch['global_input'].shape}")
    print(f"Region input shape:  {batch['region_input'].shape}")

    print("\n✓ Dataset and DataLoader test passed!")


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
    print_parameter_stats(model)

    model.freeze_global_branch()

    print("\nAfter freezing global branch:")
    print_parameter_stats(model)

    model.unfreeze_global_branch()

    print("\nAfter unfreezing global branch:")
    print_parameter_stats(model)

    print("\n✓ Freeze/unfreeze test passed!")


def test_end_to_end():
    """
    端到端测试。
    """
    print("\n" + "=" * 60)
    print("End-to-End Test")
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
        fusion_mode="concat_conv",
        align_mode="crop",
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    data = generate_synthetic_weather_data(
        batch_size=2,
        in_chans=19,
        out_chans=19,
        global_img_size=(720, 1440),
        region_img_size=(1000, 1000),
        region_lat_range=(-5, 5),
        region_lon_range=(-5, 5),
        seed=42,
    )

    x_region = data["region_input"].to(device)
    x_global = data["global_input"].to(device)
    y_region = data["region_target"].to(device)

    print(f"\nRegion input shape:  {x_region.shape}")
    print(f"Global input shape:  {x_global.shape}")
    print(f"Region target shape: {y_region.shape}")

    optimizer.zero_grad()
    output = model(x_region, x_global)
    loss = criterion(output, y_region)

    print(f"\nOutput shape: {output.shape}")
    print(f"Loss:         {loss.item():.6f}")

    loss.backward()
    optimizer.step()

    print("\n✓ End-to-end test passed!")


def run_all_tests():
    """
    运行所有测试。
    """
    print("\n" + "=" * 60)
    print("Running All Tests")
    print("=" * 60)

    tests = [
        ("Synthetic Data Generation", test_synthetic_data),
        ("Dataset and DataLoader", test_dataset_and_dataloader),
        ("Fusion Block", test_fusion_block),
        ("Feature Aligner", test_feature_aligner),
        ("Base Model", test_base_model),
        ("Hybrid Model", test_hybrid_model),
        ("Checkpoint Save/Load", test_checkpoint_save_load),
        ("Freeze/Unfreeze", test_freeze_unfreeze),
        ("End-to-End", test_end_to_end),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n✗ {test_name} test failed!")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    print("=" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
