import sys
import torch
import torch.nn as nn
import os

# 添加当前 case 目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.fourcastnet_global_region_hybrid import FourCastNetGlobalRegionHybrid
from utils.synthetic_data import generate_synthetic_weather_data, create_dataloaders
from utils.checkpoint_utils import (
    save_checkpoint,
    load_checkpoint,
    print_parameter_stats,
)


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


def test_training_loop():
    """
    测试完整训练循环。
    """
    print("\n" + "=" * 60)
    print("Training Loop Test")
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

    train_loader, _ = create_dataloaders(
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

    print(f"\nTraining for 2 epochs with {len(train_loader)} batches per epoch")

    for epoch in range(2):
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            x_region = batch["region_input"].to(device)
            x_global = batch["global_input"].to(device)
            y_region = batch["region_target"].to(device)

            optimizer.zero_grad()
            output = model(x_region, x_global)
            loss = criterion(output, y_region)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/2 - Average Loss: {avg_loss:.6f}")

    print("\n✓ Training loop test passed!")


def test_model_with_checkpoint():
    """
    测试模型的 checkpoint 保存和恢复。
    """
    print("\n" + "=" * 60)
    print("Model with Checkpoint Test")
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

    optimizer.zero_grad()
    output1 = model(x_region, x_global)
    loss1 = criterion(output1, y_region)
    loss1.backward()
    optimizer.step()

    print(f"\nFirst forward pass - Loss: {loss1.item():.6f}")

    checkpoint_path = "./test_checkpoints/integration_checkpoint.pth"
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=1,
        loss=loss1.item(),
        checkpoint_path=checkpoint_path,
    )

    print(f"Checkpoint saved to: {checkpoint_path}")

    loaded_model = FourCastNetGlobalRegionHybrid(
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

    loaded_optimizer = torch.optim.AdamW(loaded_model.parameters(), lr=1e-4)

    load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=loaded_model,
        optimizer=loaded_optimizer,
        device=device,
    )

    print("Checkpoint loaded")

    loaded_model.eval()
    with torch.no_grad():
        output2 = loaded_model(x_region, x_global)

    assert torch.allclose(output1, output2, atol=1e-6), "Outputs don't match after loading checkpoint!"

    print("\n✓ Model with checkpoint test passed!")


def test_model_with_frozen_global_branch():
    """
    测试冻结全球分支的模型。
    """
    print("\n" + "=" * 60)
    print("Model with Frozen Global Branch Test")
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

    print("\nBefore freezing global branch:")
    print_parameter_stats(model)

    model.freeze_global_branch()

    print("\nAfter freezing global branch:")
    print_parameter_stats(model)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

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

    optimizer.zero_grad()
    output = model(x_region, x_global)
    loss = criterion(output, y_region)

    print(f"\nForward pass - Loss: {loss.item():.6f}")

    loss.backward()
    optimizer.step()

    print("\n✓ Model with frozen global branch test passed!")


if __name__ == "__main__":
    try:
        test_end_to_end()
        test_training_loop()
        test_model_with_checkpoint()
        test_model_with_frozen_global_branch()
        print("\n" + "=" * 60)
        print("All Integration Tests Passed!")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"\n✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
