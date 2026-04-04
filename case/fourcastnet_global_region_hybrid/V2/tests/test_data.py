import sys
import torch
import os

# 添加当前 case 目录到路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.synthetic_data import (
    generate_synthetic_weather_data,
    SyntheticWeatherDataset,
    create_dataloaders,
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

    assert data['global_input'].shape == (2, 19, 720, 1440), "Global input shape mismatch!"
    assert data['global_target'].shape == (2, 19, 720, 1440), "Global target shape mismatch!"
    assert data['region_input'].shape == (2, 19, 1000, 1000), "Region input shape mismatch!"
    assert data['region_target'].shape == (2, 19, 1000, 1000), "Region target shape mismatch!"

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

    assert len(train_loader.dataset) == 10, "Train dataset size mismatch!"
    assert len(val_loader.dataset) == 5, "Val dataset size mismatch!"

    batch = next(iter(train_loader))
    print(f"\nBatch keys: {list(batch.keys())}")
    print(f"Global input shape:  {batch['global_input'].shape}")
    print(f"Region input shape:  {batch['region_input'].shape}")

    assert batch['global_input'].shape == (2, 19, 720, 1440), "Global input shape mismatch!"
    assert batch['region_input'].shape == (2, 19, 1000, 1000), "Region input shape mismatch!"

    print("\n✓ Dataset and DataLoader test passed!")


if __name__ == "__main__":
    try:
        test_synthetic_data()
        test_dataset_and_dataloader()
        print("\n" + "=" * 60)
        print("All Data Layer Tests Passed!")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"\n✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
