import numpy as np
import torch
from typing import Tuple, Optional


def generate_synthetic_weather_data(
    batch_size: int = 4,
    in_chans: int = 19,
    out_chans: int = 19,
    global_img_size: Tuple[int, int] = (720, 1440),
    region_img_size: Tuple[int, int] = (1000, 1000),
    region_lat_range: Tuple[float, float] = (-5, 5),
    region_lon_range: Tuple[float, float] = (-5, 5),
    seed: Optional[int] = None,
) -> dict:
    """
    生成虚拟气象数据。

    Args:
        batch_size: batch size
        in_chans: 输入变量通道数
        out_chans: 输出变量通道数
        global_img_size: 全球图像尺寸 (H, W)，默认 720x1440 (0.25度)
        region_img_size: 区域图像尺寸 (H, W)，默认 1000x1000 (0.01度, 10度范围)
        region_lat_range: 区域纬度范围 (lat_min, lat_max)
        region_lon_range: 区域经度范围 (lon_min, lon_max)
        seed: 随机种子

    Returns:
        data: 包含以下键的字典
            - global_input: (Batch, in_chans, H_global, W_global)
            - global_target: (Batch, out_chans, H_global, W_global)
            - region_input: (Batch, in_chans, H_region, W_region)
            - region_target: (Batch, out_chans, H_region, W_region)
            - region_lat_range: (lat_min, lat_max)
            - region_lon_range: (lon_min, lon_max)
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    H_global, W_global = global_img_size
    H_region, W_region = region_img_size

    global_input = np.random.randn(batch_size, in_chans, H_global, W_global).astype(np.float32)
    global_target = np.random.randn(batch_size, out_chans, H_global, W_global).astype(np.float32)

    region_input = np.random.randn(batch_size, in_chans, H_region, W_region).astype(np.float32)
    region_target = np.random.randn(batch_size, out_chans, H_region, W_region).astype(np.float32)

    global_input = torch.from_numpy(global_input)
    global_target = torch.from_numpy(global_target)
    region_input = torch.from_numpy(region_input)
    region_target = torch.from_numpy(region_target)

    data = {
        "global_input": global_input,
        "global_target": global_target,
        "region_input": region_input,
        "region_target": region_target,
        "region_lat_range": region_lat_range,
        "region_lon_range": region_lon_range,
    }

    return data


class SyntheticWeatherDataset(torch.utils.data.Dataset):
    """
    虚拟气象数据集。

    Args:
        num_samples: 样本数量
        in_chans: 输入变量通道数
        out_chans: 输出变量通道数
        global_img_size: 全球图像尺寸
        region_img_size: 区域图像尺寸
        region_lat_range: 区域纬度范围
        region_lon_range: 区域经度范围
        seed: 随机种子
    """

    def __init__(
        self,
        num_samples: int = 1000,
        in_chans: int = 19,
        out_chans: int = 19,
        global_img_size: Tuple[int, int] = (720, 1440),
        region_img_size: Tuple[int, int] = (1000, 1000),
        region_lat_range: Tuple[float, float] = (-5, 5),
        region_lon_range: Tuple[float, float] = (-5, 5),
        seed: Optional[int] = None,
    ):
        self.num_samples = num_samples
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.global_img_size = global_img_size
        self.region_img_size = region_img_size
        self.region_lat_range = region_lat_range
        self.region_lon_range = region_lon_range

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.global_inputs = []
        self.global_targets = []
        self.region_inputs = []
        self.region_targets = []

        H_global, W_global = global_img_size
        H_region, W_region = region_img_size

        for _ in range(num_samples):
            global_input = np.random.randn(in_chans, H_global, W_global).astype(np.float32)
            global_target = np.random.randn(out_chans, H_global, W_global).astype(np.float32)
            region_input = np.random.randn(in_chans, H_region, W_region).astype(np.float32)
            region_target = np.random.randn(out_chans, H_region, W_region).astype(np.float32)

            self.global_inputs.append(torch.from_numpy(global_input))
            self.global_targets.append(torch.from_numpy(global_target))
            self.region_inputs.append(torch.from_numpy(region_input))
            self.region_targets.append(torch.from_numpy(region_target))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "global_input": self.global_inputs[idx],
            "global_target": self.global_targets[idx],
            "region_input": self.region_inputs[idx],
            "region_target": self.region_targets[idx],
            "region_lat_range": self.region_lat_range,
            "region_lon_range": self.region_lon_range,
        }


def create_dataloaders(
    batch_size: int = 4,
    num_train_samples: int = 800,
    num_val_samples: int = 200,
    in_chans: int = 19,
    out_chans: int = 19,
    global_img_size: Tuple[int, int] = (720, 1440),
    region_img_size: Tuple[int, int] = (1000, 1000),
    region_lat_range: Tuple[float, float] = (-5, 5),
    region_lon_range: Tuple[float, float] = (-5, 5),
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    创建训练和验证数据加载器。

    Args:
        batch_size: batch size
        num_train_samples: 训练样本数量
        num_val_samples: 验证样本数量
        in_chans: 输入变量通道数
        out_chans: 输出变量通道数
        global_img_size: 全球图像尺寸
        region_img_size: 区域图像尺寸
        region_lat_range: 区域纬度范围
        region_lon_range: 区域经度范围
        num_workers: 数据加载线程数
        seed: 随机种子

    Returns:
        train_loader, val_loader
    """
    train_dataset = SyntheticWeatherDataset(
        num_samples=num_train_samples,
        in_chans=in_chans,
        out_chans=out_chans,
        global_img_size=global_img_size,
        region_img_size=region_img_size,
        region_lat_range=region_lat_range,
        region_lon_range=region_lon_range,
        seed=seed,
    )

    val_dataset = SyntheticWeatherDataset(
        num_samples=num_val_samples,
        in_chans=in_chans,
        out_chans=out_chans,
        global_img_size=global_img_size,
        region_img_size=region_img_size,
        region_lat_range=region_lat_range,
        region_lon_range=region_lon_range,
        seed=seed + 1,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, val_loader
