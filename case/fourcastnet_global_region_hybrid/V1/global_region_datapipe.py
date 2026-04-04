# Global + Region Hybrid DataPipe

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from onescience.datapipes.climate import ERA5Datapipe


class GlobalRegionERA5Datapipe(ERA5Datapipe):
    """
    全球+区域混合数据加载器，用于加载局部区域数据、全球数据和区域信息
    """
    def __init__(self, params, distributed=False):
        super().__init__(params, distributed)
        # 全球数据路径
        self.global_data_dir = params.dataset.get('global_data_dir', self.data_dir)
    
    def train_dataloader(self):
        """
        创建训练数据加载器
        """
        dataset = GlobalRegionERADataset(
            data_dir=self.data_dir,
            global_data_dir=self.global_data_dir,
            split='train',
            channels=self.channels,
            time_steps=self.time_steps,
            split_config=self.split_config
        )
        
        if self.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = torch.utils.data.RandomSampler(dataset)
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        return dataloader, sampler
    
    def val_dataloader(self):
        """
        创建验证数据加载器
        """
        dataset = GlobalRegionERADataset(
            data_dir=self.data_dir,
            global_data_dir=self.global_data_dir,
            split='val',
            channels=self.channels,
            time_steps=self.time_steps,
            split_config=self.split_config
        )
        
        if self.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        return dataloader, sampler


class GlobalRegionERADataset(Dataset):
    """
    全球+区域混合数据集
    """
    def __init__(self, data_dir, global_data_dir, split, channels, time_steps, split_config):
        self.data_dir = data_dir
        self.global_data_dir = global_data_dir
        self.split = split
        self.channels = channels
        self.time_steps = time_steps
        self.split_config = split_config
        
        # 加载文件列表
        self.file_list = self._get_file_list()
        self.global_file_list = self._get_global_file_list()
    
    def _get_file_list(self):
        """
        获取局部区域文件列表
        """
        # 这里实现获取局部区域文件列表的逻辑
        # 假设文件命名格式为: {year}_{month}_{day}_{hour}.nc
        file_list = []
        # 实现具体的文件列表获取逻辑
        return file_list
    
    def _get_global_file_list(self):
        """
        获取全球文件列表
        """
        # 这里实现获取全球文件列表的逻辑
        # 假设文件命名格式为: {year}_{month}_{day}_{hour}.nc
        file_list = []
        # 实现具体的文件列表获取逻辑
        return file_list
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """
        获取数据项
        
        Returns:
            local_data: 局部区域数据
            global_data: 全球数据
            region_info: 区域信息 (lat_min, lat_max, lon_min, lon_max)
            target: 目标数据
        """
        # 加载局部区域数据
        local_file = self.file_list[idx]
        local_data = self._load_data(local_file)
        
        # 加载对应的全球数据
        global_file = self.global_file_list[idx]
        global_data = self._load_data(global_file)
        
        # 提取区域信息（这里假设从文件名或其他地方获取）
        region_info = self._extract_region_info(local_file)
        
        # 加载目标数据
        target = self._load_target(local_file)
        
        return local_data, global_data, region_info, target
    
    def _load_data(self, file_path):
        """
        加载数据
        """
        # 这里实现具体的数据加载逻辑
        # 假设数据存储为numpy格式或netCDF格式
        # 示例：
        # data = np.load(file_path)
        # return torch.tensor(data, dtype=torch.float32)
        pass
    
    def _load_target(self, file_path):
        """
        加载目标数据
        """
        # 这里实现具体的目标数据加载逻辑
        # 示例：
        # target = np.load(file_path.replace('input', 'target'))
        # return torch.tensor(target, dtype=torch.float32)
        pass
    
    def _extract_region_info(self, file_path):
        """
        从文件路径中提取区域信息
        """
        # 这里实现从文件路径中提取区域信息的逻辑
        # 示例：
        # filename = os.path.basename(file_path)
        # region_info = filename.split('_')[1:5]  # 假设文件名包含区域信息
        # return tuple(map(float, region_info))
        # 暂时返回默认值
        return (30.0, 60.0, 100.0, 130.0)


if __name__ == "__main__":
    # 测试数据加载器
    import sys
    sys.path.append('/Users/zhao/Desktop/OneScience/dev-earth-function+commit')
    
    from onescience.utils.YParams import YParams
    
    # 创建参数
    config_file = '/Users/zhao/Desktop/OneScience/dev-earth-function+commit/onescience/examples/earth/fourcastnet/conf/config.yaml'
    params = YParams(config_file, 'datapipe')
    
    # 创建数据加载器
    datapipe = GlobalRegionERA5Datapipe(params)
    train_dataloader, train_sampler = datapipe.train_dataloader()
    
    # 测试数据加载
    for i, (local_data, global_data, region_info, target) in enumerate(train_dataloader):
        print(f"Batch {i}")
        print(f"Local data shape: {local_data.shape}")
        print(f"Global data shape: {global_data.shape}")
        print(f"Region info: {region_info}")
        print(f"Target shape: {target.shape}")
        break
