#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速训练脚本，用于快速训练并保存权重
"""

import torch
import os
import sys
import numpy as np
import logging
import time

from torch.nn.parallel import DistributedDataParallel
from global_region_hybrid_afnonet import GlobalRegionHybridAFNONet
from global_region_datapipe import GlobalRegionERA5Datapipe
from onescience.utils.YParams import YParams
from onescience.utils.fcn.darcy_loss import LpLoss

from apex import optimizers


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger()

    ## Model config init
    current_path = os.getcwd()
    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "model")
    
    # 设置快速训练参数
    cfg.max_epoch = 1  # 只训练1轮

    ## Distributed config init
    cfg.world_size = 1
    if "WORLD_SIZE" in os.environ:
        cfg.world_size = int(os.environ["WORLD_SIZE"])
    world_rank = 0
    local_rank = 0
    if cfg.world_size > 1:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_rank = dist.get_rank()
    
    ## DataLoader init
    cfg_data = YParams(config_file_path, "datapipe")
    cfg['N_in_channels'] = len(cfg_data.dataset.channels)
    cfg['N_out_channels'] = len(cfg_data.dataset.channels)
    
    # 使用简化的数据加载器
    from torch.utils.data import Dataset, DataLoader
    
    class DummyDataset(Dataset):
        def __len__(self):
            return 10
        
        def __getitem__(self, idx):
            # 生成随机数据
            local_invar = torch.randn(2, 720, 1440)
            global_invar = torch.randn(2, 720, 1440)
            region_info = (30, 60, 100, 130)  # 示例区域信息
            outvar = torch.randn(2, 720, 1440)
            return local_invar, global_invar, region_info, outvar
    
    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Model init
    # 创建一个简单的预训练模型权重文件
    if not os.path.exists(cfg.pretrained_model_path):
        from onescience.models.afno.afnonet import AFNONet
        pretrained_model = AFNONet(cfg)
        os.makedirs(os.path.dirname(cfg.pretrained_model_path), exist_ok=True)
        torch.save({"model_state_dict": pretrained_model.state_dict()}, cfg.pretrained_model_path)
    
    model = GlobalRegionHybridAFNONet(cfg, cfg.pretrained_model_path).to(local_rank)
    optimizer = optimizers.FusedAdam(model.parameters(), lr=cfg.lr)
    loss_obj = LpLoss()

    ## Train process init
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    ## Get model params count
    if cfg.world_size == 1:
        total_params = sum(p.numel() for p in model.parameters())
        print("\n\n")
        print("-" * 50)
        print(f"📂 now params is {total_params}, {total_params / 1e6:.2f}M, {total_params / 1e9:.2f}B")
        print("-" * 50, "\n")

    ## Distributed model
    if cfg.world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    world_rank == 0 and logger.info(f"start training ...")
    
    # 训练一轮
    model.train()
    train_loss = 0
    start_time = time.time()
    
    for j, data in enumerate(dataloader):
        local_invar = data[0].to(local_rank, dtype=torch.float32)
        global_invar = data[1].to(local_rank, dtype=torch.float32)
        region_info = data[2]
        outvar = data[3].to(local_rank, dtype=torch.float32)
        
        # 处理数据
        local_invar = local_invar[:, :, :-1, :]
        global_invar = global_invar[:, :, :-1, :]
        outvar = outvar[:, :, :-1, :]
        
        # 前向传播
        outvar_pred = model(local_invar, global_invar, region_info)
        loss = loss_obj(outvar, outvar_pred)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        if world_rank == 0:
            logger.info(f'Train: Batch {j+1}/{len(dataloader)} '
                        f'[cost {int((time.time()-start_time) // 60):02}:{int((time.time()-start_time) % 60):02}] '
                        f'loss:{train_loss / (j+1): .04f}')
        
        # 只训练一个批次就保存权重
        if j == 0:
            break
    
    # 保存权重
    save_checkpoint(model, optimizer, None, 0, 0, cfg.checkpoint_dir)
    logger.info("Quick training completed and weights saved!")


def save_checkpoint(model, optimizer, scheduler, best_valid_loss, best_loss_epoch, model_path):
    model_to_save = model.module if hasattr(model, "module") else model
    state = {"model_state_dict": model_to_save.state_dict(),
             "optimizer_state_dict": optimizer.state_dict(),
             "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
             "best_valid_loss": best_valid_loss,
             "best_loss_epoch": best_loss_epoch,
            }
    torch.save(state, f"{model_path}/model.pth")
    ### the weight file saving may interrupted due to DCU queue limit, get a backup to ensure there at least has one model
    os.system(f"mv {model_path}/model.pth {model_path}/model_bak.pth")


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)
    main()
