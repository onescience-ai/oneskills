#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型测试脚本，用于加载训练好的模型并进行测试
"""

import torch
import os
import sys
import numpy as np
import logging
from datetime import datetime

from global_region_hybrid_afnonet import GlobalRegionHybridAFNONet
from onescience.utils.YParams import YParams

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()


def test_model():
    """
    测试模型
    """
    current_path = os.getcwd()
    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "model")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 初始化模型
    model = GlobalRegionHybridAFNONet(cfg, cfg.pretrained_model_path).to(device)
    
    # 加载模型权重
    model_path = os.path.join(cfg.checkpoint_dir, "model_bak.pth")
    if os.path.exists(model_path):
        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        logger.error(f"Model weight not found at {model_path}")
        return
    
    # 设置模型为评估模式
    model.eval()
    
    # 创建结果保存目录
    result_dir = os.path.join(current_path, "result")
    os.makedirs(result_dir, exist_ok=True)
    
    # 生成测试数据
    logger.info("Generating test data...")
    local_invar = torch.randn(1, cfg.N_in_channels, 720, 1440).to(device)
    global_invar = torch.randn(1, cfg.N_in_channels, 720, 1440).to(device)
    region_info = (30, 60, 100, 130)  # 示例区域信息
    
    # 测试前向传播
    logger.info("Testing forward pass...")
    start_time = datetime.now()
    
    with torch.no_grad():
        outvar_pred = model(local_invar, global_invar, region_info)
    
    end_time = datetime.now()
    inference_time = (end_time - start_time).total_seconds()
    
    logger.info(f"Inference time: {inference_time:.4f} seconds")
    logger.info(f"Input shape: {local_invar.shape}")
    logger.info(f"Output shape: {outvar_pred.shape}")
    logger.info(f"Output min: {outvar_pred.min().item():.4f}")
    logger.info(f"Output max: {outvar_pred.max().item():.4f}")
    logger.info(f"Output mean: {outvar_pred.mean().item():.4f}")
    logger.info(f"Output std: {outvar_pred.std().item():.4f}")
    
    # 保存测试结果
    result_file = os.path.join(result_dir, "test_result.npy")
    np.save(result_file, outvar_pred.cpu().numpy())
    logger.info(f"Test result saved to {result_file}")
    
    # 测试模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    logger.info("Model test completed successfully!")


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)
    test_model()
