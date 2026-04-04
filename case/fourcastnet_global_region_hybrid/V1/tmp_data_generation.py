#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
虚拟数据生成脚本，用于生成全球+区域混合模型的测试数据
"""

import os
import numpy as np
import torch
from datetime import datetime, timedelta


def generate_virtual_data(data_dir, global_data_dir, static_dir, num_samples=10):
    """
    生成虚拟数据
    
    Args:
        data_dir: 局部区域数据保存目录
        global_data_dir: 全球数据保存目录
        static_dir: 静态数据保存目录
        num_samples: 生成的样本数量
    """
    # 创建目录
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(global_data_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)
    
    # 生成样本
    start_date = datetime(2020, 1, 1)
    for i in range(num_samples):
        # 生成日期
        date = start_date + timedelta(days=i)
        date_str = date.strftime("%Y%m%d")
        
        # 生成局部区域数据
        # 假设输入输出通道数为2，分辨率为720x1440
        local_input = np.random.randn(2, 720, 1440).astype(np.float32)
        local_output = np.random.randn(2, 720, 1440).astype(np.float32)
        
        # 生成全球数据
        global_input = np.random.randn(2, 720, 1440).astype(np.float32)
        
        # 保存数据
        # 局部区域数据
        local_input_file = os.path.join(data_dir, f"input_{date_str}.npy")
        local_output_file = os.path.join(data_dir, f"output_{date_str}.npy")
        np.save(local_input_file, local_input)
        np.save(local_output_file, local_output)
        
        # 全球数据
        global_input_file = os.path.join(global_data_dir, f"input_{date_str}.npy")
        np.save(global_input_file, global_input)
        
        print(f"Generated sample {i+1}/{num_samples}: {date_str}")
    
    # 生成静态数据
    land_mask = np.random.randn(1, 720, 1440).astype(np.float32)
    static_file = os.path.join(static_dir, "land_mask.npy")
    np.save(static_file, land_mask)
    
    print("Virtual data generation completed!")


if __name__ == "__main__":
    # 设置数据目录
    current_path = os.getcwd()
    data_dir = os.path.join(current_path, "data", "region")
    global_data_dir = os.path.join(current_path, "data", "global")
    static_dir = os.path.join(current_path, "data", "static")
    
    # 生成虚拟数据
    generate_virtual_data(data_dir, global_data_dir, static_dir, num_samples=10)
