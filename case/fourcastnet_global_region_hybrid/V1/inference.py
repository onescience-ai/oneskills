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


def main():
    current_path = os.getcwd()
    config_file_path = os.path.join(current_path, "conf/config.yaml")
    cfg = YParams(config_file_path, "model")
    cfg_data = YParams(config_file_path, "datapipe")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    model = GlobalRegionHybridAFNONet(cfg, cfg.pretrained_model_path).to(device)
    
    # 加载模型权重
    if os.path.exists(f"{cfg.checkpoint_dir}/model_bak.pth"):
        logger.info(f"Loading model from {cfg.checkpoint_dir}/model_bak.pth")
        checkpoint = torch.load(f"{cfg.checkpoint_dir}/model_bak.pth", map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        logger.error(f"Model weight not found at {cfg.checkpoint_dir}/model_bak.pth")
        return
    
    # 设置模型为评估模式
    model.eval()
    
    # 创建结果保存目录
    result_dir = os.path.join(current_path, "result")
    os.makedirs(result_dir, exist_ok=True)
    
    # 加载测试数据
    # 这里需要根据实际情况实现数据加载逻辑
    # 示例：
    # test_data = load_test_data()
    
    # 推理
    with torch.no_grad():
        # 示例推理代码
        # for i, (local_invar, global_invar, region_info, _) in enumerate(test_data):
        #     local_invar = local_invar.to(device, dtype=torch.float32)
        #     global_invar = global_invar.to(device, dtype=torch.float32)
        #     
        #     # 前向传播
        #     outvar_pred = model(local_invar, global_invar, region_info)
        #     
        #     # 保存结果
        #     save_result(outvar_pred, i, result_dir)
        
        # 暂时使用随机数据进行测试
        logger.info("Testing model with random data...")
        local_invar = torch.randn(1, cfg.N_in_channels, 720, 1440).to(device)
        global_invar = torch.randn(1, cfg.N_in_channels, 720, 1440).to(device)
        region_info = (30, 60, 100, 130)  # 示例区域信息
        
        # 前向传播
        start_time = datetime.now()
        outvar_pred = model(local_invar, global_invar, region_info)
        end_time = datetime.now()
        
        logger.info(f"Inference time: {(end_time - start_time).total_seconds():.2f} seconds")
        logger.info(f"Input shape: {local_invar.shape}")
        logger.info(f"Output shape: {outvar_pred.shape}")
        
        # 保存结果
        result_file = os.path.join(result_dir, "test_result.npy")
        np.save(result_file, outvar_pred.cpu().numpy())
        logger.info(f"Result saved to {result_file}")


def save_result(result, index, result_dir):
    """
    保存推理结果
    """
    result_file = os.path.join(result_dir, f"result_{index}.npy")
    np.save(result_file, result.cpu().numpy())
    logger.info(f"Result saved to {result_file}")


def load_test_data():
    """
    加载测试数据
    """
    # 这里实现具体的测试数据加载逻辑
    pass


if __name__ == "__main__":
    current_path = os.getcwd()
    sys.path.append(current_path)
    main()
