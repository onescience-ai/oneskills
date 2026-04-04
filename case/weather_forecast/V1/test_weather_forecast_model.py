import torch
from weather_forecast_model import WeatherForecastModel


def test_model():
    """
    测试全球短期天气预报模型
    """
    print("Testing Weather Forecast Model...")
    
    # 创建模型
    model = WeatherForecastModel()
    print("Model created successfully")
    
    # 生成随机输入
    batch_size = 1
    surface_input = torch.randn(batch_size, 7, 721, 1440)
    upper_air_input = torch.randn(batch_size, 5, 13, 721, 1440)
    
    print(f"Input shapes:")
    print(f"  Surface: {surface_input.shape}")
    print(f"  Upper air: {upper_air_input.shape}")
    
    # 前向传播
    with torch.no_grad():
        surface_output, upper_air_output = model(surface_input, upper_air_input)
    
    print(f"Output shapes:")
    print(f"  Surface: {surface_output.shape}")
    print(f"  Upper air: {upper_air_output.shape}")
    
    # 验证输出形状是否正确
    assert surface_output.shape == (batch_size, 7, 721, 1440), f"Surface output shape mismatch: {surface_output.shape}"
    assert upper_air_output.shape == (batch_size, 5, 13, 721, 1440), f"Upper air output shape mismatch: {upper_air_output.shape}"
    
    print("Model test passed! Output shapes are correct.")
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")


if __name__ == "__main__":
    test_model()
