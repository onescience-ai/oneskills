import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../../"))

from weather_forecast_model import WeatherForecastModel
from onescience.memory.checkpoint import replace_function


def test_weather_forecast_model():
    """
    测试天气预报模型的基本功能。
    """
    print("=" * 80)
    print("测试天气预报模型")
    print("=" * 80)

    Batch = 2
    surface_vars = 7
    upper_air_vars = 5
    pressure_levels = 13
    height = 721
    width = 1440

    print(f"\n模型参数:")
    print(f"  Batch: {Batch}")
    print(f"  地表变量数: {surface_vars}")
    print(f"  高空变量数: {upper_air_vars}")
    print(f"  气压层数: {pressure_levels}")
    print(f"  高度: {height}")
    print(f"  宽度: {width}")

    model = WeatherForecastModel(
        surface_vars=surface_vars,
        upper_air_vars=upper_air_vars,
        pressure_levels=pressure_levels,
        height=height,
        width=width,
        embed_dim=192,
        num_heads=6,
        window_size=(6, 12),
        depth=2,
        mlp_ratio=4.0,
        drop_path=0.0,
    ).cuda()

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    surface_input = torch.randn(Batch, surface_vars, height, width).cuda()
    upper_air_input = torch.randn(Batch, upper_air_vars, pressure_levels, height, width).cuda()

    print(f"\n输入形状:")
    print(f"  地表输入: {surface_input.shape}")
    print(f"  高空输入: {upper_air_input.shape}")

    print(f"\n开始前向传播...")
    with replace_function(model,["surface_encoder_fuser", "upper_air_encoder_fuser", "surface_middle_fuser", "upper_air_middle_fuser", "surface_decoder_fuser", "upper_air_decoder_fuser",], False):
         surface_output, upper_air_output = model(surface_input, upper_air_input)
    

    print(f"\n输出形状:")
    print(f"  地表输出: {surface_output.shape}")
    print(f"  高空输出: {upper_air_output.shape}")

    assert surface_output.shape == (Batch, surface_vars, height, width), f"地表输出形状不匹配: {surface_output.shape}"
    assert upper_air_output.shape == (Batch, upper_air_vars, pressure_levels, height, width), f"高空输出形状不匹配: {upper_air_output.shape}"

    print(f"\n形状验证通过!")

    print(f"\n测试梯度流动...")
    loss = surface_output.sum() + upper_air_output.sum()
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"  警告: 参数 {name} 的梯度为 None")
        else:
            assert not torch.isnan(param.grad).any(), f"参数 {name} 的梯度包含 NaN"
            assert not torch.isinf(param.grad).any(), f"参数 {name} 的梯度包含 Inf"

    print(f"  梯度流动验证通过!")

    print(f"\n" + "=" * 80)
    print("所有测试通过!")
    print("=" * 80)


if __name__ == "__main__":
    test_weather_forecast_model()
