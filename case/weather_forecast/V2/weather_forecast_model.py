import torch
import torch.nn as nn
from onescience.modules import OneEmbedding, OneFuser, OneSample, OneRecovery


class WeatherForecastModel(nn.Module):
    """
    全球短期天气预报模型。

    基于Pangu系列组件的轻量级编码器-解码器架构，采用分离的地表和高空分支处理。
    输入当前时刻的全球气象场，输出下一时刻的预报结果。

    Args:
        surface_vars (int): 地表变量数，默认为7
        upper_air_vars (int): 高空变量数，默认为5
        pressure_levels (int): 气压层数，默认为13
        height (int): 纬度网格数，默认为721
        width (int): 经度网格数，默认为1440
        embed_dim (int): Embedding维度，默认为192
        num_heads (int): 注意力头数，默认为6
        window_size (tuple): 窗口大小，地表分支为(6, 12)，高空分支为(2, 6, 12)
        depth (int): 网络深度，默认为2
        mlp_ratio (float): MLP比例，默认为4.0
        drop_path (float): DropPath比例，默认为0.0
    """

    def __init__(
        self,
        surface_vars=7,
        upper_air_vars=5,
        pressure_levels=13,
        height=721,
        width=1440,
        embed_dim=192,
        num_heads=6,
        window_size=(6, 12),
        depth=2,
        mlp_ratio=4.0,
        drop_path=0.0,
    ):
        super().__init__()

        self.surface_vars = surface_vars
        self.upper_air_vars = upper_air_vars
        self.pressure_levels = pressure_levels
        self.height = height
        self.width = width
        self.embed_dim = embed_dim

        surface_patch_size = (4, 4)
        upper_air_patch_size = (2, 4, 4)

        surface_patch_height = (height + surface_patch_size[0] - 1) // surface_patch_size[0]
        surface_patch_width = (width + surface_patch_size[1] - 1) // surface_patch_size[1]

        upper_air_patch_pressure = (pressure_levels + upper_air_patch_size[0] - 1) // upper_air_patch_size[0]
        upper_air_patch_height = (height + upper_air_patch_size[1] - 1) // upper_air_patch_size[1]
        upper_air_patch_width = (width + upper_air_patch_size[2] - 1) // upper_air_patch_size[2]

        self.surface_patch_resolution = (surface_patch_height, surface_patch_width)
        self.upper_air_patch_resolution = (upper_air_patch_pressure, upper_air_patch_height, upper_air_patch_width)

        surface_downsampled_height = (surface_patch_height + 1) // 2
        surface_downsampled_width = (surface_patch_width + 1) // 2

        upper_air_downsampled_pressure = upper_air_patch_pressure
        upper_air_downsampled_height = (upper_air_patch_height + 1) // 2
        upper_air_downsampled_width = (upper_air_patch_width + 1) // 2

        self.surface_downsampled_resolution = (surface_downsampled_height, surface_downsampled_width)
        self.upper_air_downsampled_resolution = (
            upper_air_downsampled_pressure,
            upper_air_downsampled_height,
            upper_air_downsampled_width,
        )

        self.surface_embedding = OneEmbedding(
            style="PanguEmbedding",
            img_size=(height, width),
            patch_size=surface_patch_size,
            Variables=surface_vars,
            embed_dim=embed_dim,
        )

        self.upper_air_embedding = OneEmbedding(
            style="PanguEmbedding",
            img_size=(pressure_levels, height, width),
            patch_size=upper_air_patch_size,
            Variables=upper_air_vars,
            embed_dim=embed_dim,
        )

        self.surface_encoder_fuser = OneFuser(
            style="PanguFuser",
            dim=embed_dim,
            input_resolution=(1, *self.surface_patch_resolution),
            depth=depth,
            num_heads=num_heads,
            window_size=(1, *window_size),
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
        )

        self.upper_air_encoder_fuser = OneFuser(
            style="PanguFuser",
            dim=embed_dim,
            input_resolution=self.upper_air_patch_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=(2, *window_size),
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
        )

        self.surface_downsample = OneSample(
            style="PanguDownSample",
            input_resolution=self.surface_patch_resolution,
            output_resolution=self.surface_downsampled_resolution,
            in_dim=embed_dim,
        )

        self.upper_air_downsample = OneSample(
            style="PanguDownSample",
            input_resolution=self.upper_air_patch_resolution,
            output_resolution=self.upper_air_downsampled_resolution,
            in_dim=embed_dim,
        )

        self.surface_middle_fuser = OneFuser(
            style="PanguFuser",
            dim=embed_dim * 2,
            input_resolution=(1, *self.surface_downsampled_resolution),
            depth=depth,
            num_heads=num_heads * 2,
            window_size=(1, *window_size),
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
        )

        self.upper_air_middle_fuser = OneFuser(
            style="PanguFuser",
            dim=embed_dim * 2,
            input_resolution=self.upper_air_downsampled_resolution,
            depth=depth,
            num_heads=num_heads * 2,
            window_size=(2, *window_size),
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
        )

        self.surface_upsample = OneSample(
            style="PanguUpSample",
            input_resolution=self.surface_downsampled_resolution,
            output_resolution=self.surface_patch_resolution,
            in_dim=embed_dim * 2,
            out_dim=embed_dim,
        )

        self.upper_air_upsample = OneSample(
            style="PanguUpSample",
            input_resolution=self.upper_air_downsampled_resolution,
            output_resolution=self.upper_air_patch_resolution,
            in_dim=embed_dim * 2,
            out_dim=embed_dim,
        )

        self.surface_decoder_fuser = OneFuser(
            style="PanguFuser",
            dim=embed_dim,
            input_resolution=(1, *self.surface_patch_resolution),
            depth=depth,
            num_heads=num_heads,
            window_size=(1, *window_size),
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
        )

        self.upper_air_decoder_fuser = OneFuser(
            style="PanguFuser",
            dim=embed_dim,
            input_resolution=self.upper_air_patch_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=(2, *window_size),
            drop_path=drop_path,
            mlp_ratio=mlp_ratio,
        )

        self.surface_recovery = OneRecovery(
            style="PanguPatchRecovery",
            img_size=(height, width),
            patch_size=surface_patch_size,
            in_chans=embed_dim * 2,
            out_chans=surface_vars,
        )

        self.upper_air_recovery = OneRecovery(
            style="PanguPatchRecovery",
            img_size=(pressure_levels, height, width),
            patch_size=upper_air_patch_size,
            in_chans=embed_dim * 2,
            out_chans=upper_air_vars,
        )

        self.surface_projection = nn.Linear(embed_dim, embed_dim * 2)
        self.upper_air_projection = nn.Linear(embed_dim, embed_dim * 2)

    def forward(self, surface_input, upper_air_input):
        """
        前向传播。

        Args:
            surface_input (torch.Tensor): 地表变量输入，形状为 (Batch, SurfaceVars, Height, Width)
            upper_air_input (torch.Tensor): 高空变量输入，形状为 (Batch, UpperAirVars, PressureLevels, Height, Width)

        Returns:
            tuple: (surface_output, upper_air_output)
                - surface_output: 地表变量输出，形状为 (Batch, SurfaceVars, Height, Width)
                - upper_air_output: 高空变量输出，形状为 (Batch, UpperAirVars, PressureLevels, Height, Width)
        """
        Batch = surface_input.shape[0]

        surface_embedded = self.surface_embedding(surface_input)
        upper_air_embedded = self.upper_air_embedding(upper_air_input)

        surface_tokens = surface_embedded.reshape(Batch, self.embed_dim, -1).permute(0, 2, 1)
        upper_air_tokens = upper_air_embedded.reshape(Batch, self.embed_dim, -1).permute(0, 2, 1)

        surface_encoder_tokens = self.surface_encoder_fuser(surface_tokens)
        upper_air_encoder_tokens = self.upper_air_encoder_fuser(upper_air_tokens)

        surface_downsampled_tokens = self.surface_downsample(surface_encoder_tokens)
        upper_air_downsampled_tokens = self.upper_air_downsample(upper_air_encoder_tokens)

        surface_middle_tokens = self.surface_middle_fuser(surface_downsampled_tokens)
        upper_air_middle_tokens = self.upper_air_middle_fuser(upper_air_downsampled_tokens)

        surface_upsampled_tokens = self.surface_upsample(surface_middle_tokens)
        upper_air_upsampled_tokens = self.upper_air_upsample(upper_air_middle_tokens)

        surface_decoder_tokens = self.surface_decoder_fuser(surface_upsampled_tokens)
        upper_air_decoder_tokens = self.upper_air_decoder_fuser(upper_air_upsampled_tokens)

        surface_projected = self.surface_projection(surface_decoder_tokens)
        upper_air_projected = self.upper_air_projection(upper_air_decoder_tokens)

        surface_feature_map = surface_projected.permute(0, 2, 1).reshape(
            Batch, self.embed_dim * 2, *self.surface_patch_resolution
        )
        upper_air_feature_map = upper_air_projected.permute(0, 2, 1).reshape(
            Batch, self.embed_dim * 2, *self.upper_air_patch_resolution
        )

        surface_output = self.surface_recovery(surface_feature_map)
        upper_air_output = self.upper_air_recovery(upper_air_feature_map)

        return surface_output, upper_air_output
