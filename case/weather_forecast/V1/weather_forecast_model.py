import torch
from torch import nn

from onescience.modules.embedding.oneembedding import OneEmbedding
from onescience.modules.sample.onesample import OneSample
from onescience.modules.recovery.onerecovery import OneRecovery
from onescience.modules.layer.layers.transformer_layers import Transformer2DBlock, Transformer3DBlock


class WeatherForecastModel(nn.Module):
    """
    全球短期天气预报模型
    
    输入当前时刻的全球气象场，输出下一时刻的预报结果
    采用分离的地表和高空分支处理，使用 Transformer 架构捕获时空依赖关系
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
    ):
        super().__init__()
        
        # 地表分支
        self.surface_embedding = OneEmbedding(
            style="PanguEmbedding",
            img_size=(height, width),
            patch_size=(4, 4),
            Variables=surface_vars,
            embed_dim=embed_dim
        )
        
        # 高空分支
        self.upper_air_embedding = OneEmbedding(
            style="PanguEmbedding",
            img_size=(pressure_levels, height, width),
            patch_size=(2, 4, 4),
            Variables=upper_air_vars,
            embed_dim=embed_dim
        )
        
        # 地表分支的分辨率变化
        self.surface_input_res = (height // 4 + (1 if height % 4 != 0 else 0), 
                               width // 4 + (1 if width % 4 != 0 else 0))
        self.surface_middle_res = (self.surface_input_res[0] // 2 + (1 if self.surface_input_res[0] % 2 != 0 else 0),
                                self.surface_input_res[1] // 2 + (1 if self.surface_input_res[1] % 2 != 0 else 0))
        
        # 高空分支的分辨率变化
        self.upper_air_input_res = (pressure_levels // 2 + (1 if pressure_levels % 2 != 0 else 0),
                                  height // 4 + (1 if height % 4 != 0 else 0),
                                  width // 4 + (1 if width % 4 != 0 else 0))
        self.upper_air_middle_res = (self.upper_air_input_res[0],
                                   self.upper_air_input_res[1] // 2 + (1 if self.upper_air_input_res[1] % 2 != 0 else 0),
                                   self.upper_air_input_res[2] // 2 + (1 if self.upper_air_input_res[2] % 2 != 0 else 0))
        
        # 地表分支下采样
        self.surface_downsample = OneSample(
            style="PanguDownSample",
            input_resolution=self.surface_input_res,
            output_resolution=self.surface_middle_res,
            in_dim=embed_dim
        )
        
        # 高空分支下采样
        self.upper_air_downsample = OneSample(
            style="PanguDownSample",
            input_resolution=self.upper_air_input_res,
            output_resolution=self.upper_air_middle_res,
            in_dim=embed_dim
        )
        
        # 地表分支上采样
        self.surface_upsample = OneSample(
            style="PanguUpSample",
            input_resolution=self.surface_middle_res,
            output_resolution=self.surface_input_res,
            in_dim=embed_dim * 2,
            out_dim=embed_dim
        )
        
        # 高空分支上采样
        self.upper_air_upsample = OneSample(
            style="PanguUpSample",
            input_resolution=self.upper_air_middle_res,
            output_resolution=self.upper_air_input_res,
            in_dim=embed_dim * 2,
            out_dim=embed_dim
        )
        
        # 地表分支编码器
        self.surface_encoder_blocks = nn.ModuleList([
            Transformer2DBlock(
                dim=embed_dim,
                input_resolution=self.surface_input_res,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0) if i % 2 == 0 else None,
                mlp_ratio=mlp_ratio
            )
            for i in range(depth)
        ])
        
        # 地表分支中间层
        self.surface_middle_blocks = nn.ModuleList([
            Transformer2DBlock(
                dim=embed_dim * 2,
                input_resolution=self.surface_middle_res,
                num_heads=num_heads * 2,
                window_size=window_size,
                shift_size=(0, 0) if i % 2 == 0 else None,
                mlp_ratio=mlp_ratio
            )
            for i in range(depth)
        ])
        
        # 高空分支编码器
        self.upper_air_encoder_blocks = nn.ModuleList([
            Transformer3DBlock(
                dim=embed_dim,
                input_resolution=self.upper_air_input_res,
                num_heads=num_heads,
                window_size=(2, 6, 12),
                shift_size=(0, 0, 0) if i % 2 == 0 else None,
                mlp_ratio=mlp_ratio
            )
            for i in range(depth)
        ])
        
        # 高空分支中间层
        self.upper_air_middle_blocks = nn.ModuleList([
            Transformer3DBlock(
                dim=embed_dim * 2,
                input_resolution=self.upper_air_middle_res,
                num_heads=num_heads * 2,
                window_size=(2, 6, 12),
                shift_size=(0, 0, 0) if i % 2 == 0 else None,
                mlp_ratio=mlp_ratio
            )
            for i in range(depth)
        ])
        
        # 地表分支解码器
        self.surface_decoder_blocks = nn.ModuleList([
            Transformer2DBlock(
                dim=embed_dim,
                input_resolution=self.surface_input_res,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0) if i % 2 == 0 else None,
                mlp_ratio=mlp_ratio
            )
            for i in range(depth)
        ])
        
        # 高空分支解码器
        self.upper_air_decoder_blocks = nn.ModuleList([
            Transformer3DBlock(
                dim=embed_dim,
                input_resolution=self.upper_air_input_res,
                num_heads=num_heads,
                window_size=(2, 6, 12),
                shift_size=(0, 0, 0) if i % 2 == 0 else None,
                mlp_ratio=mlp_ratio
            )
            for i in range(depth)
        ])
        
        # 特征融合模块
        self.fusion_surface = nn.Linear(embed_dim * 2, embed_dim * 2)
        self.fusion_upper_air = nn.Linear(embed_dim * 2, embed_dim * 2)
        
        # 地表分支输出恢复
        self.surface_recovery = OneRecovery(
            style="PanguPatchRecovery",
            img_size=(height, width),
            patch_size=(4, 4),
            in_chans=embed_dim * 2,
            out_chans=surface_vars
        )
        
        # 高空分支输出恢复
        self.upper_air_recovery = OneRecovery(
            style="PanguPatchRecovery",
            img_size=(pressure_levels, height, width),
            patch_size=(2, 4, 4),
            in_chans=embed_dim * 2,
            out_chans=upper_air_vars
        )
    
    def forward(self, surface_input, upper_air_input):
        """
        前向传播
        
        Args:
            surface_input: 地表变量输入，形状为 (Batch, SurfaceVars, Height, Width)
            upper_air_input: 高空变量输入，形状为 (Batch, UpperAirVars, PressureLevels, Height, Width)
        
        Returns:
            surface_output: 地表变量输出，形状为 (Batch, SurfaceVars, Height, Width)
            upper_air_output: 高空变量输出，形状为 (Batch, UpperAirVars, PressureLevels, Height, Width)
        """
        # 1. Embedding 层
        surface_emb = self.surface_embedding(surface_input)  # (Batch, embed_dim, H1, W1)
        upper_air_emb = self.upper_air_embedding(upper_air_input)  # (Batch, embed_dim, P1, H1, W1)
        
        # 2. 形状转换：从特征图转换为 token 序列
        B, C, H1, W1 = surface_emb.shape
        surface_tokens = surface_emb.view(B, C, -1).transpose(1, 2)  # (Batch, H1*W1, embed_dim)
        
        B, C, P1, H1, W1 = upper_air_emb.shape
        upper_air_tokens = upper_air_emb.view(B, C, -1).transpose(1, 2)  # (Batch, P1*H1*W1, embed_dim)
        
        # 3. 编码阶段 - 地表分支
        for blk in self.surface_encoder_blocks:
            surface_tokens = blk(surface_tokens)
        
        # 保存跳连接
        surface_skip = surface_tokens
        
        # 地表分支下采样
        surface_tokens = self.surface_downsample(surface_tokens)  # (Batch, H2*W2, 2*embed_dim)
        
        # 地表分支中间层
        for blk in self.surface_middle_blocks:
            surface_tokens = blk(surface_tokens)
        
        # 4. 编码阶段 - 高空分支
        for blk in self.upper_air_encoder_blocks:
            upper_air_tokens = blk(upper_air_tokens)
        
        # 保存跳连接
        upper_air_skip = upper_air_tokens
        
        # 高空分支下采样
        upper_air_tokens = self.upper_air_downsample(upper_air_tokens)  # (Batch, P2*H2*W2, 2*embed_dim)
        
        # 高空分支中间层
        for blk in self.upper_air_middle_blocks:
            upper_air_tokens = blk(upper_air_tokens)
        
        # 5. 特征融合
        surface_tokens = self.fusion_surface(surface_tokens)
        upper_air_tokens = self.fusion_upper_air(upper_air_tokens)
        
        # 6. 解码阶段 - 地表分支
        surface_tokens = self.surface_upsample(surface_tokens)  # (Batch, H1*W1, embed_dim)
        
        # 融合跳连接
        surface_tokens = surface_tokens + surface_skip
        
        for blk in self.surface_decoder_blocks:
            surface_tokens = blk(surface_tokens)
        
        # 7. 解码阶段 - 高空分支
        upper_air_tokens = self.upper_air_upsample(upper_air_tokens)  # (Batch, P1*H1*W1, embed_dim)
        
        # 融合跳连接
        upper_air_tokens = upper_air_tokens + upper_air_skip
        
        for blk in self.upper_air_decoder_blocks:
            upper_air_tokens = blk(upper_air_tokens)
        
        # 8. 形状转换：从 token 序列转换回特征图
        surface_tokens = surface_tokens.transpose(1, 2).view(B, -1, H1, W1)  # (Batch, 2*embed_dim, H1, W1)
        upper_air_tokens = upper_air_tokens.transpose(1, 2).view(B, -1, P1, H1, W1)  # (Batch, 2*embed_dim, P1, H1, W1)
        
        # 9. 输出恢复
        surface_output = self.surface_recovery(surface_tokens)  # (Batch, SurfaceVars, Height, Width)
        upper_air_output = self.upper_air_recovery(upper_air_tokens)  # (Batch, UpperAirVars, PressureLevels, Height, Width)
        
        return surface_output, upper_air_output


if __name__ == "__main__":
    # 模型测试
    model = WeatherForecastModel()
    
    # 生成随机输入
    batch_size = 2
    surface_input = torch.randn(batch_size, 7, 721, 1440)
    upper_air_input = torch.randn(batch_size, 5, 13, 721, 1440)
    
    # 前向传播
    surface_output, upper_air_output = model(surface_input, upper_air_input)
    
    # 打印输出形状
    print(f"Surface output shape: {surface_output.shape}")
    print(f"Upper air output shape: {upper_air_output.shape}")
