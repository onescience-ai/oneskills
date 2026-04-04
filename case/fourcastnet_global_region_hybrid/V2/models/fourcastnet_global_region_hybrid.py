import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

import sys
import os

# 添加 OneScience 根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from onescience.modules import OneEmbedding, OneFuser
from components.global_region_fusion import GlobalRegionFusionBlock, GlobalFeatureAligner


class FourCastNetGlobalRegionHybrid(nn.Module):
    """
    全球+区域混合 FourCastNet 模型。

    该模型包含两个分支：
    1. 全球分支：使用预训练的 FourCastNet 提取全球特征
    2. 区域分支：训练一个新的 FourCastNet 变体，在中间层融合全球特征

    Args:
        global_img_size (tuple): 全球图像尺寸 (H_global, W_global)
        global_patch_size (tuple): 全球 patch 尺寸 (ph_global, pw_global)
        global_embed_dim (int): 全球 embedding 维度
        global_depth (int): 全球 trunk 层数
        global_checkpoint_path (str): 全球模型 checkpoint 路径

        region_img_size (tuple): 区域图像尺寸 (H_region, W_region)
        region_patch_size (tuple): 区域 patch 尺寸 (ph_region, pw_region)
        region_lat_range (tuple): 区域纬度范围 (lat_min, lat_max)
        region_lon_range (tuple): 区域经度范围 (lon_min, lon_max)
        region_embed_dim (int): 区域 embedding 维度
        region_depth (int): 区域 trunk 层数

        in_chans (int): 输入变量通道数
        out_chans (int): 输出变量通道数
        fusion_layer_idx (int): 融合层位置（在区域 trunk 中的索引）
        fusion_mode (str): 融合模式，可选 "concat_mlp", "concat_conv", "gate"
        align_mode (str): 全球特征对齐模式，可选 "crop", "interpolate"

        mlp_ratio (float): MLP 隐层放大倍数
        drop_rate (float): dropout 比例
        drop_path_rate (float): Stochastic Depth 比例
        num_blocks (int): AFNO 通道分块数
        sparsity_threshold (float): AFNO soft shrink 阈值
        hard_thresholding_fraction (float): AFNO 保留频率模式比例
    """

    def __init__(
        self,
        global_img_size=(720, 1440),
        global_patch_size=(8, 8),
        global_embed_dim=768,
        global_depth=12,
        global_checkpoint_path=None,

        region_img_size=(1000, 1000),
        region_patch_size=(8, 8),
        region_lat_range=(-5, 5),
        region_lon_range=(-5, 5),
        region_embed_dim=768,
        region_depth=6,

        in_chans=19,
        out_chans=19,
        fusion_layer_idx=3,
        fusion_mode="concat_conv",
        align_mode="crop",

        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        num_blocks=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
    ):
        super().__init__()

        self.global_img_size = global_img_size
        self.global_patch_size = global_patch_size
        self.global_embed_dim = global_embed_dim
        self.global_depth = global_depth
        self.global_checkpoint_path = global_checkpoint_path

        self.region_img_size = region_img_size
        self.region_patch_size = region_patch_size
        self.region_lat_range = region_lat_range
        self.region_lon_range = region_lon_range
        self.region_embed_dim = region_embed_dim
        self.region_depth = region_depth

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.fusion_layer_idx = fusion_layer_idx
        self.fusion_mode = fusion_mode
        self.align_mode = align_mode

        self.num_blocks = num_blocks

        global_num_patches = (global_img_size[1] // global_patch_size[1]) * (global_img_size[0] // global_patch_size[0])
        self.global_patch_grid_height = global_img_size[0] // global_patch_size[0]
        self.global_patch_grid_width = global_img_size[1] // global_patch_size[1]

        region_num_patches = (region_img_size[1] // region_patch_size[1]) * (region_img_size[0] // region_patch_size[0])
        self.region_patch_grid_height = region_img_size[0] // region_patch_size[0]
        self.region_patch_grid_width = region_img_size[1] // region_patch_size[1]

        drop_path_global = np.linspace(0, drop_path_rate, global_depth).tolist()
        drop_path_region = np.linspace(0, drop_path_rate, region_depth).tolist()

        self.global_pos_embed = nn.Parameter(torch.zeros(1, global_num_patches, global_embed_dim))
        self.global_pos_drop = nn.Dropout(p=drop_rate)

        self.global_patch_embed = OneEmbedding(
            style="FourCastNetEmbedding",
            img_size=global_img_size,
            patch_size=global_patch_size,
            in_chans=in_chans,
            embed_dim=global_embed_dim,
        )

        self.global_blocks = nn.ModuleList([
            OneFuser(
                style="FourCastNetFuser",
                dim=global_embed_dim,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=drop_path_global[i],
                num_blocks=num_blocks,
                sparsity_threshold=sparsity_threshold,
                hard_thresholding_fraction=hard_thresholding_fraction,
            )
            for i in range(global_depth)
        ])

        self.region_pos_embed = nn.Parameter(torch.zeros(1, region_num_patches, region_embed_dim))
        self.region_pos_drop = nn.Dropout(p=drop_rate)

        self.region_patch_embed = OneEmbedding(
            style="FourCastNetEmbedding",
            img_size=region_img_size,
            patch_size=region_patch_size,
            in_chans=in_chans,
            embed_dim=region_embed_dim,
        )

        self.region_blocks = nn.ModuleList([
            OneFuser(
                style="FourCastNetFuser",
                dim=region_embed_dim,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=drop_path_region[i],
                num_blocks=num_blocks,
                sparsity_threshold=sparsity_threshold,
                hard_thresholding_fraction=hard_thresholding_fraction,
            )
            for i in range(region_depth)
        ])

        self.global_feature_aligner = GlobalFeatureAligner(
            global_grid_size=(self.global_patch_grid_height, self.global_patch_grid_width),
            global_img_size=global_img_size,
            align_mode=align_mode,
        )

        self.fusion_block = GlobalRegionFusionBlock(
            dim_region=region_embed_dim,
            dim_global=global_embed_dim,
            dim_out=region_embed_dim,
            fusion_mode=fusion_mode,
            dropout=drop_rate,
        )

        self.head = nn.Linear(
            region_embed_dim,
            self.out_chans * self.region_patch_size[0] * self.region_patch_size[1],
            bias=False,
        )

        self._init_weights()

    def _init_weights(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(init_weights)
        nn.init.trunc_normal_(self.global_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.region_pos_embed, std=0.02)

    def load_global_pretrained(self, checkpoint_path=None):
        """
        加载全球模型预训练权重。

        Args:
            checkpoint_path (str): checkpoint 文件路径
        """
        if checkpoint_path is None:
            checkpoint_path = self.global_checkpoint_path

        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            if "model" in state_dict:
                state_dict = state_dict["model"]

            global_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("global_") or k in ["patch_embed", "blocks", "pos_embed", "head"]:
                    new_key = k
                    if not k.startswith("global_"):
                        new_key = f"global_{k}"
                    global_state_dict[new_key] = v

            self.load_state_dict(global_state_dict, strict=False)
            print(f"Loaded global pretrained weights from {checkpoint_path}")

    def freeze_global_branch(self):
        """
        冻结全球分支参数。
        """
        for name, param in self.named_parameters():
            if name.startswith("global_"):
                param.requires_grad = False
        print("Global branch frozen")

    def unfreeze_global_branch(self):
        """
        解冻全球分支参数。
        """
        for name, param in self.named_parameters():
            if name.startswith("global_"):
                param.requires_grad = True
        print("Global branch unfrozen")

    def forward(self, x_region, x_global=None):
        """
        Args:
            x_region: (Batch, in_chans, H_region, W_region)
            x_global: (Batch, in_chans, H_global, W_global), optional

        Returns:
            out: (Batch, out_chans, H_region, W_region)
        """
        Batch = x_region.shape[0]

        if x_global is None:
            x_global = torch.zeros(Batch, self.in_chans, *self.global_img_size, device=x_region.device)

        x_region = self.region_patch_embed(x_region)
        x_region = x_region + self.region_pos_embed
        x_region = self.region_pos_drop(x_region)
        x_region = x_region.reshape(Batch, self.region_patch_grid_height, self.region_patch_grid_width, self.region_embed_dim)

        x_global = self.global_patch_embed(x_global)
        x_global = x_global + self.global_pos_embed
        x_global = self.global_pos_drop(x_global)
        x_global = x_global.reshape(Batch, self.global_patch_grid_height, self.global_patch_grid_width, self.global_embed_dim)

        for blk in self.global_blocks:
            x_global = blk(x_global)

        global_feat_aligned = self.global_feature_aligner(
            x_global,
            self.region_lat_range,
            self.region_lon_range,
            (self.region_patch_grid_height, self.region_patch_grid_width),
        )

        for i, blk in enumerate(self.region_blocks):
            x_region = blk(x_region)

            if i == self.fusion_layer_idx:
                x_region = self.fusion_block(x_region, global_feat_aligned)

        x_region = self.head(x_region)
        x_region = rearrange(
            x_region,
            "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=self.region_patch_size[0],
            p2=self.region_patch_size[1],
            h=self.region_patch_grid_height,
            w=self.region_patch_grid_width,
        )

        return x_region


class FourCastNetBase(nn.Module):
    """
    基座 FourCastNet 模型，用于快速训练获取 checkpoint。

    Args:
        img_size (tuple): 图像尺寸 (H, W)
        patch_size (tuple): patch 尺寸 (ph, pw)
        in_chans (int): 输入变量通道数
        out_chans (int): 输出变量通道数
        embed_dim (int): embedding 维度
        depth (int): trunk 层数
        mlp_ratio (float): MLP 隐层放大倍数
        drop_rate (float): dropout 比例
        drop_path_rate (float): Stochastic Depth 比例
        num_blocks (int): AFNO 通道分块数
        sparsity_threshold (float): AFNO soft shrink 阈值
        hard_thresholding_fraction (float): AFNO 保留频率模式比例
    """

    def __init__(
        self,
        img_size=(720, 1440),
        patch_size=(8, 8),
        in_chans=19,
        out_chans=19,
        embed_dim=768,
        depth=12,
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        num_blocks=8,
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = num_blocks

        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        drop_path = np.linspace(0, drop_path_rate, depth).tolist()

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.patch_grid_height = img_size[0] // self.patch_size[0]
        self.patch_grid_width = img_size[1] // self.patch_size[1]

        self.patch_embed = OneEmbedding(
            style="FourCastNetEmbedding",
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.blocks = nn.ModuleList([
            OneFuser(
                style="FourCastNetFuser",
                dim=embed_dim,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                drop_path=drop_path[i],
                num_blocks=num_blocks,
                sparsity_threshold=sparsity_threshold,
                hard_thresholding_fraction=hard_thresholding_fraction,
            )
            for i in range(depth)
        ])

        self.head = nn.Linear(
            embed_dim,
            self.out_chans * self.patch_size[0] * self.patch_size[1],
            bias=False,
        )

        self._init_weights()

    def _init_weights(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(init_weights)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        Args:
            x: (Batch, in_chans, Height, Width)

        Returns:
            out: (Batch, out_chans, Height, Width)
        """
        Batch = x.shape[0]

        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = x.reshape(Batch, self.patch_grid_height, self.patch_grid_width, self.embed_dim)
        for blk in self.blocks:
            x = blk(x)

        x = self.head(x)
        x = rearrange(
            x,
            "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            h=self.patch_grid_height,
            w=self.patch_grid_width,
        )
        return x
