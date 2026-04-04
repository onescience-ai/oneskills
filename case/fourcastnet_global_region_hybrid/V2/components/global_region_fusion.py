import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalRegionFusionBlock(nn.Module):
    """
    全球特征与区域特征融合模块。

    支持多种融合方式：
    - concat + MLP: 拼接后通过 MLP 融合
    - concat + Conv: 拼接后通过 Conv 融合
    - gate: 门控融合

    Args:
        dim_region (int): 区域特征维度
        dim_global (int): 全球特征维度
        dim_out (int): 输出特征维度
        fusion_mode (str): 融合模式，可选 "concat_mlp", "concat_conv", "gate"
        dropout (float): dropout 比例
    """

    def __init__(
        self,
        dim_region=768,
        dim_global=768,
        dim_out=768,
        fusion_mode="concat_mlp",
        dropout=0.0,
    ):
        super().__init__()
        self.dim_region = dim_region
        self.dim_global = dim_global
        self.dim_out = dim_out
        self.fusion_mode = fusion_mode

        if fusion_mode == "concat_mlp":
            self.fusion = nn.Sequential(
                nn.Linear(dim_region + dim_global, dim_out * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_out * 2, dim_out),
            )
        elif fusion_mode == "concat_conv":
            self.fusion = nn.Sequential(
                nn.Conv2d(dim_region + dim_global, dim_out * 2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv2d(dim_out * 2, dim_out, kernel_size=3, padding=1),
            )
        elif fusion_mode == "gate":
            self.gate = nn.Sequential(
                nn.Linear(dim_region + dim_global, dim_out),
                nn.Sigmoid(),
            )
            self.global_proj = nn.Linear(dim_global, dim_out)
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

        self.norm = nn.LayerNorm(dim_out)

    def forward(self, region_feat, global_feat):
        """
        Args:
            region_feat: (Batch, H, W, dim_region)
            global_feat: (Batch, H, W, dim_global)

        Returns:
            fused_feat: (Batch, H, W, dim_out)
        """
        if self.fusion_mode == "concat_mlp":
            x = torch.cat([region_feat, global_feat], dim=-1)
            x = self.fusion(x)
            x = self.norm(x)
            return x

        elif self.fusion_mode == "concat_conv":
            B, H, W, C = region_feat.shape
            x = torch.cat([region_feat, global_feat], dim=-1)
            x = x.permute(0, 3, 1, 2)
            x = self.fusion(x)
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            return x

        elif self.fusion_mode == "gate":
            gate = self.gate(torch.cat([region_feat, global_feat], dim=-1))
            global_proj = self.global_proj(global_feat)
            x = gate * region_feat + (1 - gate) * global_proj
            x = self.norm(x)
            return x


class GlobalFeatureAligner(nn.Module):
    """
    全球特征对齐模块。

    支持两种对齐方式：
    - crop: 裁切全球特征到区域范围
    - interpolate: 插值全球特征到区域分辨率

    Args:
        global_grid_size (tuple): 全球 patch grid 尺寸 (H_global, W_global)
        global_img_size (tuple): 全球图像尺寸 (H_img, W_img)
        align_mode (str): 对齐模式，可选 "crop", "interpolate"
    """

    def __init__(
        self,
        global_grid_size=(90, 180),
        global_img_size=(720, 1440),
        align_mode="crop",
    ):
        super().__init__()
        self.global_grid_size = global_grid_size
        self.global_img_size = global_img_size
        self.align_mode = align_mode

    def forward(self, global_feat, region_lat_range, region_lon_range, region_grid_size):
        """
        Args:
            global_feat: (Batch, H_global, W_global, dim)
            region_lat_range: (lat_min, lat_max) 纬度范围 [-90, 90]
            region_lon_range: (lon_min, lon_max) 经度范围 [-180, 180]
            region_grid_size: (H_region, W_region) 区域 patch grid 尺寸

        Returns:
            aligned_feat: (Batch, H_region, W_region, dim)
        """
        B, H_global, W_global, dim = global_feat.shape

        if self.align_mode == "crop":
            lat_min, lat_max = region_lat_range
            lon_min, lon_max = region_lon_range

            lat_start = int((lat_min + 90) / 180 * H_global)
            lat_end = int((lat_max + 90) / 180 * H_global)
            lon_start = int((lon_min + 180) / 360 * W_global)
            lon_end = int((lon_max + 180) / 360 * W_global)

            aligned_feat = global_feat[:, lat_start:lat_end, lon_start:lon_end, :]

            H_crop, W_crop = aligned_feat.shape[1], aligned_feat.shape[2]
            H_region, W_region = region_grid_size

            if H_crop != H_region or W_crop != W_region:
                aligned_feat = aligned_feat.permute(0, 3, 1, 2)
                aligned_feat = F.interpolate(
                    aligned_feat,
                    size=region_grid_size,
                    mode="bilinear",
                    align_corners=False,
                )
                aligned_feat = aligned_feat.permute(0, 2, 3, 1)

            return aligned_feat

        elif self.align_mode == "interpolate":
            aligned_feat = global_feat.permute(0, 3, 1, 2)
            aligned_feat = F.interpolate(
                aligned_feat,
                size=region_grid_size,
                mode="bilinear",
                align_corners=False,
            )
            aligned_feat = aligned_feat.permute(0, 2, 3, 1)
            return aligned_feat

        else:
            raise ValueError(f"Unknown align_mode: {self.align_mode}")
