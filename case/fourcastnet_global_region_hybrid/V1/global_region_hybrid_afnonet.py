# Global + Region Hybrid FourCastNet Model

import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from onescience.utils.fcn.img_utils import PeriodicPad2d
from onescience.models.afno.afnonet import AFNONet, Block, PatchEmbed, Mlp


class GlobalFeatureExtractor(nn.Module):
    """
    全球特征提取器，用于从预训练的FourCastNet模型中提取最后一层特征
    """
    def __init__(self, pretrained_model_path):
        super().__init__()
        # 加载预训练模型
        self.pretrained_model = AFNONet()
        checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        self.pretrained_model.load_state_dict(checkpoint['model_state_dict'])
        
        # 冻结预训练模型的参数
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
    
    def forward(self, global_input):
        """
        提取全球特征
        """
        return self.pretrained_model.forward_features(global_input)


class RegionFeatureFusion(nn.Module):
    """
    区域特征融合模块，用于融合全球特征和局部区域特征
    """
    def __init__(self, embed_dim, global_feature_dim, num_blocks=8):
        super().__init__()
        # 全球特征投影
        self.global_feature_proj = nn.Linear(global_feature_dim, embed_dim)
        
        # 特征融合模块
        self.fusion_block = Block(
            dim=embed_dim,
            mlp_ratio=4.,
            drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            double_skip=True,
            num_blocks=num_blocks,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0
        )
    
    def forward(self, local_features, global_features, region_mask):
        """
        融合局部特征和全球特征
        
        Args:
            local_features: 局部区域特征
            global_features: 全球特征
            region_mask: 区域掩码，用于选择与局部区域对应的全球特征
        """
        # 提取与局部区域对应的全球特征
        global_features = global_features * region_mask
        
        # 投影全球特征到与局部特征相同的维度
        global_features = self.global_feature_proj(global_features)
        
        # 融合特征
        fused_features = local_features + global_features
        fused_features = self.fusion_block(fused_features)
        
        return fused_features


class GlobalRegionHybridAFNONet(nn.Module):
    """
    全球+区域混合FourCastNet模型
    """
    def __init__(
            self,
            params,
            pretrained_model_path,
            img_size=(720, 1440),
            patch_size=(16, 16),
            in_chans=2,
            out_chans=2,
            embed_dim=768,
            depth=12,
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.,
            num_blocks=16,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
        ):
        super().__init__()
        self.params = params
        self.img_size = img_size
        self.patch_size = params.patch_size
        self.in_chans = params.N_in_channels
        self.out_chans = params.N_out_channels
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = params.num_blocks
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # 局部区域特征提取
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.h = img_size[0] // self.patch_size[0]
        self.w = img_size[1] // self.patch_size[1]

        # 前半部分blocks用于提取局部特征
        self.local_blocks = nn.ModuleList([
            Block(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
            num_blocks=self.num_blocks, sparsity_threshold=sparsity_threshold, hard_thresholding_fraction=hard_thresholding_fraction) 
        for i in range(depth // 2)])

        # 全球特征提取器
        self.global_feature_extractor = GlobalFeatureExtractor(pretrained_model_path)

        # 区域特征融合模块
        self.feature_fusion = RegionFeatureFusion(embed_dim, embed_dim, num_blocks=self.num_blocks)

        # 后半部分blocks用于处理融合后的特征
        self.fusion_blocks = nn.ModuleList([
            Block(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
            num_blocks=self.num_blocks, sparsity_threshold=sparsity_threshold, hard_thresholding_fraction=hard_thresholding_fraction) 
        for i in range(depth // 2, depth)])

        self.head = nn.Linear(embed_dim, self.out_chans*self.patch_size[0]*self.patch_size[1], bias=False)

        # 初始化权重
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x, global_input, region_info):
        """
        前向传播特征提取
        
        Args:
            x: 局部区域输入
            global_input: 全球输入
            region_info: 区域信息，包含经纬度范围
        """
        B = x.shape[0]
        
        # 提取局部特征
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        x = x.reshape(B, self.h, self.w, self.embed_dim)
        for blk in self.local_blocks:
            x = blk(x)
        
        # 提取全球特征
        global_features = self.global_feature_extractor(global_input)
        
        # 生成区域掩码
        region_mask = self._generate_region_mask(region_info, global_features.shape[1:3])
        region_mask = region_mask.unsqueeze(0).unsqueeze(-1).repeat(B, 1, 1, 1)
        
        # 融合特征
        x = self.feature_fusion(x, global_features, region_mask)
        
        # 处理融合后的特征
        for blk in self.fusion_blocks:
            x = blk(x)

        return x

    def _generate_region_mask(self, region_info, global_size):
        """
        生成区域掩码，用于选择与局部区域对应的全球特征
        
        Args:
            region_info: 区域信息，包含经纬度范围 (lat_min, lat_max, lon_min, lon_max)
            global_size: 全球特征的大小 (H, W)
        """
        lat_min, lat_max, lon_min, lon_max = region_info
        
        # 计算经纬度对应的索引范围
        H, W = global_size
        lat_start = int((90 - lat_max) / 180 * H)
        lat_end = int((90 - lat_min) / 180 * H)
        lon_start = int((lon_min + 180) / 360 * W)
        lon_end = int((lon_max + 180) / 360 * W)
        
        # 生成掩码
        mask = torch.zeros(global_size)
        mask[lat_start:lat_end, lon_start:lon_end] = 1.0
        
        return mask

    def forward(self, x, global_input, region_info):
        """
        前向传播
        
        Args:
            x: 局部区域输入
            global_input: 全球输入
            region_info: 区域信息，包含经纬度范围
        """
        x = self.forward_features(x, global_input, region_info)
        x = self.head(x)
        x = rearrange(
            x,
            "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            h=self.img_size[0] // self.patch_size[0],
            w=self.img_size[1] // self.patch_size[1],
        )
        return x


if __name__ == "__main__":
    # 测试模型
    import sys
    sys.path.append('/Users/zhao/Desktop/OneScience/dev-earth-function+commit')
    
    from onescience.utils.YParams import YParams
    
    # 创建参数
    config_file = '/Users/zhao/Desktop/OneScience/dev-earth-function+commit/onescience/examples/earth/fourcastnet/conf/config.yaml'
    params = YParams(config_file, 'model')
    params.N_in_channels = 2
    params.N_out_channels = 2
    params.patch_size = (16, 16)
    params.num_blocks = 16
    
    # 假设预训练模型路径
    pretrained_model_path = '/path/to/pretrained/fourcastnet/model.pth'
    
    # 创建模型
    model = GlobalRegionHybridAFNONet(params, pretrained_model_path)
    
    # 测试输入
    local_input = torch.randn(1, 2, 720, 1440)  # 局部区域输入
    global_input = torch.randn(1, 2, 720, 1440)  # 全球输入
    region_info = (30, 60, 100, 130)  # 区域经纬度范围 (lat_min, lat_max, lon_min, lon_max)
    
    # 前向传播
    output = model(local_input, global_input, region_info)
    print(f"Input shape: {local_input.shape}")
    print(f"Output shape: {output.shape}")
