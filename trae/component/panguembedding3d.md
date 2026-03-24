component:
  meta:
    name: PanguEmbedding3D
    中文名称: 盘古三维面片嵌入层
    别名: 3D Patch Embedding, 三维体嵌入
    version: 1.0
    领域: 深度学习/气象AI
    分类: 神经网络组件
    子类: 嵌入层/编码器
    作者: OneScience
    tags:
      - 气象AI
      - 3D建模
      - Patch Embedding
      - Pangu-Weather

  concept:
    描述: >
      将三维大气场数据（气压层×经度×纬度×变量）分割为不重叠的三维patch，
      并通过卷积映射到高维潜在空间，为后续3D Transformer提供统一的token表示。
    直觉理解: >
      就像把一个立方体蛋糕切成小块，每小块代表局部的大气状态。
      通过神经网络将每小块压缩成一个"特征向量"，
      这样既保留了空间结构信息，又大幅降低了计算量。
    解决的问题:
      - 高分辨率全球场计算复杂度: 直接在原始分辨率(13×1440×721)上计算注意力代价极高，通过patch降采样将token数量减少至可处理规模
      - 三维体统一表示: 将上空多层变量与地表变量统一嵌入到相同维度空间，便于在单一网络中联合建模垂直-水平耦合关系
      - 局部特征提取: 通过卷积核捕获patch内的局部空间模式，为全局建模提供更抽象的特征表示

  theory:
    核心公式:
      三维卷积嵌入:
        表达式: "Y = Conv3D(Pad(X), kernel=patch_size, stride=patch_size)"
        变量说明:
          X:
            名称: 输入三维场
            形状: "[B, C_in, P, H, W]"
            描述: "批次×输入通道×气压层×纬度×经度，表示原始大气变量场"
          Pad(X):
            名称: 零填充后的输入
            形状: "[B, C_in, P', H', W']"
            描述: "对不能被patch_size整除的维度进行零填充，确保完整分割"
          Y:
            名称: 嵌入后的特征
            形状: "[B, C_embed, P'', H'', W'']"
            描述: "批次×嵌入维度×层patch数×纬度patch数×经度patch数，其中P''=⌈P/patch_p⌉"
          patch_size:
            名称: 三维patch尺寸
            形状: "(patch_p, patch_h, patch_w)"
            描述: "在气压层、纬度、经度三个维度上的patch大小，如(2,4,4)表示2层×4纬度×4经度"

  structure:
    架构类型: 卷积嵌入网络
    计算流程:
      - 计算各维度填充量：对不能被patch_size整除的维度，计算所需的零填充
      - 零填充：在输入张量的6个面（前后、上下、左右）进行对称或非对称填充
      - 3D卷积投影：使用kernel_size=stride=patch_size的卷积将每个patch映射到embed_dim维
      - 可选归一化：若指定norm_layer，对嵌入特征进行归一化处理
    计算流程图: |
      输入 [B,C,P,H,W]
        ↓
      ZeroPad3d (处理不整除情况)
        ↓
      [B,C,P',H',W']
        ↓
      Conv3d(kernel=patch_size, stride=patch_size)
        ↓
      [B,embed_dim,P'',H'',W'']
        ↓
      可选: LayerNorm (permute → norm → permute)
        ↓
      输出 [B,embed_dim,P'',H'',W'']

  interface:
    参数:
      img_size:
        类型: tuple[int, int, int]
        默认值: (13, 721, 1440)
        描述: 输入图像尺寸(气压层数, 纬度格点数, 经度格点数)，对应ERA5的13层×0.25°分辨率全球网格
      patch_size:
        类型: tuple[int, int, int]
        默认值: (2, 4, 4)
        描述: 三维patch尺寸(层方向, 纬度方向, 经度方向)，决定下采样率和局部感受野大小
      in_chans:
        类型: int
        默认值: 5
        描述: 输入通道数，对应大气变量数量(Z/Q/T/U/V五个上空变量)
      embed_dim:
        类型: int
        默认值: 192
        描述: 嵌入后的特征维度，即每个patch token的向量长度
      norm_layer:
        类型: nn.Module
        默认值: null
        描述: 可选的归一化层(如LayerNorm)，用于稳定特征分布
    输入:
      x:
        类型: Tensor
        形状: "[B, C, P, H, W]"
        描述: 批次×通道×气压层×纬度×经度的五维张量，表示三维大气场的多变量数据
    输出:
      embedded:
        类型: Tensor
        形状: "[B, embed_dim, P', H', W']"
        描述: 嵌入后的特征张量，其中P'=⌈P/patch_p⌉, H'=⌈H/patch_h⌉, W'=⌈W/patch_w⌉

  types:
    AtmosphericField3D:
      形状: "[B, C, P, H, W]"
      描述: 三维大气场张量，P为气压层维度(垂直方向)，H为纬度维度，W为经度维度，C为变量通道数
    PatchTokens3D:
      形状: "[B, D, P', H', W']"
      描述: 三维patch token张量，D为嵌入维度，P'/H'/W'为各维度的patch网格尺寸

  constraints:
    shape_constraints:
      - 规则: "P % patch_p 可以不为0，H % patch_h 可以不为0，W % patch_w 可以不为0"
        描述: 通过零填充自动处理不整除情况，确保所有输入尺寸都能正确嵌入
      - 规则: "输出尺寸 P' = ⌈P / patch_p⌉, H' = ⌈H / patch_h⌉, W' = ⌈W / patch_w⌉"
        描述: 输出patch网格尺寸由输入尺寸和patch_size向上取整决定
    parameter_constraints:
      - 规则: "patch_size各维度 > 0"
        描述: patch尺寸必须为正整数
      - 规则: "embed_dim > 0"
        描述: 嵌入维度必须为正整数
    compatibility_rules:
      - 输入类型: AtmosphericField3D
        输出类型: PatchTokens3D
        描述: 将连续的三维大气场离散化为patch token序列，降低空间分辨率但增加特征维度

  implementation:
    框架: pytorch
    示例代码: |
      import torch
      from torch import nn

      class PanguEmbedding3D(nn.Module):
          def __init__(self, img_size=(13, 721, 1440), patch_size=(2, 4, 4),
                       in_chans=5, embed_dim=192, norm_layer=None):
              super().__init__()
              level, height, width = img_size
              l_patch_size, h_patch_size, w_patch_size = patch_size

              # 计算填充
              padding = [0] * 6
              for i, (size, p_size) in enumerate([(level, l_patch_size),
                                                   (height, h_patch_size),
                                                   (width, w_patch_size)]):
                  if size % p_size:
                      pad = p_size - (size % p_size)
                      padding[2*i] = pad // 2
                      padding[2*i+1] = pad - padding[2*i]

              self.pad = nn.ZeroPad3d(tuple(reversed(padding)))
              self.proj = nn.Conv3d(in_chans, embed_dim,
                                    kernel_size=patch_size, stride=patch_size)
              self.norm = norm_layer(embed_dim) if norm_layer else None

          def forward(self, x):
              x = self.pad(x)
              x = self.proj(x)
              if self.norm:
                  x = self.norm(x.permute(0,2,3,4,1)).permute(0,4,1,2,3)
              return x

  usage_examples:
    - 描述: "标准Pangu-Weather上空变量嵌入 (13层×721×1440 → 7×181×360)"
      示例代码: |
        embed = PanguEmbedding3D(
            img_size=(13, 721, 1440),
            patch_size=(2, 4, 4),
            in_chans=5,
            embed_dim=192
        )
        x = torch.randn(2, 5, 13, 721, 1440)  # 2个样本，5变量
        out = embed(x)
        # out.shape → (2, 192, 7, 181, 360)

    - 描述: "地表变量嵌入 (1×721×1440 → 1×181×360)"
      示例代码: |
        embed_surface = PanguEmbedding3D(
            img_size=(1, 721, 1440),
            patch_size=(1, 4, 4),
            in_chans=4,
            embed_dim=192
        )
        x_surface = torch.randn(2, 4, 1, 721, 1440)  # 4个地表变量
        out_surface = embed_surface(x_surface)
        # out_surface.shape → (2, 192, 1, 181, 360)

    - 描述: "带LayerNorm的嵌入"
      示例代码: |
        embed_norm = PanguEmbedding3D(
            img_size=(13, 128, 256),
            patch_size=(2, 4, 4),
            in_chans=5,
            embed_dim=192,
            norm_layer=nn.LayerNorm
        )
        x = torch.randn(4, 5, 13, 128, 256)
        out = embed_norm(x)
        # out.shape → (4, 192, 7, 32, 64)

  knowledge:
    应用说明: >
      PanguEmbedding3D属于3D体建模策略中的嵌入模块。在气象AI建模中，嵌入层负责将原始物理场转换为神经网络可处理的token表示。
      3D体建模策略的核心是将大气状态的垂直维度（气压层）显式建模为空间维度，与经纬度平等对待，从而利用3D注意力同时捕捉垂直与水平邻域的耦合关系。
      该策略解决了2D模型将垂直层当作通道而难以捕捉层间联系的问题，在Pangu-Weather中显著提升了对不同气压层间大气状态关系的建模能力。
    热点模型:
      - 模型: Pangu-Weather
        年份: 2023
        场景: 全球中期天气预报，需要同时建模13个气压层的5个大气变量及4个地表变量
        方案: >
          使用(2,4,4)的3D patch将上空13×1440×721×5体降采样为7×360×181×192的token体，
          地表使用(1,4,4)的patch得到1×360×181×192，两者在层维拼接形成8×360×181×192的统一3D体，
          输入到3D Earth-specific Transformer进行编码-解码
        作用: 作为3DEST主干的输入接口，将高分辨率物理场压缩为可处理的token表示，同时保留三维空间结构
        创新: 首次在气象AI中采用3D patch embedding统一上空与地表变量，为3D窗口注意力提供统一的体表示
      - 模型: FuXi
        年份: 2023
        场景: 全球中期天气预报，采用级联模型结构
        方案: >
          在嵌入阶段使用3D卷积(时间×空间)处理输入，但高度维度仍编码为通道维，
          主干网络使用2D Swin Transformer V2处理空间维度
        作用: 通过Cube Embedding捕获时空局部模式，但未将高度作为显式空间维度
        创新: 结合时间维度的3D卷积嵌入，但垂直建模能力弱于Pangu的完全3D体策略
    最佳实践:
      - patch_size选择应平衡计算效率与信息保留：过大导致局部细节丢失，过小导致token数量过多
      - 对于气压层维度，patch_size通常为1-2，因为垂直层数较少(13层)且层间关系重要
      - 对于经纬度维度，patch_size通常为4-8，与图像领域的ViT类似
      - 零填充策略确保任意输入尺寸都能处理，提升模型泛化性
      - 可选的归一化层有助于稳定深层网络训练，但会增加计算开销
    常见错误:
      - 忘记处理不整除情况导致维度不匹配错误
      - 混淆输入格式(B,C,P,H,W)与其他约定(如B,P,H,W,C)导致维度错误
      - patch_size设置不当导致输出尺寸过小或过大
      - 在气压层维度使用过大的patch_size导致垂直分辨率损失严重
    论文参考:
      - 标题: "Accurate medium-range global weather forecasting with 3D neural networks"
        作者: Kaifeng Bi et al.
        年份: 2023
        摘要: 提出Pangu-Weather模型，首次采用3D Earth-specific Transformer显式建模大气垂直结构，在ERA5数据上全面超越ECMWF-IFS

  graph:
    类型关系:
      - 神经网络层
      - 嵌入层
      - 卷积层
    所属结构:
      - 3D Earth-specific Transformer (3DEST)
      - Pangu-Weather编码器
    依赖组件:
      - nn.Conv3d
      - nn.ZeroPad3d
      - nn.LayerNorm (可选)
    变体组件:
      - PatchEmbed2D (二维面片嵌入)
      - PatchEmbed (标准ViT嵌入)
      - CubeEmbedding (FuXi的时空立方体嵌入)
    使用模型:
      - Pangu-Weather
    类型兼容:
      输入:
        - AtmosphericField3D
      输出:
        - PatchTokens3D
