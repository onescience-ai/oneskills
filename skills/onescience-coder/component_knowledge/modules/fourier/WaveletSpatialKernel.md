component:

  meta:
    name: WaveletSpatialKernel
    alias: MultiwaveletSpatialSparseKernel
    version: 1.0
    domain: physics_informed_machine_learning
    category: neural_network
    subcategory: spatial_operator
    author: OneScience
    tags:
      - multi-wavelet
      - spatial_convolution
      - high_frequency_capture
      - sparse_kernel

  concept:

    description: >
      WaveletSpatialKernel 构成了多小波变换系统中的微观/物理维度捕获算子。
      多小波傅里叶核（WaveletFourierKernel）在频域中强行截断了高频信号以获取全局低频依赖；
      而真实偏微分系统的解（如湍流边缘、边界层剧烈变化）在物理空间上富含高频的局部拓扑特征。
      WaveletSpatialKernel 直接在物理网格空间上基于标准的多维卷积（Conv2D/Conv3D），针对
      多小波展开后的隐空间局部系数进行处理，弥补傅里叶操作丢失的局部突变细节。

    intuition: >
      在一个预测气流或流体涡旋的模型中，大尺度的流动交由傅里叶系统处理；而涡流破碎或边界碰撞产生的局部细节，
      则交由局部核（SpatialKernel，3x3 的标准卷积）在微观区域内进行模式识别。两者（全局频域 + 局部空域）
      的特征交织，才得以实现 $2^N$ 层次的多小波完备表达系统。

    problem_it_solves:
      - 弥补傅里叶高频截断所导致的高频、局部特征丢失问题（Ring Artifacts）。
      - 为多小波变换系统提供纯正物理空间的平移一致性微约束。
      - 为复杂的、物理边界存在尖锐梯度的现象（如激波）提供足够的局部感受野进行非线性拟合。

  theory:

    formula:

      spatial_feature_extraction:
        expression: |
          Y_{conv} = \sigma (W_{conv} \ast X_{\text{permuted}} + b_{conv})
          Y_{final} = W_{linear} \cdot Y_{conv}

    variables:

      X_{\text{permuted}}:
        name: ReshapedWaveletCoefficients
        shape: [batch, c \times k^2, Nx, Ny]
        description: 为适应标准空间卷积维度要求，将通道和正交基合并作为通道维度后的输入张量

      W_{conv}:
        name: ConvolutionWeights
        shape: [out\_channels, in\_channels, 3, 3]
        description: 用于提取局部相互依存关系的卷积核，默认内核大小为 3x3（或 3x3x3）以保持紧凑的局部感受野

      \sigma:
        name: Activation
        description: 能够阻断无物理意义负相位的非线性激活函数，通常采用 ReLU

      W_{linear}:
        name: ProjectionWeights
        description: 用于特征维度重整的线性映射矩阵，将卷积中间的扩张通道降维匹配原有基张量大小

  structure:

    architecture: SpatialSparseKernel

    pipeline:
      - name: ChannelPermutation
        operation: "[B, Nx, Ny, c*k^2] -> [B, c*k^2, Nx, Ny]"
      - name: ConvolutionBlock
        operation: Conv2d(3x3) / Conv3d(3x3x3) + ReLU
      - name: SpatialRestoration
        operation: "[B, c*k^2, Nx, Ny] -> [B, Nx, Ny, c*k^2]"
      - name: LinearProjection
        operation: Linear(alpha * k^2, c * k^2)

  interface:

    parameters:

      k:
        type: int
        description: 多小波基参数/网格分辨率层级，控制输出表示的最末维度数量

      alpha:
        type: int
        description: 空间卷积扩维系数倍率，决定中间特征通道数

      c:
        type: int
        default: 1
        description: 基础通道因子

    inputs:
      x:
        type: Tensor
        shape: [batch, spatial_dims..., c, k^2]
        description: 多小波投影特征系数

    outputs:
      out:
        type: Tensor
        shape: [batch, spatial_dims..., c, k^2]
        description: 并行结合了局部空间细节修正的同构张量

  types:
    Tensor:
      description: PyTorch Tensor 向量

  implementation:

    framework: pytorch

    code: |
      import torch
      import torch.nn as nn
      
      class WaveletSpatialKernel2D(nn.Module):
          """
          二维空间稀疏核层 (2D Spatial Sparse Kernel)。
          (原 sparseKernel2d / sparseKernel)
      
          该模块在物理空间中使用标准的二维卷积（Conv2d）来处理多小波系数。
          它先通过卷积层（包含 ReLU 激活）提取特征，然后通过线性层进行特征投影。
          常用于多小波变换中处理高频细节系数，捕捉局部物理特征。
      
          Args:
              k (int): 多小波块大小参数。输入特征的最后一个维度应为 k^2。
              alpha (int): 控制卷积层输出通道数的倍率因子。
              c (int, optional): 通道缩放因子。默认值: 1。
              nl (int, optional): 保留参数。默认值: 1。
              initializer (callable, optional): 初始化函数（保留接口）。
      
          形状:
              输入 x: (B, N_x, N_y, c, k^2)。
              输出: (B, N_x, N_y, c, k^2)。
      
          Example:
              >>> layer = WaveletSpatialKernel2D(k=3, alpha=4, c=1)
              >>> x = torch.randn(4, 64, 64, 1, 9)
              >>> out = layer(x)
              >>> print(out.shape)
              torch.Size([4, 64, 64, 1, 9])
          """
          def __init__(self, k, alpha, c=1, nl=1, initializer=None, **kwargs):
              super().__init__()
              self.k = k
              self.conv = self.convBlock(k, c * k ** 2, alpha)
              self.Lo = nn.Linear(alpha * k ** 2, c * k ** 2)
      
          def forward(self, x):
              B, Nx, Ny, c, ich = x.shape  
              x = x.view(B, Nx, Ny, -1).permute(0, 3, 1, 2)
              x = self.conv(x)
              x = x.permute(0, 2, 3, 1)
              x = self.Lo(x)
              x = x.view(B, Nx, Ny, c, ich)
              return x
      
          def convBlock(self, k, W, alpha):
              och = alpha * k ** 2
              net = nn.Sequential(
                  nn.Conv2d(W, och, 3, 1, 1),
                  nn.ReLU(inplace=True),
              )
              return net
      
      
      class WaveletSpatialKernel3D(nn.Module):
          """
          三维空间稀疏核层 (3D Spatial Sparse Kernel)。
          (原 sparseKernel3d)
      
          类似于 WaveletSpatialKernel2D，但在三维物理空间上使用 Conv3d 进行局部特征提取。
          适用于处理 3D 体数据中的局部高频特征，作为多小波变换处理三维细节系数的核心组件。
      
          Args:
              k (int): 多小波参数。输入最后维度应为 k^2。
              alpha (int): 通道倍率因子。
              c (int, optional): 通道因子。默认值: 1。
              nl (int, optional): 保留参数。默认值: 1。
              initializer (callable, optional): 初始化函数。
      
          形状:
              输入 x: (B, N_x, N_y, T, c, k^2)。通常用于 2D 空间 + 1D 时间，或者 3D 空间。
              输出: (B, N_x, N_y, T, c, k^2)。
      
          Example:
              >>> layer = WaveletSpatialKernel3D(k=3, alpha=4, c=1)
              >>> x = torch.randn(2, 32, 32, 10, 1, 9)
              >>> out = layer(x)
              >>> print(out.shape)
              torch.Size([2, 32, 32, 10, 1, 9])
          """
          def __init__(self, k, alpha, c=1, nl=1, initializer=None, **kwargs):
              super().__init__()
              self.k = k
              self.conv = self.convBlock(alpha * k ** 2, alpha * k ** 2)
              self.Lo = nn.Linear(alpha * k ** 2, c * k ** 2)
      
          def forward(self, x):
              B, Nx, Ny, T, c, ich = x.shape  
              x = x.view(B, Nx, Ny, T, -1).permute(0, 4, 1, 2, 3)
              x = self.conv(x)
              x = x.permute(0, 2, 3, 4, 1)
              x = self.Lo(x)
              x = x.view(B, Nx, Ny, T, c, ich)
              return x
      
          def convBlock(self, ich, och):
              net = nn.Sequential(
                  nn.Conv3d(och, och, 3, 1, 1),
                  nn.ReLU(inplace=True),
              )
              return net

  skills:

    build_wavelet_spatial_kernel:
      description: 构建用于提取由于傅里叶截断而丢失的高频空间特征算子
      inputs:
        - k
        - alpha
        - c
      prompt_template: |
        构建多小波空间稀疏核。
        参数：k={{k}}, alpha={{alpha}}, c={{c}}
        要求：利用 Conv2d/Conv3d 仅在局部感受野内聚合信息，注意权重的缩放与原位激活。
        
    diagnose_spatial_kernel:
      description: 分辨和排查多小波空间核常见的感受野或维度错误
      checks:
        - channel_expansion_mismatch
        - inplace_activation_gradient_error
        - spatial_resolution_loss

  knowledge:

    usage_patterns:
      dual_path_processing:
        pipeline:
          - (Input) -> Branch1: WaveletFourierKernel (低频全局)
          - (Input) -> Branch2: WaveletSpatialKernel (高频局部)
          - (Merge) -> Add/Concat

    hot_models:
      - model: MWCNN (Multi-level Wavelet-CNN)
        year: 2018
        role: 首次将小波变换与 CNN 结合处理图像反问题，提供了空间卷积恢复高频特征的先例
        architecture: U-Net with Wavelet Transform
      - model: FNO
        year: 2020
        role: 原生依靠傅里叶算子，但在后续变体中引入局部残差连接来补偿高空域特征
        architecture: Neural Operator

    best_practices:
      - Padding需固定为1（针对3x3 kernel），Stride固定为1，以严格对齐张量分辨率尺度，防止出现多小波反变换时网格不对齐。
      - 原地（In-place）激活函数 
n.ReLU(inplace=True) 可显著减少大尺寸 3D/2D 流体数据上的 GPU 显存峰值开销。

    anti_patterns:
      - 试图使用大尺寸卷积核（如 7x7 或 9x9）。因为全局交互任务已被并行的傅里叶谱模块有效处理，空间侧使用过大卷积只会带来参数冗余和显卡吞吐下降，失去Sparse的本意。

    paper_references:
      - title: "Multi-level Wavelet-CNN for Image Restoration"
        authors: Liu et al.
        year: 2018
      - title: "Wavelet Neural Operator for solving parametric partial differential equations"
        authors: Tripura et al.
        year: 2022


  graph:
    is_a:
      - ConvolutionalKernel
      - SpatialOperator
    part_of:
      - MultiWaveletTransform
    depends_on:
      - nn.Conv2d
      - nn.Conv3d
      - nn.ReLU
      - nn.Linear
    compatible_with:
      - WaveletFourierKernel