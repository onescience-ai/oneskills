component:

  meta:
    name: UtilityFunctions
    alias: utils
    version: 1.0
    domain: deep_learning
    category: utility
    subcategory: spatial_transform_and_normalization
    author: OneScience
    license: Apache-2.0
    tags:
      - grid_sample
      - optical_flow
      - image_warping
      - spectral_normalization
      - power_iteration

  concept:

    description: >
      该模块提供了一组深度学习中常用的底层辅助函数与类。
      其中包括用于空间变换（Spatial Transformation）的特征重采样工具 `make_grid` 和 `warp`，它们基于光流（Optical Flow）或运动场对特征图进行可微的形变操作（Warping）；
      此外，还提供了一个自定义的 `spectral_norm` 模块包装器，用于通过幂迭代法（Power Iteration）计算权重矩阵的最大奇异值，从而对模块权重进行谱归一化，以约束网络的 Lipschitz 常数。

    intuition: >
      `warp`（形变操作）就像是在一张具有弹性的橡胶网格图像上，根据给定的二维向量场（Flow）推拉每一个像素，从而生成形变后的新图像。
      `spectral_norm`（谱归一化）则像是一个“信号放大限制器”，它防止某一层网络将输入信号成倍无限放大，这在训练极度不稳定的生成对抗网络（GAN）或确保物理替代模型稳定性时极为关键。

    problem_it_solves:
      - 实现可微的图像或特征图的空间重采样与形变（常用于光流估计、视频帧插值、物理场平移）
      - 将绝对坐标系与预测的相对偏移量（Flow）对齐，并归一化到 PyTorch `grid_sample` 所需的 [-1, 1] 区间
      - 通过谱归一化稳定神经网络的训练过程，防止梯度爆炸和模式崩溃（Mode Collapse）

  theory:

    formula:

      warp_grid_normalization:
        expression: vgrid_{x,y} = \frac{grid_{x,y} + flow_{x,y}}{\max(W-1, 1)} \times 2.0 - 1.0

      power_iteration:
        expression: v_{t+1} = \frac{W^T u_t}{\|W^T u_t\|_2}, \quad u_{t+1} = \frac{W v_{t+1}}{\|W v_{t+1}\|_2}

      spectral_normalization:
        expression: \sigma = u^T W v, \quad \tilde{W} = \frac{W}{\sigma}

    variables:

      grid:
        name: CoordinateGrid
        description: 图像的绝对坐标网格矩阵

      flow:
        name: OpticalFlow
        description: 预测的相对像素偏移向量场

      W:
        name: WeightMatrix
        description: 神经网络层的权重张量（通常展平为 2D 矩阵参与计算）

      u:
        name: LeftSingularVector
        description: 通过幂迭代逼近的左奇异向量

      v:
        name: RightSingularVector
        description: 通过幂迭代逼近的右奇异向量

      \sigma:
        name: SpectralNorm
        description: 权重的谱范数（即最大奇异值）

  structure:

    architecture: functional_utilities_and_wrappers

    pipeline:

      - name: CoordinateGeneration
        operation: create_meshgrid (make_grid)

      - name: SpatialWarping
        operation: add_flow_normalize_and_sample (warp)

      - name: WeightNormalization
        operation: power_iteration_and_scaling (spectral_norm)

  interface:

    parameters:

      mode:
        type: str
        default: 'bilinear'
        description: `warp` 中使用的插值模式，支持 'bilinear', 'nearest' 等

      padding_mode:
        type: str
        default: 'zeros'
        description: `warp` 中越界像素的填充模式，支持 'zeros', 'border', 'reflection'

      module:
        type: nn.Module
        description: `spectral_norm` 要包装的目标网络层（如 nn.Linear, nn.Conv2d）

      name:
        type: str
        default: 'weight'
        description: 要进行谱归一化的参数名称

      power_iterations:
        type: int
        default: 1
        description: 每次前向传播时幂迭代法更新奇异向量的次数

    inputs:

      input:
        type: Tensor
        shape: [B, C, H, W]
        dtype: float32
        description: 待形变特征图，或生成网格时的参考输入张量

      flow:
        type: Tensor
        shape: [B, 2, H, W]
        dtype: float32
        description: 形变偏移量向量场

    outputs:

      output:
        type: Tensor
        description: 形变后的张量，或谱归一化目标层前向传播的结果

  types:

    Tensor:
      shape: dynamic
      description: 任意维度的 PyTorch 张量

  implementation:

    framework: pytorch

    code: |
      import torch.nn.functional as F
      import torch
      import torch.nn as nn


      def make_grid(input):
          B, C, H, W = input.size()
          device = 'cuda' if torch.cuda.is_available() else 'cpu'
          xx = torch.arange(0, W).view(1, -1).repeat(H, 1).to(device)
          yy = torch.arange(0, H).view(-1, 1).repeat(1, W).to(device)
          xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
          yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
          grid = torch.cat((xx, yy), 1).float()

          return grid

      def warp(input, flow, grid, mode="bilinear", padding_mode="zeros"):

          B, C, H, W = input.size()
          vgrid = grid + flow

          vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
          vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
          vgrid = vgrid.permute(0, 2, 3, 1)
          output = torch.nn.functional.grid_sample(input, vgrid, padding_mode=padding_mode, mode=mode, align_corners=True)
          return output

      def l2normalize(v, eps=1e-12):
          return v / (v.norm() + eps)


      class spectral_norm(nn.Module):
          def __init__(self, module, name='weight', power_iterations=1):
              super(spectral_norm, self).__init__()
              self.module = module
              self.name = name
              self.power_iterations = power_iterations
              if not self._made_params():
                  self._make_params()

          def _update_u_v(self):
              u = getattr(self.module, self.name + "_u")
              v = getattr(self.module, self.name + "_v")
              w = getattr(self.module, self.name + "_bar")

              height = w.data.shape[0]
              for _ in range(self.power_iterations):
                  v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
                  u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

              sigma = u.dot(w.view(height, -1).mv(v))
              setattr(self.module, self.name, w / sigma.expand_as(w))

          def _made_params(self):
              try:
                  u = getattr(self.module, self.name + "_u")
                  v = getattr(self.module, self.name + "_v")
                  w = getattr(self.module, self.name + "_bar")
                  return True
              except AttributeError:
                  return False

          def _make_params(self):
              w = getattr(self.module, self.name)

              height = w.data.shape[0]
              width = w.view(height, -1).data.shape[1]

              u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
              v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
              u.data = l2normalize(u.data)
              v.data = l2normalize(v.data)
              w_bar = nn.Parameter(w.data)

              del self.module._parameters[self.name]

              self.module.register_parameter(self.name + "_u", u)
              self.module.register_parameter(self.name + "_v", v)
              self.module.register_parameter(self.name + "_bar", w_bar)


          def forward(self, *args):
              self._update_u_v()
              return self.module.forward(*args)

  skills:

    apply_warp_transform:

      description: 使用网格和偏移量对输入特征图进行可微的重采样形变

      inputs:
        - input_tensor
        - predicted_flow_field

      prompt_template: |
        调用 make_grid 生成基础坐标网格，
        然后利用 warp(input_tensor, predicted_flow_field, grid) 得到形变后的输出。

    apply_spectral_norm:

      description: 在网络层外层包装谱归一化，以稳定高方差的对抗训练或强化学习过程

      inputs:
        - target_module (如 nn.Conv2d)

      prompt_template: |
        为模块 {{target_module}} 添加谱归一化包装：
        layer = spectral_norm({{target_module}})

  knowledge:

    usage_patterns:

      optical_flow_estimation:

        pipeline:
          - Feature Extractor
          - Predict Flow
          - Warp (将过去的帧根据预测的 Flow 对齐到当前帧进行误差比对)

      stable_gan_discriminator:

        pipeline:
          - spectral_norm(Conv2d)
          - LeakyReLU
          - spectral_norm(Linear)

    design_patterns:

      module_decorator:

        structure:
          - `spectral_norm` 作为装饰器模式（Wrapper），不改变内部模块的原始逻辑结构，只在每次 `forward` 调用前拦截并更新（缩放）其内部参数字典中的 `weight`。

    hot_models:

      - model: Spatial Transformer Networks (STN)
        year: 2015
        role: 提出在网络内部进行可微空间变换的概念
        architecture: grid_generator_and_sampler

      - model: Spectral Normalization GAN (SNGAN)
        year: 2018
        role: 稳定生成对抗网络训练的里程碑工作
        architecture: gan_discriminator_with_sn

    model_usage_details:

      Physics Surrogates:

        usage: 在基于 AI 的流体力学（如平流方程）预测中，`warp` 函数可用于实现半拉格朗日格式（Semi-Lagrangian scheme），根据预测的速度场精确平移流体特征。

    best_practices:

      - `F.grid_sample` 严格要求输入的坐标网格尺寸归一化到 `[-1, 1]`。当前代码的 `vgrid[:, 0, :, :] = 2.0 * ... / max(W - 1, 1) - 1.0` 精确处理了这种归一化转换，且使用了 `align_corners=True` 保证角点对齐。
      - 当输入的尺寸 `(H, W)` 在整个训练和验证过程中保持不变时，应考虑将 `make_grid` 的输出缓存（Cache），避免在每次前向传播时重复生成，造成资源浪费。

    anti_patterns:

      - `make_grid` 函数中硬编码了 `device = 'cuda' if torch.cuda.is_available() else 'cpu'`。这会导致多 GPU 并行（DataParallel/DDP）以及特定设备指定（如 cuda:1）时的张量不在同一设备的崩溃问题。更优雅的做法是使用 `input.device` 获取上下文设备环境。
      - （注意）当前代码提供了自定义的 `spectral_norm`，但在较新的 PyTorch 版本中，应首选官方高度优化的内置函数 `torch.nn.utils.spectral_norm`。

    paper_references:

      - title: "Spatial Transformer Networks"
        authors: Jaderberg et al.
        year: 2015

      - title: "Spectral Normalization for Generative Adversarial Networks"
        authors: Miyato et al.
        year: 2018

  graph:

    is_a:
      - HelperFunctions
      - TransformationModule
      - WeightNormalization

    part_of:
      - FlowNet
      - VoxelMorph
      - SNGAN
      - PhysicsInformedModels

    depends_on:
      - torch.nn.functional.grid_sample
      - torch.mv
      - torch.t

    variants:
      - torch.nn.utils.spectral_norm (PyTorch官方内置实现)
      - affine_grid (处理仿射变换而不是密集光流场)

    used_in_models:
      - 流场形变与预测模型
      - GAN 鉴别器 (Discriminator)

    compatible_with:

      inputs:
        - Tensor

      outputs:
        - Tensor