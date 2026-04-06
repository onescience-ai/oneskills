component:

  meta:
    name: GroupEquivariantConvolutions
    alias: GroupConv
    version: 1.0
    domain: deep_learning
    category: neural_network
    subcategory: equivariant_operator
    author: OneScience
    license: Apache-2.0
    tags:
      - group_equivariance
      - c4
      - d4
      - conv2d
      - conv3d


  concept:

    description: >
      GroupEquivariantConv2d/3d 实现了对离散旋转群（C4）及可选反射群（D4）的等变卷积。
      模块通过对基础卷积核做旋转、反射与群维度重排来构造完整群卷积核，
      在 first_layer（lifting）、middle layer（group->group）和 last_layer（projection）三种模式下
      分别组织权重与偏置形状，以保持群作用下表征的一致性。

    intuition: >
      普通卷积对平移等变，但对旋转/镜像不天然稳定。
      该模块把“旋转与反射”显式编码为群元素，并让权重在群轨道上共享，
      从而让特征对群变换具有可控响应。

    problem_it_solves:
      - 在 2D/3D 场景中引入旋转与反射等变性
      - 用参数共享减少对每个方向单独学习的需求
      - 统一 lifting/group/projection 三类群卷积阶段
      - 支持 Hermitian 频谱权重构造以约束复数对称性（2D）


  theory:

    formula:

      group_conv:
        expression: |
          (f *_{G} \psi)(g, x) = \sum_{y} f(y)\,\psi(g^{-1}(x-y))

      lifted_conv:
        expression: |
          f: \mathbb{Z}^d \to \mathbb{R}^{C_{in}}
          \Rightarrow \hat{f}: G \times \mathbb{Z}^d \to \mathbb{R}^{C_{out}}

      weight_orbit:
        expression: |
          W_g = T_g(W_0),\ g \in G

    variables:

      G:
        name: DiscreteGroup
        description: C4 或 D4 群

      W_0:
        name: BaseKernel
        description: 基础卷积核参数

      T_g:
        name: GroupActionOnKernel
        description: 群元素 g 对卷积核的旋转/反射变换

      f:
        name: InputFeature
        description: 输入特征场


  structure:

    architecture: group_equivariant_convolution_family

    pipeline:

      - name: BaseWeightInit
        operation: kaiming_uniform

      - name: GroupWeightGeneration
        operation: rotate_reflect_permute_weight_orbits

      - name: BiasExpansion
        operation: repeat_or_flatten_bias_by_group_mode

      - name: Convolution
        operation: conv2d_or_conv3d_with_generated_weights


  interface:

    parameters:

      in_channels:
        type: int
        description: 输入通道数

      out_channels:
        type: int
        description: 输出通道数

      kernel_size:
        type: int|tuple
        description: 卷积核尺寸（奇数）

      bias:
        type: bool
        description: 是否使用偏置

      first_layer:
        type: bool
        description: 是否为 lifting 层

      last_layer:
        type: bool
        description: 是否为 projection 层

      reflection:
        type: bool
        description: 是否启用反射（D4）

      spectral:
        type: bool
        description: 2D 权重是否使用复数频谱参数

      Hermitian:
        type: bool
        description: 2D 频谱权重是否强制 Hermitian 对称

    inputs:

      x2d:
        type: GroupFeature2D
        shape: [batch, channels*(|G| or 1), H, W]
        dtype: float32|complex64
        description: 2D 输入特征

      x3d:
        type: GroupFeature3D
        shape: [batch, channels*(|G| or 1), D, H, W]
        dtype: float32
        description: 3D 输入特征

    outputs:

      y2d:
        type: GroupFeature2D
        description: 2D 群卷积输出

      y3d:
        type: GroupFeature3D
        description: 3D 群卷积输出


  types:

    GroupFeature2D:
      shape: [batch, channels_grouped, H, W]
      description: 含群维编码的 2D 特征

    GroupFeature3D:
      shape: [batch, channels_grouped, D, H, W]
      description: 含群维编码的 3D 特征


  implementation:

    framework: pytorch

    code: |

      class GroupEquivariantConv2d(nn.Module):
          def __init__(self, in_channels, out_channels, kernel_size, bias=True, first_layer=False, last_layer=False, spectral=False, Hermitian=False, reflection=False):
              super().__init__()
              ...
              self.get_weight()

          def get_weight(self):
              # 通过旋转/反射和群维度重排构造完整卷积核
              ...
              return self.weights

          def forward(self, x):
              self.get_weight()
              return F.conv2d(input=x, weight=self.weights, bias=self.bias)

      class GroupEquivariantConv3d(nn.Module):
          def __init__(self, in_channels, out_channels, kernel_size, bias=True, first_layer=False, last_layer=False, reflection=False):
              super().__init__()
              ...

          def get_weight(self):
              ...
              return self.weights

          def forward(self, x):
              self.get_weight()
              return F.conv3d(input=x, weight=self.weights, bias=self.bias)


  skills:

    build_group_equivariant_conv:

      description: 构建支持 C4/D4 的 2D/3D 群等变卷积层

      inputs:
        - in_channels
        - out_channels
        - kernel_size
        - first_layer
        - last_layer
        - reflection

      prompt_template: |

        构建 GroupEquivariantConv 模块。

        参数：
        in_channels = {{in_channels}}
        out_channels = {{out_channels}}
        kernel_size = {{kernel_size}}
        first_layer = {{first_layer}}
        last_layer = {{last_layer}}
        reflection = {{reflection}}

        要求：
        通过群变换生成卷积核轨道并执行 group-aware 卷积。


    diagnose_equivariance:

      description: 分析群等变卷积中的等变性与实现错误

      checks:
        - even_kernel_size_violation
        - wrong_group_permutation_order
        - inconsistent_first_last_layer_channel_layout
        - hermitian_weight_reconstruction_error


  knowledge:

    usage_patterns:

      lifting_group_projection:

        pipeline:
          - LiftingConv
          - GroupConvBlocks
          - ProjectionConv

      symmetry_aware_modeling:

        pipeline:
          - EncodeSymmetryGroup
          - SharedOrbitWeights
          - EquivariantInference


    hot_models:

      - model: G-CNN
        year: 2016
        role: 群卷积网络奠基工作
        architecture: group_equivariant_cnn

      - model: E(2)-CNN Family
        year: 2019
        role: 系统化二维等变卷积构造
        architecture: steerable_equivariant_network


    best_practices:

      - 保持 kernel_size 为奇数以保证中心对齐。
      - 明确区分 first_layer/middle/last_layer 的通道组织方式。
      - 引入反射群前先验证任务是否具有镜像对称先验。


    anti_patterns:

      - 在群维度置换中误用索引导致等变性破坏。
      - 在 3D 场景中错误旋转深度维度。


    paper_references:

      - title: "Group Equivariant Convolutional Networks"
        authors: Cohen and Welling
        year: 2016

      - title: "A General Theory of Equivariant CNNs on Homogeneous Spaces"
        authors: Cohen et al.
        year: 2019


  graph:

    is_a:
      - NeuralNetworkComponent
      - EquivariantOperator

    part_of:
      - EquivariantNetworks
      - SymmetryAwareModels

    depends_on:
      - GroupActions
      - WeightOrbitGeneration
      - Conv2d
      - Conv3d

    variants:
      - GroupEquivariantConv2d
      - GroupEquivariantConv3d

    used_in_models:
      - G-CNN
      - E(2)-Equivariant CNN Variants
      - Symmetry-aware Forecasting Models

    compatible_with:

      inputs:
        - GroupFeature2D
        - GroupFeature3D

      outputs:
        - GroupFeature2D
        - GroupFeature3D