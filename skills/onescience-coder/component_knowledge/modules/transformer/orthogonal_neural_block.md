component:

  meta:
    name: OrthogonalNeuralBlock
    alias: OrthogonalNeuralOperatorBlock
    version: 1.0
    domain: ai_for_science
    category: neural_network
    subcategory: neural_operator
    author: OneScience
    license: Apache-2.0
    tags:
      - orthogonal_neural_operator
      - cholesky_decomposition
      - dual_branch
      - deep_learning
      - over_smoothing


  concept:

    description: >
      正交神经 Transformer 块是 Orthogonal Neural Operator (ONO) 的核心组件。
      它采用独特的双分支架构同步更新：特征分支使用注意力机制提取非线性特征，
      物理分支在特征分支的引导下，通过维护协方差矩阵和 Cholesky 分解进行正交化投影。
      这种机制保证了信号传播过程中的特征独立性。

    intuition: >
      在非常深的网络中，不同维度的特征往往会趋于同质化（也就是大家都变得差不多，失去了细节），这叫“过度平滑”。
      这个模块就像是一个“特征梳理器”：它专门分出一条线（物理分支），
      利用线性代数里的正交化手段（把特征强制分配到互不干扰的正交基底上），
      确保每一层传递下去的物理信息都是独立且清晰的，从而允许网络搭得更深而不丢失物理高频细节。

    problem_it_solves:
      - 深层神经算子（Neural Operators）中普遍存在的过度平滑（Over-smoothing）问题
      - 物理场高频细节在多层前向传播中的衰减和丢失
      - 大规模张量求逆操作中的数值不稳定性（通过带有 jitter 的 psd_safe_cholesky 解决）


  theory:

    formula:

      covariance_update:
        expression: $$ \Sigma_{new} = \alpha \cdot \Sigma_{old} + (1 - \alpha) \cdot \text{Cov}(x') $$

      cholesky_orthogonalization:
        expression: |
          $$ L L^T = \Sigma $$
          $$ x_{ortho} = x' L^{-T} $$

      physics_branch_projection:
        expression: $$ fx_{out} = (x_{ortho} \cdot \text{softplus}(\mu)) (x_{ortho}^T fx_{in}) + fx_{in} $$

    variables:

      x:
        name: FeatureBranch
        description: 用于计算注意力并指导正交投影的辅助特征分支

      fx:
        name: PhysicsBranch
        description: 承载核心物理量的分支，通过正交投影进行更新

      \Sigma:
        name: FeatureCovariance
        description: 动量更新的特征协方差矩阵 (feature_cov)

      L:
        name: CholeskyFactor
        description: 协方差矩阵的下三角 Cholesky 分解结果


  structure:

    architecture: dual_branch_orthogonal_network

    pipeline:

      - name: FeatureAttention
        operation: nystrom / linear / self_attention (更新 x)

      - name: FeatureMLP
        operation: one_mlp (更新 x)

      - name: FeatureProjection
        operation: linear_projection (提取用于计算协方差的特征 x')

      - name: CovarianceEstimation
        operation: exponential_moving_average (维护 feature_cov)

      - name: OrthogonalBasisGeneration
        operation: psd_safe_cholesky + inverse (生成正交基)

      - name: PhysicsOrthogonalProjection
        operation: matrix_multiplication + residual_add (更新 fx)

      - name: PhysicsMLP
        operation: one_mlp_or_linear (最终处理 fx)


  interface:

    parameters:

      num_heads:
        type: int
        description: 特征分支注意力机制的头数

      hidden_dim:
        type: int
        description: 隐藏层的通道维度

      dropout:
        type: float
        description: 注意力机制的 Dropout 概率

      attn_type:
        type: str
        default: "nystrom"
        description: 注意力类型，支持 "nystrom" (高效近似), "linear", "selfAttention"

      mlp_ratio:
        type: int
        default: 4
        description: MLP 的隐藏层扩展倍率

      last_layer:
        type: bool
        default: False
        description: 是否为最后一层。若为 True，物理分支的最后一步将采用简单的线性投影而非 MLP

      momentum:
        type: float
        default: 0.9
        description: 协方差矩阵滑动平均的动量率

      psi_dim:
        type: int
        default: 8
        description: 投影到正交空间的隐式维度尺寸

      out_dim:
        type: int
        default: 1
        description: last_layer 为 True 时的物理输出维度

    inputs:

      x:
        type: FeatureTensor
        shape: [batch, num_points, hidden_dim]
        dtype: float32
        description: 辅助特征张量

      fx:
        type: PhysicsTensor
        shape: [batch, num_points, hidden_dim]
        dtype: float32
        description: 待更新的核心物理特征张量

    outputs:

      output:
        type: Tuple[FeatureTensor, PhysicsTensor]
        shape: "([batch, N, hidden_dim], [batch, N, out_dim/hidden_dim])"
        description: 更新后的双分支特征组


  implementation:

    framework: pytorch

    code: |

      # 请参考源文件 orthogonal_neural_block.py 获取完整实现。
      # 核心逻辑包含在 psd_safe_cholesky 函数和 forward 函数内的协方差动量更新中：
      
      # batch_cov = torch.einsum("blc, bld->cd", x_, x_) / (B * L)
      # self.feature_cov.mul_(self.momentum).add_(batch_cov, alpha=1 - self.momentum)
      # L = psd_safe_cholesky(batch_cov)
      # L_inv_T = L.inverse().transpose(-2, -1)
      # x_ = x_ @ L_inv_T
      # fx = (x_ * softplus(self.mu)) @ (x_.transpose() @ fx) + fx


  skills:

    build_orthogonal_block:

      description: 构建能缓解特征平滑的双分支正交算子

      inputs:
        - num_heads
        - hidden_dim
        - attn_type
        - psi_dim

      prompt_template: |

        请实例化 OrthogonalNeuralBlock。
        对于超大规模序列，建议保持 attn_type="nystrom" 以加速矩阵逼近。
        确保在调用 forward 时，同时传入 (x, fx) 两个分支的张量。


    diagnose_cholesky_decomposition:

      description: 排查正交化过程中的数值计算崩溃

      checks:
        - non_positive_definite_covariance (协方差矩阵不是半正定，导致 Cholesky 分解返回 NaN)
        - jitter_failure (添加了最大抖动 1e-6 后依然无法分解，通常是因为梯度爆炸导致输入特征包含无穷大或 NaN)



  knowledge:

    usage_patterns:

      orthogonal_neural_operator:

        pipeline:
          - Initializer (从输入生成初始的 x 和 fx)
          - [OrthogonalNeuralBlock (last_layer=False)] x N
          - OrthogonalNeuralBlock (last_layer=True)


    hot_models:

      - model: Orthogonal Neural Operator (ONO)
        role: 结合正交多项式与深度学习架构的先进偏微分方程求解器


    best_practices:

      - `psd_safe_cholesky` 函数是整个模块的心脏。如果在训练中频繁触发 `RuntimeWarning` (added jitter to the diagonal)，说明特征学习出现了严重的不稳定，此时应该考虑降低学习率或检查输入数据的归一化情况。
      - `momentum` 参数决定了协方差矩阵对当前 batch 的敏感度。如果 batch size 很小，建议调大 momentum (如 0.99) 以保证协方差矩阵的统计稳定性。


    anti_patterns:

      - 在验证集或测试集中开启 `.train()` 模式。这会导致用单个测试样本去污染全局维护的 `feature_cov`，使得正交投影产生剧烈偏差。此模块严格依赖正确的 `if self.training:` 状态切换。


  graph:

    is_a:
      - TransformerBlock
      - NeuralOperator
      - DualBranchNetwork

    part_of:
      - ONOModel
      - AIforScienceFramework

    depends_on:
      - OneAttention (Nystrom/Linear)
      - CholeskyDecomposition

    variants:
      - None

    used_in_models:
      - Orthogonal Neural Operators

    compatible_with:

      inputs:
        - FeatureTensor
        - PhysicsTensor

      outputs:
        - UpdatedFeatureTensor
        - UpdatedPhysicsTensor