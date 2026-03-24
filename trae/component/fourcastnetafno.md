component:
  meta:
    name: FourCastNetAFNO2D
    中文名称: FourCastNet自适应傅里叶神经算子二维混合层
    别名: AFNO2D, Adaptive Fourier Neural Operator 2D
    version: 1.0
    领域: 深度学习/气象
    分类: 神经网络组件
    子类: 空间混合层/Token Mixer
    作者: OneScience
    tags:
      - 气象AI
      - 频域混合
      - 全局感受野
      - 高分辨率预报

  concept:
    描述: >
      FourCastNetAFNO2D是一种基于傅里叶变换的空间token混合层，通过2D FFT将输入映射到频域，
      在频域内对通道块执行稀疏化的复数MLP混合，再通过逆FFT还原到空间域，以O(N log N)复杂度
      实现全局感受野的token混合，替代标准Transformer的自注意力机制。
    直觉理解: >
      就像用频谱分析仪处理音频信号一样，AFNO将二维气象场转换到"频率空间"，在那里用神经网络
      调整不同频率成分的强度（类似均衡器），然后再转回空间域。这样可以同时"看到"全球所有位置
      的信息，而不像卷积只能看局部邻域。
    解决的问题:
      - 高分辨率全局依赖建模: 在0.25°分辨率（720×1440网格）上，标准自注意力的O(N²)复杂度难以承受，需要可扩展的全局token混合器。
      - 长程空间依赖捕捉: 标准卷积在极高分辨率下需要加深网络或增大卷积核才能扩大感受野，内存和计算成本迅速膨胀；AFNO通过频域全局卷积直接建模长程依赖。
      - 多尺度结构建模: 通过频域稀疏化和软阈值，AFNO能自适应地抑制高频噪声、强化结构化谱特征，同时捕捉从行星尺度到中小尺度的相互作用。

  theory:
    核心公式:
      AFNO空间混合:
        表达式: |
          z_{m,n} = DFT(X)_{m,n}
          z̃_{m,n} = S_λ(MLP(z_{m,n}))
          y_{m,n} = IDFT(z̃)_{m,n} + X_{m,n}
        变量说明:
          X:
            名称: 输入token网格
            形状: [B, H, W, C]
            描述: B为批次大小，H为纬度方向patch数，W为经度方向patch数，C为通道维度（hidden_size）
          z_{m,n}:
            名称: 频域表示
            形状: [B, H, W//2+1, num_blocks, block_size]（复数）
            描述: 2D DFT后的频域张量，(m,n)为频率索引，rfft2仅保留非冗余频率分量
          MLP(z):
            名称: 频域块对角MLP
            形状: 复数 → 复数
            描述: 对每个频率点的通道向量进行两层复数MLP变换，权重在所有频率位置共享且为块对角结构
          S_λ:
            名称: 软阈值稀疏化算子
            形状: 复数 → 复数
            描述: S_λ(x) = sign(x) * max(|x| - λ, 0)，λ为sparsity_threshold，对频域表示进行稀疏化以抑制不重要频率
          y:
            名称: 输出token网格
            形状: [B, H, W, C]
            描述: 逆FFT后的空间域张量，与输入X做残差连接

  structure:
    架构类型: 频域神经算子/Fourier Neural Operator变体
    计算流程:
      - 输入token网格X [B, H, W, C]
      - 2D实数FFT变换到频域 → z [B, H, W//2+1, num_blocks, block_size]（复数）
      - 频域硬阈值化：仅保留kept_modes = int((H//2+1) * hard_thresholding_fraction)个低频模式
      - 频域块对角MLP第一层：复数线性变换 + ReLU → o1 [B, H, W//2+1, num_blocks, block_size*hidden_size_factor]
      - 频域块对角MLP第二层：复数线性变换 → o2 [B, H, W//2+1, num_blocks, block_size]
      - 软阈值稀疏化：F.softshrink(o2, lambd=sparsity_threshold)
      - 2D逆FFT回到空间域 → y [B, H, W, C]
      - 残差连接：输出 = y + X
    计算流程图: |
      输入 X [B,H,W,C]
        ↓
      rfft2 (2D实数FFT)
        ↓
      频域张量 z [B,H,W//2+1,num_blocks,block_size] (复数)
        ↓
      硬阈值化 (保留kept_modes个低频)
        ↓
      复数MLP第1层 (w1, b1) + ReLU
        ↓
      o1 [B,H,W//2+1,num_blocks,block_size*factor] (复数)
        ↓
      复数MLP第2层 (w2, b2)
        ↓
      o2 [B,H,W//2+1,num_blocks,block_size] (复数)
        ↓
      softshrink (软阈值λ)
        ↓
      irfft2 (2D逆FFT)
        ↓
      y [B,H,W,C]
        ↓
      残差: y + X → 输出

  interface:
    参数:
      hidden_size:
        类型: int
        默认值: 768
        描述: 输入token的通道数，必须能被num_blocks整除
      num_blocks:
        类型: int
        默认值: 8
        描述: 通道分块数，block_size = hidden_size // num_blocks
      sparsity_threshold:
        类型: float
        默认值: 0.01
        描述: 软阈值化的阈值λ，用于稀疏化频域低幅度分量
      hard_thresholding_fraction:
        类型: float
        默认值: 1.0
        描述: 保留的频率模式比例，取值范围(0,1]
      hidden_size_factor:
        类型: int
        默认值: 1
        描述: 频域MLP中间层扩展倍数
    输入:
      x:
        类型: Tensor
        形状: [B, H, W, C]
        描述: 输入token网格，B为批次，H为纬度patch数，W为经度patch数，C为通道维度
    输出:
      output:
        类型: Tensor
        形状: [B, H, W, C]
        描述: 经过频域混合和残差连接后的输出，形状与输入一致

  types:
    TokenGrid:
      形状: [B, H, W, C]
      描述: 二维token网格，对应patch embedding后的气象场
    FrequencyDomain:
      形状: [B, H, W//2+1, num_blocks, block_size]（复数）
      描述: 频域表示，W//2+1为实数FFT的非冗余频率数

  constraints:
    shape_constraints:
      - 规则: hidden_size % num_blocks == 0
        描述: 通道数必须能被块数整除
      - 规则: H和W必须为正整数
        描述: 空间维度必须为正整数以支持FFT
    parameter_constraints:
      - 规则: 0 < hard_thresholding_fraction <= 1
        描述: 频率保留比例必须在(0,1]范围
      - 规则: sparsity_threshold >= 0
        描述: 软阈值必须非负
    compatibility_rules:
      - 输入类型: TokenGrid
        输出类型: TokenGrid
        描述: 输入输出形状一致，支持残差连接和多层堆叠

  implementation:
    框架: pytorch
    示例代码: |
      import torch
      import torch.nn as nn
      import torch.nn.functional as F

      class FourCastNetAFNO2D(nn.Module):
          def __init__(self, hidden_size=768, num_blocks=8,
                       sparsity_threshold=0.01, hard_thresholding_fraction=1,
                       hidden_size_factor=1):
              super().__init__()
              self.hidden_size = hidden_size
              self.num_blocks = num_blocks
              self.block_size = hidden_size // num_blocks
              self.sparsity_threshold = sparsity_threshold
              self.hard_thresholding_fraction = hard_thresholding_fraction

              scale = 0.02
              self.w1 = nn.Parameter(scale * torch.randn(
                  2, num_blocks, self.block_size, self.block_size * hidden_size_factor))
              self.b1 = nn.Parameter(scale * torch.randn(
                  2, num_blocks, self.block_size * hidden_size_factor))
              self.w2 = nn.Parameter(scale * torch.randn(
                  2, num_blocks, self.block_size * hidden_size_factor, self.block_size))
              self.b2 = nn.Parameter(scale * torch.randn(2, num_blocks, self.block_size))

          def forward(self, x):
              bias = x
              B, H, W, C = x.shape

              # FFT到频域
              x = torch.fft.rfft2(x.float(), dim=(1, 2), norm="ortho")
              x = x.reshape(B, H, W//2+1, self.num_blocks, self.block_size)

              # 硬阈值化
              total_modes = H // 2 + 1
              kept_modes = int(total_modes * self.hard_thresholding_fraction)

              # 复数MLP
              o1_real = torch.zeros([B, H, W//2+1, self.num_blocks,
                                     self.block_size * self.hidden_size_factor], device=x.device)
              o1_imag = torch.zeros_like(o1_real)

              o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
                  torch.einsum('...bi,bio->...bo',
                               x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real,
                               self.w1[0]) -
                  torch.einsum('...bi,bio->...bo',
                               x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag,
                               self.w1[1]) + self.b1[0])

              o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
                  torch.einsum('...bi,bio->...bo',
                               x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag,
                               self.w1[0]) +
                  torch.einsum('...bi,bio->...bo',
                               x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real,
                               self.w1[1]) + self.b1[1])

              o2_real = torch.zeros(x.shape, device=x.device)
              o2_imag = torch.zeros(x.shape, device=x.device)

              o2_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = (
                  torch.einsum('...bi,bio->...bo', o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) -
                  torch.einsum('...bi,bio->...bo', o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) + self.b2[0])

              o2_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = (
                  torch.einsum('...bi,bio->...bo', o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) +
                  torch.einsum('...bi,bio->...bo', o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) + self.b2[1])

              # 软阈值稀疏化
              x = torch.stack([o2_real, o2_imag], dim=-1)
              x = F.softshrink(x, lambd=self.sparsity_threshold)
              x = torch.view_as_complex(x)

              # 逆FFT
              x = x.reshape(B, H, W//2+1, C)
              x = torch.fft.irfft2(x, s=(H, W), dim=(1,2), norm="ortho")

              return x.type(bias.dtype) + bias

  usage_examples:
    - 描述: FourCastNet标准配置
      示例代码: |
        afno = FourCastNetAFNO2D(hidden_size=768, num_blocks=8)
        x = torch.randn(2, 90, 180, 768)
        out = afno(x)

    - 描述: 降低频率保留比例
      示例代码: |
        afno = FourCastNetAFNO2D(hidden_size=768, num_blocks=8,
                                  hard_thresholding_fraction=0.5)
        x = torch.randn(2, 90, 180, 768)
        out = afno(x)

  knowledge:
    应用说明: >
      FourCastNetAFNO2D属于AFNO范式的空间混合模块，在气象AI建模中作为Vision Transformer的token mixer替代方案。
      该组件通过频域全局卷积实现O(N log N)复杂度的全局感受野建模，解决了高分辨率气象场上标准自注意力O(N²)
      复杂度不可承受的问题。AFNO将Fourier Neural Operator思想引入全球天气预报，在频域通过块对角MLP和
      软阈值稀疏化实现自适应频谱滤波，能够同时捕捉从行星尺度到中小尺度的多尺度大气相互作用。
    热点模型:
      - 模型: FourCastNet
        年份: 2022
        场景: 全球高分辨率（0.25°）中期天气预报，预测未来1周内20个大气变量的演化
        方案: |
          采用AFNO作为ViT骨干的空间token混合层，替代标准自注意力。输入经过patch embedding（8×8 patch）
          后形成token网格，通过12层AFNO block（每层包含AFNO空间混合+MLP通道混合）进行特征提取。
          AFNO在频域对token进行全局混合，通过硬阈值保留重要频率模式、软阈值稀疏化抑制噪声，
          最后通过线性解码器重建下一时刻的气象场。
        作用: |
          作为主干网络的核心空间混合模块，AFNO实现了全局长程依赖建模，使模型能够捕捉大气中的遥相关模式
          （如ENSO、NAO等）和跨尺度相互作用。相比卷积网络，AFNO在有限层数内即可覆盖全球范围；
          相比标准Transformer，AFNO将复杂度从O(N²)降至O(N log N)，使得在720×1440高分辨率网格上的
          训练和推理成为可能。
        创新: |
          1) 将Fourier Neural Operator引入全球天气预报，首次在气象AI中系统应用频域神经算子；
          2) 通过块对角MLP结构在频域实现通道间的稀疏耦合，平衡表达能力与参数效率；
          3) 结合硬阈值和软阈值的双重频域正则化策略，增强对结构化谱特征的学习；
          4) 在0.25°分辨率上实现与ECMWF IFS相当的预报技能，推理速度快4-5个数量级。

      - 模型: FourCastNet降水诊断模型
        年份: 2022
        场景: 从主干模型预测的20个大气变量中诊断6小时累计降水，捕捉极端降水事件
        方案: |
          使用与主干模型相同的AFNO架构，但输入为主干预测的20变量场，输出为单通道降水场。
          在AFNO主干后接2D卷积（周期padding）+ReLU强制非负输出。对降水应用log变换缓解稀疏和长尾分布问题。
        作用: |
          AFNO作为降水诊断网络的特征提取器，从多变量大气状态中学习降水的复杂非线性关系。
          频域全局视野使模型能够捕捉降水相关的大尺度环流模式（如气河、热带气旋）和局地对流结构。
        创新: |
          首次展示DL诊断降水模型在全球尺度上与IFS具有可比skill，能够解析小尺度降水结构。
          通过独立建模降水避免其稀疏分布干扰主干训练，同时利用AFNO的多尺度建模能力改善极端降水预报。

    最佳实践:
      - 通道数hidden_size应设为num_blocks的整数倍，典型配置为768维分8块（block_size=96）
      - hard_thresholding_fraction默认为1.0保留全部频率；若需降低计算量可设为0.5-0.8保留低频模式
      - sparsity_threshold典型值为0.01，过大会过度稀疏化导致信息丢失
      - 与token-wise MLP交替堆叠形成AFNO block，先AFNO空间混合后MLP通道混合
      - 输入前需进行patch embedding和位置编码，输出后需线性解码回原始分辨率
      - 频域计算对数值精度敏感，forward中将输入转为float32进行FFT计算 [基于通用知识补充]
    常见错误:
      - 忘记hidden_size必须被num_blocks整除，导致reshape失败
      - 混淆输入形状约定：AFNO要求[B,H,W,C]而非[B,C,H,W]
      - 硬阈值化kept_modes计算错误，应为int((H//2+1) * fraction)
      - 软阈值lambd设置过大导致频域表示过度稀疏
      - 复数MLP权重初始化scale过大导致训练初期梯度爆炸 [基于通用知识补充]
    论文参考:
      - 标题: "FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators"
        作者: Pathak et al.
        年份: 2022
        摘要: |
          提出FourCastNet，一个基于AFNO的全球高分辨率（0.25°）天气预报模型。通过将Fourier Neural
          Operator引入Vision Transformer架构，实现O(N log N)复杂度的全局token混合。模型在多个变量上
          与ECMWF IFS性能相当，推理速度快4-5个数量级，支持100-1000成员的大规模集合预报。

  graph:
    类型关系:
      - 神经网络层
      - 空间混合模块
      - Token Mixer
    所属结构:
      - FourCastNet主干网络
      - AFNO Block
      - Vision Transformer变体
    依赖组件:
      - Patch Embedding
      - 位置编码
      - Token-wise MLP
      - 线性解码器
    变体组件:
      - 标准Transformer自注意力
      - Swin Transformer窗口注意力
      - 卷积层
    使用模型:
      - FourCastNet
      - FourCastNet降水诊断模型
    类型兼容:
      输入:
        - TokenGrid
      输出:
        - TokenGrid
