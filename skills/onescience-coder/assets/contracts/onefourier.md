# Contract: OneFourier

## 基本信息

- 组件名：`OneFourier`
- 所属模块族：`fourier`
- 统一入口：`direct_import`
- 注册名：`style="<FourierStyle>"`

## 组件职责

为频域卷积、几何谱卷积和多种 wavelet/fourier 变换提供统一注册入口，常用于神经算子模型中的全局谱域特征建模。

补充说明：

- 调用层通过 `style` 选择具体谱算子实现
- 当前与 CFD 任务最相关的 style 主要是 `FNOSpectralConv1d/2d/3d` 与 `GeoSpectralConv2d`
- wrapper 不统一不同 style 的 forward 签名；调用层必须知道目标算子的真实参数形式

## 支持输入

- 2D 输入：`depends_on_style`
- 3D 输入：`depends_on_style`

内部统一做法：

- 先检查 `style` 是否已注册
- 再将构造参数透传给对应谱算子
- forward 时不做额外 shape 修正

## 构造参数

- `style`
  - 具体谱算子注册名
- `in_channels`
  - 输入通道数
- `out_channels`
  - 输出通道数
- `modes1 / modes2 / modes3`
  - 频域截断模态数
- `s1 / s2`
  - `GeoSpectralConv2d` 这类几何投影层常用的潜在网格尺寸
- `**kwargs`
  - 其余参数直接透传给具体 Fourier 层

## 输出约定

- 2D 输出：`depends_on_style`
- 3D 输出：`depends_on_style`

额外约束：

- 常规 `FNOSpectralConv*d` 会保持 batch 和空间维结构，只变换通道
- `GeoSpectralConv2d` 额外依赖 `x_in/x_out/iphi` 等参数，不能按普通卷积层的调用方式理解

## 典型调用位置

- `CFD_Benchmark` 中 `FNO / U_FNO / U_NO / U_Net`
- 使用 Geo-FNO 投影的非结构网格 2D CFD 任务

## 典型参数

- 结构网格 FNO
  - `style="FNOSpectralConv2d"`
  - `in_channels=n_hidden`
  - `out_channels=n_hidden`
  - `modes1=modes, modes2=modes`
- 非结构 2D 几何投影
  - `style="GeoSpectralConv2d"`
  - `s1=96, s2=96`

## 风险点

- `GeoSpectralConv2d` 的 forward 签名和普通 `FNOSpectralConv2d` 不同，直接替换很容易出错
- `modes*` 不能脱离真实分辨率和 `shapelist` 单独设置，否则会出现频域截断与网格不匹配
- 当前 `CFD_Benchmark` 几个模型的非结构分支主要围绕 2D Geo 投影写法组织，不要默认它们已经自然覆盖任意 3D 非结构任务

## 源码锚点

- `./onescience/src/onescience/modules/fourier/onefourier.py`
- `./onescience/src/onescience/modules/fourier/fno_layers.py`
- `./onescience/src/onescience/modules/fourier/geo_spectral.py`
