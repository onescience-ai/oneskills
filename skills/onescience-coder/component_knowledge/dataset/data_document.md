# 数据说明

各个领域数据的说明，包括数据来源、数据格式、数据用途、使用领域、是否存储、读取接口、适用模型等信息，方便用户了解和使用数据。
## 生信

| 数据名称 | 数据大小 | 数据格式 |数据来源 | 数据用途|使用领域|是否存储|读取接口|适用模型|备注|
|-------|-------|-------|-------|-------|--------|--------|-------|-------|-------|
|wwPDB|772GB|.cif，.pdb，.pkl等| [下载地址](https://af3-dev.tos-cn-beijing.volces.com/release_data.tar.gz) |训练/推理|蛋白质/复合物推理训练|新一代机器集群|--|Protenix，Alphafold3|--|
|Alphafold3|860G|cif、fasta、fa等|--| MSA搜索|蛋白质、复合物推理|新一代机器集群|--|Alphafold3|--|
|alphafold2.3.0|--|pdb、cif等|--|MSA搜索|蛋白质单体推理|新一代机器集群|--|alphafold2.3.0|--|
|proteinmpnn|--|pdb|--| -- |--|新一代机器集群|--|--|--|
|evo2|5T|Json、fasta格式|--| 训练 |基因|新一代机器集群|--|evo2|--|

## 海洋

| 数据名称 | 数据大小 | 数据格式 |数据来源 | 数据用途|使用领域|是否存储|读取接口|适用模型|备注|
|---------|---------|---------|---------|---------|----------|----------|---------|---------|---------|
|ERA5|418G/每年|.h5| [ERA5官方下载](https://cds.climate.copernicus.eu/) |气象海洋模型训练|气象/海洋|新一代机器集群|onescience.datapipes.climate.ERA5DF5Datapipe|所有气象海洋类模型均适用|--|
| CMEMS |161M/每年*25|.h5| [哥白尼海洋数据中心](https://cmems-du.eu/) |气象海洋模型训练|海洋|新一代机器集群|onescience.utils.fcn.data_loader_ocean.py|所有气象海洋类模型均适用|--|

## 材料

| 数据名称 | 数据大小 | 数据格式 |数据来源 | 数据用途|使用领域|是否存储|读取接口|适用模型|备注|
|---------|---------|---------|---------|---------|----------|----------|---------|---------|---------|
|MPtrj|11.35GB|json|[下载地址](https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842/2)|模型训练、微调、推理|无机晶体：材料发现和性能预测|是|--|MACE和UMA|--|
|OC20|1.1T|extxyz预处理以后变为aselmdb|[下载地址](https://fair-chem.github.io/catalysts/datasets/oc20.html)|微调、推理|催化剂：加速催化剂发现|是|--|UMA|--|
|碳酸盐类分子液体数据集|16.9 MB|xyz格式|从github上拉[下载地址](https://github.com/imagdau/Tutorials)|训练、推理|碳酸盐类分子液体研究|是|--|UMA|--|

## 流体

| 数据名称 | 数据大小 | 数据格式 |数据来源 | 数据用途|使用领域|是否存储|读取接口|适用模型|备注|
|---------|---------|---------|---------|---------|----------|----------|---------|---------|---------|
|ShapeNet-car|17GB|vtk、预处理后为npy|[下载地址](http://www.nobuyuki-umetani.com/publication/mlcfd_data.zip)|汽车仿真|空气动力学|是|--|Transolver、CFD_Benchmark|--|
|AirfRANS|9GB|vtk|[下载地址](https://github.com/Extrality/AirfRANS)|翼型仿真|空气动力学|是|--|Transolver|--|
|deepmind涡激数据集|15GB|tfrecord|[下载地址](https://storage.googleapis.com/dm-meshgraphnets/)|圆柱绕流|基础流体力学仿真|是|--|MeshGraphNet|--|
|deepmind粒子数据集|8GB|tfrecord|[下载地址](https://storage.googleapis.com/dm-meshgraphnets//)|拉格朗日粒子流动问题|多相流|是|--|MeshGraphNet|--|
|Eagle无人机数据集|91GB|npy|[下载地址](https://datasets.liris.cnrs.fr/eagle-version1)|湍流模拟|湍流|是|--|EagleMeshTransformer|--|
|CFDBench数据集（4个经典问题）|13.4GB|npy|[下载地址](https://huggingface.co/datasets/chen-yingfa/CFDBench)|坝流，圆柱绕流，方腔流。管道流动|基础流体力学仿真|是|--|CFDBench|--|
|PDEBench|91GB|npy|[下载地址](https://storage.googleapis.com/dm-meshgraphnets//)|拉格朗日粒子流动问题|多相流|是|--|MeshGraphNet|--|
|DeepCFD管道流|91GB|npy|[下载地址](https://storage.googleapis.com/dm-meshgraphnets//)|拉格朗日粒子流动问题|多相流|是|--|MeshGraphNet|--|
|复杂边界椭圆偏微分方程数据集|1GB|npy|[下载地址](https://drive.google.com/file/d/11PbUrzJ-b18VhFGY_uICSciCkeGrsaTZ/view)|复杂边界椭圆偏微分方程数据集|通用PDE仿真|是|--|beno|--|

