# CFD Datapipe 编写指南 (Skills)

本文档是 OneScience CFD 领域 Datapipe 的编写规范与模板，基于已有的 AirfRANS、ShapeNetCar、DeepCFD、Eagle、DeepMind_CylinderFlow、DeepMindLagrangian、BENO、CFDBench、DrivAerML 等数据集总结而成。

---

## 1. 整体架构

每个 CFD 数据集由 **两个类** 组成：

| 类 | 职责 |
|---|---|
| `XXXDataset(BaseDataset)` | 继承 `BaseDataset`，负责数据加载、归一化、预处理、`__getitem__` |
| `XXXDatapipe` | 包装 Dataset，提供 `train_dataloader()` / `val_dataloader()` / `test_dataloader()` |

文件命名：`xxx.py`，放在 `src/onescience/datapipes/cfd/` 目录下。

---

## 2. Dataset 类模板

```python
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from onescience.datapipes.core import BaseDataset
from onescience.distributed.manager import DistributedManager


class XXXDataset(BaseDataset):
    """
    XXX CFD 数据集

    继承自 BaseDataset，用于加载和处理 XXX 数据。
    """

    # ========== 类元数据 (必须覆盖) ==========
    DOMAIN = "cfd"
    TASK = "regression"  # 或 "forecasting"
    DATA_FORMATS = ["vtu", "npy", "pkl", "npz", "tfrecord"]  # 按实际填写

    def __init__(
        self,
        config: Union[Dict[str, Any], Any],
        mode: str = "train",
        coef_norm: Optional[Tuple] = None,
    ):
        """
        Parameters
        ----------
        config : Dict or DictConfig
            数据集配置 (Hydra/OmegaConf)
        mode : str
            'train', 'val', 'test'
        coef_norm : tuple, optional
            (mean_in, std_in, mean_out, std_out)。
            train 模式下可为 None（会自动计算），val/test 必须由 Datapipe 传入。
        """
        self.mode = mode
        self._provided_coef_norm = coef_norm
        self.coef_norm = None
        self.data_list_names = []  # 样本名/路径列表

        # 分布式管理器 (判断 rank、控制日志)
        self.dist = DistributedManager()

        # 调用 BaseDataset.__init__ → 设置 self.config, self.logger
        super().__init__(config)

        # 初始化流程
        self._init_paths()       # 解析路径、加载 manifest、划分 split
        self._load_metadata()    # 加载或计算归一化统计量
        self._init_data()        # (可选) 预加载数据到内存

        # 控制非主进程日志
        if self.dist.rank != 0:
            self.logger.setLevel(logging.WARNING)

        if self.dist.rank == 0:
            self.logger.info(
                f"[{self.mode}] XXXDataset initialized with {len(self)} samples."
            )

    # =====================================================================
    #  路径与数据划分
    # =====================================================================
    def _init_paths(self):
        """
        初始化数据路径，加载 manifest/metadata，按 mode 划分样本列表。

        常见模式:
        - manifest.json 中按 key 划分 (AirfRANS)
        - 按比例随机划分 (DeepCFD)
        - 按文件夹结构划分 (Eagle)
        """
        super()._init_paths()  # 设置 self.data_path

        # === 方式一：manifest.json 划分 ===
        # manifest_path = self.data_path / "manifest.json"
        # with open(manifest_path, "r") as f:
        #     manifest = json.load(f)
        # if self.mode == "train":
        #     self.data_list_names = manifest["train"]
        # elif self.mode == "val":
        #     self.data_list_names = manifest["val"]
        # elif self.mode == "test":
        #     self.data_list_names = manifest["test"]

        # === 方式二：按比例划分 ===
        # all_files = sorted(self.data_path.glob("*.npy"))
        # indices = list(range(len(all_files)))
        # random.Random(self.config.data.seed).shuffle(indices)
        # split_idx = int(len(indices) * self.config.data.split_ratio)
        # if self.mode == "train":
        #     selected = indices[:split_idx]
        # else:
        #     selected = indices[split_idx:]
        # self.data_list_names = [all_files[i] for i in selected]

        pass  # 按实际需求实现

    # =====================================================================
    #  归一化统计量
    # =====================================================================
    def _load_metadata(self):
        """
        加载或计算归一化系数 (mean_in, std_in, mean_out, std_out)。

        优先级：
        1. 构造函数传入的 coef_norm
        2. 已保存的 .npy 文件
        3. (仅 train) 遍历训练集计算 → 保存
        """
        stats_dir = Path(self.config.source.stats_dir)
        stats_dir.mkdir(parents=True, exist_ok=True)

        paths = {
            k: stats_dir / f"{k}.npy"
            for k in ("mean_in", "std_in", "mean_out", "std_out")
        }

        # 优先使用传入的
        if self._provided_coef_norm:
            self.coef_norm = self._provided_coef_norm
            return

        # 尝试从文件加载
        if all(p.exists() for p in paths.values()):
            arrays = {k: np.load(p) for k, p in paths.items()}
            self.coef_norm = (
                arrays["mean_in"],
                arrays["std_in"],
                arrays["mean_out"],
                arrays["std_out"],
            )
            if self.dist.rank == 0:
                self.logger.info(f"Loaded normalization stats from {stats_dir}")
            return

        # 训练模式下计算
        if self.mode == "train":
            if self.dist.rank == 0:
                self.logger.warning("Stats not found, calculating on the fly...")
            self.coef_norm = self._calculate_normalization()
            for k, arr in zip(paths.keys(), self.coef_norm):
                np.save(paths[k], arr)
            return

        raise FileNotFoundError(
            f"Normalization stats not found in {stats_dir}, and mode is not 'train'."
        )

    def _calculate_normalization(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        两遍遍历训练集：第一遍算均值，第二遍算标准差。
        返回 (mean_in, std_in, mean_out, std_out)。
        """
        assert self.mode == "train"

        mean_in = mean_out = std_in = std_out = None
        count = 0

        # Pass 1: mean
        for s in self.data_list_names:
            _, x, y, _ = self._load_single_simulation(s)
            n = x.shape[0]
            if mean_in is None:
                mean_in = x.mean(axis=0, dtype=np.float64)
                mean_out = y.mean(axis=0, dtype=np.float64)
                count = n
            else:
                new_count = count + n
                mean_in += (x.sum(axis=0, dtype=np.float64) - n * mean_in) / new_count
                mean_out += (y.sum(axis=0, dtype=np.float64) - n * mean_out) / new_count
                count = new_count

        mean_in = mean_in.astype(np.float32)
        mean_out = mean_out.astype(np.float32)

        # Pass 2: std
        count = 0
        for s in self.data_list_names:
            _, x, y, _ = self._load_single_simulation(s)
            n = x.shape[0]
            if std_in is None:
                count = n
                std_in = ((x - mean_in) ** 2).sum(axis=0, dtype=np.float64) / count
                std_out = ((y - mean_out) ** 2).sum(axis=0, dtype=np.float64) / count
            else:
                new_count = count + n
                std_in += (
                    ((x - mean_in) ** 2).sum(axis=0, dtype=np.float64) - n * std_in
                ) / new_count
                std_out += (
                    ((y - mean_out) ** 2).sum(axis=0, dtype=np.float64) - n * std_out
                ) / new_count
                count = new_count

        std_in = np.sqrt(std_in).astype(np.float32)
        std_out = np.sqrt(std_out).astype(np.float32)

        return (mean_in, std_in, mean_out, std_out)

    # =====================================================================
    #  (可选) 预加载数据到内存
    # =====================================================================
    def _init_data(self):
        """
        可选：对小数据集一次性加载到内存 (如 DeepCFD)。
        大数据集可留空，在 __getitem__ 中按需加载。
        """
        pass

    # =====================================================================
    #  加载单个样本的原始数据
    # =====================================================================
    def _load_single_simulation(
        self, s
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        加载单个模拟/样本的原始数据。

        Parameters
        ----------
        s : str or Path
            样本名称或路径

        Returns
        -------
        pos : np.ndarray, shape (N, D)
            空间坐标 (2D/3D)
        x : np.ndarray, shape (N, C_in)
            输入特征 (坐标 + 边界条件 + SDF + ...)
        y : np.ndarray, shape (N, C_out)
            目标变量 (速度 + 压力 + ...)
        extra : np.ndarray or dict
            额外信息 (如 surface 标记, 边界类型等)

        Notes
        -----
        常用数据格式和对应读取方式：
        - VTU/VTP: pyvista.read(path)
        - NPY/NPZ: np.load(path)
        - PKL: pickle.load(f)
        - TFRecord: tf.data.TFRecordDataset
        - 自定义二进制: struct.unpack / numpy.fromfile

        CFD 领域常见物理量：
        输入: 坐标(x,y,z), 入口速度(U_inf), 攻角(alpha), SDF, 法向量, Reynolds数
        输出: 速度场(Ux,Uy,Uz), 压力(p), 湍流粘度(nut), 壁面剪切应力(tau)
        """
        raise NotImplementedError("子类必须实现 _load_single_simulation")

    # =====================================================================
    #  Dataset 核心接口
    # =====================================================================
    def __len__(self) -> int:
        return len(self.data_list_names)

    def __getitem__(self, idx: int):
        """
        获取第 idx 个样本，完成以下流程:
        1. 加载原始数据
        2. 归一化
        3. 转为 Tensor
        4. (可选) 子采样
        5. (可选) 构建图结构
        6. 返回 Data/Dict

        Returns
        -------
        根据任务类型返回不同格式：
        - 图网络 (GNN): torch_geometric.data.Data
        - 网格方法 (CNN/FNO): Dict[str, Tensor]
        - 时序任务: Dict[str, Tensor] (含时间步)
        """
        sample_id = self.data_list_names[idx]

        try:
            # 1. 加载
            pos, x, y, extra = self._load_single_simulation(sample_id)

            # 2. 归一化
            if self.coef_norm:
                mean_in, std_in, mean_out, std_out = self.coef_norm
                x = (x - mean_in) / (std_in + 1e-8)
                y = (y - mean_out) / (std_out + 1e-8)

            # 3. 转 Tensor
            pos = torch.tensor(pos, dtype=torch.float32)
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)

            # 4. 子采样 (训练/验证时)
            if self.mode in ("train", "val"):
                n_sub = getattr(self.config.data, "subsampling", None)
                if n_sub and pos.size(0) > n_sub:
                    idx_s = random.sample(range(pos.size(0)), n_sub)
                    idx_s = torch.tensor(idx_s)
                    pos, x, y = pos[idx_s], x[idx_s], y[idx_s]

            # 5. 构建图 (如果使用 GNN)
            # edge_index = nng.radius_graph(pos, r=radius, max_num_neighbors=k)

            # 6. 返回
            # --- PyG Data 格式 (GNN) ---
            # from torch_geometric.data import Data
            # return Data(pos=pos, x=x, y=y, edge_index=edge_index)

            # --- Dict 格式 (CNN / FNO / Transformer) ---
            return {"x": x, "y": y, "pos": pos}

        except Exception as e:
            self.logger.error(
                f"Error loading sample {idx} ({sample_id}): {e}", exc_info=True
            )
            return {}  # 返回空对象，避免训练崩溃
```

---

## 3. Datapipe 类模板

```python
import copy
from typing import Dict, Any, Tuple, Optional
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class XXXDatapipe:
    """
    XXX 数据集的 Datapipe 封装。

    职责：
    - 实例化 train/val/test Dataset
    - 传递归一化系数 (train → val/test)
    - 提供 DataLoader 工厂方法
    """

    def __init__(self, params, distributed: bool = False):
        self.params = params
        self.distributed = distributed

        # 1. 先创建训练集 (会计算 / 加载归一化统计量)
        self.train_dataset = XXXDataset(
            config=copy.deepcopy(params), mode="train"
        )

        # 2. 获取归一化系数，传给 val/test
        self.coef_norm = self.train_dataset.coef_norm

        # 3. 创建验证集和测试集
        self.val_dataset = XXXDataset(
            config=copy.deepcopy(params), mode="val", coef_norm=self.coef_norm
        )
        self.test_dataset = XXXDataset(
            config=copy.deepcopy(params), mode="test", coef_norm=self.coef_norm
        )

    def train_dataloader(self) -> Tuple[DataLoader, Optional[DistributedSampler]]:
        sampler = (
            DistributedSampler(self.train_dataset, shuffle=True)
            if self.distributed
            else None
        )
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.params.dataloader.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.params.dataloader.num_workers,
            pin_memory=True,
            drop_last=self.distributed,
        )
        return loader, sampler

    def val_dataloader(self) -> Tuple[DataLoader, Optional[DistributedSampler]]:
        sampler = (
            DistributedSampler(self.val_dataset, shuffle=False)
            if self.distributed
            else None
        )
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.params.dataloader.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.params.dataloader.num_workers,
            pin_memory=True,
            drop_last=self.distributed,
        )
        return loader, sampler

    def test_dataloader(self) -> Tuple[DataLoader, Optional[DistributedSampler]]:
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.params.dataloader.batch_size,
            shuffle=False,
            num_workers=self.params.dataloader.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return loader, None
```

> **注意**: 如果使用 PyTorch Geometric 的图数据，需使用 `torch_geometric.loader.DataLoader` 替代标准 `DataLoader`。
> 如果使用 DGL 图数据，需使用 `dgl.dataloading.GraphDataLoader` 或自定义 collate_fn。

---

## 4. 返回数据格式速查

根据下游模型选择合适的返回格式：

| 模型类型 | 返回格式 | 示例 |
|---------|---------|------|
| **GNN (点云/非结构化网格)** | `torch_geometric.data.Data` | AirfRANS, ShapeNetCar, BENO |
| **CNN / U-Net (规则网格)** | `Dict[str, Tensor]` (BCHW) | DeepCFD |
| **时序 GNN (MeshGraphNet)** | `dgl.DGLGraph` | DeepMind_CylinderFlow, Lagrangian |
| **Transformer / FNO** | `Dict[str, Tensor]` | CFDBench |
| **预构建图** | `Dict[str, Tensor]` + 预存图 | DrivAerML |

---

## 5. 图构建方式速查

```python
# ===== PyTorch Geometric 半径图 =====
import torch_geometric.nn as nng
edge_index = nng.radius_graph(
    x=pos,               # (N, D) 节点坐标
    r=radius,             # 连接半径
    loop=True,            # 是否包含自环
    max_num_neighbors=64  # 最大邻居数
)

# ===== 从 mesh cell 提取边 (DGL) =====
import dgl
def cells_to_edges(cells):
    """将三角形/四面体面片转为边列表"""
    edges = set()
    for cell in cells:
        for i in range(len(cell)):
            for j in range(i + 1, len(cell)):
                edges.add((cell[i], cell[j]))
                edges.add((cell[j], cell[i]))
    src, dst = zip(*edges)
    return dgl.graph((src, dst))

# ===== PyG HeteroData (异质图) =====
from torch_geometric.data import HeteroData
data = HeteroData()
data["node"].x = node_features
data["node", "connects", "node"].edge_index = edge_index
```

---

## 6. 数据读取工具 (readers.py)

项目自带的 `readers.py` 提供以下工具函数：

```python
from onescience.datapipes.cfd.readers import read_vtp, read_vtu, read_cgns, read_stl
```

也可直接使用 pyvista:
```python
import pyvista as pv
mesh = pv.read("path/to/file.vtu")
points = mesh.points                      # (N, 3)
velocity = mesh.point_data["U"]          # (N, 3)
pressure = mesh.point_data["p"]          # (N,)
```

---

## 7. 配置文件 (Hydra YAML) 参考

```yaml
# config.yaml
source:
  data_dir: /path/to/dataset
  stats_dir: /path/to/stats   # 归一化统计量保存位置

data:
  seed: 42
  split_ratio: 0.8            # train/test 划分比例
  subsampling: 10000           # 子采样点数
  variables: [U, p, nut]      # 物理变量
  splits:                      # manifest 划分配置
    task: full
    train_name: full_train
    test_name: full_test
    val_split_ratio: 0.1
  sampling:                    # 采样策略
    sample_strategy: uniform   # null / uniform / mesh
    n_boot: 5000
    surf_ratio: 0.2
  crop: null                   # 裁剪边界 [xmin, xmax, ymin, ymax]

model_hparams:
  build_graph: true
  r: 0.1                      # 图连接半径
  max_neighbors: 64

dataloader:
  batch_size: 4
  num_workers: 4
```

---

## 8. 注册新数据集

在 `__init__.py` 中添加导入：

```python
# src/onescience/datapipes/cfd/__init__.py
from .xxx import XXXDataset, XXXDatapipe
```

---

## 9. Checklist

新建 CFD Datapipe 时请确认：

- [ ] 继承 `BaseDataset`，设置 `DOMAIN = "cfd"`
- [ ] `__init__` 接受 `config`, `mode`, `coef_norm` 三个参数
- [ ] 使用 `DistributedManager()` 控制日志输出
- [ ] `_init_paths()` 调用 `super()._init_paths()`，并按 mode 划分样本列表
- [ ] `_load_metadata()` 实现三级加载：传入 → 文件 → 计算
- [ ] `_load_single_simulation()` 返回 `(pos, x, y, extra)` 四元组
- [ ] `__getitem__` 包含归一化、转 Tensor、子采样流程
- [ ] `__getitem__` 用 try/except 包裹，异常时返回空对象
- [ ] Datapipe 类创建 train/val/test 三个 Dataset
- [ ] 归一化系数从 train_dataset 传递给 val/test
- [ ] DataLoader 支持 `DistributedSampler`
- [ ] 在 `__init__.py` 注册导出
