# Metad-toy: LAMMPS Metadynamics Plugin

**Metad-toy** 是一个基于 LAMMPS 插件机制开发的元动力学 (Metadynamics) 插件。它允许用户在 CUDA 加速的环境下，通过自定义集体变量 (CVs) 来驱动复杂系统的相变或化学反应模拟。

---

## ✨ 功能特性

* **多维网格加速**：支持 1D、2D、3D 的稠密与稀疏势能网格。
* **Cubic 插值**：使用三次插值算法计算网格梯度，确保原子受力的连续性，提高积分稳定性。
* **多种 CV 支持**：包括 `DISTANCE`、`STEINH`、`STRU_FACTOR`、`WEIGHT_CHEM` 及多通道 `CSS`。
* **运行时表达式图**：实验性的 `EXECUTOR GRAPH` 将逐原子表达式编译为 CUDA kernel，并连接 MPI 均值和偏置梯度回传。
* **Well-tempered 支持**：通过 `WT` 关键字实现自适应的高斯高度调整。
* **断点续算**：能够读取 `HILLS` 文件并重建势能表面。

---

## 🛠️ 安装指南

### 前提条件

* LAMMPS (29 Aug 2024 或更新版本)
* CMake 3.20+
* CUDA Toolkit (用于 `.cu` 源码编译)
* MPICH 或 OpenMPI

### 编译步骤

1. **准备依赖**：安装 CUDA Toolkit（含 NVRTC）、xxd，以及与 LAMMPS 一致的 MPI。Graph 使用 LAMMPS 自带的 fmt 头文件。下面的基础构建关闭可选 ML CV。

2. **构建插件**：根据安装位置修改路径，并将 `89` 换成目标 GPU 架构。
```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release -DENABLE_DEBUG=OFF -DUSE_ML_CVS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCLEAN_HOST_COMPILER=/usr/bin/g++ \
  -DLAMMPS_SOURCE_DIR=/path/to/lammps/src \
  -DLAMMPS_LIB_DIR=/path/to/lammps/lib \
  -DMPI_HOME=/path/to/mpi
cmake --build build -j4
```

3. **加载插件**：
`fix_crystallize_plugin.so`将在本项目下的`build`文件夹中生成，可以将它复制到任何地方使用。
编译完成后，在 LAMMPS 脚本中使用 `plugin load path/to/fix_crystallize_plugin.so` 加载。

---

## 📝 LAMMPS 输入参数

使用 `fix metad` 指令调用该插件。

### 语法示例

```lammps
fix metad all metad GAUSSIAN 0.01 0.005 10.0 &
                    PACE 10 &
                    RECORD FILE_NAME rec.log REC_PACE 1 &
                    CAL NAME Q6 STEINH Q 6 metad_group cutoff_r 3.9 cutoff_Natoms 18 &
                    CAL NAME Q4 STEINH Q 4 metad_group cutoff_r 3.9 cutoff_Natoms 18 &
                    CV_dim 1 &
                    SYMBOL v1 Q6.AVE &
                    SYMBOL v2 Q4.AVE &
                    DIM 1 0 1 1000 "v1+v2" &
                    METAD_RESTART 1 &
                    WT 0
```

### 参数说明

| 关键字 | 参数 | 描述 |
| --- | --- | --- |
| `GAUSSIAN` | `[sigma] [height] [biasf]` | 高斯宽度、初始高度和偏置因子。 |
| `PACE` | `[N]` | 每隔 N 步添加一次高斯函数。 |
| `CAL` | `NAME [cv_set_name] [{CV set}] ` | 设置一个CV的计算。 |
| `{CV set}=DISTANCE` | `[ID1] [ID2]` | 定义两个原子之间的距离作为 CV。 |
| `{CV set}=STEINH` | `[Q/L] [3/4/6] [group] <cutoff_r [r]> <cutoff_Natoms [N]>` | 定义 Steinhardt 序参数作为 CV。 |
| `CV_dim` | `[N]` | 集体变量的总维度。 |
| `SYMBOL` | `[variable name] [cv_set_name].[cv_func]` | 指定cv计算的变量名所对应的cv计算函数方法 |
| `DIM` | `[idx] [low] [up] [bins] "[expr]"` | 设置特定维度的网格范围和精度以及变量组合的表达式。 |
| `METAD_RESTART` | `[0/1]` | 是否启用 HILLS 阅读。 |
| `WT` | `[0/1]` | 是否启用 Well-tempered Metadynamics。 |

---

## 📂 项目结构

```text
.
├── src/                # 核心实现 (.cu 和 .cpp)
├── include/            # 头文件 (.h)
├── third_party/            # 头文件 (.h)
├── CMakeLists.txt      # 构建脚本
└── test/               # 测试用例 (DISTANCE/STEINH)

```

---

## 🧪 开发与调试

当前版本为 **实验性功能验证版（experimental preview）**。Graph 已接入完整 LAMMPS 流程，支持普通/化学 `STRU_FACTOR`、`CSS` 和 `WEIGHT_CHEM`；这些 CV 需要显式设置 `EXECUTOR GRAPH`。

更新内容、接口迁移和验证范围见 [更新日志](CHANGELOG.md)。独立测试入口见 [Graph 测试](benchmarks/graph/README.md)，完整流程示例见 [Stru_factor 接入说明](benchmarks/strufactor_lammps_graph/README.md)。

---

## 使用文档 / Documentation

CSS 示例见 [多元素 CSS](benchmarks/css/README.md)，化学加权接口见 [WEIGHT_CHEM](docs/manuals/CV_Weight_Chem.zh.md)。

详细参数配置与命令说明请参阅：[用户使用手册](docs/manuals/User_Manual.zh.md)

---

## 🤝 贡献与反馈

如果有任何问题或建议，请联系 **ZQC**。
gz1999zqc@163.com