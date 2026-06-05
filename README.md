# Metad-toy: LAMMPS Metadynamics Plugin

**Metad-toy** 是一个基于 LAMMPS 插件机制开发的元动力学 (Metadynamics) 插件。它允许用户在 CUDA 加速的环境下，通过自定义集体变量 (CVs) 来驱动复杂系统的相变或化学反应模拟。

---

## ✨ 功能特性

* **多维网格加速**：计划支持 1D、2D 及更高维度的势能网格，通过网格化存储高斯函数。
已完成：1D
* **Cubic 插值**：使用三次插值算法计算网格梯度，确保原子受力的连续性，提高积分稳定性。
* **多种 CV 支持**：内置 `DISTANCE` (距离) 和 `STEINH` (Steinhardt 序参数) 等集体变量。
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

1. **设置环境变量**：
```bash
export LAMMPS_SOURCE_DIR=/path/to/lammps/src
export LAMMPS_SOURCE_DIR="/home/zqc/apps/lammps/lammps-stable_29Aug2024/src/",
export LAMMPS_LIB_DIR="/path/to/lib/",
export MPI_INCLUDE_PATH="/path/to/mpich-4.1.2",
export MPI_HOME="/path/to/",
export CLEAN_HOST_COMPILER="/usr/bin/g++",
export CMAKE_BUILD_TYPE="Release",
export CMAKE_EXPORT_COMPILE_COMMANDS="true",
export METAD_PLUGIN_PATH="${workspaceFolder}/build"

```


2. **构建插件**：
```bash
mkdir build && cd build
cmake ..
make -j 4

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
                    SIMBOL v1 Q6.AVE &
                    SIMBOL v2 Q4.AVE &
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
| `SIMBOL` | `[variable name] [cv_set_name].[cv_func]` | 指定cv计算的变量名所对应的cv计算函数方法 |
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

---

## 使用文档 / Documentation

详细参数配置与命令说明请参阅：[用户使用手册](docs/User_Manual.md)

---

## 🤝 贡献与反馈

如果有任何问题或建议，请联系 **ZQC**。
gz1999zqc@163.com