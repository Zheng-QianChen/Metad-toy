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
fix 1 all metad \
   CV_dim 1 PACE 100 \
   GAUSSIAN 0.05 0.1 10.0 \
   DISTANCE 1 2 \
   DIM 1 2.0 8.0 500 \
   WT 1

```

### 参数说明

| 关键字 | 参数 | 描述 |
| --- | --- | --- |
| `CV_dim` | `[N]` | 集体变量的总维度。 |
| `GAUSSIAN` | `[sigma] [height] [biasf]` | 高斯宽度、初始高度和偏置因子。 |
| `PACE` | `[N]` | 每隔 N 步添加一次高斯函数。 |
| `DISTANCE` | `[ID1] [ID2]` | 定义两个原子之间的距离作为 CV。 |
| `STEINH` | `[Q/L] [4/6/8/12] [group]` | 定义 Steinhardt 序参数作为 CV。 |
| `DIM` | `[idx] [low] [up] [bins]` | 设置特定维度的网格范围和精度。 |
| `WT` | `[0/1]` | 是否启用 Well-tempered Metadynamics。 |

---

## 📂 项目结构

```text
.
├── src/                # 核心实现 (.cu 和 .cpp)
├── include/            # 头文件 (.h)
├── CMakeLists.txt      # 构建脚本
└── test/               # 测试用例 (DISTANCE/STEINH)

```

---

## 🧪 开发与调试

---

## 🤝 贡献与反馈

如果有任何问题或建议，请联系 **ZQC**。
gz1999zqc@163.com