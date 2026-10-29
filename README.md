# Metad-toy: LAMMPS Metadynamics Plugin

2026-10-28 统一版本（2026-10-29 收口）：[当前使用入口与验收边界](docs/RELEASE_20261028.zh.md)。
旧功能恢复、LocalQL 优化、Q/Half 设备路径、共享 SUM/在线 WT 与 A100 多卡续算已完成相应验收；
真实 Al3Sc 形核及 FES 收敛仍需独立科学证据。

普通 LAMMPS（不编译 Kokkos/Weave）同样受支持：2026-10-29 已在独立普通宿主上验收，
见 [普通版构建、数据路径与回归范围](docs/PLAIN_LAMMPS_20261029.zh.md)。MetaDtoy CV 仍使用 CUDA。

2026-10-29 独立复验后的 SUM/safe_div 测试就绪修正：标准 Graph 预设 36/36，
默认并发下 1/2/4 ranks 各重复 50 次全部通过；历史验收与当前证据见
[跨流就绪复验](docs/STREAM_READINESS_20261029.zh.md)。

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

* 支持 `plugin load` 的共享库版 LAMMPS（历史验证：29 Aug 2024；2026-10-28 整合验证：22 Jul 2025）
* CMake 3.21+（使用构建预设时）
* CUDA Toolkit (用于 `.cu` 源码编译)
* MPICH 或 OpenMPI

### 编译步骤

构建已按 view、Graph、CV、LAMMPS 适配器和测试分层，独立目标与预设见 [分层构建指南](cmake/README.md)。2026-10-28 统一了主工作区与 Weave/Kokkos 联合实现；版本来源、推荐构建和当前验证边界见 [版本整合记录](docs/VERSION_CONSOLIDATION_20261028.zh.md) 与 [本轮验收](docs/INTEGRATION_VALIDATION_20261028.zh.md)。不要凭旧 build 目录中的插件判断当前源码状态。

1. **准备依赖**：安装 CUDA Toolkit（含 NVRTC）、xxd，以及与 LAMMPS 一致的 MPI。Graph 使用 third_party/fmt 中随项目分发的头文件和许可证。下面的基础构建关闭可选 ML CV。

2. **构建插件**：根据安装位置修改路径，并将 `89` 换成目标 GPU 架构。
```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release -DENABLE_DEBUG=OFF -DUSE_ML_CVS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++ \
  -DLAMMPS_SOURCE_DIR=/path/to/lammps/src \
  -DLAMMPS_LIB_DIR=/path/to/lammps/lib \
  -DMPI_HOME=/path/to/mpi
cmake --build build -j4
```

3. **加载插件**：
`fix_crystallize_plugin.so`将在本项目下的`build`文件夹中生成，可以将它复制到任何地方使用。
编译完成后，在 LAMMPS 脚本中使用 `plugin load path/to/fix_crystallize_plugin.so` 加载。

### 安装后自检

```bash
python3 benchmarks/run.py smoke --build-dir build --lammps lmp
# 多 rank 检查
python3 benchmarks/run.py smoke --build-dir build --lammps lmp --np 1 2 4 --mpi mpirun
```

快速自检只需要 Python 标准库和 LAMMPS 可执行文件；自动生成初态、检查当前非 ML CV 的前后向，并保存日志和通过/失败结果。完整有限差分、Graph CTest 和 O₂ 重建的入口见 [benchmarks 指南](benchmarks/README.md)。

---

## 📝 LAMMPS 输入参数

使用 `fix metad` 指令调用该插件。

### 语法示例

```lammps
fix metad all metad EXECUTOR GRAPH GAUSSIAN 0.01 0.005 10.0 &
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
├── benchmarks/         # 安装自检、有限差分和采样示例
└── test/               # 历史输入与开发实验

```

---

## 🧪 开发与调试

当前版本为 **实验性功能验证版（experimental preview）**。Graph 已接入完整 LAMMPS 流程，支持 Global `DISTANCE`、`STEINH Q/L`、普通/化学 `STRU_FACTOR`、`CSS` 和 `WEIGHT_CHEM`。图接口使用 `EXECUTOR GRAPH`，均值与值开关由图表达式提供。`STEINH Q/L` 统一要求 Graph；原 CV 内部的 AVE／SW_FUNC 标量接口已移除，`SW_FUNC_r` 继续用于径向权重。

更新内容、接口迁移和验证范围见 [更新日志](CHANGELOG.md)。独立测试入口见 [Graph 测试](benchmarks/graph/README.md)，完整流程示例见 [Stru_factor 接入说明](benchmarks/strufactor_lammps_graph/README.md)。

---

## 使用文档 / Documentation

- [安装与测试指南](benchmarks/README.md)
- [用户使用手册](docs/manuals/User_Manual.zh.md)
- [非 Local Q 的 Graph 接入](benchmarks/steinhardt_graph/README.md)与 [Local Q 接入](benchmarks/localql_graph/README.md)
- [多元素 CSS](benchmarks/css/README.md)与 [WEIGHT_CHEM](docs/manuals/CV_Weight_Chem.zh.md)
- [CV、线性组合与 Graph 有限差分](benchmarks/finite_difference/README.md)
- [DISTANCE 修复验证](benchmarks/finite_difference/DISTANCE_RESULTS.md)与 [O₂ Morse 重建示例](benchmarks/o2_distance/README.md)

## 贡献与反馈

问题与建议可通过 GitHub issue 或联系 ZQC：gz1999zqc@163.com。

## 偏置层与高维沉积

偏置核心可独立于 LAMMPS/CUDA 构建，支持 1–3D dense/sparse 网格和任意维 HILLS/PBMetaD。默认 AUTO 在 D>3 选择 PBMetaD，可显式选择后端。接口、输入与可运行测试见 [偏置层说明](benchmarks/highdim/README.md)，正确性和性能数据见 [验证记录](benchmarks/highdim/RESULTS.md)。

## 离线命令行分析

CLI 也可通过 CMake 安装为 `metadtoy` 命令：

```bash
cmake --preset cli -DPython3_EXECUTABLE=/path/to/python -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build --preset cli
cmake --install build/cli --component cli
export PATH="$HOME/.local/bin:$PATH"
metadtoy --help
```

安装会在前缀内管理独立的 Python 环境和分析依赖。随 MD 构建时，可增加
`-DZQC_BUILD_CLI=ON`。Python 打包方式、离线安装及源码入口见 [CLI 使用说明](tools/README.md)。



统一入口为 `python tools/metadtoy.py hills ...`，支持 HILLS 参数检查、沉积记录 CSV 导出、
1D/2D 偏势或 WTMetaD 自由能估计的 CSV/PNG 预览，以及高维偏势的显式切片。
原生 LAMMPS/GPUMD HILLS 共用读取层；PBMetaD 联合 FES 需要轨迹重加权，不能仅由 hills 恢复。
默认 FES 使用逐帧重加权，可通过 `python tools/metadtoy.py fes reconstruct COLVAR ...` 调用；原 HILLS 渐近估计需 `--method hills`。支持原生 WT 的离线 c(t) 及已给校正权重，输出包含逐帧权重与 ESS。
安装、示例和后续轨迹/训练接口规划见 [CLI 使用说明](tools/README.md)，数值对照见
[CLI 验证](benchmarks/cli/README.md)。

### LAMMPS CUDA/Kokkos

可选使用 lammps-kokkos 预设；输入仍可写 fix metad，-sf kk 选择设备桥接。
数组同步、构建依赖、旧版 LAMMPS 补丁及测试命令见 [Kokkos 接入指南](benchmarks/kokkos/README.md)。
