# 安装后验证与 benchmarks

建议先执行快速自检，再选择数值回归或重建示例。下面的命令在仓库根目录运行，也可以用 `benchmarks/run.py` 的绝对路径从其他目录运行。

## 1. 快速自检：先确认插件可以工作

```bash
python3 benchmarks/run.py smoke --build-dir build --lammps lmp
# 单独复制了插件时：
python3 benchmarks/run.py smoke --plugin /path/fix_crystallize_plugin.so --lammps /path/lmp
# 检查多 rank：
python3 benchmarks/run.py smoke --build-dir build --lammps lmp --np 1 2 4 --mpi mpirun
```

只需要 Python 3.9+ 标准库、支持 `plugin load` 的 LAMMPS 可执行文件、已编译插件及其运行库；逐原子图需要 NVIDIA GPU、CUDA driver/runtime 和 NVRTC。1 rank 直接启动 LAMMPS，多 rank 才需要 MPI launcher。这里不要求 LAMMPS Python 模块、numpy、mpi4py 或编译头文件。

脚本自建 32 原子初态，覆盖 DISTANCE、Q3/4/6、Local Q3/4/6、普通/化学结构因子、WEIGHT_CHEM 和三元素 CSS。检查插件加载、NVRTC、前后向、非零偏置力、有限数值、距离参考值，以及不同 rank 下的 CV/偏势/力/virial 一致性。4 个场景包含多个 CV；它是安装检查，逐 CV 的导数精度用下一节的回归验证。

每次运行有独立目录，默认在 `build/benchmarks/<suite>-<timestamp>/`；`--build-dir` 会改变此位置。`--output` 可指定尚不存在的目录。保存输入、初态、LAMMPS 日志、原子力及 `summary.json`。成功退出码 0，测试失败为 1，路径或参数错误为 2。进程失败、超时和数值失败都计入失败，不静默跳过。`summary.json` 记录实际命令、插件 SHA-256、构建信息和逐项结果。

## 2. 有限差分：验证力与 virial

使用能导入本机 LAMMPS 的 Python：

```bash
python benchmarks/run.py regression --build-dir build --np 1 2 4 \
  --mpi mpirun --mpicxx mpicxx --lammps-src /path/lammps/src
```

需要同一 MPI/LAMMPS 安装对应的 `lammps`、`mpi4py`、numpy、MPI C++ 编译器和 LAMMPS 头文件。launcher 与 Python 模块必须使用兼容的 MPI；测试加载的 Python LAMMPS 库也应与插件构建时的一致。若环境中有多套工具，请显式指定路径并选对 Python。

该入口执行 [finite_difference/run.py](finite_difference/run.py) 的所有 23 个场景；默认 1 rank，共 23 组，`--np 1 2 4` 为 69 组。每组扫描四个差分步长，检查 CV 梯度、偏置力和六分量应变/virial；另检查 CV 线性组合的梯度叠加。结果包含逐步长误差、阈值和失败明细。

更完整的迁移、空 rank、非连续 ID 和异常输入测试见 [DISTANCE 验证](finite_difference/DISTANCE_RESULTS.md)；Q/Local 的组变更和邻居排除测试见 [steinhardt_graph](steinhardt_graph/README.md)、[localql_graph](localql_graph/README.md)。

## 3. 独立 Graph：验证 NVRTC 与图反向

```bash
cmake -S . -B build/graph -DZQC_GRAPH_STANDALONE=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build/graph -j4
python3 benchmarks/run.py graph --graph-build-dir build/graph
```

将 `89` 换为目标 GPU 架构。需要 CUDA Toolkit、xxd；fmt 头文件随项目分发。默认 MPI 测试另需 MPI，`-DZQC_TEST_MPI=OFF` 可只保留单进程验证。此模式不查找 LAMMPS 头文件或库，也不要求 LAMMPS Python 模块。入口执行全部已配置的 CTest，默认包含固定的 1/2/4 rank 场景；`--np` 只用于 smoke/regression。需要指定工具时，可设置 `MPI_CXX_COMPILER`、`MPIEXEC_EXECUTABLE`、`MPIEXEC_PREFLAGS`，入口支持 `--ctest /path/ctest`。纯 C++ 接口验证可用 `cmake --preset host-tests`；完整目标列表见 [分层构建指南](../cmake/README.md)。

## 4. O₂：完整偏置采样与曲线重建

```bash
python benchmarks/run.py o2 --build-dir build --lammps lmp
```

需要 numpy、matplotlib 和 LAMMPS 可执行文件，采用 1 rank。自动跑三条 2 ns 的经典 Morse 示例，读取 HILLS 重建相对势能，并生成对照图。验证采样覆盖、每条曲线 RMSE < 25 meV、最大误差 < 75 meV、势阱位置误差 < 0.01 Å；失败返回非零退出码。这里的统计阈值比本机测量误差宽，适用于默认长轨迹。更短的自定义采样可能尚未收敛。

模型、well-tempered 缩放、径向熵修正与结果边界见 [o2_distance](o2_distance/README.md)。

## 路径与 MPI 参数

- 插件：显式 `--plugin` → `METAD_PLUGIN` → `--build-dir` 下的插件；保留 `build/graph-lammps` 的开发目录回退。
- LAMMPS：显式 `--lammps` → `LAMMPS_EXECUTABLE` → 插件旁 CMakeCache 记录的安装目录 → PATH 中的 `lmp` / `lmp_mpi`。
- MPI：显式 `--mpi` / `--mpicxx` → `MPIEXEC` / `MPICXX` → 构建缓存的 MPI_HOME 或 PATH。高级差分脚本支持同名参数。
- 头文件：显式 `--lammps-src` → `LAMMPS_SOURCE_DIR` → 插件旁 CMakeCache。
- OpenMPI 的绑核/超额订阅参数按机器需要传入，例如 `--mpi-args="--bind-to none --oversubscribe"`；默认不注入 OpenMPI 特有参数。集群调度及 GPU 分配遵守本机作业设置。

`smoke --help` 等可查看参数。若缺少工具，先修复提示的路径；若 NVRTC 或 MPI 运行失败，检查对应 case 的 `screen.txt` / LAMMPS 日志。运行回归前，应先让 smoke 通过。

## 历史性能实验与发布验证

当前缓存/流改造的版本 A/B 和 Leaf 非连续暂存测量入口见
[runtime_performance](runtime_performance/README.md)。它接收多个明确的插件路径，
先做数值核对，再分开记录 NVRTC 初始化和稳定 timestep 的耗时。
与上传提交 `2d9ac78` 的直接比较及 Local Q 回退定位见
[上传版本复测](runtime_performance/UPLOADED_COMPARISON.md)。

带日期的目录及 `cv_communication`、`weighted_chem_switch` 等保留开发过程的专项比较，部分要求历史插件、特定初态或显式本机路径。安装用户优先使用上面四个入口。历史计时受硬件、编译选项和版本影响，不作为跨机器安装通过标准。

本轮发布检查见 [INSTALLATION_RESULTS.md](INSTALLATION_RESULTS.md)。所有新运行的原始数据默认写入被 Git 忽略的 build 目录；仓库只保留代码、文档和精选数值记录。

## 高维偏势沉积

[HILLS / PBMetaD 的输入、重启与验证](highdim/README.md) 提供任意维列表后端、
逐维宽度/周期核，以及 4/8 维完整 LAMMPS 数值与重启测试。

## GPUMD 接入验证

[GPUMD DISTANCE / StruFactor + O₂ FES](gpumd/README.md) 提供可选 EnhancedSampling 桥接构建、
数组/流与周期像近邻契约测试、StruFactor 力/virial 有限差分、与原生谐和偏置的交叉核对，以及真实 GPUMD 积分中的 Morse 盲重建。
该入口独立于 LAMMPS，所有新运行结果写入单独的 build 目录。

## Kokkos 兼容性

[Kokkos 验证](kokkos/README.md)覆盖普通/CUDA 数值交叉、Full/Half 邻居请求、
连续积分与原子迁移、力/virial/compute 输出，以及旧版 LAMMPS trim 回补。

## 冻结共享原子偏势

[Al3Sc 验证](shared_frozen/README.md)覆盖逐原子共享表、MEAN 聚合、GPU 数组种子、
普通/Kokkos 与 MPI 数值交叉、重启重放及 LocalQ6×CSS 的力/virial 有限差分。

## 共享偏势概率验证

[shared_frozen/PROBABILITY.md](shared_frozen/PROBABILITY.md) 提供参考概率已知的耦合双阱测试，
检查固定/学习后冻结的共享偏势重加权、统计误差与采样覆盖，并与生产 CUDA 求值及 CLI 交叉核对。

## CV 压缩与 FES

[shared_frozen/COMPRESSION.md](shared_frozen/COMPRESSION.md)包含平坦原势能的解析控制、
相关双阱的共同沉积和原坐标 FES 图。对压缩导致的权重退化同时报告有效覆盖与未达标项。
