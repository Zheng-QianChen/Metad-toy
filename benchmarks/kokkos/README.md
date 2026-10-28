# LAMMPS CUDA/Kokkos 接入与验证

基线：LAMMPS stable_29Aug2024、其自带 Kokkos 4.3.1、CUDA 12.2、double 精度。
当前适配的设备后端是 NVIDIA/CUDA。普通 LAMMPS 的构建与运行入口保留。

## 构建

用同一套 Host 编译器构建 LAMMPS、插件 C++ 文件、CUDA Host 部分和测试探针。
尤其要保持 `_GLIBCXX_USE_CXX11_ABI`、LAMMPS 整数宽度、Kokkos 配置与宿主一致。

```bash
cmake --preset lammps-kokkos \
  -DLAMMPS_SOURCE_DIR=/path/to/lammps/src \
  -DLAMMPS_LIB=/path/to/lammps-build/liblammps.so \
  -DKokkos_DIR=/path/to/lammps-build/cmake_packages/Kokkos \
  -DCMAKE_CXX_COMPILER=/path/to/matching/g++ \
  -DCMAKE_CUDA_HOST_COMPILER=/path/to/matching/g++
cmake --build --preset lammps-kokkos -j 4
```

Kokkos_DIR 必须来自该 liblammps 的实际构建或安装。插件只使用这份 Kokkos 的
头文件、配置宏和 liblammps 已导出的运行时符号，避免静态链接第二份 Kokkos。
插件的 CUDA 源码由 CMake CUDA 编译器编译；Kokkos 的全局 compiler launcher 关闭。

普通构建使用 `lammps` 预设，或 `-DZQC_LAMMPS_KOKKOS=OFF`。
带桥接的插件在未启动 Kokkos 的 LAMMPS 中也可使用普通 Host 适配路径。
两种插件都必须与所加载的 liblammps ABI 匹配。

启动示例：

```bash
lmp -k on g 1 -sf kk -pk kokkos gpu/aware off -in in.lammps
```

输入继续写 `fix m all metad ...`；`-sf kk` 选择注册的 `metad/kk`。
也可显式使用 `metad/kk/device`。compute 继续使用 `metad/atom`，
输出由 Host 侧按原子 localid 装配。MPI 测试使用 gpu/aware off；
GPU-aware MPI 和多 GPU 性能需要在目标环境另行验证。

## 数组存储和同步

- 坐标/类型：桥接借用 AtomKokkos DualView 的 Device 数组，保留真实 item/component stride。
  范围是 owned 前缀加 ghost 后缀，每次取 View 重新读取指针和容量，覆盖排序与扩容。
- 输入就绪：在 LAMMPS Kokkos 默认 CUDA execution-space 的流上记录事件，
  消费该 View 的 CV 流通过等待钩子排入依赖。
- group/type/tag/molecule：Host 侧用于组、ID 和邻居筛选，读取前通过 DualView 同步。
  DISTANCE 按 ID 查找端点时使用 Host 坐标。
- 邻居：请求 LAMMPS 的 Host Full/Half 候选表，按拓扑和组 generation 缓存 CSR，
  上传后交给 CUDA CV kernel。LocalQL 的 ghost 行补齐逻辑继续使用。
- 力：CV 输出 owned+ghost 增量，Host MPI reverse 合并后将 owned 增量加到 Kokkos
  Device 力数组；完成后标记 F_MASK 的 Device 副本为最新。现有 pair 力被保留。
- virial：MPI 包仍为 3 项力和 6 项 LAMMPS virial；Fix 在 Host 完成全局/逐原子 tally。
  Full9 输入仍按原有约定投影为 xx/yy/zz/xy/xz/yz。
- Graph codegen 的 kernel 参数仍是连续数组；非连续 Leaf 输入使用暂存。
  Kokkos 坐标借用并不改变这一 ABI。

当前主机邻居筛选、MPI 打包和 owned 力回写仍包含必要的数据搬运。
“坐标上传计数为 0”只表示 CV 直接借用了 Kokkos 坐标，不能据此推断整条链零拷贝。

## 旧版 LAMMPS trim 修复

未修复的 29Aug2024 在 half/full、copy trim、skip trim 中可能把 custom cutoff=0
直接作为裁剪半径。较大的 CV Full 表被用于生成 pair Half 表时，pair 表会被清空，
从而丢失物理相互作用。这是宿主旧实现的问题。

当前上游已使用按类型 cutoff 回退：
[npair_halffull.cpp](https://github.com/lammps/lammps/blob/develop/src/npair_halffull.cpp)、
[npair_skip.cpp](https://github.com/lammps/lammps/blob/develop/src/npair_skip.cpp)。

针对上述旧基线提供 [最小回补](patches/lammps-29Aug2024-trim.patch)：

```bash
cd /path/to/lammps
patch --dry-run -p1 < /path/to/Metad-toy/benchmarks/kokkos/patches/lammps-29Aug2024-trim.patch
patch -p1 < /path/to/Metad-toy/benchmarks/kokkos/patches/lammps-29Aug2024-trim.patch
cmake --build /path/to/lammps-build -j 4
```

CMake 检测已知旧源码组合并拒绝构建。必须实际重编译 liblammps；
仅修改源码不能修复已安装的二进制。检测不代表未知版本、私有修改和 ABI 混用已经验证。
本次只修补了 310 的隔离验证源码，用户原有 WSL LAMMPS 源码与库保持原样。
当前上游已经修复，因此无需重复提交新 bug issue。

## 数值回归

需要与宿主同一 MPI 的 mpi4py、LAMMPS Python 模块、NumPy；
通过 PYTHONPATH 和 LD_LIBRARY_PATH 指向待测模块/库。
下列目录必须是新目录，便于保留每次验证证据。

```bash
python benchmarks/finite_difference/run.py \
  --plugin build/lammps-kokkos/fix_crystallize_plugin.so \
  --output build/kk-fd --np 1 2 --steps 1e-5 3e-6 \
  --lammps-args="-k on g 1 -sf kk -pk kokkos gpu/aware off"
python benchmarks/finite_difference/run.py \
  --plugin build/lammps/fix_crystallize_plugin.so \
  --output build/host-fd --np 1 --steps 1e-5 3e-6
```

覆盖所有已编译非 ML CV、线性组合、混合 group、非线性 Graph、DISTANCE。
Graph 恢复误差要求仍为 1e-10；LEGACY 的 exprtk 数值导数沿用力/virial
差分容差，坐标和 CV 值的恢复仍要求 1e-10。

实际 Full/Half 请求及周期自镜像验证：

```bash
cmake --build build/lammps-kokkos --target zqc_kokkos_neighbors_probe -j 4
python benchmarks/finite_difference/engine_neighbors.py \
  --plugin build/lammps-kokkos/fix_crystallize_plugin.so \
  --probe build/lammps-kokkos/zqc_kokkos_neighbors_probe.so \
  --output build/kk-neighbors --np 1 2 \
  --lammps-args="-k on g 1 -sf kk -pk kokkos gpu/aware off"
```

连续积分测试用相同 MPI/C++ ABI 编译 adapter_probe.cpp，
并链接待测插件（MPI wrapper 的 Host compiler 必须与插件一致）：

```bash
mpicxx -std=c++17 -shared -fPIC benchmarks/kokkos/adapter_probe.cpp \
  -I/path/to/lammps/src -Iinclude/core \
  "$PWD/build/lammps-kokkos/fix_crystallize_plugin.so" \
  -o build/kk-adapter-probe.so
mpiexec -np 2 python benchmarks/kokkos/lifecycle.py \
  --fixture build/kk-fd/nonlinear_graph_np1 \
  --probe build/kk-adapter-probe.so \
  --plugin build/lammps-kokkos/fix_crystallize_plugin.so \
  --output build/kk-lifecycle --newton off \
  --lammps-args="-k on g 1 -sf kk -pk kokkos gpu/aware off neigh full newton off"
```

lifecycle.py 先用独立解析 LJ 力验证 pair 相互作用，再加固定偏置，积分 200 步。
检查真实 owner 迁移、排序、类型/组刷新、单列/多列 compute、逐原子 stress、
重复 run 0，以及 Kokkos 坐标上传次数为 0。替换为 neigh half/newton on、
neigh half/newton off，并去掉 lammps-args 跑 Host 参考；各跑 1/2 rank。
compare.py 接收两组有限差分目录和上述 lifecycle 目录，比较完整状态；
lifecycle 参数的第一项应为单 rank Host 参考。

验证记录见 [RESULTS_310.md](RESULTS_310.md)。