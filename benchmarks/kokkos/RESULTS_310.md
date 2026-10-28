# CUDA/Kokkos 验证记录（310，2026-10-27）

本次实现支持 NVIDIA/CUDA 的 LAMMPS Kokkos 数组接入，同时保留普通 LAMMPS 路径。
验证全部在用户指定的 310 节点直接运行，未使用 Slurm。

## 环境与范围

- GPU：NVIDIA RTX 4090 D，单卡；MPI 分别为 1/2 rank。
- CUDA 12.2.91；Kokkos 4.3.1；LAMMPS stable_29Aug2024。
- Host/CUDA Host compiler 统一为 devtoolset-9 GCC，C++17，旧 libstdc++ 字符串 ABI。
- MPI：MPICH 3.0.4；测试使用 gpu/aware off。
- 宿主、插件和所有探针的编译器/MPI ABI 一致。
- 隔离工作目录：/home/saitang/work/MetadToy-kokkos-20261027。
- 多 GPU、多节点、GPU-aware MPI、HIP/SYCL，以及其他 LAMMPS/Kokkos 版本未在本轮验证。

## 结果

| 验证 | 结果 |
| --- | --- |
| 普通插件构建（Kokkos 依赖关闭） | 通过 |
| CUDA/Kokkos 插件构建 | 通过 |
| 所有非 ML CV、线性组合、混合组、非线性 Graph、DISTANCE 有限差分 | 69/69 最终配置通过 |
| 普通与 Kokkos 状态交叉：坐标、力、CV、bias energy、virial | 最大绝对差 7.549516567451064e-14 |
| Host/full/half/half-off × 1/2 rank，200 步连续积分 | 8/8 通过 |
| 连续轨迹跨后端比较 | 最大绝对差 8.881784197001252e-15 |
| 实际 Half/Full 邻居请求契约 | 324 项通过，最大误差 4.433786671143025e-12 |
| 最终连续积分 CUDA compute-sanitizer memcheck | 0 errors |
| 旧版 trim 源码检测 | 未修复源码拒绝，回补源码接受 |

69 个配置包含普通模式 23 个、Kokkos 模式 46 个。有限差分覆盖 96 个坐标分量
（单 rank）或跨域抽样（双 rank），并验证 6 个应变分量。两种步长为 1e-5 和 3e-6。

LEGACY 解释器使用 exprtk 的数值导数。原测试把其力/virial 的恢复也要求到 1e-10，
在普通和 Kokkos 两端均出现约 2.2e-8 的舍入波动；两端的导数差分检查均通过。
测试现对 LEGACY 力/virial 恢复采用既有导数容差，坐标和值仍要求 1e-10，
Graph 反向的严格恢复要求保持原样。6 个 LEGACY 配置已按该标准重新执行。

连续积分使用非零 lj/cut 势和固定偏置，独立计算 LJ 力作为初始参考。
检查实际 owner 列表变化、周期边界迁移、排序、原子类型 2→3、CV 绑定组改变、
compute 单列/多列/组过滤、逐原子 stress 与总 virial、一致的重复 run 0。
Kokkos 组的 position_uploads 均为 0；Host 对照大于 0。

Half/Full 测试包含 Newton on/off、正交/倾斜盒、小盒周期自镜像、空组、
相交与不相交组、组变化、候选缓存和力/virial reverse。

## 验证中发现并处理的问题

1. 插件重复链接静态 Kokkos 会在退出时重复释放全局对象。现只使用宿主运行时，
   并在链接时要求全部符号可解析；独立 Python worker 显式 finalize Kokkos。
2. nvcc_wrapper 的 Host compiler 随环境变化，曾导致 GCC 的字符串 ABI 混用。
   本轮构建固定 C++ 与 CUDA Host compiler；原生 CUDA 编译路径关闭 Kokkos 的全局 launcher。
3. 普通坐标上传 View 原先遗漏就绪事件，现从 GpuBuffer::view() 取得等待钩子。
4. 删除未使用的 C++20 source_location 头文件，保持项目的 C++17 编译要求。
5. compute 插件工厂按 LAMMPS 三参数 ABI 注册，并支持绑定 metad/kk 和 metad/kk/device。

## 旧 LAMMPS trim 缺陷

在原始 29Aug2024 下，较大 CV Full 表被用于生成 pair Half 表时，
零 custom cutoff 使 pair 邻居表被清空。加入真实 LJ 积分后，
普通与 Kokkos 的初始力相差约 0.1699；独立 Kokkos pair 表保持了正确相互作用。

提供的三文件回补在 custom cutoff=0 时使用按类型的 cutneighsq，
与当前上游的回退语义一致。回补后，普通/Kokkos 的连续轨迹恢复一致。
用户原有 WSL LAMMPS 未修改；仅 310 的隔离宿主用于修复验证。

独立复现文件 trim_reproducer.cpp / trim_reproducer.in 只申请 Full 邻居表，
不包含 MetaD 计算或力写入。回补后两原子 LJ 能量：

- 添加 Full 请求前：-0.8909652875830758
- 添加 Full 请求后：-0.8909652875830758
- 删除 Full 请求后：-0.8909652875830758

当前上游已修复：[half/full](https://github.com/lammps/lammps/blob/develop/src/npair_halffull.cpp)、
[skip](https://github.com/lammps/lammps/blob/develop/src/npair_skip.cpp)。
因此本轮没有新发 issue。源码检测不能代替重编译宿主二进制，
生命周期测试中的独立 LJ 参考负责继续检查实际运行的库。

## 证据

- results/comparison_310.json：46 个跨模式状态比较和 8 组轨迹比较。
- results/neighbors_310.json：邻居测试数量、最大误差和插件 SHA256。
- results/guard_rejected_310.log：旧版源码被拒绝的配置日志。
- results/memcheck_310.log：最终连续积分 CUDA 内存检查。
- results/finite_difference_310.json：按最终恢复标准合并的 69 个配置结果。
- 原始 finite-difference、lifecycle 日志保存在上述 310 隔离目录。

本轮测量用于正确性和兼容性验收，没有据此宣称整体性能提升。
Kokkos 插件 SHA256：025b664853663f679acc09d01d573c0469bc8d654b8f8f9c3b01dedb29dac48f。
回补后的 liblammps SHA256：9c73a3ec4f224d62b5106565902cb1fbfccf16a90351117614488546ad2dbd85。
