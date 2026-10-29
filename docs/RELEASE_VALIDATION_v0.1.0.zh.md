# v0.1.0 发布验证 — 2026-10-29

本页记录发布前本机重新执行的构建和测试。历史恢复、A100 生产运行、性能和科学控制模型
各保留原始日期及平台，不能合计成“本次发布全部重新测过”。功能范围以
[v0.1.0 版本说明](../RELEASE_NOTES.md)为准。

## 主库与统一版的合并

GitHub `main` 的合并前基线为 `2d9ac7801ed7f093d5e83312f707e2167a337595`，
本地主库为 `785d1ae`；二者均是统一版 `68d6dfc` 的祖先。
本次保留全部已有提交，以快进方式把统一成果纳入本地主库，再通过发布分支上传。

主库工作区的 34 个未提交新文件与统一版逐文件 SHA256 相同，涵盖科学 benchmark 源码、
图和整理后的结果；它们已在统一历史中。其余 6 个已修改文件为开发记录、说明或较早构建清单。
合并前保存全部 40 个文件的原样归档、diff 与哈希；原主库改动另以 Git stash 留档。
统一版保留较新的 SUM/在线共享测试清单和科学边界说明，并补回科学控制模型的历史更新记录。

本次发布元数据基于 `68d6dfc`，新增统一 `VERSION=0.1.0`，CMake 读取该值；
CLI 版本为 0.1.0。生产计算实现及现有数值容差没有因发布而改变。

## 当前执行结果

| 配置 / 用例 | 本次结果 | 覆盖 |
|---|---|---|
| Host 预设 | 6/6 | 无 CUDA 的基础接口和数学层测试 |
| Graph 预设 | 36/36 | CUDA、NVRTC、MPI、SUM/AVE/safe_div、流契约与共享偏势 |
| 普通 LAMMPS 插件重建 | 40/40 CTest | 完全无 Kokkos 宿主，另含 CV 和 GPUMD 运行库独立测试 |
| Kokkos 插件重建 | 40/40 CTest | 匹配的 Kokkos/CUDA 宿主，无 Weave |
| Kokkos+Weave 插件重建 | 40/40 CTest | 匹配的 Kokkos/CUDA 宿主及可选 Weave |
| 普通原生 lmp smoke | 12/12 | 1/2/4 ranks × DISTANCE、Q、LocalQL、化学/CSS 组合 |
| Kokkos 真实 Al–Sc MEAM | 6/6 | 1/2 ranks、LocalQL×CSS、坐标/应变差分、门控、空中心与实际续算 |
| Kokkos+Weave 真实 Al–Sc MEAM | 6/6 | 同上，实际加载 Weave 插件 |
| 全新 GPUMD v5.9.1 构建 | 5/5 | 邻居、EngineAdapter、C ABI、真实 DISTANCE/WT、真实 StruFactor |

不同配置会重复执行相同测试，表中各项不能当成互不重复的独立测试总数。
GPUMD 两项端到端测试实际运行新构建的 `gpumd_metadtoy` 与 `libmetadtoy.so.1`，
DISTANCE 包含 2000 步 WT 沉积；StruFactor 包含坐标力、对角应变 virial 差分和缓存/重建对照。
这比单独编译桥接头文件的覆盖更广，但没有新增长程 FES 或大体系性能结论。

本机环境：WSL Ubuntu 22.04、RTX 4080 SUPER、driver 610.88、CUDA 12.6.20、
GCC 11.4、OpenMPI 5.0.6；LAMMPS 为 22 Jul 2025。
GPUMD 上游为 v5.9.1，tree `11ba6fbdb4d71b2cdb351b39972fa057521585b3`。
所有上述运行显式移除了 `CUDA_DEVICE_MAX_CONNECTIONS` 和 `CUDA_LAUNCH_BLOCKING`。
MPI 使用 `self,tcp` / `ob1`，固定匹配的库路径及每进程单线程。

## 可追溯数据及复现

仓库中的 [机器可读验证清单](releases/v0.1.0-validation.json) 保存逐项测试名、插件 SHA256、
测试/运行输入文件 SHA256 和范围。`source_inputs_sha256` 排序规则写在 JSON 内；
排除 Markdown、图片与 docs，以避免文档和验证清单自引用。
最终发布提交由 `v0.1.0` 标签固定，GitHub Release 附带原始构建/CTest 日志、XML、
宿主数值结果摘要及先前跨流隔离/稳定性记录；附件不含凭据或机器专用运行库。

源码构建见 [集成指南](../benchmarks/integration/README.md)和
[GPUMD 指南](../benchmarks/gpumd/README.md)。本次使用全新的
`build/release-v0.1.0/gpumd`，现有三个插件构建目录分别重建；
没有把旧阶段冻结插件替换为发布时生成的新文件。

本次补充运行没有再次执行 A100 长运行/多节点性能、GPUMD 长程 FES 或真实 Al3Sc
科学收敛验证。此前默认并发 SUM 150 次等稳定性记录见
[跨流就绪复验](STREAM_READINESS_20261029.zh.md)，与本页当前构建矩阵分开记录。

原始证据和主库合并前备份保存在
`C:/Users/Administrator/.codex/reports/metadtoy-release-20261029/`；
大型构建、可执行文件和完整轨迹留在本地，不纳入源码 Git 历史。
