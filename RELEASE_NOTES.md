# MetaDtoy v0.1.0

发布日期：2026-10-29。本版本整合主库与统一工作树，支持普通 LAMMPS、LAMMPS Kokkos/CUDA，
并提供部分 GPUMD 接入。源码版本以 Git 标签 `v0.1.0` 为准，构建入口读取根目录 `VERSION`。
这是按下述范围发布的工程版本；真实材料的形核机制、自由能收敛和速率需要各自的科学验证。

## LAMMPS

- 支持普通共享库版 LAMMPS，通过 `plugin load` 加载插件；不要求编译 Kokkos、GPU package 或 Weave。
  普通宿主上传坐标与 Host 邻居 CSR，CV、Graph 及偏势计算仍使用 CUDA。
- 支持 LAMMPS Kokkos 的 CUDA 后端，通过设备 View 借用坐标/类型及设备邻居表，
  完成 CV 前后向、偏置力与 virial 回写；普通适配器路径保留。
- Weave 批通信是可选优化；提供普通 Host、Kokkos、Kokkos+Weave 独立构建配置。
- 包含 DISTANCE、Steinhardt Q/LocalQL、StruFactor/CSS、化学加权及 LocalGeom 等现有非 ML CV，
  支持 Graph 组合、SUM/AVE/safe_div、力与 virial 导数。
- 支持全局 MetaD/WT、HILLS 与续算；共享原子偏势支持 SUM/MEAN、在线沉积、WT、校验历史和续算。
  总能量归约与每次沉积高度归约独立；默认 `SHARED_DEPOSIT MEAN` 将一次总基础高度按全局中心分摊。

主要统一验收宿主为 LAMMPS 22 Jul 2025。历史 29 Aug 2024 的构建需通过 trim 兼容守卫，
不能将“历史上跑过”理解成任意未打补丁版本均可使用。插件、宿主、MPI、CUDA 和 Kokkos
必须使用匹配的 ABI；不能把旧目录里同名 `.so` 当作本版插件。

构建与实际范围：[普通 LAMMPS](docs/PLAIN_LAMMPS_20261029.zh.md)、
[Kokkos 接入](benchmarks/kokkos/README.md)、[三种集成构建](benchmarks/integration/README.md)、
[共享偏势手册](docs/manuals/SHARED_BIAS.zh.md)。

## GPUMD：部分支持

接入目标为 GPUMD v5.9.1。构建系统在 build 目录中准备上游副本，通过 EnhancedSampling
薄桥接调用 `libmetadtoy.so.1`；公开 C ABI 版本为 1。无需把 GPUMD 完整源码纳入本仓库。

已接入：

- 单进程、固定原子顺序、正交固定盒；NVE 与 NVT_LAN/NVT_NHC/NVT_BDP。
- DISTANCE、StruFactor、指定固定中心组、Graph 表达式及一维网格 WTMetaD。
- 设备数组与流依赖、力/势能/九分量 virial 累加，独立周期像邻居和候选缓存。
- 独立运行库构建/安装、C ABI 检查、真实 GPUMD DISTANCE 与 StruFactor 数值回归。

尚未接入：NPT、完整 MD 状态重启、固定/移动组控制、GPUMD 原生 group 编号映射、
LocalQL 所需的 ghost 中心环境，以及 LAMMPS 全部 CV/共享偏势配置。
`start fresh` 只表示新建偏置历史，不能视为 MD 重启功能。
邻居重建目前含 CPU 空间分箱；本版不承诺 GPUMD 的全设备邻居或大体系加速。

配置、可复现测试与限制见 [GPUMD 指南](benchmarks/gpumd/README.md)。
此前真实 GPUMD 数值与内存检查的阶段记录见
[共享库验收](benchmarks/gpumd/RESULTS_SHARED_310.md)和
[StruFactor 验收](benchmarks/gpumd/RESULTS_STRUFACTOR_310.md)。

## 本版修正及证据口径

- LocalQL 删除无用逐边球谐缓存并借用设备 CSR，普通 LAMMPS 的上传 CSR 路径同样保留。
- SUM/safe_div MPI 测试修正 pageable H2D 上传的完成依赖；增加默认流/非阻塞流的延迟生产者回归。
  默认并发下 SUM 的 1/2/4 ranks 各重复 50 次全部通过；AVE 共 60 次、流契约共 20 次通过。
  `MAX_CONNECTIONS=32` 的 SUM 附加对照另有 60 次通过，未用串行诊断环境替代默认验收。
- 标准及全新 Graph 预设均 36/36，Host 6/6。早期 34/34 不包含新增 SUM 三条；
  恢复当晚 37/37 为历史记录，当前复验单独列出。详见 [跨流就绪复验](docs/STREAM_READINESS_20261029.zh.md)。
- 历史 A100 长运行、多卡/双节点与同机性能数据保留原平台和测试口径；本次发布不把它们
  改称为发布当天重新测得的结果。约 300 ms/步等其他平台数字不能直接移用。
- 科学 benchmark 保留失败、覆盖率和不确定度；控制模型的 FES 恢复不替代真实 Al3Sc 验收。

发布前合并检查与当前补充验证见 [发布验证记录](docs/RELEASE_VALIDATION_v0.1.0.zh.md)。
完整历史与来源见 [更新日志](CHANGELOG.md)和[统一工作记录](docs/RELEASE_20261028.zh.md)。

## 构建入口

需要 Linux/WSL、C++17、CUDA Toolkit（含 NVRTC）、匹配的 MPI 和宿主源码/库。
预设使用 CMake 3.24+ 的 `native` GPU 架构；在构建文件中显式指定目标架构可用于其他环境。
普通 Host、Kokkos 与 Weave 使用独立目录，按 [集成构建](benchmarks/integration/README.md) 的实际依赖配置。

```bash
# Host 单元测试不需要 CUDA；并不代表 CV 可在无 GPU 环境计算。
cmake --preset host-tests
cmake --build --preset host-tests --parallel 4
ctest --preset host-tests

# CUDA Graph / NVRTC / MPI 回归。
cmake --preset graph-tests
cmake --build --preset graph-tests --parallel 4
ctest --preset graph-tests

# 可选 GPUMD 全程序与桥接验证，需要用户自行准备 GPUMD v5.9.1 源码。
cmake --preset gpumd-tests -DZQC_GPUMD_SOURCE=/absolute/path/to/GPUMD
cmake --build --preset gpumd-tests --parallel 4
ctest --preset gpumd-tests --output-on-failure
```

源码发布包不附带特定机器的 LAMMPS/GPUMD 可执行文件、CUDA/MPI 库、大型轨迹或构建缓存。
请针对宿主和目标 GPU 重新构建；GPUMD C ABI/SONAME 版本与本项目发布版本分别管理。
