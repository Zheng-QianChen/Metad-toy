# MetaDtoy 版本整合记录（2026-10-28）

本次以主工作区的完整开发状态为底座，吸收另存的 Weave/Kokkos GPU 联合实现。
主工作区原 HEAD 为 `2d9ac7801ed7f093d5e83312f707e2167a337595`，但其代码不代表实际工作区：
当时仍有 52 个已跟踪文件修改和三百余个未跟踪文件。整合前保存的完整快照提交为
`5c2e8a6ef4f01acaf8641a61728cea219bee1b4f`。

## 版本的去向

| 版本 | 处理方式与依据 |
| --- | --- |
| `/home/zqc/work/Metad-toy` | 保留最新 Graph、ArrayView、通用引擎、批次接口、偏置层、CLI、GPUMD 和冻结共享表实现，作为统一底座。 |
| Windows `weave_metad_joint_20261027/metadtoy` 与 WSL `Al3Sc-kokkos-shared-20261028/source/metadtoy` | 相同的 GPU 联合源码快照；吸收设备力提交、Weave reverse 和全局 virial 归约，按最新批次接口重写接合处。 |
| `Metad-toy-al3sc-kokkos-joint-20261027` | 名称不能代表内容：审计时核心文件与主工作区相同，未包含实际通过联合验证的设备补丁。保留作历史副本。 |
| `Metad-toy-localql-css-l12`、`Metad-toy-localql-css-sw-no-wt` | LocalQL×CSS 及 product 开关可用当前 Graph 表达；保留旧实现作迁移参照，不恢复专用类。 |
| `Metad-toy-views-graph/q/radial` | ArrayView 迁移中间状态，已被当前通用接口覆盖。 |
| `build/bias-layer-20261023/grid-worktree` | BiasGrid 早期草稿，当前主工作区包含后续沉积边界等修正。 |
| `build/gpumd-adapter-worktree`、`gpumd-dynamic-review`、`gpumd-neighbors-worktree` | GPUMD 适配中间状态；保留主工作区较新的邻居、周期像力与 virial、runtime 实现。 |
| `Metad-toy-plumed-compare` 与其 `Metad-toy-fork` | LocalGeom、软固体数等旧轴尚无完整替代，登记在延期功能清单。 |
| `Metad-toy-hybrid-mpi` | 独有价值主要是单卡 MPI/OpenMP 运维说明与 Slurm 模板；作为参考保留，未验证的集群模板不混入默认运行入口。 |
| `lammps-metad-toy` | 无 Git 的早期平铺源码；保留历史快照。 |
| `metad-verify-build` 等 | 构建产物，不作为代码来源。 |

18 份源码归档的路径与 SHA256 见 `version-sources-20261028.json`。归档保存选定源码文件和
对应已跟踪改动补丁；不声称包含历史目录中的全部轨迹、编译缓存或忽略文件。
原目录和运行中的生产任务均保留。

## 本次统一的接口

- Graph 仍使用 `begin/finish/cancel_force_virial_batch`。Fix 只提供本步 virial 标志，不再嵌套第二套开闭批次。
- Q/LocalQL 的梯度在 GPU 取负生成物理力，以 CUDA view 提交；逐原子梯度诊断按需下载。
- Weave 是显式可选的构建后端，`ZQC_LAMMPS_WEAVE=OFF` 为默认值。
- 启用 Weave 时，Host/CUDA/owned 单原子贡献合入一个设备批次。不能根据某个 rank 是否收到 Host 单原子贡献单独改选 Host reverse。
- 提交返回前结束对调用方缓冲的读取；取消会排空并丢弃待提交结果；有 ghost 贡献时一次 reverse，纯 owned 贡献跳过 reverse。
- 仅需要全局 virial 时下载 6 个归约值；请求逐原子 stress 时下载 owned 六分量并由 Fix tally，避免重复累加。
- `WEAVE_METAD_DEVICE_COMMIT=0` 保留 Host 合批回退，便于用同一插件数值对照。
- `minimum_image` 调用兼容 LAMMPS 2024/2025 的签名；兼容代码存在不等于所有版本均完成本轮运行验证。

## 使用和复现

构建与验证入口位于 `benchmarks/integration/`。所有运行显式指定源码、LAMMPS、MPI、CUDA、
Kokkos 和 Weave 路径，不根据目录名或旧 `.so` 的存在推断版本。
Weave 运行脚本在创建 box 前加载匹配的 `weavebatchplugin.so` 并调用 `weave/batch_comm`。
Kokkos 配置必须来自同一份宿主 `liblammps.so`，不能额外链接另一份静态 Kokkos runtime。

本轮验收的实际命令、二进制 SHA256、退出码和数值结果由构建目录及验证目录保存。
验收汇总见 `INTEGRATION_VALIDATION_20261028.zh.md`，不要把历史验证数据归到新二进制名下。

## 项目仍未完成的部分

代码版本统一与完整科学验证是不同的验收项。当前生产 `SHARED_FROZEN` 是固定表的 MEAN 模式，
不是在线共享沉积；生产 SUM 模式和共享 WT 历史尚未实现。概率/压缩实验仍属于原型，
不因本次收进统一源码而变成已验证的生产功能。

邻居元数据、ghost 补行、CSR 构造及部分计数仍涉及 Host。LocalQL6 每条边的球谐缓存等内存问题
尚未在本轮优化或完成大规模性能剖析。短程正确性回归也不能替代 Al3Sc 形核路径、自由能收敛与
重加权有效样本数的验证。远端固定势大体系运行只能支持其实际覆盖的稳定性和容量结论。

后续功能见 [延期功能清单](DEFERRED_FEATURES_20261028.zh.md)。
