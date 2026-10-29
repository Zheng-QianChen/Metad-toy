# 历史独有实现及恢复记录

以下保留功能语义和源文件位置；D01–D03 已按当前 Graph/引擎接口恢复，见
[恢复验证](LEGACY_RESTORATION_20261028.zh.md)。历史代码均已有源码
归档，目录与 SHA256 见 `version-sources-20261028.json`。未以文件不同或旧关键字消失作为功能缺失的唯一依据。

## D01：LocalGeom 软配位与局部几何总数

来源：`/home/zqc/work/Metad-toy-plumed-compare/include/CVsets/CV_LocalGeom.h:12`、
`src/cvs/CVs_LocalGeom.cu:243` 和同目录下 `Metad-toy-fork` 的同名文件。

- `n_i = sum_j sigma(r_ij)`：按邻居选择得到软配位数。
- `NTET = sum_i g_tet(n_i)`：接近指定配位数的局部环境数量；别名 `NSOLID`。
- `NCOORD = sum_i n_i`：软配位总数；别名 `SUM_N`。`AVE_N` 再除中心总数。
- 已新增 LOCAL_GEOM 的 COORD/TET 叶和 Graph SUM；CSS 评分不能替代软配位总数。

原暂缓原因：旧实现绑定旧 LAMMPS/NeighHub 与标量 bias_force 接口。本轮已重写设备前后向、
virial 和 Graph 规约，没有复制旧的 Host 邻居遍历。

最小验收：独立 CPU 构型参考、1/2/4 rank 一致性、空 group/无中心 rank、周期边界、坐标和六分量
应变有限差分、变化选择集后的缓存失效。旧阈值/开关参数需原样记录后再比较。

## D02：开关加权平均邻距 MEAN_RNN

来源：同一 `CVs_LocalGeom.cu:280`；`MEAN_RNN`/`AVE_RNN`。旧代码在 `:528` 明确拒绝偏置力，
仅用于监视。保留这个限制，不能将其标成可用 MetaD 轴。

已提供 rnn/rnn_valid 逐原子诊断，由 owned 项规约恢复原全局监视量。已对照独立参考，
覆盖零邻居/空中心和 MPI；没有把 RNN 暴露为可偏置的 Graph 叶。

## D03：旧 LocalQL 的 NSOLID 和 MEAN_SOLID

来源：`/home/zqc/work/Metad-toy-plumed-compare/Metad-toy-fork/src/cvs/CVs_Steinhardt_Local.cu:639`。

| 旧轴 | 数学含义 | 当前处理 |
| --- | --- | --- |
| `NSOLID`（旧 `SW_FUNC` 别名） | `sum_i f(q_i)` | 已由中心组感知的 `SUM(f(q))` 恢复。 |
| `MEAN_SOLID`（该旧 fork 中的 `AVE` 别名） | `sum_i q_i f(q_i) / sum_i f(q_i)` | 已由 `safe_div(SUM(q*f(q)),SUM(f(q)),0)` 恢复。 |
| `FRAC` | `sum_i f(q_i)/N` | `safe_div(SUM(f(q)),SUM(constant_one(q)),0)`，保持中心组边界。 |

安全除法的 epsilon=0 保留旧版恰好零分母约定；正 epsilon 明确改变近零区间行为。
不同年代的 `AVE` 仍不视为同一个物理量，旧语法继续给出迁移提示。

验收：逐项对照旧公式、零分母和近零分母、尺寸扩展时的缩放、MPI/空 rank、坐标与应变差分。

## D04：hybrid-mpi 集群运行模板

来源：`/home/zqc/work/Metad-toy-hybrid-mpi/docs/HYBRID_MPI_OPENMP.zh.md` 和
`scripts/slurm_hybrid_single_gpu.sbatch`。功能是单卡单 MPI rank、OpenMP 线程与资源绑定的运行约定。

保留作迁移模板；后续去除旧绝对插件路径，与集群实际 LAMMPS/MPI/CUDA ABI、Slurm 绑定和线程
配置验证后，再作为统一运行入口。此次没有替换正在运行的作业。

## 已有替代、无需恢复旧实现的项目

| 历史实现 | 当前替代与限制 |
| --- | --- |
| `LOCALQL_CSS` 的 `JOINT.MEAN` | `STEINH L` + `CSS`，逐原子先乘后取 `AVE(q*c)`，只归一一次；不是 `AVE(q)*AVE(c)`。 |
| `SW_FUNC_product` / `SW_MEAN` | `AVE(q*c*fermi(q*c,...))` 等匹配开关表达；这仍是均值，不是总固体数。 |
| `apply_center_weights` 专用回调 | Graph adjoint。 |
| views 的 `BorrowedArray` | 当前 `ArrayView`、stride、readiness 和通用引擎接口。 |
| `GH_t0_uniformGrid`、`GH_t1_sparseHash`、旧 `zqc_gaussian.h` | 当前 BiasGrid/hill bias。 |
| grid/GPUMD 开发工作树 | 主工作区已有后续实现；不把中间草稿整包合回。 |

LocalQL_CSS 的表达式替代已纳入小体系能量与导数回归；这不等于对所有旧液/固轨迹、所有旧输入
参数完成二进制逐项对照。旧输入文件可继续从归档取回开展这类专门验收。

## 与旧功能恢复分开追踪的开发项

- 生产共享势 SUM 与在线共享沉积/WT 历史已实现，见 [共享功能验收](SHARED_ONLINE_20261028.zh.md)。
- 概率压缩策略的统计验证和未知相比例下的校准。
- Q/Half、属性 forward 及无 Weave 设备力合批已完成，见 [路径验收](DEVICE_COVERAGE_20261028.zh.md)。
  LocalQL 无用球谐缓存已移除、直接借用设备 CSR，见 [A100/WSL 性能](LOCALQL_PROFILE_20261028.zh.md)。
  仍保留 Host 选择集元数据、LocalQL 一个纯 J 标量计数回读、无 Weave 的 reverse 边界包及诊断下载。
  不能称为全项目零 Host；更多硬件与规模的性能仍需对应平台证据。
- Al3Sc 形核机制、FES 收敛和重加权验收。

其中尚未完成的生产规模或科学验收，不能因“版本已统一”一并勾选完成。
