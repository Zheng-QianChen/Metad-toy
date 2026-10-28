# 暂不直接合并的独有实现

以下保留功能语义和源文件位置，便于后续按当前 Graph/引擎接口重新实现。历史代码均已有源码
归档，目录与 SHA256 见 `version-sources-20261028.json`。未以文件不同或旧关键字消失作为功能缺失的唯一依据。

## D01：LocalGeom 软配位与局部几何总数

来源：`/home/zqc/work/Metad-toy-plumed-compare/include/CVsets/CV_LocalGeom.h:12`、
`src/cvs/CVs_LocalGeom.cu:243` 和同目录下 `Metad-toy-fork` 的同名文件。

- `n_i = sum_j sigma(r_ij)`：按邻居选择得到软配位数。
- `NTET = sum_i g_tet(n_i)`：接近指定配位数的局部环境数量；别名 `NSOLID`。
- `NCOORD = sum_i n_i`：软配位总数；别名 `SUM_N`。`AVE_N` 再除中心总数。
- 当前没有对应 LocalGeom 叶和可直接配置的 Graph SUM 节点；CSS 评分不能替代软配位总数。

暂不合并原因：旧实现绑定旧 LAMMPS/NeighHub 与标量 bias_force 接口。重新实现需要软配位叶、
完整反向和全局规约，不能仅复制注册类或把 SUM 改为 MEAN。

最小验收：独立 CPU 构型参考、1/2/4 rank 一致性、空 group/无中心 rank、周期边界、坐标和六分量
应变有限差分、变化选择集后的缓存失效。旧阈值/开关参数需原样记录后再比较。

## D02：开关加权平均邻距 MEAN_RNN

来源：同一 `CVs_LocalGeom.cu:280`；`MEAN_RNN`/`AVE_RNN`。旧代码在 `:528` 明确拒绝偏置力，
仅用于监视。保留这个限制，不能将其标成可用 MetaD 轴。

后续可先移植为监视型逐原子/全局诊断。验收包含独立距离与开关参考、零邻居/空中心行为、MPI
规约；若要用于偏置，另行实现并验证梯度和 virial。

## D03：旧 LocalQL 的 NSOLID 和 MEAN_SOLID

来源：`/home/zqc/work/Metad-toy-plumed-compare/Metad-toy-fork/src/cvs/CVs_Steinhardt_Local.cu:639`。

| 旧轴 | 数学含义 | 当前处理 |
| --- | --- | --- |
| `NSOLID`（旧 `SW_FUNC` 别名） | `sum_i f(q_i)` | 需要 SUM/总数语义，登记延期。 |
| `MEAN_SOLID`（该旧 fork 中的 `AVE` 别名） | `sum_i q_i f(q_i) / sum_i f(q_i)` | 当前普通 AVE 不等价；需加权均值或安全除法及零分母约定。 |
| `FRAC` | `sum_i f(q_i)/N` | 数学上可由 Graph `AVE(f(q))` 表达；需要一致的中心选择、开关参数及迁移验收，不必恢复旧类。 |

注意：加权均值不必依赖新增 SUM；同一个中心集合下也可用两个均值之比实现，但当前内置 Graph
节点尚无通用除法。这与 NSOLID 的总数缺口是两个问题。迁移时不能继续把不同年代的 `AVE`
关键字当成同一个物理量。

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

- 生产共享势 SUM 与在线共享沉积/WT 历史。
- 概率压缩策略的统计验证和未知相比例下的校准。
- 邻居/CSR 完全设备化、LocalQL 球谐缓存和大体系性能。
- Al3Sc 形核机制、FES 收敛和重加权验收。

这些是尚未完成的开发或科学验收，不能在本次“版本已统一”结论中一并勾选完成。
