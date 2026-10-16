# Local Q 接入 Graph

`STEINH L 3/4/6` 提供逐中心原始值和 adjoint，可以与普通 Q、Stru_factor 等已接入的 CV 使用同一张 Graph。CVAssembly 的 owned 前缀给出中心 I 的顺序，Graph 汇总所有输出对每个中心的梯度，再通过 Local Q 的 I→J→K 链传播到坐标。

## 输入示例

```lammps
fix m all metad EXECUTOR GRAPH GAUSSIAN 0.10 0.10 10 PACE 100000 &
    CAL NAME LQ4 STEINH L 4 all cutoff_r 3.30 d_block_size 64 &
    CAL NAME LQ6 STEINH L 6 all cutoff_r 3.30 d_block_size 128 &
    SYMBOL q4 LQ4.RAW SYMBOL q6 LQ6.RAW &
    CV_dim 2 DIM 1 -1 2 300 "AVE(q4*q6)" &
             DIM 2 -1 2 300 "AVE(fermi(q6,0.45,12))" &
    METAD_RESTART 0 WT 0
```

- `.RAW` / `.value` 提供公共中心并集上的逐原子 LQ，组外补零。
- Graph 的 `.AVE` / `.MEAN` 对该 CV 自己的 owned 中心组求算术均值；表达式中的 `AVE(expr)` 使用公共并集计数。
- `SW_FUNC_r` 继续对两层邻居距离加权。Graph 中的值开关通过表达式定义，配置 `SW_FUNC_cv` 会提示改写到 DIM。
- `STEINH L` 统一要求 `EXECUTOR GRAPH`。旧 CV 内部的 `AVE` / `MEAN_SOLID`、`SW_FUNC` / `NSOLID`、`FRAC` 及 `SW_FUNC_cv` 已移除，旧输入给出迁移提示。
- 迁移时注意旧 `MEAN_SOLID` / `AVE` 是 `Σqf/Σf`，`NSOLID` 是 `Σf`，`FRAC` 是 `Σf/N`。Graph `.AVE` 是原始 LQ 算术均值；`AVE(fermi(...))` 是公共中心并集上的平均开关值。Graph 当前尚未提供 SUM 规约。

## 存储与执行

- `d_group_indices[calctag]` 的前 `group_count` 项是 owned I，后面追加计算 q_lm 所需的 J，J 可为 ghost；`d_calc_tag[localid]` 提供逆映射。
- 原始 q_lm、平均后的 LQ_lm 按 `[localid][2*(L+1)]` 存储，m=0..L、实部与虚部交错；最终 LQ 值按 localid 存储。
- `d_center_adjoint[center]` 对应 CVAssembly 的 owned 前缀。`d_dcvdx_rjk_prefix[calctag][2*(L+1)]` 汇总所有 I 传向 q_lm 的梯度前因子，供 J→K 阶段使用。
- 力和 virial 按 `[localid][3]`、`[localid][6]` 累积。使用两层 ghost halo，forward 宽度为 0；reverse 宽度为 9，顺序为六分量 virial、三分量坐标梯度。

同一环境由首个 CV 更新，其余阶数复用；每个 CV 按自己的 block size 启动。Graph 每轮使几何失效，候选邻居缓存随新 run、邻居重建、localid 布局及组表变化刷新。空中心、孤立原子和空 rank 都有明确的零输出路径。

本机 LAMMPS 29Aug2024 的自定义 cutoff 仅覆盖 owned 邻居行，ghost 行仍使用 pair cutoff。当它小于 Local Q 的候选半径时，插件对 owned I 所依赖的 ghost J 用已有 halo 坐标和空间分箱补齐候选行，保留 type/group/molecule 排除规则。这些行随邻居缓存复用，最终仍由 GPU 应用物理 cutoff 和径向权重。

## 验证入口

使用可导入 `lammps`、`mpi4py`、`numpy`、`scipy` 的 Python：

```bash
python benchmarks/steinhardt_graph/run.py --kind L --output build/localql-check
python benchmarks/steinhardt_graph/run.py --kind mixed --output build/localql-mixed-check
python benchmarks/cv_communication/run.py --after-only \
  --cases local_graph q_local_graph \
  --initial build/graph-lammps/shared-mean-validation/initial.data \
  --output build/localql-comm-check
```

`--kind L` 检查 L3/L4/L6；`mixed` 检查 Q3/L4/L6。CPU 参考使用 SciPy 球谐函数计算 q_lm 和局域平均，与 CUDA 多项式实现独立。测试覆盖前向值、坐标及六分量应变有限差分、MPI 1/2/4 rank、共享环境、不同原子组、组变更、迁移、同一步重算、孤立原子、空 rank、径向开关以及邻居排除规则。

测试记录及源码/插件摘要在本目录 JSON 中。`legacy_regression_results.json` 保存初次接入时的历史对照，当时旧规约入口仍可用；`communication_results.json` 记录该阶段的实际通信包宽。接口清理后的记录使用 `cleanup_` 前缀。此次验证功能与数值正确性，尚未进行长时间性能对照。

## 初次接入结果（2026-10-16，接口清理前）

- Local Q Graph 21 组、Q/Local 混合 Graph 15 组、普通 Q 回归 9 组均通过，均覆盖 1/2/4 rank。
- Local Q 原始值对独立 CPU 参考最大误差 `4.44e-16`；Local/mixed 的力、virial 有限差分最大误差分别为 `1.21e-12`、`2.03e-08`。
- 6 组通信探针检查通过。9 组 LEGACY 修改前后对照的 CV/能量最大差 `0.00e+00`、力最大差 `5.08e-21`。
- 既有 16 项 CTest 通过；2 rank 的不同组／组变更场景经 Compute Sanitizer memcheck 检查，两 rank 均为 0 errors。

## 接口清理后结果（2026-10-16）

Release 插件已重新构建，`STEINH Q/L` 统一使用 Graph。Local Q 的旧规约、值开关状态、额外 MPI 求和及 seed kernel 已移除。

- Local Q 21 组、Q/Local 混合图 12 组、普通 Q 回归 6 组通过，覆盖 1/2/4 rank。
- Local Q 原始值对独立 CPU 参考最大误差 `4.44e-16`；本轮力、virial 有限差分最大误差分别为 `1.21e-12`、`1.44e-08`。
- Local Q 的 AVE、MEAN_SOLID、SW_FUNC、NSOLID、FRAC 旧回调及 `SW_FUNC_cv` 共 6 项拒绝检查通过；Q/mixed 的旧接口拒绝回归也通过。
- 清理前后插件的 6 组 Graph 对照通过：CV/能量最大差 `0.00e+00`，力最大差 `3.39e-21`，压力最大差 `9.54e-18`。每 rank 的通信包宽、调用次数和发送量一致。
- 既有 16 项 CTest 全部通过，包含 NVRTC 前向/反向及 MPI 场景。

原始输入、screen/log、构建日志保存在 `build/localql-interface-cleanup/`。本目录的 `cleanup_*.json` 保存本轮结果；`cleanup_metadata.json` 记录源码和插件 SHA-256。
