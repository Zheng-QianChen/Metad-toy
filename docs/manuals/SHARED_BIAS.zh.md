# 共享原子偏势：冻结、在线沉积、SUM/MEAN 与续算

一个 family 的 owned 中心共同使用同一张一维势表 u(s)。ghost 参与 CV 导数和力通信，不重复沉积。令全局有效中心数为 N：

- `SHARED_AGGREGATION MEAN`（兼容默认）：V = Σu(sᵢ)/N，dV/dsᵢ = u′(sᵢ)/N。
- `SHARED_AGGREGATION SUM`：V = Σu(sᵢ)，dV/dsᵢ = u′(sᵢ)。

两种模式的 `f_bias` 都返回实际总偏势 V，`f_bias[1]` 都返回同一 family 的 MEAN(s)。family 必须全局非空，本地空 rank 正常参与规约。当前只接受一个非周期逐原子 Graph 表达式。

`thermo_modify norm yes` 会按 LAMMPS 规则把 SUM 模式的 thermo 能量显示除以体系总原子数；要直接显示总 V，使用 `thermo_modify norm no`。程序接口 `extract_fix` 的能量始终是总 V。

## 冻结已有表

```lammps
fix bias all metad EXECUTOR GRAPH SHARED_FROZEN shared.table Sc &
  SHARED_AGGREGATION SUM PACE 0 &
  CAL NAME Q STEINH L 6 Sc cutoff_r 4.0 &
  SYMBOL q Q.RAW CV_dim 1 DIM_EXPR 1 "q"
```

表格式保持 `# METAD_TOY_SHARED_FROZEN 1`，随后为等间隔的 `s u du/ds` 三列。三次 Hermite 插值的能量与导数来自同一多项式；冻结表区间外按端点斜率线性延拓。冻结模式不接受 WT、非零 PACE 或 METAD_RESTART。

## 在线共同沉积

```lammps
fix bias all metad EXECUTOR GRAPH SHARED_ONLINE Sc -0.2 1.2 401 &
  SHARED_AGGREGATION SUM SHARED_DEPOSIT MEAN &
  GAUSSIAN 0.05 0.001 10 TEMP 1000 PACE 100 WT 1 &
  HILLS_FILE shared.history METAD_RESTART 0 &
  CAL NAME Q STEINH L 6 Sc cutoff_r 4.0 &
  SYMBOL q Q.RAW CV_dim 1 DIM_EXPR 1 "q"
fix_modify bias energy yes virial yes
```

`SHARED_ONLINE <group> <lower> <upper> <nodes>` 指定共享网格。上下界必须有限且递增，节点数 4–1,000,000，间距不得大于 sigma/2。所有沉积中心必须落在网格范围内，越界会报错而不是静默丢弃 hill。普通逐步求值继续使用同一个 C1 插值。参数中的能量和 `TEMP` 依 LAMMPS 当前单位，`units metal` 下 height 为 eV。

用户选定的默认是 `SHARED_DEPOSIT MEAN`：每次沉积的**基准总振幅**为 height，各中心分配 height/N。显式 `SHARED_DEPOSIT SUM` 则让每个中心获得完整 height，基准总振幅为 N×height。这与上面的势能 SUM/MEAN 是两个独立设置。WT 开启后，各中心振幅还会被温度因子降低，实际振幅和不再固定等于基准值。

在一个沉积步，所有中心读取更新前的同一张 u，并同时生成增量：

```text
h_i = height × (1/N if SHARED_DEPOSIT MEAN else 1)
      × exp[-u_old(s_i) / (kBT × (bias_factor - 1))]  # WT 1
Δu(s) = Σ_i h_i exp[-(s-s_i)^2/(2σ²)]
```

WT 0 去掉指数温度因子。WT 温度因子使用未乘全局聚合系数的共享函数 u(sᵢ)。一批中的后一个中心不会看到前一个中心刚沉积的势，避免遍历顺序/MPI 划分改变 WT 定义。GPU 在每个网格节点计算完整高斯及解析导数，不截断尾部；MPI 只规约网格增量，不收集全部中心到 Host。普通求值复杂度 O(N)，沉积步为 O(N×nodes)，故网格精度与 PACE 也影响训练成本。

## 历史与续算

`shared.history` 使用独立版本头 `METAD_TOY_SHARED_ONLINE 1`，不是全局 MetaD 的普通 HILLS 文件。初始零表和每次沉积后的完整表都写一条校验记录，保存步号、累计次数、网格值和导数。配置含网格、sigma、height、温度、WT、聚合/沉积方式及 CV 表达式/定义的身份摘要。

- 使用相同 MD restart、相同 CV 配置和对应历史文件，设置 `METAD_RESTART 1`。
- 已有历史时，`METAD_RESTART 0` 拒绝覆盖；需要新训练时使用新文件名。
- 历史晚于当前 MD 步、参数/表达式不匹配、顺序错误、校验失败或尾部记录未写完整，都拒绝续算。
- `run 0`、setup、写 MD restart 不沉积；同一积分步不会重复沉积。
- `PACE 0` 可读取已训练历史后冻结。PACE 允许调整，不属于历史配置身份；其他历史参数仍须一致。
- 记录会 flush；程序不会静默忽略损坏尾部。此格式没有承诺断电时的文件系统持久性。

导出某一已记录步的冻结表，不覆写已有输出：

```bash
python tools/shared_history.py shared.history --output trained.table
python tools/shared_history.py shared.history --step 2000 --output step2000.table
```

随后用 `SHARED_FROZEN trained.table Sc SHARED_AGGREGATION SUM`（聚合方式与训练对应）运行冻结采样。工具先验证全部历史；省略 `--output` 时输出历史配置和沉积数量。

## 验收边界

独立 CUDA/MPI 测试将 8 种聚合/沉积/WT 组合与解析高斯能量和导数比较，并验证空 rank、有限差分、重启去重、冻结和损坏历史拒绝。真实 Al–Sc MEAM 的 Q/LocalQL/CSS 联合描述符另有 NPT 连续/续算对照。

这些证明实现遵循上述定义，**不自动证明** `−γ/(γ−1)u(s)` 就是相关原子体系的成核自由能。生产 FES、持续沉积时的重加权和采样收敛仍需对应统计检验。冻结后应使用实际总偏势 V 重加权；MEAN 与 SUM 的总能量不能混用。family 宜按固定身份或元素定义，硬空间分组的跳变不是光滑保守势。
