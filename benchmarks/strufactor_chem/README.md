# Stru_factor_chem 接入 Graph 的验证

化学版原始值保持为

`S_i = 1 + sum_j w(type_i) * w(type_j) * sinc(q*r_ij) * s(r_ij)`。

常数自项为 1。`s` 沿用 Stru_factor 的径向定义：`r^2 <= 0.8*rc^2` 时为 1，之后以 `1-u^3` 降到 0，`u=(r^2-0.8*rc^2)/(0.2*rc^2)`。截断处的值连续，一阶导数仍沿用原定义。

## 输入与数据流

继续使用 `STRU_FACTOR ... Chemical`，在 `fix ... metad EXECUTOR GRAPH` 中可以写以下片段：

```text
CAL NAME C STRU_FACTOR selected cutoff_r 3.3 q_factor 1.7 Chemical Chem_map (1,1.0) (2,-0.7) (3,0.35)
SYMBOL c C.RAW
CV_dim 1 DIM 1 0 10 500 "AVE(c*c)"
```

`C.AVE` / `C.MEAN` 可直接作为全局均值的 SYMBOL 输入。`C.RAW` / `C.value` 是逐中心原始值，允许先组合、再规约。数值常量用于算符参数，例如 `linear(AVE(c),0.3,0)` 表示均值乘 0.3。

`Chem_map` 的类型键仍使用 LAMMPS 数字类型编号。允许正、负、零权重；未指定的类型使用 `exp(-0.5*((type-Chem_ctarget)/Chem_sigma)^2)`。高斯参数默认是 0、1。这是对类型编号的映射，实际元素权重宜显式指定。

普通版和化学版共用装配、Graph 生命周期、力/virial 缓冲和反向通信。化学版选择同一数值实现的编译期加权特化。输出按 localid 保存；读取邻居类型时与筛选后的几何使用同一槽位。图返回的每个中心 `dV/dS_i` 分发到邻居项两端，ghost 贡献经 reverse communication 回到 owner。均值由 Graph 的 AVE 节点处理。

共享化学环境的键包含完整 Chem_map 和高斯参数。不同权重的 CV 各自保留对应类型表。邻居请求按 CV cutoff 建立，并覆盖所需的 ghost 距离。

## 验证入口

```sh
cmake --build build/graph -j4
ctest --test-dir build/graph --output-on-failure
cmake --build build/graph-lammps -j4
python benchmarks/strufactor_chem/run.py --output build/strufactor-chem-new-run
```

`run.py` 使用与现有 CSS 测试相同的本地 LAMMPS、MPI、插件路径，输出目录需为新的运行目录。

- 独立 kernel 测试：`benchmarks/graph/strufactor_graph.cu` 分别编译普通与化学版本，覆盖 1/2/4 rank、空 rank、localid 重排、邻居段空槽、正/负/零权重、组外力、零 seed、小距离、六分量 virial 及非线性 Graph。
- 元素权重全为 1 时，对照普通 Stru_factor 的值、力和逐原子 virial。
- 实际 LAMMPS：五种配置各测试 1/2/4 rank，共十五组。包含不同 Chem_map 共存、高斯后备权重、小中心组、CV cutoff 大于 pair cutoff，以及 RAW 非线性组合。
- 三种非法参数检查：零 sigma、越界类型编号、无限权重。

本次 13 项 CTest 全部通过。化学版最大力有限差分误差约 `6.4e-11`；实际 LAMMPS 原始值对独立 CPU 公式最大误差约 `1.4e-15`，MPI 分区逐原子状态最大差约 `7.4e-17`，压力六分量最大差约 `2.75e-11`。各场景均检查非零力、组外力和总力守恒。

普通 Stru_factor 的实际 LAMMPS 1/2 rank 回归也通过，最大力差约 `2.3e-17`。详细化学版结果保存在 `results.json`；完整运行日志位于 `build/strufactor-chem-validation`、`build/strufactor-chem-raw-validation-v2` 和 `build/strufactor-chem-invalid-validation`。

测试输入修正记录：RAW 非线性场景使用 `linear(...)` 表达常数缩放，并将第一维网格设为 `[0,10]` 以覆盖约 6.9 的输出；零偏置力不作为反向验证通过的条件。
