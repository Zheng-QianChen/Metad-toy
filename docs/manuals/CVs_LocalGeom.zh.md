# LocalGeom：软配位、局部几何计数与邻距监视

`LOCAL_GEOM` 通过当前引擎的设备 Full 邻居表计算 owned 中心的逐原子值，
邻居可以是组外原子和周期 ghost。Graph 负责规约和反向传播。

```lammps
CAL NAME radial SW_FUNC FERMI r_0 3.0 alpha -8
CAL NAME coord LOCAL_GEOM Sc SW_FUNC_r radial cutoff_eps 1e-12 OUTPUT COORD
CAL NAME tetra LOCAL_GEOM Sc SW_FUNC_r radial cutoff_eps 1e-12 OUTPUT TET TET_CENTER 4 TET_SIGMA 0.5
SYMBOL n coord.RAW
SYMBOL t tetra.RAW
```

每个中心的软配位为 `n_i = sum_j radial(r_ij)`；TET 输出为
`t_i = exp(-0.5*((n_i-TET_CENTER)/TET_SIGMA)^2)`。名称 TET 延续旧版，
此式只匹配配位数，不测量键角，不能仅凭它判定四面体结构。

参数：中心 group 必须提供；`OUTPUT` 默认 `COORD`，还可选 `TET`；
`TET_CENTER=4`、`TET_SIGMA=0.5`；`NEIGH_TYPE 0` 默认包含所有类型，
非零类型或类型标签由引擎解析。`cutoff_r` 显式指定时覆盖自动半径；
未指定径向开关时为硬截断邻居计数、默认半径 4。
给定径向开关但未指定半径时，由 `cutoff_eps` 求半径。
径向贡献小于 `cutoff_eps` 或距离小于 `1e-12` 时忽略。

旧物理量的迁移：

| 旧功能 | 当前表达 |
| --- | --- |
| `NCOORD` / `SUM_N` | `SUM(n)` |
| `AVE_N` | 非空组使用 `AVE(n)`；允许全局空组时用 `safe_div(SUM(n),SUM(constant_one(n)),0)` |
| `NTET` / LocalGeom 的 `NSOLID` | `SUM(t)` |

这些表达用于 `DIM_EXPR` 或 `DIM`，不能再把旧轴名直接接在 CAL 后。
硬截断和数值尾部筛选在边界处不光滑；有限差分应避开截断点。

`compute metad/atom` 提供 `coord.value`、`coord.n`、`coord.rnn` 和
`coord.rnn_valid`。其中 `rnn_i = sum_j r_ij radial(r_ij) / n_i`，
仅当 `n_i>cutoff_eps` 时有效；否则 rnn 和有效标记均为零。
组外与 ghost 输出槽位为零，MPI 规约仅计 owned 中心。

恢复旧 `MEAN_RNN` / `AVE_RNN` 监视量：

```lammps
compute rg all metad/atom m coord.rnn coord.rnn_valid
compute rg_sum all reduce sum c_rg[1] c_rg[2]
variable mean_rnn equal c_rg_sum[1]/max(c_rg_sum[2],1)
```

这是对有效中心的逐中心均值，不是把全体系所有邻居键放在一起加权平均。
RNN 仅作为诊断暴露，不接受 Graph SYMBOL，保持旧版“不能施加偏置”的边界。
COORD/TET 的力与六分量 virial 则由 Graph adjoint 完整传播。

回归入口：`benchmarks/localgeom/run.py`。覆盖独立 CPU 参考、周期边界、
类型筛选、排除、空集合、孤立原子、组变化、迁移、坐标/六分量应变差分和 MPI 分解。
