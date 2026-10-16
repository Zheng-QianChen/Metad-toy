# 非 Local Steinhardt Q 接入 Graph

`STEINH Q 3/4/6` 现在提供逐中心原始值与梯度接口。这里的 Q 是每个中心的径向加权球谐模长；`STEINH L` 表示再对邻居 qlm 做平均的 Local Q，其后续 Graph 接入见 [Local Q 验证](../localql_graph/README.md)。

## 输入

```lammps
fix m all metad EXECUTOR GRAPH GAUSSIAN 0.10 0.10 10 PACE 100000 &
    CAL NAME Q4 STEINH Q 4 all cutoff_r 3.30 d_block_size 128 &
    CAL NAME Q6 STEINH Q 6 all cutoff_r 3.30 d_block_size 128 &
    SYMBOL q4 Q4.RAW SYMBOL q6 Q6.RAW &
    CV_dim 2 DIM 1 0 1 300 "AVE(q4*q6)" &
             DIM 2 0 1 300 "AVE(fermi(q6,0.45,12))" &
    METAD_RESTART 0 WT 0
```

- `.RAW` / `.value` 提供公共中心并集上的逐原子 q_l，组外位置补零。
- `.AVE` / `.MEAN` 是该 Q 自己的 owned 中心组全局平均；`AVE(expr)` 按公共并集 owned 数计数。
- `SW_FUNC_r` 继续作用于邻居距离，支持已有径向开关配置及尾部截断。对 q_l 的变换在 Graph 表达式中写出。
- Q 统一使用 `EXECUTOR GRAPH`，原有 LEGACY 的 AVE／SW_FUNC 回调及 `SW_FUNC_cv` 参数已移除。Local Q 后续也完成相同清理，见 [Local Q 验证](../localql_graph/README.md)。
- 旧 Q `SW_FUNC` 的语义是 `Σ q_i*f(q_i)`，Graph 的 `AVE(fermi(...))` 是平均开关值，两者含义不同；当前 Graph 尚未实现 SUM 规约。

## 数据与通信

CVAssembly 提供中心 localid 排列。q_l 按 `localid` 存储，qlm 按 `localid*2*(L+1)+2*m+real/imag` 存储，只保存 m=0..L。Graph 先汇总多个输出对 Leaf 的贡献，再提取每个 Q 的 owned `dV/dq_i`。

反向保留原子对线程布局。一个半对需要两端的 qlm、q_l、径向权重和及 `dV/dq_i`，因此 forward 每原子发送 `2*(L+1)+3` 个 double：Q3/Q4/Q6 分别为 11/13/17。随后计算坐标梯度和 virial，通过宽度 9 的 reverse 将 ghost 贡献归还 owner。virial 顺序为 `xx,yy,zz,xy,xz,yz`，每项向原子对两端各分一半。

每轮 Graph 求值统一使共享环境失效，首个 Q 刷新几何，后续 Q 复用。同一步修改坐标后求值也会刷新；run 重新初始化、原子迁移和组成员变化会使候选邻居缓存失效。各 Q 按自己的线程块大小计算启动规模。

## 验证

```bash
cmake --build build/graph-lammps -j4
# 使用能导入 lammps、mpi4py、numpy 的 Python。
python benchmarks/steinhardt_graph/run.py --output build/q-graph-new-check
```

脚本支持 `--kind Q/L/mixed`、`--plugin`、`--mpi`、`--np`、`--cases`，输出目录必须是新目录。每个配置使用独立进程，保存输入、LAMMPS 日志、逐原子结果和数值误差。

接口清理后的检查覆盖六种场景各 1/2/4 rank，共 18 组完整 LAMMPS 运行。覆盖 Q3/Q4/Q6、多个输出、不同中心组、组成员变化、不同线程块大小共享环境、平滑径向函数、CV cutoff 大于 pair cutoff、迁移、孤立原子和空 rank。

- CPU 前向使用球谐加法定理和 Legendre 多项式，原始值最大误差 `3.37e-15`。
- 非线性、不同组及径向开关场景各检查 12 个坐标自由度和 6 个应变方向的有限差分；力与 virial 最大误差分别为 `1.25e-12`、`6.25e-09`。
- 检查总力守恒、逐原子 virial 求和、MPI 分区一致性、同一步恢复坐标，以及多次 run 和迁移后 NVRTC 只编译一次。
- 另检查 LEGACY Q 的 AVE、SW_FUNC 和 Q 的 `SW_FUNC_cv` 输入均给出迁移提示，结果见 `interface_results.json`。
- 通信对照使用本次清理前的插件，检查 Q4/Q6、Q3/Q6 和 Local Q4/Q6 的 1/2/4 rank，共 9 组通过，CV/能量一致，最大力差 `1.69e-20`。结果见 `cleanup_communication_results.json`。
- 既有 16 项 CTest 全部通过。

当前数值结果在 `results.json`，构建信息及误差摘要在 `metadata.json`。首次迁移时的 21 组结果保存在 `initial_graph_results.json`；`communication_results.json` 和 `legacy_regression_results.json` 也保留为迁移前后对照记录。本轮验证接口清理的正确性，未做长时间性能测试。
