# WEIGHT_CHEM Graph 与 virial 验证

原始值保持为 `W_i = 1 + sum_j w(type_i)*w(type_j)*s(r_ij)`。
径向开关 `s` 在 `r² <= 0.8*rc²` 时为 1，外侧为 `1-u³`，其中 `u=(r²-0.8*rc²)/(0.2*rc²)`。常数自项为 1，截断方式沿用原定义。

## 输入

在 `fix ... metad EXECUTOR GRAPH` 内使用以下片段，各行之间按 LAMMPS 规则添加续行符：

```text
CAL NAME W WEIGHT_CHEM selected cutoff_r 3.05 d_block_size 128 Chem_map (1,1.0) (2,-0.7) (3,0.35)
SYMBOL w W.RAW
CV_dim 1 DIM 1 0 1 500 "AVE(fermi(w,-4,0.4))"
```

`W.RAW` 提供逐中心原始值；`W.AVE` / `W.MEAN` 提供该 CV 中心组的全局平均。多个不同组的 RAW 输入装配到公共并集，缺失位置为零，通用 `AVE(expr)` 对该公共并集计数；直接使用 `W.AVE` 时按 W 的中心组计数。

`Chem_map` 接受 LAMMPS 数字类型编号，权重可正、负或零。未列出的类型使用 `exp(-0.5*((type-Chem_ctarget)/Chem_sigma)^2)`，默认参数为 0、1。`(1,1.0)`、引号括起的 `(1 1.0)` 以及两个 token 的 `(1 1.0)` 均可读取。

CV 值上的 `SW_func` 已移到 Graph 表达式。径向开关仍在邻居循环内执行。原来的 LEGACY AVE/COUNT 接口已移除；当前 Graph 提供均值规约，`AVE(fermi(...))` 是平均开关值，不能当成总 COUNT。总计数规约尚未实现。

## 存储与反向

`Weighted_chem_pair` 复用 `Stru_factor` 的 Graph 生命周期和 GpuBuffer：每个实例保存自己的 localid 原始值，环境保存可共享的中心表、邻居几何、类型和元素权重。相同组、cutoff、完整权重配置可与 `STRU_FACTOR Chemical` 共享环境。

Graph 返回压缩中心排列的 `dV/dW_i`，每线程处理一个中心，将邻居项的贡献累加到两端原子。力按 `3*localid+c` 存放；virial 按 `6*localid+c` 存放，顺序为 `xx,yy,zz,xy,xz,yz`，每个邻居项向两端各分一半。ghost 的力和 virial 通过每原子 9 个 double 的 reverse communication 汇回 owner。逐原子诊断保留 `W.chem_pair_r`。

## 复现

```sh
cmake --build build/graph -j4
ctest --test-dir build/graph --output-on-failure
cmake --build build/graph-lammps -j4
python3 benchmarks/weighted_chem/run.py --output build/weighted-chem-new-run
python3 benchmarks/cv_communication/run.py --after-only --cases chem_mix \
  --initial build/weighted-chem-new-run/signed_maps.data \
  --output build/weighted-chem-new-comm
```

脚本沿用仓库现有测试的本地 LAMMPS/MPI 路径。运行目录应为新目录。

## 2026-10-15 验证结果

- 16 项 CTest 全部通过；WEIGHT_CHEM 覆盖 1/2/4 rank。最大坐标力有限差分误差 `2.08e-10`，并通过六个应变方向的 virial 有限差分、逐原子各分一半及累加布局检查。
- 完整 LAMMPS 共 21 组：七种配置分别测试 1/2/4 rank。覆盖正负零权重、高斯后备权重、小中心组及空 rank、CV cutoff 大于 pair cutoff、RAW+Fermi、与化学结构因子共享环境、不同中心组。
- 独立 CPU 原始值最大误差 `1.34e-14`；不同 MPI 分区逐原子位置/力/原始值最大差 `1.57e-17`；六分量全局压力最大差 `1.26e-12`。逐原子应力求和与全局压力最大差 `3.42e-13`。
- 各有效场景都检查非零偏置力、组外邻居受力和总力守恒。六种非法输入按预期报错。
- 通信探针确认 1/2/4 rank 只出现宽度 9 的 reverse，forward 宽度为 0。

`results.json` 记录数值结果及原始日志路径，`communication_results.json` 记录通信验证。SW_func 分离/内联性能对照见 `../weighted_chem_switch/`。
