# 偏置层与高维沉积

运行时计算全部在 C++；Python 只用于离线测试和统计。纯 C++17 的 `zqc_bias` 库提供 `BiasSession`、`HillBias` 和 `BiasGrid`，可在关闭 CUDA、CV、LAMMPS 时独立构建。

## 后端选择

| BIAS | 表示 | 存储 / 求值 |
|---|---|---|
| AUTO（默认） | D≤3 选择 GRID，D>3 选择 PBMETAD | 按所选后端 |
| GRID | 1–3D 连续均匀网格 | O(∏bins)，局部三次插值 |
| SPARSE_GRID | 1–3D 哈希网格 | 已写格点，局部三次插值 |
| HILLS | 完整 D 维 Gaussian hill 列表 | O(H×D)，每次求值 O(H×D) |
| PBMETAD | D 个一维偏势的 PBMetaD 组合 | O(H×D)，每次求值 O(H×D) |

H 为沉积批次数。PBMetaD 改变采样算法，不能视为完整 D 维 MetaD 的无误差存储优化；要保留完整高维 hills 请显式使用 `BIAS HILLS`。启动日志打印最终后端、维数、历史条数、核类型、线程数和加载时间。

`Gaussian_Hill_type` 的 0/1/2/3 分别对应 GRID/SPARSE_GRID/HILLS/PBMETAD。新输入推荐 `BIAS`。

## 输入

```lammps
fix m all metad EXECUTOR GRAPH BIAS AUTO &
 GAUSSIAN 0.35 0.12 9 TEMP 300 PACE 100 &
 CAL NAME D1 DISTANCE 1 2 CAL NAME D2 DISTANCE 1 3 &
 SYMBOL d1 D1.RAW SYMBOL d2 D2.RAW &
 CV_dim 4 SIGMAS 0.35 0.45 0.8 0.5 &
 DIM_EXPR 1 "d1" DIM_EXPR 2 "d2" &
 DIM_EXPR 3 "d1*d2" DIM_EXPR 4 "linear(d1,0.4,0)-d2" &
 HILLS_FILE bias.hills HILLS_WORKERS 4 METAD_RESTART 0 WT 1
```

- `GAUSSIAN`：默认 sigma、初始高度、bias factor。高度与 kBT 使用宿主能量单位；LAMMPS 用 `force->boltz*TEMP`。WT=1 要求 factor>1。
- `CV_dim` 先于 `DIM`、`DIM_EXPR`、`SIGMAS`、`PERIODS`；各表达式产生一个全局标量。
- 网格用 `DIM index lower upper bins "expression"`，每轴 bins≥4。列表/PB 可用 `DIM_EXPR index "expression"`，无需网格范围。
- `SIGMAS`、`PERIODS` 各接 D 个值；省略时宽度统一、周期为 0。正 period 表示周期长度。网格周期必须与 upper−lower 一致。
- `PB_KERNEL PLUMED_2_10` 为 PB 默认核；`SMOOTH_GAUSSIAN` 可选择光滑周期核。核类型记录在历史头。
- `PB_NORMALIZATION MEAN` 为默认，另可选 `SUM`，只改变固定能量零点。
- `HILLS_WORKERS 1..16` 控制读取时解析和网格重建的工作线程，默认 1。只在 root rank 加载。
- `HILLS_FILE` 默认 HILLS，各 Fix 应使用独立文件。
- `PACE 0` 冻结自动沉积。`setup/run 0` 只施加已有偏势；积分到 PACE 整数倍时先沉积，再计算当前能量与梯度。同一步重复回调不会重复沉积。

[四原子完整示例](in.pbmetad.lmp) 已在 LAMMPS 29 Aug 2024 跑通。请在新的输出目录运行：

```sh
lmp -var plugin /absolute/path/fix_crystallize_plugin.so \
    -in /absolute/path/Metad-toy/benchmarks/highdim/in.pbmetad.lmp
```

## PBMetaD 数学与 PLUMED 兼容范围

先从同一份旧历史求出各轴 V_j，计算
`w_j=exp(-V_j/kBT)/sum_k(exp(-V_k/kBT))`。
各维沉积高度为 `height0*w_j`；WT 再乘 `exp[-V_j/(kBT*(gamma-1))]`。
所有轴高度计算完成后统一追加，保证结果不依赖轴遍历顺序。

组合势 `V_PB=-kBT*log(mean(exp(-V_j/kBT)))`，梯度为 `w_j*dV_j/ds_j`。内部先减最小 V_j，避免指数下溢。默认 MEAN 与 [PLUMED 2.10 PBMetaD::calculate 源码](https://github.com/plumed/plumed2/blob/v2.10/src/bias/PBMetaD.cpp) 一致，空历史为零；手册公式写的是 SUM，两者差 `kBT*log(D)`。

`PLUMED_2_10` 核按上述源文件的固定宽度、无 GRID 路径实现：周期轴使用最小镜像，`q=0.5*(delta/sigma)^2`，q<6.25 时为 `height*exp(-q)`，否则为零。该版本 PBMetaD 中 stretchA/B 保持 1/0，不能仅凭输出的 `kerneltype` 标签推定为平移后的核。截断面处有跳跃，有限差分应避开该面。源码常数在 [tools/Tools.h](https://github.com/plumed/plumed2/blob/v2.10/src/tools/Tools.h)。

兼容的是固定宽度 PB/WT 沉积公式及上述核求值。当前未实现 PLUMED 输入语法、原生多文件 HILLS 读写、ADAPTIVE、partitioned families 或多 walker；也未用 PLUMED 二进制做轨迹逐步交叉测试。PLUMED calculate/update 与本适配器的沉积时序不同，不能承诺逐步轨迹相同。核心的 evaluate/advance 已分开，宿主可以明确选择调用阶段。

完整 HILLS 的非周期轴使用 Gaussian，周期轴使用平滑核
`q=[sin(pi*delta/L)/(pi*sigma/L)]²`；`SMOOTH_GAUSSIAN` PB 使用同一核。完整协方差仍保留在原型中，生产接口当前是逐轴 sigma。

二维偏势可以通过固定其余坐标求切片。二维联合 FES 需要轨迹重加权统计；两个 PB 一维边缘无法恢复变量相关性。

## 网格与重启

网格统一采用运行时轴描述，dense 按末轴连续的行主序存储，sparse 用相同线性索引作哈希键。能量与梯度来自同一套 Catmull–Rom 张量插值，覆盖 1/2/3D 和不等 bins。

格点位于 bin 中心；非周期轴有效插值区为 `lower+1.5*dx` 到 `upper-1.5*dx`。区间外能量夹紧，该轴梯度为零；区间端点可能有导数折角，边界不充当物理墙。周期轴环绕插值。非周期 hill 更新范围为每轴 6 sigma；周期轴更新整轴。请按 CV 采样范围留足网格边距。

原生 HILLS v3 五行头记录模式、D、初始高度、kBT、WT/factor、核类型、sigma、period 和列名。数据按 `step D个中心 1或D个实际高度` 存储，17 位有效数字可恢复 FP64。此高度不同于 PLUMED WT 文件中经自由能比例缩放的高度。

加载先校验完整文件，再替换历史。配置不符、非法字段、未结束行、未来步号会拒绝；允许显式手工放置的同一步多条记录，自动沉积仍按步去重。v1/v2 只能以其原有光滑核读取。旧网格 `step centers height sigma` 文件也可加载/续写，但必须使用统一 sigma 和非周期轴。

并行解析按行区间分工，最终按原顺序装配。网格重建按第一轴划分格点区域，每个格点仍按时间顺序累加，串并行结果逐位一致。偏势文件应与匹配步号的宿主 checkpoint 一起保存；当前是单模拟写入格式。

## 宿主接口

详见 [ENGINE_ADAPTER.md](../graph/ENGINE_ADAPTER.md#bias-session)。

`BiasHost` 只接收 root 标志、`broadcast(double*,size_t)` 和日志回调。数据包为 `[V,dV/ds0,...]`，MPI 每次求值广播一次。单进程 GPUMD 可省略广播。核心抛 C++ 异常，由宿主转换报错；没有 LAMMPS、MPI 或 CUDA 头依赖。

GPUMD 具体 EnhancedSampling 适配尚未接入。CV/Graph 的值、种子和宿主 force/virial 装配仍沿用现有 view 层。

## 构建与验证

```sh
cmake --preset host-tests
cmake --build --preset host-tests -j 4
ctest --preset host-tests

# 需要 numpy；偏势求值使用刚构建的生产 C++ 库
python benchmarks/highdim/scientific_validation.py \
 --library build/host-tests/benchmarks/graph/libzqc_bias_probe.so \
 --output build/highdim-science.json

# 需要 numpy、mpi4py、LAMMPS Python 包，以及已构建的插件
python benchmarks/highdim/run.py \
 --plugin build/lammps/fix_crystallize_plugin.so \
 --np 1 2 4 --output build/highdim-validation

# Host 并行加载、网格/列表交叉
build/host-tests/benchmarks/graph/zqc_bias_rebuild_benchmark

# CUDA 仅供性能评估原型
cmake --preset graph-tests
cmake --build --preset graph-tests --target zqc_bias_evaluate_benchmark
build/graph-tests/benchmarks/graph/zqc_bias_evaluate_benchmark
```

LAMMPS 测试输出目录必须尚不存在。结果保存输入、HILLS、日志、差分误差和插件 SHA-256。[RESULTS.md](RESULTS.md) 说明数据、测试含义和性能取舍。
