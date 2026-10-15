# 多元素 CSS

`include/CVsets/CV_CSS.h` 与 `src/cvs/CVs_CSS.cu` 实现独立的 CSS 描述量。
中心由 LAMMPS group 选择；一个 CSS 中的中心共用同一套目标环境。
邻居类型可以使用 LAMMPS atom type 编号或已定义的类型标签。二元、三元及更多元素使用相同接口。

## 定义

第 c 个通道指定邻居类型 t_c、径向函数 s_c、目标配位数 mu_c 和容差 sigma_c：

```text
n_ic = sum_(j: type[j] == t_c) s_c(r_ij)
CSS_i = exp(-0.5 * sum_c ((n_ic - mu_c) / sigma_c)^2)
```

不同通道的高斯评分相乘，所有目标同时满足时评分为 1。
每个通道的 sigma 表示平滑配位数容差；它和径向开关的距离参数分别设置。
增加通道会增加约束，评分的分布和偏置参数需要重新标定。

## 输入

```lammps
# 示例约定：type 1 = Al，type 2 = Sc；中心组为 Sc。
group Sc type 2

# 下列 CAL/SYMBOL 行放入 fix ... metad EXECUTOR GRAPH 的参数中。
CAL NAME first SW_FUNC FERMI r_0 3.30 alpha -20 &
CAL NAME outer SW_FUNC FERMI r_0 4.50 alpha -20 &
CAL NAME C CSS Sc &
    CHANNEL 1 first 12 2.0 &
    CHANNEL 2 first 0 0.45 &
    CHANNEL 2 outer 6 1.5 &
    cutoff_eps 1e-12 &
SYMBOL c C.AVE
```

`CHANNEL <邻居类型编号或标签> <径向开关名> <目标配位数> <容差>` 可重复任意次。
类型标签通过 `labelmap` 命令，或 data 文件中的 `Atom Type Labels` 定义。例如：

```lammps
# read_data/create_box 之后、fix ... metad 之前定义标签。
labelmap atom 1 Fe 2 "Fe3+" 3 O

# 以下行放入 fix ... metad 的参数中；first 为已定义的径向开关。
CAL NAME C CSS centers &
    CHANNEL Fe first 6 1.5 &
    CHANNEL "Fe3+" first 2 1.0 &
    CHANNEL O first 4 1.5
```

上述目标配位数仅为语法示例。编号和标签可混用；未定义的标签由 LAMMPS 报错。
类型标签区分大小写，`Fe` 和 `Fe3+` 是两个独立名称。它们表示用户定义的 atom type；
实际电荷由 LAMMPS 的原子属性单独设置，CSS 根据 type 进行选择。
单个原子的 atom ID 是另一个整数属性；选择特定中心原子可先使用 `group ... id ...`。
标签在读取 CSS 配置时解析成编号，后续 GPU 核函数继续使用整数类型比较。

每个开关必须先通过 `CAL NAME ... SW_FUNC ...` 定义。支持 FERMI、TANH、RATIONAL；
径向函数要求在大距离处衰减。`cutoff_eps` 默认 1e-12，用于求各通道的数值尾部截断半径。

上例的 outer 是累计范围，包含内层；它并不是自动减去 first 得到的壳层。
L1₂ 中第一壳层的 Sc 目标为 0，因此这里外层 Sc 的 6 来自第二壳层。
给同一元素添加多个半径，可约束多层化学环境；对于其它结构需重新指定目标。
这些数值展示输入语义，生产模拟应按实际温度、密度和径向函数标定平滑配位数目标。

三元环境按相同形式增加第三种类型，例如：

```lammps
CAL NAME C CSS centers &
    CHANNEL 1 first 6 1.5 &
    CHANNEL 2 first 4 1.5 &
    CHANNEL 3 first 2 1.0
```

此例表示每个中心周围分别期望 6、4、2 个邻居，目标数只是示例。
若 A 中心和 B 中心需要不同目标，分别声明 `CSS groupA ...` 与 `CSS groupB ...`，再在 Graph 中组合。
未列出的邻居类型不进入评分；需要排除某类近邻时，为它添加目标为 0 的通道。

原 Al3Sc CSS 的单通道定义为 `CSS Sc CHANNEL 2 first 0 0.45`。
本实现按推导的 cutoff 截断径向尾部；旧实现还逐项检查 `w < eps`，
边界舍入处可能有 eps 量级的差异。

## Graph 与逐原子输出

- `C.RAW` / `C.value`：每个中心的 CSS 分数。
- `C.AVE` / `C.MEAN`：只对中心组 owned 原子求全局均值，由 Graph 做 MPI 规约。
- `compute pc all metad/atom m C.css C.n1 C.n2`：输出评分与声明顺序对应的通道配位数。

坐标邻居可包含组外原子和 ghost。CVAssembly 提供中心排列；CSS 在 GPU 上计算配位数、
评分及反向力/virial，复用现有 Stru_fact_env 的几何数据与 NeighHub。
每次反向通过 Fix 的单 CV reverse 通信归还 ghost 的三分量力和六分量 virial。
逐原子诊断请求时才下载评分与配位数。

CSS 对同一元素、同一半径只记录总平滑配位数，不包含角度和连通性。
多通道能区分更多化学环境，但相同配位向量仍可对应不同结构。
高斯在远离目标时会饱和到接近零；sigma 需结合采样范围设置。

## 验证

独立 CUDA 测试 `benchmarks/graph/css.cu` 使用 CPU 独立公式检查单通道、二通道、
三元素四通道、同元素不同半径、无邻居中心、零中心、下溢评分和缓冲区边界。
对 24 个坐标自由度及 6 个应变方向做有限差分，检查坐标力、virial 与净力。

```bash
cmake --build build/graph -j 2
ctest --test-dir build/graph --output-on-failure
python benchmarks/css/run.py --output build/graph-lammps/css-validation
```

LAMMPS 测试使用实际 Graph 执行器与 NVRTC 编译，覆盖原单通道、三元、多个统计半径、
非连续小中心组及共享几何的两个 CSS，分别运行 1/2/4 个 MPI rank。
检查独立 CPU 前向、逐原子通道输出、非零偏置力、组外受力、净力及跨 rank 力/压力一致性。
测试输出目录需为新目录，汇总结果保存在该目录的 `results.json`。

静态区分验证：

```bash
python benchmarks/css/calibrate.py --output build/graph-lammps/css-calibration
```

固定晶格常数 4.05 的 FCC 几何，原单通道对理想 L1₂ 与孤立 Sc 的评分均接近 1；
示例三通道的评分分别为 0.9999994 与 0.0003355。这是静态对照，不代表实际温度下的分类准确率。
本轮 10 项 CTest、12 组完整 LAMMPS、静态对照和 6 项非法输入检查的结果记录在 `results.json`。

类型标签验证：

```bash
python benchmarks/css/type_labels.py --output build/graph-lammps/css-type-labels --np 2
```

使用两个 MPI rank 比较编号、带引号/不带引号的 `Fe3+` 标签、编号与标签混用，
以及 data 文件类型标签的 CSS、通道配位数、力、能量与压力。上述写法的输出一致。
同时检查未知标签、未定义 labelmap 和越界编号的报错。
