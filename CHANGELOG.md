# 更新日志

## 2026-10-16 — 实验性功能验证版

本次将运行时编译的 Graph 接入完整 LAMMPS 偏置计算流程，并增加 CSS、化学结构因子与 WEIGHT_CHEM 的 Graph 接口、力和 virial 验证。版本定位为 experimental preview，供功能验证和接口迭代使用。

### Graph 与运行时编译

- 新增 `EXECUTOR GRAPH` 执行路径，连接 CV 原始值、表达式图、网格偏置及梯度回传。多个 DIM 可以共享节点，反向时累加各输出的贡献。
- 逐元素表达式生成 CUDA 前向/反向代码，交给 NVRTC 编译。底层设备函数集中在 `zqc_graph_device.cuh`，构建时通过 xxd 自动嵌入运行库，源码变更无需再维护一份字符串。
- Graph 负责拓扑、表达式解析与执行计划；节点构建按算符注册，具体实现拆入 `src/utils`。公开 Graph 接口与设备函数保留为两个头文件。
- 增加 Host 全局标量运算和 MPI `AVE`。每个 mean 将本地和与 owned 数量打包为 `double[2]`，通过一次 `MPI_Allreduce` 得到全局平均。
- 新增 CVAssembly：维护 owned/ghost localid 与公共压缩中心排列，各 CV 从设备数组装配 Leaf，缺失位置补零。原子迁移或组成员变化时刷新映射和长度，复用缓冲容量及已编译 module。
- NVRTC 编译信息写入 rank 0 的 LAMMPS screen/log，包括版本、架构、源码大小、编译及加载耗时和编译器日志。

### CV 与通信

- 普通 `STRU_FACTOR` 改为提供原始值与 adjoint 接口，CV 值的均值和 switching function 移到 Graph。设备 kernel 并回原 CV 实现，共享筛选后的邻居几何；每轮求值刷新环境，支持同一 timestep 修改坐标后重算。
- `STRU_FACTOR ... Chemical` 接入 Graph，保留元素权重与高斯后备映射。共享环境键包含完整权重配置，避免不同 Chem_map 相互覆盖。
- 新增多通道 `CSS`：每个通道指定邻居类型、径向函数、目标配位数和容差，支持多元素及多个统计半径。邻居类型接受编号或已定义的 LAMMPS type label，如 `Fe`、`Fe3+`。多个半径表示累计范围，壳层差值需另行定义。
- `WEIGHT_CHEM` 接入 Graph，保留 `1 + sum_j w_i*w_j*s(r_ij)` 的原始定义，补齐坐标力和六分量 virial。它与化学结构因子共用邻居几何和编译期 kernel 特化。
- 上述 CV 将 ghost 的三分量力和六分量 virial 合并为宽度 9 的 reverse communication。Fix 按当前 CV 的实际宽度通信，检查打包长度，并清理多余同步。
- Steinhardt 球谐函数及梯度改用单位方向与笛卡尔分量计算，调整邻居对计算和中间缓冲，补充 MPI 与 virial 回归记录。
- 修复 NeighHub 邻居请求匹配：同时检查 requestor 与请求编号，避免不同请求者之间编号碰撞。
- 修复稀疏网格递归中的局部边界数组遮蔽参数问题。

### 输入迁移

全局默认执行器仍为 `LEGACY`；使用普通/化学 `STRU_FACTOR`、`CSS`、`WEIGHT_CHEM` 时显式指定 `EXECUTOR GRAPH`。

```lammps
fix m all metad EXECUTOR GRAPH GAUSSIAN 0.10 0.10 10.0 PACE 100000 &
    CAL NAME S STRU_FACTOR all cutoff_r 3.30 q_factor 1.7 &
    SYMBOL s S.RAW &
    CV_dim 1 DIM 1 0 1 500 "AVE(fermi(s,-4,0.4))" &
    METAD_RESTART 0 WT 0
```

- `S.RAW` / `S.value` 提供公共压缩并集上的逐原子值；`S.AVE` / `S.MEAN` 按 S 自己的中心组 owned 数量求全局平均。
- 通用 `AVE(expr)` 按公共并集的 owned 数量平均。不同组缺失位置补零，非线性函数可能把零变为非零，需要按此语义选择表达式。
- 已迁移 CV 的旧 `COUNT` 与 CV 值 `SW_func` 接口移除；使用 Graph 的 `fermi`、`tanh`、`rational` 等算符。`AVE(fermi(...))` 是平均值，总计数规约尚未实现。
- 每个 DIM 最终必须得到 Global 标量。当前仅实现 Graph 自身的算符与语法，不兼容完整 ExprTk，也未提供通用向量运算或 Global 到 PerAtom 的广播。常数缩放可使用 `linear(x,scale,offset)`。
- Stru_factor/CSS/WEIGHT_CHEM 的邻居循环仍独立执行，Graph 融合逐元素表达式及梯度分发。CSS 的径向 `SW_FUNC` 定义继续用于邻居计算。

### 验证与性能

2026-10-16 上传前重新构建 Release LAMMPS 插件和独立 Graph 测试，**16/16 项 CTest 通过**。覆盖 NVRTC、开关函数、装配、MPI mean、普通/化学结构因子、CSS、WEIGHT_CHEM；MPI 场景包括 1/2/4 rank。验证环境为 CUDA 12.6、RTX 4080 SUPER、LAMMPS 29 Aug 2024，`USE_ML_CVS=OFF`。

本次开发保留的完整 LAMMPS 记录包括 12 组 CSS、15 组化学结构因子及 21 组 WEIGHT_CHEM 场景，涵盖 CPU 公式对照、组外受力、空 rank、不同中心组、共享环境、六分量压力与逐原子应力一致性。独立测试包含坐标力和六个应变方向的 virial 有限差分。

- [Graph 构建与测试](benchmarks/graph/README.md)
- [Stru_factor 完整流程与共享环境对照](benchmarks/strufactor_lammps_graph/README.md)
- [CSS 与类型标签](benchmarks/css/README.md)
- [化学结构因子](benchmarks/strufactor_chem/README.md)
- [WEIGHT_CHEM 与 virial](benchmarks/weighted_chem/README.md)
- [通信宽度验证](benchmarks/cv_communication/README.md)

SW_func 分离/融合对照使用 4000 原子、1000 步预热、20000 步计时，每种配置重复 5 次，计时排除 NVRTC 初始化。完整 LAMMPS 流程中，1 rank 的分离/融合中位耗时约为 763.56/765.78 微秒每步，2 rank 为 1421.97/1418.77 微秒每步；差异均小于 0.3%，没有观察到稳定的完整流程收益。生产路径保留独立 Graph 开关计算。微基准及数据见 [SW_func 性能对照](benchmarks/weighted_chem_switch/RESULTS.md)。

Stru_factor 的 pair 线程布局保留为 [独立实验](benchmarks/strufactor_pair_20261015/README.md)，生产实现继续按中心原子分配线程。历史 Q 性能和 MPI 测试记录一并归档；其中旧插件失败记录以报告日期为准。

### 当前限制

- 这是功能验证版本，接口仍可能调整；本轮未验证 ML CV 构建。
- 多 MPI rank 的测试共用一张 GPU，性能数据不代表多 GPU 扩展性。
- Stru_factor 历史长 NVE 对照存在尚未归因的轨迹差异；性能对照采用受控固定运动，详情见对应基准报告。
- PBMetaD、OPES 等高维偏置方案仍处于讨论阶段，本次没有加入这些算法。

## 2026-10-14 — Graph 独立原型

- 提供独立图构建、源码打印、NVRTC 前向/反向及手写 kernel 对照入口。
- 统一设备函数源码与构建时嵌入，增加数学及 switching function 节点。
- 本次上传同时包含此前尚未推送的原型提交 `f70101c`；完整 LAMMPS 接入见上方 2026-10-16 更新。
