# 更新日志

## Graph 测试跨流就绪修正 — 2026-10-29

- 复现 SUM/safe_div MPI 三条失败；仅同步测试 H2D、链接原 STATIC 库的隔离版本通过 15/15。
- SUM/safe_div 与 AVE 测试改用 GpuBuffer 上传/下载契约；新增交替默认流/非阻塞流的延迟生产者回归。
- 标准及全新 Graph 预设均 36/36；默认环境 SUM/safe_div 重复 150 次、AVE 60 次、流契约 20 次通过，
  MAX_CONNECTIONS=32 附加 SUM 对照 60 次通过；Host 6/6。生产计算代码及数值容差未改。
- 区分早期 34 项、恢复当晚 37 项与当前复验记录，见 [完整证据](docs/STREAM_READINESS_20261029.zh.md)。

## 普通 LAMMPS 兼容验收 — 2026-10-29

- host 构建/验证不再强制提供 Weave；仅 Kokkos/Weave 构建传入 Kokkos_DIR。
- 全新关闭 Kokkos/GPU package 的普通宿主通过 1/2/4 ranks CV、ghost/cache、真实 MEAM、
  共享在线 WT/续算及原生 lmp 加载测试。明确区分 CPU LAMMPS 宿主与 CUDA CV 计算。
- 构建命令、Python 库固定方法与实际验收范围见 [普通版说明](docs/PLAIN_LAMMPS_20261029.zh.md)。

## 统一候选版 — 2026-10-28 开始，2026-10-29 收口

- 按 5→3→4→2→1→6 完成受测工程项；A100 10 万步、9 次续算与真实 1/2/4 GPU、跨两节点通过。
- 合并其他工作区的 Langevin、同表配对及 FES 控制模型，保留原始通过与失败结果；统一构建上重新采样的重加权二维 RMSE=0.01454 kBT。
- 用不加载 LAMMPS/MetaDtoy 的最小程序复现 OpenMPI 5.0.6 CUDA 延迟初始化/退出问题；提供不禁用 CUDA-aware MPI 的 early_cuda.py，在线 NPT/续算的 memcheck 与 initcheck 均 0 错误。
- 固定源码、依赖、插件哈希与验证边界；真实 Al3Sc 形核/FES 收敛未标记为完成。见 [发布入口](docs/RELEASE_20261028.zh.md)。

## 未发布 — 2026-10-28 设备路径、LocalQL 优化与在线共享偏势

- 删除 LocalQL 无用球谐缓存，直接借用设备 CSR；WSL 与 A100 同机重复对照通过。
- Q/Half 邻居准备与 Q 属性 forward 设备化，无 Weave 设备力合批仅暂存原生 reverse 边界包。
- 固定共享表支持 SUM；新增在线 Gaussian/WT、独立的能量和沉积归约、校验历史、冻结与续算。
- 默认一次沉积总基础高度由全局中心分摊。新增共享历史导出工具；普通 WT FES 工具不解释该格式。
- 共享功能的 40 项 CTest、三种构建 20 项真实 NPT/重启矩阵、CUDA 内存检查与导出表参考对照通过。
- 具体环境、误差和限制见 [共享验收](docs/SHARED_ONLINE_20261028.zh.md)、
  [性能验收](docs/LOCALQL_PROFILE_20261028.zh.md)及[设备覆盖](docs/DEVICE_COVERAGE_20261028.zh.md)。

## 未发布 — 2026-10-28 旧独有 CV 功能恢复

- 新增设备 LocalGeom COORD/TET 叶，恢复软配位总数与目标配位计数的 Graph 前后向和 virial。
- 新增 SUM 与 safe_div，恢复 LocalQL NSOLID/MEAN_SOLID/FRAC；CVGraph 的 SUM 按中心组掩蔽值和伴随，多组表达式可显式指定规约组。
- MEAN_RNN 保持诊断接口；零邻居不进入平均分母。Global Leaf 广播伴随跨 MPI 汇总。
- LocalGeom、LocalQL 旧统计、MPI/差分及兼容插件验证见 [恢复记录](docs/LEGACY_RESTORATION_20261028.zh.md)。

## 未发布 — 2026-10-28 CSS / LocalQL 设备邻居准备

- CSS/StruFactor 与 LocalQL 直接请求 Kokkos device full/ghost 邻居表，在 GPU 筛选、扫描并生成 CSR；保留 Host/全局 Q/Half 兼容路径。
- LocalQL 将 CSR 行重排和中心标记移至 GPU，只回读 I/J 中心数、有效 pair 数；CSS 中心顺序检查按拓扑/装配版本复用。
- 两层 halo、周期镜像 localid、动态组和每轮几何更新语义保持一致。新增设备邻居计数，扩展 ghost 缓存与 LocalQL 回归入口以支持 Kokkos/Weave。

## 未发布 — 2026-10-28 CV 压缩与 FES 验证

- 新增光滑 90%→10% 概率压缩对照，输出解析平坦势、共同沉积偏势与原坐标二维 FES 图。
- 增加独立 CPU 采样桥接和实际 CUDA 表/链式梯度核对，继续复用 HillBias、HILLS 及 CLI。
- 在 310 记录初轮与同链延长结果：普通 CV 和解析形状 MEAN 达标；学习压缩 MEAN 的覆盖和强 SUM 的重加权仍未达标，完整保留失败证据。
- 方法、复现与结果见 [CV 压缩测试](benchmarks/shared_frozen/COMPRESSION.md)。

## 未发布 — 2026-10-28 共享偏势平衡概率验证

- 新增耦合双阱统计原型：独立数值积分对照固定共享表和共同沉积后冻结的重加权分布。
- 复用 HillBias/HILLS 与实际 CUDA FrozenSharedBias，检查表插值、偏势导数、重启一致性和现有 CLI。
- 增加独立链误差、联合分布、权重 ESS、FES 覆盖门槛及错误重加权对照。
- 310 初轮 12 组中 10 组达标；两组强 SUM 在同一链延长 8 倍后达到原判据，保留初轮失败记录。
- 提供[复现说明](benchmarks/shared_frozen/PROBABILITY.md)和[实测结果](benchmarks/shared_frozen/PROBABILITY_RESULTS_310.md)。
  结论限于冻结平衡采样；主程序的生产 CV/偏势算法未改动。

## 未发布 — 2026-10-27 Graph 力与 virial 合并回传

- Graph 反向新增力输出批次：各 CV 贡献累加后统一完成一次 ghost reverse 和一次 owned 回写；Kokkos 最终力只上传一次。
- 统一收集 DISTANCE 的 owned 贡献，支持空中心 rank、取消和重复求值；批次外直接调用保留立即提交行为。
- 新增真实适配器通信计数与布局回归，检查六/九分量、Host/CUDA、AoS/SoA 和源数组复用。
- Al3Sc 普通/Kokkos 单双 rank 有限差分与新旧状态对照通过。说明与性能复测见 [310 记录](benchmarks/shared_frozen/FORCE_BATCH_310.md)。

## 未发布 — 2026-10-27 冻结共享原子偏势原型

- 增加 SHARED_FROZEN：一张一维固定表按中心组逐原子求值，再按全局 owned 数量取 MEAN；三次 Hermite 插值同时给出能量与导数。
- Graph 执行器支持 PerAtom 输出和数组反向种子，可与 Global 输出共同反传；CVGraph 对外提供明确的 owned 槽位选择集。
- 共享偏势层独立于 LAMMPS，使用 CUDA 查表/规约和 SumReducer；组外梯度为零，空 rank 参与全局规约，日志记录表内容指纹。
- Al3Sc LocalQ6×CSS：6 个普通/Kokkos 有限差分配置通过；6912 原子短积分、选择集刷新、restart 重放通过；完整 CUDA 内存检查 0 错误。
- 提供[输入与复现指南](benchmarks/shared_frozen/README.md)、解析测试表及[310 验证记录](benchmarks/shared_frozen/RESULTS_310.md)。此阶段使用冻结表验证接口与导数。

## 未发布 — 2026-10-27 LAMMPS CUDA/Kokkos 接入

- 增加可选 Kokkos 桥接和 lammps-kokkos 构建预设，注册 metad/kk 与 metad/kk/device，保留普通 LAMMPS 路径。
- 坐标/类型借用 Kokkos Device View 并保留真实 stride；输入带 CUDA event 等待钩子，力在 reverse 后回写 Device 并更新 DualView 状态。邻居候选、Host MPI 和 virial tally 沿用适配器接口。
- Kokkos 运行时由 liblammps 唯一持有，避免插件重复静态链接；固定原生 CUDA 构建边界，compute 支持 /kk Fix，并修正工厂函数 ABI。
- 补齐普通坐标上传 View 的就绪事件，删除未使用的 source_location 头文件。
- 为旧版 LAMMPS 的零 cutoff trim 缺陷提供源码检测和三文件回补。当前上游已修复，用户原有 LAMMPS 源码与库未修改。
- 310 单卡 1/2 rank：69 个有限差分配置、8 组连续积分、324 项邻居契约检查通过；最终 CUDA memcheck 为 0 错误。详情见 [接入指南](benchmarks/kokkos/README.md)和[验证记录](benchmarks/kokkos/RESULTS_310.md)。

## 未发布 — 2026-10-27 可安装的 metadtoy 命令

- 增加标准 Python 包及 metadtoy 命令入口；依赖清单与源码 CLI 共用。
- CMake 增加 cli 预设、metadtoy-cli 打包目标和 cli 安装组件；ZQC_CLI_ONLY 仅配置 Python 工具，ZQC_BUILD_CLI 可随现有构建启用，默认关闭。
- 安装前缀下管理独立 Python 环境和分析依赖；启动命令无需激活环境，支持同版本开发代码更新。首次配置、构建和安装的步骤见 [CLI 安装指南](tools/README.md)。
- 310 上 10 项安装检查通过，包含 FES/进度输出、路径空格、前缀覆盖、增量更新及 Host 库共存；本机 WSL 已实际安装并验证入口。详见 [安装记录](benchmarks/cli/INSTALLATION_310.md)。

## 未发布 — 2026-10-26 FES 批处理与终端进度

- COLVAR 增加有界批读取，保持 step 的 int64 精度；范围扫描、直方图及对数权重累计按批执行。
- 离线 c(t) 在规则网格上复用逐轴高斯因子与缓冲区，相邻沉积之间复用 offset，保持原积分网格和沉积时序。
- FES 命令新增 --progress auto/always/never、--sample-batch；交互终端默认显示阶段进度和预计剩余时间，进度事件可供后续 TUI 复用。
- 310 上 52 项回归测试通过，包括分批等价、周期网格核、沉积步边界、大整数步号、stdout/stderr 分离及计算取消。新旧版本交替计时与数值对照见 [FES 性能记录](benchmarks/cli/FES_PERFORMANCE_310.md)。

## 未发布 — 2026-10-26 轨迹重加权 FES

- FES 默认采用逐帧重加权，使用 log_weight、rbias 或 (bias-c(t))/kBT。新增 fes reconstruct，旧 hills preview --quantity fes 使用同一默认方法；原渐近估计通过 --method hills 保留。
- 为原生 1–3D 普通 WTMetaD 增加离线 c(t) 积分，按记录时序追加 hills，显式区分积分域与绘图域；PB/高维可读取已给出的校正权重。
- 导出 FES CSV/PNG、来源与参数 JSON、逐帧权重 CSV；保存样本数和权重 ESS，空 bin 留空，缺少动态归一化时明确报错。
- 310 上 45 项测试通过；已有 O2 轨迹的 180001 帧完成重建和 c(t) 网格加密检查。详见 [验证记录](benchmarks/cli/FES_RESULTS_310.md)。

## 2026-10-16 — Graph CV 与安装验证更新（experimental preview）

本次整合下方 Q、Local Q、DISTANCE 的 Graph 接入及数值验证，继续作为实验性功能验证版发布。

- 新增 `benchmarks/run.py` 的 smoke/regression/graph/o2 四个入口；smoke 使用 Python 标准库和 LAMMPS 可执行文件，自动生成小体系并检查插件、各 CV、NVRTC、力及 MPI 一致性。
- 当前数值验证与 O₂ 脚本移除作者机器的硬编码依赖路径，支持命令行、环境变量、CMake 缓存及 PATH 查找；MPI 附加参数可配置。
- 测试保存独立输出目录、日志、实际执行命令与插件摘要，失败返回非零退出码；O₂ 增加曲线误差、势阱位置和覆盖范围的验收标准。
- 新增 [安装测试指南](benchmarks/README.md)与 [发布验证记录](benchmarks/INSTALLATION_RESULTS.md)。

## 未发布 — DISTANCE 修复、Graph 接入与 O₂ 重建验证

- DISTANCE 改用全局 tagint ID 查找 owner，一次规约两端坐标和 owner 数；支持迁移、排序、倾斜盒最小镜像，补齐偏置力和六分量 virial，并校验缺失/重复 ID 与零距离。
- CV Graph 增加 Global 标量输入和 adjoint 回传接口；DISTANCE 可独立执行，或参与逐原子 CV 规约后的表达式。LEGACY 距离入口继续可用。
- 修复后 69 组有限差分、9 组线性叠加、46 组 MPI 状态对照、12 组距离边界/迁移检查、6 组异常输入检查及 19 项 CTest 通过；混合图增补更小差分步长验证二阶收敛。详见 [DISTANCE 验证](benchmarks/finite_difference/DISTANCE_RESULTS.md)。
- 新增 [O₂ Morse 示例](benchmarks/o2_distance/README.md)：3 条独立 2 ns 轨迹，以 DISTANCE 做 well-tempered MetaD，再从 HILLS 重建并补径向熵修正。1.05–1.60 Å 内平均曲线 RMSE 为 1.36 meV；附复跑脚本、曲线及参数记录。

## 未发布 — CV 与 Graph 有限差分验证

- 新增 `benchmarks/finite_difference`，在固定偏置历史下扫描三个差分步长，分别检查 CV 坐标导数、六分量应变导数、偏置力及 virial。
- 覆盖 Q3/4/6、Local Q3/4/6、普通/化学结构因子、WEIGHT_CHEM、CSS、不同中心组及线性/非线性组合；Graph 路径 54/54 组通过，9 组线性梯度叠加与 36 组 MPI 状态对照通过。
- 新增 Graph 独立 Jacobian/VJP 测试，覆盖 Host、NVRTC、MPI 空 rank、共享节点和重复反向；CTest 扩为 19 项并全部通过。
- 旧 DISTANCE 的 6 组检查失败：单 rank 缺少 virial，MPI 下误用本地下标导致距离/力错误；两距离线性组合在 2 rank 还出现非有限力。保留失败记录用于后续修复。
- 测试方法、阈值和逐 CV 数值见 [有限差分报告](benchmarks/finite_difference/RESULTS.md)。

## 未发布 — Local Q 的 Graph 接入

- `STEINH L 3/4/6` 接入 Graph，支持 `.RAW` / `.value`、图侧 `.AVE` / `.MEAN`，以及与普通 Q 等 CV 共享图表达式。
- Graph 的逐中心梯度沿 I→J→K 传播，复用原有几何导数与 virial 算法；Local Q forward 宽度为 0，reverse 每原子 9 个 double。
- 中心排列使用 CVAssembly 的 owned 前缀；修复同一步重算、共享环境不同 block size、空中心和 Local Q Host 值数组扩容／清零问题。
- 修复本机 LAMMPS 自定义 cutoff 未覆盖 ghost 邻居行导致的第二层漏邻居；按现有 halo 坐标补齐必要的 ghost 候选行，并保留邻居排除规则。
- Local Q 统一要求 `EXECUTOR GRAPH`。删除 CV 内部的 AVE／MEAN_SOLID、SW_FUNC／NSOLID、FRAC 回调、CPU 规约与 seed kernel；移除 `SW_FUNC_cv` 状态和解析入口，旧输入给出迁移提示。图侧 `.AVE` / `.MEAN`、`AVE(expr)` 和值开关继续可用，`SW_FUNC_r` 保留径向权重。
- 接入阶段完成独立 CPU 参考、有限差分、MPI 与两 rank CUDA memcheck。接口清理后重新构建插件：21 组 Local Graph、12 组混合 Graph、6 组普通 Q 回归、6 组清理前后 Graph 通信对照及 Local Q 的 6 项旧接口拒绝检查通过；既有 16 项 CTest 通过。详见 [输入及验证记录](benchmarks/localql_graph/README.md)。

## 未发布 — 非 Local Q 的 Graph 接入

- `STEINH Q 3/4/6` 增加 RAW/value、AVE/MEAN Graph 输入及逐中心 adjoint，径向开关保留在邻居计算中。
- 原子对反向接收两端 `dV/dq_i`，Q3/Q4/Q6 forward 包宽为 11/13/17；力和六分量 virial 继续使用宽度 9 的 reverse。
- 使用 CVAssembly 统一中心排列，刷新共享环境与新 run 的邻居缓存。修复 Q 数组扩容指针、不同线程块共享环境的尾部漏算和空 rank 的零网格启动问题。
- 接口清理后 18 组完整 LAMMPS 检查通过，覆盖 1/2/4 rank、独立 CPU 前向、坐标/应变有限差分、迁移及组成员变化；3 项旧接口报错检查、9 组清理前后对照和既有 16 项 CTest 通过。
- Q 统一要求 `EXECUTOR GRAPH`，删除内部 AVE／SW_FUNC 标量回调、规约与对应反向分支。`.AVE` / `.MEAN` 仍由 Graph 构建；值开关通过 `.RAW` 和图表达式定义，`SW_FUNC_r` 保留径向权重。旧 `SW_FUNC_cv` 输入会提示迁移。
- 该阶段曾将 Local Q 的规约状态移入 Local 类；随上方 Local Q Graph 接入及接口清理，相关旧规约已移除。详见 [Q 接入说明与验证记录](benchmarks/steinhardt_graph/README.md)。

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
