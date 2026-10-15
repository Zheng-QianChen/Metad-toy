# Stru_factor 的完整 LAMMPS Graph 对照

这是实验性接入验证。普通 Stru_factor 现在通过 `EXECUTOR GRAPH` 使用：CV 提供逐原子值和梯度回传，均值与 switching function 由图节点处理。全局 executor 默认值仍为 LEGACY，以兼容其他 CV，因此使用普通 Stru_factor 时须显式选择 GRAPH。

```lammps
fix m all metad EXECUTOR GRAPH GAUSSIAN 0.10 0.10 10.0 PACE 100000 &
CAL NAME S1 STRU_FACTOR all cutoff_r 3.30 q_factor 1.7 d_block_size 128 &
CAL NAME S2 STRU_FACTOR all cutoff_r 3.30 q_factor 2.3 d_block_size 128 &
SYMBOL s1 S1.AVE SYMBOL s2 S2.AVE &
CV_dim 2 DIM 1 -5 5 500 "s1*s2" DIM 2 -5 5 500 "s1+s2"
```

## 当前边界

- Graph 已接入普通/化学 Stru_factor、CSS 和 WEIGHT_CHEM。这些 CV 内部原来的 AVE／COUNT 回调及 CV 值 `SW_func` 参数已移除；`S1.AVE` 仍是有效的图输入写法。其他尚无原始梯度接口的 CV 在选择 GRAPH 时会明确报错。后续迁移见 [化学结构因子](../strufactor_chem/README.md) 和 [WEIGHT_CHEM](../weighted_chem/README.md)；本文早期插件对照保留当时的接口与结果。
- `SYMBOL s S1.AVE` / `.MEAN` 是按 **S1 自己的全局 owned 组成员数** 求平均；ghost 不计数。
- `SYMBOL x S1.RAW`（或省略方法）得到公共压缩并集上的逐原子 Leaf。可以写 `AVE(x*x)`、`AVE(fermi(x, ...))` 等现有 Graph 表达式；这里 `AVE(expr)` 的分母是公共并集 owned 数，而不是某个 CV 自己的组数。组外输入补零，非线性函数可能把零变为非零，应按这个语义选表达式。
- 每个 DIM 最终必须是 Global 标量。目前仅支持 Graph 已实现的语法与节点，并非完整 ExprTk 语法兼容；不支持的表达式报错，不会偷偷回退。
- switching function 使用 Graph 的 `fermi`、`tanh`、`rational` 等节点，作用于 RAW 输入；`CAL ... SW_FUNC` 命名注册表尚未连接到 Graph。
- `get_env()` 在 GPU 筛选 cutoff 内的邻居，并缓存位移、距离平方和邻居 localid。同组同 cutoff 的 CV 共用这份几何数据，前向和反向均复用；筛选结果保存在显存。
- Graph 每轮先统一使输入 CV 的共享环境失效，再调用全部 `base_calc()`；第一个使用者更新环境，其他 CV 复用。每轮都准备，保证同一 timestep 修改坐标后重新求值也会刷新。
- 通用 `AVE(expr)` 与 `CV.AVE` 都把 `{local_sum, local_count}` 打包为 `double[2]`，每个 mean 只调用一次 `MPI_Allreduce`；不同 mean 仍各自规约。计数必须小于 `2^53`，全局空组报错。
- ghost 力与 virial 通过 `FixMetadynamics::communicate_reverse(*this)` 回传，每次只发送当前 CV 的 9 个 double；通信宽度与数值回归见 [CV 通信验证](../cv_communication/README.md)。
- CV 先完成 `base_calc()`，再装配 Leaf。当前用默认 CUDA 流的顺序依赖。全部输出种子一起反传，每个 CV 执行一次 `apply_adjoint()`。
- 原子迁移/组掩码变化会刷新装配表和有效长度。显存按容量复用，NVRTC module 不因长度改变而重编译。
- 本例的 `.AVE` 之后乘加发生在 Host。融合核负责规约前的逐元素计算和梯度分发，**Stru_factor 的邻居计算 kernel 仍独立执行**。

## 构建（本机）

```sh
cmake -S . -B build/graph-lammps \
  -DCMAKE_BUILD_TYPE=Release -DENABLE_DEBUG=OFF -DUSE_ML_CVS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCLEAN_HOST_COMPILER=/usr/bin/g++ \
  -DMPI_HOME=/home/zqc/apps/deepmd \
  -DLAMMPS_SOURCE_DIR=/home/zqc/apps/lammps/lammps-stable_29Aug2024/src \
  -DLAMMPS_LIB_DIR=/home/zqc/apps/deepmd/lib
OMPI_CXX=/usr/bin/g++ cmake --build build/graph-lammps -j 3
```

使用与 LAMMPS 相同的 MPI。此机的 Conda MPI wrapper 默认宿主编译器不在 PATH，故构建时显式设置 `OMPI_CXX`。

## 验证及计时

以下命令从仓库根目录运行；输出目录必须是新的目录，以免覆盖原始证据。`run.py` 可以通过 `--plugin`、`--lmp`、`--mpi` 改用其他路径。

```sh
python3 benchmarks/strufactor_lammps_graph/run.py --validate --cells 6 \
  --np 1 2 4 --repeats 1 --output build/graph-lammps/check-new
python3 benchmarks/strufactor_lammps_graph/groups.py \
  --initial build/graph-lammps/check-new/initial.data \
  --output build/graph-lammps/groups-new
python3 benchmarks/strufactor_lammps_graph/same_step.py \
  --initial build/graph-lammps/check-new/initial.data \
  --output build/graph-lammps/same-step-new
python3 benchmarks/strufactor_lammps_graph/run.py --cells 10 --steps 3000 \
  --warmup 200 --np 1 2 4 --repeats 3 --motion translate \
  --output build/graph-lammps/timing-new
```

`run.py` 默认只运行 GRAPH；显式 `--executors LEGACY GRAPH` 和 `compare.py` 用于仍含旧普通 Stru_factor 标量入口的历史插件。整理前后对照需要预先保留的插件，以下脚本比较两版 Graph 受力和化学加权兼容路径：

```sh
python3 benchmarks/strufactor_lammps_graph/cleanup_regression.py \
  --baseline-plugin build/graph-lammps/before-strufactor-cleanup.so \
  --initial build/graph-lammps/check-new/initial.data \
  --output build/graph-lammps/cleanup-check-new
```

主计时采用固定编号的 4000 原子 FCC 初态、两个 Stru_factor、两个 DIM、`pair_style zero`。先产生 hill 并扰动结构，随后 `fix move linear` 让两条路径接受完全相同的非零受力工作量；偏置力被完整计算，但不反馈到规定的运动。平移仍会触发邻居重建及 rank 间迁移。它是完整 LAMMPS timestep 的 CV 路径基准，不代表带真实势函数的 NVE/NPT 性能。

每个配置先预热 200 步，再计 3000 步，重复三次并交替顺序。主指标为最后一段 `run` 的 LAMMPS `Loop time / steps`，不含 NVRTC/setup、进程启动及预热。`wall_s` 单独保留冷启动全过程时间。所有 MPI rank 共享 GPU 0、`OMP_NUM_THREADS=1`、`--bind-to none`，不是多 GPU 强缩放。

可选 `--motion nve` 用于额外自然演化检查。自由演化可能离开 hill，使 Graph 的零种子分支跳过邻居反向计算，因此不能把这种混合工作量当作持续非零受力的主要计时。短程非零受力验证与自然演化计时分别保存。

`groups.py` 验证不同且非连续组、空组 rank、组外邻居受力、总力守恒和 1/2/4 rank 一致性，并用独立 CPU 求和检查 CV。通用 Graph 的 MPI 测试还覆盖输入长度变化、失效的旧 forward、只编译一次、重复日志句柄以及编译失败诊断。

## NVRTC 日志

rank 0 同时写到 LAMMPS 的 screen 和 logfile，不限于 `metad_debug_logging.txt`。每个 rank 各自编译，记录的是 rank 0 的耗时。

```text
[MetaD] executor=GRAPH dimensions=2
[MetaD Graph] NVRTC begin: version=12.6 --gpu-architecture=sm_89 source_bytes=... inputs=2 reductions=2
[MetaD Graph] NVRTC compiler log:
...
[MetaD Graph] NVRTC ready: compile_ms=... load_ms=... cubin_bytes=... kernels=cv_forward,cv_backward (reused across timesteps)
```

成功时也保留编译器 warning。失败时输出 compiler log 和 `NVRTC/module failed`，随后通过 LAMMPS 的 rank-local 错误入口终止 communicator，避免其他 rank 一直等待规约。日志在编译前后刷新，不逐步刷屏。

原始输入、screen、log.lammps、dump 和逐次计时 JSON 保存在 `build/graph-lammps/`（已由仓库 build 忽略规则覆盖）。本目录的 JSON 文件保存汇总与验证结果，便于审查。

## 初次接入结果与限制（共享环境修复前）

`results.json` 是相同规定轨迹的主计时，保存全部 18 个样本、中位数、最小/最大值和编译耗时；`validation.json` 与 `groups.json` 保存数值验证；`environment.json` 保存硬件、构建参数和所测插件 SHA-256。

本机主计时中位数（原路径 → Graph）：1 rank 为 2.127 → 2.496 ms/step；2 rank 为 3.589 → 3.804 ms/step；4 rank 为 5.969 → 5.823 ms/step。Graph 分别约慢 17.3%、慢 6.0%、快 2.4%。共享桌面 GPU 下的三次重复不足以支持普遍加速结论，尤其应结合样本范围判断几个百分点的差距。

这不是相同工作量的设备微基准：Graph 还计算和回传 per-atom virial，而旧 Stru_factor AVE 没有相应的 virial 累加。当前用例也只有两个 AVE 后的 Host 标量组合；它检验的是完整接入成本，不是复杂逐原子算式的融合上限。

短程非零偏置力对照的最大绝对力差为 5.285e-12，Graph 跨 rank 的力差为 1.36e-19；不同组的独立 CPU CV 参考误差为 4.45e-15。输入长度随 MPI 迁移变化而只编译一次，已在实际 LAMMPS 和独立 MPI 回归中验证。

额外的自由 NVE 长测保存在 `build/graph-lammps/timing/`。约 200 步时两条路径 CV 差在 3e-12 量级，3200 步后最大差约 5.19e-5，且均已离开 hill、偏置能量为零。差异尚未作根因归因，不将其记为长期轨迹等价通过；主计时使用规定轨迹避免该差异及零种子快速返回干扰。

可重新生成主汇总：

```sh
python3 benchmarks/strufactor_lammps_graph/summarize.py \
  build/graph-lammps/timing-matched \
  --output benchmarks/strufactor_lammps_graph/results.json
```

## 共享环境与 mean 规约修复

共享环境的失效标记移到 Graph 每轮的准备阶段，随后同环境的 CV 只触发一次几何更新。`base_calc()` 的独立调用保留自行失效行为。普通 `AVE` 和组内 `CV.AVE` 共用 `reduce_sum_count()`，空 rank 传入 `{0, 0}`；反向仍使用全局 count 分发种子。

`shared_mean_validation.json` 保存本次 9 项 CTest、完整 LAMMPS 的 1/2/4 rank 对照、不同组和同 timestep 重算结果。新增回归不增加 timestep，在两次 `run 0` 之间改变坐标，并分别对照独立 CPU 参考，避免缓存复用掩盖过时数据：

```sh
python3 benchmarks/strufactor_lammps_graph/same_step.py \
  --initial build/graph-lammps/shared-mean-validation/initial.data \
  --output build/graph-lammps/same-step-new
```

调用计数使用独立的 `LD_PRELOAD` 探针 `count_calls.cpp`，不修改插件且不参与性能测量。两个共享环境 CV、9 轮评估时，每 rank 的环境 kernel 从 18 次减到 9 次；mean 规约从 18 次单 double 加 18 次单计数，变为 18 次 double 二元组。在 1/2 rank 下分别验证。这里统计的是 collective 调用次数，不是网络报文数，也不包含 CUDA Driver API 启动的融合 kernel。

```sh
OMPI_CXX=/usr/bin/g++ /home/zqc/apps/deepmd/bin/mpicxx \
  -shared -fPIC benchmarks/strufactor_lammps_graph/count_calls.cpp \
  -I/usr/local/cuda/include -L/usr/local/cuda/lib64 \
  -Wl,--no-as-needed -lcudart -ldl -o build/graph-lammps/count_calls.so
# 对已有短测试输入执行一次；不用此探针的结果作为计时样本。
LD_PRELOAD="$PWD/build/graph-lammps/count_calls.so" \
  /home/zqc/apps/deepmd/bin/mpirun --bind-to none -np 2 \
  /home/zqc/apps/deepmd/bin/lmp_mpi -in <测试输入路径>
```

修复前后分别保留插件并交替运行同一规定轨迹：4000 原子、预热 200 步、计时 3000 步、每种配置 3 次。`shared_mean_results.json` 保存两个插件的 SHA-256、18 个原始样本和汇总。均为 Graph 路径，未加载计数探针，NVRTC 和 setup 不计入步耗时；未将此前另一批 LEGACY 计时混入本次比较。

- 1 rank：2.508 → 1.810 ms/step，中位数耗时降低 27.8%。
- 2 rank：3.743 → 2.871 ms/step，中位数耗时降低 23.3%。
- 4 rank：5.842 → 4.611 ms/step，中位数耗时降低 21.1%。

每一对样本的末端 CV 和非零偏置能量均通过 `1e-10` 绝对误差检查。本次是共享环境复用和 mean 通信合并的合计效果，未分别隔离两项收益。桌面共享 GPU 上的小规模基准不能直接外推到其它原子规模或真实势函数负载；初次接入中记录的长 NVE 轨迹差异仍保留为独立待查事项。

## Stru_factor 与环境整理

普通 Stru_factor 现在只负责逐原子前向和原始 adjoint 回传。CV 内部的 AVE／COUNT、旧导数缓冲和 forward 通信入口已删除；`S1.AVE` 由 Graph 的组内 mean 节点提供。原来的 `SW_func` CV 参数改为 RAW 输入上的 Graph 表达式。化学加权版本仍在子类维护原有标量入口，本次只做共享环境所需的适配。

`get_env()` 保留 GPU cutoff 筛选和邻居几何缓存：每个中心将有效邻居写到自己的候选段开头，前向和反向按有效数量遍历。同环境的多个 CV 复用缓存。删除筛选结果的三组 Host 镜像与 D2H 回传，移除多余同步，邻居编号缓冲按实际候选槽位容量申请。owned+ghost 坐标继续用 LAMMPS localid 索引。

`strufactor_cleanup_validation.json` 保存 9 项 CTest、整理前后 1/2/4 rank 完整 LAMMPS 对照、不同组和同一步重算的结果。普通 Graph 路径最大绝对力差约 5.42e-20，CV 和偏置能量一致；化学兼容路径的全组、单原子类型用例也通过对照。输入检查确认普通 Stru_factor 的旧 LEGACY 入口和旧 SW_func 参数给出明确错误，GRAPH 正常执行。

`strufactor_cleanup_results.json` 保存 18 个计时样本、插件 SHA-256、样本范围和末端 CV/能量。整理前基线已包含上一轮共享环境及 mean 通信修复；两版均走 Graph。4000 原子、预热 200 步、计时 3000 步、每配置三轮交替运行，不含 NVRTC/setup：

- 1 rank：1.796 → 1.237 ms/step，中位数耗时降低 31.1%。
- 2 rank：2.856 → 2.182 ms/step，中位数耗时降低 23.6%。
- 4 rank：4.612 → 3.451 ms/step，中位数耗时降低 25.2%。

这是本轮整体整理的收益，未单独比较有无筛选 kernel。原始日志在 `build/graph-lammps/strufactor-cleanup-*`，计时版本另存为 `after-strufactor-cleanup-timed.so`；随后仅调整了旧参数的错误输出及注释，并重建和检查输入。
