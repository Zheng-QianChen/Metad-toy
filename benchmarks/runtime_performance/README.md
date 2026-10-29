# 缓存、流与 Leaf 布局性能

最新结果见[暂停生产任务后的 Al3Sc 对照](AL3SC_PRODUCTION_ISOLATED.md)，已完成计时并恢复生产任务。

本机三阶段对照、完整数据及结论见 [2026-10-22 性能记录](RESULTS.md)。
与上次上传提交 `2d9ac78` 的直接对照见 [上传版本复测](UPLOADED_COMPARISON.md)：
覆盖两种邻居重建条件，并定位到大体系 Local Q 的 ghost 补表性能回退。

实际 Al3Sc 的 MEAM＋NPT＋LocalQL×CSS 参数对照见
[Al3Sc 生产参数测试](AL3SC_PRODUCTION.md)，同时比较生产专用插件、上传版和当前版。

`run.py` 比较多个插件版本的完整 LAMMPS timestep。每个插件必须使用相同的
LAMMPS/MPI ABI、Release 设置及 GPU 架构。脚本只需 Python 标准库；运行时使用
LAMMPS 可执行文件、MPI launcher、CUDA/NVRTC。

```bash
python3 benchmarks/runtime_performance/run.py \
  --variant before_cache=/path/to/before_cache/fix_crystallize_plugin.so \
  --variant before_streams=/path/to/before_streams/fix_crystallize_plugin.so \
  --variant current=/path/to/current/fix_crystallize_plugin.so \
  --lmp /path/to/lmp --mpi /path/to/mpirun --mpi-args="--bind-to none" \
  --cells 4 16 --np 1 2 --repeats 3 --warmup 100 --target-seconds 3 \
  --output build/runtime-performance
```

`--mpi-args` 按 launcher 选择；示例参数适用于 OpenMPI。脚本继承作业的 GPU
可见性设置，各次运行串行执行。第一个版本作为数值/加速比基线，最后一个版本
用来校准计时长度。输出目录须尚不存在。若只比较两个版本，传入两个 `--variant`。

`--neighbor-rebuild displacement`（默认）按原子位移判断是否重建邻居表；
`--neighbor-rebuild every-step` 对应 `neigh_modify every 1 delay 0 check no`，
用于测量邻居表和候选表频繁失效时的成本。两种条件应使用独立输出目录；
每次结果中的 `neighbor_builds` 保存 LAMMPS 在计时段实际报告的重建次数。

比较上传版本时，应将具体提交的完整源码导出到独立目录，使用相同 LAMMPS、
MPI、CUDA 架构和 Release 设置重新构建。记录提交号和插件 hash；本地开发快照
适合比较单个改造阶段，不能替代上传提交的版本对照。

## 工作负载与测量边界

- `shared`：相同 group/cutoff 的两个 Stru_factor，共享环境，图输出为均值的和与积。
- `mixed`：在 shared 上增加三元素 CSS、Q6、Local Q6，包含多个几何环境和通信。
- `css`、`q6`、`localq6`：各自独立运行一个 CV，复用 mixed 的径向参数，用于定位
  混合负载中的性能变化。图输出为该 CV 的均值和均值的线性变换。
- 每个规模为 FCC 的 `4*cells^3` 个原子，pair 势为 zero；主要测增强采样开销。
  这不是生产势函数下整个 MD 的加速比。
- 初始沉积一颗 hill，再扰动相对位置，保证偏置力非零。正式运行统一平移原子，
  保留偏置求值和通信工作，避免不同版本的积分误差改变后续轨迹。
- 每个配置先对照 CV、偏势、逐原子力和压力/virial，再计时。相同 rank 的各版本
  读取同一初态；MPI ranks 共用一张 GPU 的场景须单独解释。
- 首次 NVRTC 编译在 setup 完成，单独记录 compile/load 毫秒数。预热后取最后一段
  LAMMPS `Loop time`；进程总墙钟另列，包含启动和编译。
- 默认用最后一个版本的 200 步试跑校准公共步数，目标每次约 3 秒，上限 20000 步。
  所有版本使用相同步数。`--steps` 可显式固定长度；计时段在下一颗 hill 沉积前结束。
- 每轮随机化版本顺序，固定随机种子，报告中位数、样本范围及每次原始值。
  避免在测试时编译或启动其他 GPU 作业；WSL 显示负载、频率变化仍可能造成抖动。

`--pair-cutoff` 默认为 4.0，只改变底层 `pair_style zero` 的邻居覆盖范围。
CSS/Q6/Local Q6 的 FERMI 有效半径约 4.239；当底层原生 ghost 邻居行覆盖不足时，
适配器会补齐 Local Q 的环境。因此可用 `--pair-cutoff 5.0` 对照这条回退路径的成本，
同时保持 CV 参数和原子初态一致。该对照仍包含原生 pair 邻居表规模变化的影响。

输出包括 `metadata.json`（插件/脚本 SHA-256）、`validation.json`、`calibration.json`、
`results.json`、`summary.json`，以及各次输入、日志、初态和验证 dump。
LAMMPS 阶段计时可以辅助解释 Modify/Comm 占比，无法单独分离流并行与减少同步的收益。

## Leaf 暂存微基准

```bash
cmake --preset graph-tests
cmake --build --preset graph-tests --target zqc_leaf_layout_benchmark -j2
build/graph-tests/benchmarks/graph/zqc_leaf_layout_benchmark 2000 > build/leaf-layout.csv
```

需要 NVCC 可通过 PATH 或 `CUDACXX` 找到。程序比较连续、仅输入 stride=3、输入和梯度
均 stride=3/8 的同一 `AVE(x*x+0.3*x)` 图，覆盖 256、16384、262144 项。
每个布局验证值、梯度及 padding，预热后计时，每种布局重复三次并打乱顺序。

`graph_forward_backward_us` 包含完整 Graph 前向/反向、下载与 Host 规约；
`staging_device_interval_us` 单独测批量 Leaf gather/scatter 提交的 CUDA event 区间，
包含 Host 提交间隙，是摊销适配时间。当前 codegen 要求连续标量地址，stride 输入
由 Leaf 暂存；微基准比较的是这个边界的成本，不是已实现的 stride kernel 特化。

`summarize.py` 可将计时、快照清单和 Leaf CSV 归档，并核对长计时结束后的 CV/偏势/
压力是否一致；绘图另需 numpy 和 matplotlib。所有原始样本及验证阈值随结果保留。

若比较多个邻居重建条件，可使用 `summarize_conditions.py`，它会检查版本和配置
是否一致、每个配置的重复次数是否齐全，以及强制重建是否实际发生在每一步。
`--provenance` 指向记录基线提交、构建命令及插件 hash 的 JSON 文件。

```bash
python3 benchmarks/runtime_performance/summarize_conditions.py \
  --input displacement=build/performance/uploaded_comparison_displacement \
  --input every-step=build/performance/uploaded_comparison_every-step \
  --provenance build/performance/uploaded-provenance.json \
  --output build/performance/uploaded-report
```

## Al3Sc 生产参数复测

`al3sc_production.py` 使用已冻结的生产 checkpoint、HILLS、势函数和三版插件。
本机准备目录为 `build/al3sc-production-benchmark-20261022`；原始资产保留在 build
目录，精简后的计时、数值核对及 hash 归档在 `al3sc_20261022.json`。

```bash
python3 benchmarks/runtime_performance/al3sc_production.py \
  --prepared build/al3sc-production-benchmark-20261022 \
  --output build/al3sc-production-retest \
  --contention concurrent --warmup 100 --steps 500 --repeats 3
```

每个样本从相同的 restart/HILLS 开始，先预热，再运行正式计时段。它保留真实的
MEAM 势、NPT、WT 沉积和 thermo 周期，首轮 NVRTC 编译在计时段之外。
`--contention concurrent` 明确记录生产任务同时运行；`production-paused` 要求外部
生产进程已由使用者安排暂停，脚本会检查进程状态，但不会发送暂停或恢复信号。

准备目录需要 `assets/`、`provenance.json` 和 `<variant>_validate/input.lmp`，
其中 variant 为 production、uploaded、current。该入口针对本次生产快照：
6912 原子、1 rank、1450 K、1 bar、1 fs、PACE 200；换体系时应同时更新输入、
条件元数据和数值核对。它不会自行从正在写入的生产文件取样。

## Local Q ghost 缓存复测

2026-10-23 的 [实现、数值与性能结果](GHOST_CACHE_RESULTS.md) 比较本次修改前后工作区插件，
覆盖两种规模、1/2 rank、原生 ghost 半径对照和 Al₃Sc LocalQL×CSS 生产参数。
