# Al3Sc LocalQL × CSS 生产参数性能对照

最新结果见[暂停生产任务后的 Al3Sc 对照](AL3SC_PRODUCTION_ISOLATED.md)，已完成计时并恢复生产任务。

2026-10-22，同一台 WSL2 工作站，使用实际生产任务的势函数、restart 和 HILLS。
当前版总步时中位数为 **65.0824 ms**，比生产专用旧插件减少 **3.65%**，比上次上传版减少 **1.46%**。
生产任务在测量期间继续运行，因此这是共享 CPU/GPU 条件下的结果，不能据此保证独占资源下同样的加速幅度。

## 版本与计时

每版重复三次；每次重新读取相同初态，预热 100 步，再计时 500 步。
取最后一段 LAMMPS `Loop time`，包含 MEAM、NPT、偏置求值、WT 沉积、邻居重建和 thermo 输出。
NVRTC 编译及启动位于计时段之前。三轮版本顺序保存在原始数据中。

| 版本 | 总步时中位数，ms | 三次范围，ms | Modify 中位数，ms/步 | MEAM Pair 中位数，ms/步 |
|---|---:|---:|---:|---:|
| 正在生产使用的专用 LOCALQL_CSS 插件 | 67.5458 | 67.4528–68.2352 | 7.1570 | 59.1220 |
| 上次上传 `2d9ac78`，Graph 接口 | 66.0436 | 65.7916–66.1026 | 5.2538 | 59.4460 |
| 当前工作树，Graph 接口 | 65.0824 | 64.6784–65.2770 | 4.4754 | 59.1880 |

三次原始总步时：

- 生产旧插件：68.2352、67.4528、67.5458 ms。
- 上传版：65.7916、66.1026、66.0436 ms。
- 当前版：65.0824、65.2770、64.6784 ms。

当前版 Modify 比生产旧版减少 37.47%，比上传版减少 14.82%。这个阶段包含偏置 fix、
NPT 等 fix 的执行及其中发生的等待和通信，不能直接当作纯 CV kernel 耗时。
Pair 已占当前总步时约 91%，所以整个 MD 的收益明显小于 Modify 的收益。
各阶段分别取中位数，它们的和不必等于总步时中位数。

当前版 NVRTC compile 中位数为 46.069 ms，上传版为 46.138 ms；每次启动均只编译一次。
这里的速度提升也不能单独归因于 Half、流并行或缓存中的某项变化：对照测量的是完整插件。

## 实际参数与等价表达式

- 6912 个原子，其中 Sc 1728 个；1 MPI rank，1 OpenMP 线程。
- MEAM Al–Sc 势，NPT 1450 K、1 bar，时间步 1 fs；Tdamp 0.05 ps、Pdamp 2 ps。
- 邻居 skin 2 Å，`delay 0 every 1 check yes`，通信 cutoff 12 Å。
- Local Q6：Sc 中心，径向 FERMI `r_0=3.50, alpha=-20, cutoff_eps_r=1e-12`。
- CSS：Sc 中心、type 2 邻居，径向 FERMI `r_0=3.30, alpha=-20, cutoff_eps=1e-12`；
  配位目标 0、宽度 0.45。
- 乘积开关 FERMI `r_0=0.35, alpha=15`。
- WT factor 50，sigma 0.00125，输入 height 0.2 eV，PACE 200，TEMP 1450。
- DIM 范围 [-0.02, 1.02]、8320 bins，Gaussian_Hill_type 0。
- thermo 每 200 步，包含 CV、逐原子描述符均值、总压力/物理压力/偏置压力。
- 初态为生产第 6220000 步，已有 31100 个 hill；各次运行结束于第 6220600 步。

生产专用实现使用原始 LocalQ6，计算 `p = LocalQ6 * CSS` 后输出 `mean(p * FERMI(p))`。
输入中 LQ6 自身的 `SW_FUNC_cv SW_f6` 不会套用到 JOINT 的 SOURCE 值上。
Graph 对应表达式为：

```text
q = LQ6.RAW
c = JOINT.RAW
s = AVE(q*c*fermi(q*c,0.35,15))
```

本次使用 Graph 内建 `fermi`。当前 LAMMPS Graph 包装层没有传入命名开关表，
因此 `sw_func(q*c,SW_product)` 会报未注册开关；直接传相同常数保持数学等价。
这一限制记录于准备目录的 `comparison_plan.json`。

本次 MEAM 的邻居覆盖范围足以提供 Local Q 所需的环境，与之前 `pair_style zero 4.0`
触发 ghost 补表的测试条件不同。每个计时段均发生 17 次邻居重建，新增 3 个 WT hill。

## 数值核对

初态逐原子比较（当前版与生产旧插件）：

| 项目 | 最大绝对差 |
|---|---:|
| 力分量 | 1.281e-11 eV/Å |
| LocalQ6 | 5.551e-17 |
| CSS | 3.109e-15 |
| LocalQ6 × CSS | 6.661e-16 |
| 软配位数 | 3.331e-15 |
| 偏置压力 | 2.928e-9 bar |

随后比较每轮全部 thermo 采样点及三颗新增 hill。当前版与同轮生产旧版的最大绝对差：

- CV：3.236e-11；偏置能：4.151e-8 eV。
- 偏置压力：1.331e-5 bar；温度：1.398e-7 K。
- hill 中心：3.236e-11；height：9.132e-12 eV；沉积步数、sigma 完全一致。

初态检验隔离了力与描述符的实现差异；短轨迹检验覆盖积分和实际 WT 沉积。
数值通过本次容差检查，尚不构成长时间轨迹逐位一致的承诺。具体阈值及每个样本误差
保存在 JSON 的 `validation` 字段；9 次运行全部通过。

## 复现与归档

设备为 RTX 4080 SUPER、i7-14700KF，WSL2 可见 8 个逻辑 CPU；CUDA 12.6。
上传版和当前版均为 Release、CUDA architecture 89，使用相同 LAMMPS 可执行文件。
生产基线直接复制正在使用的二进制，完整 hash 如下：

| 版本 | 插件 SHA-256 |
|---|---|
| production | `974145ab6e87a2a125be808c5f29cd27e67a11875aa13690b1a269b6f6667599` |
| uploaded | `67b9ca52375508986a734af1de767f1d9bbc779df42e0b73a8fa1f1db17b1ef2` |
| current | `fc34be4362139c99588470ce50c2ce68ecf3fcee333761f0f50833bfaeb4afc6` |

[完整计时及数值归档](al3sc_20261022.json) 包含各次数据、版本 hash、准备时源码状态、
checkpoint 来源和条件元数据。[运行脚本](al3sc_production.py) 使用 Python 标准库。

本机原始材料位于 `build/al3sc-production-benchmark-20261022/`：

- `assets/`：冻结的 restart、HILLS、MEAM 参数及三版插件，每份都有 SHA-256。
- `*_validate/`：初态验证输入、输出与逐原子 dump。
- `concurrent-timing/`：9 次完整输入、日志、HILLS、结果、验证以及实际执行的脚本副本。
- `validate_and_archive.py`：核对轨迹/沉积并归档数据的脚本。

```bash
python3 benchmarks/runtime_performance/al3sc_production.py \
  --prepared build/al3sc-production-benchmark-20261022 \
  --output build/al3sc-production-retest \
  --contention concurrent --warmup 100 --steps 500 --repeats 3
```

本次测试只新增基准材料及文档；生产进程持续推进，测试使用独立的 HILLS 和 restart 副本。
500 步计时窗没有覆盖生产每 10000 步的轨迹写盘和分段重启开销，也未测试多 rank、
其他 GPU 或不同温度/体系规模。因此应把这组数字用于本机同款参数的比较。
