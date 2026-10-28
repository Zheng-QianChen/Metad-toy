# FES 批处理和终端进度验证（310，2026-10-26）

## 范围和方法

基线为本轮修改前的 CLI 源码快照，不是已发布二进制。改动集中在 COLVAR
批读取、FES 批累计、规则网格 c(t) 求值及终端进度。测试未启动 MD。
所有测试与计时通过 SSH 直接在 310（excelsior）运行。

- CPU：Intel Xeon Platinum 8163 @ 2.50 GHz。
- Python 3.10.6，NumPy 1.23.4；OMP_NUM_THREADS=1、OPENBLAS_NUM_THREADS=1。
- 每次计时启动新的 Python 进程，包括导入、两遍轨迹读取、SHA256 校验、权重/FES CSV、
  JSON 和 PNG 输出。每个用例先预热，随后交替新旧版本各三次，报告中位数。
- 采用温热文件/字体缓存。没有 GPU、跨机器或冷缓存计时；共享机器存在调度波动。
- cProfile 只用于定位热点，其耗时不参与下表。

## 完整命令耗时

| 用例 | 输入 / c(t) 网格 | 修改前 | 修改后 | 加速比 |
|---|---|---:|---:|---:|
| O₂，原生 WT HILLS 重加权 | 180001 选中帧、19800 hills，1024 点 | 10.149 s | 5.666 s | 1.79× |
| 2D 合成 WT 格式数据 | 5000 帧、500 hills，128² 点 | 1.324 s | 1.044 s | 1.27× |
| 3D 合成 WT 格式数据，输出二维边缘 FES | 3000 帧、300 hills，48³ 点 | 4.238 s | 2.071 s | 2.05× |
| O₂，读取已有 log_weight | 180001 帧，无 c(t) 回放 | 7.176 s | 4.217 s | 1.70× |

O₂ 输入为已有 GPUMD seed_73517 的 COLVAR/HILLS，范围为 step 200000–2000000；
直方图为 1.05–1.60 Å 的 120 个分箱，c(t) 积分域为 0.85–3.0 Å。
2D/3D 合成数据使用周期与非周期混合坐标，生成方法保存在 perf_fes.py。
这些合成数据用于核/批处理等价和性能验证，不代表真实采样收敛性验证。

优化包括：

1. COLVAR 解析共享校验逻辑，默认以 8192 帧为一批产生列数组，step 独立保存为 int64。
2. 范围扫描、权重和分箱索引按批计算；重复 bin 使用 logaddexp.at 累加，保持每帧贡献。
3. 规则积分网格使用逐轴高斯因子和复用缓冲区，减少重复计算与临时数组。
4. 相邻沉积之间复用 c(t)，每个沉积步仍按 hill.step <= sample.step 更新。

网格分辨率、积分范围和核定义维持不变。3D 结果仍来自完整三维积分，
不是减小维度或删掉高斯尾部的近似。

## 数值对照与回归

52 项 unittest 全部通过（6.680 s），包含原有 45 项测试，C++ bias 参考交叉测试未跳过。
新增验证覆盖：

- 批大小 1 / 7 / 8192 的 FES、权重及 ESS 等价；大于 2^53 的步号保持精确；
- 1D/2D/3D、周期/非周期高斯因子与原通用求值器对照；
- 同一步多 hill、批边界、沉积步前后 c(t) 的标量/批量一致；
- 进度阶段、ETA、默认非终端静默、stderr/stdout 分离；
- 计算中取消保留旧输出，done 在输出发布后触发。

新旧版本 CSV 数值最大绝对差（检查所有数值列，含每个 bin 的 ESS）：

| 用例 | FES CSV 所有数值列 | 逐帧权重 CSV | 全局 ESS 相对差 |
|---|---:|---:|---:|
| O₂ WT | 0 | 0 | 3.98e-13 |
| 2D 合成 | 9.24e-14 | 2.22e-15 | 0 |
| 3D 合成 | 4.26e-14 | 1.80e-15 | 0 |
| O₂ 已有权重 | 0 | 0 | 4.23e-13 |

整数计数、步号及空值位置要求完全一致；其余数值采用 atol=2e-12、rtol=2e-11。
全局 ESS 的差异来自浮点求和分组。

## 进度条开销

同一 O₂ WT 工作量交替开关进度各三次：

- 关闭：5.747 / 5.605 / 5.710 s，中位数 5.710 s。
- 开启：5.624 / 5.683 / 5.597 s，中位数 5.624 s。

这组数据没有测到超过运行波动的进度开销，不能把开启后的较低中位数解释为额外加速。
强制开启测试将 stderr 捕获为文本，覆盖渲染/输出成本；未单独测量真实终端绘制耗时。
交互终端刷新限流为约 0.15 s，重定向为约 1 s，阶段切换和完成会及时输出。

实际输出示例：

    重加权 [=====               ]  25.3% 45536/180001 frames 已用 1.2s 剩余约 3.4s HILLS 6400
    重加权 [===============     ]  79.9% 143840/180001 frames 已用 3.2s 剩余约 0.8s HILLS 16256
    重加权 [====================] 100.0% 180001/180001 frames 已用 3.9s

首次扫描显示已处理帧数；重加权阶段的总帧数来自扫描，ETA 对应当前阶段。
当前是 CLI 终端进度，独立 TUI 界面尚未实现；回调可直接提供给之后的 TUI。

## 复跑与证据

    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    export METADTOY_BIAS_REFERENCE=/path/to/libbias_reference.so
    python -m unittest discover -s benchmarks/cli -p 'test_*.py' -v

    python benchmarks/cli/perf_fes.py \
        --baseline /path/to/before-checkout --optimized /path/to/after-checkout \
        --o2 /path/to/seed_73517 --out /path/to/fes-perf-results --repeats 3

perf_fes.py 输出完整耗时、数值对照、源码 SHA256 和进度记录。
它同时生成 2D/3D 合成数据，不修改 O₂ 输入。

本次远端证据目录：
/home/saitang/work/MetadToy-gpumd-20261023/build/fes-perf-20261026

- [原始性能数据](fes_performance_310.json)
- [测试日志](fes_performance_tests_310.log)
- [实际进度输出](fes_progress_310.log)

本地另存 baseline-source.tgz、optimized-source.tgz、cProfile 摘要与执行脚本，
位于 Windows 工作区 work_MetaDtoy/fes_perf_20261026。
