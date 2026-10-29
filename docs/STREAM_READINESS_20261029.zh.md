# SUM/safe_div 跨流就绪复验 — 2026-10-29

本轮在独立审查使用的同一台 WSL / RTX 4080 SUPER / driver 610.88 上复现并修复了
`zqc_graph_sum_safe_div_mpi_{1,2,4}` 的失败。直接原因是测试夹具把裸 pageable
`cudaMemcpy(H2D)` 返回误当作设备写入完成，随后由 nonblocking Graph 执行流读取同一缓冲区。
改用已有的阻塞式 `GpuBuffer::upload_from` 后，默认并发环境中的三条测试各重复 50 次全部通过。

改动限于测试 I/O、新增流依赖回归及接口注释；`src/` 生产实现、计算公式和数值容差未改。
没有在实现中强制 `CUDA_DEVICE_MAX_CONNECTIONS=1`、`CUDA_LAUNCH_BLOCKING=1` 或全设备同步。

## 复现与隔离

源码基线为 `73b409013d606004ae332b131ec3bc8c45b9c1b1`。
原 `build/graph-tests/benchmarks/graph/zqc_graph_sum_safe_div_test` 的 SHA256 为
`73bf1263e3f3728a898690557909be73e2ddf413e534ad1e7c49a2d90cff08ab`。

1. 两个 CUDA 诊断环境变量均未设置：原二进制的 1/2/4 ranks 三条测试全部失败。
   `until-fail:2` 在每条的第一次失败即停止，实际是 3 次失败，不能计成 6 次。
2. 只设 `CUDA_DEVICE_MAX_CONNECTIONS=1`：原二进制每条重复两次，6/6 通过。
3. 从原测试源复制隔离版本，只在原 `put()` 的 H2D 后补一条 `cudaStreamSynchronize(0)`，
   重新编译测试对象，链接未修改的原 STATIC Graph 库；默认环境下 1/2/4 ranks 各跑五次，15/15 通过。
   该隔离版本只保存在外部证据目录，未合入源码树。

这把本次失败定位到了测试上传和消费之间缺少完成依赖；不能从 `MAX_CONNECTIONS=1` 的改善
直接推断是 SUM 生产核有缺陷。CUDA 的“同步”H2D API 对 pageable Host 内存可能在暂存后返回，
而 DMA 尚未完成，见 [CUDA Runtime 同步语义](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html)。
默认流与 nonblocking 流之间也不能凭调用先后假定设备依赖。

原测试的风险有两处：有限差分循环上传新的输入后 Graph 仍可能读取旧值；
测试上传梯度清零值后，DMA 可能晚于 backward 写回，覆盖计算结果。这分别对应旧 forward 值和零梯度。

## 实际修正与生产路径核对

- `benchmarks/graph/graph_sum_safe_div.cu`：`put/read` 接收缓冲区，分别使用
  `upload_from/download_to`；返回前等待指定流，下载也遵守缓冲区已有的就绪依赖。
- `benchmarks/graph/graph_ave_mpi.cu`：修正相同的测试上传模式，数值判据不变。
- `benchmarks/graph/stream_contract.cu`：新增 SUM、AVE 和嵌套 safe_div 的延迟生产者测试。
  预热后交替使用 legacy 默认流与显式 nonblocking 流，设备 kernel 写入输入与梯度哨兵，
  `record_write` 发布完成事件，Graph 等待后执行；另一条流下载梯度。
  六轮覆盖事件复用，并用独立 CPU 公式核对 forward、梯度及 `Leaf::clear()` 后的零值。
- `include/utils/zqc_graph.cuh`：说明外部写入必须已完成或发布就绪事件；裸指针写入本身不会自动更新事件。

本次失败走的是 `GraphExecution` 融合路径。该路径在执行流中等待 Leaf 就绪、启动融合 kernel、
下载规约输入，再同步执行流后进入 Host/MPI 规约；它通过 `reduce_host/reduction_seed`
处理 SUM/AVE，没有调用独立节点的 `SUM/AVE::forward/backward` 默认流 kernel。
`Leaf::clear()` 原实现也已在返回前同步其默认流清零。因此本轮没有修改这些生产原语。
这项结论限定于本次复现路径，不等于证明项目所有异步调用均不存在竞态。

## 当前实测结果

以下默认环境均显式移除了 `CUDA_DEVICE_MAX_CONNECTIONS` 和 `CUDA_LAUNCH_BLOCKING`。
全新构建与普通预设目录都保留了命令、退出码、原始 CTest 日志和 XML。

| 范围 | 实际执行结果 | 原始日志 |
|---|---|---|
| 全新 Graph 构建（含 NVRTC） | 36/36 通过 | `full-graph-default.log` |
| 标准 `graph-tests` 预设重新构建 | 36/36 通过 | `preset-ctest.log` |
| SUM/safe_div 默认环境，1/2/4 ranks 各 50 次 | 150/150 通过 | `sum-default-repeat50.log` |
| AVE 默认环境，1/2/4 ranks 各 20 次 | 60/60 通过 | `ave-default-repeat20.log` |
| 流契约默认环境，20 次，每次含新增六轮 | 20/20 通过 | `stream-default-repeat20.log` |
| 故意删除新增生产者事件的反例 | 3/3 正确检出旧值并失败 | `negative-control-{1,2,3}.log` |
| SUM/safe_div，MAX_CONNECTIONS=32，各 20 次 | 60/60 通过 | `sum-max32-repeat20.log` |
| 全新 Host 构建 | 6/6 通过 | `host-tests.log` |

重复次数按原始日志中的逐次结果统计。CTest 的重复运行 XML 仅保留每个测试的末次记录，
不能单独用 XML 的 tests 数量作为执行总数。MAX_CONNECTIONS=32 的结果作为附加并发对照，
与默认环境结果分开计数。本轮未重新运行 A100 性能或 LAMMPS 全矩阵；生产代码和阶段插件未替换。

反例仅在外部复制的测试源中删除新增两次 `record_write`，仍链接同一生产库；三次均报告
`SUM read stale producer values` 并返回失败。它确认延迟生产者用例能检出依赖缺失，
而不只是依赖启动时序碰巧通过；该反例未合入源码树。

环境为 CUDA 12.6.20、GCC 11.4、OpenMPI 5.0.6、WSL Ubuntu 22.04。
MPI 采用 `self,tcp` / `ob1`，`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`；
库搜索路径显式绑定同一套 MPI 与 CUDA，避免误加载其他 LAMMPS/DeepMD 同名库。

## 复验命令

先按本机依赖设置编译器、MPI 和 CUDA。不要把本次绝对路径当作可移植安装配置。
在已配置的统一树中：

```bash
unset CUDA_DEVICE_MAX_CONNECTIONS CUDA_LAUNCH_BLOCKING
cmake --preset graph-tests
cmake --build --preset graph-tests --parallel 4
ctest --preset graph-tests --output-on-failure
ctest --test-dir build/graph-tests --output-on-failure \
  -R '^zqc_graph_sum_safe_div_mpi_(1|2|4)$' --repeat until-fail:50
ctest --test-dir build/graph-tests --output-on-failure \
  -R '^zqc_graph_ave_mpi_(1|2|4)$' --repeat until-fail:20
ctest --test-dir build/graph-tests --output-on-failure \
  -R '^zqc_stream_contract$' --repeat until-fail:20
CUDA_DEVICE_MAX_CONNECTIONS=32 ctest --test-dir build/graph-tests --output-on-failure \
  -R '^zqc_graph_sum_safe_div_mpi_(1|2|4)$' --repeat until-fail:20
```

`validate_fix.py` 另以独立的 `build/stream-readiness-20261029/{graph,host}` 完成全新构建，
`validate_presets.py` 在保存旧二进制和 STATIC 库后更新标准 `build/graph-tests`。
完整依赖路径和实际命令保存在外部证据的 JSON 中。

## 历史记录与交付范围

- 2026-10-28 17:24 的 `final-weave-ctest/ctest.log`：34/34 是早期清单，未包含新增 SUM/safe_div 三条。
- 2026-10-28 21:17 的 `build/restoration-20261028/ctest-37.log`：当晚清单为 37 项，包含三条
  `SUM_SAFE_DIV_PASS`。它支持“当时通过”，不能取代当前复现。
- 恢复提交 `dd8da2c` 时间为 2026-10-28 21:28:45 +0800，早期 34 项的记录不能覆盖该提交的恢复功能。
- 原 Windows 阶段归档没有收录上述恢复日志，本次补存原日志与哈希，旧归档保持原样。
  当前 36 项 `graph-tests` 与历史带适配器的 37/40 项清单范围不同，不能以总数相互替代。

本轮证据保存于 Windows
`C:/Users/Administrator/.codex/reports/metadtoy-stream-readiness-20261029/`，
WSL 对应 `/mnt/c/Users/Administrator/.codex/reports/metadtoy-stream-readiness-20261029/`。
其中保留修复前失败日志、旧二进制/STATIC 库及哈希、隔离实验、完整复验、历史原日志，
并以 `manifest.json` 固定最终源码提交和归档哈希。报告和大型二进制留在源码库之外。
