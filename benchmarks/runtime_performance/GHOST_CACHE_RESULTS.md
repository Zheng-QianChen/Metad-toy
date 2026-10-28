# Local Q ghost 候选缓存：验证与性能

日期：2026-10-23。对比基线为**本次修改前的工作区插件**，两版只有本次适配器缓存改动。
此前上传版本及旧生产插件的结果保留在 [Al₃Sc 历史对比](AL3SC_PRODUCTION_ISOLATED.md)。

## 修改及有效期

当 Local Q cutoff 超过 LAMMPS pair 提供的 ghost 行范围时，适配器需要补齐二跳环境。
之前每次求值都进行空间搜索；现在原始 ghost 候选表跟随宿主邻居拓扑版本复用。
组成员变化只重筛输出 CSR。宿主重建步使用 R+skin，延迟首次需求使用 R+2skin，
保持每步按当前坐标筛物理 cutoff。详细存储与生命周期见 [适配器约定](../graph/ENGINE_ADAPTER.md#local-q-ghost-候选缓存2026-10-23)。

## 计时

RTX 4080 SUPER，i7-14700KF，WSL Ubuntu 22.04，CUDA 12.6，LAMMPS stable_29Aug2024。
每个 rank 使用一个 CPU 线程，多 rank 共用一张 GPU。生产任务在授权暂停窗口内停止，
完成后自动恢复；窗口约 473.5 秒。每组随机排列版本顺序、各重复 3 次，表中为中位数。
首轮 NVRTC 与 100 步预热均在计时之外。

| 场景 | 原子数 | rank | 修改前 ms/步 | 修改后 ms/步 | 耗时变化 |
|---|---:|---:|---:|---:|---:|
| Local Q6 | 256 | 1 | 2.379 | 1.291 | -45.7% |
| Local Q6 | 256 | 2 | 3.815 | 3.122 | -18.2% |
| Local Q6 | 16384 | 1 | 21.738 | 5.287 | -75.7% |
| Local Q6 | 16384 | 2 | 16.898 | 6.677 | -60.5% |
| 混合 CV | 256 | 1 | 3.694 | 2.691 | -27.1% |
| 混合 CV | 256 | 2 | 6.632 | 5.880 | -11.3% |
| 混合 CV | 16384 | 1 | 27.531 | 10.570 | -61.6% |
| 混合 CV | 16384 | 2 | 22.599 | 12.901 | -42.9% |
| Local Q6，pair cutoff 5.0 | 16384 | 1 | 5.139 | 5.239 | +2.0% |
| Al₃Sc LocalQL×CSS，MEAM+NPT+WT | 6912 | 1 | 64.627 | 64.501 | -0.2% |

前八组采用 pair zero、相同初态与固定平移轨迹，计时 300 步；
pair cutoff=4.0 Å，CV 有效 cutoff≈4.239 Å，skin=0.35 Å，使用位移重建检查。
混合 CV 包含两个共享环境 Stru_factor、CSS、Q6、Local Q6。
两版所有这些计时段的宿主邻居重建次数均为 0；此前额外开销发生在适配器补行。
16384 原子、单 rank 的 Local Q Modify 阶段由 21.562 降至 5.121 ms/步。
这些是完整插件循环耗时，包含 Host、MPI 与 GPU，不等同于单 kernel 计时。

pair cutoff=5.0 的对照组保留相同 CV 和坐标。其原生 ghost 行已覆盖需要的范围，
两版耗时范围重叠，没有显示明确的性能改善。
Al₃Sc 的 pair cutoff=5.0 Å，Local Q 有效半径≈4.882 Å，也不触发该补行慢路径；
本次 Al₃Sc 耗时变化同样处于重复样本波动范围内。

Al₃Sc 每次从同一个 6220000 步 restart 和 31100 条 HILLS 开始，保留 1450 K、
1 bar、1 fs、skin=2 Å、comm cutoff=12 Å、WT factor=50、PACE=200 和原始
LocalQL×CSS 图表达式。预热 100 步后计时 500 步，每个样本包含 17 次邻居重建，
从初态至结束新增 3 条 hill。只读冻结资产，运行输出写入独立 benchmark 目录。

## 正确性

- 插件重新构建成功。
- 新增 ghost 缓存测试 12/12，现有缓存测试 27/27，均覆盖 1/2/4 rank。
- Local Q3/Q4/Q6、混合组、非线性 Graph 有限差分 15/15，四种差分步长。
- 共用邻居转换路径的 Half 表回归 486/486，覆盖 1/2/4 rank。
- Ghost 测试覆盖固定盒邻居进入 cutoff、延迟首次需求、动态组、NPT、倾斜盒、
  排序迁移与 skin 修改。逐步枚举实际二跳边，并比较缓存与冷重建。
  最大力差为 9.445e-14；CV 最大差为 5.551e-17。
- 性能夹具 18 次前置数值检查均通过；修改前后最大力差为 2.168e-19，
  最大压力差为 4.441e-16。非零偏置力检查通过；9 个计时配置的全部重复样本最终 CV/偏置能/压力也通过同样阈值。
- Al₃Sc 六个样本的相同初态 CV、偏置能、诊断列、偏置压力及最大力一致，
  最大差异小于 1e-8。轨迹会因浮点微差逐渐分叉，速度判断采用相同初态与相同步数。

## 复现

冻结前后插件后，在没有外部竞争负载的环境运行：

```bash
python3 benchmarks/runtime_performance/run.py \
  --variant before=/absolute/before.so --variant after=/absolute/after.so \
  --lmp /path/lmp --mpi /path/mpiexec \
  --cases localq6 mixed --cells 4 16 --np 1 2 \
  --warmup 100 --steps 300 --repeats 3 --output build/ghost-cache-timing

python benchmarks/finite_difference/ghost_cache.py \
  --plugin /absolute/after.so --np 1 2 4 --output build/ghost-cache-check
```

半径对照使用相同计时入口，加 `--pair-cutoff 5.0 --cases localq6 --cells 16 --np 1`。
[归档 JSON](ghost_cache_20261023.json) 保存全部计时样本、范围、验证误差及插件 hash。
本机完整输入、日志、实现差异和暂停/恢复凭据位于
`build/ghost-neighbor-cache-20261023/`；Al₃Sc 本次冻结运行器为该目录的
`al3sc_before_after.py`，资产位于 `al3sc-prepared/`。

修改前 SHA-256：`fc34be4362139c99588470ce50c2ce68ecf3fcee333761f0f50833bfaeb4afc6`。
修改后 SHA-256：`3564a9fa299dc5b52911930d7ba2da4d68f942a03228924ed263e13392f6f53c`。
