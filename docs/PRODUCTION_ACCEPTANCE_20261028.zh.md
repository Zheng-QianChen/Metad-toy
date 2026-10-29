# A100 生产运行验收

运行内核固定在提交 `8482cf744aaae566ab1b65fb09fe63415219b67a`。
本轮验证第 1 项的长轨迹、显存、反复续算与真实多卡/跨节点；不等同形核科学收敛。

## 平台与输入

- 曙光 N32-H，ARM64/Kunpeng，A100 PCIe 40GB，驱动 535.104.12、CUDA 12.2、
  GCC 11.3、OpenMPI 4.1.5，LAMMPS 22Jul2025 与匹配的 Weave/Kokkos。
- 真实 6,912 原子 Al–Sc、MEAM、Q6+LocalQL6+CSS 混合逐原子表达式；共享能量 SUM、
  默认沉积 MEAN、WT bias factor 10，401 网格节点、sigma=0.05、height=0.001 eV。
- GPU-aware MPI 保持开启。每 rank 1 个 OpenMP 线程，MPI 使用 ob1/self,tcp；
  本轮没有测 UCX/RDMA 吞吐，不把应用层设备指针通信等同于全网络零 Host 暂存。

## 长轨迹与显存

作业 **1602815**，节点 `paraai-n32-h-01-agent-15`，1 GPU；Slurm COMPLETED，exit 0。
1450 K NPT，dt=0.0002 ps，连续物理时间 20 ps：10 段各 10000 步、9 次新进程续算。
PACE=100，最终严格 1000 次沉积、末次步号 100000；全部历史校验通过。
每段都保存相互匹配的 MD restart 与偏势历史。

温度、势能、偏势、CV、体积和六分量偏势压力全部有限，原子数保持 6912，
各段 dangerous builds 均为 0。10 段每步耗时中位数 **8.982 ms**，范围 8.935–9.074 ms；
这包括 MEAM、三个 CV、在线 WT 的运行循环，不包含进程启动/重启文件时间。
它不是裸 MEAM 与 CV 的配对开销测试，不能直接从该数字推导 CV 百分比。

约每 2 秒采样 GPU 进程显存，10 个长段均有不少于 20 个样本。
剔除每进程前三个启动样本后，观测上限 **608 MiB**；各段前后四分位时间窗的
显存中位数增幅最大 **4 MiB**。完整曲线及每进程统计保留；仅支持该大小、热状态及
每进程 10000 步窗口下的内存稳定性，不声称无限时无泄漏或所有百万原子体系均通过。

## 真实多 GPU 与跨节点

作业 **1602816**，节点 `paraai-n32-h-01-agent-[231-232]`，每节点分配两张 GPU，
QOS=gpugpu；Slurm COMPLETED，exit 0，40 分钟限额内约 3 分钟结束。

每种布局比较连续 100 步与 5 段各 20 步的重新启动；同时与单卡连续状态比较。

| 布局 | MPI ranks | GPU 数 | 节点数 |
|---|---:|---:|---:|
| one | 1 | 1 | 1 |
| two-local | 2 | 2 | 1 |
| two-nodes | 2 | 2 | 2 |
| four | 4 | 4 | 2 |

全部 24 个短运行通过。最大重启状态误差 **7.10e-12**，最大跨布局状态误差
**1.83e-12**，重启后共享表最大误差 **1.46e-15**。
状态包含 id/type、周期等价位置、速度、力；另独立比较盒子和 thermo。
每组均为 50 次沉积、最终步号 100，所有历史记录校验通过。

hostfile、Slurm 节点/step 记录、每 rank local rank/CUDA_VISIBLE_DEVICES 与 LAMMPS
命令均保留。batch 节点的 nvidia-smi 可见两张实际 GPU；它不用于推断未记录的网络带宽。

## 初轮问题与证据

保留所有初轮失败输入/日志，不归因为生产内核错误：

1. 精简源码包遗漏两个 CMake 所需 probe 文件，配置阶段退出；补齐并逐文件核对哈希。
2. thermo 解析器遗漏表头前导空格，模拟本身正常完成；用真实输出修复读取。
3. write_restart 会重新映射周期坐标，裸坐标差约一个盒长 52.022 Å；按最小镜像比较后
   位置差小于 2.0e-13 Å，速度/力/盒子检查保持严格。
4. 多节点首次提交缺少 gpugpu QOS，按调度器要求补充；账户 GRES 限额导致等待长作业释放。

远端独立目录：
`/home/bingxing2/home/scx8q1h/work/MetadToy-Al3Sc-ARM64-A100-20261028/production-8482cf7-20261028`。
Windows 归档：`Work_MetaD/artifacts/metadtoy-production-20261028`，含 source、原始记录、
分析程序、535 个归档成员和 SHA256。其他用户作业与原生产目录未修改。

上述结果证明受测配置的运行和续算；没有把 20 ps 轨迹标作 Al3Sc 形核/FES 收敛。
百万原子性能对照属于第 3 项，见 [LocalQL 性能记录](LOCALQL_PROFILE_20261028.zh.md)。
