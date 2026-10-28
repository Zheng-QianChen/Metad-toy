# LocalQL 缓存与设备邻居优化验收（2026-10-28）

对应完成顺序第 3 项。实现提交 `a576294ba7c0689857e3fa864a3fb3b84e69023f`；上一版本为 `dd8da2c`。本项已经完成同机性能与数值验收，不代表后续设备覆盖、生产稳定性或科学收敛已经完成。

## 实现

删除 LocalQL 前向写入、反向从不读取的逐 pair 球谐缓存。反向继续沿用原来的解析球谐计算；删除随之失去用途的 active-pair 前缀扫描和每轮末端计数回传。直接借用 Engine 的设备 CSR，通过 localid 映射原生行起止，避免二次复制整张候选表。I/J 标记合并到一次原子 OR；没有证据表明这项单独加速了环境核，不作该项性能声明。

位置与活动几何每次求值都更新，拓扑与组版本控制邻居缓存。周期镜像 localid、两层邻域、力和 virial 定义不变。每次求值仍有一个 pure-J 数量标量回传；没有宣称零 Host 同步。

## 同机性能

采用真实 Al–Sc MEAM、相同输入、表偏势、温压控制与排序设置；bare 为 MEAM+Weave，不含 CV，old 为前一设备邻居版本，new 为本次实现。三个交错重复取中位数，单位 ms/步。

| 平台与原子数 | bare | old | new | 整步下降 | new 相对 bare 附加开销 |
|---|---:|---:|---:|---:|---:|
| WSL / RTX 4080 SUPER，6912 | 19.9645 | 22.6224 | 21.6631 | 4.24% | 8.51% |
| WSL / RTX 4080 SUPER，55296 | 74.7109 | 85.8818 | 81.4635 | 5.14% | 9.04% |
| WSL / RTX 4080 SUPER，442368 | 441.530 | 538.506 | 483.728 | 10.17% | 9.56% |
| ARM Kunpeng 920 / A100 PCIe 40GB，55296 | 17.36916 | 24.1056 | 19.7998 | 17.86% | 13.99% |
| ARM Kunpeng 920 / A100 PCIe 40GB，3981312 | 1362.908 | 1782.184 | 1512.616 | 15.13% | 10.98% |

WSL 为 50 步预热、100 步计时；A100 为 20 步预热、50 步计时，均为 1 MPI rank、1 OMP 线程、1 GPU。A100 作业 `1602547` 在 `paraai-n32-h-01-agent-11` 正常完成（exit 0），未修改其他生产作业。55296 的 old 中位数为最终全部三次样本的 24.1056；中途只有两次样本时的数值不是最终中位数。

WSL 大体系 old 有一个 613.696 ms 的慢样本，另外两次为 538.506、535.405 ms，Pair 也发生波动。原始样本全部保留，不将整步差值全部归因于 LocalQL 核函数。A100 的 Pair 部分在 1357–1361 ms 附近，CV Modify 部分下降更明确。

历史“百万原子约 300 ms/步”来自另一套四张 RTX 4090 的纯 Al 测试，不能作为单张 A100 的 Al–Sc MEAM 基线；此处裸 MEAM 已需 1363 ms/步。历史约 +30% 的 CV 开销来自 6912 原子、4090D 和旧 LAMMPS。另一个 −33.7% 是整个版本的速度提升，不是 CV 附加开销。这些历史口径及原始出处见 DEVICE_NEIGHBORS_20261028.zh.md。

## 分段证据

442368 原子、50 步预热后捕获 20 步，以 CUDA event 记录实际流上的核耗时。该诊断程序只在捕获结束同步，性能比较使用未插桩的独立运行。

逐 pair 缓存删除后，qlm 前向由 41.679 降至 4.765 ms/步；进一步直接借用 CSR 后为 4.674 ms/步。环境核约 6.8–6.9 ms，j-k 反向约 14.8–15.0 ms，后者仍是后续可优化项。候选行复制原约 1.628 ms/次重建，新映射约 0.0143 ms/次重建。预热后已用显存由 5,161,484,288 降至 3,764,781,056 bytes，减少约 1.40 GB。

Nsight Systems 2024.5 明确报告不支持当前 CUDA driver 13.3，导出的报告没有 GPU kernel/copy 记录。因此没有用该报告声称测得 GPU 分段耗时；保留失败诊断，并用独立 CUDA-event 插桩补充。

## 数值与兼容验收

- LocalQL Graph 21/21，旧统计接口迁移后 18/18，ghost/cache 8/8；覆盖 1/2/4 MPI ranks、空组/空 rank、动态组、迁移、排除与周期镜像。
- Host 与无 Weave Kokkos 各 6/6；完整 CTest 37/37。
- WSL 新旧 9 对轨迹在步骤 0/50/150 的输出最大缩放差 4.18e-13；A100 6 对轨迹在 0/20/70 的最大缩放差 6.62e-15，门槛均为 1e-8。这是短轨迹一致性，不是长时间轨迹逐位相同或物理收敛证明。
- compute-sanitizer memcheck：6912 原子真实联合输入，ERROR SUMMARY 0 errors。此项单 rank 诊断设置 `OMPI_MCA_accelerator=null`，规避之前裸 MEAM 亦可复现的 WSL OpenMPI CUDA finalize 注册内存故障；所有性能样本没有使用该覆盖，不能称默认 MPI 环境已通过 memcheck。

## 可复查产物

Windows：`F:\Users\Administrator\Documents\ChatGPT\Work_MetaD\artifacts\metadtoy-localql-profile-20261028`。含 local/A100 分析 JSON、每次输入和原始日志、构建及环境元数据、插桩源码、采样脚本、source manifest 与 patch。A100 下发归档以 dd8da2c 为基线并包含本次两文件改动，内容对应 a576294；manifest 中基线字段不是候选实现版本号。

WSL：`/home/zqc/work/Metad-toy-unified-20261028/build/localql-profile-20261028`。最终三种插件、验证日志和 profile 数据按 manifest 冻结，后续工作不得覆盖此目录内的冻结插件。
