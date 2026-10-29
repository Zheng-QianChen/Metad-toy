# CSS / LocalQL 设备邻居准备：实现与验证（2026-10-28）

实现提交：8b02540，分支 codex/metadtoy-unified-20261028。
隔离源码：/home/zqc/work/Metad-toy-unified-20261028。本轮没有替换主仓库的已安装插件，也没有更换正在运行的生产任务。
预编译新版本已另存到 /home/zqc/work/Metad-toy/build/device-neighbors-20261028/，
包括 weave、kokkos、host 三种插件及 manifest.json。主工作区存在其他 agent 的未提交改动；
本轮源码保存在上述隔离分支，没有覆盖这些改动。

## 实现

- NeighborRequest::host_csr=false 允许消费者放弃 Host offsets/neighbors 镜像；默认保留兼容语义，请求去重包含该字段。
- CSS/StruFactor、LocalQL 在 Kokkos 下请求 device Full / Full+ghost 表。桥接借用 d_numneigh、d_neighbors 的真实 stride，并等待 Kokkos 生产流事件；GPU 计数、扫描、筛选与 CSR 填充替代 Host 邻居遍历。
- 实际 AlSc 日志显示 CSS 为 copy/kk/device、copy from MEAM full list；LocalQL 为 full/bin/ghost/kk/device。是否复用由 LAMMPS 的请求匹配决定，不强制把不兼容的 cutoff 或排除语义混在一起。
- Kokkos ghost 行使用自定义 R+skin；保留 2R+skin halo，无须普通 Host 路径的 cutneighghostsq 补行。周期镜像保持不同 localid，不按 tag 合并。
- LocalQL 的 CSR 行重排、中心反向索引放在 GPU。中心组上传按成员/布局 generation 缓存；有效几何、开关权重和 I/J/K 标记仍每轮更新。每轮仅回读 I/J 中心数与有效 pair 数。
- CSS 的中心顺序检查按邻居 generation 和 Leaf 装配对象缓存。普通 Host、全局 Q、Half 的原有兼容路径保留。

## 实测

单位 ms/步，下表只列各模式均已收齐三次交错运行的中位数。WSL 为 50 步预热、100 步计时；A100 为 20 步预热、50 步计时。
每个平台内部使用同一个 LAMMPS/Weave、数据、势函数、NPT 参数和固定共享表。测量关闭 dump/restart，仅保留计时端点 thermo。
WSL 保留 sort 5；A100 完全沿用原生产输入的默认 sorting。因此不能直接把两个平台的毫秒数归结为 GPU 型号差异。
本轮运行宿主均为 LAMMPS 22Jul2025；29Aug2024 仅补充核查了 Kokkos ghost cutoff 与设备 View 字段，未把源码一致性当作该版本运行验证。
每次计时的范围、阶段数据和邻居重建次数均保存在 summary.json / measurements.csv；旧版 Host 阶段存在样本波动，表中使用中位数。

| 平台 | 原子数 | 裸 MEAM | 旧联合 CV | 新联合 CV | 总耗时减少 | 新 CV 相对裸 MEAM 额外开销 | 仅 CSS | 仅 LocalQL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| N32H A100 40GB | 55,296 | 17.347 | 39.835 | 23.852 | 40.12% | 37.50% | 18.132 | 23.677 |
| N32H A100 40GB | 3,981,312 | 1365.892 | 2628.280 | 1727.270 | 34.28% | 26.46% | 1400.504 | 1715.274 |
| WSL RTX4080SUPER | 6,912 | 19.261 | 23.273 | 21.896 | 5.91% | 13.68% | 20.221 | 21.635 |
| WSL RTX4080SUPER | 55,296 | 68.002 | 85.447 | 80.201 | 6.14% | 17.94% | 70.151 | 79.792 |
| WSL RTX4080SUPER | 442,368 | 411.212 | 536.961 | 484.678 | 9.74% | 17.87% | 422.083 | 482.837 |

旧联合 CV：WSL 用本次优化前的不可变统一插件；A100 用原生产 joint 插件。新旧均为 LocalQ6×CSS 的 raw product、SHARED_FROZEN MEAN、PACE 0。
仅 CSS 与仅 LocalQL 使用相同固定表，分别以 c 或 q 为输入。这两种有偏轨迹不同，耗时差不能当作可精确相加的 kernel profiler 结果。

A100 3,981,312 原子阶段计时：

| 阶段 | 裸 MEAM | 旧联合 CV | 新联合 CV |
|---|---:|---:|---:|
| Pair | 1358.520 | 1363.160 | 1359.380 |
| Neigh | 0.673 | 694.460 | 2.027 |
| Modify | 2.263 | 568.320 | 364.460 |
| Comm | 1.160 | 0.889 | 0.781 |

Modify 包括 CV 几何准备、前向、反向、同步、偏势和积分 fix；它不是 LocalQL kernel 的独立计时。

## 正确性与兼容性

- LocalQL 14 组（7 场景×1/2 rank）通过：独立 CPU 参考、坐标/virial 有限差分、径向开关、组刷新、owner 迁移、排除、孤立原子和空 rank。
- ghost 缓存 8 组（4 场景×1/2 rank）通过：周期两跳边跨 cutoff、延迟首次需求、动态组、NPT、倾斜盒、迁移、skin 改变。探针对照当前坐标逐边枚举，并断言设备路径 ghost_row_rebuilds=0、neighbor_uploads=0。
- Al3Sc 联合测试 6 组通过：raw/gated 有限差分、空中心组、1/2 rank、实际 restart 重放。
- CTest 34/34 通过。
- 新旧轨迹端点（温度、势能、压力、体积、偏势、CV）比较：15/15 通过相对尺度容差 1e-8；完整误差保存在 thermo-checks.json。
- build-host: PASS
- validate-host: PASS
- build-kokkos: PASS
- validate-kokkos: PASS
- global-history: PASS
- memcheck: FAIL

默认 WSL OpenMPI CUDA accelerator 环境的 memcheck 返回 99，唯一报告来自退出阶段
cuMemHostUnregister 的 CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED；裸 MEAM 对照同样复现。
单 rank 设置 OMPI_MCA_accelerator=null 后，新插件完整 memcheck 为 0 错误。
未禁用 memcheck 的 API 报告；该诊断环境不用于计时或生产，也不代表跨 rank
设备 MPI 在 memcheck 下没有 API 问题。详见 memcheck-controls.json 和原始日志。

## 与旧性能说法的关系

此前的“几百万原子约 300 ms/步”来自曙光 4×RTX4090 的纯 Al MEAM 测试：8,388,608 原子中位数约 240 ms/步，没有 CV。
当前为单 A100 40GB、Al–Sc 二元 MEAM、NPT、单 MPI rank/单 OpenMP 线程。当前裸 MEAM 本身约 1.36 秒/步，不能拿前者的绝对时间作为同配置目标。
此前约 +30% 是 310 单 RTX4090D 上的 LocalQL+CSS 联合开销，另外有一次 -33.7% 表示版本优化后的总耗时下降；两者不是同一个比率。
本轮以同一 A100 分配中的裸 MEAM、旧版、新版直接对照，区分平台基线和软件邻居准备开销。

## 尚未完成的优化

Host 选择集/类型/ID 元数据、按需诊断下载和两个 LocalQL 标量计数仍存在。设备 CSR 还要转换为 LocalQL 的 dense 行，球谐缓存和有效 pair 数组也仍占显存。
全局 Q/Half 的邻居准备、全局 Q 的原子属性 forward 通信，以及未启用 Weave 的力通信仍有 Host 路径。本轮不宣称“所有 CV 全程无 Host”。
需要下一轮优化时，应先 profile LocalQL 前向/反向与大球谐缓存；不应继续把剩余 Modify 时间归咎于已经移除的 CPU ghost 补行。

## 证据与复现

本地日志：build/neighbor-device-20261028/ 与 build/validation-20261028/neighbor-device-*。
审计产物：F:/Users/Administrator/Documents/ChatGPT/Work_MetaD/artifacts/metadtoy-neighbor-device-20261028/；保留脚本、逐次计时、输入、stdout、误差、源码快照和 SHA-256。
本地 Weave 插件 SHA-256：2d54da3cd726a152386886bf87d83c884ccd7fa1f68e6c489885b6f341beb999。
A100 的二进制/资产 SHA、节点和 GPU UUID 见 a100-metadata.json；作业状态：

1602110|COMPLETED|0:0|paraai-n32-h-01-agent-60|00:37:44
1602110.batch|COMPLETED|0:0|paraai-n32-h-01-agent-60|00:37:44
1602110.extern|COMPLETED|0:0|paraai-n32-h-01-agent-60|00:37:44

最初两次 A100 作业因传输构建包漏掉探针源码及 third_party 头文件失败；补齐后使用同一隔离目录重新构建。失败日志保留，不计入性能数据。
