# 统一源码验收（2026-10-28）

本页保留早期整合的历史记录。`final-weave-ctest/ctest.log` 的 34/34 对应
2026-10-28 17:24 的测试清单，早于 21:28 的旧功能恢复提交，不包含后来新增的
`zqc_graph_sum_safe_div_mpi_{1,2,4}`，不能作为这三条的验收证据。
2026-10-29 的失败复现、上传就绪契约修正和当前重复验证另见
[SUM/safe_div 跨流复验](STREAM_READINESS_20261029.zh.md)。

本轮三个构建配置均来自同一份源码，源码文件指纹为 `875f3cc6093d06c086b0dd30e143d5cf301bb27f7c78537e5dbdf674ddd9f168`。
逐文件 SHA256、插件 SHA256、结果路径见 `integration-validation-20261028.json`。
Git 提交号另随主目录 `build/unified-20261028/manifest.json` 记录，避免把旧 HEAD 的历史测试归给新源码。

| 验收 | 结果 | 覆盖范围 |
| --- | --- | --- |
| 构建 | 3/3 | Weave+Kokkos、普通 Kokkos、普通 LAMMPS；可选 ML 关闭。 |
| 早期 CTest（10-28 17:24） | 34/34 | 当时的 Host/Graph/CUDA/流、MPI、CV、冻结表、GPUMD 测试清单；不包含后加的 SUM/safe_div 三条。 |
| 全局 CV 有限差分 | 14/14 | Q6、LocalQL6、传统 DISTANCE/线性组合、Graph DISTANCE、混合域、非线性图；1/2 rank，两个步长。 |
| Al3Sc 固定表 | 18/18 运行 | Weave 下 1/2 rank，原始乘积/开关乘积、空中心、选择集刷新、NVE、读写 restart；三条回退配置各 4 个运行。 |
| 真实适配器批次 | 4/4 运行 | 1/2 rank × 普通/单中心；每 rank 12 种 Host/CUDA/混合、AoS/SoA、6/9 分量组合；取消、空批次、复用、一次 reverse/写回。 |
| WT + NPT + restart | 2/2 | 全局 Q6/LocalQL6/CSS，1/2 rank；6 次 hill，run 0 不沉积，续算力/能量/CV/体积/virial 与连续运行比较。 |
| 后端数值对照 | 9/9 | Weave 与关闭设备提交、普通 Kokkos、普通 LAMMPS 比较；最大绝对差 `2.898e-14`，容差 `atol=rtol=2e-10`。 |
| 离线 CLI | 52/52 | 包含生产 C++ bias 参考探针，无跳过项。 |

Weave 最终插件 SHA256：`304ad0ee9c51d0f264a5941d68b21f50d0466bf27f62642a8b05c9bb4f8aa5dc`。
普通 Kokkos：`f73ff146d6b358568eb8692360bf88a48d515f5bad2e83b643bb34b428802cbf`。
普通 LAMMPS：`f41b35d2e8b35fc23148bd92c4bc239e26ab7469c305eb4f552becf35689eca8`。

完整本地日志：`/home/zqc/work/Metad-toy-unified-20261028/build/validation-20261028`。构建目录还保存 configure/compile 日志、完整命令和 build receipt。
推荐读取主目录 `build/unified-20261028/manifest.json` 选择对应插件；不要继续使用旧默认 build 中的
文件名来推断是否包含新实现。GPU 联合插件仍依赖该 manifest 所列匹配的 LAMMPS/Weave 运行库。

第一次新增 NPT 测试发现测试脚本在恢复 NPT fix 前执行了 run 0，LAMMPS 日志报告 unused restart
fix info，导致恒温/恒压器状态丢失。已修正测试顺序，保留原失败日志；最终两种 rank 数均通过。
这项失败没有通过放宽轨迹容差绕过。

本轮仍是实验版整合验收：未重跑 LAMMPS 2024、完整 GPUMD 长轨迹、ML CV、远端 ARM/A100 新包
或大体系性能；不能由小体系正确性推断形核采样/FES 已收敛。在线共享沉积、生产 SUM、旧独有 CV
和性能工作见 `DEFERRED_FEATURES_20261028.zh.md`。
