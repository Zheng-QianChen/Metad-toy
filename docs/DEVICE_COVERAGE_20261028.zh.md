# Q / Half / 无 Weave 设备路径验收

对应用户指定完成顺序第 4 项。本项在独立 q-device 工作树实现，基于 LocalQL 优化 a576294。下面的“设备”指明确的数据与算子路径，不等于整个 MD 程序没有 Host 工作。

## 改动

Q 现请求设备 Full CSR，在 GPU 上映射中心行、筛选和压缩 half-pair 索引。拓扑变化时仍上传组/ID 元数据并读取少量校验/计数标量，但不再把邻居边数组下载到 Host 遍历。Q 的球谐、q、权重和与 adjoint 合并为设备属性行，交给原生 CommKokkos forward，保留 GPU-aware MPI 的选择与 halo relay 顺序。普通 Host 引擎继续使用兼容通信。

通用 Half 请求支持设备 CSR：相同选择集借用原生 Newton-on half 表，不同选择集在设备 Full 表上执行原有的双向成员判定、全局 ID 及周期自镜像方向规则。半对归属不依赖 pair potential 的 Newton 开关。测试独立以 Full 有向表计算 pair 数、能量、两端力和 virial 作为参考。

没有 Weave 时，力和 virial 也在设备合批、归约和写回。LAMMPS 22Jul2025 的原生 Fix reverse 仍是 Host 接口，故只将 halo 包暂存到 Host，由原生 communicator 交换，再在设备累加。没有下载整份 owned+ghost 力/virial 场。Weave 版本继续使用设备 reverse。`WEAVE_METAD_DEVICE_COMMIT=0` 保留原整场 Host 回退，现也适用于无 Weave Kokkos。

## 当前路径边界

| 路径 | 正常 CUDA/Kokkos 路径 | 仍可能使用 Host 的位置 |
|---|---|---|
| CSS / StruFactor / 化学扩展 | 设备候选、坐标、CV 与反向 | 组、ID、类型元数据和按需诊断 |
| LocalQL | 设备两层邻域、球谐和反向 | pure-J 标量；组与布局元数据 |
| Q | 设备候选、half-pair、球谐和属性 forward | 拓扑标量；Host 引擎或用户通信回退 |
| LocalGeom COORD/TET | 设备邻居、值和反向 | 诊断输出、组与类型元数据 |
| 通用 Half | 请求 `host_csr=false` 后设备筛选 | 显式要求 Host CSR 的兼容消费者 |
| 力 reverse：Weave + GPU-aware MPI | 设备包交换与设备写回 | 用户关闭 GPU-aware 或显式关闭设备提交 |
| 力 reverse：原生 Kokkos | 设备累加/写回，Host halo 包 | 当前 LAMMPS Fix reverse 接口 |
| DISTANCE、全局 bias、Graph 标量规约 | 小规模标量/选定原子路径 | 这些路径本来就含 Host 标量及 MPI 规约 |

本次未启用可选 ML/ONNX 模块，不将其列为已验收。`compute metad/atom`、stress/atom、dump 等消费者要求 Host 数组时，相应下载是按需输出，不应与常规 CV 计算混为一谈。

## 验证

本地 RTX 4080 SUPER、LAMMPS 22Jul2025、CUDA 12.6、OpenMPI 5.0.6：

- Q：Weave 21/21、无 Weave Kokkos 21/21、Host 6/6；覆盖 Q3/Q4/Q6、多个中心组、软径向权重、空 rank、迁移和排除。
- 混合 Q/LocalQL 图 12/12；LocalQL 无 Weave 9/9；GPU-aware 关闭和 comm host 各 6/6；显式整场力回退 4/4。
- Half：Weave 和原生 Kokkos 各 486 个选择/构型检查，最大误差 4.44e-12，覆盖三斜盒、周期自镜像、Newton on/off、组变更与排序。
- 12 组真实力批次测试（两种构建 × 1/2/4 ranks × 普通/空中心），每 rank 含 12 种 AoS/SoA、Host/CUDA/混合输入、六/九分量与生命周期检查；参考误差为零。
- Q 坐标有限差分最大误差 2.57e-12、应变有限差分 6.26e-9；门槛分别 2e-7、2e-6。CTest 37/37。
- Q memcheck 与 initcheck 均 `ERROR SUMMARY: 0 errors`。同先前诊断，仅这些单 rank sanitizer 用 `OMPI_MCA_accelerator=null` 避开已复现的 WSL OpenMPI finalize 故障，不代表默认 MPI 环境已解决。

通信计数实测：2 ranks、44 次 Q 图求值，每 rank 132 次属性 forward、792 次原生设备打包、0 次 classic 打包。Weave 的力 ghost D2H/H2D 为零。无 Weave 每 rank 约 1.02 MB halo D2H/H2D，且 owned 力保持设备。此宿主在 `gpu/aware off` 时选用 classic Fix forward，实测计数也是 classic；因此没有把该样本声称为原生设备 MPI。

Python 测试夹具原来只 close LAMMPS，退出时会打印 Kokkos 未 finalize。现显式 finalize，另行复跑 1/2 ranks 的正常 Q 路径后该诊断消失。故意触发旧接口拒绝的失败进程仍可能有 finalize 提示；这些不是正常成功路径。初次 Half 夹具漏装 weave/batch_comm 被明确拒绝，补上测试夹具入口后完成全部矩阵；失败日志保留。

## 产物

源工作树 `/home/zqc/work/Metad-toy-q-device-20261028`；原始记录在其 `build/device-coverage-20261028`，另外保留 `build/q-no-weave`、`build/q-host`、`build/half-device-weave`。冻结插件和源码/证据归档见 Windows `F:\Users\Administrator\Documents\ChatGPT\Work_MetaD\artifacts\metadtoy-device-coverage-20261028`。

本轮证明功能、数值与数据通路，没有对 Q 或无 Weave 版本给出未经测量的加速百分比。多物理 GPU、跨节点、长时间运行仍属于下一阶段生产验收。
