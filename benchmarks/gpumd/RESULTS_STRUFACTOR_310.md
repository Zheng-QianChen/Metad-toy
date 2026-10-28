# StruFactor → GPUMD 验证（310，2026-10-23）

本轮在 libmetadtoy 中加入独立周期像候选表及 StruFactor 输入。GPUMD 可执行文件
与 C ABI v1 均未修改；6 项回归通过，随后补测底势 cutoff=2.0 Å、
CV cutoff=4.3 Å 也通过。编译与运行均通过 SSH 直接在 310 执行，未使用 Slurm。

## 数值结果

- 小盒 2.8×3.2×3.7 Å、CV cutoff=4.3 Å；独立枚举周期像，包括原子自身的非零平移像。
- 验证 CV 值、总势能、逐原子力及完整九分量 virial。
- 力中心差分最大误差约 7.27e-9 eV/Å。
- 对角应变 virial 差分最大误差约 2.79e-8 eV，包含自像对盒应变的贡献。
- 不同中心组、不同 cutoff 的两个 StruFactor 均值与 DISTANCE 组成 a*b+r，通过解析对照。
- 逐原子非线性图 (s*s).AVE 经 NVRTC 前后向执行，通过解析值与有限差分。
- 非周期、部分周期和完全周期条件均通过。
- 200 个积分步（含初始力，共 201 次求值）：skin=0.3 建表 1 次；skin=0 建表 201 次；
  COLVAR 最大差为 0。

近邻契约测试还覆盖 Full/Half、非对称选择集的正反方向、跨周期边界的小位移缓存、
超出 skin/2 和盒长变化时重建，以及 Host/CUDA 两种输入的 owner 力/Full9 归并。

CUDA compute-sanitizer memcheck：独立近邻测试与真实 GPUMD+NVRTC 路径均为 0 errors。
数值明细和可执行文件哈希见 [RESULTS_STRUFACTOR_310.json](RESULTS_STRUFACTOR_310.json)。

## 使用范围与成本

当前宿主入口仍为单进程、固定原子顺序、正交固定盒；支持 owned 中心的邻居请求。
LocalQL 所需的 ghost 中心完整环境与 GPUMD 原生 group 编号映射尚未接入。

候选 CSR 在 CPU 使用空间分箱重建；每轮 GPU 位移检查回传一个标志，再在 GPU
刷新周期像坐标和类型。较大的 cutoff/盒长会增加像数量。本轮未对大体系做性能测量。

配置与复现命令见 [README.md](README.md#strufactor中心组与周期像)。
