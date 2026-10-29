# 统一版的科学证据与运行环境

本页分别记录可重复的工程/控制模型证据，以及真实 Al3Sc 科学验收仍需回答的问题。
不能用短积分、完整编译或一个控制模型的通过替代真实形核和 FES 收敛。

## 其他开发工作区的科学实现

从仍在开发的 `/home/zqc/work/Metad-toy` 只读保存 34 个新增 benchmark/结果文件和
已跟踪改动补丁，再合入本统一分支。原目录与其未提交内容均未改写。
合入内容包括 BAOAB 动力学、删失首次到达、两层配对 bootstrap、同表冻结/继续、
独立固定时长 FES 采样和实际 CUDA 插值器核对。生产 CV/偏势内核没有因此变化。

原 FES 记录中的 `kinetics_probe.cpp`、`fes_recovery_sample.py`、`fes_recovery.py`、
`fes_recovery_cuda.py` 四个 SHA256 均与取回的源码一致；完整快照和逐文件清单位于
Windows `Work_MetaD/artifacts/metadtoy-release-20261028`。

## 统一版上的重新采样

使用本分支构建的 `zqc_shared_kinetics_probe`，重新生成 16 条带偏独立轨迹和
16 条无偏对照；每条重新平衡 500、采样 10000 约化时间，dt=0.0005，每 0.1 记录。
沉积/冻结协议、种子和预先规定的误差门槛与原报告相同，没有根据本轮结果调整门槛。

| 指标 | 原始带偏 | 完整总偏势重加权 | 无偏对照 |
|---|---:|---:|---:|
| 二维 FES RMSE / kBT | 0.68917 | 0.01454 | 0.03478 |
| 参考概率质量覆盖 | 99.9787% | 99.9773% | 99.9295% |
| 联合分布总变差 | 0.26821 | 0.00486 | 0.01559 |
| P(x>0) | 0.66929 | 0.49754 | 0.48614 |
| P(x>0) 独立轨迹 SEM | 0.00496 | 0.00597 | 0.01715 |
| 核心跃迁数 | 3334 | 同一轨迹 | 844 |
| 全部门槛 | 不通过 | 通过 | 精度不通过 |

160 万帧的现有 CLI 重建与独立实现概率最大差 6.87e-15。
实际生产 CUDA 冻结表在 512 对输入上的能量差 3.26e-14、导数差 1.52e-13、
有限差分差 6.45e-10、置换误差 0。原始数据保留在
`build/release-fes-recovery-20261028`，包括每条轨迹、CSV、协议、表、历史和图。

本轮与原 310 数值略有差别，统计判据与定性结论一致；不要求长混沌轨迹跨构建逐位一致。
重加权使用整帧总偏势 B，weight ESS 只量化权重集中度，不等于时间去相关的独立样本数。

## 科学适用范围

- 当前控制模型采用恒高原型沉积、MEAN 冻结势；不等同生产在线 WT 的默认分摊参数。
- 同表配对记录支持本次物理坐标的双向收益，但压缩 CV 的反向可变慢。
  不把一次分布压平视为通用动力学加速。
- 冻结后平衡重加权可恢复该控制模型联合分布；不能将共享一体表的负值直接当作
  相关多中心的联合 FES，也不能用构型重加权系数恢复真实物理时间。
- Al3Sc 仍需指定科学状态点、独立训练/验证轨迹、核尺寸/结构判据、正反向覆盖、
  分块/独立重复不确定度以及时间依赖偏势的有效归一化方案。
  10 万步稳定性验收不证明这些问题已解决。

详见 [原 FES 实验](../benchmarks/shared_frozen/FES_RECOVERY.md)、
[动力学](../benchmarks/shared_frozen/KINETICS.md)和[同表配对](../benchmarks/shared_frozen/MATCHED_MD.md)。

## WSL OpenMPI 5.0.6 退出阶段问题

原单 rank memcheck 的 `cuMemHostUnregister` 错误 713 已由裸 MEAM 复现。
本轮进一步使用只链接 MPI/CUDA 的最小程序（没有 LAMMPS 或 MetaDtoy）：

| 初始化/通信顺序 | sanitizer 退出码 | 错误数 |
|---|---:|---:|
| MPI_Init → CUDA 初始化 → MPI_Finalize | 99 | 1，错误 713 |
| CUDA 初始化 → MPI_Init → MPI_Finalize | 0 | 0 |
| MPI_Init → CUDA 初始化 → 设备缓冲自通信 → MPI_Finalize | 0 | 0 |
| 单 rank 禁用 MPI accelerator 的诊断对照 | 0 | 0 |

verbose 日志显示第一种顺序没有注册测试内存，退出仍尝试注销；其他两种 CUDA 路径注册成功。
这与 [OpenMPI 5.0.6 组件源码](https://github.com/open-mpi/ompi/blob/v5.0.6/opal/mca/accelerator/cuda/accelerator_cuda_component.c)
的延迟初始化和 finalize 行为相符。没有通过屏蔽 API 错误来判定通过，也没有替换已安装 MPI。

最小程序为 `benchmarks/integration/mpi_cuda_init.cpp`，以匹配宿主的 mpicxx、CUDA 头文件和
libcudart 编译。依次传入 late-cuda、early-cuda、device-exchange 参数，并在外层运行
`compute-sanitizer --tool memcheck --error-exitcode 99`；默认测试只运行一个进程。

新增显式诊断入口 `benchmarks/integration/early_cuda.py --device 0 <test.py> ...`，
在 Python 测试导入 MPI 前初始化所选设备。用最终冻结插件跑完整共享在线 NPT/重启测试，
memcheck 与 initcheck 均退出 0、错误数 0，**没有设置 `OMPI_MCA_accelerator=null`**，
CUDA-aware MPI 和设备属性/力路径保持启用。

该入口解决已验证的 Python 诊断启动顺序；没有宣称所有第三方 MPI/驱动版本均已修复。
每个 rank 的 device 编号在其 CUDA_VISIBLE_DEVICES 命名空间内显式选择，不能在多卡环境
盲目让所有 ranks 使用 0。生产插件没有额外插入自通信或初始化 MPI。

构建仍可能报告 DeepMD 与 CUDA Toolkit 两处 libcudart 搜索目录，以及历史未使用变量警告。
验收固定实际依赖路径和哈希；这些警告未通过关掉诊断来隐藏。
