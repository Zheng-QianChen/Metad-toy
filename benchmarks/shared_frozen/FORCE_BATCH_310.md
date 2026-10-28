# Graph 力与 virial 合并回传（310，2026-10-27）

## 实现

CVGraphExecution 在各 CV 反向前开始输出批次，所有贡献提交后调用 finish。
LAMMPS 适配器用一份 Host AoS 缓冲累加 owned+ghost 力 [total,3] 和 virial [total,6]，
整批只调用一次 ghost reverse，随后一次性回写 owned 力与 virial。
Kokkos 最终 owned 力只上传一次。各 CV 的源数组在提交返回后即可复用。
批次外调用仍立即提交；GPUMD 默认批次钩子保留设备提交流程。

取消丢弃尚未写回的贡献，空批次不通信、不回写；批次内拒绝重新开始 evaluation 或改变拓扑。
DISTANCE 的 owned 贡献加入公共缓冲，只有 owned 贡献时不触发 ghost reverse。

## 通信次数与批次契约

每次独立 run 0 的实际计数均为：**2 次 CV 提交、1 次 reverse 流程、1 次输出回写**。
reverse 流程可包含多个邻域交换阶段，这里统计调用次数，不等于 MPI 消息数。

| 配置 | 提交 / reverse / 回写（每 rank） | Host/CUDA × AoS/SoA × 6/9 分量 |
|---|---|---|
| host-np1 | [[2, 1, 1]] | 8/8 通过，最大误差 0 |
| host-np2 | [[2, 1, 1], [2, 1, 1]] | 8/8 通过，最大误差 0 |
| kk-np1 | [[2, 1, 1]] | 8/8 通过，最大误差 0 |
| kk-np2 | [[2, 1, 1], [2, 1, 1]] | 8/8 通过，最大误差 0 |
| empty-host | [[2, 1, 1], [2, 1, 1]] | 8/8 通过，最大误差 0 |
| empty-kk | [[2, 1, 1], [2, 1, 1]] | 8/8 通过，最大误差 0 |

合计48组配置/布局用例。参考结果按全局原子 ID 独立求和，包含周期 ghost 和跨 rank ghost。
覆盖延迟写回、输入复用、取消后重开、空批次、错误元数据、非法生命周期与重复求值。
empty-host/empty-kk 中两个 CV 均只以 atom 1 为中心，两 rank 的 owned 中心数为 [1,0]。

## 数值与兼容回归

- 108 原子 Al3Sc LocalQ6×CSS：普通/Kokkos × 1/2 ranks，使用1e-5和3e-6两个差分步长，均通过。
- 最大力有限差分误差：1.682617004170224e-12 eV/Å；最大 virial 有限差分误差：2.823739560318256e-12 eV。
- 与改前插件比较初态、30步积分、重复求值和选择集刷新：最大总力差 1.119937476090627e-14 eV/Å，最大 virial 差 2.602085213965211e-18 eV，偏势能差为0。
- nonlinear_graph、distance_mixed、distance_graph 在普通/Kokkos × 1/2 ranks 下12/12通过。
- 独立 engine_contract、shared_frozen CTest 2/2通过。
- GPUMD engine、neighbors、runtime_abi CTest 3/3通过。

## 改前/改后交错计时

同一310单卡RTX4090 D、6912原子/1728Sc、MEAM、NPT 1450 K/1 bar、1 fs。
全局和共享偏势均冻结；每次从相同restart启动，预热50步；1 rank正式300步，2 ranks正式200步。
新旧插件随机交错三轮，共24次运行。结果为完整步时中位数，排除启动/NVRTC/预热。
两个rank共享同一张GPU，OMP_NUM_THREADS=1，gpu/aware off。

| 模式 | ranks | 改前 ms/步 | 改后 ms/步 | 完整步时变化 | Modify 改前→改后 ms/步 |
|---|---:|---:|---:|---:|---:|
| global | 1 | 27.4077 | 27.1284 | -1.02% | 4.5347 → 4.3703 |
| shared | 1 | 27.3344 | 27.1872 | -0.54% | 4.5097 → 4.3913 |
| global | 2 | 52.6725 | 53.1360 | +0.88% | 7.5190 → 7.2525 |
| shared | 2 | 52.7795 | 52.5935 | -0.35% | 7.7075 → 7.2720 |

各轮样本（ms/步）：

- kk_np1_global：改前 [27.889766666666667, 27.4077, 27.3197]；改后 [27.128433333333334, 27.368666666666662, 27.1284]。
- kk_np1_shared：改前 [27.324600000000004, 27.37316666666667, 27.334433333333333]；改后 [27.1872, 27.25376666666667, 27.151200000000003]。
- kk_np2_global：改前 [52.8545, 52.6725, 52.668499999999995]；改后 [53.136, 53.291000000000004, 52.935]。
- kk_np2_shared：改前 [52.75299999999999, 54.657, 52.7795]；改后 [52.593500000000006, 53.671499999999995, 52.4605]。

负的步时变化表示该组中位数变快。Modify 包含偏置处理和积分器；
总步时还受 MEAM Pair、邻居表、进程调度等影响。通信次数减少不代表完整步时按同一比例下降。
每次启动前后均未检测到其他GPU计算进程；节点未独占预约，小幅差异仍需结合重复波动解释。

## 当前边界与复现

公共累计缓冲仍位于 Host。LocalQL/CSS 的 GPU→Host 下载以及 LocalQL 内部计数回读仍存在；
本轮合并了跨 CV 的 ghost 通信和 owned 回写，GPU常驻累计及单rank设备ghost归并是后续工作。

- [批次探针与运行说明](../finite_difference/README.md#graph-力和-virial-合并回传)。
- [数值与计数原始结果](results/force_batch_numerical_310.json)。
- [性能汇总](results/force_batch_performance_310.json)、[全部样本](results/force_batch_samples_310.json)、[元信息](results/force_batch_metadata_310.json)。
- [普通全局图回归](results/force_batch_global_host_310.json)、[Kokkos全局图回归](results/force_batch_global_kk_310.json)。
- [CTest日志](results/force_batch_ctest_310.log)、[GPUMD日志](results/force_batch_gpumd_310.log)。
- 310隔离目录：/home/saitang/work/MetadToy-force-batch-20261027。
- 新插件：/home/saitang/work/MetadToy-force-batch-20261027/build/fix_crystallize_plugin.so。
- 宿主仍为旧隔离目录中的 lammps-build/lmp；现用安装及原基线二进制均保持原样。

改前插件 SHA256：697a00c24b2dbc08394696a769ac881b906af63ac0fbaf8fa18160cb1133d7c7。
改后插件 SHA256：c44535597489be7d56895dca9c233d23dd09bd07ee8317694567506d82a8f9cd。
