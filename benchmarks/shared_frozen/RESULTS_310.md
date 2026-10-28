# 冻结共享偏势验证（310，2026-10-27）

实现入口为 SHARED_FROZEN，第一版使用一张冻结的一维表和 MEAN 聚合。
Graph 保留原有 Global 标量输出，并增加 PerAtom 输出及数组反向种子。
本地查表、导数和求和使用 CUDA；MPI 只规约 sum_u、sum_s、count、invalid 四个值。

## 环境

- 310 节点直接执行，未使用 Slurm；RTX 4090 D 单卡。
- LAMMPS stable_29Aug2024 + Kokkos 4.3.1，CUDA 12.2.91，GCC devtoolset-9，MPICH 3.0.4。
- 独立宿主启用 MEAM、PLUGIN、KOKKOS，沿用已回补 trim 的隔离源码。
- 验证根目录：/home/saitang/work/MetadToy-shared-frozen-20261027。
- 用户正在运行的生产任务、原 LAMMPS 安装及之前的隔离二进制均未修改。

## 数值结果

- 108 原子扰动 L12 Al3Sc：普通模式和 Kokkos 各 1/2 rank，合计 4 个配置。
- 单中心 family：普通和 Kokkos 各 2 rank，合计 2 个配置；另一个 rank 的 family 为空。
- 上述 6 个配置均使用 1e-5、3e-6 两个步长，检查 8 颗原子的 24 个坐标分量及全部六个应变分量。
- 偏置力由“有偏总力减去同构型的无偏 MEAM 力”取得；差分只对冻结偏势能进行。
- 最大力差分误差：1.0738861189185656e-11 eV/Å。
- 最大 virial 差分误差：1.1287659338037742e-11 eV。
- 势能另与逐原子 compute 输出构造的解析二次势均值比较，独立检查 family 分母和组外 padding。

生产冻结初态使用原来的 6912 原子、1728 Sc 和同款 LocalQ6/CSS 径向参数：
两种模式各积分 30 步 NVE，步长 0.5 fs，保留 restart 的初始速度。
这是一段固定势下的计算验证，原生产任务使用的 NPT/WT 沉积不在此短跑中复现。

普通/Kokkos 全部记录状态的最大差异：

- 坐标（周期等价）：2.842170943040401e-14 Å。
- 总力：5.508926648190027e-13 eV/Å。
- LocalQ6/CSS：1.0402789740737717e-13。
- 总偏势能：0 eV。
- 偏势 virial：1.214306433183765e-17 eV。

两种模式分别从新写出的 restart 重新启动，重放偏势能均一致。
最大力差为 5.932476732084524e-13 eV/Å；最大 virial 差为 7.112366251504909e-17 eV。
写 restart 会把部分原子坐标移动整数盒矢量；检查先按周期等价处理坐标，
其余力、CV 和能量仍使用原有严格容差。选择集成员刷新和重复 run 0 均通过。

## 回归和内存检查

- 新共享表回归及原有 Graph/NVRTC、装配、开关函数、流契约：5/5 通过。
- 原有 nonlinear_graph、distance_mixed 的 1/2 rank 有限差分：4/4 通过。
- 共享表单元回归 Compute Sanitizer：0 错误。
- 完整 Al3Sc MEAM/kk + Graph + 共享表，3 步积分及选择集刷新 memcheck：0 错误。
- 表插值的解析导数检查容差为 2e-12；有限差分容差为 1e-9，实际误差 3.4662e-12。
  后者考虑了能量相减再除以步长的舍入放大。

## 参数与范围

验证表为 u(s)=0.4*(s-0.35)^2 eV，s=q*c*fermi(q*c,0.35,15)，
q 为原始 LocalQ6，c 为 CSS。它是便于独立核对的解析测试势。
共享历史的自适应沉积、PBMetaD-PF、多个 family 和正式 FES 统计属于后续工作。

MEAM/kk 需要 Newton pair on；本轮使用
-k on g 1 -sf kk -pk kokkos gpu/aware off neigh half newton on。
本轮没有多 GPU、多节点或吞吐性能结论。

插件 SHA256：697a00c24b2dbc08394696a769ac881b906af63ac0fbaf8fa18160cb1133d7c7。
表 SHA256：da480bce08df316ee7605aec28f24d8c0b2bc5be8b47bbba87b40500a36f1f41。

汇总见 results/comparison_310.json；4 个原有差分结果见 results/global_regression_310.json；
两个 memcheck 日志和最终 Graph CTest 日志也保存在 results/。
完整输入、逐原子状态、frames.csv、restart 和编译日志留在上述 310 隔离目录。
