# Al3Sc 冻结共享原子偏势原型

当前正式实现已增加 `SHARED_AGGREGATION SUM`、`SHARED_ONLINE`、共享 WT 历史和续算；下面保留最初 MEAN 冻结原型的定义与记录。新接口与数学定义见 [共享偏势手册](../../docs/manuals/SHARED_BIAS.zh.md)。`online.py` 检查真实 MEAM/NPT 的共享在线历史与续算，`al3sc.py --aggregation SUM` 检查生产 SUM 的力/virial。

本原型使用 Sc 中心上的原始 LocalQ6、CSS 及其组合：

```text
s_i = q_i * c_i * fermi(q_i*c_i, 0.35, 15)
V = sum_{i in Sc, owned globally} u(s_i) / N_Sc
dV/ds_i = u'(s_i) / N_Sc
```

分母是跨 MPI 的有效中心总数。ghost 用于 CV 导数及力通信；公共 Leaf 中组外的补零槽位
由明确的 family 选择集排除。`f_bias` 返回 V，`f_bias[1]` 返回同一 family 上的 MEAN(s)。
Al 原子作为环境邻居仍会收到反作用力。

## LAMMPS 输入

CUDA/Kokkos 的构建方式见 ../kokkos/README.md。

```lammps
group Sc type 2
fix bias all metad EXECUTOR GRAPH SHARED_FROZEN shared.table Sc PACE 0 &
  CAL NAME SW_r6 SW_FUNC FERMI r_0 3.50 alpha -20 &
  CAL NAME SW_css SW_FUNC FERMI r_0 3.30 alpha -20 &
  CAL NAME LQ6 STEINH L 6 Sc cutoff_eps_r 1e-12 SW_FUNC_r SW_r6 &
  CAL NAME JOINT CSS Sc CHANNEL 2 SW_css 0 0.45 cutoff_eps 1e-12 &
  SYMBOL q LQ6.RAW SYMBOL c JOINT.RAW &
  CV_dim 1 DIM_EXPR 1 "q*c*fermi(q*c,0.35,15)"
```

SHARED_FROZEN 后两个参数分别为表文件和中心组。第一版接受一个一维逐原子表达式，
采用 MEAN 聚合。省略 PACE 时自动冻结；显式非零 PACE、WT 或 METAD_RESTART 报错。
表在构造时读入并保持不变，日志给出内容指纹。每次重启使用同一输入与表。
所有 rank 必须读取同一张表。

表格式：

```text
# METAD_TOY_SHARED_FROZEN 1
# coordinate energy derivative
0.0 0.049 -0.28
0.5 0.009 0.12
1.0 0.169 0.52
```

坐标严格递增且等间距；三列分别是 CV 坐标、势能、对 CV 的导数。
上面三点来自 u(s)=0.4*(s-0.35)^2。能量采用 LAMMPS 当前单位，units metal 为 eV。
区间内采用三次 Hermite 插值，能量与导数来自同一多项式；两端按端点斜率线性延拓。
请将表区间覆盖目标采样区间，并确认端点斜率符合预期的外推力。

## 实现边界

- Graph 的 Global 输出继续通过 forward() 返回；PerAtom 输出通过 per_atom_output(index) 借用。
- backward(global_seeds, atom_seeds) 中两组种子分别按对应输出的先后顺序排列。
- 共享表批量查值与本地求和在 GPU 完成，只传四个汇总量给 Host/MPI。
- FrozenSharedBias 仅依赖 ArrayView、CUDA 和 SumReducer，可供其他宿主接入。
- 当前全局 AVE 节点的已有 Host 规约路径保留；这次 GPU 规约用于共享表路径。
- 每轮梯度视图带完成事件；完成消费后才能覆写或销毁相应存储。
- family 应按元素或固定身份定义。空间硬分组改变成员会令势能函数分段跳变，
  不能把这类成员变化下的轨迹直接视作同一个光滑固定势的平衡采样。

## 验证

al3sc.py 使用用户现有 Al-Sc MEAM 文件。小夹具是扰动的 108 原子 L12 Al3Sc，
沿用生产的 LocalQ6/CSS 径向参数。测试表为解析二次势，用于独立核对，
不代表从采样学习得到的成核自由能。

```bash
python benchmarks/shared_frozen/al3sc.py \
  --plugin /path/to/fix_crystallize_plugin.so \
  --assets /path/to/Al-Sc-potential-directory \
  --output /path/to/new-test-directory
```

assets 目录包含 Al-Sc.library.meam 和 Al-Sc.meam。MPI 与 Kokkos 示例：

```bash
mpiexec -np 2 python benchmarks/shared_frozen/al3sc.py \
  --plugin /path/to/fix_crystallize_plugin.so \
  --assets /path/to/assets --output /path/to/new-kk-test \
  --lammps-args="-k on g 1 -sf kk -pk kokkos gpu/aware off neigh half newton on"
```

增加 --single 仅让 atom 1 贡献，检查本地空 family 与 MPI 归一化。
使用 --restart /path/to/restart.start --skip-fd 检查生产初态。
程序写出 result.json、frames.csv、shared.table 和 restart.final。
frames.csv 保存每帧实际总偏势，可在固定势下平衡后使用 exp(V/kBT) 重加权；
短验证轨迹只用于实现检查，FES 的采样覆盖和统计收敛需要单独评估。

独立 GPU 回归：`ctest -R zqc_shared_frozen --output-on-failure`。
覆盖混合 Global/PerAtom 输出、多输出种子、流依赖、长度刷新、组外梯度、
非连续 View、表导数、端点延拓、空 rank、非法槽位和固定表重载。

完整结果见 [310 验证记录](RESULTS_310.md)。本目录的 [shared.table](shared.table)
就是验证使用的解析测试表，可用绝对路径传给 SHARED_FROZEN。

310 上已编译的验证插件：
/home/saitang/work/MetadToy-shared-frozen-20261027/build/fix_crystallize_plugin.so。
它匹配同目录上层 lammps-build 中的隔离宿主；在其他安装上使用时需匹配编译器和宿主 ABI。


## 稳态运行效率

performance.py 以同一份 6912 原子 Al3Sc restart 对比三条完整积分路径：
仅 MEAM、原有 Global Graph＋冻结全局网格偏势、PerAtom Graph＋冻结共享表。
三者都采用 NPT 1450 K／1 bar、1 fs 步长，因此包含每步所需的 virial 计算。
全局偏势作用于 AVE(s)，共享偏势计算 MEAN(u(s))；计时不代表两种采样方法的统计效率相同。

```bash
python benchmarks/shared_frozen/performance.py \
  --lammps /path/to/lmp --mpiexec /path/to/mpiexec \
  --plugin /path/to/fix_crystallize_plugin.so \
  --assets /path/to/assets --table /path/to/shared.table \
  --output /path/to/new-performance-directory
```

assets 需要 restart.start、Al-Sc.library.meam 和 Al-Sc.meam。
脚本默认检查普通／Kokkos、1／2 MPI rank。每次从相同初态启动，预热 50 步，
排除 NVRTC、读表和预热，再随机交错执行三轮计时。根据预跑速度选择同组一致的步数，
默认每轮至少 200 步、最多 3000 步，目标时间为最快模式约 6 秒。
MPI、CUDA 与 LAMMPS 动态库应提前加入环境；OMP_NUM_THREADS 设为 1。
脚本在每次启动前检查 GPU 计算进程，有其他计算任务时终止计时。

输出保留完整输入、日志、原始样本、二进制和输入文件 SHA256。
summary.json 同时给出以“无 CV 的完整 MEAM 步时”和“基线 Pair 势计算时间”为单位的结果。
新增开销是有偏完整步时减去无偏完整步时，包含 CV、图、偏势、额外邻居表与通信。
Modify 分项还包含 NPT 积分器；扣除基线 Modify 后只能用作偏置处理阶段的近似归因。

实测汇总见 [310 性能记录](PERFORMANCE_310.md)。

## 平衡概率与重加权

新增[共享偏势概率验证](PROBABILITY.md)：耦合双阱参考分布、共同沉积后冻结、
HILLS/表格重启、实际 CUDA 求值器和 CLI 交叉检查。统计验证独立于上述力/virial 测试。

[CV 压缩测试](COMPRESSION.md)进一步检验 90% 概率压进 10% CV 区间后的共同沉积、
偏势高度、原坐标 FES 和重加权样本退化，保留未达标结果及其覆盖掩码。
