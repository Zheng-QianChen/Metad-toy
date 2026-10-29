# CV 与 Graph 有限差分验证

验证当前可构建的所有非 ML CV：STEINH Q3/Q4/Q6、Local Q3/Q4/Q6、STRU_FACTOR、STRU_FACTOR Chemical、WEIGHT_CHEM、CSS 和 DISTANCE。化学权重同时检查显式映射与高斯回退，CSS 检查二元、三元与多个半径通道。`src/ml_cvs/mlcvs.cu` 目前只有 ONNX provider 探测函数，没有可注册的 ML CV。

最新结果见 [DISTANCE 修复报告](DISTANCE_RESULTS.md)：69/69 场景通过，包括全局标量和混合 Graph；[RESULTS.md](RESULTS.md) 保留修复前的检查记录。

## 完整 LAMMPS 验证

```bash
# 先按项目现有配置构建插件；Python 需能导入 lammps、mpi4py、numpy。
cmake --build build/graph-lammps -j4
python benchmarks/finite_difference/run.py --output build/finite-difference-new
```

可通过 `--plugin`、`--mpi`、`--mpicxx`、`--lammps-src` 指定本机路径，`--mpi-args` 传入 launcher 附加参数，`--np 1 2 4` 指定 rank 数。输出目录必须尚不存在。`--cases` 支持挑选场景，例如：

```bash
python benchmarks/finite_difference/run.py --np 1 2 4 \
  --cases q3 q4 q6 l3 l4 l6 structure chemical weighted css_ternary \
  --output build/finite-difference-cvs
```

`bias_probe.cpp` 是独立测试动态库：从 Fix 公开接口取得当前 `dV/ds`，并在偏离当前 CV 的位置放一个固定 hill。生产 CV、Graph、装配和 MPI 通信均由 LAMMPS `run 0` 执行。夹具使用 `pair_style zero`，所以测到的力和 virial 全部来自偏置。`PACE 0` 固定偏置历史，所有扰动都保持同一 timestep。

每个坐标的数值导数为：

```text
D_h f(x) = [f(x+h) - f(x-h)] / (2h)
```

单输出场景同时检查两层关系：

```text
dV/dx = -force              对照偏置能的差分
ds/dx = -force / (dV/ds)    对照 CV 值的差分
```

对三个对角应变和 xy/xz/yz 三个剪切应变，仿射变换晶胞与原子坐标，检查 `dV/dε = -virial` 和 `ds/dε = -virial/(dV/ds)`。双输出非线性 Graph 直接检查总偏置能的导数。固定 hill 保证种子非零，测试会拒绝零信号和非有限力。

- 初态为带确定性扰动、三种原子类型的 32 原子周期 FCC；单 rank 遍历全部 96 个坐标，多 rank 检查选定的 8 个原子、24 个坐标，包含跨子域与组外邻居。
- 每个场景扫描 `h=1e-4,3e-5,1e-5,3e-6`，并检查扰动后恢复原态。
- 通过标准为 `abs(error) <= atol + rtol*max(abs(analytic),abs(numerical))`；两个较小步长均须通过。力 `(atol,rtol)=(1e-8,2e-5)`，virial `(1e-7,3e-5)`，CV 坐标导数 `(2e-7,2e-5)`，CV 应变导数 `(2e-6,3e-5)`。
- `ql_linear`、`chemical_linear`、`all_linear` 使用正负系数，并额外对照独立 CV 的值、坐标梯度和应变导数的加权和。
- `mixed_groups` 检查不同中心组及零填充；`nonlinear_graph` 检查逐原子乘积、开关、规约后的非线性和两个输出。

差分验证适用于这些采样点的光滑区域。开关阈值、硬截断边界等不可导点应单独检验边界语义，不能用普通中心差分通过与否判断其导数。

## 独立 Graph 验证

```bash
cmake --build build/graph -j4
ctest --test-dir build/graph -R zqc_graph_fd --output-on-failure -V
```

`graph.cu` 对实际 `GraphExecution::forward()` 做差分，对照 `backward()` 的完整 Leaf Jacobian。Host 路径是两个 Global 输入、14 个输出；NVRTC 路径是两个各含 19 个样本的 PerAtom Leaf 加一个 Global 输入，共 39 个变量、18 个输出。

覆盖加减乘、重复输入、负号、linear、exp、constant_one、step 常值区、fermi、tanh、两种 rational 幂配置、sw_func 适配器、共享乘积和共享 AVE。每个输出逐项作为种子，再检查正负混合种子 VJP、零种子及重复 backward 清零。MPI 1/2/4 rank 使用同一组全局样本，多 rank 时 rank 0 为空，其余 rank 不均匀分配。

步长为 `1e-3,1e-4,1e-5`，后两个步长均要求误差满足 `atol=3e-8, rtol=2e-6`。大步长用于观察截断误差。

## 记录格式

`run.py` 输出逐场景 JSON、逐坐标/应变明细、可复现的 LAMMPS 输入、布局诊断与日志。`summarize.py` 将本轮固定目录布局下的精选记录生成到本目录；原始大文件保存在 `build/finite-difference/`。`metadata.json` 区分完整扫描时的脚本摘要与后续仅增加失败诊断的版本。

## compute 与逐原子 stress 输出

`compute_outputs.py` 复用 WEIGHT_CHEM、化学 Stru_factor 和 CSS 的生产夹具，检查单列/多列输出一致、group 外零填充、连续两帧数据、逐原子 stress 求和与全局 pressure，以及按全局 ID 对齐的 MPI 一致性。

```bash
python benchmarks/finite_difference/compute_outputs.py \
  --plugin build/graph-lammps/fix_crystallize_plugin.so \
  --np 1 2 4 --output build/compute-output-validation
```

该脚本只用 Python 标准库，调用实际 LAMMPS 可执行文件；可通过 `--lmp`、`--mpi` 指定安装路径。输出目录必须尚不存在。详细范围与数值记录见 [引擎适配验证](../graph/ENGINE_ADAPTER.md#virial-与-compute-输出补测)。

## virial 引擎边界

`engine_virial.py` 复用本目录 `run.py` 生成的 `structure_npN` 夹具，向真实适配器注入带已知分量值的 owned/ghost 数组。参考值按全局原子 ID 独立规约，检查 LAMMPS 六项投影和 reverse 累加。

```bash
python benchmarks/finite_difference/engine_virial.py \
  --plugin build/graph-lammps/fix_crystallize_plugin.so \
  --fixture build/engine-adapter-validation/structure_np2 \
  --np 1 2 4 --output build/virial-boundary-validation
```

每个 rank 配置包含 8 种组合：六/九分量、Host/CUDA、AoS/带预留容量的 SoA。另检查单个 owned 原子的六/九分量提交、非法形状拒绝和源数组保持原样。需要带 `lammps`、`mpi4py` 的 Python；脚本从插件的 CMake 缓存寻找 NVCC/LAMMPS 头文件，也接受 `--nvcc`、`--lammps-src`、`--mpi`、`--mpicxx`。

这个检查验证输出接口和通信；力/virial 的物理导数由 `run.py` 验证。GPU 完整九分量保留由 `zqc_engine_geometry` 检查，真实 GPUMD 后端仍待接入。

## 数据需求与跨步缓存

`engine_cache.py` 复用 `run.py` 的 structure 夹具，检查未引用 Leaf、共享选择集、
compute 独立需求和移除、动态组、坐标微移、真实排序/迁移/ghost 重建，以及全局 DISTANCE。
对缓存计算与同一步冷重建比较值和力，并通过适配器计数器检查扫描/上传次数。
原生探针额外覆盖同长度下的成员交换、类型变化和稳定 ID 重排。

```bash
python benchmarks/finite_difference/engine_cache.py \
  --plugin build/lammps/fix_crystallize_plugin.so \
  --fixture build/validation/structure_np1 \
  --np 1 2 4 --output build/engine-cache-validation
```

输出目录应尚不存在；脚本自动编译测试探针，保存结果和被测插件 SHA-256。
计数器验证数据准备行为，耗时收益需要另外在实际体系上测量。

## Half 邻居表与配对归属

`engine_neighbors.py` 自动生成 12 原子的周期夹具，通过真实 LAMMPS 请求 Half 表，并与独立筛选的 Full 表参考比较配对数、二次 pair 能量、两端力、九分量总 virial 和逐原子 reverse 回写。

```bash
python benchmarks/finite_difference/engine_neighbors.py \
  --plugin build/lammps/fix_crystallize_plugin.so \
  --np 1 2 4 --output build/half-neighbor-validation
```

需要能导入 `lammps`、`mpi4py` 的 Python；路径选项与 `engine_virial.py` 相同。每个 rank 配置覆盖 orthogonal/triclinic/小周期盒三种几何、`newton on/off`、初始/平移排序/组变更三个阶段，以及九种选择集组合（同组、异组、重叠、单侧 all、空组与 all 别名）。小盒的 cutoff 大于盒长，显式覆盖同 ID 的周期自镜像。测试还验证新 CUDA 流等待候选上传事件、稳定 topology 下复用缓存，以及 `Half + ghost_centers` 的非法组合拒绝。

2026-10-22 验证：MPI 1/2/4 rank 共 486 组选择集/配置检查通过，最大绝对差异 `4.434e-12`。本次原始记录位于 `build/half-neighbors-review/mpi-validation/`，后续安装验证使用上面的命令生成新记录。

独立 `zqc_production_cv_views` 检查所有生产 CV 的九分量形变有限差分与总 virial 对称性；`zqc_engine_contract` 使用旋转不变的三体解析例子验证逐原子非对称项能正确抵消。对称性检查仅针对具有旋转不变性的测试能量。

## Local Q ghost 候选跨步复用

```bash
python benchmarks/finite_difference/ghost_cache.py \
  --plugin build/lammps/fix_crystallize_plugin.so \
  --np 1 2 4 --output build/ghost-cache-validation
```

测试自动构造 pair cutoff=2.5 Å、CV cutoff≈4.239 Å 的二跳邻域，
用独立逐对枚举检查当前物理邻居是否全部包含在缓存 CSR 中：

- 两原子分别在 skin/2 内移动，跨过真实 cutoff，宿主邻居表保持不变；
- 一个请求延迟到两原子远离时才首次求值，随后它们靠近，验证延迟创建的余量；
- 动态组每步交换成员，检查几何候选复用和成员筛选；
- NPT 与倾斜盒变形、排序迁移、run 间修改 skin；
- 同一构型缓存/冷重建的 CV、偏置能、非零力及压力乘体积比较。

倾斜盒采用偶数步 `fix deform`，在没有末尾重映射的奇数步取样，使两次力计算
对应相同的坐标与盒子。`ghost_row_rebuilds` 单独检查空间搜索次数。
输出包含逐步计数器、最大数值误差、输入与插件 SHA-256；支持 `--cases` 选择场景。
Python 环境需包含 `lammps`、`mpi4py`、`numpy`，路径参数与 `engine_cache.py` 相同。


## Graph 力和 virial 合并回传

`force_batch.py` 使用真实 LAMMPS 适配器检查 LocalQL＋CSS 每轮的
“2 次贡献提交、1 次 ghost reverse、1 次回写”。C++ 探针对 Host/CUDA、
AoS/SoA、六/九分量 virial 共八种组合按全局原子 ID 独立求和，另外检查
延迟写回、输入复用、取消、空批次、非法生命周期和重复求值。

```bash
cmake --build /path/to/build --target zqc_force_batch_probe
mpiexec -np 2 python benchmarks/finite_difference/force_batch.py \
  --plugin /path/to/build/fix_crystallize_plugin.so \
  --probe /path/to/build/zqc_force_batch_probe.so \
  --fixture /path/to/completed-small-al3sc-fixture \
  --output /path/to/new-batch-test
```

fixture 使用 `benchmarks/shared_frozen/al3sc.py` 生成的小体系目录，
包含 initial.data、shared.table 和 in.lammps。测试可加 `--single-center`，
使两个 CV 都只以 atom 1 为中心，验证其他 rank 空中心时仍参加统一通信。
Kokkos 参数为
`--lammps-args="-k on g 1 -sf kk -pk kokkos gpu/aware off neigh half newton on"`。
探针只在独立测试进程中使用，测试结束会恢复 MD 力和 virial。

新旧插件性能对照使用 `force_batch_performance.py`，复用 shared_frozen/performance.py
的相同 Al3Sc NPT 输入，交错运行三轮，排除 NVRTC 和预热。
验证结果及范围见 [310 合并回传记录](../shared_frozen/FORCE_BATCH_310.md)。
