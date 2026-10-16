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
