# GPUMD 接入：DISTANCE / StruFactor → Graph → 偏置

这是一条真实 GPUMD v5.9.1 积分路径。CMake 在构建目录复制上游源码，再给
EnhancedSampling Action 加可选转发入口；MetaD-toy 的 CV、Graph、偏势数学层继续共用。
未选择 `backend metadtoy` 的 EnhancedSampling 配置仍使用 GPUMD 原生实现。

## 构建

准备 GPUMD v5.9.1 源码（包含 `src/` 和 `LICENSES/`）。本仓库不捆绑其完整源码。

```sh
cmake -S . -B build/gpumd \
  -DZQC_BUILD_LAMMPS=OFF \
  -DZQC_BUILD_GPUMD=ON \
  -DZQC_BUILD_TESTS=ON -DZQC_TEST_MPI=OFF \
  -DZQC_GPUMD_SOURCE=/absolute/path/to/GPUMD \
  -DZQC_GPUMD_VALIDATION_MORSE=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build/gpumd -j4
ctest --test-dir build/gpumd -L gpumd --output-on-failure
```

也可以使用 `cmake --preset gpumd-tests -DZQC_GPUMD_SOURCE=/path/to/GPUMD`。
`gpumd` 预设只构建程序；`gpumd-tests` 额外构建验证目标。

`native` 需要 CMake ≥3.24；较早版本显式指定架构，例如 RTX 4090 的 `89`。
C++17、CUDA、cuBLAS/cuSOLVER/cuFFT 是程序构建依赖。Python ≥3.8 用于构建目录补丁；
NumPy 用于端到端验证，Matplotlib 用于 FES 图。MD 每步计算均为 C++/CUDA。

`ZQC_GPUMD_VALIDATION_MORSE` 默认关闭。开启后额外提供 `metadtoy_morse`，
它只接受两个 O 原子，是核对能量、力、virial 和 FES 的已知参考模型。
真实材料计算应使用 GPUMD 自己的势。

## 配置与数据边界

在 GPUMD 的 `run.in` 中加入：

```text
enhanced_sampling sampling.in
```

`sampling.in`：

```text
backend metadtoy
start fresh
output COLVAR 10
cv r distance atoms 0 1
expression r
warmup 20000
bias metad temperature 300 sigma .015 height .02 biasfactor 20 pace 100
grid .85 3.0 4300
history HILLS
```

- `backend metadtoy` 必须是第一条有效指令；`#` 开始注释。
- 原子编号是 `model.xyz` 中从 0 开始的行号，进入内部 CV 时转为 `AtomId=row+1`。
- 可声明多个 DISTANCE，再由一个 Graph 表达式组合，例如 `expression a*b+a`。
- 距离遵循盒子的周期标记。长度用 Å，能量用 eV，温度用 K。
- `warmup` 期间不沉积；之后在步号满足 PACE 时沉积。`setup_force` 只计算初始力。
- COLVAR 列为 `step value bias_eV dV_dvalue`。梯度为能量对图输出的导数，
  CV backward 转为原子力。HILLS 保存实际沉积高度及版本化参数。
- 确定性测试也支持 `bias harmonic center 1.21 kappa 8`，可与 GPUMD 原生谐和偏置交叉核对。

第一版支持单进程、固定原子顺序、正交固定盒，NVE 或恒温 NVT_LAN/NVT_NHC/NVT_BDP。
目前开放 DISTANCE、StruFactor、单输出 Graph 和一维网格 WTMetaD；NPT、固定/移动组
及完整 MD 重启会明确报错。`start fresh` 会新建本次偏置历史，不能替代 MD 状态重启。

原生 position/force 为 `[x(N),y(N),z(N)]` SoA，virial 为九个长度 N 的分量块，
顺序 `xx yy zz xy xz yz yx zx zy`。ArrayView 保存实际 stride。数组所有权归 GPUMD；
每次 `bind_frame` 重新借用并记录生产流事件，`finish` 让 MD 流等待偏置回写。
偏置力与 virial 累加到已有结果，总偏置能量均分至每原子 potential。
当前 DISTANCE 只读取点名原子的坐标到 Host；全局标量 Graph 在 Host 执行。


## StruFactor、中心组与周期像

例如，对指定中心组的逐原子结构因子求均值，再施加谐和偏置：

```text
backend metadtoy
start fresh
output COLVAR 10
neighbor_skin 0.3
group centers atoms 0 1 2 3
cv s strufactor group centers cutoff 4.3 q 2.1 reduce mean
expression s
bias harmonic center 1.0 kappa 2.0
```

`group centers atoms ...` 声明固定的 0-based 原子行号集合，`group all`
选择全部 owned 原子。这里的名称由 MetaD-toy 配置管理；GPUMD 原生 group
编号目前尚未映射。邻居环境包括所有原子及其周期像，中心组外的邻居同样获得力。

`reduce mean` 在 Graph 层对该 CV 中心组求均值，分母排除组外 padding 与周期像。
`reduce raw`（省略 reduce 时的默认值）提供逐原子 Leaf，可用
`expression exp(s).AVE` 对逐原子变换后求均值。多个 raw CV 的公共 Leaf
按各中心组的 owned 并集装配，组外补零；对公共表达式的 `.AVE` 按该并集计数。
需要各组单独均值时，应为各输入明确指定 `reduce mean` 后再组合。

例如同时定义不同中心组、不同 cutoff 的 `a`、`b`，以及 DISTANCE `r`，
可使用 `expression a*b+r`。q 的单位为 Å⁻¹，cutoff 与 skin 的单位为 Å。

邻居表由运行库独立构建，CV cutoff 可以大于底势 cutoff。重建时在 CPU
使用空间网格分箱，并形成 Full/Half CSR；每轮在 GPU 检查相对重建构型的
位移，Host 读取一个标志。位移超过 skin/2、盒子或请求变化时重建，否则
GPU 刷新扩展坐标和类型。skin=0 可用于每轮重建的对照。

周期像保存为 owned 后面的扩展原子行，每行保存 owner 与整数晶格平移。
支持 cutoff 超过盒长：同一原子的不同平移分别参与求和，零平移的中心自身
排除。像上的三分量力和九分量 virial 由 adapter 累加回 owner；自像可以
具有非零 virial，而净力相消。候选缓存沿用重建时的坐标参考，跨周期边界
的小位移会保持几何连续。

当前 provider 提供 owned 中心的 Full/Half 请求；需要 ghost 中心完整环境的
LocalQL 请求尚未接入。高密度或 cutoff/盒长很大时，像数量及重建成本会增加。
本轮仅验证正确性，尚未给出大体系性能结论。

运行端到端验证：

```sh
python benchmarks/gpumd/strufactor.py --gpumd build/gpumd/bin/gpumd_metadtoy \
  --output build/gpumd-strufactor
```

此验证用 cutoff=2.0 Å 的 Morse 夹具，对照 cutoff=4.3 Å 的 StruFactor；
检查多像、非周期/混合周期、独立中心组、逐原子非线性图、与 DISTANCE
组合、力与对角应变 virial 有限差分、缓存与每步重建等价。近邻契约测试另
覆盖 Half 的正反选择方向、跨边界缓存及 Host/CUDA 两种 Full9 回写路径。

## 可重复验证

快速测试会检查 Morse 解析值、GPUMD 原生与桥接谐和偏置、非线性图、
两个 CV 共用原子的梯度累计、实际 GPUMD 总能量的中心有限差分，
以及 2000 步 Langevin/WTMetaD 的沉积时序：

```sh
python benchmarks/gpumd/run.py --gpumd build/gpumd/bin/gpumd_metadtoy \
  --mode smoke --output build/gpumd-smoke
```

完整的最小 FES 例子：

```sh
python benchmarks/gpumd/run.py --gpumd build/gpumd/bin/gpumd_metadtoy \
  --mode fes --steps 2000000 --seed 73517 --output build/gpumd-o2
python benchmarks/gpumd/reconstruct.py build/gpumd-o2
python benchmarks/gpumd/validate_fes.py build/gpumd-o2
```

显式输出目录必须不存在；省略 `--output` 时自动新建目录，便于重复运行 CTest。
输入、GPUMD 日志、HILLS、COLVAR 和检查 JSON 都保存在该目录。

重建脚本只读取 HILLS 与采样元数据，先得到径向自由能
`F(r) = -gamma/(gamma-1) V(r) + C`，然后对自由转动双原子使用
`U(r) = F(r) + 2 k_B T log(r/Å) + C`。验证脚本才读取已知 Morse 参考。
采样前固定比较区间为 1.05–1.60 Å，目标 RMSE <25 meV、最大误差 <75 meV，
并检查轨迹覆盖。短 smoke 只能证明接入和沉积正确，不能证明 FES 收敛。
这是 Morse 流程验证，不是对真实氧分子电子势能面的预测。


## 动态库与 GPUMD 的更新边界

GPUMD 构建产物现在是 `bin/gpumd_metadtoy` 与 `lib/libmetadtoy.so.1`。
EnhancedSampling 的薄桥接仅包含宿主类型转换和生命周期转发；
配置解析、EngineAdapter、CV、Graph、NVRTC 和 BiasSession 均在共享库内。

公开接口为 `include/adapters/zqc_metadtoy_api.h`：版本化 C 函数表、
不透明状态句柄和按元素描述跨度的 CUDA 数组。GPUMD 的 Atom/Box/Integrate、
项目内部 C++ 类、STL 容器和异常均由各自一侧管理。错误通过宿主持有的文本
缓冲回传；日志通过同步回调转发。每轮库先等待宿主流的输入，再让宿主流
等待力/virial/势能回写完成。运行库与宿主需使用兼容的 CUDA 和平台依赖。

初次启用桥接需要构建 GPUMD。之后只修改库内算法时，在原构建目录执行：

```sh
cmake --build build/gpumd-tests --target metadtoy -j
```

也可以在没有 GPUMD 源码的源码树中独立构建运行库：

```sh
cmake --preset gpumd-runtime
cmake --build --preset gpumd-runtime -j
cmake --install build/gpumd-runtime --prefix /path/to/metadtoy-package --component runtime
```

部署目录保持 `bin/` 与 `lib/` 并列，可执行文件通过 `$ORIGIN/../lib` 查找库。
更新应在新进程启动前完成，保持 `libmetadtoy.so.1` 的 SONAME 和公开 ABI
兼容。运行中的任务继续使用已加载的版本；发布时采用新文件加原子替换，
避免原地改写已加载的库文件。仅输入表达式或参数变化时直接重新运行。

公开结构布局、函数签名或宿主所需服务发生不兼容变化时，需要提升 ABI /
SONAME 并重建相应桥接。GPUMD 升级后也需重新核对注入位置和宿主数组约定。

`zqc_gpumd_runtime_abi` 只包含公开头文件，验证版本拒绝、错误字符串、
日志回调、非默认流和 Distance 力/virial；`zqc_gpumd_distance_smoke`
继续运行真实 GPUMD 积分和短程 WTMetaD。两项均由 `gpumd-tests` 预设构建。

StruFactor 的独立周期像候选表已在共享库内接入，公开 C ABI 保持版本 1。
本次仅重新构建运行库，既有 GPUMD 可执行文件的 SHA-256 保持不变。

动态库拆分的 310 验证结果见 [RESULTS_SHARED_310.md](RESULTS_SHARED_310.md)。

StruFactor 接入的结果与范围见 [RESULTS_STRUFACTOR_310.md](RESULTS_STRUFACTOR_310.md)。
