# 分层构建

根 CMakeLists 负责选择模块；每个子目录创建自己的 target。头文件路径、宏和链接库通过 target 传播。

## 目标与依赖

| 目标 | 内容 | 外部依赖 |
| --- | --- | --- |
| `zqc_views` | ArrayView、VirialView、EngineAdapter、诊断与求和回调接口 | C++17 |
| `zqc_bias` | 完整 hill 列表、PBMetaD、历史格式与解析梯度 | C++17 |
| `zqc_graph_runtime` | Graph、Node、Leaf、表达式解析、NVRTC 执行 | CUDA、NVRTC、fmt、xxd |
| `zqc_cv_assembly` | 选择组索引、CV 值装配、梯度收集 | CUDA |
| `zqc_cvs` | 各种生产 CV 和注册工厂 | CUDA、view、assembly |
| `zqc_cv_graph` | CV 到 Leaf 的装配与图执行桥接 | Graph、assembly |
| `zqc_sampling` | 网格、维度与偏势的 LAMMPS 桥接 | CUDA、LAMMPS、MPI、exprtk |
| `zqc_lammps_adapter` | 宿主数组、近邻表、通信与日志适配 | CUDA、LAMMPS、MPI |
| `fix_crystallize_plugin` | LAMMPS fix/compute 入口 | 上述运行模块 |
| `zqc_ml_cvs` | 可选 ONNX Runtime 模块 | CUDA、ONNX Runtime、cuDNN |

`zqc_cvs` 使用 OBJECT 库保留各 CV 的注册构造函数。Graph 设备函数文本由单独的 `zqc_graph_embedded` 静态库提供，修改 `zqc_graph_device.cuh` 会自动重新生成。独立数值测试共用 `zqc_validation_kernels`，减少同一设备源文件的重复编译。

zqc_bias 可在纯 Host 预设中独立编译和测试。网格/采样桥接层仍有宿主依赖，因此归入 LAMMPS 构建范围。生产 CV、Graph/Leaf、装配层和 CVGraph 桥接均通过通用 view/回调访问宿主；`core` 可独立编译这些模块。

## 常用命令

在仓库根目录运行。预设需要 CMake 3.24+，手动 `-S/-B` 配置支持 3.18+。
CUDA 预设默认 `CMAKE_CUDA_ARCHITECTURES=native`，为配置时可见的 GPU 编译，避免落入 nvcc 的旧架构默认值。
在无 GPU 的构建机、交叉编译或发布给其他 GPU 时，显式传入目标架构，例如 `-DCMAKE_CUDA_ARCHITECTURES=89` 或 `'-DCMAKE_CUDA_ARCHITECTURES=80;89'`。
`native` 自 [CMake 3.24](https://cmake.org/cmake/help/v3.31/release/3.24.html) 起支持；纯 Host 预设不启用 CUDA。依赖路径按本机安装调整。
编译器应在首次配置或 toolchain 中指定；换编译器时使用新的构建目录。

纯 C++ 接口测试：

```bash
cmake --preset host-tests
cmake --build --preset host-tests -j2
ctest --preset host-tests
```

Graph/NVRTC 与数值验证：

```bash
cmake --preset graph-tests \
  -DMPI_CXX_COMPILER=/path/to/mpicxx -DMPIEXEC_EXECUTABLE=/path/to/mpirun
cmake --build --preset graph-tests -j2
ctest --preset graph-tests
```

若只验证单进程 Graph 和 CUDA 算法，配置时追加 `-DZQC_TEST_MPI=OFF`。独立测试不会查找 LAMMPS。fmt 固定使用项目内 third_party/fmt/include 中的上游 10.2.0 头文件及许可证，不再查找系统或 LAMMPS 的 fmt。NVRTC 源码只包含设备函数，无需 fmt include 路径。

只编译计算库：

```bash
cmake --preset core
cmake --build --preset core --target zqc_graph_runtime zqc_cv_assembly -j2
cmake --build --preset core --target zqc_cvs zqc_cv_graph -j2
```

构建 LAMMPS 插件：

```bash
cmake --preset lammps \
  -DLAMMPS_SOURCE_DIR=/path/to/lammps/src \
  -DLAMMPS_LIB_DIR=/path/to/lammps/lib -DMPI_HOME=/path/to/mpi
cmake --build --preset lammps -j2
```

产物为 `build/lammps/fix_crystallize_plugin.so`。也可以在同一构建目录启用 `ZQC_BUILD_TESTS=ON`，让插件和测试复用 Graph 运行库。

日常修改后直接执行 `cmake --build`。CMake 按源文件和头文件依赖增量编译；指定 `--target zqc_graph_runtime`、`zqc_cvs` 或 `zqc_lammps_adapter` 可以缩小本次检查范围。

## 开关及兼容

- `ZQC_ENABLE_CUDA=OFF`：构建 Host 偏势库和接口测试；关闭测试时保留库与接口 target。
- `ZQC_BUILD_CVS=ON`：加入生产 CV。CVGraph 桥接层随 Graph 构建，可使用 mock CV 独立验证。
- `ZQC_BUILD_LAMMPS=ON`：加入适配器和插件，同时启用 CV。
- `ZQC_BUILD_TESTS=ON`：加入独立测试；`ZQC_TEST_MPI` 控制 MPI 测试组。
- `ZQC_GRAPH_STANDALONE=ON`：保留旧用法，启用 Graph 测试并关闭 LAMMPS 插件。
- `USE_ML_CVS=ON`：显式启用 ML 依赖检查，默认关闭。
- `CMAKE_BUILD_TYPE`：默认 Release；`ENABLE_DEBUG=ON` 额外开启 CUDA device debug 和诊断宏。
- `ENABLE_ASAN=ON`：给参与计算和插件链接的 target 加上 Host ASan。
- `CLEAN_HOST_COMPILER`：兼容旧变量；优先使用标准 `CMAKE_CUDA_HOST_COMPILER`。

新增 CV 时，把实现文件加入 `src/cvs/CMakeLists.txt` 的显式源码列表；新宿主适配器应拥有自己的 target 和依赖，不修改 Graph/CV 的包含路径。

## 本次验证

- `host-tests` 预设完成配置、编译，6/6 测试通过（含偏置核心、网格与宿主生命周期）。
- `build/layers-graph` 在 MPI 测试关闭时独立编译，10/10 测试通过；启用 MPI 后 25/25 通过。
- Graph、装配和 CVGraph 桥接的 7 份编译依赖文件均未引用 `mpi.h`、`lammps.h` 或 `fix_crystallize.h`；Graph smoke 的动态库依赖也没有 MPI/LAMMPS。
- `build/layers-lammps` 完成配置，`zqc_lammps_adapter` 目标独立编译通过。
- `zqc_cvs` 和完整 `fix_crystallize_plugin` 已编译通过。14 个生产 CV/Graph/装配/桥接翻译单元的编译依赖没有原生 LAMMPS/MPI 头文件。
- 开启 CV 和测试后新增 `zqc_production_cv_views`，使用实际 CV 工厂及计算实现验证 AoS/SoA、类型基数、带步长 Leaf 输入、力和完整九分量 virial；详见 `benchmarks/graph/ENGINE_ADAPTER.md`。

偏置迁移后的完整 layers-lammps CTest 为 30/30；完整 LAMMPS 沉积/重启/差分 33/33，测量与接口见 [高维偏置说明](../benchmarks/highdim/README.md)。

## GPUMD 可选桥接

`cmake --preset gpumd -DZQC_GPUMD_SOURCE=/path/to/GPUMD` 选择 GPUMD v5.9.1
EnhancedSampling 桥接，默认不加入两原子验证势。开启
`-DZQC_GPUMD_VALIDATION_MORSE=ON` 可编译 DISTANCE + O₂ FES 验证程序。
程序目标为 `gpumd_metadtoy`；启用 ZQC_BUILD_TESTS 后增加 engine、neighbors、runtime ABI 验证目标。构建及测试依赖、原生数组布局和
输入示例见 [GPUMD 验证](../benchmarks/gpumd/README.md)。

GPUMD 的动态运行库可用 `gpumd-runtime` 预设单独构建，不需要提供 `ZQC_GPUMD_SOURCE`；目标名为 `metadtoy`。首次宿主接入使用 `gpumd` / `gpumd-tests`，之后只更新算法库可只构建该目标。接口和部署说明见 [GPUMD 验证说明](../benchmarks/gpumd/README.md#动态库与-gpumd-的更新边界)。

## 安装离线 CLI

CLI 作为可选组件接入 CMake，使用 Python >=3.9 和 pip，目标为
`metadtoy-cli`，安装组件为 `cli`。配置 `ZQC_BUILD_CLI=ON` 可随现有 MD
构建打包；`ZQC_CLI_ONLY=ON` 则只配置 CLI，入口预设为 `cli`。

```bash
cmake --preset cli -DPython3_EXECUTABLE=/path/to/python -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build --preset cli
cmake --install build/cli --component cli
export PATH="$HOME/.local/bin:$PATH"
metadtoy --help
```

构建生成 wheel，安装创建前缀下的专用 Python 环境及 `bin/metadtoy`。
默认 CLI 开关为 OFF。依赖、离线包源和安装目录说明见 [CLI 安装](../tools/README.md)。

## CUDA/Kokkos LAMMPS

使用 lammps-kokkos 预设或 ZQC_LAMMPS_KOKKOS=ON，并指定宿主实际构建的 Kokkos_DIR。
编译器 ABI、依赖边界和旧版 trim 检测说明见 [Kokkos 指南](../benchmarks/kokkos/README.md)。
