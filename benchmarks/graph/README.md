# Experimental runtime graph prototype

这是运行时图编译的实验性原型（experimental prototype / proof of concept），用于功能验证和性能对比，不是正式发布版本。当前 LAMMPS 的 `evaluate_bias` 仍使用原有 DimSet/ExprTk 路径，尚未接入本图实现。

## 文件与职责

- `include/utils/zqc_graph.cuh`：节点、表达式解析、拓扑排序和前向/反向代码生成。
- `include/utils/zqc_graph_device.cuh`：共享的 inline 数学函数与导数，构建时通过 xxd 自动嵌入 NVRTC 源码。
- `include/utils/zqc_switch_function_device.cuh`：旧径向开关函数的参数和接口适配，复用上述公式。
- `graph_smoke.cu`：示例 main，打印融合源码、NVRTC 编译并加载前向/反向 kernel。
- `graph_sw_func.cu`：实际启动 GPU kernel，对照独立数值参考验证结果和梯度。
- `../graph_kernel_20261013/`：融合图与手写 kernel 的性能对比程序及历史报告。

## 构建与验证

在仓库根目录执行。需要 CUDA Toolkit（含 NVRTC）、可用的 NVIDIA GPU、MPI、xxd，以及 LAMMPS 源码目录中的头文件和 fmt；独立目标不链接 LAMMPS 插件。

```bash
cmake -S . -B build/graph \
  -DZQC_GRAPH_STANDALONE=ON \
  -DLAMMPS_SOURCE_DIR=/path/to/lammps/src \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/graph -j2
ctest --test-dir build/graph --output-on-failure
```

`89` 对应本次验证使用的 GPU；其他 GPU 请使用对应的编译架构。

打印生成的完整 kernel 源码：

```bash
./build/graph/benchmarks/graph/zqc_graph_smoke > build/graph/generated_sample.cu
```

`zqc_graph_smoke` 只验证生成、编译和加载。数值验证由 `zqc_graph_sw_test` 完成，涵盖 13 组开关函数参数、共享节点/输入的梯度累加、Mean 反向种子、旧径向接口、异常参数和数组边界。

已有性能对比脚本也使用 `build/graph`，可通过 `--build-dir` 改变位置：

```bash
LAMMPS_SOURCE_DIR=/path/to/lammps/src \
  python3 benchmarks/graph_kernel_20261013/run.py --build-dir build/graph
```

该性能脚本目前针对 sm_89；历史报告对应当时的源码与环境，不代表后续每次修改后的测量结果。`build*/`、CMake 缓存和基准程序的 `results*/` 为本地产物，保留源程序和精选报告供复现。

## 当前边界

- 融合路径支持等长的逐原子标量输入；维度描述尚不意味着已支持通用向量和广播。
- 多输出 DAG 可以共享节点；每张可融合图生成一个前向 kernel 和一个反向 kernel。
- `.MEAN` 只允许出现在输出端。融合 kernel 产生规约前的逐原子值，最终规约和反向种子由调用方处理，尚未接入 LAMMPS/MPI 规约。
- 头文件仍是单个 CUDA 翻译单元使用的原型；正式执行器、编译缓存和生产集成留待后续实现。
