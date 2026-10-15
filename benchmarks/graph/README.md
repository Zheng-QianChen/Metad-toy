# Experimental runtime graph prototype

这是运行时图编译的实验性原型（experimental prototype / proof of concept），用于功能验证和性能对比，不是正式发布版本。LAMMPS 的 `evaluate_bias` 已可通过 `EXECUTOR GRAPH` 接入本图实现。Stru_factor、化学结构因子、CSS 和 WEIGHT_CHEM 已提供该接口。

## 文件与职责

Graph 的头文件统一为两个：`zqc_graph.cuh` 提供完整接口，`zqc_graph_device.cuh` 提供供 NVRTC 编译的内联数值函数。使用方包含 `zqc_graph.cuh` 即可构图和执行；各类实现继续放在 `src/utils` 中。

- `include/utils/zqc_graph.cuh`：值描述、`Node` 及内置节点、表达式解析、`Graph`、`Codegen` 和 `GraphExecution` 的声明。
- `src/utils/zqc_graph_parser.cu`：只解析语法，生成符号、数值和调用组成的表达式树。
- `src/utils/zqc_graph_nodes.cu`：节点实现、各节点的参数读取函数、内置节点注册及旧 `sw_func` 适配。
- `src/utils/zqc_graph.cu`：图的所有权、注册分派、拓扑检查与前向/反向代码生成。
- `src/utils/zqc_graph_execution.cu`：NVRTC 编译与执行，调度 GPU 融合段、Host/MPI 规约及后续全局标量运算。
- `include/utils/zqc_graph_device.cuh`：共享的 inline 数学函数与导数，构建时通过 xxd 自动嵌入 NVRTC 源码。
- `include/utils/zqc_switch_function_device.cuh`：旧径向开关函数的参数和接口适配，复用上述公式。
- `graph_smoke.cu`：示例 main，打印融合源码、NVRTC 编译并加载前向/反向 kernel。
- `graph_sw_func.cu`：实际启动 GPU kernel，对照独立数值参考验证结果和梯度。
- `graph_ave_mpi.cu`：1/2/4 rank 的 AVE、空 rank、不均匀分区、共享节点梯度与子通信器验证。
- `cv_assembly.cu`：不同原子组的 GPU 数值装配、图计算和梯度回收验证。
- `include/CVsets/CV_Stru_factor.h` / `src/cvs/CVs_Strufactor.cu`：Stru_factor 的环境、CV 接口、设备数据及前向/原始反向 kernel。独立测试以 `ZQC_STRUFACTOR_KERNELS_ONLY` 编译同一源文件中的设备计算部分。
- `strufactor_graph.cu`：分别编译普通结构因子、化学结构因子和 WEIGHT_CHEM，使用同一套生产 kernel 验证原始值、图梯度、坐标力和 virial，覆盖 1/2/4 rank。
- `graph_extension.cu`：在独立翻译单元中注册组合算子和外部规约节点，验证扩展接口及多文件链接。
- `../graph_kernel_20261013/`：融合图与手写 kernel 的性能对比程序及历史报告。

## 表达式与节点扩展

流程是 `表达式文本 → GraphExpr 语法树 → 注册的创建函数 → Node 图 → 融合源码`。解析器负责括号、优先级和参数分隔；每个节点的 `create()` 负责参数含义和校验。`Graph` 持有创建出的节点，传入 `symbols` 的叶子仍由调用方持有。

```cpp
#include "zqc_graph.cuh"

Leaf cv({0.2, 0.4, 0.6});
Graph graph(Graph::Symbols{{"cv", &cv}});
graph.parse_expr("(cv * fermi(cv, 0.5, 4)).MEAN");
graph.parse_expr("rational(cv, 1.25, 6, 12)");
graph.build();
graph.print_code();
```

内置调用包括 `neg(x)`、`exp(x)`、`linear(x,k,b)`、`constant_one(x)`、`step(x,r0)`、`fermi(x,r0,alpha)`、`tanh(x,r0,alpha)`、`rational(x,r0,n,m[,d0])`；原有 `+`、`-`、`*`、括号和输出后缀 `.MEAN` 保持可用。这里的 `tanh` 对应原有 `TanhSwitch` 开关函数。`rational` 省略 `d0` 时取零，幂参数必须是正整数。数字字面量用于算子的常量参数，普通图输入仍从 `symbols` 的节点取得。

原来的 `Graph::Switches` 与 `sw_func(cv, gate)` 也保留：适配器在构图时将配置转换为具体节点，融合 kernel 内没有这层类型分派。

新增节点时，在其实现旁定义创建函数，例如：

```cpp
Node* Fermi::create(Graph& graph, const GraphExpr& call) {
    call.require_args(3);
    const double r0 = call.number_arg(1);
    const double alpha = call.number_arg(2);
    Node* input = graph.create_node(call.args[0]);
    return graph.make_node<Fermi>(input, r0, alpha);
}
```

然后注册 `graph.register_node("my_fermi", Fermi::create)`，即可使用 `my_fermi(cv, 0.5, 4)`。注册按 Graph 实例隔离，重复名称会报错。创建函数也可以组合现有节点，例如测试中的 `scaled_square(x,k)` 创建 `Mul(x,x)` 后再创建 `Linear`；不必为每个组合增加节点类。所有新建节点通过 `make_node()` 交给图管理，`parse_expr()` 失败时会撤销本次创建的节点并保留原图。

新增内置算子只需在 `register_builtin_graph_nodes()` 加入名称与创建函数的对应关系。通用的函数调用无需修改解析器；增加新的中缀运算符语法才需要扩展语法解析。注册只发生在 CPU 构图阶段，不增加 kernel 内的函数指针调用。

`Node::is_reduction()` 和 `kernel_output()` 标记 GPU 到 Host 的规约边界：后者返回需要从融合 kernel 导出的逐原子节点。`reduce_host()` 执行规约，`reduction_seed()` 将上游梯度转换成每个输入元素的梯度。目前支持元素共享同一个反向标量种子的规约；这不是通用 Max 等规约的反向接口。

## Host AVE 与全局标量执行

`AVE(expr)` 可以出现在全局标量表达式内部。最终后缀 `.AVE` 与旧 `.MEAN` 采用同一个实现。定义图与执行图分开：`Graph::build()` 准备拓扑和融合源码，`GraphExecution` 编译并执行。

```cpp
#include "zqc_graph.cuh"

// values/gradients 是调用方的 GpuBuffer<double>。
// local_count 只统计本 rank 拥有的有效输入，不包括 ghost 和组外原子。
Leaf cv(values, gradients, local_count);
Graph graph(Graph::Symbols{{"cv", &cv}}, {}, lmp->world);
graph.parse_expr("linear(AVE(cv), 2, 1)");
graph.parse_expr("AVE(cv*cv)");
GraphExecution run(graph);  // NVRTC 只编译一次

auto dimensions = run.forward();
// 网格根据 dimensions 计算 dV/dDIM：
run.backward({dV_dDIM1, dV_dDIM2});
// gradients 此时保存每个有效输入的 dV/dcv_i，交给 CV 的原始反向计算。
```

执行顺序是 GPU 前向 → 下载规约输入 → Host 求局部和 → `MPI_Allreduce` 汇总全局和/计数 → Host 全局标量运算。AVE 使用全局总和除以全局计数，不对各 rank 的局部平均再平均。空 rank 必须参加 collective，全局无元素时所有 rank 一致报错。当前先由 Host 对下载数组求和，未优化成 GPU 预规约。

反向先将所有输出的种子注入 Host 图，累加共享节点贡献，再以 `g_AVE / N_global` 为 GPU 反向种子。每个 rank 接收同一组全局输出种子，不把这些种子再次求和。输入数据及节点参数从 forward 到 backward 期间必须保持不变。

构图通过 `Graph(..., {}, communicator)` 指定 MPI 通信器；手工创建节点时传入 `AVE(input, communicator)`。传入的通信器必须在执行器使用期间存活，各 rank 必须构建相同的拓扑/规约顺序。省略通信器（`MPI_COMM_NULL`）表示明确的单进程规约，不自动使用 `MPI_COMM_WORLD`。MPI 初始化/结束仍由调用方负责。

`Global` 叶子可以用 `Leaf({value}, {Domain::Global, 1})` 构造；其 `val/adj` 指向 Host 标量。只有全局标量的图不会分配节点显存或编译 kernel。`PerAtom` 的 `val/adj` 仍指向设备内存。

当前执行器要求最终输出均为全局标量，支持一段逐原子融合计算后接 Host 规约和标量运算。暂不支持 Global 再广播回 PerAtom、多组不同本地长度或通用向量分量。图拓扑在执行器生存期间固定；本地长度改变后调用 `resize_inputs()` 更新执行缓冲，同长度的外部 GpuBuffer 地址变化会在 forward 时刷新。LAMMPS 接入层负责原子迁移后的布局更新。

普通 `Stru_factor` 已提供原始值装配及 `apply_adjoint(leaf_adjoint)` 接口：前向把按 atom 索引存储的值装配到 Leaf，反向直接接收图产生的 GPU 梯度数组。AVE 和 CV 值上的开关函数由图处理，旧 CV 内部 AVE/COUNT 路径已移除。`evaluate_bias` 在 `EXECUTOR GRAPH` 下调度这条路径。

## CV 装配表（接入的第一步）

`zqc_cv_assembly.cuh` 提供 `CVAtomLayout` 和 `CVAssembly`。外部管理层根据各 CV 的 groupbit，遍历 `nlocal + nghost` 个原子，构造公共并集和每组的 `local_ids`（localtag）与 `leaf_slots`。相同 group 共享装配表。编号是本地数组下标，不是全局 atom tag；`localtag < nlocal` 为 owned，`localtag >= nlocal` 为 ghost。

每组的 `local_ids()` / `device_local_ids()` 都包含 owned 和组内 ghost，保持 owned 在前、ghost 在后。`group_count()` 只统计 owned，`total_count()` 包含 ghost。因此 CV 可用前 `group_count` 项作计算中心，同时保留表尾 ghost 编号。`leaf_count()` 只统计公共并集中的 owned；ghost 的 `leaf_slots` 为 -1，不装配到 Leaf，也不重复参与规约。这里只处理编号，不复制坐标；邻居表仍可访问组外原子。

```cpp
MetaD_zqc::CVAtomLayout atoms;
atoms.rebuild(nlocal, nghost, host_mask, {groupbit_a, groupbit_b});
auto a_map = atoms.group(groupbit_a);
auto b_map = atoms.group(groupbit_b);

GpuBuffer<double> a_values(atoms.leaf_count()), a_grad(atoms.leaf_count());
GpuBuffer<double> b_values(atoms.leaf_count()), b_grad(atoms.leaf_count());
// Stru_factor 新接口（调用方仍负责安排 base_calc 和 Leaf 生命周期）：
cv_a.bind_assembly(a_map);
cv_b.bind_assembly(b_map);
cv_a.base_calc();
cv_b.base_calc();
cv_a.assemble_values(a_values);
cv_b.assemble_values(b_values);
Leaf a(a_values, a_grad, atoms.leaf_count());
Leaf b(b_values, b_grad, atoms.leaf_count());
```

`Stru_factor` 绑定外部表后只将前 `group_count` 个 owned 编号交给中心 kernel，不再扫描 mask 筛组，也不把表尾 ghost 复制进 owned 中心缓冲区；不绑定时保留原接口。装配在 GPU 上完成，先清零公共目标区间，再填入该 CV 的 owned 值，不读取 ghost CV 值（这些值此时可以尚未计算或通信）。Leaf 读取连续数组。目标缓冲区由调用方/Leaf 持有，装配不会偷偷扩容或缓存其地址。装配与后续读取必须在相同 CUDA stream 上，或由调用方建立同步依赖；接口返回不代表 GPU 操作已完成。

图反向后调用 `cv_a.apply_adjoint(a_grad)`。普通 Stru_factor 在内部使用 `gather_adjoint` 提取 A 实际计算过的中心梯度，顺序与 `a_map->local_ids()` 的前 `group_count` 项一致，调用方无需再做一次 gather。ghost 表尾不产生额外的中心梯度；缺失位置是常量零，即使图对那个槽位产生非零导数，也不会传给 CV。

原子迁移、排序、ghost 列表或动态 group 改变后重建并重新绑定；从前向到反向期间不能更换这份布局。`shared_ptr` 保证旧表的生命周期，但不能自动判断外部原子顺序是否过期。公共表长度变化时通过 `GraphExecution::resize_inputs()` 更新缓冲并保留已编译的 kernel。组表构造的循环代价约为 `O(nlocal + nghost + union_total * distinct_groups)`，尚无性能测量；它不保证比已有共享 CV 环境的筛选更快。

此接口已接入 `evaluate_bias`。通用 `AVE(expr)` 对整个 Leaf 区间计数：装配后的 `AVE(a+b)` 可以明确表示对公共并集平均，但 `AVE(a)` 不会自动变成 A 自身组的平均。按规约组选择元素/计数是下一步，不能仅把分母改成 `group_count`（非线性运算后的组外值可能非零）。

`cv_assembly.cu` 验证稀疏且重叠的组、ghost 编号保留且不进入 owned 计数/规约、ghost-only 组/rank、有效零值、缺失填零、共享表、NVRTC 前后向、梯度回收、边界保护、重复装配和成员变更；测试使用已知逐原子数值，尚不构成完整 Stru_factor 力或 MPI 集成验证。

## Stru_factor 的数据对象与原始反向

`Stru_fact_env::data` 是长期存在的 `StruFactorData`，集中持有 `group_count` 和五个原有 GpuBuffer：`d_group_indices`、`d_group_numneigh`、`d_neigh_in_cutoff_r`、`d_calculated_numneigh`、`d_group_dminneigh`。数组名称、元素类型和原环境的共享关系保持一致，不复制拥有显存的对象；具体 CV 的值及梯度缓冲区仍由 CV 持有。

`set_context()` 统一绑定各缓冲区的名称、日志、错误处理和 LAMMPS 对象；`grow_centers()` / `grow_neighbors()` 调用已有 `GpuBuffer::grow_to()`，保留原有申请数量和容量复用策略。启动函数接收 `const StruFactorData&`，在内部提取当前指针参数包，字段也使用原名。提取过程只复制指针和数量，不分配显存、不上传数据、不增加同步；扩容后下一次启动会自动取得新指针。独立验证通过定长构造函数使用 GpuBuffer 的无 LAMMPS 分配入口。

单组的调用顺序如下（图和执行器的创建见上文）：

```cpp
cv.bind_assembly(atoms.group(groupbit));
cv.base_calc();
cv.assemble_values(values);       // Leaf 借用 values 和 gradients
const auto dimensions = run.forward();
run.backward({dV_dDIM1, dV_dDIM2});
cv.apply_adjoint(gradients);      // 输入为 Leaf 排列的完整 GPU 梯度数组
```

新反向针对每个 owned 中心的 `dV/dS_i`，将物理力 `-dV/dr` 和六分量 virial 累加到所有受影响原子，包括组外邻居和 ghost。CV 调用 LAMMPS reverse communication 汇总 ghost 贡献，再将 owned 贡献加到力和 virial 数组。所有 rank 必须按相同 CV 顺序调用，包括空中心组的 rank；从前向到反向期间保持坐标、参数和装配表不变。化学结构因子与 WEIGHT_CHEM 共用该生命周期，根据各自径向定义选择设备函数。

`strufactor_graph.cu` 对照独立 CPU 能量公式做坐标与应变有限差分，验证 raw 值、`AVE(s*s)` / `exp(AVE(s))` 的图梯度、组外邻居受力、ghost 贡献、空 rank、孤立中心、小 qr 极限、零梯度和数组边界。MPI 基准按全局编号用 Allreduce 汇总贡献，验证分区数学；实际 LAMMPS 的装配、通信及压力验证分别见 `../strufactor_lammps_graph/`、`../strufactor_chem/` 和 `../weighted_chem/`。

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

`zqc_graph_smoke` 只验证生成、编译和加载。数值验证由 `zqc_graph_sw_test` 完成，涵盖 13 组开关函数参数、共享节点/输入的梯度累加、Mean 反向种子、旧径向接口、异常参数和数组边界；还验证直接函数调用、独立文件注册、非 `MEAN` 派生的规约节点，以及解析失败后的图状态恢复。

已有性能对比脚本也使用 `build/graph`，可通过 `--build-dir` 改变位置：

```bash
LAMMPS_SOURCE_DIR=/path/to/lammps/src \
  python3 benchmarks/graph_kernel_20261013/run.py --build-dir build/graph
```

该性能脚本目前针对 sm_89；历史报告对应当时的源码与环境，不代表后续每次修改后的测量结果。`build*/`、CMake 缓存和基准程序的 `results*/` 为本地产物，保留源程序和精选报告供复现。

## 当前边界

- 融合路径支持等长逐原子标量输入；规约后支持 Host 全局标量。维度描述尚不意味着已支持通用向量和广播。
- 多输出 DAG 可以共享节点；每张可融合图生成一个前向 kernel 和一个反向 kernel。
- `AVE`/`MEAN` 通过 Host/MPI 执行，规约后可以继续全局标量运算；LAMMPS 的 `evaluate_bias` 通过 `EXECUTOR GRAPH` 调用该执行器。
- 实现通过 `zqc_graph_runtime` 静态库链接，可从多个 CUDA 翻译单元使用。设备数值函数仍保留 inline 并通过 xxd 嵌入，供 NVRTC 内联；当前每个执行器编译一次，跨执行器的编译缓存留待后续实现。
