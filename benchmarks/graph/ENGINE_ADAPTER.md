# 引擎适配：生产 CV、Graph 与 Leaf

DISTANCE、Stru_factor/化学结构因子、CSS、WEIGHT_CHEM、Q/Local Q 的配置与计算均使用 CVContext 和 EngineAdapter。Graph/Leaf 使用 ArrayView 与 SumReducer，LAMMPS 的 fix/compute 负责宿主入口。每个 Fix 持有一个 `EngineAdapter`，所有依赖它的 CV 在适配器之前销毁。

## 代码入口

- `include/utils/zqc_array_view.h`：`ArrayView<T>`，可传入 CUDA kernel 的非拥有数组视图。
- `include/utils/zqc_cuda_execution.cuh`：环境/Graph 的非阻塞流，以及将 CUDA event 转为 View 等待钩子的桥接。
- `include/utils/zqc_virial.h`：`VirialRepresentation` 和 `VirialView`，声明对称六分量或完整九分量，内联读取完整张量。
- `include/utils/zqc_engine.h`：语义邻居请求、请求管理、通信数据包和引擎操作接口，仅依赖标准库。
- `include/utils/zqc_distance.h`：DISTANCE 前向、反向状态，依赖 `EngineAdapter`。
- `include/core/zqc_lammps_engine.h`、`src/core/zqc_lammps_engine.cu`：LAMMPS 适配器，负责原生邻居表、共享上传或 Kokkos View 借用、规约和力/virial 回写。
- `include/CVsets/CV_Stru_factor.h`、`src/cvs/CVs_Strufactor.cu`：通用几何入口及原始 CV kernel；用 `ZQC_STRUFACTOR_KERNELS_ONLY` 可独立编译，包含路径无需提供 LAMMPS。

生产接入统一使用 `ArrayView`；`make_aos_view`、`make_soa_view`、`make_kokkos_view` 构造入口保留真实跨度、容量和就绪钩子。owned 范围由引擎或选择集提供。独立目录 `prototypes/extension_lab/external_arrays` 保留其自包含的 `BorrowedArray` ABI 实验；该原型不是生产接口，也不能与 `ArrayView` 相互 reinterpret_cast。

## 一轮执行

1. CV 构建时向 `engine.neighbors().request(...)` 声明需求，保存返回句柄。DISTANCE 使用稳定原子 ID，不需要邻居请求。
2. 主流程调用 `begin_evaluation()`。求值编号递增，包括同 timestep 的重复 `run 0`。
3. Graph 从输出拓扑识别实际使用的 Leaf，准备这些 Leaf 的装配表。`compute metad/atom` 在每次运行初始化时另外登记输出需求。主流程执行两者需求的并集，CV 按需取得候选邻居、坐标、类型和选择集，共享环境在同轮只计算一次。
4. 几何 kernel 在物理 cutoff 内筛选，原始 CV kernel 计算逐原子值，装配进 Leaf 后执行 Graph。
5. Graph 返回逐中心 adjoint。径向 CV 计算 owned+ghost 力和对称六分量 virial，Q/Local Q 直接生成完整九分量张量。
6. Graph 在逐 CV 反向前调用 `begin_force_virial_batch()`。各 CV 用 `commit_force_virial()` 提交力和 virial，LAMMPS 适配器把它们累加到同一份 owned+ghost 缓冲。
7. 所有 CV 反向结束后，Graph 调用 `finish_force_virial_batch()`：合计贡献只进行一次 reverse，再一次性将 owned 力写回宿主；virial 留给 Fix tally。反向异常调用 `cancel_force_virial_batch()`，丢弃尚未回写的贡献。

公共 LAMMPS reverse 包宽为九个 double，即各 CV 合计的三分量力加六分量 virial；所有 rank 保持相同提交顺序，空中心组也参与。Q 的半对计算先通过 `AtomCommunication` forward 补齐另一端的球谐、值、权重和及图梯度，归属端由稳定全局 AtomId 决定。Local Q 使用两层 halo 完成 I→J→K；两者最终都经由 `commit_force_virial` 回写。

### 力输出批次的存储与生命周期

LAMMPS 公共缓冲在 Host 上按 localid 的 AoS 布局保存 `[total,3]` 力和 `[total,6]` virial，
owned 行在前、ghost 行在后。容量跨步复用，批次开始时清零有效范围。
Host 输入按 stride 直接累加；CUDA 输入先等待生产事件，再通过复用的暂存区下载、累加。
提交返回后调用方可以复用源数组，适配器不持有其借用地址。

批次期间禁止开始下一轮 evaluation 或改变拓扑。取消只丢弃尚未回写的缓冲；
成功 finish 后调用方才可进入下一轮 MD。空批次不写回。
DISTANCE 的 owned 单原子贡献也加入相同缓冲；只有 owned 贡献的批次不触发 ghost reverse。
批次外的直接提交保留立即回传、立即写回行为。GPUMD 当前无需 MPI，采用默认批次钩子并沿用其设备提交路径。

当前合并的是通信和 owned 回写，CUDA 输入的下载仍按 CV 发生。
Kokkos 在 finish 后只上传一次最终 owned 力；GPU 常驻公共缓冲属于后续优化。
`SyncStats` 中的 `force_virial_submissions`、`force_reverse_calls` 和 `force_virial_flushes`
分别记录批量 CV 提交、ghost reverse 流程、输出回写次数。一次 reverse 流程可能包含多个交换阶段。

## 数组约定

`ArrayView` 描述逻辑二维数组 `[item, component]`。`item_count` 是有效项数；`storage_size` 是从 `data` 开始可访问的 T 元素数。两个 stride 都以 T 元素计量。

- AoS xyz：`item_stride=3`，`component_stride=1`。
- SoA xyz：`item_stride=1`，`component_stride=分配的原子槽位数`。预留槽位多于有效原子数时，应使用实际 stride。
- 力形状为 `[total,3]`。virial 使用 `VirialView`：`Symmetric6` 为 `[total,6]`，顺序 `xx yy zz xy xz yz`；`Full9` 为 `[total,9]`，顺序 `xx yy zz xy xz yz yx zx zy`。数值是含偏置种子的构型 virial，单位为能量。
- owned 原子在 `[0,owned)`，ghost 在 `[owned,total)`。`LocalAtomIndex=int` 只用于本地数组，稳定全局编号另用 `AtomId=uint64_t`。
- 视图借用内存。下一轮求值、底层扩容或重新排序后重新取得；前向、反向之间保持同一布局。
- CUDA 消费者在提交 kernel 前调用 `view.wait_on(consumer_stream)`；CPU 访问前调用 `view.wait_host()`。空钩子表示生产者已完成写入，不能用空钩子发布仍在其他流上更新的数据。

GPU 计算中的访问是 `coordinates(atom,component)` 的内联寻址。虚函数调用发生在 Host 准备/回写阶段。

### Graph 融合 kernel 的 ABI 边界

上述 stride 寻址适用于 CV 设备算法及装配接口。Graph 的 codegen 目前仍生成
`const double*` / `double*` 参数，以 `values[i]`、`gradients[i]` 访问连续标量数组；
生成的融合 kernel 不接收 ArrayView 或 stride 参数。

Leaf 在这个边界适配外部布局。连续 CUDA 标量数组直接借用设备地址；非连续 CUDA
输入在每次 forward 前由 gather kernel 拷贝一次到 Leaf 的连续暂存区。非连续梯度
输出在 backward 后由 scatter kernel 从连续暂存区写回一次。暂存区按容量复用，
但每轮 gather/scatter 仍有 kernel 启动和显存读写成本。Host 输入/梯度还涉及 CPU
打包或散写以及 H2D/D2H 传输。值和梯度分别判断是否需要暂存。

因此 View 接口接受非连续数组，并不代表融合 codegen 已经支持直接 stride 寻址。
是否为常见 stride 生成特化 kernel，应依据暂存成本在实际工作负载中的占比决定。
实现入口为 `Leaf::apply_bound_views()`、`stage_values_to_device()` 和
`scatter_gradients_from_device()`（`src/utils/zqc_graph_nodes.cu`）。

本机连续/stride 输入的实测成本，以及缓存前、改流前、当前版的完整 LAMMPS
对照，见 [性能记录](../runtime_performance/RESULTS.md)。当前 LAMMPS 桥接装配后的
Leaf 为连续 CUDA 数组，非连续暂存微基准单独验证这个接口边界。

## 流与就绪依赖

每个共享几何环境持有一条 `cudaStreamNonBlocking` 流。同环境的 Stru_factor/CSS 或不同阶数 Q 共用这条流，几何缓存、CUB scratch 和反向临时区按提交顺序复用。不同环境的 GPU 工作可以重叠。Graph 执行器另有融合流，普通 LAMMPS 适配器另有传输流。Host 仍按确定顺序发起 MPI 集合操作和 ghost 通信。

`base_calc()` 和 `assemble_values()` 表示计算已提交；装配完成的事件由 `CV::device_readiness()` 返回。Graph 将事件放入 Leaf 的值视图，融合流在加载前等待。Graph 的前向返回前完成 Host/MPI 规约，反向返回前完成外部梯度写回。Compute-only CV 在 Fix 返回 MD 前也等待完成，保证下一步坐标更新时已经结束旧坐标的借用。

外部 MD 在自己的流上更新数组后，先记录事件，再发布视图：

```cpp
// md_ready 由后端持有，以 cudaEventDisableTiming 创建。
// MD 的坐标写入已经提交到 md_stream。
execution_check(cudaEventRecord(md_ready, md_stream));
auto coordinates = position_buffer.view(total_atoms, 3);
coordinates.readiness = cuda_event_readiness(md_ready);

// CV 的 Host 启动入口建立依赖，然后提交几何 kernel。
coordinates.wait_on(cv_stream);
// geometry_kernel<<<blocks, threads, 0, cv_stream>>>(coordinates, ...);
```

坐标、类型、选择集标记、邻居 CSR、Leaf 值/梯度和力/virial 视图都沿用这个约定。`ArrayView` 的只读转换会保留钩子；自行切片时也应复制 `readiness`。Host 视图可提供自己的完成回调；CUDA event 的 Host 回调使用 `cudaEventSynchronize`。`wait_on(nullptr)` 表示在 CUDA 默认流排入等待，CPU 读取必须使用 `wait_host()`。

事件和数组由生产者持有；消费者提交等待前，生产者不能重新记录同一事件来表示后续轮次，也不能覆盖、释放或重排仍在使用的数组。`CudaExecutionStream::ready()` 复用一个事件，因此应在当前求值内立即交给消费者。普通 LAMMPS 路径的 Host 上传函数返回时传输已完成，返回的输入视图可用空钩子；GPUMD/Kokkos 后端借用异步生产的数组时，应发布实际生产流的事件。后端尚需实现自己的引擎接入。

同步点按用途保留：

| 边界 | 等待方式 |
|---|---|
| 同环境连续 kernel、清零、CUB scan | 同一流顺序执行 |
| MD → CV、CV 装配 → Graph | `cudaStreamWaitEvent` |
| Q/Local Q 筛选数量供 CPU 分配或启动 | 对应流的下载与等待 |
| Q forward 打包、力/virial reverse 打包 | 下载完成后再进入 MPI |
| Graph AVE/全局标量 | 融合流下载完成，再 Host/MPI 规约 |
| 诊断输出、公开 backward 返回、MD 复用数组 | 对应流/事件完成等待 |

`GpuBuffer::clear_async()` 只提交清零；`upload_from()` / `download_to()` 保留返回时 Host 内存可复用的约定，等待范围缩小到所传入的流。扩容仍使用 `cudaMalloc/cudaFree`，这些冷路径可能触发驱动同步；稳态复用容量。生产 CV 和融合执行器中已移除 `cudaDeviceSynchronize/cuCtxSynchronize`，独立节点的默认流执行只等待默认流。

`zqc_stream_contract` 故意阻塞模拟 MD 流，检查消费依赖以及无关流上的 buffer 操作；另外覆盖装配、NVRTC、非连续 Leaf 的前向和梯度回写。`zqc_production_cv_views` 在独立 MD 流延迟发布坐标，对全部生产 CV 验证 AoS/SoA、力和九分量 virial 的有限差分。运行：

```bash
ctest --test-dir build/layers-lammps -R 'zqc_stream_contract|zqc_production_cv_views' --output-on-failure -j1
```

## 六分量与九分量 virial

`VirialRepresentation` 说明张量有哪些独立分量，`ArrayView` 的 stride 说明这些分量如何存储。
`VirialView::validate()` 检查表示、列数和可达内存范围；形状九列但声明 `Symmetric6` 会被拒绝。
Host 验证后，kernel 可调用 `full_component(atom,component)` 按上述 Full9 顺序读取：对称六分量的后三项来自 `xy/xz/yz`，完整九分量则读取独立的 `yx/zx/zy`。

以下是借用九分量 SoA 数组的例子，`slots` 是实际分配槽位数，可能大于有效原子数：

```cpp
VirialView virials{
    {device_virial, total_atoms, 9, 1, slots, 9*slots, MemorySpace::Cuda},
    VirialRepresentation::Full9
};
engine.commit_force_virial(forces, virials);
```

全局 CV 的 `add_owned_force_virial(local,force,virial)` 使用恰好一行的 Host virial 视图，同样接受两种表示。
当前 DISTANCE、径向 Stru_factor、化学版、WEIGHT_CHEM 与 CSS 明确声明 `Symmetric6`，继续使用现有六分量缓冲。
这几类 CV 的每个有向 pair virial 在两端各保存一半，适配器保持原有分配；总张量等于逐原子贡献之和。
采用可能非对称的逐原子 virial 分配时，计算端保存 `Full9`；保证逐原子对称的分配可以使用 `Symmetric6`。旋转不变能量的完整总张量满足对称性，不要求每个原子的分配项对称，也不能由总量对称推回已丢失的逐原子转置项。测试先汇总九分量，再检查三个转置差值与九个形变方向的能量有限差分。

普通 LAMMPS 后端显式选取 `xx yy zz xy xz yz` 写入现有 tally 和六列 stress 输出。这是命名分量投影，既不转置，也不取上下三角平均；非对称九分量无法由该六列输出完整表达。
适配器保留调用方的九分量输入及原 stride，当前六分量连续下载仍沿用一次拷贝，reverse 仍传输 `3+6` 个 double。
GPUMD 后端应读取完整九分量并按其原生 stride 累加；实际 EnhancedSampling 后端尚待实现。不同引擎的逐原子分配方式也须在接入时核对，布局一致本身不证明物理分配一致。

## 邻居约定与缓存

请求身份包含中心选择集、邻居选择集、物理 cutoff、full/half 以及是否需要 ghost 中心。只对完全相同的请求去重。当前 LAMMPS 后端支持 owned 中心及含 ghost 中心的 full list，以及 owned 中心的 half list。Local Q 声明 ghost_centers 后，由适配器准备两倍物理 cutoff 加 skin 的 halo，并补齐可能受宿主 pair cutoff 限制的 ghost 邻居行，保留 type/group/molecule 排除语义。

Half 约定每个无序原子对在全体系只计算一次，返回视图声明 `kind=Half`、`pair_ownership=UniqueAcrossRanks`。消费者计算完整 pair 能量，向两端累加力，将 pair virial 各半分到两端，然后经 `commit_force_virial` 归还 ghost 贡献。此约定独立于宿主 pair 的 `newton on/off`。

- 相同选择集使用原生 `REQ_NEWTON_ON` half 表；`all_atoms` 与 LAMMPS 的 all 组按同一选择集处理。
- 不同选择集从 full 候选表筛选：只允许一个方向时保留该方向，两个方向都允许时按稳定 AtomId 选一端。同 ID 的周期自镜像再按镜像位移选一侧，因此需要启用 atom IDs。候选 CSR 仍随宿主重建/选择集变化缓存。
- `Half + ghost_centers` 会在请求验证时拒绝：ghost 中心行用于读取完整邻域，声明该需求应使用 Full。Full 返回 `DirectedByCenter`，生产 CV 的中心环境公式按有向项计算。

选择集编号由后端解释。LAMMPS 后端当前使用 group 编号；CV 的中心选择集和邻居选择集独立，后者默认全部原子。原生 neighbor 的特殊标志在适配器内去除，kernel 只收到 localid。

候选表为分段数组：`centers[k]` 是中心 localid，`offsets[k]..offsets[k+1]` 是该中心的候选段。几何筛选保留段起点，将有效邻居放在段首，并另存有效数量。Leaf 装配顺序与中心顺序在每轮进行核对。

LAMMPS 提供带周期镜像的 ghost 坐标，几何层直接计算 `r_neighbor-r_center`。需要 minimum image 的后端必须显式实现对应几何约定，当前入口会拒绝 `MinimumImageRequired`。

数据准备由活动 CV 的读取请求触发，同一选择集和邻居请求共享缓存。坐标每轮最多上传一次；类型每轮按需比较，内容变化才上传。稳定全局 ID 表及 DISTANCE 的 owned ID→localid 映射按拓扑版本重建。

LAMMPS 的 `init`、`setup_post_neighbor`、`post_neighbor` 和 `min_post_neighbor` 回调推进拓扑版本，覆盖排序、迁移、ghost 重建和相同 timestep 的重复 `run 0`。静态组按此版本缓存；动态组每轮检查当前 mask，成员变化推进选择集版本。宿主在上述生命周期之外直接改动 localid/ID/静态组时，须显式调用 `invalidate_topology()`。原子数或指针相同不代表版本有效。

`SelectionView::generation` 为非零时保证覆盖成员和 localid 身份变化，Graph 据此复用装配表；提供者不能保证版本时可保留 0，Graph 改为逐项比较。只有装配布局变化时才复制成员列表。dense 0/1 flags 仅在消费者请求时生成，只有 CUDA 消费者需要且内容变化时才上传。

普通候选表使用 LAMMPS perpetual full list（物理 cutoff 加 skin），缓存键包含原生拓扑版本及中心/邻居选择集版本。`topology_generation` 只在依赖版本或 CSR 内容变化时推进。Local Q 补齐的 ghost 候选行按同一个宿主拓扑版本缓存；动态选择集变化时从原始候选行重新筛选 CSR。物理 cutoff 筛选、有效 pair 压缩和 I/J/K 标记每轮按当前坐标计算。原生邻居的排除规则遵循 LAMMPS 邻居重建时序。

各类标记的用途：

- LAMMPS `atom->mask`：Host 组筛选；`all` 的成员资格直接由布局确定。
- `NEIGHMASK`：解码原生邻居编号的位掩码；转换后的设备 CSR 保存纯 localid。
- Q 的 selection flags：半对归属和两端反向贡献所需的 0/1 成员资格。
- Local Q 的 `d_LQ_mask`：本轮 I/J/K 身份和计数，随几何重新计算；Local Q 已移除未使用的 Q selection-flags 参数。

原生邻居请求仍在 CV 构建阶段登记，供 LAMMPS 初始化邻居设施；未被图或 compute 引用的 CV 不执行数据准备、GPU 上传或 `base_calc`。回写的 Host 缓冲在 CV 间复用。

Stru_factor 与 Q 环境池按 EngineAdapter 所有者区分。Q 环境引用计数随 CV 析构释放，Stru_factor 池由宿主在 CV 销毁后 clear_pool(engine)；适配器的寿命覆盖所有 CV。

## 验证入口

```bash
cmake --build build/graph -j2
ctest --test-dir build/graph -R 'zqc_engine_' --output-on-failure

# Host 接口可在没有 CUDA/LAMMPS 的环境单独编译。
g++ -std=c++17 -Iinclude/utils benchmarks/graph/engine_contract.cpp -o engine_contract
./engine_contract

cmake --build build/graph-lammps -j2
python benchmarks/finite_difference/run.py --np 1 2 4 \
  --cases distance distance_graph distance_mixed structure chemical weighted css_ternary mixed_groups nonlinear_graph \
  --output build/engine-adapter-validation

# 用上一步生成的 fixture 检查环境池和带空槽的 group 编号，输出目录应尚不存在。
mpirun -np 2 python benchmarks/finite_difference/engine_lifecycle.py \
  --fixture build/engine-adapter-validation/structure_np2 \
  --output build/engine-adapter-lifecycle-2
```

`engine_contract` 检查 AoS/带预留容量的 SoA、六/九分量表示和非法形状、请求身份、64 位全局 ID、周期距离的坐标有限差分、半 pair virial、本地重排和过期 backward。
`engine_geometry` 在 GPU 上检查六转九、非对称九分量传递及输入/输出 AoS/SoA 的八种组合，并用同一生产几何入口比较值、力、virial，覆盖组外邻居、ghost、self 与 cutoff 筛选及周期约定拒绝。
完整 LAMMPS 测试检查同 timestep 的重复计算、1/2/4 rank、混合组及非线性 Graph 的力和应变 virial。

## 后续接入范围

GPUMD EnhancedSampling 和 Kokkos CUDA 的真实适配器尚未实现；当前 SoA 验证证明的是数组/几何接口，不代表已在这两个引擎上运行。它们还需实现原生邻居、周期位移、数据就绪、力/virial 布局与所有权同步。

生产 CV 的读取函数通过 CVContext 查询类型标签、选择集和径向开关。异常和 Logger 事件回到宿主，Fix 统一转发给 LAMMPS。网格/采样层仍含宿主依赖，属于下一阶段拆分范围。本轮未测量性能增益；候选表跨轮缓存仍按上文约定处理。

## 首批适配历史验证（2026-10-21）

- Release LAMMPS 插件、独立 CUDA 目标编译通过。
- CTest：25/25，通过数组 ABI、诊断、GpuBuffer、NVRTC、AVE、CV/Graph 有限差分和新增引擎边界测试。记录：`build/engine-adapter-final-ctest.log`。
- 完整 LAMMPS：9 个场景 × 1/2/4 rank，共 27/27。步长 `1e-4,3e-5,1e-5,3e-6`，沿用有限差分夹具的力、CV 导数和应变 virial 容差。原始记录：`build/engine-adapter-final-fd/`。
- 生命周期：2 rank，带空槽的 group 编号，连续删除/重建同名 Fix 三次，CV 值一致。记录：`build/engine-adapter-lifecycle-2/log.lammps`。
- 编译环境：CUDA 12.6、LAMMPS stable_29Aug2024；本轮测试使用现有 RTX 4080 SUPER。

被测插件 SHA-256：`705addbd55cc0499210d26d113ae7ee82c192717df55154e90f13530fb1f0811`。
这轮验证的是数值正确性、通信和生命周期；没有进行性能基准测试。

## virial 与 compute 输出补测

在相同插件上补跑 39/39 个输出场景（均包含 1/2/4 rank）：

- WEIGHT_CHEM：共享化学结构因子环境、不同中心组、CV cutoff 大于 pair cutoff，9 个场景。`chem_pair_r`/`stru_f` 对照独立 CPU 公式；逐原子 stress 求和对照全局 pressure。
- 化学 Stru_factor：带符号权重、小中心组、逐原子非线性图，9 个场景；`stru_f` 对照 CPU 参考。
- CSS：原始二元、三元、多壳层、小中心组，12 个场景；评分及各通道配位数对照 CPU 参考。
- 新增 `benchmarks/finite_difference/compute_outputs.py`：上述三类生产夹具各 1/2/4 rank，共 9 个场景。比较单列 `vector_atom` 与多列 `array_atom` 的同一属性、compute group 外零填充、连续两帧输出、六分量 `stress/atom` 以及 MPI 分区一致性。

观测到的最大误差：逐原子 CV 对照 CPU 为 `1.16e-14`；单列/多列与 group 掩码检查均为零；`-sum(stress/atom)/volume` 对照全局 pressure 为 `5.69e-13`，小于使用的 `1e-9` 容差。新增检查中，跨 rank 逐原子 dump 最大差异为 `8.70e-10`，小于 `2e-8` 容差。

这与之前六个应变方向的有限差分互补：有限差分检查总 virial 的导数关系，输出补测检查逐原子分配、ghost reverse 和 LAMMPS 的 stress 汇总链路。当前没有逐原子局部应变定义下的独立有限差分验证。

```bash
python benchmarks/finite_difference/compute_outputs.py \
  --plugin build/graph-lammps/fix_crystallize_plugin.so \
  --np 1 2 4 --output build/compute-output-validation
```

`--plugin` 应明确指向本次构建；仓库的 `build/` 下还可能存在旧插件。
原始结果分别位于 `build/engine-adapter-output-weighted/`、`build/engine-adapter-output-chemical/`、`build/engine-adapter-output-css/` 和 `build/engine-adapter-compute-outputs-current/`。

输出职责：适配器负责力/virial 回写，Fix 负责 LAMMPS virial 记账；`compute metad/atom` 仍是 LAMMPS 输出桥，通过 `Fix::extract -> CV::peratom_values` 取得只读视图。Fix 持有各请求的连续 Host 列，按视图 stride 读取或从 CUDA 下载。DISTANCE 为全局标量，使用 `f_m[...]` 输出。Q 的诊断梯度在请求时额外归并到 owner，同一轮 x/y/z 共用一次 reverse；正常力回写仍只需原来的通信。

## 六/九分量接口验证（2026-10-21）

- 独立 CUDA 目标与 Release LAMMPS 插件编译通过；CTest 25/25，包含扩展后的 Host/CUDA virial 测试。
- `engine_virial.py`：8 种表示/内存/布局组合 × 1/2/4 rank，24/24。批量 ghost reverse 和单个 owned 提交对照按 ID 规约的参考值，最大误差为零。记录：`build/virial-representation-lammps/`。
- `run.py`：`distance_mixed`、`structure`、`weighted`、`css_ternary` × 1/2 rank，8/8。沿用四个差分步长及力/virial 容差。记录：`build/virial-representation-fd/`。
- `compute_outputs.py`：WEIGHT_CHEM、化学 Stru_factor、CSS × 1/4 rank，6/6。单列/多列及 group 掩码误差为零，stress 求和与 pressure 最大差异为 `8.53e-14`。记录：`build/virial-representation-compute/`。

本轮插件 SHA-256：`d6f0da4d533d431ba5c7fe54f04d5ec4de5fdc8542877df5d0418f9d4c95cc42`。
上述九分量测试覆盖通用视图与 LAMMPS 输出边界；没有运行真实 GPUMD，也未做本轮性能对比。


## 独立生产 CV 验证入口

`production_cv_views.cu` 使用轻量测试引擎调用实际 CV 工厂，不包含 LAMMPS/MPI。
选择集编号为 77，Host 选择集/中心编号及类型标记带 stride，坐标在 AoS 与带预留容量的 SoA 间切换。
输入 Leaf 值和梯度也使用 stride=2。11 种配置覆盖全部生产 CV 及 Q/Local Q 的 3/4/6 阶；
比较类型编号从 0/1 开始的结果，并对全部坐标和九个应变分量作有限差分。

```bash
cmake --preset core -DZQC_BUILD_TESTS=ON -DZQC_TEST_MPI=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build --preset core --target zqc_production_cv_views -j2
ctest --test-dir build/core -R '^zqc_production_cv_views$' --output-on-failure
```

`q_compute_outputs.py` 可复用 `finite_difference/run.py` 的单 Q/Local Q 夹具，
检查 compute 的三列梯度等于负的偏置力、跨 rank owner 归并及同一步重复求值：

```bash
mpirun -np 2 python benchmarks/finite_difference/q_compute_outputs.py \
  --fixture build/validation/q3_np1 --output build/q-output-validation
```

输出目录应尚不存在；夹具父目录须保留同次构建生成的 `bias_probe.so`。

## 完成生产迁移后的验证（2026-10-22）

- 完整 LAMMPS 插件编译通过；CTest 26/26，包含新增生产 CV 的独立后端测试。
- 关闭 LAMMPS 和 MPI 的全新 `build/views-core` 配置可构建、运行 `zqc_production_cv_views`。
  其动态依赖只有 CUDA Runtime 和系统 C/C++ 库。14 个生产 CV/Graph/装配/桥接翻译单元的
  编译依赖没有 LAMMPS/MPI 原生头文件，审计记录为 `build/layers-lammps/view-dependency-audit.json`。
- 全量有限差分：23 场景 × 1/2 rank，46/46；六项线性叠加核对也通过。
  四个步长为 `1e-4,3e-5,1e-5,3e-6`，记录为 `build/view-migration-fd/`。
- Q 专项 9/9、Local Q 专项 18/18，覆盖 1/2/4 rank、动态组、原子迁移、空中心 rank、
  孤立原子、径向开关和邻居排除；另外 9 项旧接口拒绝检查通过。
- compute 输出 9/9；Q3/Local Q6 的三列梯度检查 6/6，与负偏置力误差为零。
- Stru_factor、Q3、Local Q6 在 2 rank 下分别连续删除/重建 Fix 三次，值保持一致。
- virial 适配边界 24/24：六/九分量 × Host/CUDA × AoS/SoA × 1/2/4 rank。

最终回归清单和日志入口为 `build/view-migration-checks.json`。完整差分测试后的最后修正为
空原子诊断视图、未知属性报错、宿主异常重复包装；最终回归包含上述输出/生命周期/专项检查。
全量差分被测插件 SHA-256 为 `4f8a474b73c251fc2555f9b0acead66a615b073b27068437b0403920db031de5`；
最终插件为 `27b5786ee5b37c68c20f6b84da5af13a612d6af087994c64bebaefce05d69d94`。

GPUMD EnhancedSampling 和真实 Kokkos 数组的后端接入、网格/采样层解耦仍是后续工作。

## 按需准备与缓存验证（2026-10-22）

最终插件 SHA-256：`b7bb26b64422a12b7830825aaf3bdb72279b79c80ea41f8df7bdaa1a22fde418`。

- `engine_cache.py`：9 个场景 × 1/2/4 rank，27/27。覆盖未引用的 SYMBOL、共享选择集、
  compute 独立需求及移除、Local Q 无冗余 flags、动态组/Q、小位移、排序/迁移和纯 DISTANCE。
  探针另检查相同长度下的成员交换、类型修改和 ID 重排。记录：`build/metadata-cache-release/`。
- 全量 CV/Graph 有限差分：46/46，六项线性叠加核对通过。记录：`build/metadata-final-fd/`。
- 全新 `build/metadata-standalone` 关闭 LAMMPS、MPI 和生产 CV，未传入任何 fmt 路径，
  Graph 编译和 NVRTC 前向/反向 smoke 通过。头文件依赖审计确认源码生成器只使用仓库内 fmt。
- 14 个 fmt 文件（13 个头文件加许可证）与上游 10.2.0 归档逐字节一致。

缓存计数示例：共享静态组的两个 CV 连续求值六次，组扫描和候选表上传均为一次；
坐标上传六次。Local Q 的 selection-flags 上传为零，ghost 几何修补执行六次，内容相同的
候选表只上传一次。动态组成员交替时每次更新；纯 DISTANCE 不准备整份 GPU 坐标或邻居表。
这些计数验证重复工作被省去，尚未测量真实体系的时间收益。

完整回归另包含 CTest 26/26、Q 9/9、Local Q 18/18、compute 9/9、virial 24/24，
见 `build/metadata-regression.json`；其后仅整理了注释、缩进和包含顺序。
最终插件的独立 NVRTC、缓存和全量有限差分记录汇总在 `build/metadata-final-checks.json`。

## 流与 View 等待改造验证（2026-10-22）

最终插件 SHA-256：`cfc1b871d9cff06d0e5b8edfcd33910d2f5b1302a9f078dd54fa8567598f7138`。
以下测试均使用同一个最终二进制；回归脚本在每阶段结束后核对插件 hash。

- CTest 27/27，包含 `zqc_stream_contract`、异步坐标生产者下的生产 CV 数值测试、
  独立 Graph/NVRTC 和 1/2/4 rank 的 Graph 有限差分。
- 完整 LAMMPS CV/Graph 有限差分 46/46，六项线性组合检查通过；覆盖四个差分步长。
- Q 专项 9/9，Local Q 专项 18/18；包含分组、空 rank、孤立原子、径向开关和邻居排除。
- compute 输出 9/9，virial 适配 24/24，按需准备与缓存 27/27。
  缓存检查继续覆盖共享选择集、compute-only CV、动态组、排序与原子迁移。

完整清单及日志入口：`build/streams-final-regression.json`。
有限差分数据位于 `build/streams-final-fd/`，其余专项目录使用相同的 `streams-final-` 前缀。
流隔离测试验证了等待范围与数据依赖；这些结果不包含真实体系的性能加速比。

## 重复装配的跨流依赖修复（2026-10-22）

上述 CTest 是单次通过记录。后续重复运行发现 `zqc_cv_assembly` 在
`refreshed graph output` 偶发失败；本次修复前独立运行 60 次，失败 26 次。
该检查复用原有 Leaf 和显存：`CVAssembly` 的 GpuBuffer 重载在默认流异步写入，
`Leaf::refresh_buffers()` 随后取得的 View 原先没有就绪事件，Graph 私有非阻塞流
因此可能提前读取。这个复现路径没有调用 Leaf::bind，也没有经过生产 CV 的环境流。

GpuBuffer 现在持有可复用的写完成事件，`view()` 携带该依赖；装配和梯度回收的
GpuBuffer 重载自动记录输出事件，Leaf 每轮刷新即可继承。`clear_async()`、
`scan_to()` 也发布写完成事件，下载及缓冲区算子在访问流中等待已有写入。
通过 ptr/ArrayView 自行提交的生产者使用 `wait_on(stream)` 和 `record_write(stream)`；
生产 CV 的 View 接口继续由 CVGraphExecution 在装配后附上 `device_readiness()`。
数组及事件由所有者持有，覆写或扩容前须结束上一轮消费者的访问。

`stream_contract.cu` 新增受控延迟测试：预热后用 Host gate 暂停生产流，装配提交
之后才释放 gate。它在修复前立即报 `GpuBuffer reassembly lost readiness before Graph forward`。
修复后的覆盖包括默认流/非阻塞流交替、同一分配重复装配、显式重绑、反向梯度回收，
以及异步写入 → CUB 扫描 → Host 下载的三条流依赖链。

- 原 `zqc_cv_assembly` 连续 300 次通过，记录在 `build/assembly-race-after.json`。
  修复前记录在 `build/assembly-race-before.json`。
- 将测试辅助上传/下载统一到 GpuBuffer 接口后，再连续通过 60 次，记录为
  `build/assembly-race-final.json`。上传返回后可立即重绑 Leaf；装配继续异步提交。
- 完整插件编译通过，CTest 27/27；扩展后的 `zqc_stream_contract` 连续 20 次通过。
- LAMMPS CV/Graph 有限差分 46/46，六项线性组合核对通过，覆盖 1/2 rank
  和四个差分步长。记录在 `build/assembly-race-fd/`。
- 本轮回归入口为 `build/assembly-race-checks.json`，插件 SHA-256 为
  `422782f4c882451820e3b40c831fafc1e52cef74e747ce4956681e759548c66f`。

安装者可以在 Graph 测试预设中重复检查这两条路径：

```bash
ctest --preset graph-tests -R '^zqc_(cv_assembly|stream_contract)$' \
  --repeat until-fail:100 --output-on-failure -j1
```

## Local Q ghost 候选缓存（2026-10-23）

2026-10-28 补充：本节的 `cutneighghostsq` 补行适用于普通 Host 邻居表。
CSS/StruFactor、LocalQL 现在可用 `NeighborRequest::host_csr=false` 放弃 Host
offsets/neighbors 镜像；`host_centers` 仍须与设备 centers 同序。LAMMPS Kokkos
后端为这些 Full 请求直接消费设备邻居表，在设备上生成 CSR；其 ghost 行使用
同一自定义 cutoff，因此不执行下面的 Host 空间搜索。未支持设备路径的后端仍可
同时提供两份 CSR。请求去重包含 `host_csr`，避免让需要 Host 数组的消费者取得空镜像。

LAMMPS stable_29Aug2024 的自定义 cutoff 会扩大请求的 owned 行；ghost 行的
`cutneighghostsq` 仍由 pair 的设置决定。因此 CV 半径大于 pair ghost 半径时，
适配器按需补齐 Local Q 的二跳 ghost 邻域。此前这项空间搜索每轮执行；
现在原始 ghost 候选 CSR 与宿主邻居表共用 `topology_epoch`。

缓存分两层：

- `repaired_ghost[localid]` 标记需要补齐的行；`ghost_offsets[localid:localid+2]`
  索引 `ghost_neighbors`。这些数组包含全部 owned 中心的候选 ghost J 所需的行，
  在空间搜索阶段保留所有选择集成员。动态组换成员后仍能复用几何候选。
- 输出 CSR 按请求的中心组/邻居组筛选。其缓存键包括宿主拓扑版本和两个成员版本。
  组变化会更新输出 CSR；原始 ghost 候选在宿主拓扑失效时重建。

宿主重建步首次使用时，空间搜索半径为 `R + skin`；如果请求在本次宿主重建之后
才首次求值，使用 `R + 2*skin`。后者覆盖“宿主参考构型 → 首次求值构型 →
后续构型”的相对位移余量，仍在下一次宿主重建时失效。
例如 skin=0.5 Å，一对原子在首次求值时相距 `R+0.75 Å`，之后可能靠近到
`R-0.05 Å`，同时两原子相对宿主参考位置的位移都小于 skin/2。
这种延迟需求不能直接以首次求值坐标构造 `R+skin` 后一直复用。

通信 halo 继续使用 `2*R + skin`，覆盖从 owned I 出发的实际 I→J→K 二跳邻域。
宿主的排序、迁移、ghost 重建及 run 初始化使 topology 失效；run 间修改 skin
会同时更新通信半径。每轮物理 cutoff、距离、开关权重及 I/J/K 身份仍由当前坐标计算。
排除规则跟随宿主邻居表重建。NPT 和倾斜盒沿用宿主的盒子位移/skin 判定；
使用者应保持可保证邻居完整性的 `neigh_modify` 设置。

`SyncStats::ghost_row_rebuilds` 统计实际空间搜索；`neighbor_rebuilds` 统计输出 CSR
筛选，动态组会增加后者。它们与设备上传次数分开，便于判断开销来源。

新增 `ghost_cache.py` 在 1/2/4 rank 下通过 12 个场景：邻居进入 cutoff、延迟首次
使用、动态组、NPT、倾斜盒、排序迁移与 skin 修改。逐步枚举真实 pair 检查 CSR，
并比较缓存与冷重建的 CV、偏置能、力和 virial。固定盒移动测试中宿主表 40 步
未重建，已初始化请求的 ghost 空间搜索次数也保持不变。
现有缓存测试 27/27，Local Q3/Q4/Q6、混合组及非线性 Graph 有限差分 15/15 通过；
共用转换路径的 Half 表回归 486/486 通过。
原始记录在 `build/ghost-neighbor-cache-20261023/`；
性能结果见 [ghost 缓存对比](../runtime_performance/GHOST_CACHE_RESULTS.md)。

## Bias session

偏置执行边界由 `zqc_bias.h` 的 `BiasSession` 和 `BiasHost` 定义。核心拥有轴描述、HILLS 历史、网格缓存和文件句柄；宿主拥有 Graph、原子数据和 communicator。所有 CV 标量和偏势参数使用宿主统一的长度/能量单位。

```cpp
BiasSettings settings;
settings.backend = BiasBackend::Auto;
settings.hills.axes = {{0.2, 0.0}, {0.3, 0.0}};
settings.grid_axes = {{0, 5, 0.2, 128, 0}, {0, 5, 0.3, 144, 0}};
settings.hills.thermal_energy = boltzmann_constant * temperature;
settings.history_path = "HILLS";
settings.history_workers = 4;

BiasHost host;
host.root = is_root_rank;
host.broadcast = broadcast_bias_packet; // (double*, size_t)，同一 communicator 内同步调用
host.log = write_engine_log;
BiasSession bias(settings, host);
bias.initialize(current_step);

// Graph 前向完成全局规约后，提供 D 个标量。
// setup/compute 只调用 evaluate；积分回调才调用 advance。
if (is_integrating) bias.advance(current_step, cv_values);
double energy = bias.evaluate(cv_values, cv_derivatives);
graph.backward(cv_derivatives);
```

这段顺序对应当前 LAMMPS 适配器的沉积后求值约定。若引擎要求力计算后沉积，可在自己的生命周期中先 evaluate、后 advance；checkpoint 和输出也必须采用对应阶段。initialize 可重复调用；同一步 advance 去重。

单进程 GPUMD 的 root 保持 true，可省略 broadcast。日志与异常由 EnhancedSampling 适配层转交 GPUMD；核心不保存 LAMMPS 的 lmp/error 指针。具体 GPUMD 回调尚待其适配器接入。force 与 6/9 分量 virial 的装配属于已有 engine view 层，BiasSession 只产出偏势与 CV 梯度。

并发边界：一个 session 由宿主顺序调用；HILLS 加载使用内部工作线程。调用 evaluate/advance 前，宿主需保证 Graph 的全局值已完成；此 CPU 接口不接收尚未就绪的 device 指针。

## LAMMPS CUDA/Kokkos

可选桥接位于 include/core/zqc_lammps_kokkos.h 和 src/core/zqc_lammps_kokkos.cu。
该边界维护 DualView 的 Host/Device 状态：坐标和类型保留宿主 Device stride，
group/tag 等筛选元数据同步到 Host；输入 View 通过事件等待宿主生产者。
MPI reverse 后将 owned 力增量加到 Device 力数组，完成后标记 F_MASK；
virial 在 Host 交给 Fix tally。邻居候选仍是 Host 请求与缓存 CSR。
普通 LAMMPS 使用 GpuBuffer 暂存；其坐标 View 也携带上传完成事件。

这层借用不会改变 Graph kernel 的连续参数 ABI。构建、已测范围和旧版 trim
回补见 [Kokkos 接入指南](../kokkos/README.md)。
