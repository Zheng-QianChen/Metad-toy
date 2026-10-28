# 四项扩展的独立原型

这些原型用于先验证接口和算法，再逐步合入 Metad-toy。全部内容位于 `prototypes/extension_lab/`；构建目录、模型文件和完整运行结果默认位于 `build/extension_lab/`。原型目录保留为独立参考。2026-10-23 已将对角宽度 hill 列表、周期核及 PBMetaD 迁入纯 C++ 库并接入 LAMMPS；使用与验证见 [高维沉积](../../benchmarks/highdim/README.md)。生产 timestep 使用 C++ 实现。

| 目录 | 可运行的内容 | 关键接口 |
|---|---|---|
| `external_arrays/` | GPUMD / LAMMPS Kokkos 原始 GPU 数组借用、NVRTC kernel 和力回写 | `BorrowedArray`、`borrow_soa`、`borrow_kokkos` |
| `sampling/` | 谐和伞采样、WTMetaD、WHAM 与 MPI 窗口交换 | `evaluate(s) -> (V, dV/ds)`、`update(step,s)` |
| `ml_models/` | SavedModel、冻结 `.pb`、ONNX，VJP、坐标导数和离线 XYZ 分析 | `ModelNode.forward(features)`、`vjp(features,seeds)` |
| `highdim/` | 任意维 hill 列表、周期核、完整协方差、PBMetaD、二维联合重加权 | `HillBias`、`ParallelBiasMetaD`、`frozen_bias_fes` |

`common/checks.py` 仅提供测试辅助函数。采样、ML 和高维算法采用 Python/NumPy，便于检查数学和接口；GPU 数组原型采用 C++17/CUDA。当前没有把这三套 Python 参考实现放进 C++ timestep。

## 安装和运行

建议使用独立 Python 3.10–3.12 环境：

```bash
python -m venv build/extension_lab/venv
source build/extension_lab/venv/bin/activate
python -m pip install -r prototypes/extension_lab/requirements-ml.txt
```

只验证采样和高维算法，可以安装 `requirements-numerical.txt`。`mpi4py` 必须与所选 `mpirun` 使用兼容的 MPI 库。TensorFlow 的 GPU 支持取决于该环境的 CUDA/cuDNN 依赖；CPU 也可验证文件和梯度契约。

GPU 数组原型需要 NVIDIA GPU、CUDA Toolkit（nvcc、NVRTC、Driver）、CMake ≥ 3.22、MPI 开发包，以及两份上游源码。已验证版本：

- LAMMPS `stable_29Aug2024`，使用其中的 `lib/kokkos` 和 `src/KOKKOS/kokkos_type.h`。
- GPUMD `a0e55bc87390b93d009da37c798616982573bab2`。读取其 `GPU_Vector` 头文件，构建时保留上游许可证；不需要先编译完整 GPUMD。

例如准备 GPUMD 依赖：

```bash
git clone https://github.com/brucefan1983/GPUMD.git build/extension_lab/GPUMD
git -C build/extension_lab/GPUMD checkout a0e55bc87390b93d009da37c798616982573bab2
```

完整验证入口：

```bash
python prototypes/extension_lab/run_all.py \
  --lammps-source /path/to/lammps-stable_29Aug2024 \
  --gpumd-source "$PWD/build/extension_lab/GPUMD" \
  --cuda-arch ADA89 --sanitize
```

`ADA89` 对应本次的 RTX 4080 SUPER；其他 GPU 可指定其 Kokkos 架构，或省略该参数让 Kokkos 检测。`--cmake`、`--mpi-cxx`、`--mpiexec` 可选择非 PATH 中的工具。OpenMPI 插槽不足时使用 `--mpi-extra=--oversubscribe`。

单项验证和离线分析：

```bash
python prototypes/extension_lab/run_all.py --modules sampling highdim
python prototypes/extension_lab/run_all.py --modules ml
python prototypes/extension_lab/ml_models/trajectory.py \
  --model build/extension_lab/results/models/onnx \
  --xyz build/extension_lab/results/demo.xyz \
  --csv build/extension_lab/results/analysis.csv
```

完整运行写入 `results/suite.json` 和逐项 JSON，命令与输出保存在 `logs/`。命令失败时入口返回非零，汇总文件标记 `failed`。入口同时比较运行前后的 Git 跟踪文件摘要。生成的模型、采样直方图和轨迹都可在 `build/extension_lab/results/` 查看。`results/` 目录中的报告是本次已验证的结果快照；重新运行不会覆盖这些快照。

## 1. 外部 GPU 数组

`BorrowedArray` 保存设备指针、原子 stride、分量 stride、总容量和 owned 数量，stride 单位均为 `double` 元素。它借用容器的存储，生命周期由引擎维护。GPUMD 使用 `x[N], y[N], z[N]`；Kokkos 从真实 View 查询布局。

测试直接实例化 `GPU_Vector<double>` 和 LAMMPS 的 `DAT::tdual_x_array` / `DAT::tdual_f_array`，分别覆盖默认布局与 `LMP_KOKKOS_NO_LEGACY` 的 LayoutLeft 坐标。运行 NVRTC 编译出的二次势 kernel，将力累加到引擎数组；验证 owned/ghost 边界、空任务、指针相同、重新分配后重新绑定、跨流 event、DualView 回写标记和中心差分。两种布局都有 Compute Sanitizer 检查。

以后接入时，引擎先提供有效的数组视图和数据完成事件，Graph 消费后返回完成事件。引擎在消费完成前保持数组存活；扩容、迁移和排序后重新取得视图。Kokkos 的 `sync_device` / `modify_device` 由适配层负责。

当前验证覆盖真实容器与独立 NVRTC kernel。还没有加载完整 GPUMD timestep 或 LAMMPS `fix/kk`；邻居表、跨 rank ghost 导数规约和设备选择仍由实际引擎适配层接入。当前数据类型固定为 FP64、单 GPU。

## 2. 采样与多副本

`evaluate()` 是纯求值，返回偏置能量和 CV 的 adjoint；`update()` 在步数满足条件时沉积 hill。WTMetaD 保存 hill、参数和最后一步，支持恢复后继续沉积。窗口交换单独处理 replica 通信，因此采样方法本身不持有 MPI communicator。

伞采样验证在已知一维双势阱中运行 18 个 Metropolis 窗口。WHAM 重建只读取各窗口直方图和已知 umbrella 偏置，最后才与参考势比较；同时报告相邻窗口重叠。

MPI 验证创建每个副本的 communicator 和 leader communicator，交替尝试相邻窗口交换。支持一个副本占多个 rank，并核对各 rank 状态一致、walker 标签没有丢失或重复。该原型采用同温度、相同基础势：

`log(P_new/P_old) = -[V_a(x_b)+V_b(x_a)-V_a(x_a)-V_b(x_b)] / kBT`。

当前交换的是模型体系的标量坐标。完整 MD 还要规定坐标、速度、thermostat、约束和随机数状态的交换/恢复规则；异温度交换及 shared-bias multiple walkers 可在这个边界上另加。副本交换测试验证通信和接受规则，未声称它的短轨迹已达到热力学收敛。

## 3. 模型文件与轨迹

加载节点读取 `manifest.json`，其中明确输入维度、输出维度、FP64、特征顺序、单位和模型文件。接口约定：

- `features`：`[batch, input_dimension]`。
- `outputs`：`[batch, output_dimension]`。
- `seeds`：上游给出的 `dV/doutputs`，形状同 outputs。
- `vjp(features,seeds)`：返回 `dV/dfeatures`，形状同 features。

导出器生成固定权重的 3→5→2 tanh 神经网络。SavedModel 通过 GradientTape 导出 VJP，冻结 `.pb` 保存 forward / VJP 两张图及输入输出 tensor 名，ONNX 保存 `forward.onnx` 和 `vjp.onnx`。加载器从文件执行，节点没有把 demo 的网络公式写死。

普通 ONNX 推理文件或只有 forward 的 `.pb`，不能据此承诺自动提供原子力；本原型要求同时提供明确的 VJP 契约。模型文件可以描述不同的网络，但还需匹配特征生成器、单位、动态 batch 和算子支持情况。冻结 `.pb` 的示例约定输入 tensor 名为 `features`、`seeds`。

坐标链验证使用三原子的三对距离作为输入、两个模型输出作为 CV，再构造标量偏置；检查坐标力、总力为零和应变导数定义的 virial。离线读取 80 帧 XYZ，输出 CV、偏置、原子力 CSV，以及均值、协方差和 lag-1 自相关。示例描述符固定为三原子、非周期体系；大体系/PBC、邻居描述符和流式读取应由独立描述符/轨迹适配层补入。

本次 TensorFlow 的 SavedModel 和冻结 PB 在 GPU 上运行，ONNX Runtime 使用 CPUExecutionProvider。接口使用 NumPy 输入/输出，会产生 Host/Device 传输；零拷贝模型张量接入是下一步集成事项。

## 4. 高维偏置与二维还原

`HillBias` 保存 hill 中心和高度，按 hill 分块计算能量与梯度。对角宽度的数值存储随 `hill_count × dimension` 增长，无需高维笛卡尔网格。支持非周期的完整协方差核，以及对角宽度下的平滑周期核。计算复杂度仍随 hill 数量增加；这里没有做 GPU 化、hill 合并或邻域截断。

PBMetaD 每维保存一维偏置，并按 `-kBT log(mean(exp(-V_i/kBT)))` 组合；与 PLUMED 的 sum 形式只差固定常数。沉积使用各维条件权重及 well-tempered 因子，所有维度的权重先从同一步旧状态计算。重启保存每一维的 hills 和沉积步数。

二维还原读取同时观测的 `(CV1,CV2)` 样本及偏置能量，以冻结偏置的 `exp(V/kBT)` 权重重建联合分布。验证采用相关二维高斯体系，在固定 PB 偏置下精确拒绝采样，再与参考联合自由能比较；同时展示把两个一维边缘分布相乘产生的误差。这里明确采用冻结偏置阶段，不能把同一权重公式直接用于任意持续沉积的非平衡轨迹。

128 维测试验证偏置求值和导数，不代表已经解决 128 维联合自由能面的统计采样问题。

## 后续逐步合入的顺序

建议先合入采样接口，再合入外部数组视图，随后接模型执行节点，最后替换/扩展高维偏置后端。每次只迁移一项，并保留原型作为数值参考。

1. 在现有 `CVGraphExecution::forward()` / `backward(seeds)` 之间选择 bias 方法：Graph 给出 CV，bias 给出能量和 seeds；单独规定沉积发生在 timestep 的哪个阶段。
2. 扩展 Leaf 的外部存储绑定，允许描述 stride 和完成事件。保留现有 GpuBuffer 路径，逐个引擎对齐生命周期、owned/ghost 与重排语义。
3. 将 ML 视为一个有 forward / VJP 的执行节点。由调度层连接 NVRTC 段与模型运行时，明确特征形状、单位和设备传输。
4. 让 dense grid、hill 列表和 PBMetaD 共享 bias 接口，先保留低维网格，再由用户选择高维表示；将重加权与模拟输出分开。

## 算法和接口依据

- [GPUMD 上游源码](https://github.com/brucefan1983/GPUMD/tree/a0e55bc87390b93d009da37c798616982573bab2)
- [LAMMPS Kokkos 源码](https://github.com/lammps/lammps/tree/stable_29Aug2024/src/KOKKOS)
- [ONNX Runtime Python 接口](https://onnxruntime.ai/docs/api/python/api_summary.html)
- [PLUMED PBMetaD 定义](https://www.plumed.org/doc-v2.9/user-doc/html/_p_b_m_e_t_a_d.html)
