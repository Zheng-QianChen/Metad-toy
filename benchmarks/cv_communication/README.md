# CV 通信转发验证

`FixMetadynamics` 每次保存当前 CV、方向和每原子包宽，再调用 LAMMPS 的 `forward_comm(this, width)` 或 `reverse_comm(this, width)`。四个打包/解包回调直接转发给该 CV，offset 为 0，stride 为本次宽度。

CV 的 `get_comm_forward_width()` / `get_comm_reverse_width()` 返回每原子的 double 数，0 表示该方向无需通信。Fix 在每次 init 时取各 CV 的最大宽度作为缓冲容量，校验打包返回值、容量和 MPI int 计数，并在正常结束或异常退出时清理当前目标。各 rank 按相同 CV 顺序调用，包含本地组为空的 rank。

Stru_factor、STEIN_QL、LocalQL、WEIGHT_CHEM 均使用此入口。CV 内部的 comm_mode/reverse_active 和通信前的 MPI_Barrier 已移除。LocalQL 复用基类的 reverse 实现，forward 宽度为 0。

## 运行

先保留改造前插件，再构建当前插件。基准插件需支持所选场景的同一输入语法；Q Graph 场景需使用已接入 Graph 的基准插件。以下命令从仓库根目录运行，输出目录应为新目录。初态为已有完整 LAMMPS 基准生成的 864 原子固定编号 FCC 文件；MPI、LAMMPS 和头文件路径可以通过参数覆盖。

```sh
python3 benchmarks/cv_communication/run.py \
  --baseline-plugin build/localql-interface-cleanup/before.so \
  --cases strufactor q_graph q_graph_odd local_graph \
  --initial build/graph-lammps/strufactor-merge-validation/initial.data \
  --output build/graph-lammps/cv-comm-check-new

python3 benchmarks/cv_communication/run.py \
  --baseline-plugin build/q-interface-cleanup/before.so \
  --cases strufactor q_graph --comm-style tiled --np 2 \
  --initial build/graph-lammps/strufactor-merge-validation/initial.data \
  --output build/graph-lammps/cv-comm-tiled-new

python3 benchmarks/cv_communication/run.py \
  --after-only --cases chem_mix \
  --initial build/graph-lammps/strufactor-merge-validation/initial.data \
  --output build/graph-lammps/cv-comm-chem-new
```

`trace_comm.cpp` 用 LD_PRELOAD 记录 metad Fix 的通信宽度、容量、调用次数和实际 MPI_Send 的 double 数。它支持本机 LAMMPS 的 brick/tiled C++ 符号，MPI_Send 计数通过 PMPI 转发。探针只用于通信验证，结果不作为运行速度基准。

## 初次通信重构的历史结果

- 插件和独立构建成功，9 项 CTest 全部通过。
- 双 Stru_factor、Q4/Q6、Q4/LocalQ：1/2/4 rank 的修改前后受力、CV、能量、六分量压力对照通过。
- Stru_factor、Q4/Q6 在 2 rank tiled 布局下通过相同对照。
- 不同/非连续原子组、空组 rank、组外邻居受力、同一步修改坐标后重算均通过已有回归。
- WEIGHT_CHEM 修复了一处重复 Fixmetad 成员遮蔽基类指针的问题。旧插件在首次 forward 通信时崩溃，关闭探针也复现；此项用修复后 1 rank 为参考，验证 2/4 rank 的受力、CV、能量及压力。化学值在测试表达式中乘 0.05，使其位于偏置网格内并保持非零受力。

通信轮数和 MPI_Send 次数与原版相同。2/4 rank 下，双 Stru_factor 的 reverse 数据量降低 50%；Q4/Q6 的 forward、reverse 均降低 50%；Q4/LocalQ 的 reverse 降低 50%，forward 数据量保持相同。统计只覆盖跨 rank 发出的 Fix 通信 double，不包含本地周期镜像拷贝和其他 LAMMPS 通信。

完整汇总、插件 SHA-256、每 rank 包宽及数值误差在 `results.json`。输入、dump、screen、探针逐 rank 日志和构建记录保存在 `build/graph-lammps/cv-comm-*`。

## WEIGHT_CHEM Graph 接入后的当前入口

`chem_mix` 现使用 `STRU_FACTOR Chemical + WEIGHT_CHEM` 的 Graph 组合，两者共享同组同 cutoff 的化学环境。forward 通信宽度为 0，reverse 每原子为 9 个 double（力 3 + virial 6）。此场景用 `--after-only --cases chem_mix` 检查 1/2/4 rank，首次迁移前的旧插件不支持 WEIGHT_CHEM Graph；before/after 对照应显式选择双方都支持的场景。

上文记录的 `STEINH + WEIGHT_CHEM`、forward 宽度 1 属于迁移前的历史测试。当前 WEIGHT_CHEM 的 CPU 对照、Graph 开关、不同组、逐原子应力与全局压力验证见 `benchmarks/weighted_chem/`。

## Q 接口清理后的场景

当前 `q_graph` / `q_graph_odd` 分别验证 Q4/Q6 与 Q3/Q6，forward 宽度为 13/17 与 11/17，reverse 为 9。旧 `q_pair` / `local_mix` / `local_pair` 输入依赖已移除的 LEGACY Q/Local Q，保留在历史结果中。

## Local Q Graph 场景

`local_graph` 使用 L4（block size 64）与 L6（128），forward 为 0、reverse 为 9；`q_local_graph` 使用 Q4 与 L6，forward 为 13、reverse 为 9。当前脚本统一使用 Graph。`local_weighted`、`local_counts` 的 LEGACY 加权均值和 NSOLID/FRAC 对照作为迁移阶段的历史记录保留；这些入口已从当前脚本删除。
