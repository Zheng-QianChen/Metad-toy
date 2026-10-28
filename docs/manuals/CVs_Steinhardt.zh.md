# 集体变量解析：STEINH (键级取向序参数)

本算子对应元动力学插件中的键级取向序参数计算模块，底层核心实现位于 `CVs_Steinhardt.cu`（映射为 `MetaD_zqc::Steinhardt` 类）。该算子专门针对金属及合金凝固过程中的晶体形核、晶格对称性破缺以及多晶型选择（Polymorph Selection）场景设计，提供基于 C++/CUDA 异构并发的高性能局域取向几何求解。

---

## 1. 核心命令语法原型

在 `fix metad` 的 `CAL` 字段中声明该算子时，其标准手写语法规范如下：

```lammps
CAL NAME <cv_name> STEINH <Q/L> <L_num> <group-name> &
         [cutoff_r <float>] &
         [cutoff_eps_r <float>] &
         [SW_FUNC_r <switch-name>] &
         [d_block_size <int>]
```

Attention: 

1-使用该参数，我们会强制要求lammps提供全近邻列表。

2-如果指定 Q ，ghost原子的范围将被设置为 cutoff_r + neighbor->skin
  如果指定 L ，ghost原子的范围将被设置为 2*cutoff_r + neighbor->skin

---

## 2. 源码级运行约束与关键字参数字典

根据底层 `create()` 解析引擎状态机的标记扫描逻辑，各参数的物理含义与硬性控制边界如下：

### 2.1 基础必填项（位置参数）

| 参数项 | 源码校验规则与类型断言 | 物理与运行含义 |
| --- | --- | --- |
| `<Q/L>` | 必须为 `'Q'` 或 `'L'`（字符串） | `Q` 计算逐中心球谐模长；`L` 对邻居 qlm 进一步平均（Local Q）。 |
| `<Q_num>` | 必须为 `3, 4, 6` 之一（整型） | **球谐函数展开阶数 $l$**：`4`：适用于体心立方（BCC）或面心立方（FCC）对称性识别；`6`：适用于密堆积结构（FCC/HCP）以及超冷液体形核前沿。 |
| `<L_num>` | 必须为 `3, 4, 6` 之一（整型） | **球谐函数展开阶数 $l$**：`4`：适用于体心立方（BCC）或面心立方（FCC）对称性识别；`6`：适用于密堆积结构（FCC/HCP）以及超冷液体形核前沿。 |
| `<group-name>` | 必须在 LAMMPS 中合法定义过 | **分析原子组**：只有属于该 Group ID 的原子才会被激活作为中心原子求解 $Q_l$ 矩阵。 |

### 2.2 高级可选关键字参数

当状态机探测到基础参数后，会向前继续扫描以下关键字。若不指定，系统将自动加载底层的标准生产默认值：

| 关键字 | 类型 | 默认值 | 源码控制逻辑与边界 |
| --- | --- | --- | --- |
| `cutoff_r` | 浮点型 | `4.0` | 近邻列表物理截断半径极限（单位：$\text{Å}$）。用于构建微观几何环境。 |
| `cutoff_eps_r` | 浮点型 | `1e-6` | 径向开关的截断容差；对无限支撑开关，据此确定有效半径。 |
| `SW_FUNC_r` | 名称 | 常数权重 | 引用已定义的径向开关；有限或容差支撑半径会替换 `cutoff_r`。Local Q 的两层邻居计算均使用它。 |
| `d_block_size` | 整型 | `128` | 底层 CUDA 核函数并发执行时的 **Thread Block Size**。必须大于 `0`。用于调节 GPU 算力吞吐。 |

> 💡 **源码特性提示（底层环境合并机制）**：
> 算子内部引入了 `Steinhardt_env::get_or_create` 智能单例工厂。如果在同一个 `fix metad` 中为不同的模板声明了相同的 Q/L 类型、`group_id`、有效 `cutoff_r`、`cutoff_eps_r` 和径向开关配置，**插件在底层会自动合并近邻列表的 CUDA 计算请求**，生成唯一的虚拟环境密钥 `env_setNum`，避免多维模拟时 GPU 显存和 Neighbor 搜索开销的无谓浪费。

---

## 3. 规约函数与 SYMBOL 映射

通过 `CAL` 计算出的 `STEINH` 数据是一组高维的、针对单个原子的局域值矩阵。在后续的 `SYMBOL` 中将其降维为一维集体变量：

```lammps
SYMBOL <symbol_name> <cv_name>.<RAW/value/AVE/MEAN/SUM>
```

### 3.1 `STEINH Q`（非 Local）

必须指定 `EXECUTOR GRAPH`：

- `.AVE` / `.MEAN`：由图节点对该 CV 的中心组求全局算术平均。
- `.SUM`：对该 CV 的中心组全局求和；空组为 0。
- `.RAW` / `.value`：逐原子原始值，用于 `AVE(q4*q6)`、`SUM(fermi(q6,0.45,12))`、`safe_div(...)` 等图表达式。
- `SW_FUNC_r`：保留邻居距离的径向权重。

Q 原有 LEGACY 的 `AVE`、`SW_FUNC` 回调及 `SW_FUNC_cv` 参数已移除。对 q 值的开关在图中定义。例如 `SYMBOL q6 Q6.RAW` 配合 `DIM ... "AVE(fermi(q6,0.45,12))"` 得到平均开关值。

旧 Q 的 `SW_FUNC` 实际计算 $\sum_i q_i f(q_i)$，不是平均开关值；若目标是总数而非均值，改用下方 `SUM` 迁移式，不要把旧 `SW_FUNC` 写成 `AVE`。

### 3.2 `STEINH L`（Local / LQ）

Local Q 必须使用 `EXECUTOR GRAPH`，提供 `.RAW` / `.value`、`.AVE` / `.MEAN` 及 `.SUM`。Graph `.AVE` 是原始 LQ 的算术均值，值开关写在 DIM 中，例如 `AVE(fermi(q6,0.45,12))`。输入与验证见 [Local Q Graph](../../benchmarks/localql_graph/README.md)。

旧 CV 内部的 `AVE` / `MEAN_SOLID`、`SW_FUNC` / `NSOLID`、`FRAC` 和 `SW_FUNC_cv` 已移除，输入这些旧配置会提示迁移到 Graph。对应 Graph 写法：

| 旧轴 | 数学含义 | Graph 迁移 |
| --- | --- | --- |
| `NSOLID`（旧 `SW_FUNC` 别名） | $\sum_i f(q_i)$ | `SUM(fermi(q,...))`；空集合（含全局空）为 0 |
| `MEAN_SOLID`（该旧 fork 中的 `AVE` 别名） | $\sum_i q_i f(q_i)/\sum_i f(q_i)$ | `safe_div(SUM(q*fermi(q,...)), SUM(fermi(q,...)), epsilon)`；当 $\lvert b\rvert\le\epsilon$ 时值与两路导数均为 0（$\epsilon$ 为有限非负），不做其它静默正则 |
| `FRAC` | $\sum_i f(q_i)/N$ | `safe_div(SUM(fermi(q,...)),SUM(constant_one(q)),0)`；空中心返回 0 |

`.AVE` / `.MEAN` 按该 CV 自己的中心组求均值；`.SUM` 按同一中心组求原始值总和。
在 CVGraph 中，`SUM(expr)` 从逐原子输入推断唯一中心组，并在 GPU 上同时掩蔽值和反向种子；
因此 `fermi(0)` 等非线性变换不会把公共数组的组外补零计入总数。
多个不同中心组参与同一个表达式时，必须显式指定 `SUM(expr, q)`，其中 `q` 是目标组的 RAW 符号。
`AVE(expr)` 保留原来的公共中心并集语义；独立 Graph（无 CVGraph/中心组）中的 SUM 仍对所有输入项求和。

旧 MEAN_SOLID 的分母只在等于零时返回 0，因此迁移时 `epsilon=0` 保持该约定；
正 epsilon 是显式选择的近零区间约定，会改变该区间的值和导数，边界不光滑。
`safe_div` 支持 Global/PerAtom 同域，以及 Global Leaf 与 PerAtom 的广播；广播产生的 Global
伴随在执行器内跨 rank 求和，调用方不能再次求和。规约结果回灌逐原子融合段尚不支持。

---

## 4. 典型手写生产运行用例

### 用例一：局域短程序识别（经典 $Q_6$ 序参数计算）

以下 CAL/SYMBOL 片段放在 `fix ... metad EXECUTOR GRAPH ...` 中：

```lammps
# 声明一个名为 Q6 的局域六角对称性特征算子
CAL NAME Q6 STEINH Q 6 metad_group cutoff_r 3.2 d_block_size 128
# 将高维原子矩阵规约为一维符号变量 v1
SYMBOL v1 Q6.AVE
```

### 用例二：立方晶格对称性追踪（$Q_4$ 序参数计算）

以下片段同样使用 `EXECUTOR GRAPH`，以 Q4 的中心组均值作为符号：

```lammps
# 计算 Q4 并由 Graph 求均值
CAL NAME Q4 STEINH Q 4 bcc_group cutoff_r 3.5 &
SYMBOL v2 Q4.AVE
```

### 用例三：Local LQ6 + 平均类固分数

```lammps
fix m all metad EXECUTOR GRAPH GAUSSIAN 0.10 0.10 10 PACE 100000 &
    CAL NAME SW_r SW_FUNC FERMI r_0 3.0 alpha -20 &
    CAL NAME LQ6 STEINH L 6 metad_group cutoff_eps_r 1e-4 SW_FUNC_r SW_r &
    SYMBOL q6 LQ6.RAW &
    CV_dim 1 DIM 1 0 1 500 "AVE(fermi(q6,0.40,20))" &
    METAD_RESTART 0 WT 0
```

径向开关作用于邻居距离，Graph 的 `fermi` 作用于逐中心 LQ6；梯度依次通过这两部分传播到坐标。

---

## 5. 常见底层异常排查

1. **"Error: Steinhardt order L must be 3, 4, or 6."**
* *原因*：用户传入了不支持的阶数（例如配置成了 `STEINH Q 5 ...`），底层 `ERR_COND` 断言失败。


2. **"Error: Steinhardt type must be 'Q' (per-center) or 'L' (neighbor-averaged)."**
* *原因*：第三位参数未能识别。请检查是否发生了拼写错误（如写成了小写 `q` 或是其他非关键字符）。


3. **"Error: 'd_block_size' must be > 0"**
* *原因*：高级调优参数错误。请确保 GPU 线程块大小配置为合法的正整数（通常推荐为 `64`, `128` 或 `256`）。
