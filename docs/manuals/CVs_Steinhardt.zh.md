# 集体变量解析：STEINH (键级取向序参数)

本算子对应元动力学插件中的键级取向序参数计算模块，底层核心实现位于 `CVs_Steinhardt.cu`（映射为 `MetaD_zqc::Steinhardt` 类）。该算子专门针对金属及合金凝固过程中的晶体形核、晶格对称性破缺以及多晶型选择（Polymorph Selection）场景设计，提供基于 C++/CUDA 异构并发的高性能局域取向几何求解。

---

## 1. 核心命令语法原型

在 `fix metad` 的 `CAL` 字段中声明该算子时，其标准手写语法规范如下：

```lammps
CAL NAME <cv_name> STEINH <Q/L> <L_num> <group-name> &
         [cutoff_r <float>] &
         [cutoff_Natoms <int>] &
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
| `<Q/L>` | 必须为 `'Q'` 或 `'L'`（字符串） | **取向序空间类型**：`Q` 代表**局域键级取向序**（Local Order）；`L` 代表**局域平均键级取向序**（Global Order）。 |
| `<Q_num>` | 必须为 `3, 4, 6` 之一（整型） | **球谐函数展开阶数 $l$**：`4`：适用于体心立方（BCC）或面心立方（FCC）对称性识别；`6`：适用于密堆积结构（FCC/HCP）以及超冷液体形核前沿。 |
| `<L_num>` | 必须为 `3, 4, 6` 之一（整型） | **球谐函数展开阶数 $l$**：`4`：适用于体心立方（BCC）或面心立方（FCC）对称性识别；`6`：适用于密堆积结构（FCC/HCP）以及超冷液体形核前沿。 |
| `<group-name>` | 必须在 LAMMPS 中合法定义过 | **分析原子组**：只有属于该 Group ID 的原子才会被激活作为中心原子求解 $Q_l$ 矩阵。 |

### 2.2 高级可选关键字参数

当状态机探测到基础参数后，会向前继续扫描以下关键字。若不指定，系统将自动加载底层的标准生产默认值：

| 关键字 | 类型 | 默认值 | 源码控制逻辑与边界 |
| --- | --- | --- | --- |
| `cutoff_r` | 浮点型 | `4.0` | 近邻列表物理截断半径极限（单位：$\text{Å}$）。用于构建微观几何环境。 |
| `cutoff_Natoms` | 整型 | `12` | **显存安全红线**：每个中心原子允许分配的最大近邻原子数上限。用于开辟静态 GPU 线程块显存。 |
| `d_block_size` | 整型 | `128` | 底层 CUDA 核函数并发执行时的 **Thread Block Size**。必须大于 `0`。用于调节 GPU 算力吞吐。 |

> 💡 **源码特性提示（底层环境合并机制）**：
> 算子内部引入了 `Steinhardt_env::get_or_create` 智能单例工厂。如果在同一个 `fix metad` 中为不同的模板声明了相同的 `group_id`、`cutoff_r` 和 `cutoff_Natoms`，**插件在底层会自动合并近邻列表的 CUDA 计算请求**，生成唯一的虚拟环境密钥 `env_setNum`，避免多维模拟时 GPU 显存和 Neighbor 搜索开销的无谓浪费。

---

## 3. 规约函数与 SYMBOL 映射

通过 `CAL` 计算出的 `STEINH` 数据是一组高维的、针对单个原子的局域值矩阵。在后续的 `SYMBOL` 中，通常将其映射为空间系统均值：

```lammps
SYMBOL <symbol_name> <cv_name>.func_name

```

* **`.AVE`（空间系统均值）**：计算该原子组内所有激活原子的 $Q_l$ 值的算术平均数，将其降维规约为元动力学（MetaD）自由能网格所需的一维标量。

---

## 4. 典型手写生产运行用例

### 用例一：局域短程序识别（经典 $Q_6$ 序参数计算）

用于捕捉液相向密堆积晶体（FCC/HCP）形核演化的标准配置：

```lammps
# 声明一个名为 Q6 的局域六角对称性特征算子
CAL NAME Q6 STEINH Q 6 metad_group cutoff_r 3.2 cutoff_Natoms 12 d_block_size 128
# 将高维原子矩阵规约为一维符号变量 v1
SYMBOL v1 Q6.AVE
```

### 用例二：立方晶格对称性追踪（$Q_4$ 序参数计算）

用于追踪 BCC 或高密堆积相变，适当放宽近邻容纳上限：

```lammps
# 探测四角/立方局域对称性，近邻上限强行开辟至 14
CAL NAME Q4 STEINH Q 4 bcc_group cutoff_r 3.5 cutoff_Natoms 14 &
SYMBOL v2 Q4.AVE
```

---

## 5. 常见底层异常排查

1. **"Error: Steinhardt order L must be 3, 4, 6, 8, or 12."**
* *原因*：用户传入了不支持的阶数（例如配置成了 `STEINH Q 5 ...`），底层 `ERR_COND` 断言失败。


2. **"Error: Steinhardt type must be 'Q' (local) or 'L' (global)."**
* *原因*：第三位参数未能识别。请检查是否发生了拼写错误（如写成了小写 `q` 或是其他非关键字符）。


3. **"Error: 'd_block_size' must be > 0"**
* *原因*：高级调优参数错误。请确保 GPU 线程块大小配置为合法的正整数（通常推荐为 `64`, `128` 或 `256`）。