# 集体变量 (CV) 专项详解：WEIGHT_CHEM

本篇文档详细解析元动力学插件中参数最复杂的异构加速集体变量（CV）—— **多组元化学加权关联对（WEIGHT_CHEM）**。该算子底层由 `MetaD_zqc::Weighted_chem_pair` 驱动，专用于捕捉和控制多组元合金（如 Al-Sc、W-based 合金）形核前沿的溶质原子偏聚与化学短程序（SRO）波动。

---

## 1. 命令语法原型

在 `fix metad` 的 `CAL` 字段中，该算子的手写声明规范如下：

```lammps
...&
CAL NAME <CAL_name> WEIGHT_CHEM <group-name> &
         [cutoff_r <float>] &
         [d_block_size <int>] &
         [SW_func <TYPE> [r_0 <float>] [d_0 <float>] [alpha <float>] [n <int>] [m <int>]] &
         [Chem_map (<type1> <weight1>) (<type2> <weight2>) ...] &
         [Chem_ctarget <float>] &
         [Chem_sigma <float>] &
SYMBOL <SYMBOL_name> <CAL_name>.<AVE/COUNT> &
...
```

`SYMBOL` 命令可接受 CV 计算指令为：
`AVE`: 全原子平均
`COUNT`: 通过指定切换函数，过滤出有特定的 `WEIGHT_CHEM` 值的原子

注意， `Chem_ctarget` 是将
---

## 2. 核心控制关键字参数字典

| 关键字 | 类型 | 默认值 | 物理与控制边界含义 |
| --- | --- | --- | --- |
| `WEIGHT_CHEM` | 关键字 | *必填* | 激活多组元化学加权关联特征算子（底层映射样式名：`CHEM_PAIR`）。 |
| `group-name` | 字符串 | *必填* | LAMMPS 中定义的原子组名。只有属于该组的原子才会作为中心原子参与近邻环境分析。 |
| `cutoff_r` | 浮点型 | `8.0` | 近邻列表截断半径极限（单位：$\text{Å}$）。 |
| `d_block_size` | 整型 | `128` | 底层 CUDA 核函数执行时的线程块大小（Thread Block Size），用于调节 GPU 并发吞吐量。 |
| `Chem_map` | 括号对 | *可选* | 自定义各原子类型的结构权重因数，形如 `(type weight)`。支持空格或逗号分隔。 |
| `Chem_ctarget` | 浮点型 | *可选* | 化学锁定机制的目标偏置中心值 $c_{\text{target}}$。 |
| `Chem_sigma` | 浮点型 | *可选* | 化学锁定高斯分布的宽度因子 $\sigma$。与 `Chem_ctarget` 共同激活**化学锁（Chemical Lock）**。 |

---

### 连续截断平滑开关函数 (SW_func)

若脚本中声明了 `SW_func <TYPE>`，系统将启动内层状态机解析亚参数。
*注意，如果在 SYMBOL 中指定 集体计算模式为 "COUNT" 格式的时候，才需要指定 SW_func。否则将会忽视该参数。*
未显式声明的子参数将自动加载标准默认值：

#### 1 FERMI (费米函数)

* **数学形式**：$f(r) = \frac{1}{1 + e^{(r - r_0)/\alpha}}$
* **子参数**：`r_0`（过渡中心，默认 `1.0`）、`alpha`（边界平滑锐度，默认 `20.0`）。

#### 2 TANH (双曲正切函数)

* **数学形式**：$f(r) = \frac{1}{2} \left[1 - \tanh\left(\frac{r - r_0}{\alpha}\right)\right]$
* **子参数**：`r_0`（默认 `1.0`）、`alpha`（默认 `20.0`）。

#### 3 RATIONAL (有理分式连续函数)

* **数学形式**：$f(r) = \frac{1 - ((r-d_0)/r_0)^n}{1 - ((r-d_0)/r_0)^m}$
* **子参数**：`r_0`（标度半径，默认 `1.25`）、`d_0`（物理平移量，默认 `0.0`）、`n`（分子幂次，默认 `6`）、`m`（分母幂次，默认 `12`）。

> ⚠️ **源码级特性提示**：本插件的内层循环解析器具备**边界安全断点**机制。当解析开关函数遭遇不属于该模型的外部关键字（如 `Chem_map` 或下一个 `CAL`）时，会自动触发 `break` 退出内层状态机，确保后续主参数完美合流。

---

## 3. 典型手写生产运行用例

### 用例一：紧凑型多组元元素加权映射

用于在常规元动力学中为不同元素赋予不同的形核权重贡献：

```lammps
CAL NAME CP1 WEIGHT_CHEM metad_group cutoff_r 4.5 &
        SW_func RATIONAL r_0 1.25 n 6 m 12 &
        Chem_map (1,0.1) (2,3.0)
```

### 用例二：分立型传参兼容性用例（空格分隔）

针对超冷液体传参时可能发生的 Token 拆分，插件底层内置了自适应向前探测拼接机制，以下语法等价且安全：

```lammps
CAL NAME CP2 WEIGHT_CHEM metad_group cutoff_r 4.5 &
        Chem_map (1 0.1) (2 3.0)
```

### 用例三：全参数激活（费米开关函数 + 高斯化学锁控制）

用于自由能表面特定亚稳态位置的精确拓扑拓荒，强行锁定临界溶质聚集核心：

```lammps
CAL NAME CP_Lock WEIGHT_CHEM solid_group cutoff_r 6.0 &
        d_block_size 256 SW_func FERMI r_0 2.0 alpha 10.0 &
        Chem_map (1,1.0) (2,1.0) Chem_ctarget 1.1 Chem_sigma 2.691
```