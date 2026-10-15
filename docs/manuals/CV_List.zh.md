# 集体变量 (CVs) 完整列表与参数规范

本页详细列出了 **MetaD-toy** 插件当前支持的所有集体变量（Collective Variables）。所有算子均基于底层 CUDA 进行高并发异构加速。

---

### 1. 键级取向序参数 (Steinhardt Order Parameters)
* **对应源码**：`CVs_Steinhardt.cu`
* **关键字**：`STEINH`
* **物理意义**：用于捕捉局域晶格对称性破缺，识别固液相变及多晶型演化。

| 进阶子参数 | 类型 | 默认值 | 物理含义 |
| :--- | :--- | :--- | :--- |
| `Q/L` | 字符串 | 必填 | `Q` 代表局域取向序（Local），`L` 代表全局取向序（Global） |
| `L_num` | 整型 | 必填 | 展开阶数，通常可选 `4`（立方结构）或 `6`（密堆积/液体形核） |
| `cutoff_r` | 浮点型 | `4.0` | 近邻列表截断半径（单位：$\text{Å}$） |
| `cutoff_Natoms` | 整型 | `12` | 中心原子允许的最大近邻原子数上限 |

---

### 2. 结构因子与元素加权近邻统计

以下 CV 使用 `EXECUTOR GRAPH`，提供 `.RAW` / `.value` 和 `.AVE` / `.MEAN` 输入。

| 关键字 | 逐中心原始值 | 配置说明 |
| --- | --- | --- |
| `STRU_FACTOR` | `1 + sum_j sinc(q*r_ij)*s(r_ij)` | 指定中心组、`cutoff_r`、`q_factor`。 |
| `STRU_FACTOR ... Chemical` | `1 + sum_j w_i*w_j*sinc(q*r_ij)*s(r_ij)` | 增加 `Chem_map` 或高斯后备权重参数。 |
| `WEIGHT_CHEM` | `1 + sum_j w_i*w_j*s(r_ij)` | 元素加权的径向统计，默认 `cutoff_r=8.0`。 |

`Chem_map` 使用数字类型编号，允许正、负、零权重。未指定的类型通过 `Chem_ctarget`、`Chem_sigma` 映射。CV 值的 switching function 写入 Graph 表达式。

输入示例与测试见 [普通结构因子](../../benchmarks/strufactor_lammps_graph/README.md)、[化学结构因子](../../benchmarks/strufactor_chem/README.md) 和 [WEIGHT_CHEM 手册](CV_Weight_Chem.zh.md)。

---

### 3. 基础几何距离 (Geometric Distance)
* **对应源码**：`CVs_DISTANCE.cu`
* **关键字**：`DISTANCE`
* **物理意义**：计算指定两组原子质心之间的标量距离，用于追踪简单的反应路径或集团拉伸。

---

### 4. 多通道化学环境评分 CSS

`CAL NAME C CSS <center-group>` 下可重复声明 `CHANNEL <neighbor-type> <radial-function> <target> <sigma>`。每个通道统计特定元素的平滑配位数，最终按与目标的偏差给出高斯评分。邻居类型支持数字编号及已定义的 type label；同一元素可配置多个统计半径。

使用 `EXECUTOR GRAPH`，输入为 `C.RAW` / `C.AVE` 等。完整定义、类型标签与多元素示例见 [CSS 说明](../../benchmarks/css/README.md)。
