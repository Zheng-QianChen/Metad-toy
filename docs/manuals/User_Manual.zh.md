# 基于 CUDA 加速的相变元动力学插件（MetaD-toy）用户使用手册

本手册旨在指导用户如何在 LAMMPS 分子动力学软件中加载并配置 **MetaD-toy** 元动力学（Metadynamics）模拟插件。该插件专门针对金属及合金（如铝基体）的凝固形核、多相转变等场景设计，通过底层的 C++/CUDA 异构并发架构，提供高并发的局域晶格序参数求解以及高效的偏置势沉积计算。

---

## 1. 插件加载机制

本插件采用非侵入式动态链接库接口设计。在 LAMMPS 脚本（`in` 文件）中，必须首先通过 `plugin load` 命令显式挂载扩展模块：

```lammps
plugin load ${METAD_PLUGIN_PATH}/fix_crystallize_plugin.so
```

> **注意**：`${METAD_PLUGIN_PATH}` 为插件编译产物所在的绝对或相对路径，确保环境中有执行权限。

---

## 2. 核心命令：fix metad 语法规范

加载插件后，通过定义一个 `fix` 关键字为 `metad` 的系统级挂载实例，来控制整个元动力学框架。

### 2.1 命令基本原型

```lammps
fix <fix-ID> <group-ID> metad <KEYWORD> <Args> ...
```

* **fix-ID**：该 fix 的自定义名称（如 `m`）。
* **group-ID**：参与元动力学偏置力作用的原子组（通常为 `all`）。
* **metad**：插件注册的样式名（Style Name）。

### 2.2 核心配置参数项（Keywords）

主程序（`FixMetadynamics` 类）内置了一个命令流标记（Token）状态机扫描引擎。支持的参数关键字及配置说明如下：

#### ① GAUSSIAN

* **语法**：`GAUSSIAN <sigma> <height0> <biasf>`
* **示例**：`GAUSSIAN 0.003 0.05 10.0`
* **含义**：
* `sigma`：高斯偏置山峰的截断或演化半宽度（$\sigma$）。
* `height0`：初始高斯势垒高度（$W_0$），能量单位与 LAMMPS 的 `units` 设置一致（例如在 `real` 单位下为 $\text{kcal/mol}$，在 `metal` 单位下为 $\text{eV}$）。
* `biasf`：偏置因子（Bias Factor），用于调控井深。



#### ② PACE

* **语法**：`PACE <timesteps>`
* **示例**：`PACE 1`
* **含义**：高斯偏置势（HILLS）的沉积周期步长。设置为 `1` 表示每一步演化均会无缝向网格堆叠新高斯势。

#### ③ RECORD

* **语法**：`RECORD FILE_NAME <filename> REC_PACE <pace>`
* **示例**：`RECORD FILE_NAME rec.log REC_PACE 1`
* **含义**：历史轨迹及元动力学日志输出配置。`FILE_NAME` 指定输出日志文件名；`REC_PACE` 指定输出当前 CV 值的步长间隔。

#### ④ CV_dim

* **语法**：`CV_dim <dimension_integer>`
* **示例**：`CV_dim 2`
* **含义**：定义自由能势能面集合变量（Collective Variables, CVs）的总空间维度，当前底层架构优化支持 `1` 到 `3` 维空间。

#### ⑤ CAL（计算变量声明）—— 双层派发机制

* **语法**：`CAL NAME <cal_name> <CV_Type> <Sub_Args...>`
* **含义**：声明并动态创建底层的微观几何量/序参数计算引擎。控制权会由主 Fix 移交给对应的子工具类：
* **STEINH（键级取向序参数）**：
* *语法*：`STEINH Q <L_num> <group> cutoff_r <r> cutoff_Natoms <N>`
* *示例*：`STEINH Q 6 metad_group cutoff_r 3.2 cutoff_Natoms 12`
* *说明*：求解指定原子组的 Steinhardt 序参数 $Q_l$（如 $Q_6$）。`cutoff_r` 为近邻阶段截断半径（$\text{Å}$），`cutoff_Natoms` 为中心原子最大容纳的近邻数上限。


> 💡 **核心提示**：本插件支持多种经由 CUDA 加速的晶格序参数与多组元化学特征算子（如 `STEINH`, `WEIGHT_CHEM` ）。关于每种 `CV_Type` 的详细底层数学物理定义、参数列表及示例，请参阅：
> **[👉 集体变量 (CVs) 完整列表与参数规范](CV_List.zh.md)**




#### ⑥ SYMBOL

* **语法**：`SYMBOL <symbol_name> <cal_name>.<func_name>`
* **示例**：`SYMBOL v1 Q6.AVE` ； `SYMBOL v2 CP.AVE`
* **含义**：将上步 `CAL` 定义的具体物理变量矩阵映射为数学公式解析引擎可调用的符号变量。`.AVE` 表示计算该组原子的系统空间平均值。

如果CAL是原子特征，那么需要使用SYMBOL指定CAL中的原子特征映射为集体特征的方式。

#### ⑦ DIM（空间边界离散）

* **语法**：`DIM <dim_index> <lower_bound> <upper_bound> <bins> "<symbol_expression>"`
* **示例**：
```lammps
DIM 1 0.0 1.0 10000 "v1"
DIM 2 0.0 10  10000 "v2"
```


* **含义**：对自由能超曲面各个维度（1-based 索引）进行边界约束和网格离散化：
* `<lower_bound>` / `<upper_bound>`：设定 CV 轴的物理求值上下界。
* `<bins>`：网格离散数。例如 `10000` 代表将势能面分成 10000 个离散栅格。
* `"<symbol_expression>"`：用双引号包裹的数学表达式，支持符号库级联计算（如 `"v1"` 或 `"(v1+v2)/2"`）。



#### ⑧ Gaussian_Hill_type（高级性能设置）

* **语法**：`Gaussian_Hill_type <type_id>`
* **说明**：设置高斯山在显存空间中的存储模型。`0` 代表**均匀密集网格（Uniform Grid）**，适用于1至3维的中低精度密集计算；`1` 代表**去饱和稀疏哈希自由能网格（Sparse Hash）**，能防止高维大生命周期模拟时的显存爆炸。*(若脚本不指定，主程序默认依据维度构建优化网格)*。

#### ⑨ METAD_RESTART 与 WT

* **语法**：`METAD_RESTART <0/1>`：是否断点续训，`1` 表示从已有自由能文件中恢复势能面。
* **语法**：`WT <0/1>`：井温元动力学（Well-Tempered Metadynamics）开关，`0` 表示不激活。

---

## 3. 计算节点与原子信息导出：compute metad/atom

### 3.1 命令基本原型

```lammps
compute <compute-ID> <group-ID> metad/atom <fix-ID> <cv_name_1> <cv_name_2> ...
```

* **compute-ID**：该计算节点的自定义名称（如 `CvPa`）。
* **group-ID**：指定计算的目标原子组（只有属于该组的原子才会被提取数据，其他组原子默认填充 `0.0`）。
* **metad/atom**：指定调用元动力学插件扩展的原子级属性导出器。
* **fix-ID**：指定与之绑定的、正在运行的元动力学 `fix` 实例 ID（如 `m`）。
* **cv_name_X**：请求导出的具体集体变量名称（如 `Q6.stein_q`、`CP.chem_pair_r`），支持同时请求单个或多个变量。

### 3.2 底层数据路由机制与多维输出逻辑

本软件的 `compute` 模块采用**低耦合、高内聚的动态路由总线设计**。在初始化阶段（`init()`），软件会自动捕获上层绑定的 `fix-ID`，并严格校验其合法性。在时间步演化及 `dump` 触发时，系统通过以下两个自动化机制完成数据极速拷贝：

1. **自适应输出维度降级（Polymorphic Output）**：
* **单变量请求**：若用户只请求了一个 CV（如单个 `Q6.stein_q`），系统自动将输出声明为**每原子标量（vector_atom）**。
* **多变量请求**：若用户请求了多个变量，系统自动将输出转换为**每原子矩阵（array_atom）**，列数等于请求的变量总数。


2. **跨模块指针直连（extract 握手）**：
系统会根据用户输入的 `cv_name` 自动拼接底层通信密钥（如 `"colvar_peratom:Q6.stein_q"`），通过 LAMMPS 原生的 `extract()` 接口强行敲开绑定的 `fix` 内部显存，通过指针直连以 $O(N)$ 的吞吐量完成本地原子（`nlocal`）数据的安全覆写与掩码过滤。

### 3.3 完整的配置与 Dump 示例

```lammps
# 1. 动态载入异构计算插件
plugin load ${METAD_PLUGIN_PATH}/fix_crystallize_plugin.so

# 2. 配置多维元动力学系统
fix m all metad GAUSSIAN 0.003 0.05 10.0 &
                    PACE 1 &
                    RECORD FILE_NAME rec.log REC_PACE 1 &
                    CAL NAME Q6 STEINH Q 6 metad_group cutoff_r 3.2 cutoff_Natoms 12 &
                    CAL NAME CP WEIGHT_CHEM metad_group cutoff_r 3 Chem_map (1,0.1) (2,3.0) &
                    SYMBOL v1 Q6.AVE &
                    SYMBOL v2 CP.AVE &
                    CV_dim 2 &
                    DIM 1 0.0 1.0 10000 "v1" &
                    DIM 2 0.0 10  10000 "v2" &
                    METAD_RESTART 1 &
                    WT 0

# 3. 关联原子级单变量提取（输出为标量，通过 c_CvPa 调用）
compute CvPa all metad/atom m Q6.stein_q
dump 1 all custom 1000 single_traj.lammpstrj id type x y z c_CvPa

# 4. 关联原子级多变量提取（输出为矩阵，通过 c_CvPaM[1], c_CvPaM[2] 调用）
compute CvPaM all metad/atom m Q6.stein_q CP.chem_pair_r
dump 2 all custom 1000 multi_traj.lammpstrj id type x y z c_CvPaM[1] c_CvPaM[2]
```

---

## 4. 完整的配置示例与说明书参考

在配置涉及铝稀土合金结晶、多维自由能跨越的复杂体系时，脚本的书写规范如下：

```lammps
# 1. 动态载入异构计算插件
plugin load ${METAD_PLUGIN_PATH}/fix_crystallize_plugin.so

# 2. 配置多维元动力学系统（采用 & 续行符连结）
fix m all metad GAUSSIAN 0.003 0.05 10.0 &
                    PACE 1 &
                    RECORD FILE_NAME rec.log REC_PACE 1 &
                    CAL NAME Q6 STEINH Q 6 metad_group cutoff_r 3.2 cutoff_Natoms 12 &
                    CAL NAME CP WEIGHT_CHEM metad_group cutoff_r 3 Chem_map (1,0.1) (2,3.0) &
                    SYMBOL v1 Q6.AVE &
                    SYMBOL v2 CP.AVE &
                    CV_dim 2 &
                    DIM 1 0.0 1.0 10000 "v1" &
                    DIM 2 0.0 10  10000 "v2" &
                    METAD_RESTART 1 &
                    WT 0

# 3. 关联原子受力更新与属性 Dump
compute CvPa all metad/atom m Q6.stein_q CP.chem_pair_r
dump 1 all custom 1000 traj.lammpstrj id type x y z c_CvPa[1] c_CvPa[2]
```

## 5. 常见错误与异常排查

当系统运行失败时，可检查根目录自动生成的白盒调试日志文件：`metad_debug_logging.txt`。

1. **"Error: GAUSSIAN command requires 3 arguments..."**
* *原因*：`GAUSSIAN` 后面遗漏了参数，或续行符 `&` 导致参数断流。


2. **"Only 1D-3D are supported for optimized grid..."**
* *原因*：`CV_dim` 设置的值超过了 3，或者定义的 `DIM` 数量与维度总数不匹配。


3. **"Error: CAL requires NAME keyword."**
* *原因*：子类参数解析模块中语法有误，请确保每个变量都以 `CAL NAME <自定义名> <类型>` 开头。