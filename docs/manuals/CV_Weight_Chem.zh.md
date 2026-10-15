# WEIGHT_CHEM：元素加权近邻和

`WEIGHT_CHEM` 为每个中心原子计算一个元素加权的径向近邻和，适合描述局部成分或偏聚。它保留径向统计，不含结构因子的 `sinc(q*r)` 项，也不包含角度信息。当前通过 `EXECUTOR GRAPH` 接入偏置计算。

## 定义

```text
W_i = 1 + sum_j w(type_i) * w(type_j) * s(r_ij)

s(r) = 1                         r² <= 0.8*rc²
       1 - u³                    0.8*rc² < r² < rc²
       0                         r² >= rc²
u = (r² - 0.8*rc²) / (0.2*rc²)
```

常数自项为 1。邻居可包含中心组外的原子。径向截断沿用原定义：函数值连续，截断处一阶导数不连续。

元素权重优先读取 `Chem_map`；未列出的类型使用：

```text
w(type) = exp(-0.5 * ((type - Chem_ctarget) / Chem_sigma)^2)
```

这里的 type 是 LAMMPS 数字类型编号。编号本身没有连续化学含义，实际元素权重建议显式给出。

## 参数

```text
CAL NAME <name> WEIGHT_CHEM <center-group> \
    cutoff_r <radius> d_block_size <threads> \
    Chem_map (<type>,<weight>) ... \
    Chem_ctarget <target> Chem_sigma <sigma>
```

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `center-group` | 必填 | 选定中心原子的 LAMMPS group。 |
| `cutoff_r` | `8.0` | 正的邻居截断半径，距离单位由 LAMMPS `units` 决定。 |
| `d_block_size` | `128` | CUDA 线程块大小，取值 1–1024。 |
| `Chem_map` | 未设置 | 有限的元素权重，可正、负或零；键为数字类型编号。 |
| `Chem_ctarget` | `0.0` | 未显式映射类型的高斯权重中心。 |
| `Chem_sigma` | `1.0` | 高斯权重宽度，必须为正。 |

`Chem_map (1,1.0) (2,-0.7)` 与空格形式 `(1 1.0) (2 -0.7)` 均可读取。此接口使用数字类型；CSS 的 CHANNEL 另外支持 type label。

## Graph 输入示例

以下片段假定体系已有三种 atom type，且已定义中心组 `selected`：

```lammps
fix m all metad EXECUTOR GRAPH GAUSSIAN 0.10 0.10 10.0 PACE 100000 &
    CAL NAME W WEIGHT_CHEM selected cutoff_r 3.05 &
        Chem_map (1,1.0) (2,-0.7) (3,0.35) &
    SYMBOL w W.RAW &
    CV_dim 1 DIM 1 0 1 500 "AVE(fermi(w,-4,0.4))" &
    METAD_RESTART 0 WT 0
```

- `W.RAW` / `W.value`：公共压缩中心并集上的逐原子值，组外位置补零。
- `W.AVE` / `W.MEAN`：按 W 自己的全局 owned 中心数求平均。
- `AVE(expr)`：按参与装配的公共并集 owned 数求平均。多个 CV 使用不同中心组时，注意补零经过非线性函数后的贡献。

`fermi(x,r0,alpha) = 1/(1+exp(-alpha*(x-r0)))`。alpha 为正时随 x 增大，为负时随 x 减小；上例参数仅展示接口，应按原始值分布设置。

CV 值上的旧 `SW_func` 参数已移到 Graph 表达式，邻居距离上的径向函数仍在 CV 内部计算。旧 LEGACY AVE/COUNT 回调已移除；总 COUNT 规约尚未实现，`AVE(fermi(...))` 表示平均开关值。

## 力、virial 与诊断

Graph 返回每个中心的 `dV/dW_i`，CV 将每个邻居项的贡献累加到两端原子。三分量力按 `3*localid+c` 保存；六分量 virial 按 `6*localid+c` 保存，顺序为 `xx,yy,zz,xy,xz,yz`，每项向两端各分一半。ghost 贡献通过宽度 9 的 reverse communication 回到 owner。

逐原子原始值可用以下接口查看：

```lammps
compute raw all metad/atom m W.chem_pair_r
```

数值验证与复现入口见 [WEIGHT_CHEM 测试](../../benchmarks/weighted_chem/README.md)，分离/融合开关函数的速度对照见 [性能报告](../../benchmarks/weighted_chem_switch/RESULTS.md)。
