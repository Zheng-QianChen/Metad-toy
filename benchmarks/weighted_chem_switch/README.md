# CV 值上 SW_func 的分离 / 内联对照

比较 `WEIGHT_CHEM` 原始值上的 `fermi(w,-4,0.4)`。两版使用相同的设备 Fermi 与导数、每线程一个中心的邻居循环、双精度计算以及力和六分量 virial。径向开关始终在邻居循环中。

- 分离版：生产 CV kernel 生成 W，实际 Graph/NVRTC kernel 计算 Fermi；反向依次执行 Graph 导数和 CV 坐标力。
- 内联版：在 CV 前向末尾直接调用 Fermi；反向先乘 Fermi 导数，再进入相同邻居循环。保留原始 W 供反向使用。

`prepare.py` 从当前生产 kernel 提取函数体，用断言校验插入位置，生成仅用于测试的融合版本。在 `build` 中复制必要源文件并构建临时插件；生产目录在准备和汇总时通过 SHA-256 检查。更换 kernel 结构后，应重新准备实验副本。

## 测量范围

`benchmark.cu`：预先构造周期 FCC 邻居几何，中心数为 64/256/4,000/32,000，邻居数为 12/42/134。分别测前向、反向和前向+反向。每组先验证输出、力和 virial，并要求力非零。用 CUDA Graph 捕获 32 次操作并通过 CUDA event 计时，每个样本约 30 ms，随机化 A/B 顺序，重复 9 次。这样测的是 GPU 计算段；Host 每次提交 kernel 的开销由完整 LAMMPS 对照覆盖。反向计时包含两版相同的力/virial 清零。

`run_lammps.py`：4,000 个三元 FCC 原子，`pair_style zero`，中心为 all，cutoff 为 3.05，Chem_map 为 1/-0.7/0.35。同一 GPU 分别运行 1/2 MPI rank，每个 rank 一个 OpenMP 线程。先做非零偏置力的 NVE 数值对照，再采用相同平移轨迹计时，以保持两版工作量一致。每次预热 1,000 步、计时 20,000 步，重复 5 次；取最后一段 LAMMPS Loop time，排除启动、NVRTC 与预热。

完整流程的分离版表达式为 `AVE(fermi(w,-4,0.4))`，内联测试插件输出已切换的值，表达式为 `AVE(w)`。两版都保留 Graph 装配及均值路径，因此完整 timestep 仍有多个 kernel。这个实验衡量的是把 CV 值上的 Fermi 放回 CV kernel 的收益。

## 复现

先构建现有 `build/graph` 与 `build/graph-lammps` Release 目标；脚本中的 CUDA、MPI、LAMMPS 路径沿用本机设置，GPU 架构为 sm_89。以下命令依次执行，避免 GPU 测量互相干扰。

```sh
python3 benchmarks/weighted_chem_switch/prepare.py --output build/switch-new
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 build/switch-new/benchmark build/switch-new 9 > build/switch-new/micro.log
python3 benchmarks/weighted_chem_switch/run_lammps.py \
  --prepared build/switch-new --output build/switch-new-lammps \
  --steps 20000 --warmup 1000 --repeats 5 --np 1 2
python3 benchmarks/weighted_chem_switch/summarize.py \
  --micro build/switch-new --lammps build/switch-new-lammps --output build/switch-new-report
```

`--resume` 仅用于继续编译已经准备好的同一实验副本；源码改变后使用新的输出目录。完整 LAMMPS 日志保留在 build；精选原始样本及报告保留在本目录。

## 结果的适用范围

结果见 [RESULTS.md](RESULTS.md)。当前只测了 Fermi 与 WEIGHT_CHEM，其他 CV、邻居分布、开关表达式和 GPU 需要单独测量。此次使用桌面 GPU/WSL2，重复样本存在波动；微小差异不能作为稳定加速的证据。

本次完整 timestep 的中位数：1 rank 分离 763.56、内联 765.78 µs/step；2 rank 分离 1421.97、内联 1418.77 µs/step。差异小于重复样本波动，暂未观察到稳定的整体加速。GPU 段在 4,000 中心、12 邻居时约 1.20 倍，42/134 邻居时接近持平。当前保留生产 Graph 分离方案。
