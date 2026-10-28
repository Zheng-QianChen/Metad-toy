# 共享偏势的平衡概率验证

本目录的 probability.py 实现统计验证的前两步：固定共享偏势重加权，以及
两个相关局域变量共同沉积后冻结、重新平衡、再重加权。它使用独立的二维参考积分，
输出联合分布、矩、误差条和带采样覆盖检查的 FES 误差。

## 测试的物理问题

采用无量纲坐标 x、y 和 kBT=1：

    U(x,y) = 4[(x²−1)²+(y²−1)²] + κ(x−y)²/2
    B(x,y) = a[u(x)+u(y)]

κ=0 和 2 分别提供独立与相关变量；a=1 或 1/2 分别代表 SUM 与 MEAN。
高斯偏势表故意不对称，让错误的概率校正能够被观察到。

参考概率由二维 Gauss–Legendre 积分计算：每个直方图单元分别积分，
8 与 16 阶结果交叉检查，积分区域由 [-3,3]² 扩大到 [-4,4]² 再检查尾部。
FES 对照使用相同单元的概率，避免把直方图概率与单元中心的势能混为一谈。

每组用 8 条独立 Metropolis 链，初始位置覆盖四个正负号组合。
每条链丢弃 20000 次预热，随后每 8 个 sweep 保留一帧，拒绝移动也保留驻留。
这是平衡构型采样；sweep 不代表 MD 时间，结果不用于成核速率或动力学路径判断。

## 实际复用的生产代码

- HillBias 负责高斯求值、导数、HILLS 写入与恢复。
- 独立训练时，两个中心每 200 sweeps 各沉积高度 0.025 的 hill，sigma=0.45，
  共同写入同一份历史。共训练 24000 sweeps；两个中心使用相同沉积步号。
- 输出 METAD_TOY_SHARED_FROZEN 1 格式，769 个点覆盖 [-3,3]，间距 1/128。
- CPU 采样使用基准程序中的三次 Hermite 表参考；其值、导数和端点延拓
  与实际 CUDA FrozenSharedBias 在 256 组双中心输入上逐点核对。
- CUDA 执行器目前提供 MEAN。SUM 在统计原型中通过总能量和导数的整体缩放构造。
- 两张独立学习的表均测试 HILLS 参数/记录、求值和导数的逐位恢复，
  导出表文件逐字节一致，并检查同一 RNG 种子下整条冻结采样链逐位重现。
- FES 使用工具目录现有 create_reweighted_fes，显式 static-bias，
  对每帧完整 B 重加权。导出的二维密度与独立 numpy 统计逐单元核对。

主程序仍使用冻结表。这里的共同沉积循环属于独立验证程序。

## 构建和运行

从仓库根目录执行。Python 依赖沿用 tools/requirements-analysis.txt。

纯 Host 统计测试：

    cmake --preset host-tests
    cmake --build --preset host-tests --target zqc_shared_probability_probe -j 4
    .venv-analysis/bin/python benchmarks/shared_frozen/probability.py \
      --library build/host-tests/benchmarks/graph/libzqc_shared_probability_probe.so \
      --output build/shared-probability-host

加上 CUDA 生产求值器交叉验证：

    cmake --preset graph-tests
    cmake --build --preset graph-tests \
      --target zqc_shared_probability_probe zqc_shared_probability_cuda -j 4
    .venv-analysis/bin/python benchmarks/shared_frozen/probability.py \
      --library build/graph-tests/benchmarks/graph/libzqc_shared_probability_probe.so \
      --cuda-library build/graph-tests/benchmarks/graph/libzqc_shared_probability_cuda.so \
      --output build/shared-probability

默认每条链保留 60000 帧，共 12 组。输出目录必须新建。
这套统计测试显式启动，不加入每次安装的快速 CTest。

针对强偏势 SUM 的采样不足，保持判据不变，将两组的链延长 8 倍：

    .venv-analysis/bin/python benchmarks/shared_frozen/probability.py \
      --library build/graph-tests/benchmarks/graph/libzqc_shared_probability_probe.so \
      --cuda-library build/graph-tests/benchmarks/graph/libzqc_shared_probability_cuda.so \
      --strong-only --samples 480000 --output build/shared-probability-strong

两次使用相同 seed 规则；长链包含原短链作为前缀。
重新分析已有样本可使用 --reuse-samples，--samples/--replicates 必须与原样本匹配。
该模式验证表 SHA256 和每帧偏势后重建报告，保留原始 samples.npz。
强偏势子集重分析还须传入 --strong-only。

## 判据与结果解释

预先固定的通过条件记录在 probability.py 的 CRITERIA 和 summary.json：

- 六个观测量：P(x>0)、P(y>0)、P(xy>0)、〈xy〉、〈x²〉、〈y²〉。
- 观测量最大绝对误差 <0.035，独立链均值的最大标准误 <0.010，
  各项误差 <4 个标准误 +0.003。
- 二维概率的总变差距离 <0.035。
- FES 使用每格权重 ESS≥60 的单元，且这些格必须覆盖参考概率质量的 98% 以上。
- 在上述覆盖区间，对齐任意加法常数后，按参考概率加权的 FES RMSE <0.12 kBT。

误差条由独立链估计，同时输出每链 8 块的分块统计以及半长度结果。
权重 ESS 只反映权重集中程度，不是扣除时间相关性后的有效样本数；
FES RMSE 的覆盖阈值与独立链误差检查须同时满足。

错误对照包括不加权和只使用 a*u(x) 加权。在相关强偏势组，
二者对 P(x>0) 的误差必须大于 0.05，否则测试未有效区分错误实现。
变量独立时，仅用本变量的偏势有时确实能恢复其边缘分布，因此必须有相关组。

summary.json 给出每项判据、原始/正确/错误对照统计、输入源码和二进制哈希。
每组保存 samples.npz、histograms.npz；两个代表组另存 CLI CSV、预览图、权重审计。
返回码 0 表示全部选中配置通过，1 表示未满足判据或程序异常。
采样不足也会返回失败；不会因曲线看起来平滑就判定通过。

这套测试不证明持续沉积时的重加权、任意原子数的效率，或 Al3Sc 成核 FES 已收敛。
后续两步分别验证在线时间依赖和真实 Al3Sc 体系。

实测结果见 [310 统计验证记录](PROBABILITY_RESULTS_310.md)。

连续动力学见 [KINETICS.md](KINETICS.md)；直接坐标沉积、冻结、重新平衡后，
仅读取 CV 和完整总偏势的重建见 [FES_RECOVERY.md](FES_RECOVERY.md)。
