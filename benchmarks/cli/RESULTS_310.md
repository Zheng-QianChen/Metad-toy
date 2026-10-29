# CLI 首版验证（310，2026-10-25）

## 环境

- 主机：310 / excelsior，通过 SSH 直接运行。
- Python 3.10.6、NumPy 1.23.4、Matplotlib 3.6.2。
- 使用现有 mambaforge3 的 Python 创建独立 venv310，复用已有数值/绘图包。
- 验证目录：/home/saitang/work/MetadToy-gpumd-20261023/build/cli-20261025-validation。
- 源码功能入口：tools/metadtoy.py；具体命令见 tools/README.md。

## 正确性

11 项测试全部通过，包含生产 C++ 交叉检查，无跳过项：

- 原生 HILLS v1/v2/v3 与旧格式；大于 2^53 的 int64 步号保持精确。
- CSV 浮点保存、逐记录 sigma、截止步号筛选和分块求值。
- 1D/2D 图片、轴顺序、3D 显式切片和 WT FES 缩放。
- PBMetaD 核截断、周期核及 mean/sum 零点约定。
- 非法数值、不完整尾行、倒序记录、未知 PLUMED 文件和输出覆盖保护。
- inspect/export 在 Python -S 下运行，验证标准库路径。
- C++ HillBias 对照覆盖 1/2/3/8/32/128 维，每种维数检查完整 HILLS、
  PB SmoothGaussian 和 PB PLUMED_2_10，共 18 种配置，含混合周期轴。
  使用 rtol=2e-12、atol=2e-14 的判据，全部通过。
- 另核对 hills preview --help 可正常展示参数。

## 真实沉积文件

读取此前 GPUMD O2 验证的 gpumd-o2-seeded-2m/seed_73517/HILLS：

- 19,800 条沉积，步号 20,100 至 2,000,000。
- 原生 v3、1D、WT gamma=20、sigma=0.015。
- 输出完整沉积 CSV，以及 1.05–1.60 区间的 250 点 FES CSV/PNG/JSON。
- 图中能量单位标注 eV；数值使用 -gamma/(gamma-1)*V+C，并以所选网格最低点为零。
- 这是对既有 MD 结果的离线分析；所得径向自由能未施加 O2 坐标测度修正。

图片已检查，坐标、单位、标题可读。图片体现离线重建流程，收敛程度仍应由采样验证评估。

## 功能边界

第一版提供 CLI 与独立服务函数，后续 TUI 可以复用。当前支持 MetaD-toy 自有 HILLS，
未实现 PLUMED 原生 HILLS importer、通用 MD 轨迹读取或 ML 训练命令。PB histories
只输出组合偏势/切片；联合 FES 需要额外轨迹统计与正确的重加权方法。
解析 hill 求和与运行时网格插值存在数值表示差别，JSON 明确记录求值方式。
