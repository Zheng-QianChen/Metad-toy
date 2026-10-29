# 轨迹重加权 FES：310 验证

日期：2026-10-26；主机 excelsior（310_right），Python 3.10.6、NumPy 1.23.4。
验证目录：/home/saitang/work/MetadToy-gpumd-20261023/build/fes-20261026-validation。
全部验证在 310 CPU 上直接运行，不经过 Slurm，不启动新的 MD。

## 功能与默认值

- 新增 fes reconstruct COLVAR；hills preview --quantity fes 默认也调用轨迹重加权。
- method=hills 显式保留 F=-gamma/(gamma-1)*V；quantity=bias 行为保持原定义。
- 权重来源为 log_weight、rbias、bias-c(t)，或原生 WT HILLS 的离线 c(t) 积分。
- 原始动态 bias 缺少归一化时拒绝推断；static-bias 需要显式选择。
- 支持 1D/2D 边际直方图及高维已给权重的投影。原生 c(t) 积分支持 1–3D 普通 WT，
  PBMetaD 需要提供校正权重。输出 CSV/PNG/JSON 和按帧权重 CSV。

## 回归结果

45 项测试通过，无跳过，最终一轮用时 4.869 秒：
原有 HILLS 11 项、轨迹/数据集 17 项、新增重加权 17 项。
生产 C++ HillBias 探针显式加载，原有三模式六维度交叉核对继续通过。

新增验证覆盖：
- 已知自由能的离散重要性重加权恢复，以及分时段加减同一偏置零点的不变性；
- 对数权重达到 10000 时的稳定计算和解析 ESS；
- 1D/2D/3D c(t) 与独立直接积分参考；
- 同一步多 hill 的包含规则、丢弃早期样本仍使用完整历史；
- 零偏置、空 history、空 bin、周期边界、显示范围外权重比例；
- 二维轴顺序及对未绘图坐标的边际化；
- PB 使用已给权重，拒绝将普通 WT 的 c(t) 公式直接用于 PB；
- 无校正量报错、静态权重显式选择、多偏置列下优先使用完整 log_weight；
- int64 步号、CSV/无表头列映射、重复步/非有限值/截断、不等间隔；
- 默认路径、旧方法显式入口、输入输出冲突和输入变化时保留原输出。

## 已有 O2 轨迹

输入来自既有 GPUMD seed_73517 的 COLVAR 与 HILLS：
/home/saitang/work/MetadToy-gpumd-20261023/build/gpumd-o2-seeded-2m/seed_73517。

选取 step >= 200000，显示距离 1.05–1.60，120 bins。
c(t) 积分域使用原运行的偏置网格域 [0.85,3.0]，采用记录的 kBT/gamma。

- 选中 180001 帧；显示范围内 178966 帧。
- 120 个 bin 均有样本。
- 总权重 ESS：53193.89952833448；未作时间相关修正。
- c(t) 网格从 1024 加密到 2048 后，FES 最大差 6.661338147750939e-16 eV。
- PNG 已视觉检查；曲线与权重、参数随结果一起保存。

加密差异仅检查这条轨迹在该积分域内的离散化稳定性，不代表采样误差、c(t)
准平稳近似误差或域截断误差。未施加径向熵修正，输出是距离坐标 FES，不是 Morse
势能曲线。本轮没有进行新的 MD，也没有将结果与真实势能作精度验收。

## 复现

从仓库根目录，在已安装 tools/requirements-analysis.txt 的 Python 环境中：

    METADTOY_BIAS_REFERENCE=/absolute/libbias_reference.so \
        python -m unittest discover -s benchmarks/cli -p 'test_*.py' -v

O2 默认方法：

    python tools/metadtoy.py hills preview /path/to/HILLS --quantity fes \
        --colvar /path/to/COLVAR --out analysis/o2-reweighted \
        --from-step 200000 --range 1 1.05 1.60 --bins 120 \
        --ct-range 1 .85 3.0 --ct-bins 1024 --energy-unit eV

将 --ct-bins 改为 2048，使用另一个输出前缀，即可做本报告的加密检查。
日志见 fes_tests_310.log；接口与适用条件见 ../../tools/README.md。
