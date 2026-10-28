# 旧独有 CV 功能恢复与验收

本轮对应用户指定顺序 **5 → 3 → 4 → 2 → 1 → 6** 的第 5 项。
基于隔离分支 `codex/metadtoy-unified-20261028` 的 `3f7764a` 实现；没有切换生产作业。

## 恢复内容

- LocalGeom 的 `n_i=sum_j sigma(r_ij)`、软配位总数、平均配位数与高斯目标配位评分。
  新 `LOCAL_GEOM ... OUTPUT COORD|TET` 叶通过当前引擎的 Full 设备邻居表工作，
  包含按类型筛选、径向阈值、力和六分量 virial。无需两跳 ghost 中心行。
- Graph `SUM(expr)`、`.SUM`、`safe_div(a,b,epsilon)`，恢复 NSOLID 与加权 MEAN_SOLID。
  CVGraph 从输入推断唯一中心组；多组表达式必须指定 `SUM(expr, raw_family_symbol)`。
  设备掩码同时限制值和反向种子，避免 fermi(0) 把公共数组的组外补零计入总数。
- SUM 全局空集为 0；AVE 原有空集报错语义不变。safe_div 在 `|b|<=epsilon`
  时值及两路导数为 0，epsilon=0 对应旧 MEAN_SOLID 的恰好零分母约定。
  避免除数平方下溢造成错误无穷导数；Global Leaf 广播的伴随由执行器跨 rank 汇总。
- MEAN_RNN 提供 rnn/rnn_valid 诊断：先对每个中心加权平均邻距，再对有效中心平均。
  仍不能作为可偏置的 Graph 叶。

使用及迁移见 [LocalGeom](manuals/CVs_LocalGeom.zh.md) 和 [Steinhardt](manuals/CVs_Steinhardt.zh.md)。
恢复功能不代表恢复含糊的旧语法；旧 AVE/NSOLID 等 CV 内部轴继续提示迁移到 Graph。

## 验收

运行宿主：WSL RTX4080SUPER，LAMMPS 22Jul2025；与上一轮一致的 CUDA/MPI/Weave 安装。
以下为最终实现的验证，日志根目录为 `build/restoration-20261028/`：

| 验证 | 结果 | 日志子目录 |
| --- | --- | --- |
| LocalGeom SUM，8 场景 × 1/2/4 rank | 24/24 | localgeom-sum-final |
| LocalQL NSOLID/MEAN_SOLID，6 场景 × 1/2/4 rank | 18/18 | localql-statistics-final |
| 显式多组 SUM，变化中心组及差分 × 1/2/4 rank | 3/3 | explicit-family |
| Host 适配插件，COORD/TET/空集 × 1/2 rank | 6/6 | localgeom-host-final |
| Kokkos 无 Weave 插件，同上 | 6/6 | localgeom-kokkos-final |
| 完整 CTest（含 SUM/safe_div 的 1/2/4 rank） | 37/37 | ../integration-weave/Testing/Temporary/LastTest.log |

LocalGeom 包含独立 CPU 公式、周期边界、类型筛选、排除、孤立原子、全局空组、空 rank、
选择集更新、迁移和 NPT。坐标力差分最大约 4.8e-11；六分量应变差分最大约 1.1e-6。
LocalQL 旧统计的值误差约 7.2e-15 内，坐标力差分约 1.5e-12 内，应变差分约 8.3e-9 内。
另有加权均值嵌套 SUM 的独立梯度差分、零/近零分母、极小非零分母及广播 Global 伴随测试。

初轮失败日志保留：LocalGeom SUM 夹具曾把 GAUSSIAN 的 height 当作 sigma 缩放；
LocalQL 总数轴曾沿用均值轴的窄 sigma，位移后偏势已截为 0，不能检查非零力。
修正夹具的 sigma/轴宽后使用新目录完整复跑，未放宽最终正确性判据。
此前普通 Host/设备 LocalGeom AVE 的 21 组通过记录作为补充保留，不重复计入上表。

## 边界

本轮没有完成共享偏势生产 SUM 或在线 WT 沉积；Graph SUM 与共享偏势归一化是不同接口。
SUM/AVE 规约仍使用已有 Host 求和边界，设备路径覆盖按后续第 4 项处理。
Global 广播仅支持 Leaf，不能把规约结果回灌逐原子融合段。
默认 WSL MPI/CUDA 退出诊断问题沿用上一轮记录；本轮未把常规回归成功当作新的完整 sanitizer 验收。
集群运行模板、生产切换与科学采样验收留在后续 1/6 阶段。
