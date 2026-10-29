# 轨迹与 Distance 数据集：310 验证

日期：2026-10-25。主机 excelsior（SSH 别名 310_right）。
Python 3.10.6，NumPy 1.23.4；CPU 离线验证，无 Slurm、无新 MD 运行。
验证目录：/home/saitang/work/MetadToy-gpumd-20261023/build/trajectory-20261025-validation。

结果：28 项测试全部通过，无跳过，首轮用时 2.551 秒。
其中原有 HILLS 11 项、新增轨迹/数据集 17 项。原有生产 C++ HillBias 探针被
显式加载，继续核对 1/2/3/8/32/128 维的三种 hill 模式。

新增验证包含：
- LAMMPS ID 行重排、超过 2^53 的 ID/步号、gzip；
- restricted triclinic 包围盒反解、scaled 坐标和 extxyz 同几何结果；
- general triclinic 原点/向量、unwrap 坐标与显式 PBC；
- 两种斜盒、三种 PBC 组合，共 180 个随机位移，与独立周期像枚举交叉核对；
- 变胞、多源、缺失 step/time、扩展 XYZ 重排 Properties；
- 固定行序确认及 ID/类型改变，缺失 PBC 和冲突 PBC；
- 截断、NaN、重复 ID/字段、奇异盒、倒序步号/时间等错误；
- CSV/JSONL/manifest 对应、时间块划分与 gap、stride；
- 源文件变化与输入/输出冲突、失败保留已有输出、单位冲突、CLI 返回码。

实际 GPUMD 文件烟雾验证：
使用此前 O2 运行的 build/gpumd-o2-seeded-2m/seed_73517/model.xyz。
这是一个两原子初始结构，不是生产多帧轨迹。输出 r_OO=1.2100000000000009 angstrom；
step/time 保持空值。它验证实际 GPUMD 文件兼容性，多帧场景由受控测试覆盖。

范围：
这里只实现离线 Distance 数据集。未开展新的采样、训练、COLVAR 重加权或模型上线。
斜盒欧氏最近像已独立验证，尚未与 LAMMPS domain 的斜盒选像约定逐例交叉核对。
