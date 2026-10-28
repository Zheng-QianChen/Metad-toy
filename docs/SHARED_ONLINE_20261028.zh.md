# 共享 SUM、在线沉积和历史验收

本轮完成顺序中的第 2 项。`SHARED_FROZEN` 支持生产 SUM/MEAN；新增 `SHARED_ONLINE`，
复用同一张 Hermite 表和 Graph 反向、设备力与 virial 通道。
使用说明与限制见 [共享偏势手册](manuals/SHARED_BIAS.zh.md)。

## 算法与持久化

能量规约 `SHARED_AGGREGATION` 与沉积规约 `SHARED_DEPOSIT` 独立。按用户选择，
沉积默认为 MEAN：一次事件的基础 hill 高度由全局 owned 中心数分摊。
WT 再按各中心在沉积前的同一张表上的势能衰减，不在中心之间顺序更新。
SUM 能量是全局广延量，LAMMPS `thermo_modify norm yes` 会再除体系原子数；
`extract_fix` 返回原始全局能量。

中心坐标、Gaussian 及导数计算留在设备；每次沉积只传递网格增量和标量，
通过 MPI 汇总后更新共享表。每个网格节点由一个 CUDA block 归约所有 owned 中心，
保留完整 Gaussian，不做尾部截断。普通步复杂度 O(N)，沉积步 O(N×网格节点数)。

历史记录配置与 CV 身份、步号、沉积序号、完整表值/导数和校验和。root 写入，
载入和错误广播到所有 ranks；拒绝损坏、截断、配置不符及晚于 MD restart 的历史。
重复 setup、run 0 和 write_restart 不会沉积。支持 PACE 0 冻结历史及导出固定表。
文件 flush 不等同断电事务；续算须匹配 MD restart 与其时刻的历史快照。

## 验证

- 独立 CUDA/MPI 测试在 1/2/4 ranks 通过，覆盖能量规约×沉积规约×WT 的 8 种组合、
  无中心 rank、解析 Gaussian、差分、同时 WT 更新、冻结/续算及异常历史。
  单个 rank 的非法中心或重复槽位会集体拒绝，历史不提交。
- 完整 CTest 40/40；新增在线测试纳入标准 graph 测试入口。
- 真实 Al–Sc MEAM/NPT 的最终矩阵 20/20：Weave 12 项，Kokkos 无 Weave 4 项，Host 4 项；
  覆盖 SUM/MEAN、1/2/4 ranks、单中心空 rank、Q+LocalQL+CSS、排序、run 0、thermo 归一化。
  连续运行与重启后重放的坐标/力最大差 7.55e-15，偏势能差 2.78e-17，
  virial 差 4.52e-17，体积差 0；历史表最大差 4.21e-17。
- SUM 固定势的真实 MEAM 10 步、坐标与应变有限差分通过。
- 在线历史导出的固定表在生产 CUDA 求值器与独立 Hermite 参考的 1,202 个点上对照；
  SUM 能量最大差 1.23e-16，导数最大差 3.71e-15；MEAN 同样通过。
- 在线路径 memcheck/initcheck 均 0 错误。WSL 单 rank 检查采用
  `OMPI_MCA_accelerator=null` 绕过已由裸 MEAM 复现的 MPI 退出问题；正常矩阵没有使用该覆盖。

运行环境：WSL Ubuntu 22.04、RTX 4080 SUPER、CUDA 12.6、OpenMPI 5.0.6、
LAMMPS 22Jul2025。多 rank 共用这一张 GPU，不充当实际多卡证据。

最终矩阵之前的失败日志保留：一个输入使用 Graph 不支持的裸数字字面量，改为 linear；
两个 thermo 检查分别误用 get_thermo 自定义键及同一步的缓存，修正测试读取后重跑完整矩阵。
实际生产 SUM 的 extensive 标志也已修正，并由最终矩阵验证。

## 证据与边界

源目录 `build/shared-online-20261028` 保存原始输入、日志、MD restart、历史、三个冻结插件。
Windows `Work_MetaD/artifacts/metadtoy-shared-online-20261028` 保存源码/证据归档、哈希清单、
分析脚本和逐项结果。历史和输入与源码同时版本化。

这些测试证明实现、导数和续算语义；不证明 Al3Sc 已发生形核或得到收敛自由能。
共享一体表的 WT 形状不能直接当作相关多中心联合 FES。长轨迹、多 GPU、跨节点与科学统计
仍由后续第 1、6 项单独验收。
