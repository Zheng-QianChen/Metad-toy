# 普通 LAMMPS 构建与兼容验收

普通 LAMMPS 是正式支持的宿主路径。2026-10-29 使用全新构建的 LAMMPS 22 Jul 2025，
显式关闭 Kokkos、GPU package 和 OpenMP package，验证当前统一版插件。
本页的“普通版”指 CPU LAMMPS 宿主加 CUDA MetaDtoy 插件；当前 CV 内核仍需要 CUDA GPU。
完全没有 CUDA 的 CPU CV 后端尚未实现。

## 两条邻居路径

Kokkos 宿主通过设备原生邻居表准备设备 CSR；普通宿主读取 LAMMPS Host 邻居表，
整理 CSR，在内容变化时上传到 GPU。两者经 EngineAdapter 的同一 view 接口提供给 CV。
LocalQL 的行边界映射和逐边球谐缓存删除适用于两条路径。

普通版每次求值上传当前坐标；候选 CSR 按拓扑和选择集版本更新。
有效距离、截断和开关权重仍在每次求值时重新计算。J 的 ghost 行和两层邻域由普通版
适配器准备；偏置力及 virial 通过 Host 暂存、普通 LAMMPS 通信和写回路径提交。
其数据传输成本与 Kokkos 不同，A100/WSL 的 Kokkos 加速百分比不能直接套用到普通版。

## 构建入口修正

`benchmarks/integration/build.py --profile host` 显式设置
`ZQC_LAMMPS_KOKKOS=OFF`、`ZQC_LAMMPS_WEAVE=OFF`，不再要求 Weave 路径或传入 Kokkos_DIR。
`--profile kokkos` 同样不要求 Weave；只有 `--profile weave` 必须提供 Weave 源码和库。
验证脚本的 `--profile host` 不传入 `-k on`、`-sf kk` 或 Weave 插件。
具体命令见 [集成指南](../benchmarks/integration/README.md)。

使用普通宿主实际构建的 MPI、整数大小和匹配源码头文件；插件应重新构建到独立目录。
本轮为 x86-64、GCC 11.4、OpenMPI 5.0.6、CUDA 12.6、RTX 4080 SUPER/SM89。
普通 LAMMPS 启用共享库、PLUGIN、MANYBODY 和 MEAM，不加载 Weave。

Python 测试第一次受 DeepMD Python 可执行文件的 RPATH 影响，误载环境内的另一份 liblammps，
在实例初始化时报告缺少 lammps_expand。该失败原样保留。修正方式为只在本轮 build 目录
复制匹配的 Python package，并在 package 内显式链接本轮 liblammps.so；未修改系统安装。
最终实例实测 `has_package('KOKKOS') == False`、`has_package('GPU') == False`，
版本为 20250722。动态链接检查没有 Kokkos/Weave 库或缺失依赖。
普通宿主的 CUDA CV 计算仍依赖 CUDA runtime，此处没有将它表述为 CPU-only 插件。

## 验收结果

- LocalQL 21 项、Q 9 项、Q/LocalQL 混合图 6 项通过：覆盖 1/2/4 MPI ranks、独立 CPU
  前向参考、坐标/应变有限差分、不同中心组、径向权重、孤立原子、空 rank 和邻居排除。
  逐原子值最大误差 3.38e-15，图输出最大误差 5.00e-16，跨 rank 布局最大误差 2.23e-16。
- ghost/cache 8 项通过：2/4 ranks 下检查跨 cutoff、动态组、NPT 和三斜盒、延后请求、
  排序/迁移与 skin 改变；独立物理邻接枚举与冷重建对照验证缓存。
- 混合 CV 导数 4 项通过：1/2 ranks，包含 Q、LocalQL、结构因子、化学/加权 CV、CSS
  的非线性图，以及 Graph DISTANCE；使用两种差分步长。
- 真实 Al–Sc MEAM 集成 6 项通过：1/2 ranks 的 LocalQL×CSS 固定共享势、门控、单中心
  空 rank 与实际重启；包括坐标/应变差分和逐原子应力检查。
- 共享在线 WT 8 项通过：SUM/MEAN × 1/2/4 ranks，以及 4 ranks 单中心测试。
  同时检查每 2 步沉积、run 0/write_restart 不重复沉积、独立表能量和续算。
  续算原子状态最大差 1.89e-15，历史最大差 1.39e-17。
- 原生普通 `lmp` 可执行文件以 2 MPI ranks 直接加载插件，完成 10 步 MEAM + LocalQL×CSS
  和 restart 输出，退出码为 0；该检查不依赖 Python 驱动 MD。

所有通过均为上述实际测试范围，未新增普通版长运行或性能结论。
真实 Al3Sc 形核/FES 科学验收的开放范围仍见 [发布说明](RELEASE_20261028.zh.md)。

## 后续维护与证据位置

邻居、缓存、通信和力/virial 接口的后续修改，需同时检查普通 Host、无 Weave Kokkos、
Weave 三个构建路径。至少覆盖单 rank、多 rank、空中心 rank、坐标/应变导数和续算；
不能用 Kokkos 关闭运行选项的结果代替完全无 Kokkos 宿主的构建检查。

本轮运行目录：`/home/zqc/work/Metad-toy-unified-20261028/build/plain-lammps-20261029`。
普通宿主位于 `lammps/`，插件位于 `plugin/fix_crystallize_plugin.so`，匹配 Python 包位于 `python/`。

Windows 证据：
`F:/Users/Administrator/Documents/ChatGPT/Work_MetaD/artifacts/metadtoy-plain-lammps-20261029`。
其中保存构建/运行脚本、analysis.json、源码快照、原始日志与二进制哈希；旧候选版归档未覆盖。
