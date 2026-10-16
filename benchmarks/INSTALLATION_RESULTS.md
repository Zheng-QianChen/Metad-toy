# 安装入口发布验证 — 2026-10-16

本轮使用 Ubuntu 22.04/WSL、LAMMPS 29 Aug 2024、CUDA 12.6、RTX 4080 SUPER（sm_89）及当前 LAMMPS 配套 MPI。插件摘要和入口脚本摘要见 [installation_results.json](installation_results.json)。插件与上一轮 DISTANCE 验证相同，本轮重点验证用户入口和可复跑性。

| 入口/检查 | 结果 | 内容 |
|---|---|---|
| smoke，1/2/4 rank | 12/12 | 全部现有非 ML CV 的组合、非零偏置力、有限数值、MPI 状态对照 |
| 独立复制插件 | 4/4 | 插件/输出路径含空格，从 `/tmp` 启动，标准库 Python，无插件旁 CMakeCache |
| regression，1/2/4 rank | 69/69 | 所有场景统一四步长 `1e-4,3e-5,1e-5,3e-6`；最后两个步长均通过 |
| 线性叠加对照 | 9/9 | 单 CV 梯度加权和与组合图反向一致 |
| graph | 19/19 | 空目录重新配置/编译后执行所有 CTest |
| o2 | 通过 | 三条 2 ns Morse 轨迹、盲重建、覆盖和误差验收 |
| 外部程序失败 | 通过 | 用 `/bin/false` 模拟启动后失败，4 项标记 FAIL，入口退出码为 1 |

O₂ 平均重建曲线在 1.05–1.60 Å 的 RMSE 为 1.36 meV，最大误差 4.13 meV；测试按每条曲线而非仅平均值检查误差阈值。

原始日志分别在本机 `build/install-smoke`、`build/install-portable smoke`、`build/install-regression`、`build/install-graph`、`build/install-ctest` 和 `build/install-o2`。这些路径只记录本次验证位置；用户运行时由自己的 `--output` 和 `--build-dir` 决定。

本次仅验证上述硬件、CUDA 与 MPI 组合。测试入口不预设作者机器目录或 OpenMPI 绑核参数；其他安装环境以其实际运行结果为准。历史性能数据不作为这里的通过阈值。
