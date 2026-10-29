# 离线 CLI 验证

测试覆盖原生 v1/v2/v3、旧格式及 int64 步号、逐行校验、CSV 精度、不同轴顺序、
3D 切片、WT 缩放、PB 核截断/零点、分块等价、无依赖 inspect/export 和输出保护。

在指定验证机器上运行：

    python -m pip install -r tools/requirements-analysis.txt
    python benchmarks/cli/test_cli.py

可构建一个小型 CPU 探针，交叉核对生产 C++ HillBias：

    g++ -std=c++17 -O2 -shared -fPIC -pthread -Iinclude/utils \
        benchmarks/cli/bias_reference.cpp src/sampling/zqc_hill_bias.cpp \
        -o /absolute/validation/libbias_reference.so
    METADTOY_BIAS_REFERENCE=/absolute/validation/libbias_reference.so \
        python benchmarks/cli/test_cli.py

交叉测试覆盖 1/2/3/8/32/128 维、混合周期轴、完整 HILLS、PB SmoothGaussian 与
PLUMED_2_10 核。没有提供探针时该项明确标记 skip，不能将其计为通过。
数值核对和 CLI 测试在 310 上通过 SSH 直接运行；不需要 Slurm 或 GPU。


## 轨迹与数据集测试

    METADTOY_BIAS_REFERENCE=/absolute/validation/libbias_reference.so \
        python -m unittest discover -s benchmarks/cli -p 'test_*.py' -v

test_trajectory.py 包括流式 LAMMPS/XYZ 读取、斜盒最近像的独立枚举、原子身份、
单位、数据集元数据和时间块划分。受控测试可在 CPU 上完成；没有 CUDA 依赖。
最新记录见 TRAJECTORY_RESULTS_310.md 和 trajectory_tests_310.log。
O2 的实际文件检查使用旧运行的初始 model.xyz；不应将其解读成新跑的多帧 MD 验证。


## 轨迹重加权 FES

test_reweight.py 纳入上面的 unittest discover 命令。
覆盖 c(t) 的 1–3D 独立积分、解析权重、投影与默认入口。
本轮 45 项通过及已有 O2 COLVAR 的实际重建记录见 FES_RESULTS_310.md；
完整日志为 fes_tests_310.log。
