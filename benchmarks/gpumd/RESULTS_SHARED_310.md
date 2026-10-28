# GPUMD 动态库验证（310，2026-10-23）

所有编译与 GPU 验证通过 SSH 在 310 直接运行，未使用 Slurm。

- 首次构建生成 `bin/gpumd_metadtoy` 和 `lib/libmetadtoy.so.1`。
- 修改运行库源码后，仅构建 `metadtoy`；GPUMD 文件 SHA-256 保持不变。
- Engine、公开 C ABI、真实 GPUMD Distance smoke、现有生产 CV View 四项测试全部通过。
- 单独配置 `ZQC_BUILD_GPUMD=OFF`、`ZQC_BUILD_GPUMD_RUNTIME=ON`，并把
  `ZQC_GPUMD_SOURCE` 指向不存在的位置：运行库独立构建和安装成功。
- 将原可执行文件复制到新安装目录后，Distance smoke 再次通过，验证相对路径部署。
- 公开头文件通过 C11 编译检查；共享库仅导出 C 入口 `zqc_metadtoy_get_api`。

真实 GPUMD 数值结果：

- 非线性图及共享原子图的能量、力、virial 最大解析误差：8.88e-16。
- 总势能中心差分的力最大误差：6.98e-9 eV/Å。
- 2000 步、18 次沉积的 HILLS 与此前静态版本同种子结果逐字节一致，
  SHA-256 为 `36c53f8a8816fc31669f09d18e3112941a19ca23808db83888e518aa6d61bd70`。

本轮验证覆盖部署方式和既有 DISTANCE 行为。StruFactor 近邻接入尚待完成；
本轮未重跑长程 FES，也不把短程耗时作为性能对比。详细数值及可执行文件
哈希见 [RESULTS_SHARED_310.json](RESULTS_SHARED_310.json)。

310 验证目录：`/home/saitang/work/MetadToy-gpumd-20261023/build/gpumd-runtime-package`。
