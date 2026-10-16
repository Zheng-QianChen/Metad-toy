# DISTANCE 修复后的验证

插件已重新构建，SHA-256 为 `2392cf3855a95ce6f66ac0c9f01ec1ebde8772dd73bfb46e0a01f45686034806`。精选数值见 [distance_results.json](distance_results.json)。[旧报告](RESULTS.md)保留修复前的失败证据。

## 实现变化

- 输入是两个正的、互不相同的 LAMMPS 全局 atom ID，采用 `tagint` 读取。
- 每轮按 tag 查找 owned 原子，打包两组 `(x,y,z,owner_count)`，一次 MPI Allreduce 得到一致坐标并校验 owner 唯一性。
- 使用 LAMMPS minimum_image 处理周期边界和倾斜盒；零距离及非有限距离报错。
- 力写入 owned 原子，六分量 virial 两端各保存一半；迁移和排序后重新定位。
- Graph 通过 Host Global Leaf 读取标量和回传标量 adjoint，支持 `.value` / `.RAW`。旧 LEGACY 的 `.DISTANCE` / `.value` / `.RAW` 继续工作。

## 验证结果

- 全套 23 场景 × 1/2/4 rank：**69/69 通过**，含单 CV、线性/非线性组合，以及 DISTANCE 的 LEGACY、Global Graph、双距离组合、与 `AVE(STRU_FACTOR)` 的混合图。
- 初次三步长扫描为 66/69；混合图的应变偏置能差分有明显截断误差。增加 `h=3e-6` 后，保持原容差，两个最小步长均通过。其 virial 最大绝对误差从 `h=1e-4` 时的 `3.399e-3`，降至 `h=1e-5` 的 `3.399e-5`，再降至 `h=3e-6` 的 `3.059e-6`，符合中心差分的二阶收敛。
- 纯 DISTANCE Graph 在 `h=1e-5`：CV 坐标导数最大误差 `2.23e-11`，CV 应变导数 `4.49e-11`，偏置力 `2.58e-9`，偏置 virial `2.20e-8`。
- 9 组线性叠加对照通过；46 组 MPI 的 CV/偏置能/owned 力/virial 状态对照通过，最大差异 `6.89e-15`。
- `distance_edges.py`：非连续 ID 11/901、倾斜盒跨周期、连续迁移/排序、非周期边界，3 种 rank 数共 12 组；同时覆盖无 owned 原子的 rank。最大值/力/virial 误差 `4.44e-16`。
- `distance_invalid.py`：相同 ID、缺失 ID、零距离，在 1/4 rank 下均按预期终止，共 6 组。
- 独立 Graph CTest：**19/19 通过**。

原始目录为 `build/finite-difference/distance-fixed`、`distance-mixed-refined`、`distance-invalid`；CTest 日志为 `distance-ctest.log`。新的默认差分步长包括 `3e-6`，结果仍要求最后两个步长同时通过。

## 复跑

```bash
python benchmarks/finite_difference/run.py --output build/distance-fd-new \
  --cases distance distance_linear distance_graph distance_graph_linear distance_mixed
mpirun -np 4 python benchmarks/finite_difference/distance_edges.py \
  --plugin build/graph-lammps/fix_crystallize_plugin.so \
  --probe build/distance-fd-new/bias_probe.so --output build/distance-edges-new
python benchmarks/finite_difference/distance_invalid.py \
  --plugin build/graph-lammps/fix_crystallize_plugin.so --output build/distance-invalid-new
ctest --test-dir build/graph --output-on-failure
```

最小镜像切换面处距离梯度不连续，零距离处方向未定义；当前差分夹具位于光滑区域。
