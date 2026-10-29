#pragma once
// GPUMD 周期显式镜像邻居提供者：独立于 LAMMPS NeighHub，不修改 GPUMD ABI。
// 布局约定与 EngineAdapter 一致：owned 前缀 [0,N)、ghost 后缀 [N,total)；
// AtomId = owner_row+1；选择集 flags 按 owner 成员资格覆盖镜像行。
// 父适配器在 begin_evaluation 之后、图准备选择集/布局之前，于 transfer_stream 上
// 调用一次 update；消费者经 ArrayView::wait_on 等待本提供者的 record_write。
#include "zqc_engine.h"
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <cuda_runtime.h>

namespace MetaD_zqc {

// 单进程正交盒上的 FULL/HALF CSR 邻居缓存，带 Verlet skin 与 GPU 位移检查。
// 配置阶段可 define_selection / register_request；active() 当且仅当已登记请求。
class GpumdNeighbors {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    explicit GpumdNeighbors(Logger logger = {});
    ~GpumdNeighbors();

    GpumdNeighbors(const GpumdNeighbors&) = delete;
    GpumdNeighbors& operator=(const GpumdNeighbors&) = delete;

    // 已登记至少一个 NeighborRequest 时为 true；父层通常仅在 active 时调用 update。
    bool active() const;
    // 默认 0.3；要求有限且 >= 0。变更后下次 update 强制重建。
    void set_skin(double skin);

    // 以稳定名称登记 owned 行选择集；返回适配器内 SelectionId（非 all_atoms）。
    // owned_rows 为输入坐标行号；镜像成员在首次/重建 update 后补入。
    SelectionId define_selection(const std::string& name, const std::vector<int>& owned_rows);
    SelectionId selection_id(const std::string& name) const;

    // 登记并校验请求；ghost_centers 显式拒绝（仅 owned 中心，供 StruFactor 等）。
    // 返回提供者句柄，供 list(handle) 取视图；身份变化使缓存失效。
    int register_request(const NeighborRequest& request);

    // 在给定流上消费 MD 坐标就绪依赖，重建或复用镜像+CSR，并 record_write 发布坐标/类型。
    // positions/types 须可 validate；types 长度与原子数一致。evaluation 写入 NeighborListView。
    void update(ArrayView<const double> positions, ArrayView<const int> types,
                const std::array<double, 3>& box, const std::array<int, 3>& periodic,
                cudaStream_t stream, std::uint64_t evaluation);

    int total_atoms() const;
    // Cuda：扩展后 AoS；Host：仅在显式请求时下载整包扩展坐标。
    ArrayView<const double> positions(MemorySpace memory);
    ArrayView<const int> types(MemorySpace memory);
    // Host：每 localid 的稳定 AtomId（= owner+1，镜像与 owner 相同）。
    ArrayView<const AtomId> ids() const;
    // 每 localid 的 owner 行号；Host/Cuda 均可。
    ArrayView<const int> owners(MemorySpace memory) const;

    SelectionView selection_members(SelectionId id);
    ArrayView<const int> selection_flags(SelectionId id, MemorySpace memory);
    NeighborListView list(int handle);
    std::uint64_t rebuild_count() const;
};

} // namespace MetaD_zqc
