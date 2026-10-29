#pragma once
#include "../utils/zqc_engine.h"
#include <array>
#include <cuda_runtime.h>
#include <memory>

namespace MetaD_zqc {

// 单进程 GPUMD 本轮借用的 MD 数组与盒子描述。
// positions/forces 为 [atom, xyz]；virials 为原生 Full9，分量顺序
// xx yy zz xy xz yz yx zx zy；potential 为逐原子势能。
// 真实跨度由各 ArrayView 的 item/component_stride 保存，可描述 SoA 与 AoS。
// stream 是 MD 生产/消费流：bind_frame 在其上记录坐标就绪事件，
// finish 在适配器回写完成后让该流等待，避免 cudaDeviceSynchronize。
struct GpumdFrame {
    ArrayView<const double> positions{};
    ArrayView<double> forces{};
    ArrayView<double> virials{};
    ArrayView<double> potential{};
    // 纯 DISTANCE 路径可不提供；其余 CV 需要与原子数一致的类型表。
    ArrayView<const int> types{};
    std::array<double, 3> box_lengths{{0, 0, 0}};
    std::array<int, 3> periodic{{0, 0, 0}};
    std::int64_t step = 0;
    cudaStream_t stream = nullptr;
};

// 借用 GPUMD CUDA SoA（或任意合法 stride）力场数组的引擎适配器。
// 内部 AtomId = 行号+1（稳定 1..N）；GPUMD 前端负责把 0-based 索引换成该编号。
// 独立候选表持有 owned+周期像坐标；像的力/virial 按 owner 归回宿主原子。
class GpumdEngineAdapter final : public EngineAdapter {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    explicit GpumdEngineAdapter(Logger logger = {});
    ~GpumdEngineAdapter() override;

    // 校验正交盒、有限边长、形状/内存/原子数一致性，记录 MD 流完成事件。
    // 地址可在步间更换；调用方须保证上一轮消费者已结束对本帧数组的借用。
    void bind_frame(const GpumdFrame& frame);
    // 上传本轮 Host 贡献并累加到 MD 力/virial，
    // 将 bias_energy/N 加到逐原子 potential，再令 MD 流等待适配器完成事件。
    void finish(double bias_energy);

    // 选择集使用固定的 0-based 原子行号；周期像自动继承其 owner 的成员资格。
    SelectionId define_selection(const std::string&, const std::vector<int>& owned_rows);
    SelectionId selection_id(const std::string&) const override;
    SelectionView selection_members(SelectionId) override;
    ArrayView<const int> selection_flags(SelectionId, MemorySpace) override;
    // skin 是候选缓存余量；位移超过 skin/2 或盒子/请求变化时重建。
    void set_neighbor_skin(double);
    std::uint64_t neighbor_rebuild_count() const;
    void begin_evaluation() override;
    std::uint64_t evaluation_id() const override;
    int owned_atom_count() const override;
    int total_atom_count() const override;
    ArrayView<const double> positions(MemorySpace) override;
    ArrayView<const int> atom_types(MemorySpace) override;
    ArrayView<const AtomId> atom_ids() override;
    Logger logger() const override;
    std::int64_t timestep() const override;
    int register_neighbors(const NeighborRequest&) override;
    NeighborListView prepare_neighbors(int) override;
    // 仅按请求 ID 收集坐标：小缓冲 D2H，等待适配器私有流；缺失/越界 owner_count=0。
    void read_owned_atoms(const AtomId*, size_t, double*, int*) override;
    // 单进程固定顺序：规约空操作。
    void sum(double*, size_t) override;
    void minimum_image(double[3]) const override;
    // 接受已校验的 Host/Cuda 力与 6/9 分量 VirialView；等待就绪后按 stride 累加完整张量。
    void commit_force_virial(ArrayView<const double>, VirialView) override;
    // 收集 Host 贡献（Symmetric6 经 VirialView::full_component 扩成 Full9），finish 时统一回写。
    void add_owned_force_virial(int, const double[3], VirialView) override;
};

} // namespace MetaD_zqc
