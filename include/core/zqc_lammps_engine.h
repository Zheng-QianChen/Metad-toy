#pragma once
#include "../utils/zqc_engine.h"
#include <memory>
namespace LAMMPS_NS { class LAMMPS; class FixMetadynamics; }
namespace MetaD_zqc {
// LAMMPS 数组适配器。普通运行持有 AoS 设备暂存；CUDA/Kokkos 运行借用
// 宿主 DualView 的坐标/类型，保留其 stride。Host 元数据用于组、ID 和邻居筛选。
// 候选邻居 CSR 按拓扑/组 generation 缓存；MPI reverse 后将 owned 力加回宿主。
// PImpl 将 LAMMPS、MPI、GpuBuffer 的实现依赖留在 .cu 文件。
class LammpsEngineAdapter final : public EngineAdapter {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    // 累计执行数据准备分支的次数，供测试核对缓存命中；空数组准备也计一次。
    struct SyncStats {
        std::uint64_t position_uploads = 0, type_uploads = 0, id_rebuilds = 0;
        std::uint64_t selection_scans = 0, selection_flag_uploads = 0;
        std::uint64_t neighbor_rebuilds = 0, neighbor_uploads = 0;
        std::uint64_t device_neighbor_rebuilds = 0;
        // 实际执行 ghost 空间搜索的次数；动态组重新筛选 CSR 单独计入 neighbor_rebuilds。
        std::uint64_t ghost_row_rebuilds = 0;
        // 仅统计输出力/virial：批量 CV 提交、统一回写和 reverse 流程的次数。
        // 一次 reverse 可能包含多个邻域交换阶段，计数不等于 MPI 消息数。
        std::uint64_t force_virial_submissions = 0, force_virial_flushes = 0;
        std::uint64_t force_reverse_calls = 0;
        std::uint64_t device_property_forwards = 0;
    };
    LammpsEngineAdapter(LAMMPS_NS::LAMMPS*, LAMMPS_NS::FixMetadynamics*, Logger = {});
    ~LammpsEngineAdapter() override;
    void begin_evaluation() override;
    // 由宿主 init 和 post_neighbor 回调通知。覆盖排序、迁移、ghost 重建，
    // 包括原子数和 timestep 都没有变化的 run 0。
    void invalidate_topology();
    SyncStats sync_stats() const;
    std::uint64_t evaluation_id() const override;
    int owned_atom_count() const override;
    int total_atom_count() const override;
    ArrayView<const double> positions(MemorySpace) override;
    ArrayView<const int> atom_types(MemorySpace) override;
    ArrayView<const AtomId> atom_ids() override;
    int atom_type_count() const override;
    int atom_type_begin() const override { return 1; }
    int atom_type_id(const std::string&) const override;
    SelectionId selection_id(const std::string&) const override;
    SelectionView selection_members(SelectionId) override;
    ArrayView<const int> selection_flags(SelectionId, MemorySpace) override;
    Logger logger() const override;
    int rank() const override;
    std::int64_t timestep() const override;
    void communicate_forward(AtomCommunication&) override;
    void communicate_reverse(AtomCommunication&) override;
    bool communicate_forward_device(ArrayView<double>) override;
    int register_neighbors(const NeighborRequest&) override;
    NeighborListView prepare_neighbors(int) override;
    void read_owned_atoms(const AtomId*, size_t, double*, int*) override;
    void sum(double*, size_t) override;
    void minimum_image(double[3]) const override;
    int reverse_communication_width() const override;
    // Graph 合批生命周期：Host 累加与可选 Weave 设备累加共用同一 begin/finish/cancel，
    // 禁止嵌套第二套批次。configure 在 Graph backward 前声明本步 virial 需求。
    // finish 后若设备已写入全局 virial，force_virial_batch_device_complete() 为真，Fix 勿再 v_tally。
    void configure_force_virial_batch(bool global_virial, bool atom_virial);
    bool force_virial_batch_device_complete() const;
    // 公共 Host 缓冲按 localid 保存 [total,3] 力和 [total,6] virial。
    // finish 将所有 CV 的 ghost 贡献合并回传，再一次性累加 owned 输出。
    void begin_force_virial_batch() override;
    void finish_force_virial_batch() override;
    void cancel_force_virial_batch() noexcept override;
    // 普通 LAMMPS virial 接口使用 xx yy zz xy xz yz。Full9 显式投影到这六项；
    // 非对称输入的 yx/zx/zy 仍保存在调用方数组中，当前六列 stress 输出仅包含选中的六项。
    void commit_force_virial(ArrayView<const double>, VirialView) override;
    void add_owned_force_virial(int, const double[3], VirialView) override;
};
} // namespace MetaD_zqc
