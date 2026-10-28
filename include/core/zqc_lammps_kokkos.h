#pragma once
#include "zqc_array_view.h"
#include <memory>
#include <vector>
namespace LAMMPS_NS { class LAMMPS; class Fix; class NeighList; }
namespace MetaD_zqc {
struct KokkosNeighborView {
    ArrayView<const int> counts, neighbors;
};
// Kokkos ABI 仅存在于实现文件。借用视图保持宿主的 stride 和分配容量，
// 有效期到下一次原子排序/扩容；每次求值重新取得视图。
class LammpsKokkosBridge {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    explicit LammpsKokkosBridge(LAMMPS_NS::LAMMPS*, LAMMPS_NS::Fix*);
    ~LammpsKokkosBridge();
    bool active() const;
    void sync_metadata();
    void sync_host_positions();
    ArrayView<const double> positions(size_t count);
    ArrayView<const int> types(size_t count);
    KokkosNeighborView full_neighbors(LAMMPS_NS::NeighList*, bool ghost_centers);
    // MPI reverse 后的 owned 增量上传并加到已有力上，完成后标记 Device 为最新。
    void add_forces(const double* values, size_t count);
    void add_force(int local, const double values[3]);
    // Weave 设备合批：仅在 ZQC_LAMMPS_WEAVE 且 WEAVE_METAD_DEVICE_COMMIT!=0 时开启。
    // accumulate 返回前完成源 View 读取，调用方缓冲可立即复用。
    // cancel 排空设备队列并丢弃结果，不写回宿主力/virial。
    bool begin_force_batch(size_t total, bool need_virial);
    bool force_batch_active() const;
    void accumulate_force_virial(ArrayView<const double>, ArrayView<const double>);
    // Host/owned 增量上传到同一设备批次；后端选择不随各 rank 的 owned 贡献改变。
    void accumulate_host_force_virial(const double* forces, const double* virials, size_t total);
    // needs_reverse=false 时跳过 Weave reverse（owned-only）。
    // global&&!per_atom 时在设备归约全局 virial；per_atom 时下载 owned 六分量。
    void finish_force_batch(size_t owned, bool needs_reverse, bool global, bool per_atom,
                            double global_virial[6], std::vector<double>& atom_virial);
    void cancel_force_batch() noexcept;
};
} // namespace MetaD_zqc
