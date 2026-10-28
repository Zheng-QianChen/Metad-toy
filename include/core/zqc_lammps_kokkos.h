#pragma once
#include "zqc_array_view.h"
#include <memory>
namespace LAMMPS_NS { class LAMMPS; class Fix; }
namespace MetaD_zqc {
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
    // MPI reverse 后的 owned 增量上传并加到已有力上，完成后标记 Device 为最新。
    void add_forces(const double* values, size_t count);
    void add_force(int local, const double values[3]);
};
} // namespace MetaD_zqc
