#include "zqc_lammps_kokkos.h"
#include "zqc_cuda_execution.cuh"
#include "zqc_gpu_buffer.cuh"
#include "lammps.h"
#include "fix.h"
#include "atom_masks.h"
#include "atom_kokkos.h"
#include "kokkos.h"
#include <type_traits>

namespace MetaD_zqc {
namespace {
// setup() 期间 VerletKokkos 会启用 auto_sync。桥接自行维护 DualView 状态，
// 需临时关闭该兼容模式，避免 sync(Device) 将旧 Host 数组标成最新。
struct ExplicitSync {
    LAMMPS_NS::KokkosLMP* kokkos;
    int previous;
    explicit ExplicitSync(LAMMPS_NS::LAMMPS* lmp)
        : kokkos(lmp->kokkos), previous(kokkos->auto_sync) { kokkos->auto_sync = 0; }
    ~ExplicitSync() { kokkos->auto_sync = previous; }
};
__global__ void add_force_rows(double* force, size_t item_stride, size_t component_stride,
                               const double* increment, size_t count) {
    const size_t local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local < count)
        for (int axis = 0; axis < 3; ++axis)
            force[local*item_stride+axis*component_stride] += increment[3*local+axis];
}
__global__ void add_force_row(double* force, size_t item_stride, size_t component_stride,
                              int local, double x, double y, double z) {
    force[local*item_stride] += x;
    force[local*item_stride+component_stride] += y;
    force[local*item_stride+2*component_stride] += z;
}
}
struct LammpsKokkosBridge::Impl {
    LAMMPS_NS::LAMMPS* lmp;
    LAMMPS_NS::AtomKokkos* atom = nullptr;
    CudaExecutionStream transfer;
    GpuBuffer<double> increments;
    cudaEvent_t producer_ready = nullptr;
    // LAMMPS 29Aug2024 的 Kokkos kernels 使用默认 execution-space 实例。
    cudaStream_t producer_stream() const { return Kokkos::Cuda().cuda_stream(); }
    ViewReadiness ready() {
        execution_check(cudaEventRecord(producer_ready, producer_stream()));
        return cuda_event_readiness(producer_ready);
    }
    auto force_view() {
        atom->sync(LAMMPS_NS::Device, F_MASK);
        auto force = atom->k_f.view<LMPDeviceType>();
        static_assert(std::is_same<typename decltype(force)::non_const_value_type, double>::value,
                      "MetaD-toy requires double precision Kokkos forces");
        ready().wait_on(transfer.get());
        return force;
    }
    void finish_force() {
        execution_check(cudaGetLastError());
        // MPI/Host 消费和后续 Kokkos 积分均位于该完成边界之后。
        transfer.synchronize();
        atom->modified(LAMMPS_NS::Device, F_MASK);
    }
};
LammpsKokkosBridge::LammpsKokkosBridge(LAMMPS_NS::LAMMPS* lmp, LAMMPS_NS::Fix* fix)
    : impl_(new Impl) {
    impl_->lmp = lmp;
    if (!lmp->kokkos) return;
    if (!lmp->kokkos->kokkos_exists)
        throw std::runtime_error("MetaD-toy: Kokkos is not initialized");
    impl_->atom = static_cast<LAMMPS_NS::AtomKokkos*>(lmp->atom);
    execution_check(cudaEventCreateWithFlags(&impl_->producer_ready, cudaEventDisableTiming));
    // 所有混合 Host/Device 字段由桥接显式同步；ModifyKokkos 不再复制 ALL_MASK。
    fix->kokkosable = 1;
    fix->execution_space = LAMMPS_NS::Device;
    fix->datamask_read = EMPTY_MASK;
    fix->datamask_modify = EMPTY_MASK;
}
LammpsKokkosBridge::~LammpsKokkosBridge() {
    if (impl_->producer_ready) cudaEventDestroy(impl_->producer_ready);
}
bool LammpsKokkosBridge::active() const { return impl_->atom != nullptr; }
void LammpsKokkosBridge::sync_metadata() {
    if (!active()) return;
    ExplicitSync guard(impl_->lmp);
    impl_->atom->sync(LAMMPS_NS::Host, TYPE_MASK | MASK_MASK | TAG_MASK | MOLECULE_MASK);
}
void LammpsKokkosBridge::sync_host_positions() {
    if (!active()) return;
    ExplicitSync guard(impl_->lmp);
    impl_->atom->sync(LAMMPS_NS::Host, X_MASK);
}
ArrayView<const double> LammpsKokkosBridge::positions(size_t count) {
    ExplicitSync guard(impl_->lmp);
    impl_->atom->sync(LAMMPS_NS::Device, X_MASK);
    auto values = impl_->atom->k_x.view<LMPDeviceType>();
    static_assert(std::is_same<typename decltype(values)::non_const_value_type, double>::value,
                  "MetaD-toy requires double precision Kokkos positions");
    return {values.data(), count, 3, values.stride_0(), values.stride_1(),
            values.span(), MemorySpace::Cuda, impl_->ready()};
}
ArrayView<const int> LammpsKokkosBridge::types(size_t count) {
    ExplicitSync guard(impl_->lmp);
    impl_->atom->sync(LAMMPS_NS::Device, TYPE_MASK);
    auto values = impl_->atom->k_type.view<LMPDeviceType>();
    return {values.data(), count, 1, values.stride_0(), 1,
            values.span(), MemorySpace::Cuda, impl_->ready()};
}
void LammpsKokkosBridge::add_forces(const double* values, size_t count) {
    if (!count) return;
    ExplicitSync guard(impl_->lmp);
    auto force = impl_->force_view();
    impl_->increments.grow_to(3*count, __FILE__, __LINE__);
    impl_->increments.upload_from(values, 3*count, impl_->transfer.get());
    add_force_rows<<<(count+255)/256,256,0,impl_->transfer.get()>>>(
        force.data(), force.stride_0(), force.stride_1(), impl_->increments.ptr, count);
    impl_->finish_force();
}
void LammpsKokkosBridge::add_force(int local, const double values[3]) {
    ExplicitSync guard(impl_->lmp);
    auto force = impl_->force_view();
    add_force_row<<<1,1,0,impl_->transfer.get()>>>(
        force.data(), force.stride_0(), force.stride_1(), local, values[0], values[1], values[2]);
    impl_->finish_force();
}
} // namespace MetaD_zqc