#include "zqc_lammps_kokkos.h"
#include "zqc_cuda_execution.cuh"
#include "zqc_gpu_buffer.cuh"
#include "lammps.h"
#include "fix.h"
#include "atom_masks.h"
#include "atom_kokkos.h"
#include "kokkos.h"
#include "neigh_list_kokkos.h"
#include "comm.h"
#ifdef ZQC_LAMMPS_WEAVE
#include "weave_metad.h"
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
#ifdef ZQC_LAMMPS_WEAVE
__global__ void accumulate_rows(double* rows, int width, size_t count,
                               ArrayView<const double> forces, ArrayView<const double> virials) {
    const size_t k = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (k < count*size_t(width)) {
        const size_t atom = k/size_t(width), component = k%size_t(width);
        rows[k] += component < 3 ? forces(atom, component) : virials(atom, component-3);
    }
}
__global__ void apply_rows(double* force, size_t item_stride, size_t component_stride,
                          const double* rows, int width, size_t owned, double* sum, bool reduce) {
    const size_t atom = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (atom < owned)
        for (int c = 0; c < 3; ++c)
            force[atom*item_stride+c*component_stride] += rows[atom*size_t(width)+size_t(c)];
    if (reduce) {
        __shared__ double partial[6][256];
        for (int c = 0; c < 6; ++c)
            partial[c][threadIdx.x] = atom < owned ? rows[atom*size_t(width)+3+size_t(c)] : 0.0;
        __syncthreads();
        for (int offset = blockDim.x/2; offset; offset /= 2) {
            if (threadIdx.x < offset)
                for (int c = 0; c < 6; ++c) partial[c][threadIdx.x] += partial[c][threadIdx.x+offset];
            __syncthreads();
        }
        if (!threadIdx.x)
            for (int c = 0; c < 6; ++c) atomicAdd(sum+c, partial[c][0]);
    }
}
#endif
}
struct LammpsKokkosBridge::Impl {
    LAMMPS_NS::LAMMPS* lmp;
    LAMMPS_NS::AtomKokkos* atom = nullptr;
    CudaExecutionStream transfer;
    GpuBuffer<double> increments;
#ifdef ZQC_LAMMPS_WEAVE
    GpuBuffer<double> batch, reduced, host_virial;
    bool batch_enabled = true;
    bool batch_open = false;
    size_t batch_total = 0;
    int batch_width = 0;
    std::uint64_t batches = 0, commits = 0, global_d2h = 0, per_atom_d2h = 0, ghost_d2h = 0, ghost_h2d = 0;
#endif
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
#ifdef ZQC_LAMMPS_WEAVE
    if (const char* value = std::getenv("WEAVE_METAD_DEVICE_COMMIT")) {
        if (std::strcmp(value, "0") && std::strcmp(value, "1"))
            throw std::invalid_argument("WEAVE_METAD_DEVICE_COMMIT must be 0 or 1");
        impl_->batch_enabled = std::strcmp(value, "0") != 0;
    }
#endif
    execution_check(cudaEventCreateWithFlags(&impl_->producer_ready, cudaEventDisableTiming));
    // 所有混合 Host/Device 字段由桥接显式同步；ModifyKokkos 不再复制 ALL_MASK。
    fix->kokkosable = 1;
    fix->execution_space = LAMMPS_NS::Device;
    fix->datamask_read = EMPTY_MASK;
    fix->datamask_modify = EMPTY_MASK;
}
LammpsKokkosBridge::~LammpsKokkosBridge() {
#ifdef ZQC_LAMMPS_WEAVE
    if (active()) {
        std::printf("WEAVE_METAD_JSON {\"rank\":%d,\"device_commit\":%s,\"batches\":%llu,\"commits\":%llu,\"global_virial_d2h_bytes\":%llu,\"per_atom_virial_d2h_bytes\":%llu,\"ghost_d2h_bytes\":%llu,\"ghost_h2d_bytes\":%llu}\n",
            impl_->lmp->comm->me, impl_->batch_enabled ? "true" : "false",
            (unsigned long long)impl_->batches, (unsigned long long)impl_->commits,
            (unsigned long long)impl_->global_d2h, (unsigned long long)impl_->per_atom_d2h,
            (unsigned long long)impl_->ghost_d2h, (unsigned long long)impl_->ghost_h2d);
        std::fflush(stdout);
    }
#endif
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
KokkosNeighborView LammpsKokkosBridge::full_neighbors(LAMMPS_NS::NeighList* source, bool ghosts) {
    auto* list = dynamic_cast<LAMMPS_NS::NeighListKokkos<LMPDeviceType>*>(source);
    const size_t count = impl_->atom->nlocal + (ghosts ? impl_->atom->nghost : 0);
    if (!list || list->inum != impl_->atom->nlocal ||
        (ghosts && list->gnum != impl_->atom->nghost) ||
        list->d_numneigh.extent(0) < count || list->d_neighbors.extent(0) < count)
        throw std::runtime_error("MetaD-toy: incomplete Kokkos full neighbor list");
    const auto ready = impl_->ready();
    return {{list->d_numneigh.data(), count, 1, list->d_numneigh.stride_0(), 1,
             list->d_numneigh.span(), MemorySpace::Cuda, ready},
            {list->d_neighbors.data(), count, list->d_neighbors.extent(1),
             list->d_neighbors.stride_0(), list->d_neighbors.stride_1(),
             list->d_neighbors.span(), MemorySpace::Cuda, ready}};
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
bool LammpsKokkosBridge::begin_force_batch(size_t total, bool need_virial) {
#ifdef ZQC_LAMMPS_WEAVE
    if (!active() || !impl_->batch_enabled) return false;
    if (impl_->batch_open) throw std::logic_error("LAMMPS Kokkos: nested device force batch");
    impl_->batch_width = need_virial ? 9 : 3;
    impl_->batch_total = total;
    impl_->batch.grow_to(total*size_t(impl_->batch_width), __FILE__, __LINE__);
    if (total)
        execution_check(cudaMemsetAsync(impl_->batch.ptr, 0,
            total*size_t(impl_->batch_width)*sizeof(double), impl_->transfer.get()));
    // 清零完成后再接受累加，避免与上一轮残留竞态。
    impl_->transfer.synchronize();
    impl_->batch_open = true;
    return true;
#else
    (void)total; (void)need_virial;
    return false;
#endif
}
bool LammpsKokkosBridge::force_batch_active() const {
#ifdef ZQC_LAMMPS_WEAVE
    return impl_->batch_open;
#else
    return false;
#endif
}
void LammpsKokkosBridge::accumulate_force_virial(ArrayView<const double> force,
                                                 ArrayView<const double> virial) {
#ifdef ZQC_LAMMPS_WEAVE
    if (!impl_->batch_open || force.item_count != impl_->batch_total ||
        virial.item_count != impl_->batch_total ||
        force.memory != MemorySpace::Cuda || virial.memory != MemorySpace::Cuda)
        throw std::invalid_argument("LAMMPS Kokkos: invalid device force/virial contribution");
    force.wait_on(impl_->transfer.get());
    virial.wait_on(impl_->transfer.get());
    const size_t size = impl_->batch_total*size_t(impl_->batch_width);
    if (size)
        accumulate_rows<<<(size+255)/256,256,0,impl_->transfer.get()>>>(
            impl_->batch.ptr, impl_->batch_width, impl_->batch_total, force, virial);
    execution_check(cudaGetLastError());
    // 提交契约：返回前源 View 的异步读取必须结束，调用方可立即复用缓冲。
    impl_->transfer.synchronize();
    ++impl_->commits;
#else
    (void)force; (void)virial;
    throw std::logic_error("LAMMPS Kokkos: device force batch requires ZQC_LAMMPS_WEAVE");
#endif
}
void LammpsKokkosBridge::accumulate_host_force_virial(const double* forces,
                                                       const double* virials, size_t total) {
#ifdef ZQC_LAMMPS_WEAVE
    if (!impl_->batch_open || total != impl_->batch_total)
        throw std::logic_error("LAMMPS Kokkos: invalid Host batch contribution");
    impl_->increments.grow_to(total*3, __FILE__, __LINE__);
    impl_->host_virial.grow_to(total*6, __FILE__, __LINE__);
    impl_->increments.upload_from(forces,total*3,impl_->transfer.get());
    impl_->host_virial.upload_from(virials,total*6,impl_->transfer.get());
    accumulate_force_virial(impl_->increments.view(total,3),impl_->host_virial.view(total,6));
#else
    (void)forces; (void)virials; (void)total;
    throw std::logic_error("LAMMPS Kokkos: device force batch requires ZQC_LAMMPS_WEAVE");
#endif
}
void LammpsKokkosBridge::finish_force_batch(size_t owned, bool needs_reverse, bool global, bool per_atom,
                                            double global_virial[6], std::vector<double>& atom_virial) {
#ifdef ZQC_LAMMPS_WEAVE
    if (!impl_->batch_open || owned > impl_->batch_total)
        throw std::logic_error("LAMMPS Kokkos: invalid force batch completion");
    // 无论成功或抛错都关闭 open batch，避免异常后残留。
    struct CloseBatch {
        Impl& state;
        ~CloseBatch() {
            // Also drain on allocation/CUDA exceptions after reverse was queued.
            cudaStreamSynchronize(state.transfer.get());
            state.batch_open = false;
        }
    } close{*impl_};
    ExplicitSync guard(impl_->lmp);
    if (needs_reverse) {
        weave_metad_reverse(impl_->lmp, impl_->batch.ptr, int(impl_->batch_total), impl_->batch_width,
                            impl_->transfer.get(), &impl_->ghost_d2h, &impl_->ghost_h2d);
    }
    auto force = impl_->force_view();
    const bool reduce = global && !per_atom && impl_->batch_width >= 9;
    if (reduce) {
        impl_->reduced.grow_to(6, __FILE__, __LINE__);
        execution_check(cudaMemsetAsync(impl_->reduced.ptr, 0, 6*sizeof(double), impl_->transfer.get()));
    }
    if (owned)
        apply_rows<<<(owned+255)/256,256,0,impl_->transfer.get()>>>(
            force.data(), force.stride_0(), force.stride_1(),
            impl_->batch.ptr, impl_->batch_width, owned, impl_->reduced.ptr, reduce);
    execution_check(cudaGetLastError());
    if (reduce) {
        execution_check(cudaMemcpyAsync(global_virial, impl_->reduced.ptr, 6*sizeof(double),
            cudaMemcpyDeviceToHost, impl_->transfer.get()));
        impl_->global_d2h += 6*sizeof(double);
    }
    if (per_atom && owned && impl_->batch_width >= 9) {
        atom_virial.resize(6*owned);
        execution_check(cudaMemcpy2DAsync(atom_virial.data(), 6*sizeof(double), impl_->batch.ptr+3,
            size_t(impl_->batch_width)*sizeof(double), 6*sizeof(double), owned,
            cudaMemcpyDeviceToHost, impl_->transfer.get()));
        impl_->per_atom_d2h += 6*owned*sizeof(double);
    }
    // 结束同步同时关闭各 CV 力/virial 缓冲的借用期，并标记 Device 力最新。
    impl_->finish_force();
    ++impl_->batches;
#else
    (void)owned; (void)needs_reverse; (void)global; (void)per_atom;
    (void)global_virial; (void)atom_virial;
    throw std::logic_error("LAMMPS Kokkos: device force batch requires ZQC_LAMMPS_WEAVE");
#endif
}
void LammpsKokkosBridge::cancel_force_batch() noexcept {
#ifdef ZQC_LAMMPS_WEAVE
    if (!impl_->batch_open) return;
    // 排空已入队的累加/清零，丢弃设备结果，不写宿主力或 virial。
    cudaStreamSynchronize(impl_->transfer.get());
    impl_->batch_open = false;
#endif
}
} // namespace MetaD_zqc
