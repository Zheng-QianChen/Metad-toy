#include "zqc_gpumd_engine.cuh"
#include "zqc_gpumd_neighbors.cuh"
#include "zqc_cuda_execution.cuh"
#include "zqc_gpu_buffer.cuh"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace MetaD_zqc {
namespace {

void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

// 将 CV 力/virial 按各自 stride 累加到 GPUMD 原生 SoA（或 AoS）槽位；Full9 保留九个独立分量。
__global__ void accumulate_force_and_full_virial_kernel(
    ArrayView<double> md_forces, ArrayView<double> md_virials,
    ArrayView<const double> source_forces, VirialView source_virials) {
    const size_t atom_index = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (atom_index >= source_forces.item_count) return;
    for (int axis = 0; axis < 3; ++axis)
        md_forces(atom_index, axis) += source_forces(atom_index, axis);
    for (int component = 0; component < 9; ++component)
        md_virials(atom_index, component) += source_virials.full_component(atom_index, component);
}

// 每个周期像与 owned 行都通过 owner 表归并。全部使用 atomicAdd，
// 避免 owned 的普通 += 与其他线程的像贡献同时写入同一个原子。
__global__ void accumulate_image_force_virial_kernel(
    ArrayView<double> md_forces, ArrayView<double> md_virials,
    ArrayView<const double> source_forces, VirialView source_virials,
    ArrayView<const int> owners) {
    const size_t row=blockIdx.x*static_cast<size_t>(blockDim.x)+threadIdx.x;
    if(row>=source_forces.item_count) return;
    const int owner=owners(row);
    for(int axis=0;axis<3;++axis)
        atomicAdd(&md_forces(owner,axis),source_forces(row,axis));
    for(int component=0;component<9;++component)
        atomicAdd(&md_virials(owner,component),source_virials.full_component(row,component));
}

__global__ void accumulate_packed_host_contributions_kernel(
    ArrayView<double> md_forces, ArrayView<double> md_virials,
    const double* packed_forces_aos, const double* packed_virials_full9_aos,
    size_t atom_count) {
    const size_t atom_index = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (atom_index >= atom_count) return;
    for (int axis = 0; axis < 3; ++axis)
        md_forces(atom_index, axis) += packed_forces_aos[3 * atom_index + axis];
    for (int component = 0; component < 9; ++component)
        md_virials(atom_index, component) += packed_virials_full9_aos[9 * atom_index + component];
}

__global__ void add_uniform_bias_to_potential_kernel(
    ArrayView<double> md_potential, double bias_energy_per_atom) {
    const size_t atom_index = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (atom_index >= md_potential.item_count) return;
    md_potential(atom_index) += bias_energy_per_atom;
}

void launch_grid(size_t atom_count, dim3& grid, dim3& block) {
    block = dim3(128, 1, 1);
    const unsigned blocks = atom_count ?
        static_cast<unsigned>((atom_count + block.x - 1) / block.x) : 0u;
    grid = dim3(blocks, 1, 1);
}

} // namespace

struct GpumdEngineAdapter::Impl {
    Logger logger;
    std::unique_ptr<GpumdNeighbors> neighbors;
    CudaExecutionStream transfer_stream; // 适配器私有传输/累加流；不阻塞无关 MD 工作。
    cudaEvent_t molecular_dynamics_ready_event = nullptr;
    cudaEvent_t adapter_writeback_done_event = nullptr;
    bool frame_bound = false;
    bool evaluation_active = false;
    bool host_positions_cached = false;
    bool host_owned_contributions_dirty = false;
    std::uint64_t evaluation = 0;
    GpumdFrame frame{};
    std::vector<AtomId> stable_atom_ids;           // Host 1..N，行序固定。
    std::vector<double> host_positions_aos;        // 仅 positions(Host) 显式请求时下载。
    std::vector<double> host_force_contributions_aos;   // add_owned 累加，finish 上传。
    std::vector<double> host_virial_contributions_full9_aos;
    std::vector<double> host_commit_force_pack;
    std::vector<double> host_commit_virial_pack;
    GpuBuffer<double> device_commit_force_pack;
    GpuBuffer<double> device_commit_virial_pack;
    GpuBuffer<double> device_owned_force_pack;
    GpuBuffer<double> device_owned_virial_pack;

    Impl() {
        check_cuda(cudaEventCreateWithFlags(&molecular_dynamics_ready_event, cudaEventDisableTiming));
        const auto status = cudaEventCreateWithFlags(&adapter_writeback_done_event, cudaEventDisableTiming);
        if (status != cudaSuccess) {
            cudaEventDestroy(molecular_dynamics_ready_event);
            check_cuda(status);
        }
        device_commit_force_pack.set_name("gpumd_commit_force", logger);
        device_commit_virial_pack.set_name("gpumd_commit_virial", logger);
        device_owned_force_pack.set_name("gpumd_owned_force", logger);
        device_owned_virial_pack.set_name("gpumd_owned_virial", logger);
    }
    ~Impl() {
        if (molecular_dynamics_ready_event) cudaEventDestroy(molecular_dynamics_ready_event);
        if (adapter_writeback_done_event) cudaEventDestroy(adapter_writeback_done_event);
    }

    void require_bound_frame() const {
        if (!frame_bound) throw std::logic_error("GPUMD adapter: frame is not bound");
    }
    void require_evaluation() const {
        require_bound_frame();
        if (!evaluation_active) throw std::logic_error("GPUMD adapter: evaluation has not started");
    }
    int atom_count() const {
        require_bound_frame();
        return static_cast<int>(frame.positions.item_count);
    }
    ViewReadiness molecular_dynamics_readiness() const {
        return cuda_event_readiness(molecular_dynamics_ready_event);
    }

    void resize_host_contribution_buffers(size_t atom_count) {
        host_force_contributions_aos.assign(3 * atom_count, 0.0);
        host_virial_contributions_full9_aos.assign(9 * atom_count, 0.0);
        host_owned_contributions_dirty = false;
    }

    void validate_and_store_frame(const GpumdFrame& incoming) {
        incoming.positions.validate();
        incoming.forces.validate();
        incoming.virials.validate();
        incoming.potential.validate();
        if (incoming.types.item_count) incoming.types.validate();

        if (incoming.positions.memory != MemorySpace::Cuda ||
            incoming.forces.memory != MemorySpace::Cuda ||
            incoming.virials.memory != MemorySpace::Cuda ||
            incoming.potential.memory != MemorySpace::Cuda)
            throw std::invalid_argument("GPUMD adapter: MD arrays must reside in Cuda memory");
        if (incoming.positions.component_count != 3 || incoming.forces.component_count != 3)
            throw std::invalid_argument("GPUMD adapter: positions/forces require 3 components");
        if (incoming.virials.component_count != 9)
            throw std::invalid_argument("GPUMD adapter: virials require native Full9 (9 components)");
        if (incoming.potential.component_count != 1)
            throw std::invalid_argument("GPUMD adapter: potential requires one component per atom");

        const size_t atom_count = incoming.positions.item_count;
        if (incoming.forces.item_count != atom_count ||
            incoming.virials.item_count != atom_count ||
            incoming.potential.item_count != atom_count)
            throw std::invalid_argument("GPUMD adapter: force/virial/potential atom counts must match positions");
        if (incoming.types.item_count && (incoming.types.item_count != atom_count || incoming.types.component_count != 1))
            throw std::invalid_argument("GPUMD adapter: type table length must match atom count");
        if (atom_count > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::length_error("GPUMD adapter: atom count exceeds int range");

        for (int axis = 0; axis < 3; ++axis) {
            if (incoming.periodic[axis] != 0 && incoming.periodic[axis] != 1)
                throw std::invalid_argument("GPUMD adapter: periodic flags must be 0 or 1");
            if (!std::isfinite(incoming.box_lengths[axis]) || !(incoming.box_lengths[axis] > 0))
                throw std::invalid_argument("GPUMD adapter: orthogonal box lengths must be finite and positive");
        }

        // 显式 View 生产者依赖先归入 MD 流，再发布本帧统一的完成事件。
        incoming.positions.wait_on(incoming.stream);
        incoming.forces.wait_on(incoming.stream);
        incoming.virials.wait_on(incoming.stream);
        incoming.potential.wait_on(incoming.stream);
        if (incoming.types.item_count) incoming.types.wait_on(incoming.stream);
        frame = incoming;
        evaluation_active = false;
        // 在 MD 生产流队尾记录就绪事件；positions(Cuda) 通过该事件暴露消费依赖。
        check_cuda(cudaEventRecord(molecular_dynamics_ready_event, frame.stream));
        frame.positions.readiness = molecular_dynamics_readiness();
        frame.forces.readiness = {};
        frame.virials.readiness = {};
        frame.potential.readiness = {};
        if (frame.types.item_count && frame.types.memory == MemorySpace::Cuda)
            frame.types.readiness = molecular_dynamics_readiness();

        if (stable_atom_ids.size() != atom_count) {
            stable_atom_ids.resize(atom_count);
            for (size_t row = 0; row < atom_count; ++row)
                stable_atom_ids[row] = static_cast<AtomId>(row + 1);
        }
        host_positions_cached = false;
        host_positions_aos.clear();
        resize_host_contribution_buffers(atom_count);
        frame_bound = true;
    }

    void wait_for_molecular_dynamics_arrays() {
        // bind_frame 记录的事件位于 MD 生产流队尾；回写前等待，避免与基线力/virial 竞态。
        check_cuda(cudaStreamWaitEvent(transfer_stream.get(), molecular_dynamics_ready_event, 0));
    }

    void upload_and_add_packed_contributions(
        const double* host_forces_aos, const double* host_virials_full9_aos,
        GpuBuffer<double>& device_forces, GpuBuffer<double>& device_virials) {
        const size_t atom_count = frame.positions.item_count;
        if (!atom_count) return;
        wait_for_molecular_dynamics_arrays();
        device_forces.grow_to(3 * atom_count, __FILE__, __LINE__);
        device_virials.grow_to(9 * atom_count, __FILE__, __LINE__);
        device_forces.upload_from(host_forces_aos, 3 * atom_count, transfer_stream.get());
        device_virials.upload_from(host_virials_full9_aos, 9 * atom_count, transfer_stream.get());
        dim3 grid, block;
        launch_grid(atom_count, grid, block);
        if (grid.x) {
            accumulate_packed_host_contributions_kernel<<<grid, block, 0, transfer_stream.get()>>>(
                frame.forces, frame.virials, device_forces.ptr, device_virials.ptr, atom_count);
            check_cuda(cudaGetLastError());
        }
    }

    void commit_device_views(ArrayView<const double> forces, VirialView virials) {
        wait_for_molecular_dynamics_arrays();
        forces.wait_on(transfer_stream.get());
        virials.values.wait_on(transfer_stream.get());
        dim3 grid, block;
        launch_grid(forces.item_count, grid, block);
        if (grid.x) {
            if (neighbors->active()) {
                const auto owners=neighbors->owners(MemorySpace::Cuda);
                owners.wait_on(transfer_stream.get());
                accumulate_image_force_virial_kernel<<<grid, block, 0, transfer_stream.get()>>>(
                    frame.forces,frame.virials,forces,virials,owners);
            } else {
                accumulate_force_and_full_virial_kernel<<<grid, block, 0, transfer_stream.get()>>>(
                    frame.forces, frame.virials, forces, virials);
            }
            check_cuda(cudaGetLastError());
        }
    }

    void commit_host_views(ArrayView<const double> forces, VirialView virials) {
        forces.wait_host();
        virials.values.wait_host();
        const size_t atom_count = frame.positions.item_count;
        host_commit_force_pack.assign(3 * atom_count,0.0);
        host_commit_virial_pack.assign(9 * atom_count,0.0);
        const auto owners=neighbors->active() ? neighbors->owners(MemorySpace::Host) : ArrayView<const int>{};
        for (size_t row = 0; row < forces.item_count; ++row) {
            const size_t owner=neighbors->active() ? static_cast<size_t>(owners(row)) : row;
            for (int axis = 0; axis < 3; ++axis)
                host_commit_force_pack[3 * owner + axis] += forces(row, axis);
            for (int component = 0; component < 9; ++component)
                host_commit_virial_pack[9 * owner + component] += virials.full_component(row, component);
        }
        upload_and_add_packed_contributions(host_commit_force_pack.data(),
                                            host_commit_virial_pack.data(),
                                            device_commit_force_pack, device_commit_virial_pack);
    }
};

GpumdEngineAdapter::GpumdEngineAdapter(Logger logger) : impl_(new Impl) {
    impl_->logger = std::move(logger);
    impl_->neighbors.reset(new GpumdNeighbors(impl_->logger));
    impl_->device_commit_force_pack.set_name("gpumd_commit_force", impl_->logger);
    impl_->device_commit_virial_pack.set_name("gpumd_commit_virial", impl_->logger);
    impl_->device_owned_force_pack.set_name("gpumd_owned_force", impl_->logger);
    impl_->device_owned_virial_pack.set_name("gpumd_owned_virial", impl_->logger);
}

GpumdEngineAdapter::~GpumdEngineAdapter() = default;

void GpumdEngineAdapter::bind_frame(const GpumdFrame& frame) {
    impl_->validate_and_store_frame(frame);
}

void GpumdEngineAdapter::begin_evaluation() {
    impl_->require_bound_frame();
    if (impl_->evaluation == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("GPUMD adapter: evaluation counter exhausted");
    ++impl_->evaluation;
    impl_->evaluation_active = true;
    // 每轮清空 Host 贡献；MD 基线力/virial 由宿主保留，适配器只做加法。
    impl_->resize_host_contribution_buffers(impl_->frame.positions.item_count);
    impl_->host_positions_cached = false;
    // 所有请求的周期像布局在 Graph 读取 total/selection 前确定，保证本轮地址和下标一致。
    if (impl_->neighbors->active()) {
        impl_->wait_for_molecular_dynamics_arrays();
        impl_->neighbors->update(impl_->frame.positions,impl_->frame.types,
            impl_->frame.box_lengths,impl_->frame.periodic,impl_->transfer_stream.get(),impl_->evaluation);
    }
}

std::uint64_t GpumdEngineAdapter::evaluation_id() const { return impl_->evaluation; }
int GpumdEngineAdapter::owned_atom_count() const { return impl_->atom_count(); }
int GpumdEngineAdapter::total_atom_count() const {
    return impl_->neighbors->active() && impl_->evaluation_active ?
        impl_->neighbors->total_atoms() : impl_->atom_count();
}
Logger GpumdEngineAdapter::logger() const { return impl_->logger; }
std::int64_t GpumdEngineAdapter::timestep() const {
    impl_->require_bound_frame();
    return impl_->frame.step;
}

ArrayView<const double> GpumdEngineAdapter::positions(MemorySpace memory) {
    impl_->require_evaluation();
    auto& state = *impl_;
    if (state.neighbors->active()) return state.neighbors->positions(memory);
    if (memory == MemorySpace::Cuda) {
        auto view = state.frame.positions;
        view.readiness = state.molecular_dynamics_readiness();
        return view;
    }
    if (memory != MemorySpace::Host)
        throw std::invalid_argument("GPUMD adapter: unknown position memory space");
    // Host 全量坐标仅在显式请求时懒下载；DISTANCE 走 read_owned_atoms，保持紧凑。
    const size_t atom_count = state.frame.positions.item_count;
    if (!state.host_positions_cached) {
        state.host_positions_aos.resize(3 * atom_count);
        state.frame.positions.wait_on(state.transfer_stream.get());
        for (size_t atom_index = 0; atom_index < atom_count; ++atom_index)
            for (int axis = 0; axis < 3; ++axis) {
                const double* device_component =
                    state.frame.positions.data +
                    atom_index * state.frame.positions.item_stride +
                    static_cast<size_t>(axis) * state.frame.positions.component_stride;
                check_cuda(cudaMemcpyAsync(
                    state.host_positions_aos.data() + 3 * atom_index + axis,
                    device_component, sizeof(double), cudaMemcpyDeviceToHost,
                    state.transfer_stream.get()));
            }
        check_cuda(cudaStreamSynchronize(state.transfer_stream.get()));
        state.host_positions_cached = true;
    }
    return {state.host_positions_aos.data(), atom_count, 3, 3, 1,
            state.host_positions_aos.size(), MemorySpace::Host};
}

ArrayView<const int> GpumdEngineAdapter::atom_types(MemorySpace memory) {
    impl_->require_evaluation();
    if (impl_->neighbors->active()) return impl_->neighbors->types(memory);
    if (!impl_->frame.types.item_count) return {};
    if (memory != impl_->frame.types.memory)
        throw std::invalid_argument("GPUMD adapter: atom type memory space mismatch");
    auto view = impl_->frame.types;
    if (view.memory == MemorySpace::Cuda)
        view.readiness = impl_->molecular_dynamics_readiness();
    return view;
}

ArrayView<const AtomId> GpumdEngineAdapter::atom_ids() {
    impl_->require_evaluation();
    if (impl_->neighbors->active()) return impl_->neighbors->ids();
    return {impl_->stable_atom_ids.data(), impl_->stable_atom_ids.size(), 1, 1, 1,
            impl_->stable_atom_ids.size(), MemorySpace::Host};
}

SelectionId GpumdEngineAdapter::define_selection(const std::string& name,const std::vector<int>& rows) {
    if (impl_->evaluation_active) throw std::logic_error("Define selections between evaluations");
    return impl_->neighbors->define_selection(name,rows);
}
SelectionId GpumdEngineAdapter::selection_id(const std::string& name) const {
    return impl_->neighbors->selection_id(name);
}
SelectionView GpumdEngineAdapter::selection_members(SelectionId selection) {
    impl_->require_evaluation();
    return impl_->neighbors->selection_members(selection);
}
ArrayView<const int> GpumdEngineAdapter::selection_flags(SelectionId selection,MemorySpace memory) {
    impl_->require_evaluation();
    return impl_->neighbors->selection_flags(selection,memory);
}
void GpumdEngineAdapter::set_neighbor_skin(double skin) {
    if (impl_->evaluation_active) throw std::logic_error("Set neighbor skin between evaluations");
    impl_->neighbors->set_skin(skin);
}
std::uint64_t GpumdEngineAdapter::neighbor_rebuild_count() const {
    return impl_->neighbors->rebuild_count();
}
int GpumdEngineAdapter::register_neighbors(const NeighborRequest& request) {
    if (impl_->evaluation_active) throw std::logic_error("Register neighbor requests between evaluations");
    return impl_->neighbors->register_request(request);
}
NeighborListView GpumdEngineAdapter::prepare_neighbors(int handle) {
    impl_->require_evaluation();
    return impl_->neighbors->list(handle);
}

void GpumdEngineAdapter::read_owned_atoms(const AtomId* requested_ids, size_t requested_count,
                                          double* coordinates_and_counts, int* owned_indices) {
    impl_->require_evaluation();
    if ((requested_count && (!requested_ids || !coordinates_and_counts || !owned_indices)))
        throw std::invalid_argument("GPUMD adapter: read_owned_atoms requires valid buffers");
    std::fill_n(coordinates_and_counts, 4 * requested_count, 0.0);
    std::fill_n(owned_indices, requested_count, -1);
    if (!requested_count) return;

    auto& state = *impl_;
    const size_t atom_count = state.frame.positions.item_count;
    // 先建立对 MD 坐标生产流的依赖，再只拷贝被点名的原子分量。
    state.frame.positions.wait_on(state.transfer_stream.get());
    for (size_t endpoint = 0; endpoint < requested_count; ++endpoint) {
        const AtomId atom_id = requested_ids[endpoint];
        if (!atom_id || atom_id > atom_count) continue; // 缺失/越界：owner_count 保持 0。
        const size_t local_row = static_cast<size_t>(atom_id - 1);
        owned_indices[endpoint] = static_cast<int>(local_row);
        for (int axis = 0; axis < 3; ++axis) {
            const double* device_component =
                state.frame.positions.data +
                local_row * state.frame.positions.item_stride +
                static_cast<size_t>(axis) * state.frame.positions.component_stride;
            check_cuda(cudaMemcpyAsync(
                coordinates_and_counts + 4 * endpoint + axis,
                device_component, sizeof(double), cudaMemcpyDeviceToHost,
                state.transfer_stream.get()));
        }
    }
    check_cuda(cudaStreamSynchronize(state.transfer_stream.get()));
    for (size_t endpoint = 0; endpoint < requested_count; ++endpoint)
        if (owned_indices[endpoint] >= 0) coordinates_and_counts[4 * endpoint + 3] = 1.0;
}

void GpumdEngineAdapter::sum(double*, size_t) {
    // 单进程固定顺序：无需跨 rank 规约。
}

void GpumdEngineAdapter::minimum_image(double displacement[3]) const {
    impl_->require_bound_frame();
    if (!displacement) throw std::invalid_argument("GPUMD adapter: null displacement");
    const auto& box = impl_->frame.box_lengths;
    const auto& periodic = impl_->frame.periodic;
    for (int axis = 0; axis < 3; ++axis) {
        if (!periodic[axis]) continue;
        displacement[axis] -= box[axis] * std::round(displacement[axis] / box[axis]);
    }
}

void GpumdEngineAdapter::commit_force_virial(ArrayView<const double> forces, VirialView virials) {
    impl_->require_evaluation();
    forces.validate();
    virials.validate();
    const size_t atom_count = static_cast<size_t>(total_atom_count());
    if (forces.item_count != atom_count || virials.values.item_count != atom_count ||
        forces.component_count != 3)
        throw std::invalid_argument("GPUMD adapter: invalid force/virial shape for commit");
    if (forces.memory == MemorySpace::Cuda && virials.values.memory == MemorySpace::Cuda)
        impl_->commit_device_views(forces, virials);
    else if (forces.memory == MemorySpace::Host && virials.values.memory == MemorySpace::Host)
        impl_->commit_host_views(forces, virials);
    else
        throw std::invalid_argument("GPUMD adapter: force and virial memory spaces must match");
    // 提交契约允许调用方在返回后复用源数组。
    check_cuda(cudaStreamSynchronize(impl_->transfer_stream.get()));
}

void GpumdEngineAdapter::add_owned_force_virial(int local_index, const double force[3],
                                               VirialView virial) {
    impl_->require_evaluation();
    virial.validate();
    virial.values.wait_host();
    if (!force || virial.values.item_count != 1 || virial.values.memory != MemorySpace::Host)
        throw std::invalid_argument("GPUMD adapter: owned contribution requires one Host row");
    if (local_index < 0 || local_index >= owned_atom_count())
        throw std::out_of_range("GPUMD adapter: force target is not owned");
    auto& state = *impl_;
    for (int axis = 0; axis < 3; ++axis)
        state.host_force_contributions_aos[3 * local_index + axis] += force[axis];
    // Symmetric6 经 full_component 补齐转置项；Full9 保留独立的 yx/zx/zy。
    for (int component = 0; component < 9; ++component)
        state.host_virial_contributions_full9_aos[9 * local_index + component] +=
            virial.full_component(0, component);
    state.host_owned_contributions_dirty = true;
}

void GpumdEngineAdapter::finish(double bias_energy) {
    impl_->require_evaluation();
    if (!std::isfinite(bias_energy))
        throw std::invalid_argument("GPUMD adapter: non-finite bias energy");
    auto& state = *impl_;
    const size_t atom_count = state.frame.positions.item_count;

    if (state.host_owned_contributions_dirty) {
        state.upload_and_add_packed_contributions(
            state.host_force_contributions_aos.data(),
            state.host_virial_contributions_full9_aos.data(),
            state.device_owned_force_pack, state.device_owned_virial_pack);
        state.host_owned_contributions_dirty = false;
    }

    if (atom_count && bias_energy != 0.0) {
        state.wait_for_molecular_dynamics_arrays();
        const double bias_energy_per_atom = bias_energy / static_cast<double>(atom_count);
        dim3 grid, block;
        launch_grid(atom_count, grid, block);
        if (grid.x) {
            add_uniform_bias_to_potential_kernel<<<grid, block, 0, state.transfer_stream.get()>>>(
                state.frame.potential, bias_energy_per_atom);
            check_cuda(cudaGetLastError());
        }
    } else if (!atom_count && bias_energy != 0.0) {
        throw std::invalid_argument("GPUMD adapter: cannot distribute bias energy over zero atoms");
    }

    // 适配器私有流队尾完成事件；MD 流随后等待，保证下一步积分看到累加结果。
    check_cuda(cudaEventRecord(state.adapter_writeback_done_event, state.transfer_stream.get()));
    check_cuda(cudaStreamWaitEvent(state.frame.stream, state.adapter_writeback_done_event, 0));
    state.evaluation_active = false;
}

} // namespace MetaD_zqc
