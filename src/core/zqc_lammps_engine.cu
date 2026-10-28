#include "zqc_lammps_engine.h"
#include "fix_crystallize.h"
#include "zqc_gpu_buffer.cuh"
#include "zqc_cuda_execution.cuh"
#ifdef ZQC_LAMMPS_KOKKOS
#include "zqc_lammps_kokkos.h"
#endif
#include "atom.h"
#include "domain.h"
#include "group.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "comm.h"
#include "update.h"
#include "utils.h"
#include <array>
#include <map>
#include <algorithm>
#include <climits>
#include <unordered_map>
#include <mpi.h>

namespace MetaD_zqc {
namespace {
void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template<class T> void upload(GpuBuffer<T>& buffer, const std::vector<T>& values, cudaStream_t stream) {
    buffer.grow_to(values.size(), __FILE__, __LINE__);
    buffer.upload_from(values.data(), values.size(), stream);
}
// Host 暂存统一为逐原子连续排列；CUDA 连续输入一次下载，带 stride 的输入逐分量下载。
void download(ArrayView<const double> source, std::vector<double>& destination, cudaStream_t stream) {
    if (source.memory == MemorySpace::Host) source.wait_host();
    else source.wait_on(stream);
    source.validate();
    if (source.item_count > std::numeric_limits<size_t>::max()/source.component_count)
        throw std::length_error("LAMMPS adapter: output shape overflow");
    destination.resize(source.item_count*source.component_count);
    if (!source.item_count) return;
    if (source.memory == MemorySpace::Host) {
        for (size_t atom = 0; atom < source.item_count; ++atom)
            for (size_t component = 0; component < source.component_count; ++component)
                destination[atom*source.component_count+component] = source(atom,component);
    } else if (source.item_stride == source.component_count && source.component_stride == 1) {
        check_cuda(cudaMemcpyAsync(destination.data(),source.data,destination.size()*sizeof(double),cudaMemcpyDeviceToHost,stream));
    } else {
        for (size_t component = 0; component < source.component_count; ++component)
            check_cuda(cudaMemcpy2DAsync(destination.data()+component,source.component_count*sizeof(double),
                source.data+component*source.component_stride,source.item_stride*sizeof(double),
                sizeof(double),source.item_count,cudaMemcpyDeviceToHost,stream));
    }
    check_cuda(cudaStreamSynchronize(stream)); // MPI 即将打包 Host 数据。
}

// 读取一份借用视图并累加到公共 Host 输出。CUDA 输入复用暂存后立即释放借用，
// 因而下一 CV 可以复用生产缓冲；Host 输入按原 stride 累加。
void accumulate_output(ArrayView<const double> source, std::vector<double>& accumulated,
                       std::vector<double>& staging, cudaStream_t stream) {
    if (source.memory == MemorySpace::Host) {
        source.wait_host();
        for (size_t atom = 0; atom < source.item_count; ++atom)
            for (size_t component = 0; component < source.component_count; ++component)
                accumulated[atom*source.component_count+component] += source(atom,component);
    } else {
        download(source, staging, stream);
        for (size_t entry = 0; entry < staging.size(); ++entry)
            accumulated[entry] += staging[entry];
    }
}

// 一次 reverse 传输 [Fx,Fy,Fz,Wxx,Wyy,Wzz,Wxy,Wxz,Wyz]，包中保存本轮各 CV 的总贡献。
struct ForceVirialCommunication : AtomCommunication {
    std::vector<double> forces, virials;
    int get_comm_reverse_width() const override { return 9; }
    int pack_comm_reverse_ubuf(int count, int first, double* buffer, int offset, int stride) override {
        for (int atom = 0; atom < count; ++atom) {
            for (int component = 0; component < 3; ++component)
                buffer[offset+atom*stride+component] = forces[3*(first+atom)+component];
            for (int component = 0; component < 6; ++component)
                buffer[offset+atom*stride+3+component] = virials[6*(first+atom)+component];
        }
        return 9;
    }
    void unpack_comm_reverse_ubuf(int count, int* local_ids, double* buffer, int offset, int stride) override {
        for (int atom = 0; atom < count; ++atom) {
            for (int component = 0; component < 3; ++component)
                forces[3*local_ids[atom]+component] += buffer[offset+atom*stride+component];
            for (int component = 0; component < 6; ++component)
                virials[6*local_ids[atom]+component] += buffer[offset+atom*stride+3+component];
        }
    }
};
template<class T> ArrayView<const T> device_vector(const GpuBuffer<T>& buffer, size_t count) {
    return buffer.view(count);
}
}

// 与原生 ghost full list 一致的 type/group/molecule 排除规则。
namespace {
bool excluded_neighbor(const LAMMPS_NS::Atom* atom, const LAMMPS_NS::Neighbor* neighbor,
                       int first, int second) {
    if (neighbor->nex_type && neighbor->ex_type[atom->type[first]][atom->type[second]]) return true;
    for (int entry = 0; entry < neighbor->nex_group; ++entry) {
        const int first_bit = neighbor->ex1_bit[entry], second_bit = neighbor->ex2_bit[entry];
        if (((atom->mask[first] & first_bit) && (atom->mask[second] & second_bit)) ||
            ((atom->mask[first] & second_bit) && (atom->mask[second] & first_bit))) return true;
    }
    for (int entry = 0; entry < neighbor->nex_mol; ++entry) {
        const int group_bit = neighbor->ex_mol_bit[entry];
        if ((atom->mask[first] & group_bit) && (atom->mask[second] & group_bit)) {
            const bool same_molecule = atom->molecule[first] == atom->molecule[second];
            if (same_molecule == bool(neighbor->ex_mol_intra[entry])) return true;
        }
    }
    return false;
}
}

struct LammpsEngineAdapter::Impl {
    CudaExecutionStream transfer; // Host 引擎的上传/回写流，跨流输入依赖由 View 提供。
    LAMMPS_NS::LAMMPS* lmp;
    LAMMPS_NS::FixMetadynamics* fix;
    Logger logger;
#ifdef ZQC_LAMMPS_KOKKOS
    std::unique_ptr<LammpsKokkosBridge> kokkos;
#endif
    std::uint64_t evaluation = 0;
    std::uint64_t topology_epoch = 1;
    SyncStats stats;
    bool host_positions_ready = false, device_positions_ready = false, device_types_ready = false;
    std::vector<double> host_positions;
    std::vector<AtomId> host_ids;
    std::uint64_t ids_epoch = 0;
    std::uint64_t owned_ids_epoch = 0;
    std::unordered_map<AtomId, int> owned_local_ids; // 稳定 ID -> owned localid，DISTANCE 等按需使用。
    std::vector<int> host_types;
    GpuBuffer<double> device_positions;
    GpuBuffer<int> device_types;
    // local_ids 为 owned 前缀+ghost 后缀。静态组按宿主 epoch 缓存，动态组每轮比较。
    // dense flags 仅在消费者请求时生成，device_flags 仅在 CUDA 消费者请求时上传。
    struct Selection {
        std::uint64_t evaluation = 0;
        std::uint64_t topology_epoch = 0, generation = 0, flags_generation = 0;
        int owned_count = 0;
        bool device_ready = false;
        std::vector<int> local_ids, flags;
        GpuBuffer<int> device_flags;
    };
    std::map<SelectionId, std::unique_ptr<Selection>> selections;
    // Host AoS 输出缓冲：owned 在前、ghost 在后。批次中累加所有 CV；
    // 批次外保存单份即时提交。容量跨步复用，begin 时清零本轮有效部分。
    ForceVirialCommunication output;
    std::vector<double> force_staging, virial_staging;
    bool force_batch_active = false;
    bool force_batch_has_contributions = false, force_batch_needs_reverse = false;
    bool force_batch_host_contributions = false;
    bool batch_global_virial = false, batch_atom_virial = false;
    bool device_virial_complete = false;
    std::vector<double> device_atom_virial;
    struct Neighbors {
        NeighborRequest request;
        int native_handle = -1;
        bool native_half = false; // 相同选择集可直接请求宿主的 Newton-on half 表。
        std::uint64_t evaluation = 0;
        std::uint64_t topology_generation = 0;
        std::uint64_t topology_epoch = 0, center_generation = 0, neighbor_generation = 0;
        // 自行补齐的 ghost 行独立于 CV 的组筛选，按宿主拓扑 epoch 复用。
        // offsets 按 localid 索引；repaired 标记区分原生行与补齐行。
        std::uint64_t ghost_rows_epoch = 0;
        std::vector<unsigned char> repaired_ghost;
        std::vector<int> ghost_offsets, ghost_neighbors;
        std::vector<int> centers, offsets, neighbors;
        GpuBuffer<int> device_centers, device_offsets, device_neighbors;
    };
    // unique_ptr 保证追加请求不会移动 GpuBuffer 所有者。
    std::vector<std::unique_ptr<Neighbors>> lists;
    int selection_mask(SelectionId selection) const {
        if (selection == all_atoms) return -1;
        if (selection < 0 || selection >= static_cast<int>(sizeof(int)*CHAR_BIT) || !lmp->group->names[selection])
            throw std::invalid_argument("LAMMPS adapter: unknown group selection");
        return lmp->group->bitmask[selection];
    }
    void require_evaluation() const {
        if (!evaluation) throw std::logic_error("LAMMPS adapter: evaluation has not started");
    }
    // 调用方已经将输出规整为 AoS；所有 ghost 在一次 reverse 中归还 owner。
    // Kokkos 仅上传最终 owned 力一次，Host virial 留给 Fix 的 tally。
    void flush_force_virial(bool needs_reverse) {
        if (needs_reverse) {
            fix->communicate_reverse(output);
            ++stats.force_reverse_calls;
        }
#ifdef ZQC_LAMMPS_KOKKOS
        const bool device_output = kokkos->active();
        if (device_output) kokkos->add_forces(output.forces.data(), lmp->atom->nlocal);
#else
        constexpr bool device_output = false;
#endif
        for (int atom = 0; atom < lmp->atom->nlocal; ++atom) {
            if (!device_output)
                for (int component = 0; component < 3; ++component)
                    lmp->atom->f[atom][component] += output.forces[3*atom+component];
            if (fix->a_virial)
                for (int component = 0; component < 6; ++component)
                    fix->a_virial[6*atom+component] += output.virials[6*atom+component];
        }
        ++stats.force_virial_flushes;
    }
    void clear_force_batch_flags() noexcept {
        force_batch_active = false;
        force_batch_has_contributions = force_batch_needs_reverse = false;
        force_batch_host_contributions = false;
    }
#ifdef ZQC_LAMMPS_KOKKOS
    bool try_open_device_batch(size_t total) {
        if (kokkos->force_batch_active()) return true;
        return kokkos->begin_force_batch(total, batch_global_virial || batch_atom_virial);
    }
    void finish_device_batch(bool needs_reverse) {
        double global_virial[6] = {};
        device_atom_virial.clear();
        kokkos->finish_force_batch(lmp->atom->nlocal, needs_reverse,
            batch_global_virial, batch_atom_virial, global_virial, device_atom_virial);
        if (batch_atom_virial) {
            const int owned = lmp->atom->nlocal;
            for (int entry = 0; entry < 6*owned; ++entry)
                fix->a_virial[entry] += device_atom_virial[size_t(entry)];
            device_virial_complete = false;
        } else if (batch_global_virial) {
            for (int component = 0; component < 6; ++component)
                fix->virial[component] += global_virial[component];
            device_virial_complete = true;
        } else {
            device_virial_complete = false;
        }
        ++stats.force_virial_flushes;
        if (needs_reverse) ++stats.force_reverse_calls;
    }
#endif
};

LammpsEngineAdapter::LammpsEngineAdapter(LAMMPS_NS::LAMMPS* lmp,
    LAMMPS_NS::FixMetadynamics* fix, Logger logger) : impl_(new Impl) {
    impl_->lmp = lmp; impl_->fix = fix; impl_->logger = logger;
#ifdef ZQC_LAMMPS_KOKKOS
    impl_->kokkos = std::make_unique<LammpsKokkosBridge>(lmp, fix);
#endif
    impl_->device_positions.set_name("engine_positions", logger);
    impl_->device_types.set_name("engine_types", logger);
}
LammpsEngineAdapter::~LammpsEngineAdapter() = default;
void LammpsEngineAdapter::invalidate_topology() {
    if (impl_->force_batch_active)
        throw std::logic_error("LAMMPS adapter: topology changed during force/virial batch");
    auto& epoch = impl_->topology_epoch;
    if (epoch == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("LAMMPS adapter: topology counter exhausted");
    ++epoch;
    // run 间可修改 skin；通信半径与 NeighHub 规范化后的请求采用相同数值。
    for (const auto& list : impl_->lists) {
        const double radius = std::max(list->request.cutoff, NeighHub::canonicalize_cutoff(list->request.cutoff));
        const double halo = (list->request.ghost_centers ? 2.0 : 1.0)*radius + impl_->lmp->neighbor->skin;
        impl_->lmp->comm->cutghostuser = std::max(impl_->lmp->comm->cutghostuser,
                                                std::nextafter(halo, std::numeric_limits<double>::infinity()));
    }
}
LammpsEngineAdapter::SyncStats LammpsEngineAdapter::sync_stats() const { return impl_->stats; }
void LammpsEngineAdapter::begin_evaluation() {
    auto& state = *impl_;
    if (state.force_batch_active)
        throw std::logic_error("LAMMPS adapter: finish force/virial batch before next evaluation");
    if (state.evaluation == std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("LAMMPS adapter: evaluation counter exhausted");
    ++state.evaluation;
#ifdef ZQC_LAMMPS_KOKKOS
    state.kokkos->sync_metadata();
#endif
    state.host_positions_ready = state.device_positions_ready = state.device_types_ready = false;
}
std::uint64_t LammpsEngineAdapter::evaluation_id() const { return impl_->evaluation; }
int LammpsEngineAdapter::owned_atom_count() const { return impl_->lmp->atom->nlocal; }
int LammpsEngineAdapter::total_atom_count() const {
    const auto* atom = impl_->lmp->atom;
    if (atom->nlocal < 0 || atom->nghost < 0 || atom->nghost > INT_MAX-atom->nlocal)
        throw std::length_error("LAMMPS adapter: local atom count overflow");
    return atom->nlocal+atom->nghost;
}

ArrayView<const double> LammpsEngineAdapter::positions(MemorySpace memory) {
    auto& state = *impl_;
    state.require_evaluation();
    const size_t atom_count = total_atom_count();
#ifdef ZQC_LAMMPS_KOKKOS
    if (state.kokkos->active() && memory == MemorySpace::Cuda)
        return state.kokkos->positions(atom_count);
    state.kokkos->sync_host_positions();
#endif
    if (!state.host_positions_ready) {
        state.host_positions.resize(3*atom_count);
        for (size_t atom = 0; atom < atom_count; ++atom)
            std::copy_n(state.lmp->atom->x[atom], 3, state.host_positions.data()+3*atom);
        state.host_positions_ready = true;
    }
    if (memory == MemorySpace::Host)
        return {state.host_positions.data(), atom_count, 3, 3, 1, state.host_positions.size(), memory};
    if (memory != MemorySpace::Cuda) throw std::invalid_argument("LAMMPS adapter: unknown memory space");
    if (!state.device_positions_ready) {
        upload(state.device_positions, state.host_positions, state.transfer.get());
        ++state.stats.position_uploads;
        state.device_positions_ready = true;
    }
    return state.device_positions.view(atom_count, 3);
}
ArrayView<const int> LammpsEngineAdapter::atom_types(MemorySpace memory) {
    auto& state = *impl_;
    state.require_evaluation();
    const size_t atom_count = total_atom_count();
#ifdef ZQC_LAMMPS_KOKKOS
    if (state.kokkos->active() && memory == MemorySpace::Cuda)
        return state.kokkos->types(atom_count);
#endif
    if (memory == MemorySpace::Host)
        return {state.lmp->atom->type, atom_count, 1, 1, 1, atom_count, memory};
    if (memory != MemorySpace::Cuda) throw std::invalid_argument("LAMMPS adapter: unknown memory space");
    if (!state.device_types_ready) {
        // 类型可由 atom/swap 等 Fix 在邻居重建间隔内改变，逐次比较请求的数据。
        const int* types = state.lmp->atom->type;
        if (state.host_types.size() != atom_count ||
            !std::equal(state.host_types.begin(), state.host_types.end(), types)) {
            state.host_types.assign(types, types+atom_count);
            upload(state.device_types, state.host_types, state.transfer.get());
            ++state.stats.type_uploads;
        }
        state.device_types_ready = true;
    }
    return device_vector(state.device_types, atom_count);
}

ArrayView<const AtomId> LammpsEngineAdapter::atom_ids() {
    auto& state=*impl_;
    state.require_evaluation();
    if(state.ids_epoch!=state.topology_epoch) {
        const size_t count=total_atom_count();
        state.host_ids.resize(count);
        for(size_t atom=0;atom<count;++atom) state.host_ids[atom]=state.lmp->atom->tag[atom];
        state.ids_epoch=state.topology_epoch;
        ++state.stats.id_rebuilds;
    }
    return {state.host_ids.data(),state.host_ids.size(),1,1,1,state.host_ids.size(),MemorySpace::Host};
}
int LammpsEngineAdapter::atom_type_count() const { return impl_->lmp->atom->ntypes; }
int LammpsEngineAdapter::atom_type_id(const std::string& name) const {
    const int type = LAMMPS_NS::utils::expand_type_int(FLERR, name, LAMMPS_NS::Atom::ATOM, impl_->lmp);
    if (type < 1 || type > atom_type_count()) throw std::invalid_argument("Unknown atom type: " + name);
    return type;
}
SelectionId LammpsEngineAdapter::selection_id(const std::string& name) const {
    const int selection = impl_->lmp->group->find(name);
    if (selection < 0) throw std::invalid_argument("Unknown atom group: " + name);
    return selection;
}
SelectionView LammpsEngineAdapter::selection_members(SelectionId selection_id) {
    auto& state = *impl_;
    state.require_evaluation();
    const int mask = state.selection_mask(selection_id);
    auto& storage = state.selections[selection_id];
    if (!storage) {
        storage = std::make_unique<Impl::Selection>();
        storage->device_flags.set_name("selection_flags", state.logger);
    }
    auto& selection = *storage;
    const bool implicit_all = selection_id == all_atoms || selection_id == 0;
    const bool dynamic = !implicit_all && state.lmp->group->dynamic[selection_id];
    if (selection.topology_epoch != state.topology_epoch ||
        (dynamic && selection.evaluation != state.evaluation)) {
        std::vector<int> local_ids;
        int owned_count = 0;
        const int total = total_atom_count(), owned = owned_atom_count();
        if (!implicit_all) ++state.stats.selection_scans;
        for (int local = 0; local < total; ++local) {
            if (!implicit_all && !(state.lmp->atom->mask[local] & mask)) continue;
            local_ids.push_back(local);
            if (local < owned) ++owned_count;
        }
        // 相同整数 localid 在排序后可能指向不同原子，epoch 变化同样推进版本。
        if (selection.topology_epoch != state.topology_epoch ||
            selection.owned_count != owned_count || selection.local_ids != local_ids) {
            selection.local_ids = std::move(local_ids);
            selection.owned_count = owned_count;
            ++selection.generation;
        }
        selection.topology_epoch = state.topology_epoch;
    }
    selection.evaluation = state.evaluation;
    return {selection_id, selection.owned_count,
            {selection.local_ids.data(), selection.local_ids.size(), 1, 1, 1,
             selection.local_ids.size(), MemorySpace::Host}, selection.generation};
}
ArrayView<const int> LammpsEngineAdapter::selection_flags(SelectionId selection_id, MemorySpace memory) {
    selection_members(selection_id);
    auto& selection = *impl_->selections.at(selection_id);
    if (selection.flags_generation != selection.generation) {
        std::vector<int> flags(total_atom_count(), 0);
        for (int local : selection.local_ids) flags[local] = 1;
        if (flags != selection.flags) {
            selection.flags = std::move(flags);
            selection.device_ready = false;
        }
        selection.flags_generation = selection.generation;
    }
    if (memory == MemorySpace::Host)
        return {selection.flags.data(), selection.flags.size(), 1, 1, 1, selection.flags.size(), memory};
    if (memory != MemorySpace::Cuda) throw std::invalid_argument("LAMMPS adapter: unknown memory space");
    if (!selection.device_ready) {
        upload(selection.device_flags, selection.flags, impl_->transfer.get());
        ++impl_->stats.selection_flag_uploads;
        selection.device_ready = true;
    }
    return device_vector(selection.device_flags, selection.flags.size());
}
Logger LammpsEngineAdapter::logger() const { return impl_->logger; }
int LammpsEngineAdapter::rank() const { return impl_->lmp->comm->me; }
std::int64_t LammpsEngineAdapter::timestep() const { return impl_->lmp->update->ntimestep; }
void LammpsEngineAdapter::communicate_forward(AtomCommunication& packet) {
    impl_->fix->communicate_forward(packet);
}
void LammpsEngineAdapter::communicate_reverse(AtomCommunication& packet) {
    impl_->fix->communicate_reverse(packet);
}

int LammpsEngineAdapter::register_neighbors(const NeighborRequest& request) {
    request.validate();
    auto& state = *impl_;
    if (state.evaluation) throw std::logic_error("LAMMPS adapter: register neighbors before the first evaluation");
    state.selection_mask(request.centers);
    state.selection_mask(request.neighbors);
    auto list = std::make_unique<Impl::Neighbors>();
    list->request = request;
    NeighSpec native;
    native.occasional = 0; // 宿主每次重建时生成带 skin 的候选表。
    const auto canonical_selection = [](SelectionId selection) { return selection == all_atoms ? 0 : selection; };
    list->native_half = request.kind == NeighborKind::Half &&
        canonical_selection(request.centers) == canonical_selection(request.neighbors);
    // 异组选对保留 full 候选，按合法方向去重。原生 half 任意选择的方向可能被组筛选丢弃。
    if (request.kind == NeighborKind::Half && !list->native_half && !state.lmp->atom->tag_enable)
        throw std::invalid_argument("LAMMPS adapter: Half lists with different selections require atom IDs");
    native.full = list->native_half ? 0 : 1;
    native.newton = list->native_half ? 1 : 0; // Half 的唯一归属独立于宿主 pair 的 newton 设置。
    native.ghost = request.ghost_centers ? 1 : 0; native.use_cutoff = 1;
    native.cutoff = NeighHub::canonicalize_cutoff(request.cutoff);
    state.lmp->comm->cutghostuser = std::max(state.lmp->comm->cutghostuser,
        std::nextafter((request.ghost_centers ? 2.0 : 1.0)*std::max(request.cutoff,native.cutoff)+state.lmp->neighbor->skin,
                       std::numeric_limits<double>::infinity()));
    list->native_handle = state.fix->neigh_hub.get_or_create(native);
    list->device_centers.set_name("neighbor_centers", state.logger);
    list->device_offsets.set_name("neighbor_offsets", state.logger);
    list->device_neighbors.set_name("neighbor_candidates", state.logger);
    state.lists.push_back(std::move(list));
    return static_cast<int>(state.lists.size()-1);
}

NeighborListView LammpsEngineAdapter::prepare_neighbors(int handle) {
    auto& state = *impl_;
    state.require_evaluation();
    if (handle < 0 || static_cast<size_t>(handle) >= state.lists.size())
        throw std::out_of_range("LAMMPS adapter: invalid neighbor handle");
    auto& list = *state.lists[handle];
    const int owned = owned_atom_count(), total = total_atom_count();
    if (list.evaluation != state.evaluation) {
        const auto selection_generation = [&](SelectionId selection) {
            // 全体原子的成员资格由布局决定，邻居过滤无需生成额外的 dense 标记或成员表。
            if (selection == all_atoms || selection == 0) return state.topology_epoch;
            selection_members(selection);
            return state.selections.at(selection)->generation;
        };
        const auto center_generation = selection_generation(list.request.centers);
        const auto neighbor_generation = selection_generation(list.request.neighbors);
        const bool dependencies_changed = list.topology_epoch != state.topology_epoch ||
            list.center_generation != center_generation || list.neighbor_generation != neighbor_generation;
        if (dependencies_changed) {
#ifdef ZQC_LAMMPS_KOKKOS
            // Host 候选筛选/ghost 补行仅在拓扑或组成员变化时读取坐标。
            state.kokkos->sync_host_positions();
#endif
            ++state.stats.neighbor_rebuilds;
            state.fix->neigh_hub.ensure(list.native_handle);
            const auto* native = state.fix->neigh_hub.list(list.native_handle);
            if (!native) throw std::runtime_error("LAMMPS adapter: neighbor list is unavailable");
            const int center_mask = state.selection_mask(list.request.centers);
            const int neighbor_mask = state.selection_mask(list.request.neighbors);
            const bool all_centers = list.request.centers == all_atoms || list.request.centers == 0;
            const bool all_neighbors = list.request.neighbors == all_atoms || list.request.neighbors == 0;
            const auto* atom = state.lmp->atom;
            std::vector<int> centers, offsets(1, 0), neighbors;
            // NEIGHMASK 只在读取宿主邻居表时去掉 special-bond 编码；设备 CSR 保存纯 localid。
            const int listed_count = native->inum + (list.request.ghost_centers ? native->gnum : 0);
            if (listed_count < 0 || listed_count > total)
                throw std::runtime_error("LAMMPS adapter: invalid neighbor center count");
            std::vector<unsigned char> listed(static_cast<size_t>(total), 0);
            for (int entry = 0; entry < listed_count; ++entry) {
                const int local = native->ilist[entry];
                if (local < 0 || local >= total) throw std::runtime_error("LAMMPS adapter: invalid neighbor center");
                listed[local] = 1;
            }
            // 每个 owned 中心的候选 J 都可能成为后续动态组需要的中心。
            // 在组筛选前补齐这些 ghost 行，成员变化时只需重组输出 CSR。
            if (list.request.ghost_centers && list.ghost_rows_epoch != state.topology_epoch) {
                const double skin = state.lmp->neighbor->skin;
                const double native_radius = list.request.cutoff + skin;
                const double native_radius_sq = native_radius*native_radius;
                list.repaired_ghost.assign(static_cast<size_t>(total), 0);
                list.ghost_offsets.assign(static_cast<size_t>(total)+1, 0);
                list.ghost_neighbors.clear();
                bool needs_ghost_rebuild = false;
                for (int center = 0; center < owned; ++center) {
                    if (!listed[center]) throw std::runtime_error("LAMMPS adapter: missing owned neighbor row");
                    for (int entry = 0; entry < native->numneigh[center]; ++entry) {
                        const int neighbor = native->firstneigh[center][entry] & NEIGHMASK;
                        if (neighbor < 0 || neighbor >= total) throw std::runtime_error("LAMMPS adapter: invalid neighbor index");
                        if (neighbor < owned) continue;
                        bool incomplete = !listed[neighbor];
                        for (int type = 1; type <= atom->ntypes && !incomplete; ++type)
                            incomplete = state.lmp->neighbor->cutneighghostsq[atom->type[neighbor]][type] < native_radius_sq;
                        if (incomplete) { list.repaired_ghost[neighbor] = 1; needs_ghost_rebuild = true; }
                    }
                }
                if (needs_ghost_rebuild) {
                    // 宿主重建步使用 R+skin。若首次需求迟于该步，两个时刻相对
                    // 同一宿主参考构型的 pair 位移可合计达到 2*skin，扩大候选壳层。
                    // 仍沿用宿主 epoch 作为寿命；NPT 的盒子位移计入宿主 skin 检查。
                    const bool at_native_build = state.lmp->neighbor->lastcall == state.lmp->update->ntimestep;
                    const double candidate_radius = list.request.cutoff + (at_native_build ? skin : 2.0*skin);
                    const double candidate_radius_sq = candidate_radius*candidate_radius;
                    using Cell = std::array<long long,3>;
                    const auto cell_of = [&](int local) {
                        return Cell{static_cast<long long>(std::floor(atom->x[local][0]/candidate_radius)),
                                    static_cast<long long>(std::floor(atom->x[local][1]/candidate_radius)),
                                    static_cast<long long>(std::floor(atom->x[local][2]/candidate_radius))};
                    };
                    std::map<Cell,std::vector<int>> cells;
                    for (int local = 0; local < total; ++local) cells[cell_of(local)].push_back(local);
                    for (int center = owned; center < total; ++center) {
                        if (list.repaired_ghost[center]) {
                            const Cell cell = cell_of(center);
                            for (int dx = -1; dx <= 1; ++dx)
                                for (int dy = -1; dy <= 1; ++dy)
                                    for (int dz = -1; dz <= 1; ++dz) {
                                        const auto found = cells.find(Cell{cell[0]+dx,cell[1]+dy,cell[2]+dz});
                                        if (found == cells.end()) continue;
                                        for (int neighbor : found->second) {
                                            if (neighbor == center || excluded_neighbor(atom,state.lmp->neighbor,center,neighbor)) continue;
                                            double distance_sq = 0;
                                            for (int axis = 0; axis < 3; ++axis) {
                                                const double delta = atom->x[neighbor][axis]-atom->x[center][axis];
                                                distance_sq += delta*delta;
                                            }
                                            if (distance_sq <= candidate_radius_sq) {
                                                if (list.ghost_neighbors.size() >= static_cast<size_t>(INT_MAX))
                                                    throw std::length_error("LAMMPS adapter: ghost neighbor offsets exceed int range");
                                                list.ghost_neighbors.push_back(neighbor);
                                            }
                                        }
                                    }
                        }
                        list.ghost_offsets[center+1] = static_cast<int>(list.ghost_neighbors.size());
                    }
                    ++state.stats.ghost_row_rebuilds;
                }
                list.ghost_rows_epoch = state.topology_epoch;
            }
            const int center_limit = list.request.ghost_centers ? total : owned;
            for (int center = 0; center < center_limit; ++center) {
                if (center < owned && !all_centers && !(atom->mask[center] & center_mask)) continue;
                const bool repaired = list.request.ghost_centers && list.repaired_ghost[center];
                if (!listed[center] && !repaired) {
                    if (center < owned) throw std::runtime_error("LAMMPS adapter: requested center missing from list");
                    continue;
                }
                centers.push_back(center);
                const size_t neighbor_count = repaired ?
                    static_cast<size_t>(list.ghost_offsets[center+1]-list.ghost_offsets[center]) :
                    static_cast<size_t>(native->numneigh[center]);
                if (neighbor_count > static_cast<size_t>(INT_MAX)) throw std::length_error("LAMMPS adapter: invalid neighbor count");
                for (size_t entry = 0; entry < neighbor_count; ++entry) {
                    const int neighbor = repaired ? list.ghost_neighbors[list.ghost_offsets[center]+entry] :
                        (native->firstneigh[center][entry] & NEIGHMASK);
                    if (neighbor < 0 || neighbor >= total) throw std::runtime_error("LAMMPS adapter: invalid neighbor local index");
                    if (!all_neighbors && !(atom->mask[neighbor] & neighbor_mask)) continue;
                    if (list.request.kind == NeighborKind::Half && !list.native_half) {
                        const bool reverse_selected = (all_centers || (atom->mask[neighbor] & center_mask)) &&
                                                      (all_neighbors || (atom->mask[center] & neighbor_mask));
                        if (reverse_selected) {
                            if (atom->tag[center] > atom->tag[neighbor]) continue;
                            if (atom->tag[center] == atom->tag[neighbor]) {
                                // 相同 ID 的周期自镜像仍是不同 pair；按镜像位移选择一侧。
                                // 判断只在重建候选表时执行，普通跨步求值复用 CSR。
                                bool positive_image = false;
                                for (int axis = 2; axis >= 0; --axis) {
                                    const double displacement = atom->x[neighbor][axis]-atom->x[center][axis];
                                    if (displacement != 0) { positive_image = displacement > 0; break; }
                                }
                                if (!positive_image) continue;
                            }
                        }
                    }
                    if (neighbors.size() >= static_cast<size_t>(INT_MAX))
                        throw std::length_error("LAMMPS adapter: neighbor offsets exceed int range");
                    neighbors.push_back(neighbor);
                }
                offsets.push_back(static_cast<int>(neighbors.size()));
            }
            const bool arrays_changed = !list.topology_generation || list.centers != centers ||
                list.offsets != offsets || list.neighbors != neighbors;
            if (arrays_changed) {
                list.centers = std::move(centers);
                list.offsets = std::move(offsets);
                list.neighbors = std::move(neighbors);
                upload(list.device_centers, list.centers, state.transfer.get());
                upload(list.device_offsets, list.offsets, state.transfer.get());
                upload(list.device_neighbors, list.neighbors, state.transfer.get());
                ++state.stats.neighbor_uploads;
            }
            if (dependencies_changed || arrays_changed) ++list.topology_generation;
            list.topology_epoch = state.topology_epoch;
            list.center_generation = center_generation;
            list.neighbor_generation = neighbor_generation;
        }
        list.evaluation = state.evaluation;
    }
    return {{list.centers.data(), list.centers.size(), 1, 1, 1, list.centers.size(), MemorySpace::Host},
            device_vector(list.device_centers, list.centers.size()),
            device_vector(list.device_offsets, list.offsets.size()),
            device_vector(list.device_neighbors, list.neighbors.size()), owned, total,
            state.evaluation, list.topology_generation, NeighborCoordinates::PeriodicImages,
            {list.offsets.data(), list.offsets.size(), 1, 1, 1, list.offsets.size(), MemorySpace::Host},
            {list.neighbors.data(), list.neighbors.size(), 1, 1, 1, list.neighbors.size(), MemorySpace::Host},
            list.request.kind, list.request.kind == NeighborKind::Half ?
                NeighborPairOwnership::UniqueAcrossRanks : NeighborPairOwnership::DirectedByCenter};
}

void LammpsEngineAdapter::read_owned_atoms(const AtomId* ids, size_t count, double* values, int* indices) {
    impl_->require_evaluation();
    auto& state = *impl_;
    const auto* atom = state.lmp->atom;
#ifdef ZQC_LAMMPS_KOKKOS
    state.kokkos->sync_host_positions();
#endif
    if (!atom->tag_enable) throw std::runtime_error("LAMMPS adapter: atom IDs are disabled");
    if (state.owned_ids_epoch != state.topology_epoch) {
        state.owned_local_ids.clear();
        for (int local = 0; local < atom->nlocal; ++local)
            if (!state.owned_local_ids.emplace(static_cast<AtomId>(atom->tag[local]), local).second)
                throw std::runtime_error("LAMMPS adapter: duplicate owned atom ID");
        state.owned_ids_epoch = state.topology_epoch;
        ++state.stats.id_rebuilds;
    }
    std::fill_n(values, 4*count, 0.0); std::fill_n(indices, count, -1);
    for (size_t endpoint = 0; endpoint < count; ++endpoint) {
        const auto found = state.owned_local_ids.find(ids[endpoint]);
        if (found == state.owned_local_ids.end()) continue;
        const int local = found->second;
        indices[endpoint] = local;
        for (int axis = 0; axis < 3; ++axis) values[4*endpoint+axis] = atom->x[local][axis];
        values[4*endpoint+3] = 1.0;
    }
}
void LammpsEngineAdapter::sum(double* values, size_t count) {
    if (count > static_cast<size_t>(INT_MAX)) throw std::length_error("LAMMPS adapter: reduction too large");
    if (count && MPI_Allreduce(MPI_IN_PLACE, values, static_cast<int>(count), MPI_DOUBLE, MPI_SUM,
                               impl_->lmp->world) != MPI_SUCCESS)
        throw std::runtime_error("LAMMPS adapter: MPI sum failed");
}
namespace {
// 兼容 29Aug2024 的 minimum_image(double*) 与 22Jul2025 的 minimum_image(FLERR,double*)。
template<class Domain>
auto call_minimum_image(Domain* domain, double* displacement, int)
    -> decltype(domain->minimum_image(displacement), void()) {
    domain->minimum_image(displacement);
}
template<class Domain>
auto call_minimum_image(Domain* domain, double* displacement, long)
    -> decltype(domain->minimum_image(FLERR, displacement), void()) {
    domain->minimum_image(FLERR, displacement);
}
}
void LammpsEngineAdapter::minimum_image(double displacement[3]) const {
    call_minimum_image(impl_->lmp->domain, displacement, 0);
}
int LammpsEngineAdapter::reverse_communication_width() const {
    return impl_->lists.empty() ? 0 : 9;
}
void LammpsEngineAdapter::configure_force_virial_batch(bool global_virial, bool atom_virial) {
    impl_->batch_global_virial = global_virial;
    impl_->batch_atom_virial = atom_virial;
    impl_->device_virial_complete = false;
}
bool LammpsEngineAdapter::force_virial_batch_device_complete() const {
    return impl_->device_virial_complete;
}
void LammpsEngineAdapter::begin_force_virial_batch() {
    auto& state = *impl_;
    state.require_evaluation();
    if (state.force_batch_active)
        throw std::logic_error("LAMMPS adapter: nested force/virial batch");
#ifdef ZQC_LAMMPS_KOKKOS
    if (state.kokkos->force_batch_active())
        throw std::logic_error("LAMMPS adapter: nested device force/virial batch");
#endif
    const size_t total = total_atom_count();
    state.output.forces.assign(total*3, 0.0);
    state.output.virials.assign(total*6, 0.0);
    state.force_batch_has_contributions = state.force_batch_needs_reverse = false;
    state.force_batch_host_contributions = false;
    state.device_virial_complete = false;
    // 设备批次延后到首个纯 Cuda 提交再打开，owned-only Host 批次可跳过 Weave reverse。
    state.force_batch_active = true;
}
void LammpsEngineAdapter::finish_force_virial_batch() {
    auto& state = *impl_;
    if (!state.force_batch_active)
        throw std::logic_error("LAMMPS adapter: no force/virial batch to finish");
    // 结束路径必须关闭 open batch：成功 flush 或中途异常都走清理。
    struct Guard {
        LammpsEngineAdapter* self;
        ~Guard() { self->cancel_force_virial_batch(); }
    } guard{this};
    if (!state.force_batch_has_contributions) return;
#ifdef ZQC_LAMMPS_KOKKOS
    // Every rank selects the same backend, including ranks with no owned
    // DISTANCE endpoints. Host contributions are added to the device batch;
    // switching just one rank to Host reverse would mismatch MPI messages.
    if (state.try_open_device_batch(total_atom_count())) {
        if (state.force_batch_host_contributions)
            state.kokkos->accumulate_host_force_virial(state.output.forces.data(),
                state.output.virials.data(), total_atom_count());
        state.finish_device_batch(state.force_batch_needs_reverse);
        return;
    }
#endif
    // 包括本 rank 没有中心的情况：批量提交仍置 needs_reverse，保证各 rank 通信顺序一致。
    state.flush_force_virial(state.force_batch_needs_reverse);
    state.device_virial_complete = false;
}
void LammpsEngineAdapter::cancel_force_virial_batch() noexcept {
#ifdef ZQC_LAMMPS_KOKKOS
    if (impl_->kokkos) impl_->kokkos->cancel_force_batch();
#endif
    impl_->clear_force_batch_flags();
}
void LammpsEngineAdapter::commit_force_virial(ArrayView<const double> forces, VirialView virials) {
    auto& state = *impl_;
    state.require_evaluation();
    const size_t total = total_atom_count();
    forces.validate();
    virials.validate();
    if (forces.item_count != total || virials.values.item_count != total || forces.component_count != 3)
        throw std::invalid_argument("LAMMPS adapter: invalid force/virial shape");
    // 当前 LAMMPS tally 使用 r tensor F 的 xx yy zz xy xz yz 六项。
    // Full9 保持原 stride，只选取前六列；此操作是投影，不对转置项取平均。
    auto lammps_virials = virials.values;
    lammps_virials.component_count = 6;
    const bool cuda_pair = forces.memory == MemorySpace::Cuda &&
                           lammps_virials.memory == MemorySpace::Cuda;
    if (state.force_batch_active) {
#ifdef ZQC_LAMMPS_KOKKOS
        if (cuda_pair && state.try_open_device_batch(total)) {
            state.kokkos->accumulate_force_virial(forces, lammps_virials);
            state.force_batch_has_contributions = state.force_batch_needs_reverse = true;
            ++state.stats.force_virial_submissions;
            return;
        }
#endif
        accumulate_output(forces, state.output.forces, state.force_staging, state.transfer.get());
        accumulate_output(lammps_virials, state.output.virials, state.virial_staging, state.transfer.get());
        state.force_batch_has_contributions = state.force_batch_needs_reverse = true;
        state.force_batch_host_contributions = true;
    } else {
#ifdef ZQC_LAMMPS_KOKKOS
        // 批次外的纯 Cuda 提交：一次性设备合批，语义同单 CV Graph 批次。
        if (cuda_pair && state.try_open_device_batch(total)) {
            try {
                state.kokkos->accumulate_force_virial(forces, lammps_virials);
                state.finish_device_batch(true);
            } catch (...) {
                state.kokkos->cancel_force_batch();
                throw;
            }
            ++state.stats.force_virial_submissions;
            return;
        }
#endif
        download(forces, state.output.forces, state.transfer.get());
        download(lammps_virials, state.output.virials, state.transfer.get());
        state.flush_force_virial(true);
        state.device_virial_complete = false;
    }
    ++state.stats.force_virial_submissions;
}
void LammpsEngineAdapter::add_owned_force_virial(int local, const double force[3], VirialView virial) {
    impl_->require_evaluation();
    virial.validate();
    virial.values.wait_host();
    if (!force || virial.values.item_count != 1 || virial.values.memory != MemorySpace::Host)
        throw std::invalid_argument("LAMMPS adapter: owned contribution requires one Host row");
    if (local < 0 || local >= owned_atom_count()) throw std::out_of_range("LAMMPS adapter: force target is not owned");
    if (impl_->force_batch_active) {
        for (int component = 0; component < 3; ++component)
            impl_->output.forces[3*local+component] += force[component];
        for (int component = 0; component < 6; ++component)
            impl_->output.virials[6*local+component] += virial.values(0,component);
        impl_->force_batch_has_contributions = true;
        impl_->force_batch_host_contributions = true;
        // 纯 owned 不置 needs_reverse；若先前设备提交已置位则保持。
        return;
    }
#ifdef ZQC_LAMMPS_KOKKOS
    if (impl_->kokkos->active()) impl_->kokkos->add_force(local, force);
    else
#endif
        for (int axis = 0; axis < 3; ++axis) impl_->lmp->atom->f[local][axis] += force[axis];
    if (impl_->fix->a_virial)
        for (int component = 0; component < 6; ++component)
            impl_->fix->a_virial[6*local+component] += virial.values(0,component);
}
} // namespace MetaD_zqc
