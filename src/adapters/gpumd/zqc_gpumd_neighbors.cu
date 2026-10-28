#include "zqc_gpumd_neighbors.cuh"
#include "zqc_cuda_execution.cuh"
#include "zqc_gpu_buffer.cuh"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace MetaD_zqc {
namespace {

void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

void launch_1d(size_t count, dim3& grid, dim3& block) {
    block = dim3(128, 1, 1);
    grid = dim3(count ? static_cast<unsigned>((count + block.x - 1) / block.x) : 0u, 1, 1);
}

// 正交盒最小镜像；非周期轴原样保留。供 Host 重建与 GPU 复用同一约定。
__host__ __device__ inline void minimum_image_orthogonal(
    double delta[3], const double box[3], const int periodic[3]) {
    for (int axis = 0; axis < 3; ++axis) {
        if (!periodic[axis]) continue;
        const double length = box[axis];
        if (!(length > 0)) continue;
        delta[axis] -= length * floor(delta[axis] / length + 0.5);
    }
}

__host__ __device__ inline double displacement_norm_sq(
    const double a[3], const double b[3], const double box[3], const int periodic[3]) {
    double delta[3] = {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
    minimum_image_orthogonal(delta, box, periodic);
    return delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
}

// 将任意 stride 的 MD 坐标打成连续 AoS，避免 Host 逐分量 D2H。
__global__ void pack_positions_aos_kernel(
    ArrayView<const double> source, double* packed_aos, size_t atom_count) {
    const size_t atom = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (atom >= atom_count) return;
    packed_aos[3 * atom + 0] = source(atom, 0);
    packed_aos[3 * atom + 1] = source(atom, 1);
    packed_aos[3 * atom + 2] = source(atom, 2);
}

__global__ void pack_types_kernel(
    ArrayView<const int> source, int* packed, size_t atom_count) {
    const size_t atom = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (atom >= atom_count) return;
    packed[atom] = source(atom);
}

// 相对重建参考构型的 MIC 位移；超过阈值或坐标非有限时置位，Host 下载一个标志。
__global__ void displacement_exceeds_kernel(
    ArrayView<const double> current, const double* reference_aos,
    const double* box, const int* periodic, size_t owned_count,
    double threshold_sq, int* exceeded) {
    const size_t atom = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (atom >= owned_count) return;
    const double current_xyz[3] = {current(atom, 0), current(atom, 1), current(atom, 2)};
    const double reference_xyz[3] = {
        reference_aos[3 * atom + 0], reference_aos[3 * atom + 1], reference_aos[3 * atom + 2]};
    const double norm_sq = displacement_norm_sq(current_xyz, reference_xyz, box, periodic);
    if (!isfinite(current_xyz[0]) || !isfinite(current_xyz[1]) || !isfinite(current_xyz[2]) ||
        !isfinite(norm_sq)) atomicMax(exceeded, 2);
    else if (norm_sq > threshold_sq) atomicMax(exceeded, 1);
}

// 缓存命中：owned = reference + MIC(current-reference)；ghost = owned[owner] + shift*box。
// 参考构型保持重建时快照，不在每步推进，从而固定镜像晶格位移在 skin 内仍有效。
__global__ void update_expanded_geometry_kernel(
    ArrayView<const double> current, ArrayView<const int> current_types,
    const double* reference_aos, const int* owners, const int* shifts_aos,
    const double* box, const int* periodic,
    size_t owned_count, size_t total_count,
    double* expanded_positions_aos, int* expanded_types) {
    const size_t local = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (local >= total_count) return;
    const int owner = owners[local];
    double owned_xyz[3];
    if (static_cast<size_t>(owner) < owned_count) {
        const double cur[3] = {current(owner, 0), current(owner, 1), current(owner, 2)};
        const double ref[3] = {
            reference_aos[3 * owner + 0], reference_aos[3 * owner + 1], reference_aos[3 * owner + 2]};
        double delta[3] = {cur[0] - ref[0], cur[1] - ref[1], cur[2] - ref[2]};
        minimum_image_orthogonal(delta, box, periodic);
        owned_xyz[0] = ref[0] + delta[0];
        owned_xyz[1] = ref[1] + delta[1];
        owned_xyz[2] = ref[2] + delta[2];
    } else {
        owned_xyz[0] = owned_xyz[1] = owned_xyz[2] = 0;
    }
    if (local < owned_count) {
        expanded_positions_aos[3 * local + 0] = owned_xyz[0];
        expanded_positions_aos[3 * local + 1] = owned_xyz[1];
        expanded_positions_aos[3 * local + 2] = owned_xyz[2];
        expanded_types[local] = current_types(local);
        return;
    }
    expanded_positions_aos[3 * local + 0] =
        owned_xyz[0] + static_cast<double>(shifts_aos[3 * local + 0]) * box[0];
    expanded_positions_aos[3 * local + 1] =
        owned_xyz[1] + static_cast<double>(shifts_aos[3 * local + 1]) * box[1];
    expanded_positions_aos[3 * local + 2] =
        owned_xyz[2] + static_cast<double>(shifts_aos[3 * local + 2]) * box[2];
    expanded_types[local] = current_types(owner);
}

bool finite_vec3(const double* xyz) {
    return std::isfinite(xyz[0]) && std::isfinite(xyz[1]) && std::isfinite(xyz[2]);
}

// 将周期坐标折入 [0,L)；非周期轴保留。拒绝非有限坐标与非正边长。
void canonicalize_owned(std::vector<double>& aos, int owned, const std::array<double, 3>& box,
                        const std::array<int, 3>& periodic) {
    for (int atom = 0; atom < owned; ++atom) {
        double* xyz = aos.data() + 3 * atom;
        if (!finite_vec3(xyz))
            throw std::invalid_argument("GpumdNeighbors: non-finite owned coordinate");
        for (int axis = 0; axis < 3; ++axis) {
            if (!periodic[axis]) continue;
            const double length = box[axis];
            double value = xyz[axis] / length;
            value -= std::floor(value);
            xyz[axis] = value * length;
            if (xyz[axis] >= length) xyz[axis] = 0; // 数值上贴上边界时归零
            if (!std::isfinite(xyz[axis]))
                throw std::invalid_argument("GpumdNeighbors: non-finite canonical coordinate");
        }
    }
}

bool shift_is_positive(const int shift[3]) {
    for (int axis = 0; axis < 3; ++axis) {
        if (shift[axis] != 0) return shift[axis] > 0;
    }
    return false;
}

} // namespace

struct GpumdNeighbors::Impl {
    Logger logger;
    double skin = 0.3;
    bool geometry_ready = false;
    bool host_positions_cached = false;
    bool host_types_cached = false;
    bool structural_dirty = true;
    std::uint64_t evaluation = 0;
    std::uint64_t rebuilds = 0;
    int owned_count = 0;
    int total_count = 0;
    std::array<double, 3> box{{0, 0, 0}};
    std::array<int, 3> periodic{{0, 0, 0}};

    // Host：重建时的规范 owned 坐标；复用时 GPU 以之为参考，不在每步改写。
    std::vector<double> host_reference_aos;
    std::vector<double> host_expanded_aos;   // 懒下载整包
    std::vector<int> host_expanded_types;
    std::vector<int> host_owners;            // localid -> owner row
    std::vector<int> host_shifts_aos;        // localid*3 -> 整数晶格位移（owned 为 0）
    std::vector<AtomId> host_ids;
    // all_atoms 视图的持久快照，避免 SelectionView 悬空到临时/thread_local 缓冲。
    std::vector<int> all_atoms_local_ids;
    std::vector<int> all_atoms_flags;
    GpuBuffer<int> device_all_atoms_flags;
    bool all_atoms_device_ready = false;

    GpuBuffer<double> device_reference_aos;
    GpuBuffer<double> device_expanded_aos;
    GpuBuffer<int> device_expanded_types;
    GpuBuffer<int> device_owners;
    GpuBuffer<int> device_shifts_aos;
    GpuBuffer<double> device_pack_scratch;
    GpuBuffer<int> device_type_scratch;
    GpuBuffer<double> device_box;
    GpuBuffer<int> device_periodic;
    GpuBuffer<int> device_displacement_flag;

    struct Selection {
        std::string name;
        SelectionId id = all_atoms;
        std::vector<int> owned_rows; // 升序唯一 owned 成员
        std::vector<int> local_ids;  // owned 前缀 + 选中 owner 的镜像
        std::vector<int> flags;      // 长度 = total；按 owner 成员资格
        std::uint64_t generation = 0;
        bool device_ready = false;
        GpuBuffer<int> device_flags;
    };
    std::vector<std::unique_ptr<Selection>> selections;
    std::unordered_map<std::string, SelectionId> selection_names;

    struct ListCache {
        NeighborRequest request;
        std::vector<int> centers;
        std::vector<int> offsets;   // size = centers+1
        std::vector<int> neighbors;
        std::uint64_t topology_generation = 0;
        GpuBuffer<int> device_centers;
        GpuBuffer<int> device_offsets;
        GpuBuffer<int> device_neighbors;
    };
    std::vector<std::unique_ptr<ListCache>> lists;

    Impl() {
        device_reference_aos.set_name("gpumd_neigh_ref", logger);
        device_expanded_aos.set_name("gpumd_neigh_pos", logger);
        device_expanded_types.set_name("gpumd_neigh_types", logger);
        device_owners.set_name("gpumd_neigh_owners", logger);
        device_shifts_aos.set_name("gpumd_neigh_shifts", logger);
        device_pack_scratch.set_name("gpumd_neigh_pack", logger);
        device_type_scratch.set_name("gpumd_neigh_type_pack", logger);
        device_box.set_name("gpumd_neigh_box", logger);
        device_periodic.set_name("gpumd_neigh_pbc", logger);
        device_displacement_flag.set_name("gpumd_neigh_flag", logger);
        device_all_atoms_flags.set_name("gpumd_neigh_all_flags", logger);

    }

    void require_geometry() const {
        if (!geometry_ready)
            throw std::logic_error("GpumdNeighbors: update has not produced a geometry");
    }

    double max_candidate_radius() const {
        double radius = 0;
        for (const auto& list : lists)
            radius = std::max(radius, list->request.cutoff + skin);
        if (!std::isfinite(radius*radius))
            throw std::overflow_error("GpumdNeighbors: cutoff plus skin exceeds finite range");
        return radius;
    }

    bool selection_contains_owner(SelectionId id, int owner) const {
        if (id == all_atoms) return true;
        if (id < 0 || static_cast<size_t>(id) >= selections.size())
            throw std::out_of_range("GpumdNeighbors: unknown selection");
        const auto& rows = selections[static_cast<size_t>(id)]->owned_rows;
        return std::binary_search(rows.begin(), rows.end(), owner);
    }

    void validate_frame(ArrayView<const double> positions, ArrayView<const int> types,
                        const std::array<double, 3>& next_box,
                        const std::array<int, 3>& next_periodic) const {
        positions.validate();
        types.validate();
        if (positions.memory != MemorySpace::Cuda)
            throw std::invalid_argument("GpumdNeighbors: positions must be Cuda memory");
        if (types.memory != MemorySpace::Cuda)
            throw std::invalid_argument("GpumdNeighbors: types must be Cuda memory");
        if (positions.component_count != 3)
            throw std::invalid_argument("GpumdNeighbors: positions require 3 components");
        if (types.component_count != 1)
            throw std::invalid_argument("GpumdNeighbors: types require one component");
        if (positions.item_count != types.item_count)
            throw std::invalid_argument("GpumdNeighbors: positions/types atom counts differ");
        if (positions.item_count > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::length_error("GpumdNeighbors: atom count exceeds int range");
        for (int axis = 0; axis < 3; ++axis) {
            if (next_periodic[axis] != 0 && next_periodic[axis] != 1)
                throw std::invalid_argument("GpumdNeighbors: periodic flags must be 0 or 1");
            if (!std::isfinite(next_box[axis]) || !(next_box[axis] > 0))
                throw std::invalid_argument("GpumdNeighbors: box lengths must be finite and positive");
        }
    }

    bool box_or_pbc_changed(const std::array<double, 3>& next_box,
                            const std::array<int, 3>& next_periodic) const {
        return box != next_box || periodic != next_periodic;
    }

    // 在 owned AABB 外扩 max(cutoff+skin) 的周期层内枚举显式镜像；含自镜像（非零位移）。
    void expand_periodic_images(const std::vector<double>& owned_aos, double radius) {
        host_owners.assign(static_cast<size_t>(owned_count), 0);
        host_shifts_aos.assign(static_cast<size_t>(owned_count) * 3, 0);
        for (int atom = 0; atom < owned_count; ++atom) host_owners[static_cast<size_t>(atom)] = atom;

        if (!(radius > 0) || owned_count == 0) {
            total_count = owned_count;
            return;
        }

        double lo[3] = {owned_aos[0], owned_aos[1], owned_aos[2]};
        double hi[3] = {lo[0], lo[1], lo[2]};
        for (int atom = 1; atom < owned_count; ++atom) {
            for (int axis = 0; axis < 3; ++axis) {
                const double value = owned_aos[static_cast<size_t>(3 * atom + axis)];
                lo[axis] = std::min(lo[axis], value);
                hi[axis] = std::max(hi[axis], value);
            }
        }
        for (int axis = 0; axis < 3; ++axis) {
            if (periodic[axis]) {
                lo[axis] -= radius;
                hi[axis] += radius;
            }
        }

        const size_t max_total = static_cast<size_t>(std::numeric_limits<int>::max());
        for (int owner = 0; owner < owned_count; ++owner) {
            const double* xyz = owned_aos.data() + 3 * owner;
            int shift_lo[3] = {0, 0, 0};
            int shift_hi[3] = {0, 0, 0};
            for (int axis = 0; axis < 3; ++axis) {
                if (!periodic[axis]) continue;
                const double length = box[axis];
                // s*L + x ∈ [lo,hi] ⇒ s ∈ [(lo-x)/L, (hi-x)/L]
                const double s_min = (lo[axis] - xyz[axis]) / length;
                const double s_max = (hi[axis] - xyz[axis]) / length;
                if (s_min <= static_cast<double>(std::numeric_limits<int>::min()) + 2 ||
                    s_max >= static_cast<double>(std::numeric_limits<int>::max()) - 2)
                    throw std::overflow_error("GpumdNeighbors: image shift range overflows int");
                shift_lo[axis] = static_cast<int>(std::ceil(s_min - 1e-12));
                shift_hi[axis] = static_cast<int>(std::floor(s_max + 1e-12));
            }
            for (int sx = shift_lo[0]; sx <= shift_hi[0]; ++sx)
                for (int sy = shift_lo[1]; sy <= shift_hi[1]; ++sy)
                    for (int sz = shift_lo[2]; sz <= shift_hi[2]; ++sz) {
                        if (sx == 0 && sy == 0 && sz == 0) continue;
                        const double image[3] = {
                            xyz[0] + static_cast<double>(sx) * box[0],
                            xyz[1] + static_cast<double>(sy) * box[1],
                            xyz[2] + static_cast<double>(sz) * box[2]};
                        bool inside = true;
                        for (int axis = 0; axis < 3; ++axis) {
                            if (image[axis] < lo[axis] - 1e-12 || image[axis] > hi[axis] + 1e-12) {
                                inside = false;
                                break;
                            }
                        }
                        if (!inside) continue;
                        if (host_owners.size() >= max_total)
                            throw std::length_error("GpumdNeighbors: expanded atom count exceeds int");
                        host_owners.push_back(owner);
                        host_shifts_aos.push_back(sx);
                        host_shifts_aos.push_back(sy);
                        host_shifts_aos.push_back(sz);
                    }
        }
        total_count = static_cast<int>(host_owners.size());
    }

    void materialize_expanded_host(const std::vector<double>& owned_aos,
                                   const std::vector<int>& owned_types) {
        host_expanded_aos.resize(static_cast<size_t>(total_count) * 3);
        host_expanded_types.resize(static_cast<size_t>(total_count));
        host_ids.resize(static_cast<size_t>(total_count));
        for (int local = 0; local < total_count; ++local) {
            const int owner = host_owners[static_cast<size_t>(local)];
            const int* shift = host_shifts_aos.data() + 3 * local;
            for (int axis = 0; axis < 3; ++axis) {
                host_expanded_aos[static_cast<size_t>(3 * local + axis)] =
                    owned_aos[static_cast<size_t>(3 * owner + axis)] +
                    static_cast<double>(shift[axis]) * box[static_cast<size_t>(axis)];
            }
            host_expanded_types[static_cast<size_t>(local)] = owned_types[static_cast<size_t>(owner)];
            host_ids[static_cast<size_t>(local)] = static_cast<AtomId>(owner + 1);
        }
        all_atoms_local_ids.resize(static_cast<size_t>(total_count));
        all_atoms_flags.assign(static_cast<size_t>(total_count), 1);
        for (int local = 0; local < total_count; ++local)
            all_atoms_local_ids[static_cast<size_t>(local)] = local;
        all_atoms_device_ready = false;
        host_positions_cached = true;
        host_types_cached = true;
    }

    void rebuild_selection_snapshots() {
        for (auto& storage : selections) {
            auto& selection = *storage;
            std::vector<int> local_ids;
            local_ids.reserve(selection.owned_rows.size());
            for (int row : selection.owned_rows) {
                if (row < 0 || row >= owned_count)
                    throw std::out_of_range("GpumdNeighbors: selection row outside owned range");
                local_ids.push_back(row);
            }
            for (int local = owned_count; local < total_count; ++local) {
                if (selection_contains_owner(selection.id, host_owners[static_cast<size_t>(local)]))
                    local_ids.push_back(local);
            }
            std::vector<int> flags(static_cast<size_t>(total_count), 0);
            for (int local = 0; local < total_count; ++local) {
                if (selection_contains_owner(selection.id, host_owners[static_cast<size_t>(local)]))
                    flags[static_cast<size_t>(local)] = 1;
            }
            // 重建可能改变同一 localid 所指的周期像，版本随重建推进。
            selection.local_ids = std::move(local_ids);
            selection.flags = std::move(flags);
            selection.device_ready = false;
            if (selection.generation == std::numeric_limits<std::uint64_t>::max())
                throw std::overflow_error("GpumdNeighbors: selection generation exhausted");
            ++selection.generation;
        }
    }

    bool half_keep_pair(const NeighborRequest& request, int center, int neighbor) const {
        const int owner_i = host_owners[static_cast<size_t>(center)];
        const int owner_j = host_owners[static_cast<size_t>(neighbor)];
        const bool forward = selection_contains_owner(request.centers, owner_i) &&
                             selection_contains_owner(request.neighbors, owner_j);
        if (!forward) return false;
        const bool reverse = selection_contains_owner(request.centers, owner_j) &&
                             selection_contains_owner(request.neighbors, owner_i);
        if (!reverse) return true;
        const AtomId id_i = static_cast<AtomId>(owner_i + 1);
        const AtomId id_j = static_cast<AtomId>(owner_j + 1);
        if (id_i < id_j) return true;
        if (id_i > id_j) return false;
        // 同 owner 的周期自镜像：相对晶格位移取字典序“正”侧，对应 (i,j,s) 而非 (j,i,-s)。
        const int* shift_i = host_shifts_aos.data() + 3 * center;
        const int* shift_j = host_shifts_aos.data() + 3 * neighbor;
        const int relative[3] = {
            shift_j[0] - shift_i[0], shift_j[1] - shift_i[1], shift_j[2] - shift_i[2]};
        return shift_is_positive(relative);
    }

    void build_lists_host() {
        const double cell_width = max_candidate_radius();
        using Cell = std::array<long long, 3>;
        std::map<Cell, std::vector<int>> bins;
        const auto cell_of = [&](int local) {
            Cell cell{};
            for (int axis = 0; axis < 3; ++axis) {
                const double value = host_expanded_aos[static_cast<size_t>(3 * local + axis)];
                const double index=std::floor(value/cell_width);
                if (!std::isfinite(index) || std::abs(index)>=9.0e18)
                    throw std::overflow_error("GpumdNeighbors: spatial cell index exceeds range");
                cell[static_cast<size_t>(axis)] = static_cast<long long>(index);
            }
            return cell;
        };
        if (cell_width > 0) {
            for (int local = 0; local < total_count; ++local)
                bins[cell_of(local)].push_back(local);
        }

        for (auto& storage : lists) {
            auto& list = *storage;
            const double radius = list.request.cutoff + skin;
            const double radius_sq = radius * radius;
            std::vector<int> centers, offsets(1, 0), neighbors;
            std::vector<int> center_rows;
            if (list.request.centers == all_atoms) {
                center_rows.resize(static_cast<size_t>(owned_count));
                for (int row = 0; row < owned_count; ++row) center_rows[static_cast<size_t>(row)] = row;
            } else {
                if (list.request.centers < 0 ||
                    static_cast<size_t>(list.request.centers) >= selections.size())
                    throw std::out_of_range("GpumdNeighbors: request centers selection missing");
                center_rows = selections[static_cast<size_t>(list.request.centers)]->owned_rows;
            }
            for (int center : center_rows) {
                if (center < 0 || center >= owned_count)
                    throw std::out_of_range("GpumdNeighbors: center is not an owned row");
                centers.push_back(center);
                if (!(radius > 0) || bins.empty()) {
                    offsets.push_back(static_cast<int>(neighbors.size()));
                    continue;
                }
                const Cell cell = cell_of(center);
                for (int dx = -1; dx <= 1; ++dx)
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dz = -1; dz <= 1; ++dz) {
                            const auto found = bins.find(Cell{
                                cell[0] + dx, cell[1] + dy, cell[2] + dz});
                            if (found == bins.end()) continue;
                            for (int neighbor : found->second) {
                                if (neighbor == center) continue;
                                const int owner_j = host_owners[static_cast<size_t>(neighbor)];
                                if (!selection_contains_owner(list.request.neighbors, owner_j))
                                    continue;
                                if (list.request.kind == NeighborKind::Half &&
                                    !half_keep_pair(list.request, center, neighbor))
                                    continue;
                                double delta[3];
                                for (int axis = 0; axis < 3; ++axis) {
                                    delta[axis] =
                                        host_expanded_aos[static_cast<size_t>(3 * neighbor + axis)] -
                                        host_expanded_aos[static_cast<size_t>(3 * center + axis)];
                                }
                                const double distance_sq =
                                    delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
                                if (distance_sq > radius_sq) continue;
                                if (neighbors.size() >= static_cast<size_t>(std::numeric_limits<int>::max()))
                                    throw std::length_error("GpumdNeighbors: neighbor CSR exceeds int");
                                neighbors.push_back(neighbor);
                            }
                        }
                offsets.push_back(static_cast<int>(neighbors.size()));
            }
            // localid 对应的像身份也属于拓扑，重建时推进版本。
            list.centers = std::move(centers);
            list.offsets = std::move(offsets);
            list.neighbors = std::move(neighbors);
            if (list.topology_generation == std::numeric_limits<std::uint64_t>::max())
                throw std::overflow_error("GpumdNeighbors: topology generation exhausted");
            ++list.topology_generation;
        }
    }

    void upload_geometry_and_lists(cudaStream_t stream) {
        device_box.grow_to(3, __FILE__, __LINE__);
        device_periodic.grow_to(3, __FILE__, __LINE__);
        device_displacement_flag.grow_to(1, __FILE__, __LINE__);
        device_reference_aos.grow_to(static_cast<size_t>(owned_count) * 3, __FILE__, __LINE__);
        device_expanded_aos.grow_to(static_cast<size_t>(total_count) * 3, __FILE__, __LINE__);
        device_expanded_types.grow_to(static_cast<size_t>(total_count), __FILE__, __LINE__);
        device_owners.grow_to(static_cast<size_t>(total_count), __FILE__, __LINE__);
        device_shifts_aos.grow_to(static_cast<size_t>(total_count) * 3, __FILE__, __LINE__);

        device_reference_aos.upload_from(host_reference_aos.data(),
            static_cast<size_t>(owned_count) * 3, stream);
        device_expanded_aos.upload_from(host_expanded_aos.data(),
            static_cast<size_t>(total_count) * 3, stream);
        device_expanded_types.upload_from(host_expanded_types.data(),
            static_cast<size_t>(total_count), stream);
        device_owners.upload_from(host_owners.data(), static_cast<size_t>(total_count), stream);
        device_shifts_aos.upload_from(host_shifts_aos.data(),
            static_cast<size_t>(total_count) * 3, stream);
        device_box.upload_from(box.data(), 3, stream);
        device_periodic.upload_from(periodic.data(), 3, stream);

        for (auto& storage : lists) {
            auto& list = *storage;
            list.device_centers.set_name("gpumd_neigh_centers", logger);
            list.device_offsets.set_name("gpumd_neigh_offsets", logger);
            list.device_neighbors.set_name("gpumd_neigh_entries", logger);
            list.device_centers.grow_to(std::max<size_t>(list.centers.size(), 1), __FILE__, __LINE__);
            list.device_offsets.grow_to(std::max<size_t>(list.offsets.size(), 1), __FILE__, __LINE__);
            list.device_neighbors.grow_to(std::max<size_t>(list.neighbors.size(), 1), __FILE__, __LINE__);
            if (!list.centers.empty())
                list.device_centers.upload_from(list.centers.data(), list.centers.size(), stream);
            list.device_offsets.upload_from(list.offsets.data(), list.offsets.size(), stream);
            if (!list.neighbors.empty())
                list.device_neighbors.upload_from(list.neighbors.data(), list.neighbors.size(), stream);
            // 与坐标缓冲相同：显式写事件，供 CV 其它流 wait_on，不依赖默认流隐式同步。
            list.device_centers.record_write(stream);
            list.device_offsets.record_write(stream);
            list.device_neighbors.record_write(stream);
        }
        // upload_from 已在流上同步；额外 record_write 供跨流消费者 wait_on。
        device_expanded_aos.record_write(stream);
        device_expanded_types.record_write(stream);
        device_owners.record_write(stream);
    }

    void gather_owned_to_host(ArrayView<const double> positions, ArrayView<const int> types,
                              cudaStream_t stream, std::vector<double>& owned_aos,
                              std::vector<int>& owned_types) {
        const size_t count = static_cast<size_t>(owned_count);
        device_pack_scratch.grow_to(count * 3, __FILE__, __LINE__);
        device_type_scratch.grow_to(std::max<size_t>(count, 1), __FILE__, __LINE__);
        positions.wait_on(stream);
        types.wait_on(stream);
        dim3 grid, block;
        launch_1d(count, grid, block);
        if (grid.x) {
            pack_positions_aos_kernel<<<grid, block, 0, stream>>>(
                positions, device_pack_scratch.ptr, count);
            pack_types_kernel<<<grid, block, 0, stream>>>(
                types, device_type_scratch.ptr, count);
            check_cuda(cudaGetLastError());
        }
        owned_aos.resize(count * 3);
        owned_types.resize(count);
        if (count) {
            device_pack_scratch.download_to(owned_aos.data(), count * 3, stream);
            device_type_scratch.download_to(owned_types.data(), count, stream);
        }
    }

    bool displacement_requires_rebuild(ArrayView<const double> positions, cudaStream_t stream) {
        if (!(skin > 0)) return true;
        if (!owned_count) return false;
        positions.wait_on(stream);
        check_cuda(cudaMemsetAsync(device_displacement_flag.ptr,0,sizeof(int),stream));
        const double threshold_sq = (skin * 0.5) * (skin * 0.5);
        dim3 grid, block;
        launch_1d(static_cast<size_t>(owned_count), grid, block);
        if (grid.x) {
            displacement_exceeds_kernel<<<grid, block, 0, stream>>>(
                positions, device_reference_aos.ptr, device_box.ptr, device_periodic.ptr,
                static_cast<size_t>(owned_count), threshold_sq, device_displacement_flag.ptr);
            check_cuda(cudaGetLastError());
        }
        int exceeded = 0;
        device_displacement_flag.download_to(&exceeded, 1, stream);
        if (exceeded==2) throw std::invalid_argument("GpumdNeighbors: non-finite owned coordinate/displacement");
        return exceeded != 0;
    }

    void reuse_update_device(ArrayView<const double> positions, ArrayView<const int> types,
                             cudaStream_t stream) {
        positions.wait_on(stream);
        types.wait_on(stream);
        dim3 grid, block;
        launch_1d(static_cast<size_t>(total_count), grid, block);
        if (grid.x) {
            update_expanded_geometry_kernel<<<grid, block, 0, stream>>>(
                positions, types, device_reference_aos.ptr, device_owners.ptr, device_shifts_aos.ptr,
                device_box.ptr, device_periodic.ptr,
                static_cast<size_t>(owned_count), static_cast<size_t>(total_count),
                device_expanded_aos.ptr, device_expanded_types.ptr);
            check_cuda(cudaGetLastError());
        }
        device_expanded_aos.record_write(stream);
        device_expanded_types.record_write(stream);
        host_positions_cached = false;
        host_types_cached = false;
    }

    void rebuild(ArrayView<const double> positions, ArrayView<const int> types, cudaStream_t stream) {
        std::vector<double> owned_aos;
        std::vector<int> owned_types;
        gather_owned_to_host(positions, types, stream, owned_aos, owned_types);
        canonicalize_owned(owned_aos, owned_count, box, periodic);
        host_reference_aos = owned_aos;
        expand_periodic_images(owned_aos, max_candidate_radius());
        materialize_expanded_host(owned_aos, owned_types);
        rebuild_selection_snapshots();
        build_lists_host();
        upload_geometry_and_lists(stream);
        if (rebuilds == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("GpumdNeighbors: rebuild counter exhausted");
        ++rebuilds;
        structural_dirty = false;
        geometry_ready = true;
        logger.printf(LogLevel::Debug, "GpumdNeighbors", __FILE__, __LINE__,
                      "rebuild owned=%d total=%d lists=%zu skin=%g",
                      owned_count, total_count, lists.size(), skin);
    }
};

GpumdNeighbors::GpumdNeighbors(Logger logger) : impl_(new Impl) {
    impl_->logger = std::move(logger);
    impl_->device_reference_aos.set_name("gpumd_neigh_ref", impl_->logger);
    impl_->device_expanded_aos.set_name("gpumd_neigh_pos", impl_->logger);
    impl_->device_expanded_types.set_name("gpumd_neigh_types", impl_->logger);
    impl_->device_owners.set_name("gpumd_neigh_owners", impl_->logger);
    impl_->device_shifts_aos.set_name("gpumd_neigh_shifts", impl_->logger);
    impl_->device_pack_scratch.set_name("gpumd_neigh_pack", impl_->logger);
    impl_->device_type_scratch.set_name("gpumd_neigh_type_pack", impl_->logger);
    impl_->device_box.set_name("gpumd_neigh_box", impl_->logger);
    impl_->device_periodic.set_name("gpumd_neigh_pbc", impl_->logger);
    impl_->device_displacement_flag.set_name("gpumd_neigh_flag", impl_->logger);
    impl_->device_all_atoms_flags.set_name("gpumd_neigh_all_flags", impl_->logger);
}

GpumdNeighbors::~GpumdNeighbors() = default;

bool GpumdNeighbors::active() const { return !impl_->lists.empty(); }

void GpumdNeighbors::set_skin(double skin) {
    if (!std::isfinite(skin) || skin < 0)
        throw std::invalid_argument("GpumdNeighbors: skin must be finite and >= 0");
    if (skin != impl_->skin) {
        impl_->skin = skin;
        impl_->structural_dirty = true;
    }
}

SelectionId GpumdNeighbors::define_selection(const std::string& name,
                                             const std::vector<int>& owned_rows) {
    if (name=="all") throw std::invalid_argument("GpumdNeighbors: all is reserved");
    if (name.empty()) throw std::invalid_argument("GpumdNeighbors: selection name is empty");
    if (impl_->selection_names.count(name))
        throw std::invalid_argument("GpumdNeighbors: selection name already defined: " + name);
    if (impl_->selections.size() >= static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::length_error("GpumdNeighbors: too many selections");

    std::vector<int> rows = owned_rows;
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    for (int row : rows) {
        if (row < 0) throw std::invalid_argument("GpumdNeighbors: selection row is negative");
        if (impl_->geometry_ready && row >= impl_->owned_count)
            throw std::out_of_range("GpumdNeighbors: selection row exceeds owned count");
    }

    auto storage = std::make_unique<Impl::Selection>();
    storage->name = name;
    storage->id = static_cast<SelectionId>(impl_->selections.size());
    storage->owned_rows = std::move(rows);
    storage->device_flags.set_name("gpumd_neigh_sel_flags", impl_->logger);
    const SelectionId id = storage->id;
    impl_->selection_names.emplace(name, id);
    impl_->selections.push_back(std::move(storage));
    impl_->structural_dirty = true;
    return id;
}

SelectionId GpumdNeighbors::selection_id(const std::string& name) const {
    if (name=="all") return all_atoms;
    const auto found = impl_->selection_names.find(name);
    if (found == impl_->selection_names.end())
        throw std::invalid_argument("GpumdNeighbors: unknown selection: " + name);
    return found->second;
}

int GpumdNeighbors::register_request(const NeighborRequest& request) {
    request.validate();
    if (request.ghost_centers)
        throw std::invalid_argument(
            "GpumdNeighbors: ghost_centers unsupported (owned centers only; StruFactor uses owned)");
    if (request.centers != all_atoms) {
        if (request.centers < 0 || static_cast<size_t>(request.centers) >= impl_->selections.size())
            throw std::out_of_range("GpumdNeighbors: centers selection is not defined");
    }
    if (request.neighbors != all_atoms) {
        if (request.neighbors < 0 || static_cast<size_t>(request.neighbors) >= impl_->selections.size())
            throw std::out_of_range("GpumdNeighbors: neighbors selection is not defined");
    }
    if (impl_->lists.size() >= static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::length_error("GpumdNeighbors: too many neighbor requests");
    auto storage = std::make_unique<Impl::ListCache>();
    storage->request = request;
    impl_->lists.push_back(std::move(storage));
    impl_->structural_dirty = true;
    return static_cast<int>(impl_->lists.size() - 1);
}

void GpumdNeighbors::update(ArrayView<const double> positions, ArrayView<const int> types,
                            const std::array<double, 3>& box, const std::array<int, 3>& periodic,
                            cudaStream_t stream, std::uint64_t evaluation) {
    auto& state = *impl_;
    state.validate_frame(positions, types, box, periodic);
    state.evaluation = evaluation;

    const int next_owned = static_cast<int>(positions.item_count);
    const bool count_changed = !state.geometry_ready || next_owned != state.owned_count;
    const bool box_changed = state.geometry_ready && state.box_or_pbc_changed(box, periodic);
    state.owned_count = next_owned;
    state.box = box;
    state.periodic = periodic;

    bool rebuild = state.structural_dirty || count_changed || box_changed || !state.geometry_ready;
    if (!rebuild)
        rebuild = state.displacement_requires_rebuild(positions, stream);
    if (rebuild)
        state.rebuild(positions, types, stream);
    else
        state.reuse_update_device(positions, types, stream);
}

int GpumdNeighbors::total_atoms() const {
    impl_->require_geometry();
    return impl_->total_count;
}

ArrayView<const double> GpumdNeighbors::positions(MemorySpace memory) {
    auto& state = *impl_;
    state.require_geometry();
    if (memory == MemorySpace::Cuda) {
        return {state.device_expanded_aos.ptr, static_cast<size_t>(state.total_count), 3, 3, 1,
                state.device_expanded_aos.capacity, MemorySpace::Cuda,
                state.device_expanded_aos.readiness()};
    }
    if (memory != MemorySpace::Host)
        throw std::invalid_argument("GpumdNeighbors: unknown position memory space");
    if (!state.host_positions_cached) {
        state.host_expanded_aos.resize(static_cast<size_t>(state.total_count) * 3);
        state.device_expanded_aos.download_to(
            state.host_expanded_aos.data(), static_cast<size_t>(state.total_count) * 3, 0);
        state.host_positions_cached = true;
    }
    return {state.host_expanded_aos.data(), static_cast<size_t>(state.total_count), 3, 3, 1,
            state.host_expanded_aos.size(), MemorySpace::Host};
}

ArrayView<const int> GpumdNeighbors::types(MemorySpace memory) {
    auto& state = *impl_;
    state.require_geometry();
    if (memory == MemorySpace::Cuda) {
        return {state.device_expanded_types.ptr, static_cast<size_t>(state.total_count), 1, 1, 1,
                state.device_expanded_types.capacity, MemorySpace::Cuda,
                state.device_expanded_types.readiness()};
    }
    if (memory != MemorySpace::Host)
        throw std::invalid_argument("GpumdNeighbors: unknown type memory space");
    if (!state.host_types_cached) {
        state.host_expanded_types.resize(static_cast<size_t>(state.total_count));
        state.device_expanded_types.download_to(
            state.host_expanded_types.data(), static_cast<size_t>(state.total_count), 0);
        state.host_types_cached = true;
    }
    return {state.host_expanded_types.data(), state.host_expanded_types.size(), 1, 1, 1,
            state.host_expanded_types.size(), MemorySpace::Host};
}

ArrayView<const AtomId> GpumdNeighbors::ids() const {
    impl_->require_geometry();
    return {impl_->host_ids.data(), impl_->host_ids.size(), 1, 1, 1, impl_->host_ids.size(),
            MemorySpace::Host};
}

ArrayView<const int> GpumdNeighbors::owners(MemorySpace memory) const {
    auto& state = *impl_;
    state.require_geometry();
    if (memory == MemorySpace::Host) {
        return {state.host_owners.data(), state.host_owners.size(), 1, 1, 1, state.host_owners.size(),
                MemorySpace::Host};
    }
    if (memory != MemorySpace::Cuda)
        throw std::invalid_argument("GpumdNeighbors: unknown owner memory space");
    return {state.device_owners.ptr, static_cast<size_t>(state.total_count), 1, 1, 1,
            state.device_owners.capacity, MemorySpace::Cuda, state.device_owners.readiness()};
}

SelectionView GpumdNeighbors::selection_members(SelectionId id) {
    auto& state = *impl_;
    state.require_geometry();
    if (id == all_atoms) {
        // 隐式全体：owned 前缀 + 全部镜像；generation 跟随重建计数以便布局失效。
        return {all_atoms, state.owned_count,
                {state.all_atoms_local_ids.data(), state.all_atoms_local_ids.size(), 1, 1, 1,
                 state.all_atoms_local_ids.size(), MemorySpace::Host},
                state.rebuilds};
    }
    if (id < 0 || static_cast<size_t>(id) >= state.selections.size())
        throw std::out_of_range("GpumdNeighbors: unknown selection");
    auto& selection = *state.selections[static_cast<size_t>(id)];
    return {selection.id, static_cast<int>(selection.owned_rows.size()),
            {selection.local_ids.data(), selection.local_ids.size(), 1, 1, 1,
             selection.local_ids.size(), MemorySpace::Host},
            selection.generation};
}

ArrayView<const int> GpumdNeighbors::selection_flags(SelectionId id, MemorySpace memory) {
    auto& state = *impl_;
    state.require_geometry();
    if (id == all_atoms) {
        if (memory == MemorySpace::Host)
            return {state.all_atoms_flags.data(), state.all_atoms_flags.size(), 1, 1, 1,
                    state.all_atoms_flags.size(), MemorySpace::Host};
        if (memory != MemorySpace::Cuda)
            throw std::invalid_argument("GpumdNeighbors: unknown selection flag memory space");
        if (!state.all_atoms_device_ready) {
            state.device_all_atoms_flags.grow_to(
                std::max<size_t>(state.all_atoms_flags.size(), 1), __FILE__, __LINE__);
            if (!state.all_atoms_flags.empty())
                state.device_all_atoms_flags.upload_from(
                    state.all_atoms_flags.data(), state.all_atoms_flags.size(), 0);
            state.device_all_atoms_flags.record_write(0);
            state.all_atoms_device_ready = true;
        }
        return {state.device_all_atoms_flags.ptr, state.all_atoms_flags.size(), 1, 1, 1,
                state.device_all_atoms_flags.capacity, MemorySpace::Cuda,
                state.device_all_atoms_flags.readiness()};
    }
    if (id < 0 || static_cast<size_t>(id) >= state.selections.size())
        throw std::out_of_range("GpumdNeighbors: unknown selection");
    auto& selection = *state.selections[static_cast<size_t>(id)];
    if (memory == MemorySpace::Host) {
        return {selection.flags.data(), selection.flags.size(), 1, 1, 1, selection.flags.size(),
                MemorySpace::Host};
    }
    if (memory != MemorySpace::Cuda)
        throw std::invalid_argument("GpumdNeighbors: unknown selection flag memory space");
    if (!selection.device_ready) {
        selection.device_flags.grow_to(std::max<size_t>(selection.flags.size(), 1), __FILE__, __LINE__);
        if (!selection.flags.empty())
            selection.device_flags.upload_from(selection.flags.data(), selection.flags.size(), 0);
        selection.device_flags.record_write(0);
        selection.device_ready = true;
    }
    return {selection.device_flags.ptr, selection.flags.size(), 1, 1, 1,
            selection.device_flags.capacity, MemorySpace::Cuda, selection.device_flags.readiness()};
}

NeighborListView GpumdNeighbors::list(int handle) {
    auto& state = *impl_;
    state.require_geometry();
    if (handle < 0 || static_cast<size_t>(handle) >= state.lists.size())
        throw std::out_of_range("GpumdNeighbors: invalid neighbor handle");
    auto& list = *state.lists[static_cast<size_t>(handle)];
    NeighborListView view;
    view.host_centers = {list.centers.data(), list.centers.size(), 1, 1, 1, list.centers.size(),
                         MemorySpace::Host};
    view.centers = {list.device_centers.ptr, list.centers.size(), 1, 1, 1,
                    list.device_centers.capacity, MemorySpace::Cuda, list.device_centers.readiness()};
    view.offsets = {list.device_offsets.ptr, list.offsets.size(), 1, 1, 1,
                    list.device_offsets.capacity, MemorySpace::Cuda, list.device_offsets.readiness()};
    view.neighbors = {list.device_neighbors.ptr, list.neighbors.size(), 1, 1, 1,
                      list.device_neighbors.capacity, MemorySpace::Cuda,
                      list.device_neighbors.readiness()};
    view.owned_atom_count = state.owned_count;
    view.total_atom_count = state.total_count;
    view.evaluation_id = state.evaluation;
    view.topology_generation = list.topology_generation;
    view.coordinates = NeighborCoordinates::PeriodicImages;
    view.host_offsets = {list.offsets.data(), list.offsets.size(), 1, 1, 1, list.offsets.size(),
                         MemorySpace::Host};
    view.host_neighbors = {list.neighbors.data(), list.neighbors.size(), 1, 1, 1,
                           list.neighbors.size(), MemorySpace::Host};
    view.kind = list.request.kind;
    view.pair_ownership = list.request.kind == NeighborKind::Half
        ? NeighborPairOwnership::UniqueAcrossRanks
        : NeighborPairOwnership::DirectedByCenter;
    return view;
}

std::uint64_t GpumdNeighbors::rebuild_count() const { return impl_->rebuilds; }

} // namespace MetaD_zqc
