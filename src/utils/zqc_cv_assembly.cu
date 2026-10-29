#include "zqc_cv_assembly.cuh"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace MetaD_zqc {
namespace {
void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

// 装配/回收 kernel 按 ArrayView 寻址，支持 AoS 与带填充的 SoA stride。
__global__ void assemble_kernel(int count, const int* local_ids, const int* leaf_slots,
                                ArrayView<const double> source, ArrayView<double> target) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < count) target(leaf_slots[k]) = source(local_ids[k]);
}
__global__ void adjoint_kernel(int count, const int* leaf_slots,
                               ArrayView<const double> leaf, ArrayView<double> cv) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < count) cv(k) = leaf(leaf_slots[k]);
}
__global__ void zero_items_kernel(ArrayView<double> target) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (static_cast<size_t>(k) < target.item_count) target(k) = 0.0;
}

void require_cuda_scalar(ArrayView<const double> view, const char* what) {
    view.validate();
    if (view.memory != MemorySpace::Cuda)
        throw std::invalid_argument(std::string("CVAssembly: ") + what + " must be Cuda");
    if (view.component_count != 1)
        throw std::invalid_argument(std::string("CVAssembly: ") + what + " must be scalar");
}
void require_cuda_scalar(ArrayView<double> view, const char* what) {
    require_cuda_scalar(ArrayView<const double>(view), what);
}

bool storage_alias(const double* a, size_t a_size, const double* b, size_t b_size) {
    if (!a || !b || !a_size || !b_size) return false;
    const double* a_end = a + a_size;
    const double* b_end = b + b_size;
    return a < b_end && b < a_end;
}

void validate_selection(int owned, int ghosts, const SelectionView& selection) {
    if (selection.id < -1)
        throw std::invalid_argument("CVAtomLayout: selection id below all_atoms");
    selection.local_ids.validate();
    selection.local_ids.wait_host();
    if (selection.local_ids.memory != MemorySpace::Host)
        throw std::invalid_argument("CVAtomLayout: selection members must be Host");
    if (selection.local_ids.component_count != 1)
        throw std::invalid_argument("CVAtomLayout: selection members must be scalar indices");
    if (selection.owned_count < 0 ||
        static_cast<size_t>(selection.owned_count) > selection.local_ids.item_count)
        throw std::invalid_argument("CVAtomLayout: owned_count exceeds member list");
    const int total = owned + ghosts;
    int previous = -1;
    for (size_t index = 0; index < selection.local_ids.item_count; ++index) {
        const int local = selection.local_ids(index);
        if (local < 0 || local >= total)
            throw std::out_of_range("CVAtomLayout: selection local id out of range");
        if (local <= previous)
            throw std::invalid_argument("CVAtomLayout: selection local ids must increase");
        previous = local;
        const bool expect_owned = static_cast<int>(index) < selection.owned_count;
        if (expect_owned != (local < owned))
            throw std::invalid_argument("CVAtomLayout: owned prefix must contain only owned atoms");
    }
}
} // namespace

CVAssembly::CVAssembly(SelectionId selection_id, int nlocal, int nghost, int group_count,
                       int leaf_count, std::vector<int> local_ids, std::vector<int> leaf_slots)
    : selection_id_(selection_id), nlocal_(nlocal), nghost_(nghost),
      group_count_(group_count), leaf_count_(leaf_count),
      local_ids_(std::move(local_ids)), leaf_slots_(std::move(leaf_slots)),
      d_local_ids_(local_ids_.size()), d_leaf_slots_(leaf_slots_.size()) {
    if (total_count()) {
        d_local_ids_.upload_from(local_ids_.data(), local_ids_.size());
        d_leaf_slots_.upload_from(leaf_slots_.data(), leaf_slots_.size());
    }
}

void CVAssembly::assemble_values(ArrayView<const double> source, ArrayView<double> target,
                                 cudaStream_t stream) const {
    // 只读取 owned 前缀；即使表里有 ghost，也不要求 source 包含 ghost 值。
    source.wait_on(stream); target.wait_on(stream);
    require_cuda_scalar(source, "source");
    require_cuda_scalar(target, "target");
    if (static_cast<int>(target.item_count) != leaf_count_)
        throw std::invalid_argument("CVAssembly: target extent must equal leaf_count");
    if (group_count_) {
        const int max_local = local_ids_[group_count_-1];
        if (source.item_count <= static_cast<size_t>(max_local))
            throw std::runtime_error("CVAssembly: source does not cover owned local ids");
    }
    if (leaf_count_ && storage_alias(source.data, source.storage_size, target.data, target.storage_size))
        throw std::runtime_error("CVAssembly: source and target must not alias");
    if (leaf_count_) {
        // 连续紧凑目标可一次 memset；任意 stride 走按项清零，避免误伤填充槽。
        if (target.item_stride == 1 && target.component_stride == 1) {
            check_cuda(cudaMemsetAsync(target.data, 0, leaf_count_*sizeof(double), stream));
        } else {
            zero_items_kernel<<<(leaf_count_-1)/256+1, 256, 0, stream>>>(target);
            check_cuda(cudaGetLastError());
        }
    }
    if (group_count()) {
        assemble_kernel<<<(group_count()-1)/256+1, 256, 0, stream>>>(
            group_count(), d_local_ids_.ptr, d_leaf_slots_.ptr, source, target);
        check_cuda(cudaGetLastError());
    }
}

void CVAssembly::gather_adjoint(ArrayView<const double> leaf, ArrayView<double> cv,
                                cudaStream_t stream) const {
    leaf.wait_on(stream); cv.wait_on(stream);
    require_cuda_scalar(leaf, "leaf adjoint");
    require_cuda_scalar(cv, "cv adjoint");
    if (static_cast<int>(leaf.item_count) != leaf_count_)
        throw std::invalid_argument("CVAssembly: leaf adjoint extent must equal leaf_count");
    if (static_cast<int>(cv.item_count) != group_count())
        throw std::invalid_argument("CVAssembly: cv adjoint extent must equal group_count");
    if (group_count() && storage_alias(leaf.data, leaf.storage_size, cv.data, cv.storage_size))
        throw std::runtime_error("CVAssembly: adjoint views must not alias");
    if (group_count()) {
        adjoint_kernel<<<(group_count()-1)/256+1, 256, 0, stream>>>(
            group_count(), d_leaf_slots_.ptr, leaf, cv);
        check_cuda(cudaGetLastError());
    }
}

void CVAssembly::assemble_values(const GpuBuffer<double>& source, GpuBuffer<double>& target,
                                 cudaStream_t stream) const {
    const size_t source_items = group_count_ == 0
        ? 0
        : static_cast<size_t>(local_ids_[group_count_-1]) + 1;
    if (source.capacity < source_items || (source_items && !source.ptr))
        throw std::runtime_error("CVAssembly: source buffer is too small");
    if (target.capacity < static_cast<size_t>(leaf_count_) || (leaf_count_ && !target.ptr))
        throw std::runtime_error("CVAssembly: target buffer is too small");
    assemble_values(source.view(source_items), target.view(leaf_count_), stream);
    // Leaf(GpuBuffer&) 每轮取得新 View，因而能看到本轮装配的完成事件。
    target.record_write(stream);
}

void CVAssembly::gather_adjoint(const GpuBuffer<double>& leaf_adjoint,
                                GpuBuffer<double>& cv_adjoint, cudaStream_t stream) const {
    if (leaf_adjoint.capacity < static_cast<size_t>(leaf_count_) || (leaf_count_ && !leaf_adjoint.ptr))
        throw std::runtime_error("CVAssembly: leaf adjoint buffer is too small");
    if (cv_adjoint.capacity < static_cast<size_t>(group_count()) || (group_count() && !cv_adjoint.ptr))
        throw std::runtime_error("CVAssembly: cv adjoint buffer is too small");
    gather_adjoint(leaf_adjoint.view(leaf_count_), cv_adjoint.view(group_count()), stream);
    cv_adjoint.record_write(stream);
}

void CVAtomLayout::rebuild(int owned, int ghosts, const std::vector<SelectionView>& selections) {
    if (owned < 0 || ghosts < 0 || owned > std::numeric_limits<int>::max()-ghosts)
        throw std::invalid_argument("CVAtomLayout: invalid owned/ghost atom counts");
    const int nall = owned + ghosts;
    struct SelectionData {
        int group_count = 0;
        std::vector<int> local_ids, leaf_slots;
    };
    std::map<SelectionId, SelectionData> by_id;
    // 每个 localid 参与哪些选择集；身份编号不作 bitmask 运算。
    std::vector<std::vector<SelectionId>> membership(static_cast<size_t>(nall));
    for (const SelectionView& selection : selections) {
        validate_selection(owned, ghosts, selection);
        by_id.try_emplace(selection.id);
        for (size_t index = 0; index < selection.local_ids.item_count; ++index) {
            const int local = selection.local_ids(index);
            auto& ids = membership[static_cast<size_t>(local)];
            if (std::find(ids.begin(), ids.end(), selection.id) == ids.end())
                ids.push_back(selection.id);
        }
    }
    std::vector<int> local_ids;
    int leaf_count = 0;
    for (int atom = 0; atom < nall; ++atom) {
        if (membership[static_cast<size_t>(atom)].empty()) continue;
        const bool is_owned = atom < owned;
        const int slot = is_owned ? leaf_count++ : -1;
        local_ids.push_back(atom);
        for (SelectionId id : membership[static_cast<size_t>(atom)]) {
            auto& entry = by_id[id];
            entry.local_ids.push_back(atom);
            entry.leaf_slots.push_back(slot);
            if (is_owned) ++entry.group_count;
        }
    }
    // 未出现任何成员的请求仍保留空表，便于空选择集/空 rank 走同一路径。
    for (const SelectionView& selection : selections) by_id.try_emplace(selection.id);
    std::map<SelectionId, std::shared_ptr<const CVAssembly>> groups;
    for (auto& entry : by_id) {
        groups.emplace(entry.first, std::shared_ptr<const CVAssembly>(new CVAssembly(
            entry.first, owned, ghosts, entry.second.group_count, leaf_count,
            std::move(entry.second.local_ids), std::move(entry.second.leaf_slots))));
    }
    local_ids_.swap(local_ids);
    groups_.swap(groups);
    leaf_count_ = leaf_count;
}

std::shared_ptr<const CVAssembly> CVAtomLayout::group(SelectionId selection_id) const {
    auto it = groups_.find(selection_id);
    if (it == groups_.end()) throw std::out_of_range("CVAtomLayout: selection was not requested");
    return it->second;
}
} // namespace MetaD_zqc
