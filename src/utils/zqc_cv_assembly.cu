#include "zqc_cv_assembly.cuh"
#include <limits>
#include <stdexcept>
#include <utility>

namespace MetaD_zqc {
namespace {
void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void check_buffer(const GpuBuffer<double>& buffer, size_t count) {
    if (buffer.capacity < count || (count && !buffer.ptr))
        throw std::runtime_error("CVAssembly: buffer is too small");
}
__global__ void assemble_kernel(int count, const int* local_ids, const int* leaf_slots,
                                const double* source, double* target) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < count) target[leaf_slots[k]] = source[local_ids[k]];
}
__global__ void adjoint_kernel(int count, const int* leaf_slots,
                               const double* source, double* target) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < count) target[k] = source[leaf_slots[k]];
}
}

CVAssembly::CVAssembly(int groupbit, int nlocal, int nghost, int group_count, int leaf_count,
                       std::vector<int> local_ids, std::vector<int> leaf_slots)
    : groupbit_(groupbit), nlocal_(nlocal), nghost_(nghost),
      group_count_(group_count), leaf_count_(leaf_count),
      local_ids_(std::move(local_ids)), leaf_slots_(std::move(leaf_slots)),
      d_local_ids_(local_ids_.size()), d_leaf_slots_(leaf_slots_.size()) {
    if (total_count()) {
        check_cuda(cudaMemcpy(d_local_ids_.ptr, local_ids_.data(), local_ids_.size()*sizeof(int),
                              cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(d_leaf_slots_.ptr, leaf_slots_.data(), leaf_slots_.size()*sizeof(int),
                              cudaMemcpyHostToDevice));
    }
}

void CVAssembly::assemble_values(const GpuBuffer<double>& source, GpuBuffer<double>& target,
                                 cudaStream_t stream) const {
    // 只读取 owned 前缀；即使表里有 ghost，也不要求 source 包含 ghost 值。
    check_buffer(source, group_count_ == 0 ? 0 : static_cast<size_t>(local_ids_[group_count_-1])+1);
    check_buffer(target, leaf_count_);
    if (leaf_count_ && source.ptr == target.ptr)
        throw std::runtime_error("CVAssembly: source and target must not alias");
    if (leaf_count_)
        check_cuda(cudaMemsetAsync(target.ptr, 0, leaf_count_*sizeof(double), stream));
    if (group_count()) {
        assemble_kernel<<<(group_count()-1)/256+1, 256, 0, stream>>>(
            group_count(), d_local_ids_.ptr, d_leaf_slots_.ptr, source.ptr, target.ptr);
        check_cuda(cudaGetLastError());
    }
}

void CVAssembly::gather_adjoint(const GpuBuffer<double>& leaf_adjoint,
                                GpuBuffer<double>& cv_adjoint, cudaStream_t stream) const {
    check_buffer(leaf_adjoint, leaf_count_);
    check_buffer(cv_adjoint, group_count());
    if (group_count() && leaf_adjoint.ptr == cv_adjoint.ptr)
        throw std::runtime_error("CVAssembly: adjoint buffers must not alias");
    if (group_count()) {
        adjoint_kernel<<<(group_count()-1)/256+1, 256, 0, stream>>>(
            group_count(), d_leaf_slots_.ptr, leaf_adjoint.ptr, cv_adjoint.ptr);
        check_cuda(cudaGetLastError());
    }
}

void CVAtomLayout::rebuild(int nlocal, int nghost, const int* mask, const std::vector<int>& groupbits) {
    if (nlocal < 0 || nghost < 0 || nlocal > std::numeric_limits<int>::max()-nghost)
        throw std::invalid_argument("CVAtomLayout: invalid owned/ghost atom counts");
    const int nall = nlocal + nghost;
    if (nall && !mask) throw std::invalid_argument("CVAtomLayout: missing atom mask");
    struct Selection {
        int group_count = 0;
        std::vector<int> local_ids, leaf_slots;
    };
    std::map<int, Selection> selections;
    int union_bits = 0;
    for (int bit : groupbits) {
        if (bit == 0) throw std::invalid_argument("CVAtomLayout: zero groupbit");
        selections.try_emplace(bit);
        union_bits |= bit;
    }
    std::vector<int> local_ids;
    int leaf_count = 0;
    for (int atom = 0; atom < nall; ++atom) {
        if (!(mask[atom] & union_bits)) continue;
        const bool owned = atom < nlocal;
        const int slot = owned ? leaf_count++ : -1;
        local_ids.push_back(atom);
        for (auto& entry : selections) {
            if (!(mask[atom] & entry.first)) continue;
            entry.second.local_ids.push_back(atom);
            entry.second.leaf_slots.push_back(slot);
            if (owned) ++entry.second.group_count;
        }
    }
    std::map<int, std::shared_ptr<const CVAssembly>> groups;
    for (auto& entry : selections) {
        groups.emplace(entry.first, std::shared_ptr<const CVAssembly>(new CVAssembly(
            entry.first, nlocal, nghost, entry.second.group_count, leaf_count,
            std::move(entry.second.local_ids), std::move(entry.second.leaf_slots))));
    }
    local_ids_.swap(local_ids);
    groups_.swap(groups);
    leaf_count_ = leaf_count;
}

std::shared_ptr<const CVAssembly> CVAtomLayout::group(int groupbit) const {
    auto it = groups_.find(groupbit);
    if (it == groups_.end()) throw std::out_of_range("CVAtomLayout: group was not requested");
    return it->second;
}
} // namespace MetaD_zqc
