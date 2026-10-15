#pragma once
#include "zqc_gpu_buffer.cuh"
#include <map>
#include <memory>
#include <vector>

namespace MetaD_zqc {

// 同一 group 的 CV 可以共享此表。local_ids 是本地下标（不是全局 atom tag）：
// 前 group_count 项为 owned，其余为 ghost；local_id >= nlocal 即为 ghost。
// owned 对应有效 leaf_slots；ghost 的 leaf_slots 为 -1，不进入 Leaf 规约。
class CVAssembly {
public:
    CVAssembly(const CVAssembly&) = delete;
    CVAssembly& operator=(const CVAssembly&) = delete;

    int groupbit() const { return groupbit_; }
    int nlocal() const { return nlocal_; }
    int nghost() const { return nghost_; }
    int group_count() const { return group_count_; } // 仅本地 owned 组成员
    int total_count() const { return static_cast<int>(local_ids_.size()); } // 含组内 ghost
    int leaf_count() const { return leaf_count_; } // 公共并集的 owned 数量
    const std::vector<int>& local_ids() const { return local_ids_; }
    const int* device_local_ids() const { return d_local_ids_.ptr; }
    const std::vector<int>& leaf_slots() const { return leaf_slots_; }

    // source 按本地原子下标存储，target 按公共 owned Leaf 顺序存储。
    // 只读取前 group_count 个中心的值；ghost 值无需就绪，也不参与求和。
    // 每次覆盖有效 target 区间，组外位置写零；两缓冲区不可别名。
    void assemble_values(const GpuBuffer<double>& source, GpuBuffer<double>& target,
                         cudaStream_t stream = 0) const;
    // 仅返回 group_count 个 owned 中心梯度，顺序与 local_ids 的 owned 前缀一致。
    // 坐标力对 ghost 的贡献及 reverse communication 仍由 CV 负责。
    void gather_adjoint(const GpuBuffer<double>& leaf_adjoint, GpuBuffer<double>& cv_adjoint,
                        cudaStream_t stream = 0) const;

private:
    friend class CVAtomLayout;
    CVAssembly(int groupbit, int nlocal, int nghost, int group_count, int leaf_count,
               std::vector<int> local_ids, std::vector<int> leaf_slots);
    int groupbit_, nlocal_, nghost_, group_count_, leaf_count_;
    std::vector<int> local_ids_, leaf_slots_;
    GpuBuffer<int> d_local_ids_, d_leaf_slots_;
};

// 由 CV/维度管理层持有，在迁移、排序、ghost 列表或 group 成员改变后重建。
// 此表包含各组中心及组内 ghost；邻居表仍可访问组外原子，不裁剪邻居坐标。
class CVAtomLayout {
public:
    // mask 是 Host 数组，长度至少 nlocal+nghost；公共表为所请求 group 的并集。
    // 重复 groupbit 自动合并；所有 owned 在前、ghost 在后。
    // 重建后需重新 bind_assembly；旧 shared_ptr 在旧使用者释放前保持有效。
    void rebuild(int nlocal, int nghost, const int* mask, const std::vector<int>& groupbits);
    const std::vector<int>& local_ids() const { return local_ids_; }
    int leaf_count() const { return leaf_count_; } // 公共表中 owned 前缀长度
    int total_count() const { return static_cast<int>(local_ids_.size()); }
    std::shared_ptr<const CVAssembly> group(int groupbit) const;

private:
    int leaf_count_ = 0;
    std::vector<int> local_ids_;
    std::map<int, std::shared_ptr<const CVAssembly>> groups_;
};

} // namespace MetaD_zqc
