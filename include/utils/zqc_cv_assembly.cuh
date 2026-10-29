#pragma once
#include "zqc_engine.h"
#include "zqc_gpu_buffer.cuh"
#include <map>
#include <memory>
#include <vector>

namespace MetaD_zqc {

// 同一选择集的 CV 可共享此表。local_ids 是本地下标（不是全局原子编号）：
// 前 group_count 项为 owned，其余为 ghost；local_id >= nlocal 即为 ghost。
// owned 对应有效 leaf_slots；ghost 的 leaf_slots 为 -1，不进入 Leaf 规约。
// selection_id 是引擎分配的选择集编号（身份），不是 bitmask。
class CVAssembly {
public:
    CVAssembly(const CVAssembly&) = delete;
    CVAssembly& operator=(const CVAssembly&) = delete;

    SelectionId selection_id() const { return selection_id_; }
    int nlocal() const { return nlocal_; }
    int nghost() const { return nghost_; }
    int group_count() const { return group_count_; } // 仅本地 owned 组成员
    int total_count() const { return static_cast<int>(local_ids_.size()); } // 含组内 ghost
    int leaf_count() const { return leaf_count_; } // 公共并集的 owned 数量
    const std::vector<int>& local_ids() const { return local_ids_; }
    const int* device_local_ids() const { return d_local_ids_.ptr; }
    const std::vector<int>& leaf_slots() const { return leaf_slots_; }
    // 仅含本地 owned 中心的公共 Leaf 槽位；就绪事件随底层缓冲区一起借用。
    ArrayView<const int> owned_leaf_slots() const { return d_leaf_slots_.view(group_count_); }

    // source 按本地原子下标存储，target 按公共 owned Leaf 顺序存储。
    // 视图显式给出有效项数与 stride；调用方持有分配。
    // target.item_count 应为 leaf_count；只覆盖该区间（组外位置写零）。
    // 只读取 owned 前缀；ghost 值无需就绪。两视图存储不可别名。
    // 异步提交到 stream；View 按值传入，调用方在提交后发布生产流的完成事件。
    void assemble_values(ArrayView<const double> source, ArrayView<double> target,
                         cudaStream_t stream = 0) const;
    // leaf.item_count 应为 leaf_count；cv.item_count 应为 group_count。
    // 仅返回 owned 中心梯度，顺序与 local_ids 的 owned 前缀一致。
    void gather_adjoint(ArrayView<const double> leaf, ArrayView<double> cv,
                        cudaStream_t stream = 0) const;

    // 连续缓冲区重载在输出中自动记录完成事件，后续 view()/Leaf/下载继承依赖。
    // 默认流与显式流均异步提交；反复装配后可直接调用 GraphExecution::forward。
    void assemble_values(const GpuBuffer<double>& source, GpuBuffer<double>& target,
                         cudaStream_t stream = 0) const;
    void gather_adjoint(const GpuBuffer<double>& leaf_adjoint, GpuBuffer<double>& cv_adjoint,
                        cudaStream_t stream = 0) const;

private:
    friend class CVAtomLayout;
    CVAssembly(SelectionId selection_id, int nlocal, int nghost, int group_count, int leaf_count,
               std::vector<int> local_ids, std::vector<int> leaf_slots);
    SelectionId selection_id_;
    int nlocal_, nghost_, group_count_, leaf_count_;
    std::vector<int> local_ids_, leaf_slots_;
    // 设备端编号表与 Host 同序；扩容只发生在构造时，生命周期随本对象。
    GpuBuffer<int> d_local_ids_, d_leaf_slots_;
};

// 由 CV/维度管理层持有，在迁移、排序、ghost 列表或选择集成员改变后重建。
// 此表包含各选择集中心及组内 ghost；邻居表仍可访问组外原子，不裁剪邻居坐标。
class CVAtomLayout {
public:
    // selections 提供本 rank 的成员快照：local_ids 递增且 owned 在前。
    // 选择集编号是身份，可大于任何 bitmask 宽度；重复 id 自动合并。
    // 公共 Leaf 为所请求选择集 owned 成员的并集（按 localid 递增）。
    // 重建后需重新 bind_assembly；旧 shared_ptr 在旧使用者释放前保持有效。
    void rebuild(int owned, int ghosts, const std::vector<SelectionView>& selections);
    const std::vector<int>& local_ids() const { return local_ids_; }
    int leaf_count() const { return leaf_count_; } // 公共表中 owned 前缀长度
    int total_count() const { return static_cast<int>(local_ids_.size()); }
    std::shared_ptr<const CVAssembly> group(SelectionId selection_id) const;

private:
    int leaf_count_ = 0;
    std::vector<int> local_ids_;
    std::map<SelectionId, std::shared_ptr<const CVAssembly>> groups_;
};

} // namespace MetaD_zqc
