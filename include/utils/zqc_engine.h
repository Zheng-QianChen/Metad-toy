#pragma once
#include "zqc_array_view.h"
#include "zqc_virial.h"
#include "zqc_collective.h"
#include <string>
#include "zqc_diagnostics.h"
#include <cmath>
#include <cstdint>
#include <vector>

namespace MetaD_zqc {
using LocalAtomIndex = int;
using AtomId = std::uint64_t;
// 引擎适配器分配的选择集编号；-1 表示所有原子。编号只在当前适配器内有效。
using SelectionId = int;
constexpr SelectionId all_atoms = -1;
// 选择集的 Host 索引快照，local_ids 按 localid 递增且 owned 在前。
// owned_count 只统计本 rank 的成员，后续元素为组内 ghost。数组由引擎持有。
struct SelectionView {
    SelectionId id = all_atoms;
    int owned_count = 0;
    ArrayView<const LocalAtomIndex> local_ids;
    // 非零版本覆盖成员、owned 边界及 localid 身份变化；提供者无法保证时留 0，
    // 消费者逐项比较。版本稳定时可跨求值复用装配表，无需重复复制整个 localid 表。
    std::uint64_t generation = 0;
};

// Full 保存以中心为起点的有向环境；Half 在全体系中每个无序 pair 只保留一次。
// Half 的中心为 owned，邻居可为 ghost；消费者向两端累计力，再归还 ghost 贡献。
// 选择集不同时，pair 至少有一个方向满足 centers -> neighbors 即可入选。
enum class NeighborKind { Full, Half };
enum class NeighborPairOwnership { DirectedByCenter, UniqueAcrossRanks };
enum class NeighborCoordinates { PeriodicImages, MinimumImageRequired };

// CV 在构建时声明需求。cutoff 为物理半径，skin/特殊邻居标志属于引擎适配层。
// 中心选择集决定求值原子；邻居选择集可独立指定，默认包含组外原子与 ghost。
struct NeighborRequest {
    double cutoff = 0;
    SelectionId centers = all_atoms;
    SelectionId neighbors = all_atoms;
    NeighborKind kind = NeighborKind::Full;
    bool ghost_centers = false;
    // Device consumers can waive the Host CSR mirror. host_centers remains available
    // for Leaf assembly; providers without a device path may still return both copies.
    bool host_csr = true;
    void validate() const {
        if (!(cutoff > 0) || !std::isfinite(cutoff*cutoff) || centers < -1 || neighbors < -1 ||
            (kind != NeighborKind::Full && kind != NeighborKind::Half))
            throw std::invalid_argument("NeighborRequest: invalid radius, selection or kind");
        // ghost 中心行描述完整的邻域环境；无序 pair 的唯一计算归属必须落在 owned 端。
        if (kind == NeighborKind::Half && ghost_centers)
            throw std::invalid_argument("NeighborRequest: ghost center environments require Full lists");
    }
    bool operator==(const NeighborRequest& other) const {
        return cutoff == other.cutoff && centers == other.centers && neighbors == other.neighbors &&
               kind == other.kind && ghost_centers == other.ghost_centers && host_csr == other.host_csr;
    }
};

// 候选表按压缩中心分段：centers[k] 为 localid，offsets[k:k+2] 界定 neighbors。
// 每个 neighbor 是已去除引擎标志的 localid；距离筛选在本轮坐标上重新执行。
// host_centers 与 centers 顺序相同，供 Leaf 装配表检查；设备数组由 provider 持有。
struct NeighborListView {
    ArrayView<const LocalAtomIndex> host_centers, centers, offsets, neighbors;
    int owned_atom_count = 0;
    int total_atom_count = 0;
    std::uint64_t evaluation_id = 0;
    std::uint64_t topology_generation = 0;
    NeighborCoordinates coordinates = NeighborCoordinates::PeriodicImages;
    // 与设备 CSR 同序的 Host 视图；host_csr=false 时可为空。数据归 provider 所有。
    ArrayView<const LocalAtomIndex> host_offsets, host_neighbors;
    NeighborKind kind = NeighborKind::Full;
    NeighborPairOwnership pair_ownership = NeighborPairOwnership::DirectedByCenter;
};

class NeighborProvider {
public:
    virtual ~NeighborProvider() = default;
    virtual int register_neighbors(const NeighborRequest&) = 0;
    virtual NeighborListView prepare_neighbors(int provider_handle) = 0;
};

// 统一保存 CV 声明并去重。请求注册后保持不变；新需求获得新句柄。
// provider 决定如何借用/建立邻居表，并负责排序、迁移、选择集变化后的失效。
class NeighborManager {
    struct Entry { NeighborRequest request; int provider_handle; };
    NeighborProvider& provider_;
    std::vector<Entry> requests_;
public:
    explicit NeighborManager(NeighborProvider& provider) : provider_(provider) {}
    int request(const NeighborRequest& request) {
        request.validate();
        for (size_t index = 0; index < requests_.size(); ++index)
            if (requests_[index].request == request) return static_cast<int>(index);
        const int handle = provider_.register_neighbors(request);
        if (requests_.size() >= static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::length_error("NeighborManager: too many requests");
        requests_.push_back({request, handle});
        return static_cast<int>(requests_.size()-1);
    }
    NeighborListView prepare(int handle) {
        if (handle < 0 || static_cast<size_t>(handle) >= requests_.size())
            throw std::out_of_range("NeighborManager: unknown request");
        return provider_.prepare_neighbors(requests_[handle].provider_handle);
    }
};

// 引擎通信期间借用的 Host 数据包。pack 返回每原子的 double 数。
// forward 将 owner 属性发给 ghost；reverse 将 ghost 的贡献累加回 owner。
class AtomCommunication {
public:
    virtual ~AtomCommunication() = default;
    virtual int get_comm_forward_width() const { return 0; }
    virtual int get_comm_reverse_width() const { return 0; }
    virtual int pack_comm_forward_ubuf(int, int*, double*, int, int) { return 0; }
    virtual void unpack_comm_forward_ubuf(int, int, double*, int, int) {}
    virtual int pack_comm_reverse_ubuf(int, int, double*, int, int) { return 0; }
    virtual void unpack_comm_reverse_ubuf(int, int*, double*, int, int) {}
};

// 每次完整 forward/backward 使用同一 evaluation_id、localid 排列和数组所有者。
// 引擎可在同一 timestep 多次求值；begin_evaluation 必须每次调用。
// owned 原子占 [0, owned)，ghost 占 [owned, total)。CUDA 消费者通过 view.wait_on(consumer_stream) 建立生产者依赖。
// CPU/MPI 操作集中在这层；kernel 仅接收 ArrayView 等按值传递的数据。
class EngineAdapter : public NeighborProvider {
    NeighborManager neighbors_{*this};
public:
    EngineAdapter() = default;
    EngineAdapter(const EngineAdapter&) = delete;
    EngineAdapter& operator=(const EngineAdapter&) = delete;
    NeighborManager& neighbors() { return neighbors_; }
    virtual void begin_evaluation() = 0;
    virtual std::uint64_t evaluation_id() const = 0;
    virtual int owned_atom_count() const = 0;
    virtual int total_atom_count() const = 0;
    virtual ArrayView<const double> positions(MemorySpace) = 0;
    virtual ArrayView<const int> atom_types(MemorySpace) = 0;
    // Host 全局原子编号，按 localid 排列；ghost 镜像与 owner 使用同一编号。
    // Q 半对用它决定唯一计算端，编号在迁移/重排后保持不变。
    virtual ArrayView<const AtomId> atom_ids() { throw std::logic_error("Engine: atom IDs unavailable"); }
    // 原生类型编号保持后端语义；type_count/type_id 负责配置标签到该编号的映射。
    virtual int atom_type_count() const { throw std::logic_error("Engine: atom types unavailable"); }
    virtual int atom_type_begin() const { return 0; }
    virtual int atom_type_id(const std::string&) const { throw std::logic_error("Engine: type lookup unavailable"); }
    virtual SelectionId selection_id(const std::string&) const { throw std::logic_error("Engine: selection lookup unavailable"); }
    virtual SelectionView selection_members(SelectionId) { throw std::logic_error("Engine: selections unavailable"); }
    // 每个 localid 一项 0/1 成员标记；存储步长由 view 指定，不暴露宿主的 group bitmask。
    virtual ArrayView<const int> selection_flags(SelectionId, MemorySpace) { throw std::logic_error("Engine: selection flags unavailable"); }
    virtual Logger logger() const { return {}; }
    virtual int rank() const { return 0; }
    virtual std::int64_t timestep() const { return static_cast<std::int64_t>(evaluation_id()); }
    // 属性通信由适配器转发，CV 只定义数据包的 pack/unpack。
    virtual void communicate_forward(AtomCommunication&) { throw std::logic_error("Engine: forward communication unavailable"); }
    virtual void communicate_reverse(AtomCommunication&) { throw std::logic_error("Engine: reverse communication unavailable"); }
    // Optional owner-to-ghost transfer for a CUDA [total,width] property field.
    // false leaves the field unchanged and requests the legacy packet fallback.
    // true returns only after all writes finish, including empty-rank participation.
    virtual bool communicate_forward_device(ArrayView<double>) { return false; }
    SumReducer sum_reducer() { return [this](double* values, size_t count) { sum(values,count); }; }

    // 每个请求 ID 填写 [x,y,z,owner_count] 及本 rank owned 下标；缺失下标为 -1。
    virtual void read_owned_atoms(const AtomId* ids, size_t count,
                                  double* coordinates_and_counts, int* owned_indices) = 0;
    virtual void sum(double* values, size_t count) = 0;
    virtual void minimum_image(double displacement[3]) const = 0;
    // 预留宿主通信缓冲容量，单位为每原子的 double 数。
    virtual int reverse_communication_width() const { return 0; }
    // Graph 在全部 CV 反向前后界定一个输出批次。支持合并的适配器把各份
    // owned+ghost 贡献累加，finish 时统一 reverse 和回写；cancel 丢弃未提交贡献。
    // 批次内保持 evaluation/localid 排列不变，每个 rank 按相同 CV 顺序提交。
    // 默认实现供无需跨 rank 通信的适配器沿用其直接提交路径。
    virtual void begin_force_virial_batch() {}
    virtual void finish_force_virial_batch() {}
    virtual void cancel_force_virial_batch() noexcept {}
    // 提交一份 CV 的 owned+ghost 力 [total,3] 和 virial [total,6 或 9]。
    // VirialView 显式声明对称六分量或完整九分量；宿主适配器说明原生输出的分量映射。
    // 适配器读取前调用 View 等待钩子；调用返回后源数组可复用，内容保持原样。
    // 批次外立即完成 reverse 和 owned 回写；批次内由 finish 完成。
    // 所有 rank 按相同 CV 顺序调用，包括空中心组。
    virtual void commit_force_virial(ArrayView<const double> forces, VirialView virials) = 0;
    // 全局 CV 累加一个 owned 原子的贡献。force 是 Host 三分量，virial 必须是
    // 恰好一行的 Host 视图；表示及分量顺序与批量提交相同，调用期间借用数据。
    // 支持合并的适配器将这类贡献加入同一批次；纯 owned 批次无需 ghost reverse。
    virtual void add_owned_force_virial(int local_index, const double force[3], VirialView virial) = 0;
};
} // namespace MetaD_zqc
