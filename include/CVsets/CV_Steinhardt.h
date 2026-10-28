#pragma once
#include "zqc_cuda_execution.cuh"
#include "zqc_cv.h"
#include "zqc_switch_function.h"
#include "zqc_cv_assembly.cuh"
#include <map>
#include <vector>

namespace MetaD_zqc {
template<int L> class STEIN_QL;
template<int L> class STEIN_LocalQL;
class Steinhardt_env;
class Steinhardt : public CV {
protected:
    int num_elements = 0;
public:
    explicit Steinhardt(CVContext& context) : CV(context) {}
    static CV* create(CVContext&, int, char**, int&);
};
struct SteinhardtRequest {
    SelectionId group_id = all_atoms;
    std::string cal_name;
    const char* group_name = nullptr;
    int Q_num = 0;
    const char* Q_type_str = nullptr;
    int d_block_size = 128;
    double cutoff_r = 4.0, cutoff_eps_r = 1e-6;
    SwitchFunction* SW_FUNC_r = nullptr;
};

// 共享环境归创建它的 EngineAdapter 所有域；引用计数随 Q/Local Q 对象释放。
// 借用的坐标/选择集视图仅在当前 evaluation_id 有效；邻居拓扑按 generation 缓存。
class Steinhardt_env : public CV_info {
    friend class Steinhardt;
    template<int L> friend class STEIN_QL;
    template<int L> friend class STEIN_LocalQL;
protected:
    CudaExecutionStream execution; // 环境和共享 scratch 的串行提交流。
    static std::map<std::string,Steinhardt_env*> env_pool;
    int ref_count = 0;
    bool LOC_flag = false, init_flag = false, graph_geometry_dirty = true;
    std::uint64_t geometry_evaluation = ~std::uint64_t(0), cached_neigh_generation = 0;
    std::uint64_t center_generation = 0;
    bool centers_changed = true;
    std::shared_ptr<const CVAssembly> atom_assembly;
    SwitchFunction* my_r_SWfunc = nullptr;
    double cutoff_r, cutoff_eps_r;
    int group_count = 0, group_id, d_block_size = 128, block_num = 0;
    ArrayView<const double> positions;
    ArrayView<const int> selection_flags;
    // Host 向量拥有内存，h_* 是同步更新的别名；GpuBuffer 拥有对应设备存储。
    // Q 的候选 offsets 按 owned 中心排列；Local Q 按 localid 排列，包含 ghost J 行。
    std::vector<int> center_storage, candidate_offset_storage, candidate_neighbor_storage;
    std::vector<int> active_offset_storage;
    std::vector<double> weight_storage;
    int *h_group_indices = nullptr, *h_group_numneigh = nullptr, *h_firstneigh_ptrs = nullptr;
    int *h_calculated_firstneigh_ptrs = nullptr;
    double* h_neigh_in_switching = nullptr;
    GpuBuffer<int> d_group_indices, d_group_numneigh, d_firstneigh_ptrs;
    // 有效邻居写入候选段前缀；方向缓存 offsets 按 calctag 排列。
    GpuBuffer<int> d_neigh_in_cutoff_r, d_calculated_numneigh, d_calculated_firstneigh_ptrs;
    GpuBuffer<double> d_neigh_in_switching;
    int num_of_all_calc_fullpair = 0, n_half_candidates = 0, n_active_pairs = 0;
    GpuBuffer<int> d_half_pair_i, d_half_pair_j, d_half_to_full, d_full_to_half;
    GpuBuffer<int> d_active_pair_mask, d_active_pair_ids;
public:
    int neigh_id = -1;
    Steinhardt_env(CVContext&, SteinhardtRequest);
    virtual ~Steinhardt_env() = default;
    static Steinhardt_env* get_or_create(CVContext&, SteinhardtRequest);
    std::string get_env_key();
    void register_env() { ++ref_count; }
    void unregister_env() {
        if (--ref_count == 0) { env_pool.erase(get_env_key()); delete this; }
    }
    virtual void refresh_centers();
    virtual void get_env();
};

template<int L> class STEIN_QL : public Steinhardt {
protected:
    int stein_l = L, d_block_size, block_num = 0;
    size_t N = 0;
    std::string env_setNum;
    Steinhardt_env* my_env;
    // q_l：[localid]；q_lm：[localid][2*(L+1)]，m=0..L 实部/虚部交错。
    std::vector<double> value_storage, harmonic_storage, gradient_storage, virial_storage;
    double *stein_q = nullptr, *h_stein_qlm = nullptr, *h_dcvdx = nullptr, *h_a_virial = nullptr;
    GpuBuffer<double> d_stein_ql, d_stein_qlm, d_stein_Ylm;
    // 坐标梯度 [localid][3]；完整 virial [localid][9]，xx yy zz xy xz yz yx zx zy。
    // d_physical_force 在设备上将梯度取负为物理力，避免常规 D2H。
    GpuBuffer<double> d_dcvdx, d_a_virial, d_physical_force;
    // Graph 梯度按 owned 中心提取，forward 通信后补齐另一端的值、球谐和梯度。
    GpuBuffer<double> d_center_adjoint, d_atom_adjoint;
    std::vector<double> h_center_adjoint, h_atom_adjoint, force_storage;
    // 按需归并到 owner 的诊断梯度；同轮请求 x/y/z 时共用一次 reverse。
    std::vector<double> output_gradient;
    bool output_gradient_ready = false;
    bool graph_values_ready = false;
    std::uint64_t graph_value_evaluation = ~std::uint64_t(0);
    void check_graph_values() const;
    void commit_coordinate_adjoint();
public:
    STEIN_QL(CVContext&, std::string, int, int, Steinhardt_env*, int);
    ~STEIN_QL() override;
    CV_Calculation set_CV_calculate(std::string) override;
    CV_BiasForce set_CV_bias_force(std::string) override;
    ViewReadiness device_readiness() const override { return my_env->execution.ready(); }
    void base_calc() override;
    int graph_selection() const override { return my_env->group_id; }
    void bind_assembly(std::shared_ptr<const CVAssembly>) override;
    void prepare_graph_evaluation() override;
    void assemble_values(ArrayView<double>) override;
    void apply_adjoint(ArrayView<const double>) override;
    virtual void compute_Q_peratoms();
    virtual void steinhardt_param_calc(double*);
    void call_steinhardt_cv_ql_i_kernel();
    void summary(FILE*) override;
    int get_comm_forward_width() const override;
    int pack_comm_forward_ubuf(int,int*,double*,int,int) override;
    void unpack_comm_forward_ubuf(int,int,double*,int,int) override;
    int get_comm_reverse_width() const override { return 3; }
    int pack_comm_reverse_ubuf(int,int,double*,int,int) override;
    void unpack_comm_reverse_ubuf(int,int*,double*,int,int) override;
    ArrayView<const double> peratom_values(const std::string&) override;
};
    #define CALC_MASK_ACTIVE (1U << 24)  // 0x01000000
    #define CALC_MASK_IS_I   (1U << 25)  // 0x02000000
    #define CALC_MASK_IS_J   (1U << 26)  // 0x04000000
    #define CALC_MASK_IS_K   (1U << 27)  // 0x08000000
    // ---- 2. 计数器的位移与掩码 (Counters) ----
    #define CALC_SHIFT_I     16
    #define CALC_SHIFT_J     8
    #define CALC_SHIFT_K     0
    #define CALC_COUNTER_MASK 0xFFU      // 八位最大值 255


// calctag 前 group_count 项是 owned I，随后为 I 的邻居 J（可含 ghost）。
// calc_tag[localid] 是反向映射，未参与的原子为 -1；GPU 每轮重新标记 I/J。
class STEIN_LocalQL_env : public Steinhardt_env {
    template<int L> friend class STEIN_LocalQL;
protected:
    GpuBuffer<int> d_calc_tag, d_LQ_mask, d_pure_J_write_offset;
    GpuBuffer<int> d_neighbor_ends;
    int num_of_all_IJ_atoms = 0;
public:
    STEIN_LocalQL_env(CVContext&, SteinhardtRequest);
    void refresh_centers() override;
    void get_env() override;
};

template<int L> class STEIN_LocalQL : public STEIN_QL<L> {
protected:
    using CV::engine;
    using CV::logger;
    using CV::assembly_;
    using STEIN_QL<L>::my_env;
    using STEIN_QL<L>::block_num;
    using STEIN_QL<L>::d_block_size;
    using STEIN_QL<L>::N;
    using STEIN_QL<L>::stein_l;
    using STEIN_QL<L>::env_setNum;
    using STEIN_QL<L>::stein_q;
    using STEIN_QL<L>::d_stein_ql;
    using STEIN_QL<L>::h_stein_qlm;
    using STEIN_QL<L>::d_stein_qlm;
    using STEIN_QL<L>::h_dcvdx;
    using STEIN_QL<L>::d_dcvdx;
    using STEIN_QL<L>::h_a_virial;
    using STEIN_QL<L>::d_a_virial;
    using STEIN_QL<L>::d_center_adjoint;
    using STEIN_QL<L>::check_graph_values;
    using STEIN_QL<L>::commit_coordinate_adjoint;
    using STEIN_QL<L>::value_storage;
    using STEIN_QL<L>::gradient_storage;
    using STEIN_QL<L>::virial_storage;
    STEIN_LocalQL_env* my_loc_env;
    // [localid][2*(L+1)]：局域平均球谐值。
    GpuBuffer<double> d_stein_LQlm;
    // [calctag][2*(L+1)]：所有 owned I 汇总到内层 q_lm 的反向前因子。
    GpuBuffer<double> d_dcvdx_rjk_prefix;
    // [owned center][2*(L+1)]：外层径向权重乘邻居 q_lm 的和。
    GpuBuffer<double> sum_of_qlm_value_weights;
    void evaluate_coordinate_adjoint();
public:
    using CV::register_buffer;
    STEIN_LocalQL(CVContext&, std::string, int, Steinhardt_env*, int);
    void summary(FILE*) override;
    void apply_adjoint(ArrayView<const double>) override;
    void compute_Q_peratoms() override;
    void steinhardt_param_calc(double*) override;
    void call_steinhardt_Local_forward_kernel();
    void call_steinhardt_Local_backward_kernel();
    CV::CV_Calculation set_CV_calculate(std::string) override;
    CV::CV_BiasForce set_CV_bias_force(std::string) override;
    int get_comm_forward_width() const override { return 0; }
};
Steinhardt* create_steinhardt_cv(CVContext&,std::string,int,int,Steinhardt_env*,SteinhardtRequest);
} // namespace MetaD_zqc

// =============================================================================
// not LOCAL
// =============================================================================

__global__ void get_environment_Steinhardt_Q(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, int start_idx,
    double cutoff_r, double cut_sigma_eps,
    // in
    int *d_group_indices,
    int *d_group_numneigh,
    int *d_firstneigh_ptrs, MetaD_zqc::ArrayView<const double> d_x_flat,
    int *d_full_to_half,
    // out
    int *d_neigh_in_cutoff_r, int *d_active_pair_mask,
    double *d_neigh_in_switching,
    int *d_calculated_numneigh);


template <int L>
__global__ void steinhardt_cv_kernel(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, double cutoff_r, double cutoff_eps,
    int *d_group_indices,
    int *d_neigh_in_cutoff_r,
    int *d_group_numneigh,
    int *d_calculated_firstneigh_ptrs,
    int *d_calculated_numneigh,
    MetaD_zqc::ArrayView<const double> d_x_flat,
    double *d_neigh_in_switching,
    double *d_stein_qlm, double *d_stein_Ylm, double *d_stein_ql);


// Graph 梯度按 localid 输入；每个半对汇总两端的力和 virial。
template <int L>
__global__ void steinhardt_dcv_kernel(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int pair_all, int groupbit,
    MetaD_zqc::ArrayView<const int> d_mask,
    int *d_active_pair_ids,
    int *d_half_pair_i, int *d_half_pair_j,
    MetaD_zqc::ArrayView<const double> d_x_flat,
    double *d_neigh_in_switching,
    double *d_stein_ql, double *d_stein_qlm,
    double *d_dcvdx, double *d_a_virial,
    const double *atom_adjoint
);


// =============================================================================
// LOCAL
// =============================================================================
__global__ void get_environment_Steinhardt_LocalQ(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, int start_idx,
    double cutoff_r, double cut_sigma_eps,
    // in
    int *d_group_indices,
    int *d_calc_tag,
    int *d_group_numneigh,
    const int *d_neighbor_ends, MetaD_zqc::ArrayView<const int> candidates,
    MetaD_zqc::ArrayView<const double> d_x_flat, bool mark_neighbors,
    // out
    int *d_neigh_in_cutoff_r, int *d_LQ_mask,
    double *d_neigh_in_switching,
    int *d_calculated_numneigh);

__global__ void get_environment_Steinhardt_LocalQ_promote_pure_K(
    int group_count,
    int atom_all,
    const int *d_LQ_mask,
    int *d_pure_J_write_offset,
    int *d_group_indices,
    int *d_calc_tag);

template <int L>
__global__ void steinhardt_Local_cv_get_qlm(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, double cutoff_r, double cutoff_eps,
    int *d_group_indices,
    int *d_neigh_in_cutoff_r,
    int *d_group_numneigh,
    int *d_calculated_numneigh,
    MetaD_zqc::ArrayView<const double> d_x_flat,
    double *d_neigh_in_switching,
    double *d_stein_qlm);

template <int L>
__global__ void steinhardt_Local_cv_get_LQl(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, double cutoff_r, double cutoff_eps,
    int *d_group_indices,
    int *d_neigh_in_cutoff_r,
    int *d_group_numneigh,
    int *d_calculated_numneigh,
    MetaD_zqc::ArrayView<const double> d_x_flat,
    double *d_neigh_in_switching,
    double *sum_of_qlm_value_weights,
    double *d_stein_qlm,
    double *d_stein_LQlm, double *d_stein_ql);

template <int L>
__global__ void steinhardt_Local_backward_ij_kernel(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    const double *center_adjoint,
    int group_count,
    MetaD_zqc::ArrayView<const double> d_x_flat, double *d_neigh_in_switching,
    int *d_neigh_in_cutoff_r,
    int *d_group_indices,
    int *d_group_numneigh,
    int *d_calculated_numneigh,
    int *d_calc_tag,
    double *sum_of_qlm_value_weights,
    double *d_dcvdx_rjk_prefix,
    double *d_stein_qlm,
    double *d_stein_LQlm,
    double *d_stein_ql,
    double *d_dcvdx, double *d_a_virial);

template <int L>
__global__ void steinhardt_Local_backward_jk_kernel(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, double cutoff_r, double cutoff_eps,
    MetaD_zqc::ArrayView<const double> d_x_flat, double *d_neigh_in_switching,
    int *d_group_indices,
    int *d_group_numneigh,
    int *d_calculated_numneigh,
    int *d_calc_tag,
    int *d_neigh_in_cutoff_r,
    double *d_dcvdx_rjk_prefix,
    double *d_stein_qlm, double *d_stein_ql,
    double *d_dcvdx, double *d_a_virial);


// =========================================================================
// 🚀 1. 状态标志位操作宏 (Flags Operations) - 纯文本替换，全括号防御
// =========================================================================

// 检查状态（返回 bool 表达式）
#define MASK_IS_ACTIVE(mask) (((mask) & CALC_MASK_ACTIVE) != 0)
#define MASK_IS_I(mask)      (((mask) & CALC_MASK_IS_I)   != 0)
#define MASK_IS_J(mask)      (((mask) & CALC_MASK_IS_J)   != 0)
#define MASK_IS_K(mask)      (((mask) & CALC_MASK_IS_K)   != 0)

// 本地打标签（非并发安全，用于 CPU 或线程独占写入）
#define MASK_SET_ACTIVE(mask) ((mask) |= CALC_MASK_ACTIVE)
#define MASK_SET_I(mask)      ((mask) |= CALC_MASK_IS_I)
#define MASK_SET_J(mask)      ((mask) |= CALC_MASK_IS_J)
#define MASK_SET_K(mask)      ((mask) |= CALC_MASK_IS_K)

// 本地清除标签
#define MASK_CLEAR_ACTIVE(mask) ((mask) &= ~CALC_MASK_ACTIVE)

// =========================================================================
// 🚀 2. 计数器操作宏 (Counters Operations)
// =========================================================================

// 获取计数器的值（返回 int 表达式）
#define MASK_GET_COUNT_I(mask) (((mask) >> CALC_SHIFT_I) & CALC_COUNTER_MASK)
#define MASK_GET_COUNT_J(mask) (((mask) >> CALC_SHIFT_J) & CALC_COUNTER_MASK)
#define MASK_GET_COUNT_K(mask) (((mask) >> CALC_SHIFT_K) & CALC_COUNTER_MASK)

// =========================================================================
// 🚀 3. GPU 专用：硬件级原子并发操作宏（带 __CUDACC__ 保护）
// =========================================================================
#ifdef __CUDACC__

// 原子打标签（必须传入原始指针，例如 &d_LQ_mask[i]）
#define MASK_ATOMIC_SET_ACTIVE(mask_ptr) (atomicOr((unsigned int*)(mask_ptr), CALC_MASK_ACTIVE))
#define MASK_ATOMIC_SET_I(mask_ptr)      (atomicOr((unsigned int*)(mask_ptr), CALC_MASK_IS_I))
#define MASK_ATOMIC_SET_J(mask_ptr)      (atomicOr((unsigned int*)(mask_ptr), CALC_MASK_IS_J))
#define MASK_ATOMIC_SET_K(mask_ptr)      (atomicOr((unsigned int*)(mask_ptr), CALC_MASK_IS_K))
#define MASK_ATOMIC_SET_VAR(mask_ptr, target_mask) \
    (atomicOr((unsigned int*)(mask_ptr), CALC_MASK_ACTIVE | (target_mask)))

// 原子自增计数器
#define MASK_ATOMIC_INC_I(mask_ptr)      (atomicAdd((unsigned int*)(mask_ptr), (1U << CALC_SHIFT_I)))
#define MASK_ATOMIC_INC_J(mask_ptr)      (atomicAdd((unsigned int*)(mask_ptr), (1U << CALC_SHIFT_J)))
#define MASK_ATOMIC_INC_K(mask_ptr)      (atomicAdd((unsigned int*)(mask_ptr), (1U << CALC_SHIFT_K)))
#define MASK_ATOMIC_INC_VAR(mask_ptr, inc_val) \
    (atomicAdd((unsigned int*)(mask_ptr), (inc_val)))

#endif
