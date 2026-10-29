#pragma once
#include "CV_Stru_factor.h"

namespace MetaD_zqc {

// 原始化学加权配位数 W_i = 1 + sum_j w(type_i)*w(type_j)*s(r_ij)。
// s 在 r^2 <= 0.8*cutoff_rsq 内为 1，在外侧过渡区用 1-u^3 降至 0。
// data 借用 Stru_fact_env 的中心表、筛选后邻居和几何，布局见 StruFactorData。
// atom_types 按 localid 借用，可含非 1 的 item_stride；type_weights[type] 长度为
// atom_type_begin()+atom_type_count()，有效类型在 [begin, begin+count)。
// values 按 localid 只写 owned 中心，其余位置由调用方初始化。
// kernel 按 stream 异步提交，底层数组需保持有效直到执行完成。
// 指针重载假定连续 localid 排列，供既有独立 benchmark 使用。
void weighted_chem_forward(const StruFactorData& data, double cutoff_rsq,
                           const int* atom_types, const double* type_weights,
                           double* values, cudaStream_t stream = 0, int block_size = 128);
void weighted_chem_forward(const StruFactorData& data, double cutoff_rsq,
                           ArrayView<const int> atom_types, ArrayView<const double> type_weights,
                           ArrayView<double> values, cudaStream_t stream = 0, int block_size = 128);

// center_adjoint[k] 是图对第 k 个压缩中心的 dV/dW_i，长度 group_count。
// force[localid,3] 累加 xyz 物理力；virial[localid,6] 累加 Symmetric6。
// 独立求值前由调用方清零。对每个中心-邻居项向两端写力，virial 各分一半；
// CV 层将 ghost 贡献 reverse 回 owner。virial 可为空；空中心组直接返回。
void weighted_chem_backward(const StruFactorData& data, double cutoff_rsq,
                            const int* atom_types, const double* type_weights,
                            const double* center_adjoint, double* force, double* virial,
                            cudaStream_t stream = 0, int block_size = 128);
void weighted_chem_backward(const StruFactorData& data, double cutoff_rsq,
                            ArrayView<const int> atom_types, ArrayView<const double> type_weights,
                            ArrayView<const double> center_adjoint, ArrayView<double> force,
                            ArrayView<double> virial, cudaStream_t stream = 0, int block_size = 128);

#ifndef ZQC_STRUFACTOR_KERNELS_ONLY
// 复用 Stru_factor 的原始值、Graph 装配和力/virial 缓冲，仅替换径向计算。
// 每个实例在基类 h_stru_factor/d_stru_factor 中保存自己的 W_i，按 localid 排列。
// Host 副本每轮前向下载，供 chem_pair_r 逐原子诊断；设备副本装配到 Graph Leaf。
// 基类的 d_center_adjoint 按压缩中心排列，d_graph_force/d_graph_virial 按 owned+ghost 排列。
// 这些 GpuBuffer 由本 CV 的基类持有，共享环境只持有几何和固定元素权重。
class Weighted_chem_pair : public Stru_factor {
private:
    Stru_fact_chem_env* my_chem_env; // 借用 env_pool 中的环境，类型与本轮 localid 布局同步。
public:
    static CV* create(CVContext& context, int narg, char** arg, int& i);
    Weighted_chem_pair(CVContext& context, Stru_fact_chem_env* env, int block_size);
    CV_Calculation set_CV_calculate(std::string) override;
    CV_BiasForce set_CV_bias_force(std::string) override;
    void call_structure_factor_cv_kernel() override;
    void call_structure_factor_adjoint_kernel(ArrayView<const double> center_adjoint,
                                              ArrayView<double> atom_forces,
                                              ArrayView<double> atom_virials) override;
    ArrayView<const double> peratom_values(const std::string&) override;
};
#endif

} // namespace MetaD_zqc
