#include "CV_Stru_factor.h"
#include "zqc_cv_debug.h"
#include <cmath>

namespace MetaD_zqc {

Stru_fact_chem_env::Stru_fact_chem_env(
    CVContext& context, SelectionId selection, std::string selection_name,
    double cutoff_r, double c_target, double sigma,
    const std::map<int, double>& custom_weights)
    : Stru_fact_env(context, selection, std::move(selection_name), cutoff_r),
      c_target(c_target), sigma(sigma) {
    register_buffer(d_type_weights, "d_type_weights");
    const int type_begin = engine.atom_type_begin();
    const int type_count = engine.atom_type_count();
    if (type_count < 0 || type_begin < 0)
        throw std::invalid_argument("Stru_factor_chem: invalid atom type range");
    // 槽位 [0, begin) 为占位；有效类型下标为 [begin, begin+count)。
    const size_t weight_slots = static_cast<size_t>(type_begin) + static_cast<size_t>(type_count);
    h_type_weights.assign(weight_slots, 0.0);
    d_type_weights.grow_to(weight_slots, __FILE__, __LINE__);
    auto logger = this->logger;
    for (int atom_type = type_begin; atom_type < type_begin + type_count; ++atom_type) {
        const auto found = custom_weights.find(atom_type);
        if (found != custom_weights.end()) {
            h_type_weights[static_cast<size_t>(atom_type)] = found->second;
        } else {
            // Chem_map 优先；其余类型按类型编号与 c_target 的距离取高斯权重。
            const double normalized_delta = (double(atom_type)-c_target)/sigma;
            h_type_weights[static_cast<size_t>(atom_type)] =
                std::exp(-0.5*normalized_delta*normalized_delta);
        }
        if (engine.rank() == 0)
            LOG("atom_type=%d, type_weights=%g", atom_type,
                h_type_weights[static_cast<size_t>(atom_type)]);
    }
    if (weight_slots)
        d_type_weights.upload_from(h_type_weights.data(), weight_slots);
}

Stru_fact_chem_env::~Stru_fact_chem_env() = default;

// 原子迁移或排序后 localid 可改变，类型与本轮中心、ghost 布局同步更新。
void Stru_fact_chem_env::refresh_centers() {
    Stru_fact_env::refresh_centers();
    device_atom_types = engine.atom_types(MemorySpace::Cuda);
}

Stru_factor_chem::Stru_factor_chem(CVContext& context, Stru_fact_chem_env* environment,
                                 double q_factor, int block_size)
    : Stru_factor(context, environment, q_factor, block_size), my_chem_env(environment) {}

void Stru_factor_chem::call_structure_factor_cv_kernel() {
    const size_t nall = static_cast<size_t>(engine.total_atom_count());
    const size_t weight_slots = my_chem_env->h_type_weights.size();
    strufactor_chem_forward(my_env->data, q_factor, my_env->cutoff_r*my_env->cutoff_r,
                            my_chem_env->device_atom_types,
                            my_chem_env->d_type_weights.view(weight_slots),
                            d_stru_factor.view(nall), my_env->execution.get(), d_block_size);
}

void Stru_factor_chem::call_structure_factor_adjoint_kernel(
    ArrayView<const double> center_adjoint, ArrayView<double> atom_forces,
    ArrayView<double> atom_virials) {
    const size_t weight_slots = my_chem_env->h_type_weights.size();
    strufactor_chem_backward(my_env->data, q_factor, my_env->cutoff_r*my_env->cutoff_r,
                             my_chem_env->device_atom_types,
                             my_chem_env->d_type_weights.view(weight_slots),
                             center_adjoint, atom_forces, atom_virials, my_env->execution.get(), d_block_size);
}

} // namespace MetaD_zqc
