#include "CV_Stru_factor.h"
#include "lammps.h"
#include "atom.h"
#include "comm.h"
#include "memory.h"
#include "zqc_debug.h"
#include <cmath>

namespace MetaD_zqc {

Stru_fact_chem_env::Stru_fact_chem_env(
    LAMMPS_NS::LAMMPS* lmp, LAMMPS_NS::FixMetadynamics* Fixmetad, FILE* f_check,
    int group_id, double cutoff_r, double c_target, double sigma,
    const std::map<int, double>& custom_weights)
    : Stru_fact_env(lmp, Fixmetad, f_check, group_id, cutoff_r),
      c_target(c_target), sigma(sigma) {
    register_buffer(d_atom_types, "d_atom_types");
    register_buffer(d_type_weights, "d_type_weights");
    const int type_count = lmp->atom->ntypes;
    lmp->memory->grow(h_type_weights, type_count+1, "metad:Stru_fact_env:h_type_weights");
    d_type_weights.grow_to(type_count+1, __FILE__, __LINE__);
    h_type_weights[0] = 0.0;
    for (int atom_type = 1; atom_type <= type_count; ++atom_type) {
        const auto found = custom_weights.find(atom_type);
        if (found != custom_weights.end()) {
            h_type_weights[atom_type] = found->second;
        } else {
            // Chem_map 优先；其余类型按类型编号与 c_target 的距离取高斯权重。
            const double normalized_delta = (double(atom_type)-c_target)/sigma;
            h_type_weights[atom_type] = std::exp(-0.5*normalized_delta*normalized_delta);
        }
        if (lmp->comm->me == 0) LOG("atom_type=%d, type_weights=%g", atom_type, h_type_weights[atom_type]);
    }
    SAFE_CUDA_MEMCPY(d_type_weights.ptr, h_type_weights, (type_count+1)*sizeof(double), cudaMemcpyHostToDevice, f_check);
}

Stru_fact_chem_env::~Stru_fact_chem_env() {
    lmp->memory->destroy(h_type_weights);
}

// 原子迁移或排序后 localid 可改变，类型与本轮中心、ghost 布局同步更新。
void Stru_fact_chem_env::refresh_lmpbox() {
    Stru_fact_env::refresh_lmpbox();
    atom_types = atom->type;
    const size_t total_atom_count = static_cast<size_t>(atom->nlocal) + atom->nghost;
    d_atom_types.grow_to(total_atom_count, __FILE__, __LINE__);
    if (total_atom_count) {
        SAFE_CUDA_MEMCPY(d_atom_types.ptr, atom_types, total_atom_count*sizeof(int), cudaMemcpyHostToDevice, f_check);
    }
}

Stru_factor_chem::Stru_factor_chem(
    LAMMPS_NS::LAMMPS* lmp, LAMMPS_NS::FixMetadynamics* Fixmetad, FILE* f_check,
    Stru_fact_chem_env* environment, double q_factor, int block_size)
    : Stru_factor(lmp, Fixmetad, f_check, environment, q_factor, block_size),
      my_chem_env(environment) {}

void Stru_factor_chem::call_structure_factor_cv_kernel() {
    strufactor_chem_forward(my_env->data, q_factor, my_env->cutoff_r*my_env->cutoff_r,
                           my_chem_env->d_atom_types.ptr, my_chem_env->d_type_weights.ptr,
                           d_stru_factor.ptr, 0, d_block_size);
}

// base apply_adjoint 提供各中心 dV/dS_i；这里将加权邻居项的力和 virial 累加到两端。
void Stru_factor_chem::call_structure_factor_adjoint_kernel(
    const double* center_adjoint, double* atom_forces, double* atom_virials) {
    strufactor_chem_backward(my_env->data, q_factor, my_env->cutoff_r*my_env->cutoff_r,
                            my_chem_env->d_atom_types.ptr, my_chem_env->d_type_weights.ptr,
                            center_adjoint, atom_forces, atom_virials, 0, d_block_size);
}

} // namespace MetaD_zqc
