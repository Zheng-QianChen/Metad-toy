#include "fix_crystallize.h"

#include "lammpsplugin.h"

#include "lammps.h"
#include "update.h"
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "command.h"
#include "domain.h"
#include "force.h"
#include "group.h"
#include "version.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"         // 完整定义Neighbor类
#include "neigh_list.h"        // 定义NeighList结构
#include "pair.h"

#include "zqc_debug.h"
#include "zqc_CVs.h"
#include "CV_Stru_factor.h"

#include <cuda_runtime.h>
#include <cstring>
#include <algorithm>
#include <cmath>

using namespace LAMMPS_NS;

MetaD_zqc::Stru_fact_chem_env::Stru_fact_chem_env(LAMMPS_NS::LAMMPS *lmp, FILE *f_check,
             LAMMPS_NS::FixMetadynamics *Fixmetad, int group_id, double cutoff_r,
             double c_target, double sigma, const std::map<int, double>& custom_weights)
    :Stru_fact_env(lmp, f_check, Fixmetad, group_id, cutoff_r),
    c_target(c_target),
    sigma(sigma)
{    
    lmp->memory->create(h_type_weights, 0, "metad:Stru_fact_env:h_type_weights");

    d_atom_types.set_name("d_atom_types");
    d_type_weights.set_name("d_type_weights");

    int ntypes = lmp->atom->ntypes;
    lmp->memory->grow(h_type_weights, ntypes+1, "metad:Stru_fact_env:h_type_weights");
    d_type_weights.grow_to(ntypes+1, f_check, __FILE__, __LINE__);

    h_type_weights[0] = 0.0; // type index starts from 1 in LAMMPS, so we set the weight of type 0 to 0
    
    for (int i=1; i<=ntypes; i++){
        // 检查用户是否在 Chem_map 中显式指定了该类型的权重
        auto it = custom_weights.find(i);
        if (it != custom_weights.end()) {
            // A. 找到了：直接采用用户赋予的硬编码映射权重 (例如 1.0 或 -3.0)
            h_type_weights[i] = it->second;
        } else {
            // B. 没找到：退回到原先默认的高斯衰减函数逻辑
            double delta = (double(i)-c_target); // default weight is 1.0 for all types
            h_type_weights[i] = exp(-POW2(delta)/(2*POW2(sigma))); // Gaussian weight based on the distance from c_target
        }

        if (lmp->comm->me == 0) {
            LOG("atom_type=%d, type_weights=%g", i, h_type_weights[i]);
        }
    }

    SAFE_CUDA_MEMCPY(d_type_weights.ptr, h_type_weights, (ntypes+1)*sizeof(double), cudaMemcpyHostToDevice, f_check);
}

MetaD_zqc::Stru_factor_chem::Stru_factor_chem(
    LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check,
    std::string env_setNum, int group_id, MetaD_zqc::Stru_fact_chem_env* my_env,
    double q_factor, int d_block_size)
    : Stru_factor(lmp, Fixmetad, f_check, env_setNum, group_id, my_env, q_factor, d_block_size)
{
    // 转回来
    this->my_chem_env = my_env;

    DEBUG_LOG("Logging: Chemical-aware Structure Factor initialized with c_target=%g, sigma=%g",
              my_chem_env->c_target, my_chem_env->sigma);

    // 此时基类的构造函数已经完全执行完毕，所有的内存开辟（h_stru_factor、d_stru_factor 等）
    // 以及 KahanAverager、GPU 设备获取全部一步到位安全落盘，不需要复写任何多余代码！
}


MetaD_zqc::Stru_fact_chem_env::~Stru_fact_chem_env(){
    lmp->memory->destroy(h_type_weights);
    // the GpuBuffer will automatically release its memory, 
    // so we don't need to manually free it here
}


void MetaD_zqc::Stru_fact_chem_env::refresh_lmpbox(){
    // clear the h_group_indices
    atom = lmp->atom;
    mask = (atom)->mask;     // 原子组掩码
    atom_types = atom->type; // 直接使用原子类型作为权重索引
    LAMMPS_NS::tagint Threads_own_atoms = ((atom)->nlocal+(atom)->nghost);

    lmp->memory->grow(h_group_indices, ((atom)->nlocal), "Stru_fact:h_group_indices");
    // group_count = how many aim atoms in local
    last_group_count = group_count;
    group_count = 0; // 当前local中有
    for (int i = 0; i < (atom)->nlocal; i++) {
        if ((mask)[i] & (groupbit)){
            (h_group_indices)[(group_count)] = i; // record local index
            (group_count)++;
        }
    }
    DEBUG_LOG("group_count=%lld",((long long)group_count));

    d_mask.grow_to(Threads_own_atoms, f_check, __FILE__, __LINE__);
    SAFE_CUDA_MEMCPY((d_mask.ptr),(mask),(Threads_own_atoms)*sizeof(int),cudaMemcpyHostToDevice,f_check);

    
    d_atom_types.grow_to(Threads_own_atoms, f_check, __FILE__, __LINE__);
    SAFE_CUDA_MEMCPY(d_atom_types.ptr, atom_types, Threads_own_atoms*sizeof(int), cudaMemcpyHostToDevice, f_check);


    // set up nvidia thread number
    block_num = ((group_count) + d_block_size - 1)/d_block_size;
    N = d_block_size*block_num;
    LOG_COND((((box_x)<2*(cutoff_r))||((box_y)<2*(cutoff_r))||((box_z)<2*(cutoff_r))),"Warning: box < cutoff_r, please check your system !");
}

void MetaD_zqc::Stru_factor_chem::call_structure_factor_cv_kernel(){
    cv_kernel_structure_factor_chem<<<block_num,d_block_size>>>(
        (my_env->group_count), q_factor, POW2(my_env->cutoff_r),
        (my_env->d_group_numneigh.ptr), 
        (my_env->d_group_dminneigh.ptr), 
        (my_env->d_group_indices.ptr), 
        (my_env->d_firstneigh_ptrs.ptr),
        (my_env->d_neigh_in_cutoff_r.ptr), 
        (my_chem_env->d_atom_types.ptr), (my_chem_env->d_type_weights.ptr),
        d_stru_factor.ptr);
}

void MetaD_zqc::Stru_factor_chem::call_structure_factor_dcv_AVE_kernel(){
    // printf("[Rank:%d]d_stein_Ylm is located in %p\n",lmp->comm->me,d_stein_Ylm.ptr);
    dcv_AVE_kernel_structure_factor_chem<<<block_num,d_block_size>>>(
        (my_env->group_count), q_factor, (my_env->groupbit), 
        all_count, POW2(my_env->cutoff_r),
        (my_env->d_mask.ptr), (my_env->d_group_indices.ptr), 
        (my_env->d_calculated_numneigh.ptr), (my_env->d_group_numneigh.ptr),
        (my_env->d_neigh_in_cutoff_r.ptr), (my_env->d_group_dminneigh.ptr),
        d_stru_factor.ptr, 
        (my_chem_env->d_atom_types.ptr), (my_chem_env->d_type_weights.ptr),
        d_dcvdx.ptr);
}

__global__ void cv_kernel_structure_factor_chem(
        int group_count, double q_factor, double cutoff_rsq,
        LAMMPS_NS::tagint *d_group_numneigh,
        double *d_group_dminneigh, 
        LAMMPS_NS::tagint *d_group_indices,
        int *d_firstneigh_ptrs, 
        int *d_neigh_in_cutoff_r, 
        int *d_atom_types, double *d_type_weights,
        double *d_stru_factor){

    int c_atom = blockIdx.x * blockDim.x + threadIdx.x;
    double r_on = 0.8*cutoff_rsq;
    // double ds = 0.0;
    if(c_atom<group_count){
        int neigh_min, neigh_max;
        double chem_weight_catoms = d_type_weights[(int)d_atom_types[d_group_indices[c_atom]]];
        neigh_min = d_group_numneigh[c_atom];
        neigh_max = d_group_numneigh[c_atom] + d_neigh_in_cutoff_r[c_atom];
        d_stru_factor[c_atom] = 0;
        for (int neigh_atom = neigh_min; neigh_atom < neigh_max; neigh_atom++){
            double delt_x, delt_y, delt_z, r2, r;
            double chem_weight;
            double theta;
            double sin_theta, cos_theta;
            delt_x = d_group_dminneigh[neigh_atom*4 + 0];
            delt_y = d_group_dminneigh[neigh_atom*4 + 1];
            delt_z = d_group_dminneigh[neigh_atom*4 + 2];
            r2 = d_group_dminneigh[neigh_atom*4 + 3];
            r      = sqrt(r2);
            theta  = q_factor*r;
            sincos(theta, &sin_theta, &cos_theta);
            int n_local_tag = d_firstneigh_ptrs[neigh_atom];
            chem_weight = chem_weight_catoms*d_type_weights[(int)d_atom_types[n_local_tag]]; // 通过邻居原子类型获取化学权重
            double s = 1.0;
            if (r2 > r_on){
                s = 1.0 - POW3((r2-r_on)/(cutoff_rsq-r_on));
            }
            d_stru_factor[c_atom] += sin_theta/theta*s*chem_weight;
        }
        // d_stru_factor[c_atom] /= (double)(d_neigh_in_cutoff_r[c_atom]);
        d_stru_factor[c_atom] += 1.0;
    }
}

__global__ void dcv_AVE_kernel_structure_factor_chem(
        int group_count, double q_factor, int groupbit, 
        int all_count, double cutoff_rsq,
        int *d_mask, LAMMPS_NS::tagint *d_group_indices, 
        LAMMPS_NS::tagint *d_calculated_numneigh,
        LAMMPS_NS::tagint *d_group_numneigh,
        int *d_neigh_in_cutoff_r, double *d_group_dminneigh,
        double *d_stru_factor, 
        int *d_atom_types, double *d_type_weights,
        double *d_dcvdx){
    
    double r_on = 0.8*cutoff_rsq;
    // devise version=============
    int c_atom = blockIdx.x * blockDim.x + threadIdx.x;
    if(c_atom<group_count){
    // host version===============
    // for (int c_atom=0; c_atom<group_count; c_atom++){
        int neigh_tag, neigh_Nb;
        int neigh_min, neigh_max;
        double NeighInGroupWeight, temp, Stru_fact;
        int Stru_fact_base_id, Stru_fact_neigh_id, Neigh_Nb;
        double dcvdx_local[3] = {0.0, 0.0, 0.0};
        neigh_min = d_group_numneigh[c_atom];
        neigh_max = d_group_numneigh[c_atom] + d_neigh_in_cutoff_r[c_atom];
        int neigh_num = d_neigh_in_cutoff_r[c_atom];
        double chem_weight_catoms = d_type_weights[(int)d_atom_types[d_group_indices[c_atom]]];
        for(int i=0; i<3; i++){
            // from 0 to l, both re_part and im_part
            d_dcvdx[c_atom*3 + i] = 0;
        }
        if (neigh_num == 0) {
            neigh_max=neigh_min=0;
        }
        for(int neigh_atom=neigh_min; neigh_atom<neigh_max; neigh_atom++){
            double s = 1.0;
            double ds = 0.0;
            double dx, dy, dz, r2, r;
            double theta, sin_theta, cos_theta;
            double chem_weight;
            dx = d_group_dminneigh[ neigh_atom*4 + 0];
            dy = d_group_dminneigh[ neigh_atom*4 + 1];
            dz = d_group_dminneigh[ neigh_atom*4 + 2];
            r2     = d_group_dminneigh[ neigh_atom*4 + 3];
            r      = sqrt(r2);
            theta = q_factor*r;
            sincos(theta, &sin_theta, &cos_theta);
            // 处理 neigh 与 cv-group 重合的部分
            neigh_tag = d_calculated_numneigh[neigh_atom];
            NeighInGroupWeight = 1.0;
            if (d_mask[neigh_tag]&groupbit){
                NeighInGroupWeight += 1.0;
            }
            if (r2 > r_on){
                s = 1.0 - POW3((r2-r_on)/(cutoff_rsq-r_on));
                ds = - 3*POW2((r2-r_on)/(cutoff_rsq-r_on)) * (2.0*r) / (cutoff_rsq-r_on);
            } else {
                s = 1.0;
                ds = 0.0;
            }
            chem_weight = chem_weight_catoms*d_type_weights[(int)d_atom_types[neigh_tag]];
            temp = (NeighInGroupWeight / all_count)*(
                        (cos_theta/theta - sin_theta/POW2(theta)) *s * q_factor
                        + ds*sin_theta/theta);
            dcvdx_local[0] -= chem_weight*(temp)*dx/r;
            dcvdx_local[1] -= chem_weight*(temp)*dy/r;
            dcvdx_local[2] -= chem_weight*(temp)*dz/r;
        }
        d_dcvdx[c_atom * 3 + 0] = dcvdx_local[0];
        d_dcvdx[c_atom * 3 + 1] = dcvdx_local[1];
        d_dcvdx[c_atom * 3 + 2] = dcvdx_local[2];
    }
}