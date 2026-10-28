#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <sstream>
#include <iomanip>
#include "CV_Steinhardt.h"
#include "zqc_cv_debug.h"
#include "CV_Steinhardt_math.h"
#include "zqc_cv_parse.h"
using namespace MetaD_zqc;
namespace {
__global__ void initialize_localql_tags(int count, const int* centers, int* tags) {
    const int row=blockIdx.x*blockDim.x+threadIdx.x;
    if (row<count) tags[centers[row]]=row;
}
__global__ void localql_row_counts(int rows, ArrayView<const int> centers,
                                   ArrayView<const int> offsets, int* counts) {
    const int row=blockIdx.x*blockDim.x+threadIdx.x;
    if (row<rows) counts[centers(row)]=offsets(row+1)-offsets(row);
}
__global__ void localql_copy_rows(int rows, ArrayView<const int> centers,
                                 ArrayView<const int> offsets, ArrayView<const int> neighbors,
                                 const int* dense_offsets, int* dense_neighbors) {
    const int row=blockIdx.x*blockDim.x+threadIdx.x;
    if (row>=rows) return;
    int write=dense_offsets[centers(row)];
    for (int k=offsets(row);k<offsets(row+1);++k) dense_neighbors[write++]=neighbors(k);
}
template<class T> void resize_host(std::vector<T>& storage, T*& alias, size_t count) {
    storage.resize(std::max<size_t>(count,1));
    alias=storage.data();
}
}

template<int L> STEIN_LocalQL<L>::STEIN_LocalQL(CVContext& context,std::string key,int group,
                                               Steinhardt_env* environment,int block_size)
    : STEIN_QL<L>(context,std::move(key),group,L,environment,block_size),
      my_loc_env(static_cast<STEIN_LocalQL_env*>(environment)) {}
STEIN_LocalQL_env::STEIN_LocalQL_env(CVContext& context,SteinhardtRequest request)
    : Steinhardt_env(context,request) { d_pure_J_write_offset.grow_to(1,__FILE__,__LINE__); }
void STEIN_LocalQL_env::refresh_centers() {
    Steinhardt_env::refresh_centers();
    const int total=engine.total_atom_count();
    d_group_indices.grow_to(std::max(total,1),__FILE__,__LINE__);
    if (centers_changed) d_group_indices.upload_from(h_group_indices,group_count, execution.get());
    d_calc_tag.grow_to(std::max(total,1),__FILE__,__LINE__);
    execution_check(cudaMemsetAsync(d_calc_tag.ptr,0xff,size_t(total)*sizeof(int),execution.get()));
    if (group_count) {
        initialize_localql_tags<<<(group_count+255)/256,256,0,execution.get()>>>(
            group_count,d_group_indices.ptr,d_calc_tag.ptr);
        CUDA_LAUNCH_CHECK();
    }
}
template<int L> void STEIN_LocalQL<L>::compute_Q_peratoms() {
    if(my_loc_env->graph_geometry_dirty || my_loc_env->geometry_evaluation!=engine.evaluation_id()) {
        my_loc_env->refresh_centers();my_loc_env->get_env();my_loc_env->graph_geometry_dirty=false;
    }
    block_num=(my_loc_env->group_count+d_block_size-1)/d_block_size;
    steinhardt_param_calc(stein_q);
}
void STEIN_LocalQL_env::get_env() {
    const int atom_count=engine.total_atom_count(), storage_count=std::max(atom_count,1);
    const auto candidates=engine.neighbors().prepare(neigh_id);
    candidates.centers.wait_on(execution.get()); candidates.offsets.wait_on(execution.get());
    candidates.neighbors.wait_on(execution.get());
    candidates.centers.validate(); candidates.offsets.validate(); candidates.neighbors.validate();
    ERR_COND(candidates.coordinates!=NeighborCoordinates::PeriodicImages,
             "Local Q requires image-resolved neighbor coordinates");
    if(!init_flag || cached_neigh_generation!=candidates.topology_generation) {
        ERR_COND(candidates.centers.memory!=MemorySpace::Cuda || candidates.offsets.memory!=MemorySpace::Cuda ||
                 candidates.neighbors.memory!=MemorySpace::Cuda || candidates.offsets.item_count!=candidates.centers.item_count+1 ||
                 candidates.neighbors.item_count>size_t(INT_MAX),"Local Q requires device CSR candidates");
        // Convert compressed rows to localid-indexed rows entirely on the device.
        // The mask is temporary row-count storage until geometric I/J/K marking below.
        d_LQ_mask.grow_to(storage_count,__FILE__,__LINE__);
        d_LQ_mask.clear_async(execution.get());
        const int rows=static_cast<int>(candidates.centers.item_count);
        if (rows) localql_row_counts<<<(rows+255)/256,256,0,execution.get()>>>(
            rows,candidates.centers,candidates.offsets,d_LQ_mask.ptr);
        CUDA_LAUNCH_CHECK();
        d_group_numneigh.grow_to(atom_count+1,__FILE__,__LINE__);
        d_LQ_mask.scan_to(d_group_numneigh,atom_count,execution.get());
        d_firstneigh_ptrs.grow_to(std::max<size_t>(candidates.neighbors.item_count,1),__FILE__,__LINE__);
        if (rows) localql_copy_rows<<<(rows+255)/256,256,0,execution.get()>>>(
            rows,candidates.centers,candidates.offsets,candidates.neighbors,d_group_numneigh.ptr,d_firstneigh_ptrs.ptr);
        CUDA_LAUNCH_CHECK();
        cached_neigh_generation=candidates.topology_generation;init_flag=true;
    }
    const int candidate_pairs=static_cast<int>(candidates.neighbors.item_count);
    // 每轮重建 I/J 标记和有效邻居，避免同一步重新求值时沿用旧几何。
    d_LQ_mask.grow_to(storage_count, __FILE__, __LINE__);
    d_neigh_in_cutoff_r.grow_to(storage_count, __FILE__, __LINE__);
    d_neigh_in_switching.grow_to(storage_count, __FILE__, __LINE__);
    d_calculated_numneigh.grow_to(std::max<int>(candidate_pairs, 1), __FILE__, __LINE__);
    d_LQ_mask.clear_async(execution.get());
    d_neigh_in_cutoff_r.clear_async(execution.get());
    d_neigh_in_switching.clear_async(execution.get());
    if (group_count) {
        get_environment_Steinhardt_LocalQ<<<block_num,d_block_size,0,execution.get()>>>(
            my_r_SWfunc->params, group_count, 0, cutoff_r, cutoff_eps_r,
            d_group_indices.ptr, d_calc_tag.ptr, d_group_numneigh.ptr,
            d_firstneigh_ptrs.ptr, positions, CALC_MASK_IS_I, CALC_MASK_IS_J, CALC_SHIFT_J,
            d_neigh_in_cutoff_r.ptr, d_LQ_mask.ptr, d_neigh_in_switching.ptr, d_calculated_numneigh.ptr);
        CUDA_LAUNCH_CHECK();
    }
    int extra_neighbors = 0;
    d_pure_J_write_offset.clear_async(execution.get());
    if (group_count) {
        const int atom_blocks = (atom_count + d_block_size - 1) / d_block_size;
        get_environment_Steinhardt_LocalQ_promote_pure_K<<<atom_blocks,d_block_size,0,execution.get()>>>(
            group_count, atom_count, d_LQ_mask.ptr, d_pure_J_write_offset.ptr,
            d_group_indices.ptr, d_calc_tag.ptr);
        d_pure_J_write_offset.download_to(&extra_neighbors, 1, execution.get());
    }
    num_of_all_IJ_atoms = group_count + extra_neighbors;
    if (extra_neighbors) {
        const int neighbor_blocks = (extra_neighbors + d_block_size - 1) / d_block_size;
        get_environment_Steinhardt_LocalQ<<<neighbor_blocks,d_block_size,0,execution.get()>>>(
            my_r_SWfunc->params, extra_neighbors, group_count, cutoff_r, cutoff_eps_r,
            d_group_indices.ptr, d_calc_tag.ptr, d_group_numneigh.ptr,
            d_firstneigh_ptrs.ptr, positions, CALC_MASK_IS_J, CALC_MASK_IS_K, CALC_SHIFT_K,
            d_neigh_in_cutoff_r.ptr, d_LQ_mask.ptr, d_neigh_in_switching.ptr, d_calculated_numneigh.ptr);
        CUDA_LAUNCH_CHECK();
    }
    // calctag 前缀和给每个 I/J 的球谐方向缓存分配连续段。
    d_calculated_firstneigh_ptrs.grow_to(num_of_all_IJ_atoms+1, __FILE__, __LINE__);
    d_calculated_firstneigh_ptrs.clear_async(execution.get());
    if (num_of_all_IJ_atoms)
        d_neigh_in_cutoff_r.scan_to(d_calculated_firstneigh_ptrs, num_of_all_IJ_atoms, execution.get());
    // Only the terminal count is needed for harmonic-buffer allocation, not the CSR.
    execution_check(cudaMemcpyAsync(&num_of_all_calc_fullpair,
        d_calculated_firstneigh_ptrs.ptr+num_of_all_IJ_atoms,sizeof(int),cudaMemcpyDeviceToHost,execution.get()));
    execution.synchronize();
    geometry_evaluation = engine.evaluation_id();
}

template <int L>
void MetaD_zqc::STEIN_LocalQL<L>::apply_adjoint(ArrayView<const double> leaf_adjoint) {
    check_graph_values();
    d_center_adjoint.grow_to(assembly_->group_count(), __FILE__, __LINE__);
    assembly_->gather_adjoint(leaf_adjoint, d_center_adjoint.view(assembly_->group_count()), my_env->execution.get());
    evaluate_coordinate_adjoint();
    commit_coordinate_adjoint();
}

// 两层几何反向以本 rank 的 owned I 为源，所有 ghost 力和 virial 统一回传 owner。
template <int L>
void MetaD_zqc::STEIN_LocalQL<L>::evaluate_coordinate_adjoint() {
    const int atom_count = engine.total_atom_count();
    const size_t storage_count = std::max(atom_count, 1);
    resize_host(gradient_storage,h_dcvdx,storage_count*3);
    resize_host(virial_storage,h_a_virial,storage_count*9);
    d_dcvdx.grow_to(storage_count*3, __FILE__, __LINE__);
    d_a_virial.grow_to(storage_count*9, __FILE__, __LINE__);
    d_dcvdx.clear_async(my_env->execution.get());
    d_a_virial.clear_async(my_env->execution.get());
    d_dcvdx_rjk_prefix.grow_to(size_t(my_loc_env->num_of_all_IJ_atoms)*2*(L+1), __FILE__, __LINE__);
    d_dcvdx_rjk_prefix.clear_async(my_env->execution.get());
    call_steinhardt_Local_backward_kernel();
}

// CV 基类保留标量回调槽；Local Q 的规约和值开关由 Graph 构建。
template <int L>
auto MetaD_zqc::STEIN_LocalQL<L>::set_CV_calculate(std::string)
    -> typename MetaD_zqc::CV::CV_Calculation {
    throw std::runtime_error("STEINH L requires EXECUTOR GRAPH; use LQ.AVE or LQ.RAW with AVE(expr)");
}

template <int L>
auto MetaD_zqc::STEIN_LocalQL<L>::set_CV_bias_force(std::string)
    -> typename MetaD_zqc::CV::CV_BiasForce {
    throw std::runtime_error("STEINH L requires EXECUTOR GRAPH; gradients are supplied by Graph");
}

template <int L>
void MetaD_zqc::STEIN_LocalQL<L>::steinhardt_param_calc(double *stein_ql){
    const int atom_count = engine.total_atom_count();
    const size_t storage_count = std::max(atom_count, 1);
    const size_t harmonic_count = storage_count*2*(L+1);
    // 值与球谐分量按 localid 存储；sum_of_qlm_value_weights 按中心 calctag 存储。
    resize_host(value_storage,stein_q,storage_count);
    stein_ql = stein_q;
    d_stein_ql.grow_to(storage_count, __FILE__, __LINE__);
    d_stein_qlm.grow_to(harmonic_count, __FILE__, __LINE__);
    d_stein_LQlm.grow_to(harmonic_count, __FILE__, __LINE__);
    sum_of_qlm_value_weights.grow_to(size_t(my_loc_env->group_count)*2*(L+1), __FILE__, __LINE__);
    d_stein_Ylm.grow_to(size_t(my_loc_env->num_of_all_calc_fullpair)*2*(L+1), __FILE__, __LINE__);
    d_stein_ql.clear_async(my_env->execution.get());
    d_stein_qlm.clear_async(my_env->execution.get());
    d_stein_LQlm.clear_async(my_env->execution.get());
    sum_of_qlm_value_weights.clear_async(my_env->execution.get());
    if (my_loc_env->group_count) call_steinhardt_Local_forward_kernel();

}


template <int L>
void MetaD_zqc::STEIN_LocalQL<L>::summary(FILE* f){}

template <int L>
void MetaD_zqc::STEIN_LocalQL<L>::call_steinhardt_Local_forward_kernel(){
    int temp_block_num = (my_loc_env->num_of_all_IJ_atoms+d_block_size-1)/d_block_size;
    CUDA_LAUNCH_CHECK();
    steinhardt_Local_cv_get_qlm<L> <<<temp_block_num,d_block_size,0,my_env->execution.get()>>>(
        (my_loc_env)->my_r_SWfunc->params,
        (my_loc_env->num_of_all_IJ_atoms), (my_loc_env->cutoff_r), (my_loc_env->cutoff_eps_r),
        (my_loc_env->d_group_indices.ptr),
        (my_loc_env->d_neigh_in_cutoff_r.ptr),
        (my_loc_env->d_group_numneigh.ptr),
        (my_loc_env->d_calculated_firstneigh_ptrs.ptr),
        (my_loc_env->d_calculated_numneigh.ptr),
        (my_loc_env->positions),
        (my_loc_env->d_neigh_in_switching.ptr),
        d_stein_qlm.ptr, d_stein_Ylm.ptr) ;
    CUDA_LAUNCH_CHECK();
    steinhardt_Local_cv_get_LQl<L> <<<block_num,d_block_size,0,my_env->execution.get()>>>(
        (my_loc_env)->my_r_SWfunc->params,
        (my_loc_env->group_count), (my_loc_env->cutoff_r), (my_loc_env->cutoff_eps_r),
        (my_loc_env->d_group_indices.ptr),
        (my_loc_env->d_neigh_in_cutoff_r.ptr),
        (my_loc_env->d_group_numneigh.ptr),
        (my_loc_env->d_calculated_firstneigh_ptrs.ptr),
        (my_loc_env->d_calculated_numneigh.ptr),
        (my_loc_env->positions),
        (my_loc_env->d_neigh_in_switching.ptr),
        sum_of_qlm_value_weights.ptr,
        d_stein_qlm.ptr,
        d_stein_LQlm.ptr, d_stein_ql.ptr) ;
    CUDA_LAUNCH_CHECK();
}


template <int L>
void MetaD_zqc::STEIN_LocalQL<L>::call_steinhardt_Local_backward_kernel(){
    if (my_loc_env->group_count == 0) return;
    // I→J：传播中心梯度，并累积外层径向权重产生的力。
    steinhardt_Local_backward_ij_kernel<L> <<<block_num,d_block_size,0,my_env->execution.get()>>>(
        (my_loc_env)->my_r_SWfunc->params,
        d_center_adjoint.ptr,
        (my_loc_env)->group_count,

        (my_loc_env->positions),
        (my_loc_env->d_neigh_in_switching.ptr),
        (my_loc_env->d_neigh_in_cutoff_r.ptr),
        (my_loc_env->d_calculated_firstneigh_ptrs.ptr),
        (my_loc_env->d_group_indices.ptr),
        (my_loc_env->d_group_numneigh.ptr),
        (my_loc_env->d_calculated_numneigh.ptr),
        (my_loc_env->d_calc_tag.ptr),

        sum_of_qlm_value_weights.ptr,
        d_dcvdx_rjk_prefix.ptr,
        d_stein_qlm.ptr,
        d_stein_LQlm.ptr,
        d_stein_ql.ptr,
        d_dcvdx.ptr, d_a_virial.ptr);
    // J→K：使用已汇总的 q_lm 梯度求内层几何力和 virial。
    CUDA_LAUNCH_CHECK();
    int temp_block_num = ((my_loc_env->num_of_all_IJ_atoms)+d_block_size-1)/d_block_size;
    steinhardt_Local_backward_jk_kernel<L> <<<temp_block_num,d_block_size,0,my_env->execution.get()>>>(
        (my_loc_env)->my_r_SWfunc->params,
        (my_loc_env)->num_of_all_IJ_atoms,
        (my_loc_env)->cutoff_r,
        (my_loc_env)->cutoff_eps_r,

        (my_loc_env->positions),
        (my_loc_env->d_neigh_in_switching.ptr),
        (my_loc_env->d_group_indices.ptr),
        (my_loc_env->d_group_numneigh.ptr),
        (my_loc_env->d_calculated_firstneigh_ptrs.ptr),
        (my_loc_env->d_calculated_numneigh.ptr),
        (my_loc_env->d_calc_tag.ptr),
        (my_loc_env->d_neigh_in_cutoff_r.ptr),

        d_dcvdx_rjk_prefix.ptr,
        d_stein_qlm.ptr, d_stein_Ylm.ptr,  d_stein_ql.ptr,
        d_dcvdx.ptr, d_a_virial.ptr);
    CUDA_LAUNCH_CHECK();
}


__global__ void get_environment_Steinhardt_LocalQ(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, int start_idx,
    double cutoff_r, double cut_sigma_eps,
    // in
    int *d_group_indices,
    int *d_calc_tag,
    int *d_group_numneigh,
    int *d_firstneigh_ptrs, MetaD_zqc::ArrayView<const double> d_x_flat,
    unsigned int current_pass_mask,   // 中心原子该打的标签 (例如：CALC_MASK_IS_I 或 CALC_MASK_IS_J)
    unsigned int neighbor_mask,       // 邻居原子该打的标签 (例如：CALC_MASK_IS_J 或 CALC_MASK_IS_K)
    unsigned int neighbor_inc_val,    // 邻居原子计数器自增的位移量 (例如：1U << CALC_SHIFT_I)
    // out
    int *d_neigh_in_cutoff_r, int *d_LQ_mask,
    double *d_neigh_in_switching,
    int *d_calculated_numneigh){

    #define sw_f(r) (MetaD_zqc::SwitchFunction::f(sw_params_rij, (r)))
    // get_environment_Steinhardt_LocalQ in GPU
    int c_atom = blockIdx.x * blockDim.x + threadIdx.x;
    if(c_atom<calc_count){
        int c_atom_calctag = c_atom+start_idx;
        int c_atom_loctag = d_group_indices[c_atom_calctag];
        int temp_tag;
        d_neigh_in_cutoff_r[c_atom_calctag] = 0;
        // c_glob_tag = h_tag[c_atom_loctag];
        double c_x = d_x_flat(c_atom_loctag,0);
        double c_y = d_x_flat(c_atom_loctag,1);
        double c_z = d_x_flat(c_atom_loctag,2);
        double sum_of_sigma_CNatoms = 0;
        int sum_of_numneigh = 0;
        // give tag mask
        MASK_ATOMIC_SET_ACTIVE(&d_LQ_mask[c_atom_loctag]);
        MASK_ATOMIC_SET_VAR(&d_LQ_mask[c_atom_loctag],current_pass_mask);
        int max_ii=0;
        //find curtoff_Natoms neigh
        int start_neigh = d_group_numneigh[c_atom_loctag];
        for (int neigh_atom=start_neigh;
                neigh_atom<d_group_numneigh[c_atom_loctag+1];
                neigh_atom++){
            int n_local_tag = d_firstneigh_ptrs[neigh_atom];
            double r2,sigma_r,r;
            double temp_x,temp_y,temp_z;
            double neigh_x,neigh_y,neigh_z;
            double delt_x,delt_y,delt_z;
            // if (n_local_tag < 0 ) continue;
            // int n_glob_tag = h_tag[n_local_tag];
            neigh_x = d_x_flat(n_local_tag,0);
            neigh_y = d_x_flat(n_local_tag,1);
            neigh_z = d_x_flat(n_local_tag,2);
            delt_x = (neigh_x - c_x);
            delt_y = (neigh_y - c_y);
            delt_z = (neigh_z - c_z);
            r2 = delt_x*delt_x + delt_y*delt_y + delt_z*delt_z;
            if (r2 <= 0.0) continue;
            r = sqrt(r2);
            // The perpetual candidate list includes skin; exclude it from the CV.
            if (r >= cutoff_r) continue;
            sigma_r = sw_f(r);
            // if (!isfinite(r) || !isfinite(sigma_r)) {
            //     printf("[诊断-环境-r] c_atom_calctag=%lld n_local_tag=%d r=%e sigma_r=%e c_x=%e c_y=%e c_z=%e neigh_x=%e neigh_y=%e neigh_z=%e\n",
            //         (long long)c_atom_calctag, n_local_tag, r, sigma_r, c_x, c_y, c_z, neigh_x, neigh_y, neigh_z);
            // }
            if ((sigma_r < cut_sigma_eps)) continue;
            // sigma_r >= cut_sigma_eps
            // 将通过筛选的原子压缩到一起，方便后续的线程数计算
            d_calculated_numneigh[start_neigh + sum_of_numneigh] = n_local_tag;
            // 计算 sum_of_sigma_CNatoms
            sum_of_sigma_CNatoms += sigma_r;
            sum_of_numneigh ++;
            MASK_ATOMIC_SET_ACTIVE(&d_LQ_mask[n_local_tag]);
            MASK_ATOMIC_SET_VAR(&d_LQ_mask[n_local_tag],neighbor_mask);
            MASK_ATOMIC_INC_VAR(&d_LQ_mask[n_local_tag],neighbor_inc_val);
        }
        d_neigh_in_cutoff_r[c_atom_calctag] = sum_of_numneigh;
        d_neigh_in_switching[c_atom_calctag] = sum_of_sigma_CNatoms;
    }
    #undef sw_f
}

__global__ void get_environment_Steinhardt_LocalQ_promote_pure_K(
    int group_count,
    int atom_all,
    const int *d_LQ_mask,
    int *d_pure_J_write_offset,
    int *d_group_indices,
    int *d_calc_tag) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= atom_all) return;

    int mask = d_LQ_mask[idx];

    // 核心物理判定：是激活的近邻 J，但自己不是中心 I
    if (MASK_IS_ACTIVE(mask) && MASK_IS_J(mask) && !MASK_IS_I(mask)) {
        int my_rank = atomicAdd((unsigned int*)d_pure_J_write_offset, 1);
        d_group_indices[group_count + my_rank] = idx;
        d_calc_tag[idx] = group_count + my_rank;
    }
}


template <int L>
__global__ void steinhardt_Local_cv_get_qlm(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, double cutoff_r, double cutoff_eps,
    int *d_group_indices,
    int *d_neigh_in_cutoff_r,
    int *d_group_numneigh,
    int *d_calculated_firstneigh_ptrs,
    int *d_calculated_numneigh,
    MetaD_zqc::ArrayView<const double> d_x_flat,
    double *d_neigh_in_switching,
    double *d_stein_qlm, double *d_stein_Ylm) {
    // 此处应该计算i,j原子的qlm

    int c_atom_calctag = blockIdx.x * blockDim.x + threadIdx.x;
    #define sw_f(r) (MetaD_zqc::SwitchFunction::f(sw_params_rij, (r)))
    if (c_atom_calctag >= calc_count) return;

    int c_atom_loctag = d_group_indices[c_atom_calctag]; // 当前原子在local原子列表中的标签
    constexpr int lm_size = (L + 1) * 2;

    // 【与导数核函数完美镜像】的近邻数读取与基础寻址逻辑
    int neigh_num = d_neigh_in_cutoff_r[c_atom_calctag];
    int stein_qlm_base_id = c_atom_loctag * lm_size; // 为了通讯方便，qlm和Ylm都按照local原子标签来存储和访问
    int stein_Ylm_base_id;

    // 如果没有邻居，直接清零退出
    double NFb_i_check = d_neigh_in_switching[c_atom_calctag];
    if (neigh_num == 0 || NFb_i_check < 1e-12) return;
    // because we multiply a swfunction, so it need to devide by its weight sum
    double inv_neigh = 1.0 / (double)NFb_i_check;
    // if (!isfinite(inv_neigh) || d_neigh_in_switching[c_atom_calctag] <= 0.0) {
    //     printf("[诊断-qlm-inv_neigh] calctag=%lld neigh_num=%d neigh_switching=%e\n",
    //         (long long)c_atom_calctag, neigh_num, d_neigh_in_switching[c_atom_calctag]);
    // }

    // 在寄存器（栈）上初始化局部数组用于累加，避免频繁读写全局显存
    double local_qlm[lm_size] = {0.0};
    // d_stein_qlm[stein_qlm_base_id] = 0.0;
    // double *local_qlm = &d_stein_qlm[stein_qlm_base_id];
    double c_x = d_x_flat(c_atom_loctag,0);
    double c_y = d_x_flat(c_atom_loctag,1);
    double c_z = d_x_flat(c_atom_loctag,2);

    int neigh_base = d_group_numneigh[c_atom_loctag];
    int neigh_pair_base = d_calculated_firstneigh_ptrs[c_atom_calctag];

    for (int neigh_atom = 0; neigh_atom < neigh_num; neigh_atom++) {
        int n_local_tag = d_calculated_numneigh[neigh_base + neigh_atom];
        // Ylm 只与原子位置有关，所以可以按照group直接访问不需要扩大数组,所以用c_atom而不是c_atom_tag
        stein_Ylm_base_id = (neigh_pair_base+neigh_atom)*lm_size;
        double local_Ylm[lm_size] = {0.0};

        double neigh_x = d_x_flat(n_local_tag,0);
        double neigh_y = d_x_flat(n_local_tag,1);
        double neigh_z = d_x_flat(n_local_tag,2);
        double delt_x = (neigh_x - c_x);
        double delt_y = (neigh_y - c_y);
        double delt_z = (neigh_z - c_z);
        double r2 = delt_x*delt_x + delt_y*delt_y + delt_z*delt_z;
        double r = sqrt(r2);
        double r_weight = sw_f(r);

        // Evaluate the same unit-direction polynomials used by the Q path.
        const double inv_r = 1.0 / r;
        compute_Ylm_unit<L>(
            delt_x * inv_r, delt_y * inv_r, delt_z * inv_r, local_Ylm);
        #pragma unroll
        for (int m = 0; m < lm_size; ++m) {
            local_qlm[m] += r_weight * local_Ylm[m];
        }
        // CUDA 12.2/12.5 的设备端使用通用 memcpy；libcu++ 未提供 cuda::std::memcpy。
        ::memcpy(&d_stein_Ylm[stein_Ylm_base_id], &local_Ylm,
                            sizeof(double)*lm_size);
    }

    // // --- 循环外归一化与写回全局显存 ---

    #pragma unroll
    for (int i = 0; i < lm_size; i++) {
        local_qlm[i] *= inv_neigh;
        // d_stein_qlm[stein_qlm_base_id + i] = local_qlm[i];
    }
    ::memcpy(&d_stein_qlm[stein_qlm_base_id], &local_qlm,
                        sizeof(double)*lm_size);

    double ql_sq = local_qlm[0] * local_qlm[0];
    #undef sw_f
}
template __global__ void steinhardt_Local_cv_get_qlm<3>(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, double cutoff_r, double cutoff_eps,
    int *d_group_indices,
    int *d_neigh_in_cutoff_r,
    int *d_group_numneigh,
    int *d_calculated_firstneigh_ptrs,
    int *d_calculated_numneigh,
    MetaD_zqc::ArrayView<const double> d_x_flat,
    double *d_neigh_in_switching,
    double *d_stein_qlm, double *d_stein_Ylm);
template __global__ void steinhardt_Local_cv_get_qlm<4>(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, double cutoff_r, double cutoff_eps,
    int *d_group_indices,
    int *d_neigh_in_cutoff_r,
    int *d_group_numneigh,
    int *d_calculated_firstneigh_ptrs,
    int *d_calculated_numneigh,
    MetaD_zqc::ArrayView<const double> d_x_flat,
    double *d_neigh_in_switching,
    double *d_stein_qlm, double *d_stein_Ylm);
template __global__ void steinhardt_Local_cv_get_qlm<6>(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, double cutoff_r, double cutoff_eps,
    int *d_group_indices,
    int *d_neigh_in_cutoff_r,
    int *d_group_numneigh,
    int *d_calculated_firstneigh_ptrs,
    int *d_calculated_numneigh,
    MetaD_zqc::ArrayView<const double> d_x_flat,
    double *d_neigh_in_switching,
    double *d_stein_qlm, double *d_stein_Ylm);


template <int L>
__global__ void steinhardt_Local_cv_get_LQl(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, double cutoff_r, double cutoff_eps,
    int *d_group_indices,
    int *d_neigh_in_cutoff_r,
    int *d_group_numneigh,
    int *d_calculated_firstneigh_ptrs,
    int *d_calculated_numneigh,
    MetaD_zqc::ArrayView<const double> d_x_flat,
    double *d_neigh_in_switching,
    double *sum_of_qlm_value_weights,
    double *d_stein_qlm,
    double *d_stein_LQlm, double *d_stein_ql) {
    // 此处应该只计算i原子的LQl

    int c_atom_calctag = blockIdx.x * blockDim.x + threadIdx.x;
    #define sw_f(r) (MetaD_zqc::SwitchFunction::f(sw_params_rij, (r)))
    if (c_atom_calctag >= calc_count) return;

    int c_atom_loctag = d_group_indices[c_atom_calctag]; // 当前原子在local原子列表中的标签

    // 【与导数核函数完美镜像】的近邻数读取与基础寻址逻辑
    int neigh_num = d_neigh_in_cutoff_r[c_atom_calctag];
    int stein_qlm_base_id = c_atom_loctag * (L + 1) * 2; // 为了通讯方便，qlm和Ylm都按照local原子标签来存储和访问

    // 如果没有邻居，直接清零退出
    double NFb_i_check = d_neigh_in_switching[c_atom_calctag];
    if (neigh_num == 0 || NFb_i_check < 1e-12) return;
    // because we multiply a swfunction, so it need to devide by its weight sum
    double inv_neigh = 1.0 / (double)(NFb_i_check + 1.0);

    // 在寄存器（栈）上初始化局部数组用于累加，避免频繁读写全局显存
    constexpr int lm_size = (L + 1) * 2;
    double local_LQlm[lm_size] = {0.0};

    double c_x = d_x_flat(c_atom_loctag,0);
    double c_y = d_x_flat(c_atom_loctag,1);
    double c_z = d_x_flat(c_atom_loctag,2);

    #pragma unroll
    for (int i = 0; i < lm_size; i++) {
        local_LQlm[i] = d_stein_qlm[c_atom_loctag*lm_size + i];
    }
    double sum_of_qlm_value_weight_loc[lm_size] = {0.0};

    int neigh_base = d_group_numneigh[c_atom_loctag];
    int neigh_pair_base = d_calculated_firstneigh_ptrs[c_atom_calctag];

    for (int neigh_atom = 0; neigh_atom < neigh_num; neigh_atom++) {
        int n_local_tag = d_calculated_numneigh[neigh_base + neigh_atom];
        double neigh_x = d_x_flat(n_local_tag,0);
        double neigh_y = d_x_flat(n_local_tag,1);
        double neigh_z = d_x_flat(n_local_tag,2);
        double delt_x = (neigh_x - c_x);
        double delt_y = (neigh_y - c_y);
        double delt_z = (neigh_z - c_z);
        double r2 = delt_x*delt_x + delt_y*delt_y + delt_z*delt_z;
        double r = sqrt(r2);
        double r_weight = sw_f(r);

        #pragma unroll
        for (int i = 0; i < lm_size; i++) {
            double qlm_value_weight =  r_weight*d_stein_qlm[n_local_tag*lm_size + i];
            local_LQlm[i] +=qlm_value_weight;
            sum_of_qlm_value_weight_loc[i] +=qlm_value_weight;
        }
    }
    ::memcpy(&sum_of_qlm_value_weights[c_atom_calctag*lm_size], &sum_of_qlm_value_weight_loc, sizeof(double)*lm_size);

    // // --- 循环外归一化与写回全局显存 ---

    #pragma unroll
    for (int i = 0; i < lm_size; i++) {
        local_LQlm[i] *= inv_neigh;
        d_stein_LQlm[stein_qlm_base_id + i] = local_LQlm[i];
    }

    double QL_sq = local_LQlm[0] * local_LQlm[0];

    // 3. 对应你原代码的第三步：从 m = 1 开始累加已经归一化的各项平方和
    #pragma unroll
    for (int i = 1; i <= L; i++) {
        double re_part = local_LQlm[i * 2 + 0];
        double im_part = local_LQlm[i * 2 + 1];
        QL_sq += 2.0 * (re_part * re_part + im_part * im_part);
    }

    d_stein_ql[c_atom_loctag] = sqrt(QL_sq * 12.56637061435917295385/double(2*L + 1));

    #undef sw_f
}
template __global__ void steinhardt_Local_cv_get_LQl<3>(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, double cutoff_r, double cutoff_eps,
    int *d_group_indices,
    int *d_neigh_in_cutoff_r,
    int *d_group_numneigh,
    int *d_calculated_firstneigh_ptrs,
    int *d_calculated_numneigh,
    MetaD_zqc::ArrayView<const double> d_x_flat,
    double *d_neigh_in_switching,
    double *sum_of_qlm_value_weights,
    double *d_stein_qlm,
    double *d_stein_LQlm, double *d_stein_ql);
template __global__ void steinhardt_Local_cv_get_LQl<4>(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, double cutoff_r, double cutoff_eps,
    int *d_group_indices,
    int *d_neigh_in_cutoff_r,
    int *d_group_numneigh,
    int *d_calculated_firstneigh_ptrs,
    int *d_calculated_numneigh,
    MetaD_zqc::ArrayView<const double> d_x_flat,
    double *d_neigh_in_switching,
    double *sum_of_qlm_value_weights,
    double *d_stein_qlm,
    double *d_stein_LQlm, double *d_stein_ql);
template __global__ void steinhardt_Local_cv_get_LQl<6>(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, double cutoff_r, double cutoff_eps,
    int *d_group_indices,
    int *d_neigh_in_cutoff_r,
    int *d_group_numneigh,
    int *d_calculated_firstneigh_ptrs,
    int *d_calculated_numneigh,
    MetaD_zqc::ArrayView<const double> d_x_flat,
    double *d_neigh_in_switching,
    double *sum_of_qlm_value_weights,
    double *d_stein_qlm,
    double *d_stein_LQlm, double *d_stein_ql);


// The derivative with respect to R = x_neighbor - x_center contributes
// -R tensor dCV/dR to the CV virial. Split it equally between the endpoints.
// Keep the center half in registers until its neighbor loop has finished.
__device__ __forceinline__ void localq_tally_bond_virial(
    double dx, double dy, double dz, const double *gradient,
    int neighbor, double *center_virial, double *atom_virial) {
    const double half_virial[9] = {
        -0.5*dx*gradient[0], -0.5*dy*gradient[1], -0.5*dz*gradient[2],
        -0.5*dx*gradient[1], -0.5*dx*gradient[2], -0.5*dy*gradient[2],
        -0.5*dy*gradient[0], -0.5*dz*gradient[0], -0.5*dz*gradient[1]
    };
    #pragma unroll
    for (int component = 0; component < 9; ++component) {
        center_virial[component] += half_virial[component];
        atomicAdd(&atom_virial[9*neighbor + component], half_virial[component]);
    }
}

template <int L>
__global__ void steinhardt_Local_backward_ij_kernel(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    const double *center_adjoint,
    int group_count,
    MetaD_zqc::ArrayView<const double> d_x_flat, double *d_neigh_in_switching,
    int *d_neigh_in_cutoff_r,
    int *d_calculated_firstneigh_ptrs,
    int *d_group_indices,
    int *d_group_numneigh,
    int *d_calculated_numneigh,
    int *d_calc_tag,
    double *sum_of_qlm_value_weights,
    double *d_dcvdx_rjk_prefix,
    double *d_stein_qlm,
    double *d_stein_LQlm,
    double *d_stein_ql,
    double *d_dcvdx, double *d_a_virial) {
    // 仅遍历i原子
    // c_atom 是 calctag
    int c_atom_calctag = blockIdx.x * blockDim.x + threadIdx.x;
    #define sw_f_r(r) (MetaD_zqc::SwitchFunction::f(sw_params_rij, (r)))
    #define sw_df_r(r) (MetaD_zqc::SwitchFunction::df(sw_params_rij, (r)))
    if (c_atom_calctag >= group_count) return;

    constexpr int lm_size = (L + 1) * 2 ;

    int c_atom_loctag = d_group_indices[c_atom_calctag]; // 当前原子在local原子列表中的标签

    int neigh_num = d_neigh_in_cutoff_r[c_atom_calctag];
    int stein_qlm_base_id;
    int i_LQlm_base = c_atom_loctag * lm_size;

    // 如果没有邻居，直接清零退出
    double NFb_i_tilt = d_neigh_in_switching[c_atom_calctag];
    if (neigh_num == 0 || NFb_i_tilt < 1e-12) return;
    NFb_i_tilt = NFb_i_tilt + 1;
    double dcvdrij_fori[3] = {0.0, 0.0, 0.0};
    double center_virial[9] = {0.0};

    double c_x = d_x_flat(c_atom_loctag,0);
    double c_y = d_x_flat(c_atom_loctag,1);
    double c_z = d_x_flat(c_atom_loctag,2);

    // 与steinhardt no local 共用一个结果数组所以是d_stein_ql
    double LQ_i = d_stein_ql[c_atom_loctag];
    LQ_i = fmax(LQ_i, 1e-12);
    // Graph 提供各输出累加后的 dV/dLQ_i，再接球谐模长的几何链。
    const double F_LQ_i_with_NF = center_adjoint[c_atom_calctag] * (4*PI)/(2*L+1)/LQ_i;

    double inv_NFb = 1.0/(NFb_i_tilt);

    double accum_re = 0.0;
    double accum_im = 0.0;



    // 邻居的分发
    int neigh_base = d_group_numneigh[c_atom_loctag];
    int neigh_pair_base = d_calculated_firstneigh_ptrs[c_atom_calctag];

    for (int neigh_atom = 0; neigh_atom < neigh_num; neigh_atom++) {
        int n_local_tag = d_calculated_numneigh[neigh_base + neigh_atom];
        double neigh_x = d_x_flat(n_local_tag,0);
        double neigh_y = d_x_flat(n_local_tag,1);
        double neigh_z = d_x_flat(n_local_tag,2);
        double delt_x = (neigh_x - c_x);
        double delt_y = (neigh_y - c_y);
        double delt_z = (neigh_z - c_z);
        double r2 = delt_x*delt_x + delt_y*delt_y + delt_z*delt_z;
        double r = sqrt(r2);
        double r_weight = sw_f_r(r);

        // 接下来求m=(-l,l)
        // 长度为lm_size,同时存有m>=0的所有信息。
        stein_qlm_base_id = n_local_tag*lm_size;
        int j_calc_tag = d_calc_tag[n_local_tag];
        if (j_calc_tag < 0) continue; // 未晋升的邻居（不应出现）；跳过避免负下标写爆显存
        double rjk_prefix_in_i = (inv_NFb)*F_LQ_i_with_NF;
        // rjk_prefix_in_i = rjk_prefix_in_i*inv_NFb*POW2(1.0/d_neigh_in_switching[j_calc_tag])*(NFb_i_tilt*r_weight);
        // #pragma unroll
        // for(int m=0;m<=lm_size;m++){
        //     atomicAdd(&d_dcvdx_rjk_prefix[j_calc_tag*lm_size + m], rjk_prefix_in_i*d_stein_LQlm[i_LQlm_base + m]);
        // }
        // *d_stein_qlm[n_local_tag]
        double dcvdrij_forj[3] = {0.0, 0.0, 0.0}; // 栈上数组初始化
        // m=[-l,+l], RE(LQlm)*d(RE(LQlm))/dr + IM(LQlm)*d(IM(LQlm))/dr
        // Re(LQlm)^2 = Re(LQl-m)^2
        double prefix = rjk_prefix_in_i*(inv_NFb)*sw_df_r(r)*(1/r);
        double prefix2 = prefix*(d_stein_LQlm[i_LQlm_base + 0]*(NFb_i_tilt*d_stein_qlm[stein_qlm_base_id + 0] - d_stein_qlm[c_atom_loctag*lm_size + 0] - sum_of_qlm_value_weights[c_atom_calctag*lm_size + 0])
                        + d_stein_LQlm[i_LQlm_base + 1]*(NFb_i_tilt*d_stein_qlm[stein_qlm_base_id + 1] - d_stein_qlm[c_atom_loctag*lm_size + 1] - sum_of_qlm_value_weights[c_atom_calctag*lm_size + 1]));
        dcvdrij_forj[0] += delt_x*prefix2;
        dcvdrij_forj[1] += delt_y*prefix2;
        dcvdrij_forj[2] += delt_z*prefix2;
        atomicAdd(&d_dcvdx_rjk_prefix[j_calc_tag*lm_size + 0], rjk_prefix_in_i*r_weight*d_stein_LQlm[i_LQlm_base + 0]);
        atomicAdd(&d_dcvdx_rjk_prefix[j_calc_tag*lm_size + 1], rjk_prefix_in_i*r_weight*d_stein_LQlm[i_LQlm_base + 1]);
        #pragma unroll
        for(int m=1;m<=L;m++){
            double prefix2 = 2.0*prefix*(d_stein_LQlm[i_LQlm_base +2*m + 0]*(NFb_i_tilt*d_stein_qlm[stein_qlm_base_id +2*m + 0] - d_stein_qlm[c_atom_loctag*lm_size +2*m + 0] - sum_of_qlm_value_weights[c_atom_calctag*lm_size +2*m + 0])
                            + d_stein_LQlm[i_LQlm_base +2*m + 1]*(NFb_i_tilt*d_stein_qlm[stein_qlm_base_id +2*m + 1] - d_stein_qlm[c_atom_loctag*lm_size +2*m + 1] - sum_of_qlm_value_weights[c_atom_calctag*lm_size +2*m + 1]));
            dcvdrij_forj[0] += delt_x*prefix2;
            dcvdrij_forj[1] += delt_y*prefix2;
            dcvdrij_forj[2] += delt_z*prefix2;
            atomicAdd(&d_dcvdx_rjk_prefix[j_calc_tag*lm_size +2*m + 0], rjk_prefix_in_i*r_weight*d_stein_LQlm[i_LQlm_base + 2*m + 0]);
            atomicAdd(&d_dcvdx_rjk_prefix[j_calc_tag*lm_size +2*m + 1], rjk_prefix_in_i*r_weight*d_stein_LQlm[i_LQlm_base + 2*m + 1]);
        }
        localq_tally_bond_virial(
            delt_x, delt_y, delt_z, dcvdrij_forj,
            n_local_tag, center_virial, d_a_virial);
        // 对ij原子的贡献
        dcvdrij_fori[0] -= dcvdrij_forj[0];
        dcvdrij_fori[1] -= dcvdrij_forj[1];
        dcvdrij_fori[2] -= dcvdrij_forj[2];
        atomicAdd(&d_dcvdx[n_local_tag * 3 + 0], +dcvdrij_forj[0]);
        atomicAdd(&d_dcvdx[n_local_tag * 3 + 1], +dcvdrij_forj[1]);
        atomicAdd(&d_dcvdx[n_local_tag * 3 + 2], +dcvdrij_forj[2]);
        // if (!isfinite(LQ_i) || LQ_i < 1e-6) {
        //     printf("[诊断-LQ_i] calctag=%lld LQ_i=%e\n", (long long)c_atom_calctag, LQ_i);
        // }
        // if (!isfinite(F_LQ_i_with_NF)) {
        //     printf("[诊断-F] calctag=%lld LQ_i=%e F=%e cv_value=%e global_sw_sum=%e\n",
        //         (long long)c_atom_calctag, LQ_i, F_LQ_i_with_NF, cv_value, global_sw_sum);
        // }
        // if (r < 1e-6) {
        //     printf("[诊断-r] calctag=%lld n_local_tag=%lld r=%e (原子几乎重合!)\n",
        //         (long long)c_atom_calctag, (long long)n_local_tag, r);
        // }
        // if (!isfinite(prefix2)) {
        //     printf("[诊断-prefix2] calctag=%lld n_local_tag=%lld prefix2=%e\n",
        //         (long long)c_atom_calctag, (long long)n_local_tag, prefix2);
        // }
        // if (r < 1e-12) {
        //     printf("[诊断-jk-r0] calctag=%lld loctag=%lld n_local_tag=%lld  atom->tag(loctag)=%d atom->tag(n_local_tag)=%d\n",
        //         (long long)c_atom_calctag, (long long)c_atom_loctag, (long long)n_local_tag,
        //         /* 这里需要把 atom->tag 数组也传进kernel或者用一个等效的全局tag数组 */
        //         0, 0);  // 占位，见下面说明
        // }
    }
    // 自己也需要分发一份
    {
        int n_local_tag = c_atom_loctag;
        double r_weight = 1;

        // 接下来求m=(-l,l)
        // 长度为lm_size,同时存有m>=0的所有信息。
        stein_qlm_base_id = n_local_tag*lm_size;
        int j_calc_tag = d_calc_tag[n_local_tag];
        double rjk_prefix_in_i = (inv_NFb)*F_LQ_i_with_NF;
        atomicAdd(&d_dcvdx_rjk_prefix[j_calc_tag*lm_size + 0], rjk_prefix_in_i*r_weight*d_stein_LQlm[i_LQlm_base + 0]);
        atomicAdd(&d_dcvdx_rjk_prefix[j_calc_tag*lm_size + 1], rjk_prefix_in_i*r_weight*d_stein_LQlm[i_LQlm_base + 1]);
        #pragma unroll
        for(int m=1;m<=L;m++){
            atomicAdd(&d_dcvdx_rjk_prefix[j_calc_tag*lm_size +2*m + 0], rjk_prefix_in_i*r_weight*d_stein_LQlm[i_LQlm_base + 2*m + 0]);
            atomicAdd(&d_dcvdx_rjk_prefix[j_calc_tag*lm_size +2*m + 1], rjk_prefix_in_i*r_weight*d_stein_LQlm[i_LQlm_base + 2*m + 1]);
        }
        // 对ij原子的贡献
    }
    atomicAdd(&d_dcvdx[c_atom_loctag * 3 + 0], dcvdrij_fori[0]);
    atomicAdd(&d_dcvdx[c_atom_loctag * 3 + 1], dcvdrij_fori[1]);
    atomicAdd(&d_dcvdx[c_atom_loctag * 3 + 2], dcvdrij_fori[2]);
    #pragma unroll
    for (int component = 0; component < 9; ++component) {
        atomicAdd(&d_a_virial[9*c_atom_loctag + component], center_virial[component]);
    }

    #undef sw_f_r
    #undef sw_df_r
}
template __global__ void steinhardt_Local_backward_ij_kernel<3>(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    const double *center_adjoint,
    int group_count,
    MetaD_zqc::ArrayView<const double> d_x_flat, double *d_neigh_in_switching,
    int *d_neigh_in_cutoff_r,
    int *d_calculated_firstneigh_ptrs,
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
template __global__ void steinhardt_Local_backward_ij_kernel<4>(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    const double *center_adjoint,
    int group_count,
    MetaD_zqc::ArrayView<const double> d_x_flat, double *d_neigh_in_switching,
    int *d_neigh_in_cutoff_r,
    int *d_calculated_firstneigh_ptrs,
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
template __global__ void steinhardt_Local_backward_ij_kernel<6>(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    const double *center_adjoint,
    int group_count,
    MetaD_zqc::ArrayView<const double> d_x_flat, double *d_neigh_in_switching,
    int *d_neigh_in_cutoff_r,
    int *d_calculated_firstneigh_ptrs,
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
    int *d_calculated_firstneigh_ptrs,
    int *d_calculated_numneigh,
    int *d_calc_tag,
    int *d_neigh_in_cutoff_r,
    double *d_dcvdx_rjk_prefix,
    double *d_stein_qlm, double *d_stein_Ylm, double *d_stein_ql,
    double *d_dcvdx, double *d_a_virial) {

    int c_atom_calctag = blockIdx.x * blockDim.x + threadIdx.x;
    #define sw_f_r(r) (MetaD_zqc::SwitchFunction::f(sw_params_rij, (r)))
    #define sw_df_r(r) (MetaD_zqc::SwitchFunction::df(sw_params_rij, (r)))
    if (c_atom_calctag >= calc_count) return;

    int c_atom_loctag = d_group_indices[c_atom_calctag]; // 当前原子在local原子列表中的标签

    int neigh_num = d_neigh_in_cutoff_r[c_atom_calctag];
    constexpr int lm_size = (L + 1) * 2 ;
    int stein_qlm_base_id = c_atom_loctag * lm_size;

    // 如果没有邻居，直接清零退出
    double NFb_j_check = d_neigh_in_switching[d_calc_tag[c_atom_loctag]];
    if (neigh_num == 0 || NFb_j_check < 1e-12) return;

    // constexpr int lmdr_size = lm_size *3;
    double dcvdrjk_forj[3] = {0.0, 0.0, 0.0}; // 栈上数组初始化
    double center_virial[9] = {0.0};
    double prefix_j[lm_size] = {0.0};
    #pragma unroll
    for(int m=0; m<lm_size; m++){
        prefix_j[m] = d_dcvdx_rjk_prefix[c_atom_calctag*lm_size + m]/NFb_j_check;
    }
    double *qlm = &d_stein_qlm[stein_qlm_base_id];
    // 紧跟在 prefix_j 那个循环之后
    if (NFb_j_check < 1e-12) {
        printf("[诊断-jk-NFb_j] calctag=%lld NFb_j=%e neigh_num=%d\n",
            (long long)c_atom_calctag, NFb_j_check, neigh_num);
    }
    // for (int m = 0; m < lm_size; m++) {
    //     if (!isfinite(prefix_j[m])) {
    //         printf("[诊断-jk-prefix_j] calctag=%lld m=%d prefix_j=%e prefix_raw=%e\n",
    //             (long long)c_atom_calctag, m, prefix_j[m],
    //             d_dcvdx_rjk_prefix[c_atom_calctag*lm_size+m]);
    //     }
    // }

    double c_x = d_x_flat(c_atom_loctag,0);
    double c_y = d_x_flat(c_atom_loctag,1);
    double c_z = d_x_flat(c_atom_loctag,2);


    int neigh_base = d_group_numneigh[c_atom_loctag];
    int neigh_pair_base = d_calculated_firstneigh_ptrs[c_atom_calctag];

    for (int neigh_atom = 0; neigh_atom < neigh_num; neigh_atom++) {
        int n_local_tag = d_calculated_numneigh[neigh_base + neigh_atom];
        double neigh_x = d_x_flat(n_local_tag,0);
        double neigh_y = d_x_flat(n_local_tag,1);
        double neigh_z = d_x_flat(n_local_tag,2);
        double delt_x = (neigh_x - c_x);
        double delt_y = (neigh_y - c_y);
        double delt_z = (neigh_z - c_z);
        double r2 = delt_x*delt_x + delt_y*delt_y + delt_z*delt_z;
        double r = sqrt(r2);
        double r_weight = sw_f_r(r);

        double sum_of_sw_weight;
        sum_of_sw_weight = d_neigh_in_switching[c_atom_calctag];

        // Contract the analytic Cartesian Ylm gradient directly with prefix_j.
        // prefix_j already includes the source weights and qlm normalization.
        const double inv_r = 1.0 / r;
        double Ylm[lm_size];
        double angular_gradient[6];
        compute_Ylm_unit_and_gradient<L>(
            delt_x * inv_r, delt_y * inv_r, delt_z * inv_r, inv_r,
            r_weight, 0.0, 0, 0, prefix_j, Ylm, angular_gradient);
        double dcvdrjk_fork[3] = {0.0, 0.0, 0.0};
        double prefix1 = sw_df_r(r)/r;
        double sum_sigprefix =prefix1*(prefix_j[0]*(Ylm[0]-qlm[0])+ prefix_j[1]*(Ylm[1]-qlm[1]));
        double sum_dY_prefix[3];
        #pragma unroll
        for (int direct = 0; direct < 3; ++direct) {
            sum_dY_prefix[direct] = angular_gradient[2*direct] +
                                   angular_gradient[2*direct + 1];
        }
        #pragma unroll
        for(int m=1;m<=L;m++){
            sum_sigprefix += prefix1*2.0*(prefix_j[2*m + 0]*(Ylm[2*m + 0]-qlm[2*m + 0])
                                        + prefix_j[2*m + 1]*(Ylm[2*m + 1]-qlm[2*m + 1]));

        }

        dcvdrjk_fork[0] = delt_x*sum_sigprefix + sum_dY_prefix[0];
        dcvdrjk_fork[1] = delt_y*sum_sigprefix + sum_dY_prefix[1];
        dcvdrjk_fork[2] = delt_z*sum_sigprefix + sum_dY_prefix[2];
        localq_tally_bond_virial(
            delt_x, delt_y, delt_z, dcvdrjk_fork,
            n_local_tag, center_virial, d_a_virial);
        #pragma unroll
        for(int direct=0; direct<3; direct++){
            dcvdrjk_forj[direct] -= dcvdrjk_fork[direct];
            atomicAdd(&d_dcvdx[n_local_tag * 3 + direct], dcvdrjk_fork[direct]);
        }
        // if (r < 1e-6) {
        //     printf("[诊断-r] calctag=%lld n_local_tag=%lld r=%e (原子几乎重合!)\n",
        //         (long long)c_atom_loctag, (long long)n_local_tag, r);
        // }
        // if (r < 1e-12) {
        //     printf("[诊断-jk-r0] calctag=%lld loctag=%lld n_local_tag=%lld  atom->tag(loctag)=%d atom->tag(n_local_tag)=%d\n",
        //         (long long)c_atom_loctag, (long long)c_atom_loctag, (long long)n_local_tag,
        //         /* 这里需要把 atom->tag 数组也传进kernel或者用一个等效的全局tag数组 */
        //         0, 0);  // 占位，见下面说明
        // }
        // if (!isfinite(sum_sigprefix)) {
        //     printf("[诊断-jk-sigprefix] calctag=%lld n_local_tag=%lld sum_sigprefix=%e r=%e prefix1=%e\n",
        //         (long long)c_atom_loctag, (long long)n_local_tag, sum_sigprefix, r, prefix1);
        // }
        // for (int d=0; d<3; d++) {
        //     if (!isfinite(sum_dY_prefix[d])) {
        //         printf("[诊断-jk-dYprefix] calctag=%lld n_local_tag=%lld dir=%d sum_dY_prefix=%e\n",
        //             (long long)c_atom_loctag, (long long)n_local_tag, d, sum_dY_prefix[d]);
        //     }
        // }
    }
    #pragma unroll
    for(int direct=0; direct<3; direct++){
    atomicAdd(&d_dcvdx[c_atom_loctag * 3 + direct], dcvdrjk_forj[direct]);
    }
    #pragma unroll
    for (int component = 0; component < 9; ++component) {
        atomicAdd(&d_a_virial[9*c_atom_loctag + component], center_virial[component]);
    }

    #undef sw_f_r
    #undef sw_df_r
}

template class MetaD_zqc::STEIN_LocalQL<3>;
template class MetaD_zqc::STEIN_LocalQL<4>;
template class MetaD_zqc::STEIN_LocalQL<6>;
