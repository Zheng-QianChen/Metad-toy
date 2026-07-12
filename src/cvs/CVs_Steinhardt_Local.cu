

#include <cuda_runtime.h>
#include <cstring>
#include <algorithm>
#include <cub/cub.cuh>

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

#include "fix_crystallize.h"

#include "zqc_debug.h"
#include "CV_Steinhardt.h"
#include "CV_Steinhardt_math.h"

using namespace LAMMPS_NS;

template <int L>
MetaD_zqc::STEIN_LocalQL<L>::STEIN_LocalQL(LAMMPS_NS::LAMMPS *lmp, 
                             LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check, 
                             std::string env_setNum, int group_id, 
                             MetaD_zqc::Steinhardt_env* my_env, int d_block_size, 
                             MetaD_zqc::SteinhardtRequest req)
                        : STEIN_QL<L>(lmp, Fixmetad, f_check, env_setNum, group_id, 
                            L, my_env, d_block_size){
    my_loc_env = static_cast<STEIN_LocalQL_env*>(my_env);
    this->my_cv_SWfunc = req.SW_FUNC_cv;
    register_buffer(d_stein_LQlm,"d_stein_LQlm");
    register_buffer(d_dcvdx_rjk_prefix,"d_dcvdx_rjk_prefix");
    register_buffer(sum_of_qlm_value_weights,"sum_of_qlm_value_weights");
}

template <int L>
MetaD_zqc::STEIN_LocalQL<L>::~STEIN_LocalQL(){
}

MetaD_zqc::STEIN_LocalQL_env::STEIN_LocalQL_env(LAMMPS_NS::LAMMPS *lmp, 
             LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check, 
             MetaD_zqc::SteinhardtRequest req)
        :Steinhardt_env(lmp, Fixmetad, f_check, 
                        req.group_id, req.cutoff_r, 12),
        cutoff_eps_r(req.cutoff_eps){
    

    this->my_r_SWfunc = req.SW_FUNC_r;
    auto atom = lmp->atom;
    

    lmp->memory->grow(h_LQ_mask, ((lmp->atom)->nmax), "STEIN_LocalQL:h_LQ_mask");
    lmp->memory->grow(h_calc_tag, ((lmp->atom)->nmax), "STEIN_LocalQL:h_calc_tag");
    lmp->memory->grow(h_neigh_in_switching, ((lmp->atom)->nmax), "STEIN_LocalQL:h_neigh_in_switching");
    lmp->memory->grow(h_calculated_firstneigh_ptrs, ((lmp->atom)->nmax), "STEIN_LocalQL:h_calculated_firstneigh_ptrs");

    std::memset(h_LQ_mask, 0, (lmp->atom)->nmax * sizeof(int));
    std::memset(h_calc_tag, -1, (lmp->atom)->nmax * sizeof(LAMMPS_NS::tagint));
    std::memset(h_neigh_in_switching, 0, (lmp->atom)->nmax * sizeof(double));
    std::memset(h_calculated_firstneigh_ptrs, 0, (lmp->atom)->nmax * sizeof(LAMMPS_NS::tagint));

    register_buffer(d_pure_J_write_offset,"d_pure_J_write_offset");
    register_buffer(d_calculated_firstneigh_ptrs,"d_calculated_firstneigh_ptrs");
    register_buffer(d_LQ_mask,"d_LQ_mask");
    register_buffer(d_calc_tag,"d_calc_tag");
    register_buffer(d_neigh_in_switching,"d_neigh_in_switching");
    // register_buffer(d_is_pure_J,"d_is_pure_J");
    

    d_pure_J_write_offset.grow_to(1, __FILE__, __LINE__);
    d_calculated_firstneigh_ptrs.grow_to(atom->nmax, __FILE__, __LINE__);
    d_LQ_mask.grow_to(atom->nmax, __FILE__, __LINE__);
    d_calc_tag.grow_to(atom->nmax, __FILE__, __LINE__);
    d_neigh_in_switching.grow_to(atom->nmax, __FILE__, __LINE__);
}

MetaD_zqc::STEIN_LocalQL_env::~STEIN_LocalQL_env(){
    lmp->memory->destroy(h_LQ_mask);
    lmp->memory->destroy(h_calc_tag);
    lmp->memory->destroy(h_neigh_in_switching);
    lmp->memory->destroy(h_calculated_firstneigh_ptrs);
}

void MetaD_zqc::STEIN_LocalQL_env::refresh_lmpbox(){
    // clear the h_group_indices
    atom = lmp->atom;
    mask = (atom)->mask;     // 原子组掩码
    // int Threads_own_atoms = atom->nlocal + atom->nghost;
    int Threads_own_atoms = atom->nmax;

    lmp->memory->grow(h_group_indices, (Threads_own_atoms), "STEIN_LocalQL_env:h_group_indices");
    lmp->memory->grow(h_calc_tag, (Threads_own_atoms), "STEIN_LocalQL_env:h_calc_tag");
    std::memset(h_group_indices, 0, Threads_own_atoms * sizeof(LAMMPS_NS::tagint));
    std::memset(h_calc_tag, -1, Threads_own_atoms * sizeof(LAMMPS_NS::tagint));

    // group_count = how many aim atoms in local
    last_group_count = group_count;
    group_count = 0; // 当前local中有
    for (int i = 0; i < (atom)->nlocal; i++) {
        if ((mask)[i] & (groupbit)){
            h_group_indices[(group_count)] = i; // record local index
            h_calc_tag[i] = group_count;
            group_count++;
            DEBUG_LOG("group_count=%lld",((long long)group_count));
        }
    }
    d_calc_tag.grow_to((Threads_own_atoms), __FILE__, __LINE__);
    d_calc_tag.upload_from(h_calc_tag, (Threads_own_atoms));

    d_mask.grow_to((Threads_own_atoms), __FILE__, __LINE__);
    d_mask.upload_from(mask, (Threads_own_atoms));
    // SAFE_CUDA_MEMCPY((d_mask.ptr),(mask),((Threads_own_atoms))*sizeof(int),cudaMemcpyHostToDevice,f_check);

    // set up nvidia thread number
    block_num = ((group_count) + d_block_size - 1)/d_block_size;
    N = d_block_size*block_num;
    // LOG_COND(((group_count)<(cutoff_Natoms)),"Warning: group_count(%lld) < cutoff_Natoms(%lld), please check your system !",(long long)group_count, (long long)cutoff_Natoms);
    LOG_COND((((box_x)<2*(cutoff_r))||((box_y)<2*(cutoff_r))||((box_z)<2*(cutoff_r))),"Warning: box < cutoff_r, please check your system !");
}

// 从FIX中直接进base_calc()函数，而base_calc()函数就是compute_Q_peratoms()
// 这是第一步
template <int L>
void MetaD_zqc::STEIN_LocalQL<L>::compute_Q_peratoms(){
    // =======接受邻居更新消息,进行与设备端通信===========
    if ((lmp->update->ntimestep > lmp->neighbor->lastcall)&&(lmp->update->ntimestep != 1)&&(this->init_flag)){
        DEBUG_LOG("rebuilds = %lld", (long long)lmp->neighbor->lastcall);
        DEBUG_LOG("now = %lld", (long long)lmp->update->ntimestep);
        ERR_COND(((my_loc_env->h_group_indices) == nullptr),"h_group_indices is nullptr.");
        DEBUG_LOG("h_group_indices=%p",(my_loc_env->h_group_indices));
    } else {
        // ===重建邻居列表后重新查找local中的目标原子=======
        if (lmp->update->ntimestep > my_loc_env->last_update_step){
            my_loc_env->refresh_lmpbox();
        }
        DEBUG_LOG("refresh_lmpbox done, group_count=%d",my_loc_env->group_count);
        block_num = my_loc_env->block_num;
        N = my_loc_env->N;
        int Threads_own_atoms = lmp->atom->nlocal + lmp->atom->nghost;
        // int Threads_own_atoms = lmp->atom->nmax;
        // LOG("Threads_own_atoms=%d",Threads_own_atoms);
        // stein_q for all aim atoms
        // LOG("=====================================================================");
        lmp->memory->grow(stein_q, Threads_own_atoms, "metad:STEIN_locQL:cv_bound");
        DEBUG_LOG("d_block_size is %d, block_num is %d",d_block_size, block_num);
    }
    DEBUG_LOG("group_count=%lld",(long long)my_loc_env->group_count);

    // 对于LOCAL原子，需要更多的信息。

    // 2. calculate atoms' environment
    DEBUG_LOG("environment function in, env_setNum is %s",env_setNum.c_str());
    environment();
    DEBUG_LOG("environment function out");

    // 3. calculate atoms' other things
    // steinhardt_param(Q_hybrid);
    steinhardt_param_calc(stein_q);
    
    // 输出group中每个原子的ql值
    DEBUG_RUN(for(int c_atom=0;c_atom<my_loc_env->group_count;c_atom++)
                {
                    DEBUG_LOG("stein_ql[%lld] = %f",(long long)c_atom,stein_q[c_atom]);
                });
    DEBUG_LOG("post_force function end");
}

// template <int L>
// void MetaD_zqc::STEIN_QL<L>::environment(){
//     DEBUG_LOG("last_update_step is %lld in %d, group_count=%d", (long long)my_loc_env->last_update_step, L, my_loc_env->group_count);
//     if (lmp->update->ntimestep > my_loc_env->last_update_step){
//         my_loc_env->get_env();
//     }
//     // DEBUG_LOG("environment function in, env_setNum is %s, get_env done",env_setNum);
//     DEBUG_LOG("last_update_step is %lld in %d, group_count=%d", (long long)my_loc_env->last_update_step, L, my_loc_env->group_count);
// }

void MetaD_zqc::STEIN_LocalQL_env::get_env(){
    // DEBUG_LOG("im in get_env, current step is %lld, last_update_step is %lld", (long long)lmp->update->ntimestep, (long long)this->last_update_step);
    // if (lmp->update->ntimestep == this->last_update_step){
    //     return;
    // }
    size_t datalen = 0;
    atom = lmp->atom;
    LAMMPS_NS::tagint atom_all = atom->nlocal + atom->nghost;
    // LAMMPS_NS::tagint all_neigh_pairs = h_group_numneigh[atom_all+1];
    // =======更新一下邻居列表位置防止报错=========
    nlist = Fixmetad->listfull;
    ERR_COND((nlist == nullptr),"STEIN_QL CV failed to find neighbor list now.");
    // DEBUG_LOG("rebuilds = %d", lmp->neighbor->lastcall);
    // DEBUG_LOG("now = %d", lmp->update->ntimestep);
    // DEBUG_LOG("rebuilds_fir = %p", nlist->firstneigh);
    // DEBUG_LOG("rebuilds_num = %p", nlist->numneigh);
    // =======防止lammps运行过程体积更改==========
    ERR_COND((lmp->domain == NULL),"domain list not initialized");
    box_x = (pbc_x) ? lmp->domain->xprd : INFINITY;
    box_y = (pbc_y) ? lmp->domain->yprd : INFINITY;
    box_z = (pbc_z) ? lmp->domain->zprd : INFINITY;

    // utilize different environment set
    // such as neighbor list, atom position, box size, to get the local structure information for each atom in the group
    if ((lmp->update->ntimestep > lmp->neighbor->lastcall)&&(lmp->update->ntimestep != 1)&&((numneigh != nullptr))&&(init_flag)){
        DEBUG_LOG("we skip rebuild in environment when %lld.", (long long)lmp->neighbor->lastcall);
    } else {
        DEBUG_LOG("cutoff_Natoms is %d",cutoff_Natoms);
        DEBUG_LOG("cutoff_r is %f",cutoff_r);
        DEBUG_LOG("group_count is %d",group_count);
        // =========================================================================
        // neighbour list and its copy to devise
        // h_LQ_mask / d_LQ_mask: mask with local's index
        // // ---- 1. 状态标志位的掩码 (Flags) ----
        // #define CALC_MASK_ACTIVE (1U << 24)  // 0x01000000
        // #define CALC_MASK_IS_I   (1U << 25)  // 0x02000000
        // #define CALC_MASK_IS_J   (1U << 26)  // 0x04000000
        // #define CALC_MASK_IS_K   (1U << 27)  // 0x08000000
        // // ---- 2. 计数器的位移与掩码 (Counters) ----
        // #define CALC_SHIFT_I     16
        // #define CALC_SHIFT_J     8
        // #define CALC_SHIFT_K     0
        // #define CALC_COUNTER_MASK 0xFFU      // 八位最大值 255
        // =========================================================================
        lmp->memory->grow(h_LQ_mask, atom_all, "STEIN_LocalQL:h_LQ_mask");
        std::memset(h_LQ_mask, 0, atom_all * sizeof(int));
        d_LQ_mask.grow_to(atom_all, __FILE__, __LINE__);
        d_LQ_mask.clear_async();
        // =========================================================================
        // neighbour list and its copy to devise
        // h_group_indices / d_group_indices: where the group atoms in locals' tag
        // =========================================================================
        // DEBUG_LOG("lastcall = %d", lmp->neighbor->lastcall);
        // int *d_group_indices;
        // SAFE_CUDA_FREE(d_group_indices);
        // SAFE_CUDA_MALLOC(&d_group_indices, (atom_all)*sizeof(int), f_check);
        d_group_indices.grow_to(atom_all, __FILE__, __LINE__);
        d_group_indices.upload_from(h_group_indices, atom_all);
        // SAFE_CUDA_MEMCPY(d_group_indices.ptr,h_group_indices,(atom_all)*sizeof(LAMMPS_NS::tagint),cudaMemcpyHostToDevice,f_check);
        // alloc
        DEBUG_LOG_COND((d_group_indices.ptr == NULL),"d_group_indices list not initialized");
        DEBUG_LOG("h_group_indices list %d" ,h_group_indices[0]);
        // =========================================================================
        // h_group_numneigh / d_group_numneigh :
        //      flatten index of the neighbour list. such as we have 20 neighbour
        //      for atom 1, then the list will be : [0, 20, ...]
        // =========================================================================
        // int *numneigh = nlist->numneigh;
        // int **firstneigh = nlist->firstneigh;
        numneigh = nlist->numneigh;
        firstneigh = nlist->firstneigh;
        DEBUG_LOG_COND((numneigh == NULL),"numneigh list not initialized");
        DEBUG_LOG_COND((firstneigh == NULL),"firstneigh list not initialized");
        // 2. creating number array of start num in different c_atom's neighbor
        // LAMMPS_NS::tagint *h_group_numneigh = new LAMMPS_NS::tagint[atom_all + 1];
        // LAMMPS_NS::tagint *d_group_numneigh;
        datalen = atom_all + 1;
        lmp->memory->grow(h_group_numneigh, datalen, "STEIN_QL:h_group_numneigh");
        d_group_numneigh.grow_to(datalen, __FILE__, __LINE__);
        DEBUG_LOG_COND((h_group_numneigh == NULL),"h_group_numneigh list not initialized");
        // 3. 逐原子拷贝邻居列表数据到GPU,现在更改为将所有原子的邻居拷贝到显存
        DEBUG_LOG("group_count=%d" ,group_count);
        h_group_numneigh[0] = 0;
        for (int gr_i = 0; gr_i < atom_all; gr_i++) {
            int i = gr_i; 
            int jnum = numneigh[i];
            h_group_numneigh[gr_i+1] = h_group_numneigh[gr_i] + jnum;
            DEBUG_LOG("gr_i=%d, tag=%d, jnum=%d, sum=%d", gr_i, i,jnum,h_group_numneigh[gr_i+1]);
        }
        d_group_numneigh.upload_from(h_group_numneigh, (atom_all + 1));
        // SAFE_CUDA_MEMCPY(d_group_numneigh.ptr,h_group_numneigh,(atom_all + 1)*sizeof(LAMMPS_NS::tagint),cudaMemcpyHostToDevice,f_check);
        LAMMPS_NS::tagint all_neigh_pairs = h_group_numneigh[atom_all];
        // =========================================================================
        // h_group_numneigh / d_group_numneigh :
        //      flatten index of the neighbour list. such as we have 20 neighbour
        //      for atom 1, then the list will be : [0, 20, ...]
        // h_firstneigh_ptrs / d_firstneigh_ptrs :
        //      flatten neighbour list
        // =========================================================================
        /* delete[] h_firstneigh_ptrs;
        h_firstneigh_ptrs = nullptr; */
        // int *h_firstneigh_ptrs = new int [h_group_numneigh[atom_all]];
        // int *d_firstneigh_ptrs; // 设备端二级指针
        // h_firstneigh_ptrs = new int [h_group_numneigh[atom_all]];
        lmp->memory->grow(h_firstneigh_ptrs, all_neigh_pairs, "STEIN_QL:h_firstneigh_ptrs");
        LAMMPS_NS::tagint ba_i;
        LAMMPS_NS::tagint nnumber;
        int i;
        // SAFE_CUDA_FREE(d_firstneigh_ptrs);
        // SAFE_CUDA_MALLOC(&d_firstneigh_ptrs, (all_neigh_pairs) * sizeof(int),f_check); // 分配设备端指针数组
        d_firstneigh_ptrs.grow_to(all_neigh_pairs, __FILE__, __LINE__);
        DEBUG_LOG("generate d_firstneigh_ptrs, h_group_numneigh[atom_all]=%d",all_neigh_pairs);
        for (int gr_i = 0; gr_i < atom_all; gr_i++) {
            i = gr_i; // 获取原子索引
            ba_i = h_group_numneigh[gr_i];
            nnumber = h_group_numneigh[gr_i+1]-h_group_numneigh[gr_i];
            DEBUG_LOG("h_group_numneigh=%d, num=%d" ,ba_i,nnumber);
            DEBUG_LOG("end of firstneigh[i]=%d,h_firstneigh_ptrs[h_group_numneigh[gr_i+1]-1] = %d",firstneigh[i][nnumber-1],h_firstneigh_ptrs[h_group_numneigh[gr_i+1]-1]);
            memcpy(&(h_firstneigh_ptrs[ba_i]),firstneigh[i],(nnumber)*sizeof(int));
            DEBUG_LOG("h_firstneigh_ptrs[h_group_numneigh[gr_i+1]-1] = %d",h_firstneigh_ptrs[h_group_numneigh[gr_i+1]-1]);
            if (gr_i==10){
                for (int i =ba_i; i<ba_i+nnumber ;i++){
                    DEBUG_LOG("%d,",atom->tag[h_firstneigh_ptrs[i]]);
                }
            }
        }
        d_firstneigh_ptrs.upload_from(h_firstneigh_ptrs, (all_neigh_pairs));
        // SAFE_CUDA_MEMCPY(d_firstneigh_ptrs.ptr,h_firstneigh_ptrs,
        //     (all_neigh_pairs) * sizeof(int),cudaMemcpyHostToDevice,f_check);
        DEBUG_LOG_COND((d_firstneigh_ptrs.ptr == NULL),"d_firstneigh_ptrs list not initialized");
        DEBUG_LOG("d_firstneigh_ptrs list %d %d %d" ,h_firstneigh_ptrs[1],h_firstneigh_ptrs[2],h_firstneigh_ptrs[3]);
        DEBUG_LOG("generate end d_firstneigh_ptrs");
        if (!init_flag) {init_flag = true;}
    }
    LAMMPS_NS::tagint all_neigh_pairs = h_group_numneigh[atom_all];
    // =========================================================================
    // h_x / h_x_flat / d_x_flat :
    //      atoms coordinate position
    // =========================================================================
    // int *h_tag = atom->tag;       // 原子全局ID数组(主机)
    // int *d_tag;             // 设备端坐标二级指针
    double **h_x = atom->x;      // 原子坐标数组(主机)
    // delete[] h_x_flat;
    // h_x_flat = nullptr;
    // DEBUG_LOG("free h_x_flat");
    // h_x_flat = new double [(atom->nlocal + atom->nghost) * 3];
    DEBUG_LOG("d_x_flat=%p",d_x_flat.ptr);
    lmp->memory->grow(h_x_flat, (atom->nlocal + atom->nghost) * 3, "STEIN_QL:h_x_flat");
    for (int i = 0; i < (atom->nlocal + atom->nghost); i++) {
        memcpy(&(h_x_flat[i*3]), h_x[i], 3*sizeof(double));
    }
    DEBUG_LOG("there are %d, h_x_flat[10]=%f",(atom->nlocal + atom->nghost),h_x_flat[10]);
    // SAFE_CUDA_FREE(d_x_flat); 
    // SAFE_CUDA_MALLOC(&d_x_flat, ((atom->nlocal + atom->nghost) * 3)*sizeof(double),f_check);
    d_x_flat.grow_to((atom->nlocal + atom->nghost) * 3, __FILE__, __LINE__);
    d_x_flat.upload_from(h_x_flat, (atom->nlocal + atom->nghost) * 3);
    // SAFE_CUDA_MEMCPY(d_x_flat.ptr,h_x_flat,((atom->nlocal + atom->nghost) * 3)*sizeof(double),cudaMemcpyHostToDevice, f_check);
    // check the pointer
    // DEBUG_LOG("alloc h_x,h_tag.....");
    DEBUG_LOG_COND((h_x == NULL),"h_x list not initialized");
    DEBUG_LOG_COND((h_x_flat == NULL),"h_x_flat list not initialized");
    DEBUG_LOG_COND((d_x_flat.ptr == NULL),"d_x_flat list not initialized");
    DEBUG_LOG("d_x_flat Allocated at: %p", d_x_flat.ptr);
    cudaDeviceSynchronize(); // waiting memory

    // =========================================================================
    // create output address
    // d_neigh_in_cutoff_r : neighbour atoms that satisfied cutoff_r
    // d_neigh_in_switching : the sum of neigh_in_switching for calculated atoms
    // d_calculated_numneigh : local tag of cutoff neighbor
    // =========================================================================
    DEBUG_LOG("release gpu");
    atom_all = (atom_all > N) ? atom_all : N;
    d_neigh_in_cutoff_r.grow_to(atom_all, __FILE__, __LINE__);
    d_neigh_in_cutoff_r.clear_async();
    d_neigh_in_switching.grow_to(atom_all, __FILE__, __LINE__);
    d_neigh_in_switching.clear_async();
    d_calculated_numneigh.grow_to(all_neigh_pairs, __FILE__, __LINE__);
    d_calculated_numneigh.clear_async();
    // d_is_pure_J.grow_to(atom_all, __FILE__, __LINE__);
    // d_is_pure_J.clear_async();
    DEBUG_LOG("release end");

    // =========================================================================
    // start kernel for calculate 
    // d_neigh_in_cutoff_r  : how many neigh atoms in cutoff_r (\sigma r_cut less than)
    // h_LQ_mask            : mask atom witch is i and j
    // d_neigh_in_switching : sum of sigma(rij) for j in neigh(i)
    // =========================================================================
    DEBUG_LOG("box_lim x:%f y:%f z:%f max:%f" ,box_x,box_y,box_z,box_x+box_y+box_z );
    DEBUG_LOG("neigh finding .......");
    DEBUG_LOG("i will start a kernel");
    // kernel function will run
    cudaDeviceSynchronize(); // waiting memory
    // cudaError_t launchErr = cudaGetLastError();
    cudaError_t syncErr;
    // launchErr = cudaGetLastError();
    // LOG("=====================================================================");
    // ERR_COND((launchErr!= cudaSuccess),"Kernel execution error: %s", cudaGetErrorString(launchErr));
    // LOG("=====================================================================");

    get_environment_Steinhardt_LocalQ<<<block_num,d_block_size>>>
      ( my_r_SWfunc->params,
        group_count, 0,
        cutoff_r, cutoff_eps_r,
        // in
        d_group_indices.ptr, d_calc_tag.ptr, d_group_numneigh.ptr, 
        d_firstneigh_ptrs.ptr, d_x_flat.ptr,
        CALC_MASK_IS_I, CALC_MASK_IS_J, CALC_SHIFT_J,
        // out
        d_neigh_in_cutoff_r.ptr, d_LQ_mask.ptr,
        d_neigh_in_switching.ptr, d_calculated_numneigh.ptr) ;
    DEBUG_LOG("env refresh out, kernel launched");
    // cudaDeviceSynchronize(); //catch kernel done

    syncErr = cudaDeviceSynchronize();
    ERR_COND((syncErr != cudaSuccess),"Kernel execution error: %s\n", cudaGetErrorString(syncErr));

    // 声明一个独特的gpu上的int,防止每个线程独立创建自己的int导致踩踏
    cudaStream_t lmp_stream = 0;
    num_of_all_IJ_atoms = 0;
    d_pure_J_write_offset.upload_from(&num_of_all_IJ_atoms, 1, lmp_stream, __FILE__, __LINE__);
    int block_num_for_atom_all = (atom_all-1+d_block_size)/d_block_size;
    get_environment_Steinhardt_LocalQ_promote_pure_K<<<block_num_for_atom_all,d_block_size>>>
        (   // in
            group_count,
            atom_all,
            d_LQ_mask.ptr, 
            d_pure_J_write_offset.ptr,
            d_group_indices.ptr,
            d_calc_tag.ptr
        );
    // cudaMemcpyAsync(&num_of_all_IJ_atoms, d_pure_J_write_offset.ptr, sizeof(int), cudaMemcpyDeviceToHost, lmp_stream);
    d_pure_J_write_offset.download_to(&num_of_all_IJ_atoms, 1, lmp_stream, __FILE__, __LINE__);
    cudaStreamSynchronize(lmp_stream);

    syncErr = cudaDeviceSynchronize();
    ERR_COND((syncErr != cudaSuccess),"Kernel execution error: %s\n", cudaGetErrorString(syncErr));
    
    block_num_for_atom_all = (num_of_all_IJ_atoms-1+d_block_size)/d_block_size;
    get_environment_Steinhardt_LocalQ<<<block_num_for_atom_all,d_block_size>>>
      ( my_r_SWfunc->params,
        num_of_all_IJ_atoms, group_count,
        cutoff_r, cutoff_eps_r,
        // in
        d_group_indices.ptr, d_calc_tag.ptr, d_group_numneigh.ptr, 
        d_firstneigh_ptrs.ptr, d_x_flat.ptr,
        CALC_MASK_IS_J, CALC_MASK_IS_K, CALC_SHIFT_K,
        // out
        d_neigh_in_cutoff_r.ptr, d_LQ_mask.ptr,
        d_neigh_in_switching.ptr, d_calculated_numneigh.ptr) ;
    num_of_all_IJ_atoms += group_count;
    DEBUG_LOG("env refresh out, kernel launched");
    cudaDeviceSynchronize(); //catch kernel done

    
    syncErr = cudaDeviceSynchronize();
    ERR_COND((syncErr != cudaSuccess),"Kernel execution error: %s\n", cudaGetErrorString(syncErr));

    DEBUG_LOG("group_count=%d, atom_all=%d, all_neigh_pairs=%lld, num_of_all_IJ_atoms=%d", 
           group_count, atom_all, (long long)all_neigh_pairs, num_of_all_IJ_atoms);

    // 清空，全写0
    d_calculated_firstneigh_ptrs.grow_to(atom->nmax, __FILE__, __LINE__);
    d_calculated_firstneigh_ptrs.clear_async();
    // a[n+1] = b[0]+...+b[n], a[0]=0
    d_neigh_in_cutoff_r.scan_to(d_calculated_firstneigh_ptrs, 
                            num_of_all_IJ_atoms+1, lmp_stream);
    d_calculated_firstneigh_ptrs.download_to(h_calculated_firstneigh_ptrs, 
                            num_of_all_IJ_atoms+1, lmp_stream, __FILE__, __LINE__);
    cudaStreamSynchronize(lmp_stream);
    num_of_all_calc_fullpair = h_calculated_firstneigh_ptrs[num_of_all_IJ_atoms];

    LOG("num_of_all_calc_fullpair=%lld (from scan of %d elements), last raw d_neigh_in_cutoff_r[num_of_all_IJ_atoms]=?",
            (long long)num_of_all_calc_fullpair, num_of_all_IJ_atoms);

    syncErr = cudaDeviceSynchronize();
    ERR_COND((syncErr != cudaSuccess),"Kernel execution error: %s\n", cudaGetErrorString(syncErr));

    DEBUG_LOG("im out");
    DEBUG_LOG("neigh find finished");


    // return the array for neigh
    DEBUG_LOG("copy result array to cpu: neigh_in_cutoff_r, h_neigh_in_switching");
    DEBUG_LOG_COND((neigh_in_cutoff_r == NULL),"neigh_in_cutoff_r list not initialized");
    DEBUG_LOG_COND((h_neigh_in_switching == NULL),"h_neigh_in_switching list not initialized");
    lmp->memory->grow(neigh_in_cutoff_r, (atom_all), "STEIN_QL:neigh_in_cutoff_r");
    d_neigh_in_cutoff_r.download_to(neigh_in_cutoff_r,num_of_all_IJ_atoms+1, lmp_stream, __FILE__, __LINE__);
    // SAFE_CUDA_MEMCPY(neigh_in_cutoff_r, d_neigh_in_cutoff_r.ptr,
    //   (group_count) * sizeof(int), cudaMemcpyDeviceToHost,f_check);
    lmp->memory->grow(h_neigh_in_switching, (atom_all), "STEIN_QL:h_neigh_in_switching");
    d_neigh_in_switching.download_to(h_neigh_in_switching, atom_all, lmp_stream, __FILE__, __LINE__);
    // SAFE_CUDA_MEMCPY(h_neigh_in_switching, d_neigh_in_switching.ptr,
    //   (atom_all) * sizeof(int), cudaMemcpyDeviceToHost,f_check);
    lmp->memory->grow(calculated_numneigh, (all_neigh_pairs), "STEIN_QL:calculated_numneigh");
    d_calculated_numneigh.download_to(calculated_numneigh, all_neigh_pairs, lmp_stream, __FILE__, __LINE__);
    // SAFE_CUDA_MEMCPY(calculated_numneigh, d_calculated_numneigh.ptr,
    //   (all_neigh_pairs) * sizeof(LAMMPS_NS::tagint), cudaMemcpyDeviceToHost,f_check);
    cudaDeviceSynchronize(); //catch kernel done
    DEBUG_LOG("copy end");
    this->last_update_step = lmp->update->ntimestep;
}



// template <int L>
// auto MetaD_zqc::STEIN_QL<L>::set_CV_calculate(std::string func_name) -> CV_Calculation {
//     // 1. 按照 "." 分割 func_name
//     std::string main_func = func_name;
//     std::string sub_param = "";
    
//     size_t dot_pos = func_name.find('.');
//     if (dot_pos != std::string::npos) {
//         main_func = func_name.substr(0, dot_pos);   // 拿到 "." 前面的部分，如 "SW_FUNC"
//         sub_param = func_name.substr(dot_pos + 1);  // 拿到 "." 后面的部分，如 "Fermi" 或 "Cubic"
//     }


//     if (main_func == "AVE") {
//         return static_cast<CV_Calculation>(&STEIN_QL<L>::compute_cv_AVE);
//     } else if (main_func == "SW_FUNC") {
//         auto it = Fixmetad->get_switching_function(sub_param);
//         if (it != nullptr) {
//             // 成功让类中的成员指针指向已经构造好的 SW1 (RATIONAL 实例)
//             this->my_cv_SWfunc = it;
//         } else {
//             // 如果脚本里写错了名字（比如写成了 SW2 却没声明），直接让 LAMMPS 报错
//             ERR_COND(1, "Switching function %s used in SYMBOL but not defined in CAL!", sub_param.c_str());
//         }
//         return static_cast<CV_Calculation>(&STEIN_QL<L>::compute_cv_SW_FUNC);
//     } else {
//         ERR_COND(1, "We can't find the func %s.", main_func.c_str());
//         return nullptr;
//     }
// }

// template <int L>
// auto MetaD_zqc::STEIN_QL<L>::set_CV_bias_force(std::string func_name) -> CV_BiasForce {
//     // 1. 按照 "." 分割 func_name
//     std::string main_func = func_name;
//     std::string sub_param = "";
    
//     size_t dot_pos = func_name.find('.');
//     if (dot_pos != std::string::npos) {
//         main_func = func_name.substr(0, dot_pos);   // 拿到 "." 前面的部分，如 "SW_FUNC"
//         sub_param = func_name.substr(dot_pos + 1);  // 拿到 "." 后面的部分，如 "Fermi" 或 "Cubic"
//     }

//     if (main_func == "AVE") {
//         return static_cast<CV_BiasForce>(&STEIN_QL<L>::bias_force_AVE);
//     } else if (main_func == "LOC_AVE") {
//         // return static_cast<CV_BiasForce>(&STEIN_QL<L>::bias_force_LOC_AVE);
//     } else if (main_func == "SW_FUNC") {
//         return static_cast<CV_BiasForce>(&STEIN_QL<L>::bias_force_SW_FUNC);
//     } else {
//         ERR_COND(1, "We can't find the func %s.", main_func.c_str());
//         return nullptr;
//     }
// }

// template <int L>
// void MetaD_zqc::STEIN_QL<L>::base_calc(){
//     compute_Q_peratoms();
// }

template <int L>
double MetaD_zqc::STEIN_LocalQL<L>::compute_cv_AVE(){
    // double global_sw_sum;
    DEBUG_LOG("im in compute_cv_AVE.");
    int group_count = my_loc_env->group_count;
    int Threads_own_atoms = lmp->atom->nlocal;
    DEBUG_LOG("group_count = %d",group_count);
    double ql_sum_local=0;
    double sw_sum_local=0;
    DEBUG_LOG_COND((stein_q == NULL),"stein_q list not initialized");
    if (group_count != 0) {
        my_averager->compute_sw(Threads_own_atoms, stein_q, lmp->atom->mask, 
            my_loc_env->groupbit, ql_sum_local, sw_sum_local, my_cv_SWfunc);
    }

    for(int i=0; i<(group_count); i++){
        DEBUG_LOG("stein_q[%d]=%g",i,stein_q[i]);
    }

    // MPI_Allreduce(&ql_sum_local, &cv_value, 1, MPI_DOUBLE, MPI_SUM, lmp->world);
    double local_sums[2] = {ql_sum_local, sw_sum_local};
    double global_sums[2] = {0.0, 0.0};
    // 一次性规约两个值
    MPI_Allreduce(local_sums, global_sums, 2, MPI_DOUBLE, MPI_SUM, lmp->world);
    // 分配回变量，并计算最终的 CV
    global_ql_sum = global_sums[0];
    global_sw_sum = global_sums[1];

    cv_value = (global_sw_sum != 0.0) ? (global_ql_sum / global_sw_sum) : 0.0;

    DEBUG_LOG("group_count = %d, compute_cv_AVE = %g",group_count, cv_value);
    // this->global_cvsw_sum = global_sw_sum;
    return cv_value;
}

template <int L>
void MetaD_zqc::STEIN_LocalQL<L>::bias_force_AVE(double dVdcv){
    // pass
    DEBUG_LOG("MetaD_zqc::STEIN_QL<L>::bias_force_AVE");
    double **f = lmp->atom->f;
    double **x = lmp->atom->x;
    int c_tag;
    DEBUG_LOG("MetaD_zqc::STEIN_QL<L>::bias_force_AVE");
    this->get_dcvdx_AVE(cv_value, h_dcvdx);
    // DEBUG_LOG("cv_value = %g, dVdcv = %g, dcvdx = %g, %g, %g",cv_value, dVdcv, dcvdx[0], dcvdx[1], dcvdx[2]);
    // DEBUG_LOG("fx0,fy0,fz0  = %.6f, %.6f, %.6f", f[c_tag][0], f[c_tag][1], f[c_tag][2]);
    for (int c_atom=0; c_atom<(my_loc_env->group_count); c_atom++){
        DEBUG_LOG("dcvdx, dcvdy, dcvdz  = %g, %g, %g", h_dcvdx[c_atom*3 + 0], h_dcvdx[c_atom*3 + 1], h_dcvdx[c_atom*3 + 2]);
        DEBUG_LOG("dVdcv  = %g", dVdcv);
        c_tag = (my_loc_env->h_group_indices)[c_atom];
        DEBUG_LOG("fx0,fy0,fz0  = %g, %g, %g", f[c_tag][0], f[c_tag][1], f[c_tag][2]);
        if (isnan(f[c_tag][0])||isnan(f[c_tag][1])||isnan(f[c_tag][2])){
            printf("error: force is infinity, check your system or cv_value.\n");
             error->all(FLERR, "STEIN_QL CV error: force is infinity, check your system or cv_value.");
        }
        f[c_tag][0] -= dVdcv*h_dcvdx[c_tag*3 + 0];
        f[c_tag][1] -= dVdcv*h_dcvdx[c_tag*3 + 1];
        f[c_tag][2] -= dVdcv*h_dcvdx[c_tag*3 + 2];
        DEBUG_LOG("fx,fy,fz  = %g, %g, %g", f[c_tag][0], f[c_tag][1], f[c_tag][2]);
        ERR_COND((isnan(f[c_tag][0])||isnan(f[c_tag][1])||isnan(f[c_tag][2])),"STEIN_QL CV error: force is infinity, check your system or cv_value.");
    }
    DEBUG_LOG("post_force_r_end");
}

template <int L>
void MetaD_zqc::STEIN_LocalQL<L>::get_dcvdx_AVE(double cv_value, double *dcvdx){
    int group_count = my_loc_env->group_count;
    int Threads_own_atoms = lmp->atom->nlocal+lmp->atom->nghost;
    int last_group_count = my_loc_env->last_group_count;
    size_t datalen = 0;
    this->cv_value = cv_value;
    

    datalen = (Threads_own_atoms * (stein_l + 1) * 2);
    lmp->memory->grow(h_stein_qlm, datalen, "STEIN_LocalQL:h_stein_qlm");
    SAFE_CUDA_MEMCPY(h_stein_qlm, d_stein_qlm.ptr, datalen*sizeof(double), cudaMemcpyDeviceToHost,f_check);


    datalen = (Threads_own_atoms*3);
    lmp->memory->grow(h_dcvdx, datalen, "STEIN_LocalQL:h_dcvdx");
    d_dcvdx.grow_to(datalen, __FILE__, __LINE__);
    d_dcvdx.clear_async();
    // SAFE_CUDA_MEMCPY(d_dcvdx.ptr,h_dcvdx, datalen*sizeof(double),cudaMemcpyHostToDevice,f_check);


    datalen = (my_loc_env->num_of_all_calc_fullpair*2*(L+1)*3);
    lmp->memory->grow(h_dYlm_dr, datalen, "STEIN_LocalQL:h_dYlm_dr");
    d_dYlm_dr.grow_to(datalen, __FILE__, __LINE__);


    datalen = (my_loc_env->num_of_all_IJ_atoms)*2*(L+1);
    d_dcvdx_rjk_prefix.grow_to(datalen, __FILE__, __LINE__);
    d_dcvdx_rjk_prefix.clear_async();

    // // sync Stein_qlm and stein_q with communication
    // // then we can directly use the data in device to calculate dcvdx, 
    // // without worrying about the data consistency between MPI processes.
    // DEBUG_LOG("[Rank:%d][Before Comm] h_stein_qlm[0] = %f, ptr = %p\n",lmp->comm->me, h_stein_qlm[0], (void*)h_stein_qlm);
    // DEBUG_LOG("[Rank:%d][Before Comm] stein_q[0] = %f, ptr = %p\n",lmp->comm->me, stein_q[0], (void*)h_stein_qlm);
    // cudaDeviceSynchronize(); // waiting memory
    // MPI_Barrier(lmp->world); // ensure all processes reach this point before communication
    // comm_mode=true;
    // lmp->comm->forward_comm(Fixmetad);
    // comm_mode=false;
    // DEBUG_LOG("[Rank:%d][After Comm] h_stein_qlm[0] = %f, ptr = %p\n",lmp->comm->me, h_stein_qlm[0], (void*)h_stein_qlm);
    // DEBUG_LOG("[Rank:%d][After Comm] stein_q[0] = %f, ptr = %p\n",lmp->comm->me, stein_q[0], (void*)h_stein_qlm);

    // SAFE_CUDA_MEMCPY(d_stein_qlm.ptr, h_stein_qlm, ((Threads_own_atoms)*(L + 1)*2)*sizeof(double), cudaMemcpyHostToDevice,f_check);
    // SAFE_CUDA_MEMCPY(d_stein_ql.ptr, stein_q, Threads_own_atoms*sizeof(double), cudaMemcpyHostToDevice,f_check);
    // SAFE_CUDA_MEMCPY(my_loc_env->d_neigh_both_in_r_N.ptr, my_loc_env->neigh_both_in_r_N, Threads_own_atoms*sizeof(int), cudaMemcpyHostToDevice,f_check);


    DEBUG_LOG("i will start a kernel of ql");
    cudaDeviceSynchronize(); // waiting memory
    CUDA_SYNC_CHECK(f_check,error);
    call_steinhardt_Local_dcv_AVE_kernel();
    d_dcvdx.download_to(h_dcvdx, (Threads_own_atoms*3), 0, __FILE__, __LINE__);
    // cudaMemcpy(h_dcvdx, d_dcvdx.ptr, (Threads_own_atoms*3)*sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize(); // waiting memory
    DEBUG_LOG("i am out");

    
    // sync dcvdx in others threads' ghost
    DEBUG_LOG("[Rank:%d][Before Comm] h_dcvdx[0] = %f, ptr = %p\n",lmp->comm->me, h_dcvdx[0], (void*)h_dcvdx);
    MPI_Barrier(lmp->world); // ensure all processes reach this point before communication
    comm_mode=true;
    lmp->comm->reverse_comm(Fixmetad);
    comm_mode=false;
    DEBUG_LOG("[Rank:%d][After Comm] h_dcvdx[0] = %f, ptr = %p\n",lmp->comm->me, h_dcvdx[0], (void*)h_dcvdx);
}


template <int L>
void MetaD_zqc::STEIN_LocalQL<L>::steinhardt_param_calc(double *stein_ql){
    int cutoff_Natoms = my_loc_env->cutoff_Natoms;
    int last_group_count = my_loc_env->last_group_count;
    int group_count = my_loc_env->group_count;
    int Threads_own_atoms = lmp->atom->nlocal + lmp->atom->nghost;
    DEBUG_LOG("Threads_own_atoms=%d, nlocal=%d, nghost=%d, group_count=%d, num_of_all_IJ_atoms=%d",
            Threads_own_atoms, lmp->atom->nlocal, lmp->atom->nghost,
            my_loc_env->group_count, my_loc_env->num_of_all_IJ_atoms);
    
    // ---- 新增: 全局零近邻早退 ----
    // num_of_all_calc_fullpair==0 意味着这个 group 里没有任何一对原子
    if (my_loc_env->num_of_all_calc_fullpair == 0) {
        DEBUG_LOG("num_of_all_calc_fullpair == 0, no atom pair within cutoff, return zero directly.");
        // 保证下游依赖的显存 buffer 仍然是"已分配、且为0"的状态
        d_stein_ql.grow_to(Threads_own_atoms, __FILE__, __LINE__);
        d_stein_ql.clear_async();
        d_stein_qlm.grow_to((Threads_own_atoms*(L + 1)*2), __FILE__, __LINE__);
        d_stein_qlm.clear_async();
        d_stein_LQlm.grow_to((Threads_own_atoms*(L + 1)*2), __FILE__, __LINE__);
        d_stein_LQlm.clear_async();
        sum_of_qlm_value_weights.grow_to((Threads_own_atoms*(L + 1)*2), __FILE__, __LINE__);
        sum_of_qlm_value_weights.clear_async();
        d_stein_Ylm.grow_to(1, __FILE__, __LINE__);  // 避免 capacity 长期停留在0导致后续某次"从0到非0"的首次扩容分支被跳过
        d_stein_Ylm.clear_async();
        // host端结果也清零, 保证 compute_cv_AVE 等下游函数拿到的是干净的0
        for (int i = 0; i < group_count; i++) stein_ql[i] = 0.0;
        return;
    }

    // TODO: we can change the cuda stream to lammps stream, 
    // but we need to make sure that the stream is synchronized before we copy data back to host. 
    // For now, we will use the default stream.
    cudaStream_t lammps_stream = 0; // Assuming you want to use the default stream. Adjust if you have a specific stream.
    // in class protect
    // result array
    // every q has <2*L + 1> qlm, with complex we will times 2
    // double *h_stein_qlm = new double [group_count*(L + 1)*2];
    // for the further concentrate we need to calculate qlm*Neigh, with comple
    d_stein_Ylm.grow_to((my_loc_env->num_of_all_calc_fullpair*(L + 1)*2), __FILE__, __LINE__);

    // SAFE_CUDA_FREE(d_stein_ql);
    // SAFE_CUDA_MALLOC(&d_stein_ql, Threads_own_atoms*sizeof(double), f_check);
    d_stein_ql.grow_to(Threads_own_atoms, __FILE__, __LINE__);
    cudaMemsetAsync(d_stein_ql.ptr, 0, (Threads_own_atoms)*sizeof(double), lammps_stream);
    // SAFE_CUDA_FREE(d_stein_qlm);
    // SAFE_CUDA_MALLOC(&d_stein_qlm, (Threads_own_atoms*(L + 1)*2)*sizeof(double), f_check);
    d_stein_qlm.grow_to((Threads_own_atoms*(L + 1)*2), __FILE__, __LINE__);
    cudaMemsetAsync(d_stein_qlm.ptr, 0, (Threads_own_atoms*(L + 1)*2)*sizeof(double), lammps_stream);
    d_stein_LQlm.grow_to((Threads_own_atoms*(L + 1)*2), __FILE__, __LINE__);
    d_stein_LQlm.clear_async();

    sum_of_qlm_value_weights.grow_to((Threads_own_atoms*(L + 1)*2), __FILE__, __LINE__);
    sum_of_qlm_value_weights.clear_async();

    DEBUG_LOG("i will start a kernel of ql");
    cudaDeviceSynchronize(); // waiting memory
    cudaError_t launchErr = cudaGetLastError();
    call_steinhardt_Local_cv_AVE_kernel();
    // cudaDeviceSynchronize(); //catch kernel done
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        // 获取详细错误信息
        const char* errName = cudaGetErrorName(syncErr);
        const char* errString = cudaGetErrorString(syncErr);
        fprintf(stderr, "\n[CRITICAL ERROR] CUDA Sync Failed!\n");
        fprintf(stderr, "Code: %d\nName: %s\nDesc: %s\n", syncErr, errName, errString);
        if (f_check) {
            fprintf(f_check, "\n[CRITICAL ERROR] CUDA Sync Failed: %s (%s)\n", errName, errString);
            fflush(f_check);
        }
        // 触发 LAMMPS 的错误退出
        error->all(FLERR, "CUDA Kernel synchronization failed. Check log for details.");
    }
    DEBUG_LOG("im out");
    DEBUG_LOG("ql calculated find finished");

    // cudaMemcpy(stein_qlm, d_stein_qlm.ptr, (group_count*(L + 1)*2) * sizeof(double), cudaMemcpyDeviceToHost);
    // SAFE_CUDA_MEMCPY(stein_ql, d_stein_ql.ptr,
    //   (group_count) * sizeof(double), cudaMemcpyDeviceToHost,f_check);
    d_stein_ql.download_to(stein_ql,Threads_own_atoms, lammps_stream, __FILE__, __LINE__);
    // SAFE_CUDA_MEMCPY(h_stein_Ylm, d_stein_Ylm.ptr,
    //   (group_count*cutoff_Natoms*(L + 1)*2) * sizeof(double), cudaMemcpyDeviceToHost,f_check);

    syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        // 获取详细错误信息
        const char* errName = cudaGetErrorName(syncErr);
        const char* errString = cudaGetErrorString(syncErr);
        fprintf(stderr, "\n[CRITICAL ERROR] CUDA Sync Failed!\n");
        fprintf(stderr, "Code: %d\nName: %s\nDesc: %s\n", syncErr, errName, errString);
        if (f_check) {
            fprintf(f_check, "\n[CRITICAL ERROR] CUDA Sync Failed: %s (%s)\n", errName, errString);
            fflush(f_check);
        }
        // 触发 LAMMPS 的错误退出
        error->all(FLERR, "CUDA Kernel synchronization failed. Check log for details.");
    }
}


template <int L>
void MetaD_zqc::STEIN_LocalQL<L>::summary(FILE* f){}

template <int L>
void MetaD_zqc::STEIN_LocalQL<L>::call_steinhardt_Local_cv_AVE_kernel(){
    int temp_block_num = (my_loc_env->num_of_all_IJ_atoms+d_block_size-1)/d_block_size;
    CUDA_SYNC_CHECK(f_check,error);
    steinhardt_Local_cv_get_qlm<L> <<<temp_block_num,d_block_size>>>(
        (my_loc_env)->my_r_SWfunc->params,
        (my_loc_env->num_of_all_IJ_atoms), (my_loc_env->cutoff_r), (my_loc_env->cutoff_eps_r), 
        (my_loc_env->d_group_indices.ptr),
        (my_loc_env->d_neigh_in_cutoff_r.ptr),
        (my_loc_env->d_group_numneigh.ptr),
        (my_loc_env->d_calculated_firstneigh_ptrs.ptr),
        (my_loc_env->d_calculated_numneigh.ptr),
        (my_loc_env->d_x_flat.ptr),
        (my_loc_env->d_neigh_in_switching.ptr),
        d_stein_qlm.ptr, d_stein_Ylm.ptr) ;
    CUDA_SYNC_CHECK(f_check,error);
    steinhardt_Local_cv_get_LQl<L> <<<block_num,d_block_size>>>(
        (my_loc_env)->my_r_SWfunc->params,
        (my_loc_env->group_count), (my_loc_env->cutoff_r), (my_loc_env->cutoff_eps_r), 
        (my_loc_env->d_group_indices.ptr),
        (my_loc_env->d_neigh_in_cutoff_r.ptr),
        (my_loc_env->d_group_numneigh.ptr),
        (my_loc_env->d_calculated_firstneigh_ptrs.ptr),
        (my_loc_env->d_calculated_numneigh.ptr),
        (my_loc_env->d_x_flat.ptr),
        (my_loc_env->d_neigh_in_switching.ptr),
        sum_of_qlm_value_weights.ptr,
        d_stein_qlm.ptr, 
        d_stein_LQlm.ptr, d_stein_ql.ptr) ;
    CUDA_SYNC_CHECK(f_check,error);
}


// template <int L>
// void MetaD_zqc::STEIN_QL<L>::call_steinhardt_dcv_AVE_pre_kernel(){ 
// }


template <int L>
void MetaD_zqc::STEIN_LocalQL<L>::call_steinhardt_Local_dcv_AVE_kernel(){ 
    // 遍历所有的ij原子对
    steinhardt_Local_dcv_AVE_ij_kernel<L> <<<block_num,d_block_size>>>(
        (my_loc_env)->my_r_SWfunc->params,
        my_cv_SWfunc->params,
        this->cv_value, this->global_sw_sum,
        (my_loc_env)->group_count,

        (my_loc_env->d_x_flat.ptr),
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
        d_dcvdx.ptr);
    // 遍历所有的jk原子对
    CUDA_SYNC_CHECK(f_check,error);
    int temp_block_num = ((my_loc_env->num_of_all_IJ_atoms)+d_block_size-1)/d_block_size;
    steinhardt_Local_dcv_AVE_jk_kernel<L> <<<temp_block_num,d_block_size>>>(
        (my_loc_env)->my_r_SWfunc->params,
        (my_loc_env)->num_of_all_IJ_atoms,
        (my_loc_env)->cutoff_r,
        (my_loc_env)->cutoff_eps_r,
        
        (my_loc_env->d_mask.ptr), 
        (my_loc_env->d_x_flat.ptr),
        (my_loc_env->d_neigh_in_switching.ptr),
        (my_loc_env->d_group_indices.ptr),
        (my_loc_env->d_group_numneigh.ptr),
        (my_loc_env->d_calculated_firstneigh_ptrs.ptr),
        (my_loc_env->d_calculated_numneigh.ptr),
        (my_loc_env->d_calc_tag.ptr),
        (my_loc_env->d_neigh_in_cutoff_r.ptr),
        
        d_dcvdx_rjk_prefix.ptr,
        d_stein_qlm.ptr, d_stein_Ylm.ptr,  d_stein_ql.ptr,
        d_dYlm_dr.ptr, d_dcvdx.ptr);
    CUDA_SYNC_CHECK(f_check,error);
}


// template <int L>
// int MetaD_zqc::STEIN_QL<L>::get_comm_forward_bytes(){ 
//     // need to communicate for each atom in the list
//     // qlm[2*(L+1) ] and ql (double value) and Neigh_Nb (int value)
//     return num_elements +1 +1; // qlm + ql + Neigh_Nb
// }

// template <int L>
// int MetaD_zqc::STEIN_QL<L>::pack_comm_forward_ubuf(int n, int *list, double *u_buf, int slot_offset, int comm_forward) {
//     if (!comm_mode){
//         return (num_elements + 1 +1);
//     }
//     int m = slot_offset; 
//     int cycle_offset = comm_forward;

//     for (int i = 0; i < n; i++) {
//         int j = list[i]; // 目标本地原子标号
        
//         // 1. 先塞当前原子的所有 qlm 分量
//         for (int k = 0; k < num_elements; k++) {
//             u_buf[m + cycle_offset*i + k] = h_stein_qlm[j * num_elements + k];
//         }
        
//         // 2. 紧接着，塞当前原子的 ql 标量数据
//         u_buf[m + cycle_offset*i + num_elements] = stein_q[j]; // 假设这是你的 ql 数组

//         u_buf[m + cycle_offset*i + num_elements +1] = ubuf(my_loc_env->neigh_both_in_r_N[j]).d;
//     }
    
//     return (num_elements + 1 +1);
// }

// template <int L>
// void MetaD_zqc::STEIN_QL<L>::unpack_comm_forward_ubuf(int n, int first, double *u_buf, int slot_offset, int comm_forward) {
//     if (!comm_mode){
//         return;
//     }

//     int m = slot_offset; 
//     int cycle_offset = comm_forward;
    
//     // 从 first 开始，连续恢复 n 个 Ghost 原子的复合数据
//     for (int i = first; i < first + n; i++) {
        
//         // 1. 先剥离 qlm 倒回 qlm 跑道
//         for (int k = 0; k < num_elements; k++) {
//             h_stein_qlm[i * num_elements + k] = u_buf[ m+ cycle_offset*(i-first) + k];
//         }
        
//         // 2. 紧接着剥离 ql 倒回 ql 跑道
//         stein_q[i] = u_buf[ m+ cycle_offset*(i-first) + num_elements];

//         my_loc_env->neigh_both_in_r_N[i] = (int) ubuf(u_buf[ m+ cycle_offset*(i-first) + num_elements +1]).i;
//     }
// }


template <int L>
int MetaD_zqc::STEIN_LocalQL<L>::get_comm_reverse_bytes(){ 
    // need to communicate for each atom in the list
    return 3; // dcv/dxyz per atoms
}

template <int L>
int MetaD_zqc::STEIN_LocalQL<L>::pack_comm_reverse_ubuf(int n, int first, 
                        double *u_buf, int slot_offset, int comm_forward) {
    if (!comm_mode){
        return (3);
    }
    int m = slot_offset; 
    int cycle_offset = comm_forward;
    // int last = first + n

    for (int i = 0; i < n; i++) {
        #pragma unroll
        for (int direct=0; direct<3;direct++){
            u_buf[m + cycle_offset*i + direct] = h_dcvdx[(i+first)*3 + direct];
        }
    }
    
    return (3);
}

template <int L>
void MetaD_zqc::STEIN_LocalQL<L>::unpack_comm_reverse_ubuf(int n, int *list, 
                        double *u_buf, int slot_offset, int comm_forward) {
    if (!comm_mode){
        return;
    }
    int loctag;
    int m = slot_offset; 
    int cycle_offset = comm_forward;
    
    // 从 first 开始，连续恢复 n 个 Ghost 原子的复合数据
    for (int i = 0; i < n; i++) {
        loctag = list[i];
        #pragma unroll
        for (int direct=0; direct<3;direct++){
            h_dcvdx[loctag*3 + direct] += u_buf[m + cycle_offset*i + direct];
        }
    }
}


// template <int L>
// double* MetaD_zqc::STEIN_LocalQL<L>::get_peratom_ptr(const std::string &prop_name) {
//     // LOG("[DEBUG get_peratom_ptr] 收到的 prop_name = \"%s\"", prop_name.c_str());
//     if (prop_name == "stein_q") {
//         return stein_q;
//     }
    
//     // ---- debug: 扁平 dcvdx 三分量 (local 序), 每步只散射一次 ----
//     if ((prop_name == "dcvdx_x" || prop_name == "dcvdx_y" || prop_name == "dcvdx_z")
//         && dcvdx_flag != lmp->update->ntimestep) {
//         int nlocal = lmp->atom->nlocal;
//         int nmax   = lmp->atom->nmax;
//         // if (nmax > nmax_pa) {
//         //     delete[] h_dcvdx_x; delete[] h_dcvdx_y; delete[] h_dcvdx_z;
//         //     h_dcvdx_x = new double[nmax];
//         //     h_dcvdx_y = new double[nmax];
//         //     h_dcvdx_z = new double[nmax];
//         //     nmax_pa = nmax;
//         // }
//         lmp->memory->grow(h_dcvdx_x, nmax, "STEIN_QL:h_dcvdx_x");
//         lmp->memory->grow(h_dcvdx_y, nmax, "STEIN_QL:h_dcvdx_y");
//         lmp->memory->grow(h_dcvdx_z, nmax, "STEIN_QL:h_dcvdx_z");
//         for (int i = 0; i < nlocal; ++i) {
//             h_dcvdx_x[i] = 0.0; h_dcvdx_y[i] = 0.0; h_dcvdx_z[i] = 0.0;
//         }
//         for (int c = 0; c < my_env->group_count; ++c) {
//             // int i = (my_env->h_group_indices)[c];   // local index
//             int i = c;
//             h_dcvdx_x[i] = h_dcvdx[c*3 + 0];
//             h_dcvdx_y[i] = h_dcvdx[c*3 + 1];
//             h_dcvdx_z[i] = h_dcvdx[c*3 + 2];
//         }
//         dcvdx_flag = lmp->update->ntimestep;
//     }

//     if (prop_name == "dcvdx_x") {
//         return h_dcvdx_x;
//     }
//     if (prop_name == "dcvdx_y") {
//         return h_dcvdx_y;
//     }
//     if (prop_name == "dcvdx_z") {
//         return h_dcvdx_z;
//     }


//     return nullptr;
// }

// 需要计算的calc_tag原子的邻居列表
__global__ void get_environment_Steinhardt_LocalQ(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, int start_idx,
    double cutoff_r, double cut_sigma_eps,
    // in
    LAMMPS_NS::tagint *d_group_indices,
    LAMMPS_NS::tagint *d_calc_tag,
    LAMMPS_NS::tagint *d_group_numneigh,
    int *d_firstneigh_ptrs, double *d_x_flat,
    unsigned int current_pass_mask,   // 中心原子该打的标签 (例如：CALC_MASK_IS_I 或 CALC_MASK_IS_J)
    unsigned int neighbor_mask,       // 邻居原子该打的标签 (例如：CALC_MASK_IS_J 或 CALC_MASK_IS_K)
    unsigned int neighbor_inc_val,    // 邻居原子计数器自增的位移量 (例如：1U << CALC_SHIFT_I)
    // out
    int *d_neigh_in_cutoff_r, int *d_LQ_mask,
    double *d_neigh_in_switching,
    LAMMPS_NS::tagint *d_calculated_numneigh){

    #define sw_f(r) (MetaD_zqc::SwitchFunction::f(sw_params_rij, (r)))
    // get_environment_Steinhardt_LocalQ in GPU
    int c_atom = blockIdx.x * blockDim.x + threadIdx.x;
    if(c_atom<calc_count){
        int c_atom_calctag = c_atom+start_idx;
        int c_atom_loctag = d_group_indices[c_atom_calctag];
        int temp_tag;
        d_neigh_in_cutoff_r[c_atom_calctag] = 0;
        // c_glob_tag = h_tag[c_atom_loctag];
        double c_x = d_x_flat[c_atom_loctag*3];
        double c_y = d_x_flat[c_atom_loctag*3+1];
        double c_z = d_x_flat[c_atom_loctag*3+2];
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
            neigh_x = d_x_flat[n_local_tag*3];
            neigh_y = d_x_flat[n_local_tag*3+1];
            neigh_z = d_x_flat[n_local_tag*3+2];
            delt_x = (neigh_x - c_x);
            delt_y = (neigh_y - c_y);
            delt_z = (neigh_z - c_z);
            r2 = delt_x*delt_x + delt_y*delt_y + delt_z*delt_z;
            r = sqrt(r2);
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
    LAMMPS_NS::tagint *d_pure_J_write_offset,
    LAMMPS_NS::tagint *d_group_indices,
    LAMMPS_NS::tagint *d_calc_tag) {
    
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
    LAMMPS_NS::tagint *d_group_numneigh,
    LAMMPS_NS::tagint *d_calculated_firstneigh_ptrs,
    LAMMPS_NS::tagint *d_calculated_numneigh,
    double *d_x_flat,
    double *d_neigh_in_switching,
    double *d_stein_qlm, double *d_stein_Ylm) {
    // 此处应该计算i,j原子的qlm

    int c_atom_calctag = blockIdx.x * blockDim.x + threadIdx.x;
    #define sw_f(r) (MetaD_zqc::SwitchFunction::f(sw_params_rij, (r)))
    if (c_atom_calctag >= calc_count) return;

    LAMMPS_NS::tagint c_atom_loctag = d_group_indices[c_atom_calctag]; // 当前原子在local原子列表中的标签
    constexpr int lm_size = (L + 1) * 2;

    // 【与导数核函数完美镜像】的近邻数读取与基础寻址逻辑
    int neigh_num = d_neigh_in_cutoff_r[c_atom_calctag];
    LAMMPS_NS::tagint stein_qlm_base_id = c_atom_loctag * lm_size; // 为了通讯方便，qlm和Ylm都按照local原子标签来存储和访问
    LAMMPS_NS::tagint stein_Ylm_base_id;

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
    double c_x = d_x_flat[c_atom_loctag*3];
    double c_y = d_x_flat[c_atom_loctag*3+1];
    double c_z = d_x_flat[c_atom_loctag*3+2];

    LAMMPS_NS::tagint neigh_base = d_group_numneigh[c_atom_loctag];
    LAMMPS_NS::tagint neigh_pair_base = d_calculated_firstneigh_ptrs[c_atom_calctag];

    for (LAMMPS_NS::tagint neigh_atom = 0; neigh_atom < neigh_num; neigh_atom++) {
        LAMMPS_NS::tagint n_local_tag = d_calculated_numneigh[neigh_base + neigh_atom];
        // Ylm 只与原子位置有关，所以可以按照group直接访问不需要扩大数组,所以用c_atom而不是c_atom_tag
        stein_Ylm_base_id = (neigh_pair_base+neigh_atom)*lm_size;
        double local_Ylm[lm_size] = {0.0};
        
        double neigh_x = d_x_flat[n_local_tag*3];
        double neigh_y = d_x_flat[n_local_tag*3+1];
        double neigh_z = d_x_flat[n_local_tag*3+2];
        double delt_x = (neigh_x - c_x);
        double delt_y = (neigh_y - c_y);
        double delt_z = (neigh_z - c_z);
        double r2 = delt_x*delt_x + delt_y*delt_y + delt_z*delt_z;
        double r = sqrt(r2);
        double r_weight = sw_f(r);

        double theta = acos(fmax(-1.0, fmin(1.0, delt_z / r)));
        double phi = atan2(delt_y, delt_x);

        double sin_theta, cos_theta, sin_phi, cos_phi;
        double sin_2theta, cos_2theta, sin_2phi, cos_2phi;
        double sin_3theta, cos_3theta, sin_3phi, cos_3phi;
        double sin_4theta, cos_4theta, sin_4phi, cos_4phi;
        double sin_5theta, cos_5theta, sin_5phi, cos_5phi;
        double sin_6theta, cos_6theta, sin_6phi, cos_6phi;
        sincos(theta, &sin_theta, &cos_theta);
        sincos(phi, &sin_phi, &cos_phi);

        // 【完全通用】的三角函数倍角级联（编译期静态分支分发）
        if constexpr (L >= 2) {
            sincos(2 * theta, &sin_2theta, &cos_2theta);
            sincos(2 * phi, &sin_2phi, &cos_2phi);
        }
        if constexpr (L >= 3) {
            sincos(3 * phi, &sin_3phi, &cos_3phi);
            sincos(3 * theta, &sin_3theta, &cos_3theta);
            sincos(4 * theta, &sin_4theta, &cos_4theta);
        }
        if constexpr (L >= 4) {
            sincos(4 * phi, &sin_4phi, &cos_4phi);
            sincos(5 * theta, &sin_5theta, &cos_5theta);
            sincos(5 * phi, &sin_5phi, &cos_5phi);
            sincos(6 * theta, &sin_6theta, &cos_6theta);
        }
        if constexpr (L >= 6) {
            sincos(6 * phi, &sin_6phi, &cos_6phi);
        }
        

        // ==========================================================
        //  💥 核心艺术：利用编译期静态判断条件，杜绝任何浪费！
        // ==========================================================
        if constexpr (L == 3) {
            compute_qlm_forward_L3(
                r_weight,
                cos_theta, sin_theta, cos_phi, sin_phi,
                cos_2theta, sin_2theta, cos_2phi, sin_2phi,
                cos_3theta, sin_3theta, cos_3phi, sin_3phi,
                local_qlm, local_Ylm
            );
        } else if constexpr (L == 4) {
            compute_qlm_forward_L4(
                r_weight,
                cos_theta, sin_theta, cos_phi, sin_phi,
                cos_2phi, sin_2phi,
                cos_3phi, sin_3phi,
                cos_4phi, sin_4phi,
                local_qlm, local_Ylm
            );
        } else if constexpr (L == 6) {
            compute_qlm_forward_L6(
                r_weight,
                cos_theta, sin_theta, cos_phi, sin_phi,
                cos_2theta, sin_2theta, cos_2phi, sin_2phi,
                cos_3phi, sin_3phi,
                cos_4theta, sin_4theta, cos_4phi, sin_4phi,
                cos_5phi, sin_5phi,
                cos_6theta, sin_6theta, cos_6phi, sin_6phi,
                local_qlm, local_Ylm
            );
        }
        cuda::std::memcpy(&d_stein_Ylm[stein_Ylm_base_id], &local_Ylm, 
                            sizeof(double)*lm_size);
    }

    // // --- 循环外归一化与写回全局显存 ---
    
    #pragma unroll
    for (int i = 0; i < lm_size; i++) {
        local_qlm[i] *= inv_neigh;
        // d_stein_qlm[stein_qlm_base_id + i] = local_qlm[i];
    }
    cuda::std::memcpy(&d_stein_qlm[stein_qlm_base_id], &local_qlm, 
                        sizeof(double)*lm_size);

    double ql_sq = local_qlm[0] * local_qlm[0];    
    #undef sw_f
}
template __global__ void steinhardt_Local_cv_get_qlm<3>(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, double cutoff_r, double cutoff_eps,
    int *d_group_indices,
    int *d_neigh_in_cutoff_r,
    LAMMPS_NS::tagint *d_group_numneigh,
    LAMMPS_NS::tagint *d_calculated_firstneigh_ptrs,
    LAMMPS_NS::tagint *d_calculated_numneigh,
    double *d_x_flat,
    double *d_neigh_in_switching,
    double *d_stein_qlm, double *d_stein_Ylm);
template __global__ void steinhardt_Local_cv_get_qlm<4>(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, double cutoff_r, double cutoff_eps,
    int *d_group_indices,
    int *d_neigh_in_cutoff_r,
    LAMMPS_NS::tagint *d_group_numneigh,
    LAMMPS_NS::tagint *d_calculated_firstneigh_ptrs,
    LAMMPS_NS::tagint *d_calculated_numneigh,
    double *d_x_flat,
    double *d_neigh_in_switching,
    double *d_stein_qlm, double *d_stein_Ylm);
template __global__ void steinhardt_Local_cv_get_qlm<6>(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, double cutoff_r, double cutoff_eps,
    int *d_group_indices,
    int *d_neigh_in_cutoff_r,
    LAMMPS_NS::tagint *d_group_numneigh,
    LAMMPS_NS::tagint *d_calculated_firstneigh_ptrs,
    LAMMPS_NS::tagint *d_calculated_numneigh,
    double *d_x_flat,
    double *d_neigh_in_switching,
    double *d_stein_qlm, double *d_stein_Ylm);


template <int L>
__global__ void steinhardt_Local_cv_get_LQl(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, double cutoff_r, double cutoff_eps,
    int *d_group_indices,
    int *d_neigh_in_cutoff_r,
    LAMMPS_NS::tagint *d_group_numneigh,
    LAMMPS_NS::tagint *d_calculated_firstneigh_ptrs,
    LAMMPS_NS::tagint *d_calculated_numneigh,
    double *d_x_flat,
    double *d_neigh_in_switching,
    double *sum_of_qlm_value_weights,
    double *d_stein_qlm, 
    double *d_stein_LQlm, double *d_stein_ql) {
    // 此处应该只计算i原子的LQl

    int c_atom_calctag = blockIdx.x * blockDim.x + threadIdx.x;
    #define sw_f(r) (MetaD_zqc::SwitchFunction::f(sw_params_rij, (r)))
    if (c_atom_calctag >= calc_count) return;

    LAMMPS_NS::tagint c_atom_loctag = d_group_indices[c_atom_calctag]; // 当前原子在local原子列表中的标签

    // 【与导数核函数完美镜像】的近邻数读取与基础寻址逻辑
    int neigh_num = d_neigh_in_cutoff_r[c_atom_calctag];
    LAMMPS_NS::tagint stein_qlm_base_id = c_atom_loctag * (L + 1) * 2; // 为了通讯方便，qlm和Ylm都按照local原子标签来存储和访问

    // 如果没有邻居，直接清零退出
    double NFb_i_check = d_neigh_in_switching[c_atom_calctag];
    if (neigh_num == 0 || NFb_i_check < 1e-12) return;
    // because we multiply a swfunction, so it need to devide by its weight sum
    double inv_neigh = 1.0 / (double)(NFb_i_check + 1.0);

    // 在寄存器（栈）上初始化局部数组用于累加，避免频繁读写全局显存
    constexpr int lm_size = (L + 1) * 2;
    double local_LQlm[lm_size] = {0.0};
    
    double c_x = d_x_flat[c_atom_loctag*3+0];
    double c_y = d_x_flat[c_atom_loctag*3+1];
    double c_z = d_x_flat[c_atom_loctag*3+2];

    #pragma unroll
    for (int i = 0; i < lm_size; i++) {
        local_LQlm[i] = d_stein_qlm[c_atom_loctag*lm_size + i];
    }
    double sum_of_qlm_value_weight_loc[lm_size] = {0.0};

    LAMMPS_NS::tagint neigh_base = d_group_numneigh[c_atom_loctag];
    LAMMPS_NS::tagint neigh_pair_base = d_calculated_firstneigh_ptrs[c_atom_calctag];

    for (LAMMPS_NS::tagint neigh_atom = 0; neigh_atom < neigh_num; neigh_atom++) {
        LAMMPS_NS::tagint n_local_tag = d_calculated_numneigh[neigh_base + neigh_atom];
        double neigh_x = d_x_flat[n_local_tag*3+0];
        double neigh_y = d_x_flat[n_local_tag*3+1];
        double neigh_z = d_x_flat[n_local_tag*3+2];
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
    cuda::std::memcpy(&sum_of_qlm_value_weights[c_atom_calctag*lm_size], &sum_of_qlm_value_weight_loc, sizeof(double)*lm_size);

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
    LAMMPS_NS::tagint *d_group_numneigh,
    LAMMPS_NS::tagint *d_calculated_firstneigh_ptrs,
    LAMMPS_NS::tagint *d_calculated_numneigh,
    double *d_x_flat,
    double *d_neigh_in_switching,
    double *sum_of_qlm_value_weights,
    double *d_stein_qlm, 
    double *d_stein_LQlm, double *d_stein_ql);
template __global__ void steinhardt_Local_cv_get_LQl<4>(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, double cutoff_r, double cutoff_eps,
    int *d_group_indices,
    int *d_neigh_in_cutoff_r,
    LAMMPS_NS::tagint *d_group_numneigh,
    LAMMPS_NS::tagint *d_calculated_firstneigh_ptrs,
    LAMMPS_NS::tagint *d_calculated_numneigh,
    double *d_x_flat,
    double *d_neigh_in_switching,
    double *sum_of_qlm_value_weights,
    double *d_stein_qlm, 
    double *d_stein_LQlm, double *d_stein_ql);
template __global__ void steinhardt_Local_cv_get_LQl<6>(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, double cutoff_r, double cutoff_eps,
    int *d_group_indices,
    int *d_neigh_in_cutoff_r,
    LAMMPS_NS::tagint *d_group_numneigh,
    LAMMPS_NS::tagint *d_calculated_firstneigh_ptrs,
    LAMMPS_NS::tagint *d_calculated_numneigh,
    double *d_x_flat,
    double *d_neigh_in_switching,
    double *sum_of_qlm_value_weights,
    double *d_stein_qlm, 
    double *d_stein_LQlm, double *d_stein_ql);


template <int L>
__global__ void steinhardt_Local_dcv_AVE_ij_kernel(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    MetaD_zqc::SwitchFunctionRequest sw_params_LQ,
    double cv_value, double global_sw_sum,
    int group_count, 
    double *d_x_flat, double *d_neigh_in_switching,
    int *d_neigh_in_cutoff_r,
    LAMMPS_NS::tagint *d_calculated_firstneigh_ptrs,
    LAMMPS_NS::tagint *d_group_indices,
    LAMMPS_NS::tagint *d_group_numneigh,
    LAMMPS_NS::tagint *d_calculated_numneigh, 
    LAMMPS_NS::tagint *d_calc_tag, 
    double *sum_of_qlm_value_weights,
    double *d_dcvdx_rjk_prefix,
    double *d_stein_qlm, 
    double *d_stein_LQlm,
    double *d_stein_ql,
    double *d_dcvdx) {
    // 仅遍历i原子
    // c_atom 是 calctag
    int c_atom_calctag = blockIdx.x * blockDim.x + threadIdx.x;
    #define sw_f_r(r) (MetaD_zqc::SwitchFunction::f(sw_params_rij, (r)))
    #define sw_f_LQ(r) (MetaD_zqc::SwitchFunction::f(sw_params_LQ, (r)))
    #define sw_df_r(r) (MetaD_zqc::SwitchFunction::df(sw_params_rij, (r)))
    #define sw_df_LQ(r) (MetaD_zqc::SwitchFunction::df(sw_params_LQ, (r)))
    if (c_atom_calctag >= group_count) return;

    constexpr int lm_size = (L + 1) * 2 ;
    
    LAMMPS_NS::tagint c_atom_loctag = d_group_indices[c_atom_calctag]; // 当前原子在local原子列表中的标签

    int neigh_num = d_neigh_in_cutoff_r[c_atom_calctag];
    LAMMPS_NS::tagint stein_qlm_base_id;
    LAMMPS_NS::tagint i_LQlm_base = c_atom_loctag * lm_size;

    // 如果没有邻居，直接清零退出
    double NFb_i_tilt = d_neigh_in_switching[c_atom_calctag];
    if (neigh_num == 0 || NFb_i_tilt < 1e-12) return;
    NFb_i_tilt = NFb_i_tilt + 1;
    double dcvdrij_fori[3] = {0.0, 0.0, 0.0};
    
    double c_x = d_x_flat[c_atom_loctag*3+0];
    double c_y = d_x_flat[c_atom_loctag*3+1];
    double c_z = d_x_flat[c_atom_loctag*3+2];

    // 与steinhardt no local 共用一个结果数组所以是d_stein_ql
    double LQ_i = d_stein_ql[c_atom_loctag];
    LQ_i = fmax(LQ_i, 1e-12);
    double F_LQ_i_with_NF = (sw_df_LQ(LQ_i)*(LQ_i-cv_value)+sw_f_LQ(LQ_i))
                        /(LQ_i)*(4*PI)/(2*L+1)/global_sw_sum;
    
    double inv_NFb = 1.0/(NFb_i_tilt);

    double accum_re = 0.0;
    double accum_im = 0.0;



    // 邻居的分发
    LAMMPS_NS::tagint neigh_base = d_group_numneigh[c_atom_loctag];
    LAMMPS_NS::tagint neigh_pair_base = d_calculated_firstneigh_ptrs[c_atom_calctag];

    for (LAMMPS_NS::tagint neigh_atom = 0; neigh_atom < neigh_num; neigh_atom++) {
        LAMMPS_NS::tagint n_local_tag = d_calculated_numneigh[neigh_base + neigh_atom];
        double neigh_x = d_x_flat[n_local_tag*3+0];
        double neigh_y = d_x_flat[n_local_tag*3+1];
        double neigh_z = d_x_flat[n_local_tag*3+2];
        double delt_x = (neigh_x - c_x);
        double delt_y = (neigh_y - c_y);
        double delt_z = (neigh_z - c_z);
        double r2 = delt_x*delt_x + delt_y*delt_y + delt_z*delt_z;
        double r = sqrt(r2);
        double r_weight = sw_f_r(r);

        // 接下来求m=(-l,l)
        // 长度为lm_size,同时存有m>=0的所有信息。
        stein_qlm_base_id = n_local_tag*lm_size;
        LAMMPS_NS::tagint j_calc_tag = d_calc_tag[n_local_tag];
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
        LAMMPS_NS::tagint n_local_tag = c_atom_loctag;
        double r_weight = 1;

        // 接下来求m=(-l,l)
        // 长度为lm_size,同时存有m>=0的所有信息。
        stein_qlm_base_id = n_local_tag*lm_size;
        LAMMPS_NS::tagint j_calc_tag = d_calc_tag[n_local_tag];
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

    #undef sw_f_r
    #undef sw_f_LQ
    #undef sw_df_r
    #undef sw_df_LQ
}
template __global__ void steinhardt_Local_dcv_AVE_ij_kernel<3>(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    MetaD_zqc::SwitchFunctionRequest sw_params_LQ,
    double cv_value, double global_sw_sum,
    int group_count, 
    double *d_x_flat, double *d_neigh_in_switching,
    int *d_neigh_in_cutoff_r,
    LAMMPS_NS::tagint *d_calculated_firstneigh_ptrs,
    LAMMPS_NS::tagint *d_group_indices,
    LAMMPS_NS::tagint *d_group_numneigh,
    LAMMPS_NS::tagint *d_calculated_numneigh, 
    LAMMPS_NS::tagint *d_calc_tag, 
    double *sum_of_qlm_value_weights,
    double *d_dcvdx_rjk_prefix,
    double *d_stein_qlm, 
    double *d_stein_LQlm,
    double *d_stein_ql,
    double *d_dcvdx);
template __global__ void steinhardt_Local_dcv_AVE_ij_kernel<4>(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    MetaD_zqc::SwitchFunctionRequest sw_params_LQ,
    double cv_value, double global_sw_sum,
    int group_count, 
    double *d_x_flat, double *d_neigh_in_switching,
    int *d_neigh_in_cutoff_r,
    LAMMPS_NS::tagint *d_calculated_firstneigh_ptrs,
    LAMMPS_NS::tagint *d_group_indices,
    LAMMPS_NS::tagint *d_group_numneigh,
    LAMMPS_NS::tagint *d_calculated_numneigh, 
    LAMMPS_NS::tagint *d_calc_tag, 
    double *sum_of_qlm_value_weights,
    double *d_dcvdx_rjk_prefix,
    double *d_stein_qlm, 
    double *d_stein_LQlm,
    double *d_stein_ql,
    double *d_dcvdx);
template __global__ void steinhardt_Local_dcv_AVE_ij_kernel<6>(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    MetaD_zqc::SwitchFunctionRequest sw_params_LQ,
    double cv_value, double global_sw_sum,
    int group_count, 
    double *d_x_flat, double *d_neigh_in_switching,
    int *d_neigh_in_cutoff_r,
    LAMMPS_NS::tagint *d_calculated_firstneigh_ptrs,
    LAMMPS_NS::tagint *d_group_indices,
    LAMMPS_NS::tagint *d_group_numneigh,
    LAMMPS_NS::tagint *d_calculated_numneigh, 
    LAMMPS_NS::tagint *d_calc_tag, 
    double *sum_of_qlm_value_weights,
    double *d_dcvdx_rjk_prefix,
    double *d_stein_qlm, 
    double *d_stein_LQlm,
    double *d_stein_ql,
    double *d_dcvdx);



template <int L>
__global__ void steinhardt_Local_dcv_AVE_jk_kernel(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, double cutoff_r, double cutoff_eps,
    int *d_mask, 
    double *d_x_flat, double *d_neigh_in_switching,
    LAMMPS_NS::tagint *d_group_indices, 
    LAMMPS_NS::tagint *d_group_numneigh,
    LAMMPS_NS::tagint *d_calculated_firstneigh_ptrs,
    LAMMPS_NS::tagint *d_calculated_numneigh, 
    LAMMPS_NS::tagint *d_calc_tag, 
    int *d_neigh_in_cutoff_r, 
    double *d_dcvdx_rjk_prefix,
    double *d_stein_qlm, double *d_stein_Ylm, double *d_stein_ql,
    double *d_dYlm_dr, double *d_dcvdx) {

    int c_atom_calctag = blockIdx.x * blockDim.x + threadIdx.x;
    #define sw_f_r(r) (MetaD_zqc::SwitchFunction::f(sw_params_rij, (r)))
    #define sw_f_LQ(r) (MetaD_zqc::SwitchFunction::f(sw_params_LQ, (r)))
    #define sw_df_r(r) (MetaD_zqc::SwitchFunction::df(sw_params_rij, (r)))
    #define sw_df_LQ(r) (MetaD_zqc::SwitchFunction::df(sw_params_LQ, (r)))
    if (c_atom_calctag >= calc_count) return;
    
    LAMMPS_NS::tagint c_atom_loctag = d_group_indices[c_atom_calctag]; // 当前原子在local原子列表中的标签

    int neigh_num = d_neigh_in_cutoff_r[c_atom_calctag];
    constexpr int lm_size = (L + 1) * 2 ;
    LAMMPS_NS::tagint stein_qlm_base_id = c_atom_loctag * lm_size;

    // 如果没有邻居，直接清零退出
    double NFb_j_check = d_neigh_in_switching[d_calc_tag[c_atom_loctag]];
    if (neigh_num == 0 || NFb_j_check < 1e-12) return;
    
    // constexpr int lmdr_size = lm_size *3;
    MetaD_zqc::D_Ylm_Layout<L> local_dYlmdx;
    double dcvdrjk_forj[3] = {0.0, 0.0, 0.0}; // 栈上数组初始化
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
    
    double c_x = d_x_flat[c_atom_loctag*3+0];
    double c_y = d_x_flat[c_atom_loctag*3+1];
    double c_z = d_x_flat[c_atom_loctag*3+2];


    LAMMPS_NS::tagint neigh_base = d_group_numneigh[c_atom_loctag];
    LAMMPS_NS::tagint neigh_pair_base = d_calculated_firstneigh_ptrs[c_atom_calctag];

    for (LAMMPS_NS::tagint neigh_atom = 0; neigh_atom < neigh_num; neigh_atom++) {
        LAMMPS_NS::tagint n_local_tag = d_calculated_numneigh[neigh_base + neigh_atom];
        double neigh_x = d_x_flat[n_local_tag*3+0];
        double neigh_y = d_x_flat[n_local_tag*3+1];
        double neigh_z = d_x_flat[n_local_tag*3+2];
        double delt_x = (neigh_x - c_x);
        double delt_y = (neigh_y - c_y);
        double delt_z = (neigh_z - c_z);
        double r2 = delt_x*delt_x + delt_y*delt_y + delt_z*delt_z;
        double r = sqrt(r2);
        double r_weight = sw_f_r(r);

        double sum_of_sw_weight;
        sum_of_sw_weight = d_neigh_in_switching[c_atom_calctag];
        
        double theta = acos(fmax(-1.0, fmin(1.0, delt_z / r)));
        double phi = atan2(delt_y, delt_x);

        double sin_theta, cos_theta, sin_phi, cos_phi;
        double sin_2theta, cos_2theta, sin_2phi, cos_2phi;
        double sin_3theta, cos_3theta, sin_3phi, cos_3phi;
        double sin_4theta, cos_4theta, sin_4phi, cos_4phi;
        double sin_5theta, cos_5theta, sin_5phi, cos_5phi;
        double sin_6theta, cos_6theta, sin_6phi, cos_6phi;
        sincos(theta, &sin_theta, &cos_theta);
        sincos(phi, &sin_phi, &cos_phi);
        
        // 【完全通用】的三角函数倍角级联
        if constexpr (L >= 2) {
            cos_2theta = 2.0*cos_theta*cos_theta - 1.0;
            sin_2theta = 2.0*sin_theta*cos_theta;
            cos_2phi   = 2.0*cos_phi*cos_phi - 1.0;
            sin_2phi   = 2.0*sin_phi*cos_phi;
        }
        // 如果 L >= 3，才编译 3 倍角
        if constexpr (L >= 3) {
            // theta方向递推到4theta需要经过3theta(即使3theta本身在L=3时不直接使用)
            cos_3theta = 2.0*cos_theta*cos_2theta - cos_theta;
            sin_3theta = 2.0*cos_theta*sin_2theta - sin_theta;
            cos_4theta = 2.0*cos_theta*cos_3theta - cos_2theta;
            sin_4theta = 2.0*cos_theta*sin_3theta - sin_2theta;

            cos_3phi = 2.0*cos_phi*cos_2phi - cos_phi;
            sin_3phi = 2.0*cos_phi*sin_2phi - sin_phi;
            cos_4phi = 2.0*cos_phi*cos_3phi - cos_2phi;
            sin_4phi = 2.0*cos_phi*sin_3phi - sin_2phi;
        }
        // 如果 L >= 4，才编译 4 倍角
        if constexpr (L >= 4) {
            // theta方向继续递推到6theta (3theta,4theta已在L>=3分支算过)
            cos_5theta = 2.0*cos_theta*cos_4theta - cos_3theta;
            sin_5theta = 2.0*cos_theta*sin_4theta - sin_3theta;
            cos_6theta = 2.0*cos_theta*cos_5theta - cos_4theta;
            sin_6theta = 2.0*cos_theta*sin_5theta - sin_4theta;

            cos_5phi = 2.0*cos_phi*cos_4phi - cos_3phi;
            sin_5phi = 2.0*cos_phi*sin_4phi - sin_3phi;
        }
        if constexpr (L >= 6) {
            cos_6phi = 2.0*cos_phi*cos_5phi - cos_4phi;
            sin_6phi = 2.0*cos_phi*sin_5phi - sin_4phi;
        }

        
        if constexpr (L == 3) {
            // 当编译指定该模板为 <3> 时，编译器在这一步会直接盲切到 L3 函数。
            // 此时 L==6 的分支、以及计算 q6 所需的其他高阶 sin_5theta 变量，
            // 会被编译器判定为“死代码”彻底移除。最终生成的 GPU 二进制指令纯净无污染。
            compute_dYlmdx_gradient_L3(
                r,
                cos_theta, sin_theta, cos_phi, sin_phi,
                cos_2theta, sin_2theta, cos_2phi, sin_2phi,
                cos_3phi, sin_3phi, sin_4phi,
                cos_4theta, sin_4theta, 
                &local_dYlmdx
            );
        } else if constexpr (L == 4) {
            // 针对计算 q4，这里在前面额外多算两个高阶级联分量即可
            compute_dYlmdx_gradient_L4(
                r,
                cos_theta, sin_theta, cos_phi, sin_phi,
                cos_2theta, sin_2theta, cos_2phi, sin_2phi,
                cos_3theta, sin_3theta, cos_3phi, sin_3phi,
                cos_4theta, sin_4theta, cos_4phi, sin_4phi,
                cos_5theta, sin_5theta, cos_5phi, sin_5phi,
                cos_6theta, sin_6theta,
                &local_dYlmdx
            );
        } else if constexpr (L == 6) {
            // 针对计算 q6，这里在前面额外多算两个高阶级联分量即可
            compute_dYlmdx_gradient_L6(
                r,
                cos_theta, sin_theta, cos_phi, sin_phi,
                cos_2theta, sin_2theta, cos_2phi, sin_2phi,
                cos_3theta, sin_3theta, cos_3phi, sin_3phi,
                cos_4theta, sin_4theta, cos_4phi, sin_4phi,
                cos_5theta, sin_5theta, cos_5phi, sin_5phi,
                cos_6theta, sin_6theta, cos_6phi, sin_6phi,
                &local_dYlmdx
            );
        }

        // // d_dYlm_dr[c_atom_calctag*lm_size*3 + neigh_atom*lm_size*3] = local_dYlmdx;
        // int target_offset = c_atom_calctag*lm_size*3 + neigh_atom*lm_size*3;
        // // 由于并不确定是否基地址对齐，我们决定最终还是采用memcpy
        // // *reinterpret_cast<D_Ylm_Layout<L>*>(&d_dYlm_dr[target_offset])[0] = local_dYlmdx;
        // cuda::std::memcpy(&d_dYlm_dr[target_offset], &local_dYlmdx, sizeof(MetaD_zqc::D_Ylm_Layout<L>));

        
        // 第一项要求获得d_Ylm
        double dcvdrjk_fork[3] = {0.0, 0.0, 0.0}; // 栈上数组初始化
        double Ylm[lm_size] = {0.0};
        if constexpr (L == 3) {
            // 当编译指定该模板为 <3> 时，编译器在这一步会直接盲切到 L3 函数。
            // 此时 L==6 的分支、以及计算 q6 所需的其他高阶 sin_5theta 变量，
            // 会被编译器判定为“死代码”彻底移除。最终生成的 GPU 二进制指令纯净无污染。
            compute_Ylm_forward_L3(
                cos_theta, sin_theta, cos_phi, sin_phi,
                cos_2theta, sin_2theta, cos_2phi, sin_2phi,
                cos_3theta, sin_3theta, cos_3phi, sin_3phi,
                Ylm
            );
        } else if constexpr (L == 4) {
            // 针对计算 q4，这里在前面额外多算两个高阶级联分量即可
            compute_Ylm_forward_L4(
                cos_theta, sin_theta, cos_phi, sin_phi,
                cos_2phi, sin_2phi,
                cos_3phi, sin_3phi,
                cos_4phi, sin_4phi,
                Ylm
            );
        } else if constexpr (L == 6) {
            // 针对计算 q6，这里在前面额外多算两个高阶级联分量即可
            compute_Ylm_forward_L6(
                cos_theta, sin_theta, cos_phi, sin_phi,
                cos_2theta, sin_2theta, cos_2phi, sin_2phi,
                cos_3phi, sin_3phi,
                cos_4theta, sin_4theta, cos_4phi, sin_4phi,
                cos_5phi, sin_5phi,
                cos_6theta, sin_6theta, cos_6phi, sin_6phi,
                Ylm
            );
        }
        double* dYlmdx_flat = reinterpret_cast<double*>(&local_dYlmdx);
        double prefix1 = sw_df_r(r)/r;
        double sum_sigprefix =prefix1*(prefix_j[0]*(Ylm[0]-qlm[0])+ prefix_j[1]*(Ylm[1]-qlm[1]));
        // double prefix2 = prefix_j*r_weight;
        double sum_dY_prefix[3] = {0.0};
        #pragma unroll
        for(int direct=0; direct<3; direct++){
            sum_dY_prefix[direct] = r_weight*(prefix_j[0]*dYlmdx_flat[direct*2 + 0]
                                                 + prefix_j[1]*dYlmdx_flat[direct*2 + 1]);
        }
        #pragma unroll
        for(int m=1;m<=L;m++){
            sum_sigprefix += prefix1*2.0*(prefix_j[2*m + 0]*(Ylm[2*m + 0]-qlm[2*m + 0])
                                        + prefix_j[2*m + 1]*(Ylm[2*m + 1]-qlm[2*m + 1]));
            #pragma unroll
            for(int direct=0; direct<3; direct++){
                sum_dY_prefix[direct] += r_weight*2.0*(prefix_j[2*m + 0]*dYlmdx_flat[m*3*2 + direct*2 + 0]
                                                     + prefix_j[2*m + 1]*dYlmdx_flat[m*3*2 + direct*2 + 1]);
            }
        }
        
        dcvdrjk_fork[0] = delt_x*sum_sigprefix + sum_dY_prefix[0];
        dcvdrjk_fork[1] = delt_y*sum_sigprefix + sum_dY_prefix[1];
        dcvdrjk_fork[2] = delt_z*sum_sigprefix + sum_dY_prefix[2];
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

    #undef sw_f_r
    #undef sw_f_LQ
    #undef sw_df_r
    #undef sw_df_LQ
}
