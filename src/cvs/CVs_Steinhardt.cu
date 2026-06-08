#include <cstring>
#include <algorithm>
#include <sstream>
#include <iomanip>

#include <cuda_runtime.h>

#include "lammps.h"
#include "lammpsplugin.h"
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
#include "zqc_switch_function.h"

using namespace LAMMPS_NS;

MetaD_zqc::CV* MetaD_zqc::Steinhardt::create(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::FixMetadynamics *Fixmetad, 
                                            int narg, char **arg, int &i, FILE *f_check){
    DEBUG_LOG("In STEINH settings");
    printf("++++++++++++++++++++++++++++++im in STEINH settings, narg=%d, current arg is %s\n", narg, arg[i]);
    LAMMPS_NS::Error *error = lmp->error;

    std::string cal_name = arg[i];

    MetaD_zqc::SteinhardtRequest req;
    req.cal_name = cal_name;
    // 原子环境分析-初始设置
    // Usage: STEINH <Q/L> <4/6/8/12> <group>
    ERR_COND(i + 3 >= narg, "Error: STEINH command requires \"STEINH <Q/L> <4/6/8/12> <group> \".");
    req.Q_type_str = arg[i+1];
    req.Q_num   = utils::inumeric(FLERR, arg[i+2], false, lmp);
    req.group_name = arg[i+3];
    req.group_id = lmp->group->find(req.group_name);
    ERR_COND(req.group_id == -1, "Error: Steinhardt group name %s not found.", req.group_name);
    //参数有效性
    ERR_COND((req.Q_num != 3 && req.Q_num != 4 && req.Q_num != 6 && req.Q_num != 8 && req.Q_num != 12),"Error: Steinhardt order L must be 3, 4, 6, 8, or 12.");
    ERR_COND((strcmp(req.Q_type_str, "Q") != 0 && strcmp(req.Q_type_str, "L") != 0), "Error: Steinhardt type must be 'Q' (local) or 'L' (global).");
    // 进阶设置
    // default values
    req.cutoff_r = 4.0;
    req.cutoff_Natoms = 12;
    req.d_block_size = 128;
    int iarg=4 + i;
    while (iarg < narg) {
        if (strcmp(arg[iarg], "cutoff_r") == 0) {
            ERR_COND((iarg + 1 >= narg) ,"Error: \'cutoff_r\' keyword requires a value");
            req.cutoff_r = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
            iarg += 2;
        } else if (strcmp(arg[iarg], "cutoff_Natoms") == 0) {
            ERR_COND((iarg + 1 >= narg), "Error: \'cutoff_Natoms\' keyword requires an integer");
            req.cutoff_Natoms = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
            iarg += 2;
        } else if (strcmp(arg[iarg], "d_block_size") == 0) {
            ERR_COND((iarg + 1 >= narg), "Error: \'d_block_size\' keyword requires an integer");
            req.d_block_size = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
            ERR_COND(req.d_block_size <= 0, "Error: \'d_block_size\' must be > 0");
            iarg += 2;
        // } else if (strcmp(arg[iarg], "SW_FUNC") == 0) {
        //     ERR_COND((iarg + 1 >= narg), "Error: \'SW_FUNC\' keyword requires a value");
        //     req.SW_FUNC = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
        //     iarg += 2;
        } else {
            break;
        }
    }
    LOG("Logging: set STEINH as Q_type_str=%s Q_num=%d group_name=%s cutoff_r=%f cutoff_Natoms=%d d_block_size=%d.",
                        req.Q_type_str, req.Q_num, req.group_name, req.cutoff_r, req.cutoff_Natoms, req.d_block_size);

    // 请求lammps提供全近邻列表（full neighbor list），并设置ghost原子范围为 cutoff_r + neighbor->skin
    NeighRequest *full_request;
    full_request = lmp->neighbor->add_request(Fixmetad, NeighConst::REQ_FULL);
    full_request->set_id(2);
    double temp_neigh_cutoff;
    if (strcmp(req.Q_type_str, "Q") == 0){
        temp_neigh_cutoff = (req.cutoff_r + lmp->neighbor->skin);
    } else if (strcmp(req.Q_type_str, "L") == 0){
        temp_neigh_cutoff = (2.0 * req.cutoff_r + lmp->neighbor->skin);
    }
    if (lmp->comm->get_comm_cutoff() < temp_neigh_cutoff) {
        lmp->comm->cutghostuser = temp_neigh_cutoff;
        LOG_COND(lmp->comm->me == 0, "Increasing communication cutoff to %g for GPU pair style",
                  lmp->comm->cutghostuser);
    }

    // steinh_requests.push_back(req);
    // // 创建 CV 对象
    // TODO: 需要处理相同envs的合并问题
    MetaD_zqc::Steinhardt_env *temp_env = MetaD_zqc::Steinhardt_env::get_or_create(lmp, 
                            f_check, Fixmetad, req.group_id, req.cutoff_r, req.cutoff_Natoms);
    DEBUG_LOG("Steinhardt_env is %p", temp_env);
    std::string env_setNum = temp_env->get_env_key();
    i = iarg;
    return MetaD_zqc::create_steinhardt_cv(lmp, Fixmetad, f_check, 
                            env_setNum, req.group_id, req.Q_num, temp_env, req.Q_type_str,
                            req.cutoff_r, req.cutoff_Natoms, req.d_block_size);
}

MetaD_zqc::Steinhardt* MetaD_zqc::create_steinhardt_cv(LAMMPS_NS::LAMMPS *lmp,
                                LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check,
                                std::string env_setNum, int group_id, int Q_num,
                                MetaD_zqc::Steinhardt_env* my_env,
                                char *Q_type_str, double cutoff_r, int cutoff_Natoms, 
                                int d_block_size)
{
    if (strcmp(Q_type_str, "Q") == 0){
        if (Q_num==3){
            return new MetaD_zqc::STEIN_QL<3>(lmp, Fixmetad, f_check, env_setNum, group_id, Q_num, my_env, d_block_size);
        } else if (Q_num==4){
            return new MetaD_zqc::STEIN_QL<4>(lmp, Fixmetad, f_check, env_setNum, group_id, Q_num, my_env, d_block_size);
        } else if (Q_num==6){
            return new MetaD_zqc::STEIN_QL<6>(lmp, Fixmetad, f_check, env_setNum, group_id, Q_num, my_env, d_block_size);
        }
    } else if (strcmp(Q_type_str,"L") == 0){
        if (Q_num==4){
            // return new MetaD_zqc::STEIN_LocalQL<4>(lmp, f_check, group_id, cutoff_r, cutoff_Natoms, d_block_size);
        } else if (Q_num==6){
            // return new STEIN_LQ6(lmp, f_check, group_id, cutoff_r, cutoff_Natoms, d_block_size);
        }
    }
    return nullptr;
}

std::map<std::string, MetaD_zqc::Steinhardt_env*> MetaD_zqc::Steinhardt_env::env_pool;

MetaD_zqc::Steinhardt_env* MetaD_zqc::Steinhardt_env::get_or_create(LAMMPS_NS::LAMMPS *lmp, FILE *f_check,
                                            LAMMPS_NS::FixMetadynamics *Fixmetad,
                                            int group_id, double cutoff_r, int cutoff_Natoms) {
    // 1. generate a unique key for the environment based on its parameters
    std::ostringstream oss;
    oss << group_id << "_" << cutoff_r << "_" << cutoff_Natoms;
    std::string key = oss.str(); // 比如 cutoff_r=5.5 时，Key 为 "1_5.5_128"
    // 2. check if the environment already exist in the pool
    if (!(env_pool.count(key))) {
        // 3. new environment and store it in the pool if not exist
        MetaD_zqc::Steinhardt_env *new_env = new Steinhardt_env(lmp, f_check, Fixmetad, 
                                                    group_id, cutoff_r, cutoff_Natoms);
        env_pool[key] = new_env; // store the new environment in the pool
    }
    env_pool[key]->register_env(); // increase reference count
    return env_pool[key]; // if exits, return the existing environment
}


std::string MetaD_zqc::Steinhardt_env::get_env_key(){
    std::string key = std::to_string(this->group_id) + "_" + std::to_string(this->cutoff_r) + "_" + std::to_string(this->cutoff_Natoms);
    return key;
}

MetaD_zqc::Steinhardt_env::Steinhardt_env(LAMMPS_NS::LAMMPS *lmp, FILE *f_check,
             LAMMPS_NS::FixMetadynamics *Fixmetad, int group_id,
             double cutoff_r, int cutoff_Natoms)
    : lmp(lmp),
      f_check(f_check),
      Fixmetad(Fixmetad),
      group_id(group_id),
      cutoff_r(cutoff_r),
      cutoff_Natoms(cutoff_Natoms)
{
    error = lmp->error;

    pbc_x = (lmp->domain->xperiodic == 1);
    pbc_y = (lmp->domain->yperiodic == 1);
    pbc_z = (lmp->domain->zperiodic == 1);
    // 这里可以添加一些初始化代码，例如分配内存、设置默认值等
    DEBUG_LOG("Steinhardt_env initialized with cutoff_r=%g and cutoff_Natoms=%d", cutoff_r, cutoff_Natoms);

    // const char *group_name = arg[1];
    groupbit = lmp->group->bitmask[group_id]; // 关键：存储原子组位掩码
    init_flag = false;
    
    // group_dminneigh = new double [2]; //inintial
    // neigh_in_cutoff_r = new int [2]; //inintial
    // neigh_both_in_r_N = new int [2]; //inintial
    lmp->memory->create(h_group_numneigh, 0, "metad:STEIN_QL:h_group_numneigh");
    lmp->memory->create(h_x_flat, 0, "metad:STEIN_QL:h_x_flat");
    lmp->memory->create(h_group_indices, 0, "metad:STEIN_QL:h_group_indices");
    lmp->memory->create(h_firstneigh_ptrs, 0, "metad:STEIN_QL:h_firstneigh_ptrs");
    lmp->memory->create(group_dminneigh, 0, "metad:STEIN_QL:group_dminneigh");
    lmp->memory->create(neigh_in_cutoff_r, 0, "metad:STEIN_QL:neigh_in_cutoff_r");
    lmp->memory->create(neigh_both_in_r_N, 0, "metad:STEIN_QL:neigh_both_in_r_N");
    lmp->memory->create(calculated_numneigh, 0, "metad:STEIN_QL:calculated_numneigh");

    // comment name
    d_group_numneigh.set_name("d_group_numneigh");
    d_x_flat.set_name("d_x_flat");
    d_mask.set_name( "d_mask");
    d_group_indices.set_name("d_group_indices");
    d_firstneigh_ptrs.set_name("d_firstneigh_ptrs");
    d_group_dminneigh.set_name("d_group_dminneigh");
    d_neigh_in_cutoff_r.set_name("d_neigh_in_cutoff_r");
    d_neigh_both_in_r_N.set_name("d_neigh_both_in_r_N");
    d_calculated_numneigh.set_name("d_calculated_numneigh");
}

template <int L>
MetaD_zqc::STEIN_QL<L>::STEIN_QL(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check, 
                             std::string env_setNum, int group_id, int stein_l, 
                             MetaD_zqc::Steinhardt_env* my_env,
                             int d_block_size)
                        : Steinhardt(lmp, f_check),
                            Fixmetad(Fixmetad),
                            env_setNum(env_setNum),
                        //   group_id(group_id),
                            stein_l(stein_l),
                            my_env(my_env),
                            d_block_size(d_block_size){
    // my_averager = new MetaD_zqc::CUBAverager();
    my_averager = new MetaD_zqc::KahanAverager();
    num_elements = 2*(L+1); // Qlm needs 2*(l+1)
    DEBUG_LOG("Logging: New a Stein_Q%d file, will generate %d lines in GPU,\n     with cutoff_r=%g, cutoff_Natoms=%d",
                stein_l,d_block_size, my_env->cutoff_r, my_env->cutoff_Natoms);
    my_env->d_block_size = d_block_size;
    // gpu device settings
    cudaGetLastError(); // clear history error
    GPU_number = 0;
    cudaGetDevice(&GPU_number);
    DEBUG_LOG("GPU_number is %d",GPU_number);
    my_env->GPU_number = GPU_number;
    
    all_count = lmp->group->count(my_env->group_id);
    DEBUG_LOG("all_count = %lld", (long long)all_count);

    // Q_per_atoms_value = new double [2]; //inintial
    // stein_q = nullptr;
    lmp->memory->create(stein_q, 0, "metad:STEIN_QL:cv_bound");
    int Threads_own_atoms = lmp->atom->nlocal;
    lmp->memory->grow(stein_q, Threads_own_atoms, "metad:STEIN_QL:cv_bound");

    error = lmp->error;

    
    // comment name
    d_stein_ql.set_name("d_stein_ql");
    d_stein_Ylm.set_name("d_stein_Ylm");
    d_dYlm_dr.set_name( "d_dYlm_dr");
    d_dcvdx.set_name("d_dcvdx");
    d_stein_qlm.set_name("d_stein_qlm");
    d_stein_LQlm.set_name("d_stein_LQlm");
}

template <int L>
MetaD_zqc::STEIN_QL<L>::~STEIN_QL(){
    // delete[] stein_q;
    lmp->memory->destroy(stein_q);
    // delete[] h_stein_Ylm;
    // SAFE_CUDA_FREE(d_stein_Ylm.ptr);
    // delete[] h_dYlm_dr;
    lmp->memory->destroy(h_dYlm_dr);
    // SAFE_CUDA_FREE(d_dYlm_dr.ptr);
    // delete[] h_dcvdx;
    lmp->memory->destroy(h_dcvdx);
    // SAFE_CUDA_FREE(d_dcvdx.ptr);
    // delete[] h_stein_qlm;
    lmp->memory->destroy(h_stein_qlm);
    // SAFE_CUDA_FREE(d_stein_qlm.ptr);
    lmp->memory->destroy(h_stein_LQlm);
    // release all alloc
    // the GpuBuffer will automatically release its memory, 
    // so we don't need to manually free it here
}

MetaD_zqc::Steinhardt_env::~Steinhardt_env(){
    atoms = nullptr;
    // release all alloc
    nlist = nullptr;
    // delete[] h_group_numneigh;
    lmp->memory->destroy(h_group_numneigh);
    // SAFE_CUDA_FREE(d_group_numneigh.ptr);
    numneigh = nullptr;
    firstneigh = nullptr;
    mask = nullptr;
    // delete[] h_x_flat;
    lmp->memory->destroy(h_x_flat);
    // SAFE_CUDA_FREE(d_x_flat.ptr);
    // SAFE_CUDA_FREE(d_mask.ptr);
    // delete[] h_group_indices;
    lmp->memory->destroy(h_group_indices);
    // SAFE_CUDA_FREE(d_group_indices.ptr);
    // delete[] h_firstneigh_ptrs;
    lmp->memory->destroy(h_firstneigh_ptrs);
    // SAFE_CUDA_FREE(d_firstneigh_ptrs.ptr);
    // delete[] group_dminneigh;
    lmp->memory->destroy(group_dminneigh);
    // SAFE_CUDA_FREE(d_group_dminneigh.ptr);
    // delete[] neigh_in_cutoff_r;
    lmp->memory->destroy(neigh_in_cutoff_r);
    // SAFE_CUDA_FREE(d_neigh_in_cutoff_r.ptr);
    // delete[] neigh_both_in_r_N;
    lmp->memory->destroy(neigh_both_in_r_N);
    // SAFE_CUDA_FREE(d_neigh_both_in_r_N.ptr);
    // delete[] calculated_numneigh;
    lmp->memory->destroy(calculated_numneigh);
    // SAFE_CUDA_FREE(d_calculated_numneigh.ptr);
    // delete[] Q_per_atoms_value;
    // the GpuBuffer will automatically release its memory, 
    // so we don't need to manually free it here
}

void MetaD_zqc::Steinhardt_env::refresh_lmpbox(){
    // clear the h_group_indices
    atom = lmp->atom;
    mask = (atom)->mask;     // 原子组掩码

    // delete[] h_group_indices;
    // h_group_indices = nullptr;
    // DEBUG_LOG("free h_group_indices");
    // h_group_indices = new int [((atom)->nlocal)];

    lmp->memory->grow(h_group_indices, ((atom)->nlocal), "STEIN_QL:h_group_indices");
    // group_count = how many aim atoms in local
    last_group_count = group_count;
    group_count = 0; // 当前local中有
    for (int i = 0; i < (atom)->nlocal; i++) {
        if ((mask)[i] & (groupbit)){
            (h_group_indices)[(group_count)] = i; // record local index
            (group_count)++;
            DEBUG_LOG("group_count=%lld",((long long)group_count));
        }
    }
    // printf("group_count=%d", group_count);

    // SAFE_CUDA_FREE((d_mask));
    // SAFE_CUDA_MALLOC(&(d_mask), (group_count)*sizeof(int), f_check);
    d_mask.grow_to(((atom)->nlocal+(atom)->nghost), f_check, __FILE__, __LINE__);
    SAFE_CUDA_MEMCPY((d_mask.ptr),(mask),(((atom)->nlocal+(atom)->nghost))*sizeof(int),cudaMemcpyHostToDevice,f_check);

    // set up nvidia thread number
    block_num = ((group_count) + d_block_size - 1)/d_block_size;
    N = d_block_size*block_num;
    // LOG_COND(((group_count)<(cutoff_Natoms)),"Warning: group_count(%lld) < cutoff_Natoms(%lld), please check your system !",(long long)group_count, (long long)cutoff_Natoms);
    LOG_COND((((box_x)<2*(cutoff_r))||((box_y)<2*(cutoff_r))||((box_z)<2*(cutoff_r))),"Warning: box < cutoff_r, please check your system !");
}

void MetaD_zqc::Steinhardt_env::get_env(){
    DEBUG_LOG("im in get_env, current step is %lld, last_update_step is %lld", (long long)lmp->update->ntimestep, (long long)this->last_update_step);
    // if (lmp->update->ntimestep == this->last_update_step){
    //     return;
    // }
    size_t datalen = 0;
    atom = lmp->atom;
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
        // =========================================================================
        // neighbour list and its copy to devise
        // h_group_indices / d_group_indices: where the group atoms in locals' tag
        // =========================================================================
        DEBUG_LOG("cutoff_Natoms is %d",cutoff_Natoms);
        DEBUG_LOG("cutoff_r is %f",cutoff_r);
        DEBUG_LOG("group_count is %d",group_count);
        // DEBUG_LOG("lastcall = %d", lmp->neighbor->lastcall);
        // int *d_group_indices;
        // SAFE_CUDA_FREE(d_group_indices);
        // SAFE_CUDA_MALLOC(&d_group_indices, (group_count)*sizeof(int), f_check);
        d_group_indices.grow_to(group_count, f_check, __FILE__, __LINE__);
        SAFE_CUDA_MEMCPY(d_group_indices.ptr,h_group_indices,(group_count)*sizeof(int),cudaMemcpyHostToDevice,f_check);
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
        // LAMMPS_NS::tagint *h_group_numneigh = new LAMMPS_NS::tagint[group_count + 1];
        // LAMMPS_NS::tagint *d_group_numneigh;
        datalen = group_count + 1;
        lmp->memory->grow(h_group_numneigh, datalen, "STEIN_QL:h_group_numneigh");
        d_group_numneigh.grow_to(datalen, f_check, __FILE__, __LINE__);
        DEBUG_LOG_COND((h_group_numneigh == NULL),"h_group_numneigh list not initialized");
        // 3. 逐原子拷贝邻居列表数据到GPU
        DEBUG_LOG("group_count=%d" ,group_count);
        h_group_numneigh[0] = 0;
        for (int gr_i = 0; gr_i < group_count; gr_i++) {
            int i = h_group_indices[gr_i]; // 获取原子索引
            int jnum = numneigh[i]; // 邻居数量
            h_group_numneigh[gr_i+1] = h_group_numneigh[gr_i] + jnum;
            DEBUG_LOG("gr_i=%d, tag=%d, jnum=%d, sum=%d", gr_i, i,jnum,h_group_numneigh[gr_i+1]);
        }
        SAFE_CUDA_MEMCPY(d_group_numneigh.ptr,h_group_numneigh,(group_count + 1)*sizeof(LAMMPS_NS::tagint),cudaMemcpyHostToDevice,f_check);
        // =========================================================================
        // h_group_numneigh / d_group_numneigh :
        //      flatten index of the neighbour list. such as we have 20 neighbour
        //      for atom 1, then the list will be : [0, 20, ...]
        // h_firstneigh_ptrs / d_firstneigh_ptrs :
        //      flatten neighbour list
        // =========================================================================
        /* delete[] h_firstneigh_ptrs;
        h_firstneigh_ptrs = nullptr; */
        // int *h_firstneigh_ptrs = new int [h_group_numneigh[group_count]];
        // int *d_firstneigh_ptrs; // 设备端二级指针
        // h_firstneigh_ptrs = new int [h_group_numneigh[group_count]];
        lmp->memory->grow(h_firstneigh_ptrs, h_group_numneigh[group_count], "STEIN_QL:h_firstneigh_ptrs");
        LAMMPS_NS::tagint ba_i;
        LAMMPS_NS::tagint nnumber;
        int i;
        // SAFE_CUDA_FREE(d_firstneigh_ptrs);
        // SAFE_CUDA_MALLOC(&d_firstneigh_ptrs, (h_group_numneigh[group_count]) * sizeof(int),f_check); // 分配设备端指针数组
        d_firstneigh_ptrs.grow_to(h_group_numneigh[group_count], f_check, __FILE__, __LINE__);
        DEBUG_LOG("generate d_firstneigh_ptrs, h_group_numneigh[group_count + 1]=%d",h_group_numneigh[group_count]);
        for (int gr_i = 0; gr_i < group_count; gr_i++) {
            i = h_group_indices[gr_i]; // 获取原子索引
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
        SAFE_CUDA_MEMCPY(d_firstneigh_ptrs.ptr,h_firstneigh_ptrs,
            (h_group_numneigh[group_count]) * sizeof(int),cudaMemcpyHostToDevice,f_check);
        DEBUG_LOG_COND((d_firstneigh_ptrs.ptr == NULL),"d_firstneigh_ptrs list not initialized");
        DEBUG_LOG("d_firstneigh_ptrs list %d %d %d" ,h_firstneigh_ptrs[1],h_firstneigh_ptrs[2],h_firstneigh_ptrs[3]);
        DEBUG_LOG("generate end d_firstneigh_ptrs");
        if (init_flag) {init_flag = true;}
    }
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
    d_x_flat.grow_to((atom->nlocal + atom->nghost) * 3, f_check, __FILE__, __LINE__);
    SAFE_CUDA_MEMCPY(d_x_flat.ptr,h_x_flat,((atom->nlocal + atom->nghost) * 3)*sizeof(double),cudaMemcpyHostToDevice, f_check);
    // check the pointer
    // DEBUG_LOG("alloc h_x,h_tag.....");
    DEBUG_LOG_COND((h_x == NULL),"h_x list not initialized");
    DEBUG_LOG_COND((h_x_flat == NULL),"h_x_flat list not initialized");
    DEBUG_LOG_COND((d_x_flat.ptr == NULL),"d_x_flat list not initialized");
    DEBUG_LOG("d_x_flat Allocated at: %p", d_x_flat.ptr);
    cudaDeviceSynchronize(); // waiting memory

    // =========================================================================
    // create output address
    // d_group_dminneigh : (dx, dy, dz, r2) * pairs
    // d_neigh_in_cutoff_r : neighbour atoms that satisfied cutoff_r
    // d_neigh_both_in_r_N : neighbour atoms that satisfied both cutoff_r and N
    // =========================================================================
    DEBUG_LOG("release gpu");
    // SAFE_CUDA_FREE(d_neigh_both_in_r_N);
    // SAFE_CUDA_FREE(d_group_dminneigh);
    // SAFE_CUDA_FREE(d_neigh_in_cutoff_r);
    // SAFE_CUDA_FREE(d_calculated_numneigh);
    DEBUG_LOG("release end");
    // double *d_group_dminneigh;
    // SAFE_CUDA_MALLOC(&d_group_dminneigh, (N*cutoff_Natoms*4)*sizeof(double),f_check);
    d_group_dminneigh.grow_to(N*cutoff_Natoms*4, f_check, __FILE__, __LINE__);
    // int *d_neigh_in_cutoff_r;
    // SAFE_CUDA_MALLOC(&d_neigh_in_cutoff_r, (N*4)*sizeof(int),f_check);
    d_neigh_in_cutoff_r.grow_to(N*4, f_check, __FILE__, __LINE__);
    // int *d_neigh_both_in_r_N;
    // SAFE_CUDA_MALLOC(&d_neigh_both_in_r_N, (N)*sizeof(int),f_check);
    int Threads_own_atoms = lmp->atom->nlocal+lmp->atom->nghost;
    Threads_own_atoms = (Threads_own_atoms > N) ? Threads_own_atoms : N;
    d_neigh_both_in_r_N.grow_to(Threads_own_atoms, f_check, __FILE__, __LINE__);
    cudaMemset(d_neigh_both_in_r_N.ptr, 0, Threads_own_atoms);
    // double *d_calculated_numneigh;
    // SAFE_CUDA_MALLOC(&d_calculated_numneigh, (N*cutoff_Natoms*sizeof(LAMMPS_NS::tagint)), f_check);
    d_calculated_numneigh.grow_to(N*cutoff_Natoms, f_check, __FILE__, __LINE__);

    // box_x=box_y=box_z=40.0;
    DEBUG_LOG("box_lim x:%f y:%f z:%f max:%f" ,box_x,box_y,box_z,box_x+box_y+box_z );
    DEBUG_LOG("neigh finding .......");
    DEBUG_LOG("i will start a kernel");
    // kernel function will run
    cudaError_t launchErr = cudaGetLastError();
    cudaDeviceSynchronize(); // waiting memory
    double cutoff_rsq = cutoff_r*cutoff_r;

    // cudaDeviceSynchronize(); //catch kernel done
    launchErr = cudaGetLastError();
    get_environment_Steinhardt_Q<<<block_num,d_block_size>>>
      ( cutoff_Natoms, cutoff_rsq, box_x, box_y, box_z, 
      group_count, d_group_indices.ptr, d_group_numneigh.ptr, d_firstneigh_ptrs.ptr, d_x_flat.ptr,
      d_group_dminneigh.ptr, d_neigh_in_cutoff_r.ptr, d_neigh_both_in_r_N.ptr, d_calculated_numneigh.ptr) ;
    DEBUG_LOG("env refresh out, kernel launched");
    // double *h_group_dminneigh = new double [group_count*cutoff_Natoms*4];
    // int *h_neigh_in_cutoff_r = new int [group_count];
    // int *h_neigh_both_in_r_N = new int [group_count];
    // int atomsnumber = (atom->nlocal + atom->nghost);
    // get_environment_temp
    //   ( cutoff_Natoms, cutoff_rsq, box_x, box_y, box_z, 
    //   group_count, h_group_indices, h_group_numneigh, h_firstneigh_ptrs, h_x_flat,
    //   h_group_dminneigh, h_neigh_in_cutoff_r, h_neigh_both_in_r_N,atomsnumber) ;
    cudaDeviceSynchronize(); //catch kernel done
    // cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        // 输出到您的文件
        fprintf(f_check, "CUDA Kernel launch failed: %s\n", cudaGetErrorString(launchErr));
        fflush(f_check);
        // 尝试输出到标准错误流 (确保在 LAMMPS 终端可见)
        fprintf(stderr, "LAMMPS CUDA ERROR: Kernel launch failed: %s\n", cudaGetErrorString(launchErr));
        error->all(FLERR, "Kernel launch failed. Check output for detailed CUDA error.");
    }
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        fprintf(f_check, "Kernel execution error: %s\n", cudaGetErrorString(syncErr));
        error->all(FLERR, "Kernel execution error\n");
    }
    DEBUG_LOG("im out");
    DEBUG_LOG("neigh find finished");


    // return the array for neigh
    DEBUG_LOG("copy result array to cpu: group_dminneigh, neigh_in_cutoff_r, neigh_both_in_r_N");
    DEBUG_LOG_COND((group_dminneigh == NULL),"group_dminneigh list not initialized");
    DEBUG_LOG_COND((neigh_in_cutoff_r == NULL),"group_dminneigh list not initialized");
    DEBUG_LOG_COND((neigh_both_in_r_N == NULL),"group_dminneigh list not initialized");
    // delete[] group_dminneigh;
    // group_dminneigh = new double [group_count*cutoff_Natoms*4];
    lmp->memory->grow(group_dminneigh, (group_count*cutoff_Natoms*4), "STEIN_QL:group_dminneigh");
    SAFE_CUDA_MEMCPY(group_dminneigh, d_group_dminneigh.ptr,
      (group_count*cutoff_Natoms*4) * sizeof(double), cudaMemcpyDeviceToHost,f_check);
    // delete[] neigh_in_cutoff_r;
    // neigh_in_cutoff_r = new int [group_count];
    lmp->memory->grow(neigh_in_cutoff_r, (group_count), "STEIN_QL:neigh_in_cutoff_r");
    SAFE_CUDA_MEMCPY(neigh_in_cutoff_r, d_neigh_in_cutoff_r.ptr,
      (group_count) * sizeof(int), cudaMemcpyDeviceToHost,f_check);
    // delete[] neigh_both_in_r_N;
    // neigh_both_in_r_N = new int [group_count];
    lmp->memory->grow(neigh_both_in_r_N, (Threads_own_atoms), "STEIN_QL:neigh_both_in_r_N");
    SAFE_CUDA_MEMCPY(neigh_both_in_r_N, d_neigh_both_in_r_N.ptr,
      (Threads_own_atoms) * sizeof(int), cudaMemcpyDeviceToHost,f_check);
    // delete[] calculated_numneigh;
    // calculated_numneigh = new LAMMPS_NS::tagint [group_count*cutoff_Natoms];
    lmp->memory->grow(calculated_numneigh, (group_count*cutoff_Natoms), "STEIN_QL:calculated_numneigh");
    SAFE_CUDA_MEMCPY(calculated_numneigh, d_calculated_numneigh.ptr,
      (group_count*cutoff_Natoms) * sizeof(LAMMPS_NS::tagint), cudaMemcpyDeviceToHost,f_check);
    cudaDeviceSynchronize(); //catch kernel done
    DEBUG_LOG("copy end");
    DEBUG_LOG("group_dminneigh Allocated at: %p", group_dminneigh);
    this->last_update_step = lmp->update->ntimestep;

    
    // // 输出邻居
    // DEBUG_RUN(for (int ii=0; ii<group_count; ii++){
    //     for (int jj=0; jj<1; jj++){
    //         fprintf(f_check, "c_atom_idx=%lld,%lld,%lld : Nx:%f Ny:%f Nz:%f r2:%f\n", 
    //                 (long long)lmp->atom->tag[h_group_indices[ii]],
    //                 (long long)neigh_in_cutoff_r[ii],
    //                 (long long)neigh_both_in_r_N[ii],
    //                 group_dminneigh[ii*cutoff_Natoms*4 + jj*4 + 0],
    //                 group_dminneigh[ii*cutoff_Natoms*4 + jj*4 + 1],
    //                 group_dminneigh[ii*cutoff_Natoms*4 + jj*4 + 2],
    //                 group_dminneigh[ii*cutoff_Natoms*4 + jj*4 + 3]);
    //     }
    // });
}





template <int L>
void MetaD_zqc::STEIN_QL<L>::environment(){
    DEBUG_LOG("last_update_step is %lld in %d, group_count=%d", (long long)my_env->last_update_step, L, my_env->group_count);
    if (lmp->update->ntimestep > my_env->last_update_step){
        my_env->get_env();
    }
    // DEBUG_LOG("environment function in, env_setNum is %s, get_env done",env_setNum);
    DEBUG_LOG("last_update_step is %lld in %d, group_count=%d", (long long)my_env->last_update_step, L, my_env->group_count);
}

template <int L>
auto MetaD_zqc::STEIN_QL<L>::set_CV_calculate(std::string func_name) -> CV_Calculation {
    // 1. 按照 "." 分割 func_name
    std::string main_func = func_name;
    std::string sub_param = "";
    
    size_t dot_pos = func_name.find('.');
    if (dot_pos != std::string::npos) {
        main_func = func_name.substr(0, dot_pos);   // 拿到 "." 前面的部分，如 "SW_FUNC"
        sub_param = func_name.substr(dot_pos + 1);  // 拿到 "." 后面的部分，如 "Fermi" 或 "Cubic"
    }


    if (main_func == "AVE") {
        return static_cast<CV_Calculation>(&STEIN_QL<L>::compute_cv_AVE);
    } else if (main_func == "LOC_AVE") {
        // return static_cast<CV_Calculation>(&STEIN_QL<L>::compute_cv_LOC_AVE);
    } else if (main_func == "SW_FUNC") {
        auto it = Fixmetad->get_switching_function(sub_param);
        if (it != nullptr) {
            // 成功让类中的成员指针指向已经构造好的 SW1 (RATIONAL 实例)
            this->my_cv_SWfunc = it;
        } else {
            // 如果脚本里写错了名字（比如写成了 SW2 却没声明），直接让 LAMMPS 报错
            ERR_COND(1, "Switching function %s used in SYMBOL but not defined in CAL!", sub_param.c_str());
        }
        return static_cast<CV_Calculation>(&STEIN_QL<L>::compute_cv_SW_FUNC);
    } else {
        ERR_COND(1, "We can't find the func %s.", main_func.c_str());
        return nullptr;
    }
}

template <int L>
auto MetaD_zqc::STEIN_QL<L>::set_CV_bias_force(std::string func_name) -> CV_BiasForce {
    // 1. 按照 "." 分割 func_name
    std::string main_func = func_name;
    std::string sub_param = "";
    
    size_t dot_pos = func_name.find('.');
    if (dot_pos != std::string::npos) {
        main_func = func_name.substr(0, dot_pos);   // 拿到 "." 前面的部分，如 "SW_FUNC"
        sub_param = func_name.substr(dot_pos + 1);  // 拿到 "." 后面的部分，如 "Fermi" 或 "Cubic"
    }

    if (main_func == "AVE") {
        return static_cast<CV_BiasForce>(&STEIN_QL<L>::bias_force_AVE);
    } else if (main_func == "LOC_AVE") {
        // return static_cast<CV_BiasForce>(&STEIN_QL<L>::bias_force_LOC_AVE);
    } else if (main_func == "SW_FUNC") {
        return static_cast<CV_BiasForce>(&STEIN_QL<L>::bias_force_SW_FUNC);
    } else {
        ERR_COND(1, "We can't find the func %s.", main_func.c_str());
        return nullptr;
    }
}

template <int L>
void MetaD_zqc::STEIN_QL<L>::base_calc(){
    compute_Q_peratoms();
}

template <int L>
void MetaD_zqc::STEIN_QL<L>::compute_Q_peratoms(){
    // =======接受邻居更新消息,进行与设备端通信===========
    if ((lmp->update->ntimestep > lmp->neighbor->lastcall)&&(lmp->update->ntimestep != 1)&&(this->init_flag)){
        DEBUG_LOG("rebuilds = %lld", (long long)lmp->neighbor->lastcall);
        DEBUG_LOG("now = %lld", (long long)lmp->update->ntimestep);
        ERR_COND(((my_env->h_group_indices) == nullptr),"h_group_indices is nullptr.");
        DEBUG_LOG("h_group_indices=%p",(my_env->h_group_indices));
    } else {
        // ===重建邻居列表后重新查找local中的目标原子=======
        if (lmp->update->ntimestep > my_env->last_update_step){
            my_env->refresh_lmpbox();
        }
        DEBUG_LOG("refresh_lmpbox done, group_count=%d",my_env->group_count);
        block_num = my_env->block_num;
        N = my_env->N;
        int Threads_own_atoms = lmp->atom->nlocal+lmp->atom->nghost;
        // stein_q for all aim atoms
        lmp->memory->grow(stein_q, Threads_own_atoms, "metad:STEIN_QL:cv_bound");
        DEBUG_LOG("d_block_size is %d, block_num is %d",d_block_size, block_num);
    }
    DEBUG_LOG("group_count=%lld",(long long)my_env->group_count);

    // 2. calculate atoms' environment
    DEBUG_LOG("environment function in, env_setNum is %s",env_setNum.c_str());
    environment();
    DEBUG_LOG("environment function out");

    // 3. calculate atoms' other things
    // steinhardt_param(Q_hybrid);
    steinhardt_param_calc(stein_q);

    DEBUG_LOG_COND((my_env->group_dminneigh == NULL),"group_dminneigh list not initialized");
    DEBUG_LOG("group_dminneigh Allocated at: %p", my_env->group_dminneigh);
    
    // 输出group中每个原子的ql值
    DEBUG_RUN(for(int c_atom=0;c_atom<my_env->group_count;c_atom++)
                {
                    DEBUG_LOG("stein_ql[%lld] = %f",(long long)c_atom,stein_q[c_atom]);
                });
    DEBUG_LOG("post_force function end");
}

template <int L>
double MetaD_zqc::STEIN_QL<L>::compute_cv_AVE(){
    DEBUG_LOG("im in compute_cv_AVE.");
    int group_count = my_env->group_count;
    int Threads_own_atoms = lmp->atom->nlocal;
    DEBUG_LOG("group_count = %d",group_count);
    double ql_ave_local=0;
    DEBUG_LOG_COND((stein_q == NULL),"stein_q list not initialized");
    if (group_count != 0) {
        my_averager->compute(Threads_own_atoms, all_count, stein_q, lmp->atom->mask, 
            my_env->groupbit, ql_ave_local);
    }
    MPI_Allreduce(&ql_ave_local, &cv_value, 1, MPI_DOUBLE, MPI_SUM, lmp->world);
    DEBUG_LOG("group_count = %d, compute_cv_AVE = %g",group_count, cv_value);
    return cv_value;
}

template <int L>
void MetaD_zqc::STEIN_QL<L>::bias_force_AVE(double dVdcv){
    // pass
    DEBUG_LOG("MetaD_zqc::STEIN_QL<L>::bias_force_AVE");
    double **f = lmp->atom->f;
    double **x = lmp->atom->x;
    int c_tag;
    DEBUG_LOG("MetaD_zqc::STEIN_QL<L>::bias_force_AVE");
    this->get_dcvdx_AVE(cv_value, h_dcvdx);
    // DEBUG_LOG("cv_value = %g, dVdcv = %g, dcvdx = %g, %g, %g",cv_value, dVdcv, dcvdx[0], dcvdx[1], dcvdx[2]);
    // DEBUG_LOG("fx0,fy0,fz0  = %.6f, %.6f, %.6f", f[c_tag][0], f[c_tag][1], f[c_tag][2]);
    for (int c_atom=0; c_atom<(my_env->group_count); c_atom++){
        DEBUG_LOG("dcvdx, dcvdy, dcvdz  = %g, %g, %g", h_dcvdx[c_atom*3 + 0], h_dcvdx[c_atom*3 + 1], h_dcvdx[c_atom*3 + 2]);
        DEBUG_LOG("dVdcv  = %g", dVdcv);
        c_tag = (my_env->h_group_indices)[c_atom];
        DEBUG_LOG("fx0,fy0,fz0  = %g, %g, %g", f[c_tag][0], f[c_tag][1], f[c_tag][2]);
        if (isnan(f[c_tag][0])||isnan(f[c_tag][1])||isnan(f[c_tag][2])){
            printf("error: force is infinity, check your system or cv_value.\n");
             error->all(FLERR, "STEIN_QL CV error: force is infinity, check your system or cv_value.");
        }
        f[c_tag][0] -= dVdcv*h_dcvdx[c_atom*3 + 0];
        f[c_tag][1] -= dVdcv*h_dcvdx[c_atom*3 + 1];
        f[c_tag][2] -= dVdcv*h_dcvdx[c_atom*3 + 2];
        DEBUG_LOG("fx,fy,fz  = %g, %g, %g", f[c_tag][0], f[c_tag][1], f[c_tag][2]);
    }
    DEBUG_LOG("post_force_r_end");
}

template <int L>
void MetaD_zqc::STEIN_QL<L>::get_dcvdx_AVE(double cv_value, double *dcvdx){
    int group_count = my_env->group_count;
    int Threads_own_atoms = lmp->atom->nlocal+lmp->atom->nghost;
    int last_group_count = my_env->last_group_count;
    size_t datalen = 0;
    

    // DEBUG_RUN(
    datalen = (Threads_own_atoms * (stein_l + 1) * 2);
    lmp->memory->grow(h_stein_qlm, datalen, "STEIN_QL:h_stein_qlm");
    // if (last_group_count < group_count){
    //     delete[] h_stein_qlm;
    //     h_stein_qlm = new double[datalen];
    // }
    SAFE_CUDA_MEMCPY(h_stein_qlm, d_stein_qlm.ptr, datalen*sizeof(double), cudaMemcpyDeviceToHost,f_check);
    // );


    datalen = (group_count*3);
    lmp->memory->grow(h_dcvdx, datalen, "STEIN_QL:h_dcvdx");
    // if (last_group_count < group_count){
    //     delete[] h_dcvdx;
    //     h_dcvdx = nullptr;
    //     h_dcvdx = new double[datalen];
    // }
    // SAFE_CUDA_FREE(d_dcvdx);
    // SAFE_CUDA_MALLOC(&d_dcvdx, datalen*sizeof(double), f_check);
    d_dcvdx.grow_to(datalen, f_check, __FILE__, __LINE__);
    SAFE_CUDA_MEMCPY(d_dcvdx.ptr,h_dcvdx, datalen*sizeof(double),cudaMemcpyHostToDevice,f_check);


    datalen = (group_count*3*2);
    lmp->memory->grow(h_dYlm_dr, datalen, "STEIN_QL:h_dYlm_dr");
    // if (last_group_count < group_count){
    //     delete[] h_dYlm_dr;
    //     h_dYlm_dr = nullptr;
    //     h_dYlm_dr = new double[(group_count*3*2)];
    // }
    // SAFE_CUDA_FREE(d_dYlm_dr);
    // SAFE_CUDA_MALLOC(&d_dYlm_dr, datalen*sizeof(double), f_check);
    d_dYlm_dr.grow_to(datalen, f_check, __FILE__, __LINE__);
    // SAFE_CUDA_MEMCPY(d_dYlm_dr,h_dYlm_dr,datalen*sizeof(double),cudaMemcpyHostToDevice,f_check);

    // sync Stein_qlm and stein_q with communication
    // then we can directly use the data in device to calculate dcvdx, 
    // without worrying about the data consistency between MPI processes.
    DEBUG_LOG("[Rank:%d][Before Comm] h_stein_qlm[0] = %f, ptr = %p\n",lmp->comm->me, h_stein_qlm[0], (void*)h_stein_qlm);
    DEBUG_LOG("[Rank:%d][Before Comm] stein_q[0] = %f, ptr = %p\n",lmp->comm->me, stein_q[0], (void*)h_stein_qlm);
    cudaDeviceSynchronize(); // waiting memory
    MPI_Barrier(lmp->world); // ensure all processes reach this point before communication
    comm_mode=true;
    lmp->comm->forward_comm(Fixmetad);
    comm_mode=false;
    DEBUG_LOG("[Rank:%d][After Comm] h_stein_qlm[0] = %f, ptr = %p\n",lmp->comm->me, h_stein_qlm[0], (void*)h_stein_qlm);
    DEBUG_LOG("[Rank:%d][After Comm] stein_q[0] = %f, ptr = %p\n",lmp->comm->me, stein_q[0], (void*)h_stein_qlm);
    // for (int i=0; i<((Threads_own_atoms)*(L + 1)*2); i++){
    //     printf("stein_qlm[%d] = %f\n", i, h_stein_qlm[i]);
    // }
    // for (int i=0; i<((Threads_own_atoms)); i++){
    //     printf("my_env->neigh_both_in_r_N[%d] = %d\n", i, my_env->neigh_both_in_r_N[i]);
    // }

    SAFE_CUDA_MEMCPY(d_stein_qlm.ptr, h_stein_qlm, ((Threads_own_atoms)*(L + 1)*2)*sizeof(double), cudaMemcpyHostToDevice,f_check);
    SAFE_CUDA_MEMCPY(d_stein_ql.ptr, stein_q, Threads_own_atoms*sizeof(double), cudaMemcpyHostToDevice,f_check);
    SAFE_CUDA_MEMCPY(my_env->d_neigh_both_in_r_N.ptr, my_env->neigh_both_in_r_N, Threads_own_atoms*sizeof(int), cudaMemcpyHostToDevice,f_check);



    // dcv_steinhardt_param_calc_kernel_q4(
    //     file, cutoff_Natoms, group_count, groupbit,
    //     mask, h_group_indices, calculated_numneigh,
    //     neigh_both_in_r_N, group_dminneigh,
    //     h_stein_qlm, h_stein_Ylm, stein_q,
    //     h_dYlm_dr, h_dcvdx);
    DEBUG_LOG("i will start a kernel of ql");
    cudaDeviceSynchronize(); // waiting memory
    call_steinhardt_dcv_AVE_kernel();
    cudaDeviceSynchronize(); // waiting memory
    DEBUG_LOG("i am out");

    cudaMemcpy(h_dcvdx, d_dcvdx.ptr, (group_count*3)*sizeof(double), cudaMemcpyDeviceToHost);
    // SAFE_CUDA_MEMCPY(h_dcvdx, d_dcvdx,
    //   (group_count*3)*sizeof(double), cudaMemcpyDeviceToHost, file);
    cudaDeviceSynchronize(); // waiting memory
    DEBUG_LOG("1");

}

template <int L>
double MetaD_zqc::STEIN_QL<L>::compute_cv_SW_FUNC(){
    DEBUG_LOG("im in compute_cv_SW_FUNC.");
    int group_count = my_env->group_count;
    DEBUG_LOG("group_count = %d",group_count);
    double ql_ave_local=0;
    DEBUG_LOG_COND((stein_q == NULL),"stein_q list not initialized");
    if (group_count != 0) {
        for (int c_atom=0; c_atom<group_count; c_atom++){
            int c_tag = (my_env->h_group_indices)[c_atom];
            double Si = stein_q[c_tag];
            ql_ave_local += my_cv_SWfunc->f(Si);
        }
    }
    MPI_Allreduce(&ql_ave_local, &cv_value, 1, MPI_DOUBLE, MPI_SUM, lmp->world);
    DEBUG_LOG("group_count = %d, compute_cv_SW_FUNC = %g",group_count, cv_value);
    return cv_value;
}

template <int L>
void MetaD_zqc::STEIN_QL<L>::bias_force_SW_FUNC(double dVdcv){
    // pass
    DEBUG_LOG("MetaD_zqc::STEIN_QL<L>::bias_force_SW_FUNC");
    double **f = lmp->atom->f;
    double **x = lmp->atom->x;
    int c_tag;
    DEBUG_LOG("MetaD_zqc::STEIN_QL<L>::bias_force_SW_FUNC");
    this->get_dcvdx_SW_FUNC(cv_value, h_dcvdx);
    // DEBUG_LOG("cv_value = %g, dVdcv = %g, dcvdx = %g, %g, %g",cv_value, dVdcv, dcvdx[0], dcvdx[1], dcvdx[2]);
    // DEBUG_LOG("fx0,fy0,fz0  = %.6f, %.6f, %.6f", f[c_tag][0], f[c_tag][1], f[c_tag][2]);
    for (int c_atom=0; c_atom<(my_env->group_count); c_atom++){
        DEBUG_LOG("dcvdx, dcvdy, dcvdz  = %g, %g, %g", h_dcvdx[c_atom*3 + 0], h_dcvdx[c_atom*3 + 1], h_dcvdx[c_atom*3 + 2]);
        DEBUG_LOG("dVdcv  = %g", dVdcv);
        c_tag = (my_env->h_group_indices)[c_atom];
        DEBUG_LOG("fx0,fy0,fz0  = %g, %g, %g", f[c_tag][0], f[c_tag][1], f[c_tag][2]);
        // if (isnan(f[c_tag][0])||isnan(f[c_tag][1])||isnan(f[c_tag][2])){
        //     LOG("error: force is infinity, check your system or cv_value.\n");
        //      error->all(FLERR, "STEIN_QL CV error: force is infinity, check your system or cv_value.");
        // }
        ERR_COND((isnan(f[c_tag][0])||isnan(f[c_tag][1])||isnan(f[c_tag][2])), 
                "STEIN_QL CV error: force is infinity, check your system or cv_value.");
        f[c_tag][0] -= dVdcv*h_dcvdx[c_atom*3 + 0];
        f[c_tag][1] -= dVdcv*h_dcvdx[c_atom*3 + 1];
        f[c_tag][2] -= dVdcv*h_dcvdx[c_atom*3 + 2];
        DEBUG_LOG("fx,fy,fz  = %g, %g, %g", f[c_tag][0], f[c_tag][1], f[c_tag][2]);
    }
    DEBUG_LOG("post_force_r_end");
}

template <int L>
void MetaD_zqc::STEIN_QL<L>::get_dcvdx_SW_FUNC(double cv_value, double *dcvdx){
    int group_count = my_env->group_count;
    int Threads_own_atoms = lmp->atom->nlocal+lmp->atom->nghost;
    int last_group_count = my_env->last_group_count;
    size_t datalen = 0;
    

    // DEBUG_RUN(
    datalen = (Threads_own_atoms * (stein_l + 1) * 2);
    lmp->memory->grow(h_stein_qlm, datalen, "STEIN_QL:h_stein_qlm");
    // if (last_group_count < group_count){
    //     delete[] h_stein_qlm;
    //     h_stein_qlm = new double[datalen];
    // }
    SAFE_CUDA_MEMCPY(h_stein_qlm, d_stein_qlm.ptr, datalen*sizeof(double), cudaMemcpyDeviceToHost,f_check);
    // );


    datalen = (group_count*3);
    lmp->memory->grow(h_dcvdx, datalen, "STEIN_QL:h_dcvdx");
    // if (last_group_count < group_count){
    //     delete[] h_dcvdx;
    //     h_dcvdx = nullptr;
    //     h_dcvdx = new double[datalen];
    // }
    // SAFE_CUDA_FREE(d_dcvdx);
    // SAFE_CUDA_MALLOC(&d_dcvdx, datalen*sizeof(double), f_check);
    d_dcvdx.grow_to(datalen, f_check, __FILE__, __LINE__);
    SAFE_CUDA_MEMCPY(d_dcvdx.ptr,h_dcvdx, datalen*sizeof(double),cudaMemcpyHostToDevice,f_check);


    datalen = (group_count*3*2);
    lmp->memory->grow(h_dYlm_dr, datalen, "STEIN_QL:h_dYlm_dr");
    // if (last_group_count < group_count){
    //     delete[] h_dYlm_dr;
    //     h_dYlm_dr = nullptr;
    //     h_dYlm_dr = new double[(group_count*3*2)];
    // }
    // SAFE_CUDA_FREE(d_dYlm_dr);
    // SAFE_CUDA_MALLOC(&d_dYlm_dr, datalen*sizeof(double), f_check);
    d_dYlm_dr.grow_to(datalen, f_check, __FILE__, __LINE__);
    // SAFE_CUDA_MEMCPY(d_dYlm_dr,h_dYlm_dr,datalen*sizeof(double),cudaMemcpyHostToDevice,f_check);

    // sync Stein_qlm and stein_q with communication
    // then we can directly use the data in device to calculate dcvdx, 
    // without worrying about the data consistency between MPI processes.
    DEBUG_LOG("[Rank:%d][Before Comm] h_stein_qlm[0] = %f, ptr = %p\n",lmp->comm->me, h_stein_qlm[0], (void*)h_stein_qlm);
    DEBUG_LOG("[Rank:%d][Before Comm] stein_q[0] = %f, ptr = %p\n",lmp->comm->me, stein_q[0], (void*)h_stein_qlm);
    cudaDeviceSynchronize(); // waiting memory
    MPI_Barrier(lmp->world); // ensure all processes reach this point before communication
    comm_mode=true;
    lmp->comm->forward_comm(Fixmetad);
    comm_mode=false;
    DEBUG_LOG("[Rank:%d][After Comm] h_stein_qlm[0] = %f, ptr = %p\n",lmp->comm->me, h_stein_qlm[0], (void*)h_stein_qlm);
    DEBUG_LOG("[Rank:%d][After Comm] stein_q[0] = %f, ptr = %p\n",lmp->comm->me, stein_q[0], (void*)h_stein_qlm);
    // for (int i=0; i<((Threads_own_atoms)*(L + 1)*2); i++){
    //     LOG("stein_qlm[%d] = %f\n", i, h_stein_qlm[i]);
    // }
    // for (int i=0; i<((Threads_own_atoms)); i++){
    //     LOG("my_env->neigh_both_in_r_N[%d] = %d\n", i, my_env->neigh_both_in_r_N[i]);
    // }

    SAFE_CUDA_MEMCPY(d_stein_qlm.ptr, h_stein_qlm, ((Threads_own_atoms)*(L + 1)*2)*sizeof(double), cudaMemcpyHostToDevice,f_check);
    SAFE_CUDA_MEMCPY(d_stein_ql.ptr, stein_q, Threads_own_atoms*sizeof(double), cudaMemcpyHostToDevice,f_check);
    SAFE_CUDA_MEMCPY(my_env->d_neigh_both_in_r_N.ptr, my_env->neigh_both_in_r_N, Threads_own_atoms*sizeof(int), cudaMemcpyHostToDevice,f_check);



    // dcv_steinhardt_param_calc_kernel_q4(
    //     file, cutoff_Natoms, group_count, groupbit,
    //     mask, h_group_indices, calculated_numneigh,
    //     neigh_both_in_r_N, group_dminneigh,
    //     h_stein_qlm, h_stein_Ylm, stein_q,
    //     h_dYlm_dr, h_dcvdx);
    DEBUG_LOG("i will start a kernel of ql");
    cudaDeviceSynchronize(); // waiting memory
    call_steinhardt_dcv_SW_FUNC_kernel();
    cudaDeviceSynchronize(); // waiting memory
    DEBUG_LOG("i am out");

    cudaMemcpy(h_dcvdx, d_dcvdx.ptr, (group_count*3)*sizeof(double), cudaMemcpyDeviceToHost);
    // SAFE_CUDA_MEMCPY(h_dcvdx, d_dcvdx,
    //   (group_count*3)*sizeof(double), cudaMemcpyDeviceToHost, file);
    cudaDeviceSynchronize(); // waiting memory
    DEBUG_LOG("1");

}


template <int L>
void MetaD_zqc::STEIN_QL<L>::steinhardt_param_calc(double *stein_ql){
    int cutoff_Natoms = my_env->cutoff_Natoms;
    int last_group_count = my_env->last_group_count;
    int group_count = my_env->group_count;
    int Threads_own_atoms = lmp->atom->nlocal + lmp->atom->nghost;
    // TODO: we can change the cuda stream to lammps stream, 
    // but we need to make sure that the stream is synchronized before we copy data back to host. 
    // For now, we will use the default stream.
    cudaStream_t lammps_stream = 0; // Assuming you want to use the default stream. Adjust if you have a specific stream.
    // in class protect
    // result array
    // every q has <2*L + 1> qlm, with complex we will times 2
    // double *h_stein_qlm = new double [group_count*(L + 1)*2];
    // for the further concentrate we need to calculate qlm*Neigh, with comple
    // size_t datalen = (group_count*cutoff_Natoms*(L + 1)*2);
    // if (last_group_count < group_count){
    //     // delete[] h_stein_Ylm;
    //     // h_stein_Ylm = new double [group_count*cutoff_Natoms*(L + 1)*2];
    //     lmp->memory->grow(h_stein_Ylm, datalen, "STEIN_QL:h_stein_Ylm");
    // }
    // SAFE_CUDA_FREE(d_stein_Ylm);
    // SAFE_CUDA_MALLOC(&d_stein_Ylm, (datalen)*sizeof(double), f_check);
    d_stein_Ylm.grow_to((group_count*cutoff_Natoms*(L + 1)*2), f_check, __FILE__, __LINE__);

    // SAFE_CUDA_FREE(d_stein_ql);
    // SAFE_CUDA_MALLOC(&d_stein_ql, Threads_own_atoms*sizeof(double), f_check);
    d_stein_ql.grow_to(Threads_own_atoms, f_check, __FILE__, __LINE__);
    cudaMemsetAsync(d_stein_ql.ptr, 0, (Threads_own_atoms)*sizeof(double), lammps_stream);
    // SAFE_CUDA_FREE(d_stein_qlm);
    // SAFE_CUDA_MALLOC(&d_stein_qlm, (Threads_own_atoms*(L + 1)*2)*sizeof(double), f_check);
    d_stein_qlm.grow_to((Threads_own_atoms*(L + 1)*2), f_check, __FILE__, __LINE__);
    cudaMemsetAsync(d_stein_qlm.ptr, 0, (Threads_own_atoms*(L + 1)*2)*sizeof(double), lammps_stream);

    DEBUG_LOG("i will start a kernel of ql");
    cudaDeviceSynchronize(); // waiting memory
    call_steinhardt_cv_AVE_kernel();
    // steinhardt_param_calc_kernel_q4<<<block_num,d_block_size>>>(
    //     group_count, cutoff_Natoms,
    //     d_neigh_both_in_r_N, d_group_dminneigh,
    //     d_stein_qlm, d_stein_Ylm,
    //     d_stein_ql) ;
    cudaDeviceSynchronize(); //catch kernel done
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        fprintf(f_check, "Kernel launch failed: %s\n", cudaGetErrorString(launchErr));
        error->all(FLERR, "Kernel launch failed\n");
    }
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        fprintf(f_check, "Kernel execution error: %s\n", cudaGetErrorString(syncErr));
        error->all(FLERR, "Kernel execution error\n");
    }
    DEBUG_LOG("im out");
    DEBUG_LOG("ql calculated find finished");

    // cudaMemcpy(stein_qlm, d_stein_qlm.ptr, (group_count*(L + 1)*2) * sizeof(double), cudaMemcpyDeviceToHost);
    SAFE_CUDA_MEMCPY(stein_ql, d_stein_ql.ptr,
      (group_count) * sizeof(double), cudaMemcpyDeviceToHost,f_check);
    // SAFE_CUDA_MEMCPY(h_stein_Ylm, d_stein_Ylm.ptr,
    //   (group_count*cutoff_Natoms*(L + 1)*2) * sizeof(double), cudaMemcpyDeviceToHost,f_check);

}


template <int L>
void MetaD_zqc::STEIN_QL<L>::summary(FILE* f){}


template <int L>
void MetaD_zqc::STEIN_QL<L>::call_steinhardt_dcv_AVE_kernel(){ 
    steinhardt_dcv_AVE_kernel<L> <<<block_num,d_block_size>>>(
        (my_env->cutoff_Natoms), (my_env->group_count), (my_env->groupbit), all_count,
        (my_env->d_mask.ptr), (my_env->d_group_indices.ptr), (my_env->d_calculated_numneigh.ptr),
        (my_env->d_neigh_both_in_r_N.ptr), (my_env->d_group_dminneigh.ptr),
        d_stein_qlm.ptr, d_stein_Ylm.ptr,  d_stein_ql.ptr,
        d_dYlm_dr.ptr, d_dcvdx.ptr);
}


template <int L>
void MetaD_zqc::STEIN_QL<L>::call_steinhardt_cv_AVE_kernel(){
    ERR_COND((my_env == nullptr),"my_env is NULL! Cannot launch kernel.");
    steinhardt_cv_AVE_kernel<L> <<<block_num,d_block_size>>>(
        (my_env->group_count), (my_env->cutoff_Natoms), (my_env->d_group_indices.ptr),
        (my_env->d_neigh_both_in_r_N.ptr), (my_env->d_group_dminneigh.ptr),
        d_stein_qlm.ptr, d_stein_Ylm.ptr,
        d_stein_ql.ptr) ;
}


template <int L>
void MetaD_zqc::STEIN_QL<L>::call_steinhardt_dcv_SW_FUNC_kernel(){ 
    auto sw_params = my_cv_SWfunc->params;
    steinhardt_dcv_SW_FUNC_kernel<L> <<<block_num,d_block_size>>>(
        sw_params,
        (my_env->cutoff_Natoms), (my_env->group_count), (my_env->groupbit), all_count,
        (my_env->d_mask.ptr), (my_env->d_group_indices.ptr), (my_env->d_calculated_numneigh.ptr),
        (my_env->d_neigh_both_in_r_N.ptr), (my_env->d_group_dminneigh.ptr),
        d_stein_qlm.ptr, d_stein_Ylm.ptr,  d_stein_ql.ptr,
        d_dYlm_dr.ptr, d_dcvdx.ptr);
}


// // 直接在求均值那里做了，不需要写新的kernel
// template <int L>
// void MetaD_zqc::STEIN_QL<L>::call_steinhardt_cv_SW_FUNC_kernel(){
//     steinhardt_cv_SW_FUNC_kernel<L> <<<block_num,d_block_size>>>(
//         (my_env->group_count), (my_env->cutoff_Natoms), (my_env->d_group_indices.ptr),
//         (my_env->d_neigh_both_in_r_N.ptr), (my_env->d_group_dminneigh.ptr),
//         d_stein_qlm.ptr, d_stein_Ylm.ptr,
//         d_stein_ql.ptr) ;
// }


template <int L>
int MetaD_zqc::STEIN_QL<L>::get_comm_forward_bytes(){ 
    // need to communicate for each atom in the list
    // qlm[2*(L+1) ] and ql (double value) and Neigh_Nb (int value)
    return num_elements +1 +1; // qlm + ql + Neigh_Nb
}

template <int L>
int MetaD_zqc::STEIN_QL<L>::pack_comm_ubuf(int n, int *list, double *u_buf, int slot_offset, int comm_forward) {
    if (!comm_mode){
        return (num_elements + 1 +1);
    }
    int m = slot_offset; 
    int cycle_offset = comm_forward;

    for (int i = 0; i < n; i++) {
        int j = list[i]; // 目标本地原子标号
        
        // 1. 先塞当前原子的所有 qlm 分量
        for (int k = 0; k < num_elements; k++) {
            u_buf[m + cycle_offset*i + k] = h_stein_qlm[j * num_elements + k];
        }
        
        // 2. 紧接着，塞当前原子的 ql 标量数据
        u_buf[m + cycle_offset*i + num_elements] = stein_q[j]; // 假设这是你的 ql 数组

        u_buf[m + cycle_offset*i + num_elements +1] = ubuf(my_env->neigh_both_in_r_N[j]).d;
    }
    
    return (num_elements + 1 +1);
}

template <int L>
void MetaD_zqc::STEIN_QL<L>::unpack_comm_ubuf(int n, int first, double *u_buf, int slot_offset, int comm_forward) {
    if (!comm_mode){
        return;
    }

    int m = slot_offset; 
    int cycle_offset = comm_forward;
    
    // 从 first 开始，连续恢复 n 个 Ghost 原子的复合数据
    for (int i = first; i < first + n; i++) {
        
        // 1. 先剥离 qlm 倒回 qlm 跑道
        for (int k = 0; k < num_elements; k++) {
            h_stein_qlm[i * num_elements + k] = u_buf[ m+ cycle_offset*(i-first) + k];
        }
        
        // 2. 紧接着剥离 ql 倒回 ql 跑道
        stein_q[i] = u_buf[ m+ cycle_offset*(i-first) + num_elements];

        my_env->neigh_both_in_r_N[i] = (int) ubuf(u_buf[ m+ cycle_offset*(i-first) + num_elements +1]).i;
    }
}

template <int L>
double* MetaD_zqc::STEIN_QL<L>::get_peratom_ptr(const std::string &prop_name) {
    if (prop_name == "stein_q") {
        return stein_q;
    }
    return nullptr;
}

__global__ void get_environment_Steinhardt_Q(int cutoff_Natoms, double cutoff_rsq,
    double box_x, double box_y, double box_z,
    int group_count, int *d_group_indices, LAMMPS_NS::tagint *d_group_numneigh,
    int *d_firstneigh_ptrs, double *d_x_flat,
    double *d_group_dminneigh, int *d_neigh_in_cutoff_r, int *d_neigh_both_in_r_N,
    LAMMPS_NS::tagint *d_calculated_numneigh){
    // get_environment_Steinhardt_Q in GPU
    int c_atom = blockIdx.x * blockDim.x + threadIdx.x;
    if(c_atom<group_count){
        double r2,temp_r2,temp_x,temp_y,temp_z,neigh_x,neigh_y,neigh_z;
        double delt_x,delt_y,delt_z;
        int c_atom_tag = d_group_indices[c_atom];
        int temp_tag;
        d_neigh_in_cutoff_r[c_atom] = 0;
        // c_glob_tag = h_tag[c_atom_tag];
        double c_x = d_x_flat[c_atom_tag*3];
        double c_y = d_x_flat[c_atom_tag*3+1];
        double c_z = d_x_flat[c_atom_tag*3+2];
        int max_ii;
        // DEBUG_LOG("now im in %d, c_atom_tag=%d, cx,cy,cz:%f,%f,%f",c_atom,c_atom_tag,c_x,c_y,c_z);
        double max_r2 = (box_x+box_y+box_z)*(box_x+box_y+box_z);
        for (int i=0;i<cutoff_Natoms;i++){
            d_group_dminneigh[c_atom*4*cutoff_Natoms +i*4 + 3]=max_r2;
            d_calculated_numneigh[c_atom*cutoff_Natoms +i] = -1;
        }
        //find curtoff_Natoms neigh
        for (int neigh_atom=d_group_numneigh[c_atom]; neigh_atom<d_group_numneigh[c_atom+1]; neigh_atom++){
            int n_local_tag = d_firstneigh_ptrs[neigh_atom];
            // if (n_local_tag < 0 ) continue;
            // int n_glob_tag = h_tag[n_local_tag];
            neigh_x = d_x_flat[n_local_tag*3];
            neigh_y = d_x_flat[n_local_tag*3+1];
            neigh_z = d_x_flat[n_local_tag*3+2];
            delt_x = (neigh_x - c_x);
            delt_y = (neigh_y - c_y);
            delt_z = (neigh_z - c_z);
            // if (delt_x > box_x/2) {
            //     delt_x -= box_x;
            // } else if (delt_x < -box_x/2) {
            //     delt_x += box_x;
            // }
            // if (delt_y > box_y/2) {
            //     delt_y -= box_y;
            // } else if (delt_y < -box_y/2) {
            //     delt_y += box_y;
            // }
            // if (delt_z > box_z/2) {
            //     delt_z -= box_z;
            // } else if (delt_z < -box_z/2) {
            //     delt_z += box_z;
            // }
            // DEBUG_LOG("c_atom_tag=%d, n_local_tag=%d, nx,ny,nz:%f,%f,%f",c_atom_tag,n_local_tag,delt_x,delt_y,delt_z);
            r2 = delt_x*delt_x + delt_y*delt_y + delt_z*delt_z;
            if ((r2 > cutoff_rsq )||(r2<1e-12)) continue;
            d_neigh_in_cutoff_r[c_atom]++;
            for (int ii=0; ii<cutoff_Natoms; ii++){
                if (d_group_dminneigh[c_atom*4*cutoff_Natoms + ii*4 + 3]>r2){
                    temp_x = d_group_dminneigh[c_atom*4*cutoff_Natoms + ii*4 + 0];
                    temp_y = d_group_dminneigh[c_atom*4*cutoff_Natoms + ii*4 + 1];
                    temp_z = d_group_dminneigh[c_atom*4*cutoff_Natoms + ii*4 + 2];
                    temp_r2 = d_group_dminneigh[c_atom*4*cutoff_Natoms + ii*4 + 3];
                    temp_tag = d_calculated_numneigh[c_atom*cutoff_Natoms + ii];
                    d_group_dminneigh[c_atom*4*cutoff_Natoms + ii*4 + 0] = delt_x;
                    d_group_dminneigh[c_atom*4*cutoff_Natoms + ii*4 + 1] = delt_y;
                    d_group_dminneigh[c_atom*4*cutoff_Natoms + ii*4 + 2] = delt_z;
                    d_group_dminneigh[c_atom*4*cutoff_Natoms + ii*4 + 3] = r2;
                    d_calculated_numneigh[c_atom*cutoff_Natoms + ii] = n_local_tag;
                    delt_x = temp_x;
                    delt_y = temp_y;
                    delt_z = temp_z;
                    r2 = temp_r2;
                    n_local_tag = temp_tag;
                }
            }
        }
        if (d_neigh_in_cutoff_r[c_atom]>=cutoff_Natoms){
            d_neigh_both_in_r_N[c_atom]=cutoff_Natoms;
        }
        else{
            d_neigh_both_in_r_N[c_atom]=d_neigh_in_cutoff_r[c_atom];
        }
    }
}



template <int L>
__global__ void steinhardt_cv_AVE_kernel(
    int group_count, int cutoff_Natoms, int *d_group_indices,
    int *d_neigh_both_in_r_N, double *d_group_dminneigh,
    double *d_stein_qlm, double *d_stein_Ylm, double *d_stein_ql) {

    int c_atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_atom >= group_count) return;

    int c_atom_tag = d_group_indices[c_atom]; // 当前原子在local原子列表中的标签

    // 【与导数核函数完美镜像】的近邻数读取与基础寻址逻辑
    int neigh_num = d_neigh_both_in_r_N[c_atom];
    int stein_qlm_base_id = c_atom_tag * (L + 1) * 2; // 为了通讯方便，qlm和Ylm都按照local原子标签来存储和访问
    int stein_Ylm_base_id;

    // 如果没有邻居，直接清零退出
    if (neigh_num == 0) return;
    double inv_neigh = 1.0 / (double)neigh_num;

    // 在寄存器（栈）上初始化局部数组用于累加，避免频繁读写全局显存
    constexpr int qlm_size = (L + 1) * 2;
    double local_qlm[qlm_size] = {0.0};
    // d_stein_qlm[stein_qlm_base_id] = 0.0;
    // double *local_qlm = &d_stein_qlm[stein_qlm_base_id];

    for (int neigh_atom = 0; neigh_atom < neigh_num; neigh_atom++) {
        // 【完全通用】的坐标读取与基础三角函数
        double dx = d_group_dminneigh[c_atom * cutoff_Natoms * 4 + neigh_atom * 4 + 0];
        double dy = d_group_dminneigh[c_atom * cutoff_Natoms * 4 + neigh_atom * 4 + 1];
        double dz = d_group_dminneigh[c_atom * cutoff_Natoms * 4 + neigh_atom * 4 + 2];
        double r2 = d_group_dminneigh[c_atom * cutoff_Natoms * 4 + neigh_atom * 4 + 3];
        double r  = sqrt(r2);

        double theta = acos(dz / r);
        double phi = atan2(dy, dx);

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
        
        // Ylm 只与原子位置有关，所以可以按照group直接访问不需要扩大数组,所以用c_atom而不是c_atom_tag
        stein_Ylm_base_id = c_atom*cutoff_Natoms*(L + 1)*2 + neigh_atom*(L + 1)*2;

        // ==========================================================
        //  💥 核心艺术：利用编译期静态判断条件，杜绝任何浪费！
        // ==========================================================
        if constexpr (L == 3) {
            compute_qlm_forward_L3(
                cos_theta, sin_theta, cos_phi, sin_phi,
                cos_2theta, sin_2theta, cos_2phi, sin_2phi,
                cos_3theta, sin_3theta, cos_3phi, sin_3phi,
                &local_qlm[0], &d_stein_Ylm[stein_Ylm_base_id]
            );
        } else if constexpr (L == 4) {
            compute_qlm_forward_L4(
                cos_theta, sin_theta, cos_phi, sin_phi,
                cos_2phi, sin_2phi,
                cos_3phi, sin_3phi,
                cos_4phi, sin_4phi,
                &local_qlm[0], &d_stein_Ylm[stein_Ylm_base_id]
            );
        } else if constexpr (L == 6) {
            compute_qlm_forward_L6(
                cos_theta, sin_theta, cos_phi, sin_phi,
                cos_2theta, sin_2theta, cos_2phi, sin_2phi,
                cos_3phi, sin_3phi,
                cos_4theta, sin_4theta, cos_4phi, sin_4phi,
                cos_5phi, sin_5phi,
                cos_6theta, sin_6theta, cos_6phi, sin_6phi,
                &local_qlm[0], &d_stein_Ylm[stein_Ylm_base_id]
            );
        }
    }

    // --- 循环外归一化与写回全局显存 ---
    
    #pragma unroll
    for (int i = 0; i < qlm_size; i++) {
        local_qlm[i] *= inv_neigh;
        d_stein_qlm[stein_qlm_base_id + i] = local_qlm[i];
    }

    double ql_sq = local_qlm[0] * local_qlm[0];

    // 3. 对应你原代码的第三步：从 m = 1 开始累加已经归一化的各项平方和
    #pragma unroll
    for (int i = 1; i <= L; i++) {
        double re_part = local_qlm[i * 2 + 0];
        double im_part = local_qlm[i * 2 + 1];
        ql_sq += 2.0 * (re_part * re_part + im_part * im_part);
    }

    d_stein_ql[c_atom_tag] = sqrt(ql_sq * 12.56637061435917295385/double(2*L + 1));
}
template __global__ void steinhardt_cv_AVE_kernel<3>(int, int, int*, int*, double*, double*, double*, double*);
template __global__ void steinhardt_cv_AVE_kernel<4>(int, int, int*, int*, double*, double*, double*, double*);
template __global__ void steinhardt_cv_AVE_kernel<6>(int, int, int*, int*, double*, double*, double*, double*);


template <int L>
__global__ void steinhardt_dcv_AVE_kernel(
    int cutoff_Natoms, int group_count, int groupbit, int all_count, 
    int *d_mask, LAMMPS_NS::tagint *d_group_indices, LAMMPS_NS::tagint *d_calculated_numneigh, 
    int *d_neigh_both_in_r_N, double *d_group_dminneigh,
    double *d_stein_qlm, double *d_stein_Ylm, double *d_stein_ql,
    double *d_dYlm_dr, double *d_dcvdx) {
    int c_atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (c_atom >= group_count) return;
    double Factor_Y, Factor_Ydx, Factor_Ydy, Factor_Ydz;
    double tdx_r, tdx_i, tdy_r, tdy_i, tdz_r, tdz_i;

    int c_atom_tag = d_group_indices[c_atom];

    // 【完全通用】框架：近邻、数组清零、寻址逻辑
    int neigh_num = d_neigh_both_in_r_N[c_atom];
    for(int i = 0; i < 3; i++) {
        d_dcvdx[c_atom * 3 + i] = 0.0;
        d_dYlm_dr[c_atom * 3 * 2 + i * 2 + 0] = 0.0;
        d_dYlm_dr[c_atom * 3 * 2 + i * 2 + 1] = 0.0;
    }
    if (neigh_num == 0) return;

    double catom_ql_timesN = 1.0 / (d_stein_ql[c_atom] * neigh_num);
    int stein_qlm_base_id = c_atom * (L + 1) * 2;

    for (int neigh_atom = 0; neigh_atom < neigh_num; neigh_atom++) {
        // 【完全通用】的坐标读取与基础三角函数
        double dx = d_group_dminneigh[c_atom * cutoff_Natoms * 4 + neigh_atom * 4 + 0];
        double dy = d_group_dminneigh[c_atom * cutoff_Natoms * 4 + neigh_atom * 4 + 1];
        double dz = d_group_dminneigh[c_atom * cutoff_Natoms * 4 + neigh_atom * 4 + 2];
        double r2 = d_group_dminneigh[c_atom * cutoff_Natoms * 4 + neigh_atom * 4 + 3];
        double r  = sqrt(r2);
        
        double theta = acos(dz / r);
        double phi = atan2(dy, dx);

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
            sincos(2 * theta, &sin_2theta, &cos_2theta);
            sincos(2 * phi, &sin_2phi, &cos_2phi);
        }
        // 如果 L >= 3，才编译 3 倍角
        if constexpr (L >= 3) {
            sincos(3 * phi, &sin_3phi, &cos_3phi);
            sincos(4 * theta, &sin_4theta, &cos_4theta);
        }
        // 如果 L >= 4，才编译 4 倍角
        if constexpr (L >= 4) {
            sincos(3 * theta, &sin_3theta, &cos_3theta);
            sincos(4 * phi, &sin_4phi, &cos_4phi);
            sincos(5 * theta, &sin_5theta, &cos_5theta);
            sincos(5 * phi, &sin_5phi, &cos_5phi);
            sincos(6 * theta, &sin_6theta, &cos_6theta);
        }
        if constexpr (L >= 6) {
            sincos(6 * phi, &sin_6phi, &cos_6phi);
        }

        // 【完全通用】的近邻查找逻辑
        int Neigh_Nb = 0;
        double neigh_ql_timesN = 0.0;
        int stein_qlm_neigh_id = 0;
        int neigh_tag = d_calculated_numneigh[c_atom * cutoff_Natoms + neigh_atom];
        
        if (d_mask[neigh_tag] & groupbit) {
            Neigh_Nb = d_neigh_both_in_r_N[neigh_tag];
            neigh_ql_timesN = 1.0 / (d_stein_ql[neigh_tag] * Neigh_Nb);
            stein_qlm_neigh_id = neigh_tag * (L + 1) * 2;
            // int left = 0, right = group_count - 1;
            // while (left <= right) {
            //     int mid = left + (right - left) / 2;
            //     if (d_group_indices[mid] == neigh_tag) {
            //         Neigh_Nb = d_neigh_both_in_r_N[mid];
            //         neigh_ql_timesN = 1.0 / (d_stein_ql[mid] * Neigh_Nb);
            //         stein_qlm_neigh_id = mid * (L + 1) * 2;
            //         break;
            //     } else if (d_group_indices[mid] < neigh_tag) left = mid + 1;
            //     else right = mid - 1;
            // }
        }

        // ==========================================================
        //  利用编译期静态判断条件！
        // ==========================================================
        if constexpr (L == 3) {
            // 当编译指定该模板为 <3> 时，编译器在这一步会直接盲切到 L3 函数。
            // 此时 L==6 的分支、以及计算 q6 所需的其他高阶 sin_5theta 变量，
            // 会被编译器判定为“死代码”彻底移除。最终生成的 GPU 二进制指令纯净无污染。
            compute_Ylm_gradient_L3(
                r, 
                cos_theta, sin_theta, cos_phi, sin_phi,
                cos_2theta, sin_2theta, cos_2phi, sin_2phi,
                cos_3phi, sin_3phi,
                cos_4theta, sin_4theta, 
                catom_ql_timesN, neigh_ql_timesN,
                stein_qlm_base_id, stein_qlm_neigh_id,
                d_stein_qlm, &d_dYlm_dr[c_atom * 3 * 2]
            );
        } else if constexpr (L == 4) {
            // 针对计算 q4，这里在前面额外多算两个高阶级联分量即可
            compute_Ylm_gradient_L4(
                r, 
                cos_theta, sin_theta, cos_phi, sin_phi,
                cos_2theta, sin_2theta, cos_2phi, sin_2phi,
                cos_3theta, sin_3theta, cos_3phi, sin_3phi,
                cos_4theta, sin_4theta, cos_4phi, sin_4phi,
                cos_5theta, sin_5theta, cos_5phi, sin_5phi,
                cos_6theta, sin_6theta,
                catom_ql_timesN, neigh_ql_timesN,
                stein_qlm_base_id, stein_qlm_neigh_id,
                d_stein_qlm, &d_dYlm_dr[c_atom * 3 * 2]
            );
        } else if constexpr (L == 6) {
            // 针对计算 q6，这里在前面额外多算两个高阶级联分量即可
            compute_Ylm_gradient_L6(
                r, 
                cos_theta, sin_theta, cos_phi, sin_phi,
                cos_2theta, sin_2theta, cos_2phi, sin_2phi,
                cos_3theta, sin_3theta, cos_3phi, sin_3phi,
                cos_4theta, sin_4theta, cos_4phi, sin_4phi,
                cos_5theta, sin_5theta, cos_5phi, sin_5phi,
                cos_6theta, sin_6theta, cos_6phi, sin_6phi,
                catom_ql_timesN, neigh_ql_timesN,
                stein_qlm_base_id, stein_qlm_neigh_id,
                d_stein_qlm, &d_dYlm_dr[c_atom * 3 * 2]
            );
        }
        // printf("c_atom=%d, neigh_atom=%d, neigh_tag=%d, Neigh_Nb=%d, d_stein_ql[neigh_tag]=%g\n",
        //         c_atom, neigh_atom, neigh_tag, Neigh_Nb, d_stein_ql[neigh_tag]);
        double fx = (d_dYlm_dr[c_atom * 3 * 2 + 0 * 2 + 0] + d_dYlm_dr[c_atom * 3 * 2 + 0 * 2 + 1]);
        double fy = (d_dYlm_dr[c_atom * 3 * 2 + 1 * 2 + 0] + d_dYlm_dr[c_atom * 3 * 2 + 1 * 2 + 1]);
        double fz = (d_dYlm_dr[c_atom * 3 * 2 + 2 * 2 + 0] + d_dYlm_dr[c_atom * 3 * 2 + 2 * 2 + 1]);
        if (isnan(fx) || isnan(fy) || isnan(fz)) {
            // 只有崩成 NaN 的线程才会触发打印，不影响整体速度
            printf("[NaN Detected] c_atom = %d, neigh_tag = %d, Neigh_Nb=%d, d_stein_ql[neigh_tag]=%g, d_stein_qlm[stein_qlm_neigh_id + 0]=%g r = %f, dx = %f, dy = %f, dz = %f, sin_theta = %f\n", 
                    c_atom, neigh_tag, Neigh_Nb, d_stein_ql[neigh_tag],d_stein_qlm[stein_qlm_neigh_id + 0], r, dx, dy, dz, sin_theta);
        }
    }

    // 【完全通用】最后的总偏导汇总
    for (int i = 0; i < 3; i++) {
        d_dcvdx[c_atom * 3 + i] = d_dYlm_dr[c_atom * 3 * 2 + i * 2 + 0] + d_dYlm_dr[c_atom * 3 * 2 + i * 2 + 1];
        d_dcvdx[c_atom * 3 + i] = -(d_dcvdx[c_atom * 3 + i] * 2 * PI) / (all_count * (2 * L + 1));
    }
}
template __global__ void steinhardt_dcv_AVE_kernel<3>(int, int, int, int,
    int*,LAMMPS_NS::tagint *,LAMMPS_NS::tagint *, int *, double*, double*, double*, double*, double*, double*);
template __global__ void steinhardt_dcv_AVE_kernel<4>(int, int, int, int,
    int*,LAMMPS_NS::tagint *,LAMMPS_NS::tagint *, int *, double*, double*, double*, double*, double*, double*);
template __global__ void steinhardt_dcv_AVE_kernel<6>(int, int, int, int,
    int*,LAMMPS_NS::tagint *,LAMMPS_NS::tagint *, int *, double*, double*, double*, double*, double*, double*);


template <int L>
__global__ void steinhardt_dcv_SW_FUNC_kernel(
    MetaD_zqc::SwitchFunctionRequest sw_params,
    int cutoff_Natoms, int group_count, int groupbit, int all_count, 
    int *d_mask, LAMMPS_NS::tagint *d_group_indices, LAMMPS_NS::tagint *d_calculated_numneigh, 
    int *d_neigh_both_in_r_N, double *d_group_dminneigh,
    double *d_stein_qlm, double *d_stein_Ylm, double *d_stein_ql,
    double *d_dYlm_dr, double *d_dcvdx) {

    int c_atom = blockIdx.x * blockDim.x + threadIdx.x;
    auto sw_f = [&](double S_val) {
        return MetaD_zqc::SwitchFunction::f(sw_params, S_val);
    };
    auto sw_df = [&](double S_val) {
        return MetaD_zqc::SwitchFunction::df(sw_params, S_val);
    };
    if (c_atom >= group_count) return;
    double Factor_Y, Factor_Ydx, Factor_Ydy, Factor_Ydz;
    double tdx_r, tdx_i, tdy_r, tdy_i, tdz_r, tdz_i;

    int c_atom_tag = d_group_indices[c_atom];

    // 【完全通用】框架：近邻、数组清零、寻址逻辑
    int neigh_num = d_neigh_both_in_r_N[c_atom];
    for(int i = 0; i < 3; i++) {
        d_dcvdx[c_atom * 3 + i] = 0.0;
        d_dYlm_dr[c_atom * 3 * 2 + i * 2 + 0] = 0.0;
        d_dYlm_dr[c_atom * 3 * 2 + i * 2 + 1] = 0.0;
    }
    if (neigh_num == 0) return;

    double ql_c = d_stein_ql[c_atom];
    double catom_ql_timesN = (sw_f(ql_c)+ql_c*sw_df(ql_c)) / (ql_c * neigh_num);
    int stein_qlm_base_id = c_atom * (L + 1) * 2;

    for (int neigh_atom = 0; neigh_atom < neigh_num; neigh_atom++) {
        // 【完全通用】的坐标读取与基础三角函数
        double dx = d_group_dminneigh[c_atom * cutoff_Natoms * 4 + neigh_atom * 4 + 0];
        double dy = d_group_dminneigh[c_atom * cutoff_Natoms * 4 + neigh_atom * 4 + 1];
        double dz = d_group_dminneigh[c_atom * cutoff_Natoms * 4 + neigh_atom * 4 + 2];
        double r2 = d_group_dminneigh[c_atom * cutoff_Natoms * 4 + neigh_atom * 4 + 3];
        double r  = sqrt(r2);
        
        double theta = acos(dz / r);
        double phi = atan2(dy, dx);

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
            sincos(2 * theta, &sin_2theta, &cos_2theta);
            sincos(2 * phi, &sin_2phi, &cos_2phi);
        }
        // 如果 L >= 3，才编译 3 倍角
        if constexpr (L >= 3) {
            sincos(3 * phi, &sin_3phi, &cos_3phi);
            sincos(4 * theta, &sin_4theta, &cos_4theta);
        }
        // 如果 L >= 4，才编译 4 倍角
        if constexpr (L >= 4) {
            sincos(3 * theta, &sin_3theta, &cos_3theta);
            sincos(4 * phi, &sin_4phi, &cos_4phi);
            sincos(5 * theta, &sin_5theta, &cos_5theta);
            sincos(5 * phi, &sin_5phi, &cos_5phi);
            sincos(6 * theta, &sin_6theta, &cos_6theta);
        }
        if constexpr (L >= 6) {
            sincos(6 * phi, &sin_6phi, &cos_6phi);
        }

        // 【完全通用】的近邻查找逻辑
        int Neigh_Nb = 0;
        double neigh_ql_timesN = 0.0;
        int stein_qlm_neigh_id = 0;
        int neigh_tag = d_calculated_numneigh[c_atom * cutoff_Natoms + neigh_atom];
        
        if (d_mask[neigh_tag] & groupbit) {
            Neigh_Nb = d_neigh_both_in_r_N[neigh_tag];
            double ql_n = d_stein_ql[neigh_tag];
            neigh_ql_timesN = (sw_f(ql_n)+ql_n*sw_df(ql_n)) / (d_stein_ql[neigh_tag] * Neigh_Nb);
            stein_qlm_neigh_id = neigh_tag * (L + 1) * 2;
        }

        // ==========================================================
        //  利用编译期静态判断条件！
        // ==========================================================
        if constexpr (L == 3) {
            // 当编译指定该模板为 <3> 时，编译器在这一步会直接盲切到 L3 函数。
            // 此时 L==6 的分支、以及计算 q6 所需的其他高阶 sin_5theta 变量，
            // 会被编译器判定为“死代码”彻底移除。最终生成的 GPU 二进制指令纯净无污染。
            compute_Ylm_gradient_L3(
                r, 
                cos_theta, sin_theta, cos_phi, sin_phi,
                cos_2theta, sin_2theta, cos_2phi, sin_2phi,
                cos_3phi, sin_3phi,
                cos_4theta, sin_4theta, 
                catom_ql_timesN, neigh_ql_timesN,
                stein_qlm_base_id, stein_qlm_neigh_id,
                d_stein_qlm, &d_dYlm_dr[c_atom * 3 * 2]
            );
        } else if constexpr (L == 4) {
            // 针对计算 q4，这里在前面额外多算两个高阶级联分量即可
            compute_Ylm_gradient_L4(
                r, 
                cos_theta, sin_theta, cos_phi, sin_phi,
                cos_2theta, sin_2theta, cos_2phi, sin_2phi,
                cos_3theta, sin_3theta, cos_3phi, sin_3phi,
                cos_4theta, sin_4theta, cos_4phi, sin_4phi,
                cos_5theta, sin_5theta, cos_5phi, sin_5phi,
                cos_6theta, sin_6theta,
                catom_ql_timesN, neigh_ql_timesN,
                stein_qlm_base_id, stein_qlm_neigh_id,
                d_stein_qlm, &d_dYlm_dr[c_atom * 3 * 2]
            );
        } else if constexpr (L == 6) {
            // 针对计算 q6，这里在前面额外多算两个高阶级联分量即可
            compute_Ylm_gradient_L6(
                r, 
                cos_theta, sin_theta, cos_phi, sin_phi,
                cos_2theta, sin_2theta, cos_2phi, sin_2phi,
                cos_3theta, sin_3theta, cos_3phi, sin_3phi,
                cos_4theta, sin_4theta, cos_4phi, sin_4phi,
                cos_5theta, sin_5theta, cos_5phi, sin_5phi,
                cos_6theta, sin_6theta, cos_6phi, sin_6phi,
                catom_ql_timesN, neigh_ql_timesN,
                stein_qlm_base_id, stein_qlm_neigh_id,
                d_stein_qlm, &d_dYlm_dr[c_atom * 3 * 2]
            );
        }
        // printf("c_atom=%d, neigh_atom=%d, neigh_tag=%d, Neigh_Nb=%d, d_stein_ql[neigh_tag]=%g\n",
        //         c_atom, neigh_atom, neigh_tag, Neigh_Nb, d_stein_ql[neigh_tag]);
        double fx = (d_dYlm_dr[c_atom * 3 * 2 + 0 * 2 + 0] + d_dYlm_dr[c_atom * 3 * 2 + 0 * 2 + 1]);
        double fy = (d_dYlm_dr[c_atom * 3 * 2 + 1 * 2 + 0] + d_dYlm_dr[c_atom * 3 * 2 + 1 * 2 + 1]);
        double fz = (d_dYlm_dr[c_atom * 3 * 2 + 2 * 2 + 0] + d_dYlm_dr[c_atom * 3 * 2 + 2 * 2 + 1]);
        if (isnan(fx) || isnan(fy) || isnan(fz)) {
            // 只有崩成 NaN 的线程才会触发打印，不影响整体速度
            printf("[NaN Detected] c_atom = %d, neigh_tag = %d, Neigh_Nb=%d, d_stein_ql[neigh_tag]=%g, d_stein_qlm[stein_qlm_neigh_id + 0]=%g r = %f, dx = %f, dy = %f, dz = %f, sin_theta = %f\n", 
                    c_atom, neigh_tag, Neigh_Nb, d_stein_ql[neigh_tag],d_stein_qlm[stein_qlm_neigh_id + 0], r, dx, dy, dz, sin_theta);
        }
    }

    // 【完全通用】最后的总偏导汇总
    for (int i = 0; i < 3; i++) {
        d_dcvdx[c_atom * 3 + i] = d_dYlm_dr[c_atom * 3 * 2 + i * 2 + 0] + d_dYlm_dr[c_atom * 3 * 2 + i * 2 + 1];
        d_dcvdx[c_atom * 3 + i] = -(d_dcvdx[c_atom * 3 + i] * 2 * PI) / (2 * L + 1);
    }
}
template __global__ void steinhardt_dcv_SW_FUNC_kernel<3>(MetaD_zqc::SwitchFunctionRequest, 
    int, int, int, int, int*,LAMMPS_NS::tagint *,LAMMPS_NS::tagint *, int *, 
    double*, double*, double*, double*, double*, double*);
template __global__ void steinhardt_dcv_SW_FUNC_kernel<4>(MetaD_zqc::SwitchFunctionRequest, 
    int, int, int, int, int*,LAMMPS_NS::tagint *,LAMMPS_NS::tagint *, int *, 
    double*, double*, double*, double*, double*, double*);
template __global__ void steinhardt_dcv_SW_FUNC_kernel<6>(MetaD_zqc::SwitchFunctionRequest, 
    int, int, int, int, int*,LAMMPS_NS::tagint *,LAMMPS_NS::tagint *, int *, 
    double*, double*, double*, double*, double*, double*);


// __global__ void steinhardt_param_calc_LOCAL_kernel(int group_count, int cutoff_Natoms,
//                     int stein_l, int groupbit,
//                     int *d_mask, LAMMPS_NS::tagint *d_group_indices,
//                     LAMMPS_NS::tagint *d_calculated_numneigh, 
//                     int *d_neigh_both_in_r_N,
//                     double *d_stein_qlm, double *d_stein_LQlm,
//                     double *d_stein_ql){
//     int c_atom = blockIdx.x * blockDim.x + threadIdx.x;
//     if(c_atom<group_count){
//         // steinhardt_param_calc_kernel_q4
//         // int stein_l=4;
//         int neigh_num, neigh_tag;
//         double temp4pi_2lplus1;
//         temp4pi_2lplus1 = 12.5663706143591729538505735331/(2*stein_l+1);
//         neigh_num = d_neigh_both_in_r_N[c_atom];
//         int base_LQlm_neigh_id,LQlm_neigh_id;
//         base_LQlm_neigh_id=c_atom*(stein_l + 1)*2;
//         d_stein_ql[c_atom] = 0;
//         if (neigh_num == 0) {
//             return;
//         }
//         for(int i=0; i<(stein_l + 1)*2; i++){
//             d_stein_LQlm[base_LQlm_neigh_id + i] = d_stein_qlm[c_atom + i];
//         }
//         for(int neigh_atom=0; neigh_atom<neigh_num; neigh_atom++){
//             neigh_tag = d_calculated_numneigh[c_atom*cutoff_Natoms + neigh_atom];
//             if (d_mask[neigh_tag]&groupbit){
//                 // TODO: 此处有问题，因为邻居原子可能不在cvgroup中，此时是找不到它的LQlm的
//                 // 使用二分查找法找 neigh_tag 对应在 d_stein_ql 中的位置
//                 int left = 0;
//                 int right = group_count - 1;
//                 // neigh_q4_deN default is 0
//                 while (left <= right) {
//                     int mid = left + (right - left) / 2;
//                     if (d_group_indices[mid] == neigh_tag) {
//                         LQlm_neigh_id = mid * (stein_l + 1) * 2;
//                         for(int i=0; i<(stein_l + 1)*2; i++){
//                             d_stein_LQlm[base_LQlm_neigh_id + i] += d_stein_qlm[LQlm_neigh_id + i];
//                         }
//                         break;
//                     } else if (d_group_indices[mid] < neigh_tag) {
//                         left = mid + 1;
//                     } else {
//                         right = mid - 1;
//                     }
//                 }
//             }
//         }
//         int i=0;
//         d_stein_LQlm[base_LQlm_neigh_id + i] = d_stein_LQlm[base_LQlm_neigh_id + i] / (neigh_num+1);
//         d_stein_ql[c_atom] += d_stein_LQlm[base_LQlm_neigh_id + i] * d_stein_LQlm[base_LQlm_neigh_id + i];
//         i=1;
//         d_stein_LQlm[base_LQlm_neigh_id + i] = d_stein_LQlm[base_LQlm_neigh_id + i] / (neigh_num+1);
//         d_stein_ql[c_atom] += d_stein_LQlm[base_LQlm_neigh_id + i] * d_stein_LQlm[base_LQlm_neigh_id + i];
//         for(int i=2; i<(stein_l + 1)*2; i++){
//             d_stein_LQlm[base_LQlm_neigh_id + i] = d_stein_LQlm[base_LQlm_neigh_id + i] / (neigh_num+1);
//             d_stein_ql[c_atom] += 2*d_stein_LQlm[base_LQlm_neigh_id + i] * d_stein_LQlm[base_LQlm_neigh_id + i];
//         }
//         d_stein_ql[c_atom] = sqrt(d_stein_ql[c_atom]*temp4pi_2lplus1);
//     }
// }


// __global__ void dcv_steinhardt_param_calc_LOCAL_kernel(int group_count, int cutoff_Natoms,
//                     int stein_l, int groupbit,
//                     int *d_mask, LAMMPS_NS::tagint *d_group_indices,
//                     LAMMPS_NS::tagint *d_calculated_numneigh, 
//                     int *d_neigh_both_in_r_N,
//                     double *d_stein_qlm, double *d_stein_LQlm,
//                     double *d_stein_ql){
//     int c_atom = blockIdx.x * blockDim.x + threadIdx.x;
//     if(c_atom<group_count){
//         // steinhardt_param_calc_kernel_q4
//         // int stein_l=4;
//         int neigh_num, neigh_tag;
//         double temp4pi_2lplus1;
//         temp4pi_2lplus1 = 12.5663706143591729538505735331/(2*stein_l+1);
//         neigh_num = d_neigh_both_in_r_N[c_atom];
//         int base_LQlm_neigh_id,LQlm_neigh_id;
//         base_LQlm_neigh_id=c_atom*(stein_l + 1)*2;
//         d_stein_ql[c_atom] = 0;
//         if (neigh_num == 0) {
//             return;
//         }
//         for(int i=0; i<(stein_l + 1)*2; i++){
//             d_stein_LQlm[base_LQlm_neigh_id + i] = d_stein_qlm[c_atom + i];
//         }
//         for(int neigh_atom=0; neigh_atom<neigh_num; neigh_atom++){
//             neigh_tag = d_calculated_numneigh[c_atom*cutoff_Natoms + neigh_atom];
//             if (d_mask[neigh_tag]&groupbit){
//                 // TODO: 此处有问题，因为邻居原子可能不在cvgroup中，此时是找不到它的LQlm的
//                 // 使用二分查找法找 neigh_tag 对应在 d_stein_ql 中的位置
//                 int left = 0;
//                 int right = group_count - 1;
//                 // neigh_q4_deN default is 0
//                 while (left <= right) {
//                     int mid = left + (right - left) / 2;
//                     if (d_group_indices[mid] == neigh_tag) {
//                         LQlm_neigh_id = mid * (stein_l + 1) * 2;
//                         for(int i=0; i<(stein_l + 1)*2; i++){
//                             d_stein_LQlm[base_LQlm_neigh_id + i] += d_stein_qlm[LQlm_neigh_id + i];
//                         }
//                         break;
//                     } else if (d_group_indices[mid] < neigh_tag) {
//                         left = mid + 1;
//                     } else {
//                         right = mid - 1;
//                     }
//                 }
//             }
//         }
//         int i=0;
//         d_stein_LQlm[base_LQlm_neigh_id + i] = d_stein_LQlm[base_LQlm_neigh_id + i] / (neigh_num+1);
//         d_stein_ql[c_atom] += d_stein_LQlm[base_LQlm_neigh_id + i] * d_stein_LQlm[base_LQlm_neigh_id + i];
//         i=1;
//         d_stein_LQlm[base_LQlm_neigh_id + i] = d_stein_LQlm[base_LQlm_neigh_id + i] / (neigh_num+1);
//         d_stein_ql[c_atom] += d_stein_LQlm[base_LQlm_neigh_id + i] * d_stein_LQlm[base_LQlm_neigh_id + i];
//         for(int i=2; i<(stein_l + 1)*2; i++){
//             d_stein_LQlm[base_LQlm_neigh_id + i] = d_stein_LQlm[base_LQlm_neigh_id + i] / (neigh_num+1);
//             d_stein_ql[c_atom] += 2*d_stein_LQlm[base_LQlm_neigh_id + i] * d_stein_LQlm[base_LQlm_neigh_id + i];
//         }
//         d_stein_ql[c_atom] = sqrt(d_stein_ql[c_atom]*temp4pi_2lplus1);
//     }
// }


REGISTER_CV("STEINH", MetaD_zqc::Steinhardt::create);