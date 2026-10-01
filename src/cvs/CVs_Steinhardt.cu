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

MetaD_zqc::CV* MetaD_zqc::Steinhardt::create(LAMMPS_NS::LAMMPS *lmp, 
                                            LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check, 
                                            int narg, char **arg, int &i){
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
    // req.cutoff_Natoms = 12;
    req.d_block_size = 128;
    req.cutoff_eps_r = 1e-6;

    std::string temp_name;
    MetaD_zqc::SwitchFunction* found_sw;

    int iarg=4 + i;
    while (iarg < narg) {
        if (strcmp(arg[iarg], "cutoff_r") == 0) {
            ERR_COND((iarg + 1 >= narg) ,"Error: \'cutoff_r\' keyword requires a value");
            req.cutoff_r = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
            iarg += 2;
        // } else if (strcmp(arg[iarg], "cutoff_Natoms") == 0) {
        //     ERR_COND((iarg + 1 >= narg), "Error: \'cutoff_Natoms\' keyword requires an integer");
        //     req.cutoff_Natoms = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
        //     iarg += 2;
        } else if (strcmp(arg[iarg], "cutoff_eps_r") == 0) {
            ERR_COND((iarg + 1 >= narg), "Error: \'cutoff_eps_r\' keyword requires a value");
            req.cutoff_eps_r = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
            iarg += 2;
        } else if (strcmp(arg[iarg], "d_block_size") == 0) {
            ERR_COND((iarg + 1 >= narg), "Error: \'d_block_size\' keyword requires an integer");
            req.d_block_size = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
            ERR_COND(req.d_block_size <= 0, "Error: \'d_block_size\' must be > 0");
            iarg += 2;
        } else if (strcmp(arg[iarg], "SW_FUNC_r") == 0) {
            ERR_COND((iarg + 1 >= narg), "Error: \'SW_FUNC_r\' keyword requires a value");
            temp_name = arg[iarg + 1];
            auto it = Fixmetad->sw_registry.find(temp_name);
            if (it != Fixmetad->sw_registry.end()) {
                found_sw = it->second;
            } else {
                found_sw = nullptr;
            }
            ERR_COND(found_sw == nullptr, "Error: SwitchFunction named %s not found in registry!", temp_name.c_str());
            // 将找到的实例指针存入你的 req 请求结构体中
            req.SW_FUNC_r = found_sw;
            iarg += 2;
        } else if (strcmp(arg[iarg], "SW_FUNC_cv") == 0) {
            ERR_COND((iarg + 1 >= narg), "Error: \'SW_FUNC_cv\' keyword requires a value");
            temp_name = arg[iarg + 1];
            auto it = Fixmetad->sw_registry.find(temp_name);
            if (it != Fixmetad->sw_registry.end()) {
                found_sw = it->second;
            } else {
                found_sw = nullptr;
            }
            ERR_COND(found_sw == nullptr, "Error: SwitchFunction named %s not found in registry!", temp_name.c_str());
            // 将找到的实例指针存入你的 req 请求结构体中
            req.SW_FUNC_cv = found_sw;
            iarg += 2;
        } else {
            break;
        }
    }
    if (req.SW_FUNC_r == nullptr) {
        req.SW_FUNC_r = MetaD_zqc::SwitchFunction::get_default_step();
    }
    if (req.SW_FUNC_cv == nullptr) {
        req.SW_FUNC_cv = MetaD_zqc::SwitchFunction::get_default_step();
    }
    if (strcmp(req.Q_type_str, "L") == 0 && req.SW_FUNC_r->params.type != MetaD_zqc::STEP) {
        double auto_cutoff = MetaD_zqc::SwitchFunction::invert_for_eps(req.SW_FUNC_r->params, req.cutoff_eps_r);
        LOG("Logging: cutoff_r auto-derived from SW_FUNC_r + cutoff_eps_r: %g -> %g (原手动值将被忽略)",
            req.cutoff_r, auto_cutoff);
        req.cutoff_r = auto_cutoff;
    }
    // LOG("Logging: set STEINH as Q_type_str=%s Q_num=%d group_name=%s cutoff_r=%f cutoff_Natoms=%d d_block_size=%d.",
    //                     req.Q_type_str, req.Q_num, req.group_name, req.cutoff_r, req.cutoff_Natoms, req.d_block_size);
    LOG("Logging: set STEINH as Q_type_str=%s Q_num=%d group_name=%s cutoff_r=%f cutoff_eps_r=%g d_block_size=%d.",
                        req.Q_type_str, req.Q_num, req.group_name, req.cutoff_r, req.cutoff_eps_r, req.d_block_size);

    // NeighHub: full list + custom cutoff; Local needs ghost (→ perpetual under BIN)
    MetaD_zqc::NeighSpec nspec;
    nspec.full = 1;
    nspec.use_cutoff = 1;
    nspec.cutoff = req.cutoff_r;
    double temp_neigh_cutoff;
    if (strcmp(req.Q_type_str, "Q") == 0){
        temp_neigh_cutoff = (req.cutoff_r + lmp->neighbor->skin);
    } else if (strcmp(req.Q_type_str, "L") == 0){
        temp_neigh_cutoff = (2.0 * req.cutoff_r + lmp->neighbor->skin);
        nspec.ghost = 1;
    } else {
        temp_neigh_cutoff = (req.cutoff_r + lmp->neighbor->skin);
    }
    if (lmp->comm->get_comm_cutoff() < temp_neigh_cutoff) {
        lmp->comm->cutghostuser = temp_neigh_cutoff;
        LOG_COND(lmp->comm->me == 0, "Increasing communication cutoff to %g for GPU pair style",
                  lmp->comm->cutghostuser);
    }
    const int neigh_id = Fixmetad->neigh_hub.get_or_create(nspec);

    // steinh_requests.push_back(req);
    // // 创建 CV 对象
    // TODO: 需要处理相同envs的合并问题
    MetaD_zqc::Steinhardt_env *temp_env = MetaD_zqc::Steinhardt_env::get_or_create(lmp, 
                            Fixmetad, f_check, req);
    temp_env->neigh_id = neigh_id;
    DEBUG_LOG("Steinhardt_env is %p neigh_id=%d", temp_env, neigh_id);
    std::string env_setNum = temp_env->get_env_key();
    i = iarg;
    return MetaD_zqc::create_steinhardt_cv(lmp, Fixmetad, f_check, 
                            env_setNum, req.group_id, req.Q_num, temp_env, req);
}

MetaD_zqc::Steinhardt* MetaD_zqc::create_steinhardt_cv(LAMMPS_NS::LAMMPS *lmp,
                                LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check,
                                std::string env_setNum, int group_id, int Q_num,
                                MetaD_zqc::Steinhardt_env* my_env,
                                MetaD_zqc::SteinhardtRequest req){
    if (strcmp(req.Q_type_str, "Q") == 0){
        if (Q_num==3){
            return new MetaD_zqc::STEIN_QL<3>(lmp, Fixmetad, f_check, env_setNum, group_id, Q_num, my_env, req.d_block_size);
        } else if (Q_num==4){
            return new MetaD_zqc::STEIN_QL<4>(lmp, Fixmetad, f_check, env_setNum, group_id, Q_num, my_env, req.d_block_size);
        } else if (Q_num==6){
            return new MetaD_zqc::STEIN_QL<6>(lmp, Fixmetad, f_check, env_setNum, group_id, Q_num, my_env, req.d_block_size);
        }
    } else if (strcmp(req.Q_type_str,"L") == 0){
        if (Q_num==3){
            return new MetaD_zqc::STEIN_LocalQL<3>(lmp, Fixmetad, f_check, env_setNum, group_id, my_env, req.d_block_size, req);
        } else if (Q_num==4){
            return new MetaD_zqc::STEIN_LocalQL<4>(lmp, Fixmetad, f_check, env_setNum, group_id, my_env, req.d_block_size, req);
        } else if (Q_num==6){
            return new MetaD_zqc::STEIN_LocalQL<6>(lmp, Fixmetad, f_check, env_setNum, group_id, my_env, req.d_block_size, req);
        }
    }
    return nullptr;
}

std::map<std::string, MetaD_zqc::Steinhardt_env*> MetaD_zqc::Steinhardt_env::env_pool;

namespace {
std::string steinhardt_env_key(int group_id, double cutoff_r, double cutoff_eps_r,
                               bool loc_flag, const MetaD_zqc::SwitchFunction* sw) {
    std::ostringstream oss;
    oss << group_id << "_"
        << std::setprecision(17) << cutoff_r << "_"
        << cutoff_eps_r << "_"
        << (loc_flag ? 1 : 0) << "_";
    if (sw == nullptr) {
        oss << "nosw";
    } else {
        const MetaD_zqc::SwitchFunctionRequest& p = sw->params;
        oss << static_cast<int>(p.type) << "_"
            << p.r_0 << "_" << p.d_0 << "_" << p.alpha << "_"
            << p.n << "_" << p.m;
    }
    return oss.str();
}
}

MetaD_zqc::Steinhardt_env* MetaD_zqc::Steinhardt_env::get_or_create(LAMMPS_NS::LAMMPS *lmp,
                                            LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check,
                                            MetaD_zqc::SteinhardtRequest req
                                            ) {
    int group_id = req.group_id;
    double cutoff_r = req.cutoff_r;
    double cutoff_eps_r= req.cutoff_eps_r;
    bool LOC_flag = (strcmp(req.Q_type_str,"L") == 0);
    // Radial switch is part of the environment: different r0/d0/n/m/alpha must not share one env.
    std::string key = steinhardt_env_key(group_id, cutoff_r, cutoff_eps_r, LOC_flag, req.SW_FUNC_r);
    // 2. check if the environment already exist in the pool
    if (!(env_pool.count(key))) {
        if (LOC_flag){
            // 3. new environment and store it in the pool if not exist
            MetaD_zqc::STEIN_LocalQL_env *new_env = new MetaD_zqc::STEIN_LocalQL_env(lmp, Fixmetad, f_check, 
                                                        req);
            env_pool[key] = new_env; // store the new environment in the pool
        } else {
            // 3. new environment and store it in the pool if not exist
            MetaD_zqc::Steinhardt_env *new_env = new MetaD_zqc::Steinhardt_env(lmp, Fixmetad, f_check, 
                                                        req);
            env_pool[key] = new_env; // store the new environment in the pool
        }
    }
    env_pool[key]->register_env(); // increase reference count
    return env_pool[key]; // if exits, return the existing environment
}


std::string MetaD_zqc::Steinhardt_env::get_env_key(){
    return steinhardt_env_key(group_id, cutoff_r, cutoff_eps_r, LOC_flag, my_r_SWfunc);
}

MetaD_zqc::Steinhardt_env::Steinhardt_env(LAMMPS_NS::LAMMPS *lmp, 
             LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check,
             MetaD_zqc::SteinhardtRequest req)
    : CV_info(lmp, Fixmetad, f_check),
      group_id(req.group_id),
      cutoff_r(req.cutoff_r),
      cutoff_eps_r(req.cutoff_eps_r){
    this->lmp = lmp;
    this->f_check = f_check;
    this->Fixmetad = Fixmetad;

    this->error = lmp->error;

    this->my_r_SWfunc = req.SW_FUNC_r;
    this->LOC_flag = (req.Q_type_str != nullptr && strcmp(req.Q_type_str, "L") == 0);

    pbc_x = (lmp->domain->xperiodic == 1);
    pbc_y = (lmp->domain->yperiodic == 1);
    pbc_z = (lmp->domain->zperiodic == 1);
    // 这里可以添加一些初始化代码，例如分配内存、设置默认值等
    DEBUG_LOG("Steinhardt_env initialized with cutoff_r=%g and cutoff_eps_r=%g", cutoff_r, cutoff_eps_r);

    // const char *group_name = arg[1];
    groupbit = lmp->group->bitmask[group_id]; // 关键：存储原子组位 掩码
    init_flag = false;
    
    // group_dminneigh = new double [2]; //inintial
    // neigh_in_cutoff_r = new int [2]; //inintial
    // neigh_both_in_r_N = new int [2]; //inintial
    lmp->memory->create(h_neigh_in_switching, ((lmp->atom)->nmax), "metad:STEIN_QL:h_neigh_in_switching");
    lmp->memory->create(h_calculated_firstneigh_ptrs, ((lmp->atom)->nmax), "metad:STEIN_QL:h_calculated_firstneigh_ptrs");
    lmp->memory->create(h_group_numneigh, 0, "metad:STEIN_QL:h_group_numneigh");
    lmp->memory->create(h_x_flat, 0, "metad:STEIN_QL:h_x_flat");
    lmp->memory->create(h_group_indices, 0, "metad:STEIN_QL:h_group_indices");
    lmp->memory->create(h_firstneigh_ptrs, 0, "metad:STEIN_QL:h_firstneigh_ptrs");
    // lmp->memory->create(group_dminneigh, 0, "metad:STEIN_QL:group_dminneigh");
    lmp->memory->create(h_full_to_half, 0, "metad:STEIN_QL:h_full_to_half");
    lmp->memory->create(neigh_in_cutoff_r, 0, "metad:STEIN_QL:neigh_in_cutoff_r");
    lmp->memory->create(neigh_both_in_r_N, 0, "metad:STEIN_QL:neigh_both_in_r_N");
    lmp->memory->create(calculated_numneigh, 0, "metad:STEIN_QL:calculated_numneigh");

    std::memset(h_neigh_in_switching, 0, (lmp->atom)->nmax * sizeof(double));
    std::memset(h_calculated_firstneigh_ptrs, 0, (lmp->atom)->nmax * sizeof(LAMMPS_NS::tagint));

    // comment name
    register_buffer(d_group_numneigh,"d_group_numneigh");
    register_buffer(d_x_flat,"d_x_flat");
    register_buffer(d_mask, "d_mask");
    register_buffer(d_group_indices,"d_group_indices");
    register_buffer(d_firstneigh_ptrs,"d_firstneigh_ptrs");
    // register_buffer(d_group_dminneigh,"d_group_dminneigh");
    register_buffer(d_neigh_in_cutoff_r,"d_neigh_in_cutoff_r");
    register_buffer(d_neigh_both_in_r_N,"d_neigh_both_in_r_N");
    register_buffer(d_half_pair_i,"d_half_pair_i");
    register_buffer(d_half_pair_j,"d_half_pair_j");
    register_buffer(d_half_to_full,"d_half_to_full");
    register_buffer(d_active_pair_mask,"d_active_pair_mask");
    register_buffer(d_active_pair_ids,"d_active_pair_ids");
    register_buffer(d_full_to_half,"d_full_to_half");
    register_buffer(d_calculated_numneigh,"d_calculated_numneigh");
    register_buffer(d_neigh_in_switching,"d_neigh_in_switching");
    register_buffer(d_calculated_firstneigh_ptrs,"d_calculated_firstneigh_ptrs");

    d_neigh_in_switching.grow_to(lmp->atom->nmax, __FILE__, __LINE__);
    d_calculated_firstneigh_ptrs.grow_to(lmp->atom->nmax, __FILE__, __LINE__);
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
    lmp->memory->destroy(h_x_flat);
    lmp->memory->destroy(h_full_to_half);
    lmp->memory->destroy(h_group_indices);
    lmp->memory->destroy(h_firstneigh_ptrs);
    // lmp->memory->destroy(group_dminneigh);
    lmp->memory->destroy(neigh_in_cutoff_r);
    lmp->memory->destroy(neigh_both_in_r_N);
    lmp->memory->destroy(calculated_numneigh);
    lmp->memory->destroy(h_neigh_in_switching);
    lmp->memory->destroy(h_calculated_firstneigh_ptrs);
}

template <int L>
MetaD_zqc::STEIN_QL<L>::STEIN_QL(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check, 
                             std::string env_setNum, int group_id, int stein_l, 
                             MetaD_zqc::Steinhardt_env* my_env,
                             int d_block_size)
                        : Steinhardt(lmp, Fixmetad, f_check),
                            env_setNum(env_setNum),
                        //   group_id(group_id),
                            stein_l(stein_l),
                            my_env(my_env),
                            d_block_size(d_block_size){
    
    this->lmp = lmp;
    this->f_check = f_check;
    this->Fixmetad = Fixmetad;
    this->error = lmp->error;

    // my_averager = new MetaD_zqc::CUBAverager();
    my_averager = new MetaD_zqc::KahanAverager();
    num_elements = 2*(L+1); // Qlm needs 2*(l+1)
    DEBUG_LOG("Logging: New a Stein_Q%d file, will generate %d lines in GPU,\n     with cutoff_r=%g, cutoff_eps_r=%g",
                stein_l,d_block_size, my_env->cutoff_r, my_env->cutoff_eps_r);
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
    int Threads_own_atoms = lmp->atom->nlocal;
    lmp->memory->create(stein_q, 0, "metad:STEIN_QL:stein_q");
    lmp->memory->grow(stein_q, Threads_own_atoms, "metad:STEIN_QL:stein_q");
    lmp->memory->create(h_stein_qlm, 0, "metad:STEIN_QL:h_stein_qlm");
    lmp->memory->grow(h_stein_qlm, Threads_own_atoms*num_elements, "metad:STEIN_QL:h_stein_qlm");
    lmp->memory->create(h_dcvdx_x, 0, "metad:STEIN_QL:h_dcvdx_x");
    lmp->memory->create(h_dcvdx_y, 0, "metad:STEIN_QL:h_dcvdx_y");
    lmp->memory->create(h_dcvdx_z, 0, "metad:STEIN_QL:h_dcvdx_z");

    
    // comment name
    register_buffer(d_stein_ql,"d_stein_ql");
    register_buffer(d_stein_Ylm,"d_stein_Ylm");
    register_buffer(d_dYlm_dr, "d_dYlm_dr");
    register_buffer(d_dcvdx,"d_dcvdx");
    register_buffer(d_stein_qlm,"d_stein_qlm");
    register_buffer(d_stein_LQlm,"d_stein_LQlm");
    register_buffer(d_a_virial,"d_a_virial");
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
    lmp->memory->destroy(h_a_virial);
    // release all alloc
    // the GpuBuffer will automatically release its memory, 
    // so we don't need to manually free it here
    lmp->memory->destroy(h_dcvdx_x);
    lmp->memory->destroy(h_dcvdx_y);
    lmp->memory->destroy(h_dcvdx_z);
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
    d_mask.grow_to(((atom)->nlocal+(atom)->nghost), __FILE__, __LINE__);
    // SAFE_CUDA_MEMCPY((d_mask.ptr),(mask),(((atom)->nlocal+(atom)->nghost))*sizeof(int),cudaMemcpyHostToDevice,f_check);
    d_mask.upload_from(mask, ((atom)->nlocal+(atom)->nghost));

    // set up nvidia thread number
    block_num = ((group_count) + d_block_size - 1)/d_block_size;
    N = d_block_size*block_num;
    // LOG_COND(((group_count)<(cutoff_Natoms)),"Warning: group_count(%lld) < cutoff_Natoms(%lld), please check your system !",(long long)group_count, (long long)cutoff_Natoms);
    LOG_COND((((box_x)<2*(cutoff_r))||((box_y)<2*(cutoff_r))||((box_z)<2*(cutoff_r))),"Warning: box < cutoff_r, please check your system !");
}

void MetaD_zqc::Steinhardt_env::get_env(){
    // DEBUG_LOG("im in get_env, current step is %lld, last_update_step is %lld", (long long)lmp->update->ntimestep, (long long)this->last_update_step);
    // if (lmp->update->ntimestep == this->last_update_step){
    //     return;
    // }
    size_t datalen = 0;
    atom = lmp->atom;
    LAMMPS_NS::tagint atom_all = atom->nlocal + atom->nghost;
    // =======从 NeighHub 取已 ensure 的 list=========
    ERR_COND((neigh_id < 1),"STEIN_QL env has invalid neigh_id.");
    Fixmetad->neigh_hub.ensure(neigh_id);
    nlist = Fixmetad->neigh_hub.list(neigh_id);
    ERR_COND((nlist == nullptr),"STEIN_QL CV failed to find neighbor list now.");
    numneigh = nlist->numneigh;
    firstneigh = nlist->firstneigh;
    // =======防止lammps运行过程体积更改==========
    ERR_COND((lmp->domain == NULL),"domain list not initialized");
    box_x = (pbc_x) ? lmp->domain->xprd : INFINITY;
    box_y = (pbc_y) ? lmp->domain->yprd : INFINITY;
    box_z = (pbc_z) ? lmp->domain->zprd : INFINITY;

    // utilize different environment set
    // such as neighbor list, atom position, box size, to get the local structure information for each atom in the group
    // 昂贵展平只在 Neighbor 重建时做（与旧逻辑一致）；勿用 last_update_step 每步触发。
    const bool skip_neigh_flatten =
        (lmp->update->ntimestep > lmp->neighbor->lastcall) &&
        (lmp->update->ntimestep != 1) &&
        (numneigh != nullptr) &&
        init_flag &&
        (last_neigh_lastcall_ == (long long)lmp->neighbor->lastcall);
    if (skip_neigh_flatten) {
        DEBUG_LOG("we skip rebuild in environment when %lld.", (long long)lmp->neighbor->lastcall);
    } else {
        // =========================================================================
        // neighbour list and its copy to devise
        // h_group_indices / d_group_indices: where the group atoms in locals' tag
        // =========================================================================
        DEBUG_LOG("cutoff_eps_r is %g",cutoff_eps_r);
        DEBUG_LOG("cutoff_r is %f",cutoff_r);
        DEBUG_LOG("group_count is %d",group_count);
        // =========================================================================
        // neighbour list and its copy to devise
        // h_group_indices / d_group_indices: where the group atoms in locals' tag
        // =========================================================================
        // DEBUG_LOG("lastcall = %d", lmp->neighbor->lastcall);
        // int *d_group_indices;
        // SAFE_CUDA_FREE(d_group_indices);
        // SAFE_CUDA_MALLOC(&d_group_indices, (atom_all)*sizeof(int), f_check);
        d_group_indices.grow_to(atom_all, __FILE__, __LINE__);
        d_group_indices.upload_from(h_group_indices, lmp->atom->nlocal);
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
        ERR_COND((nlist->ilist == NULL),"ilist not initialized");
        // Ghost full lists: only inum+gnum entries are valid (via ilist).
        // Blindly scanning atom_all caused 2nd-run segfaults (wild firstneigh[i]).
        const int nlist_n = nlist->inum + nlist->gnum;
        ERR_COND((nlist_n <= 0),"neighbor list inum+gnum == 0");
        ERR_COND((nlist_n > atom_all),"neighbor list inum+gnum > nlocal+nghost");
        // 2. creating number array of start num in different c_atom's neighbor
        // LAMMPS_NS::tagint *h_group_numneigh = new LAMMPS_NS::tagint[group_count + 1];
        // LAMMPS_NS::tagint *d_group_numneigh;
        datalen = atom_all + 1;
        lmp->memory->grow(h_group_numneigh, datalen, "STEIN_QL:h_group_numneigh");
        d_group_numneigh.grow_to(datalen, __FILE__, __LINE__);
        DEBUG_LOG_COND((h_group_numneigh == NULL),"h_group_numneigh list not initialized");
        // 3. 逐原子拷贝邻居列表数据到GPU,现在更改为将所有原子的邻居拷贝到显存
        //    （经 ilist 填充；不在 list 中的 local index jnum=0）
        DEBUG_LOG("group_count=%d nlist_n=%d inum=%d gnum=%d", group_count, nlist_n, nlist->inum, nlist->gnum);
        std::vector<int> jnum_of((size_t)atom_all, 0);
        for (int ii = 0; ii < nlist_n; ++ii) {
            const int i = nlist->ilist[ii];
            ERR_COND((i < 0 || i >= atom_all),"ilist[%d]=%d out of range atom_all=%d", ii, i, (int)atom_all);
            ERR_COND((numneigh[i] > 0 && firstneigh[i] == nullptr),
                     "firstneigh[%d] is null with jnum=%d", i, numneigh[i]);
            jnum_of[(size_t)i] = numneigh[i];
            DEBUG_LOG("ii=%d, tag=%d, jnum=%d", ii, i, numneigh[i]);
        }
        h_group_numneigh[0] = 0;
        for (int gr_i = 0; gr_i < group_count; gr_i++) {
            int i = h_group_indices[gr_i]; // 获取原子索引
            int jnum = numneigh[i]; // 邻居数量
            h_group_numneigh[gr_i+1] = h_group_numneigh[gr_i] + jnum;
            DEBUG_LOG("gr_i=%d, tag=%d, jnum=%d, sum=%d", gr_i, i,jnum,h_group_numneigh[gr_i+1]);
        }
        d_group_numneigh.upload_from(h_group_numneigh, (atom_all + 1));
        // SAFE_CUDA_MEMCPY(d_group_numneigh.ptr,h_group_numneigh,(atom_all + 1)*sizeof(LAMMPS_NS::tagint),cudaMemcpyHostToDevice,f_check);
        LAMMPS_NS::tagint all_neigh_pairs = h_group_numneigh[group_count];
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
        // grow(0) → null; keep at least 1 slot for safe grow/upload
        const LAMMPS_NS::tagint grow_pairs = (all_neigh_pairs > 0) ? all_neigh_pairs : 1;
        lmp->memory->grow(h_firstneigh_ptrs, grow_pairs, "STEIN_QL:h_firstneigh_ptrs");
        lmp->memory->grow(h_full_to_half, grow_pairs, "STEIN_QL:h_full_to_half");
        std::fill(h_full_to_half, h_full_to_half + grow_pairs, (LAMMPS_NS::tagint)-1);
        
        LAMMPS_NS::tagint ba_i;
        LAMMPS_NS::tagint nnumber;
        std::vector<LAMMPS_NS::tagint> half_i;
        std::vector<LAMMPS_NS::tagint> half_j;
        std::vector<LAMMPS_NS::tagint> h_half_to_full;
        n_half_candidates = 0;
        d_full_to_half.grow_to(grow_pairs, __FILE__, __LINE__);
        // SAFE_CUDA_FREE(d_firstneigh_ptrs);
        // SAFE_CUDA_MALLOC(&d_firstneigh_ptrs, (all_neigh_pairs) * sizeof(int),f_check); // 分配设备端指针数组
        d_firstneigh_ptrs.grow_to(grow_pairs, __FILE__, __LINE__);
        DEBUG_LOG("generate d_firstneigh_ptrs, h_group_numneigh[atom_all]=%d", (int)all_neigh_pairs);
        for (int gr_i = 0; gr_i < group_count; gr_i++) {
            int loc_i = h_group_indices[gr_i]; // 获取原子索引
            ba_i = h_group_numneigh[gr_i];
            nnumber = h_group_numneigh[gr_i+1]-h_group_numneigh[gr_i];
            DEBUG_LOG("h_group_numneigh=%d, num=%d" ,ba_i,nnumber);
            if (nnumber <= 0) continue;
            // LAMMPS firstneigh 高位可能编码 special bond 标志，必须剥掉 NEIGHMASK
            ERR_COND((firstneigh[loc_i] == nullptr),"firstneigh[%d] null", loc_i);
            for (LAMMPS_NS::tagint jj = 0; jj < nnumber; ++jj) {
                int loc_j = firstneigh[loc_i][jj] & NEIGHMASK;
                int e = ba_i + jj;
                int tag_i = lmp->atom->tag[loc_i];
                int tag_j = lmp->atom->tag[loc_j];
                h_firstneigh_ptrs[e] = loc_j;
                if (((lmp->atom->mask[loc_j] & groupbit) == 0) || (tag_i <= tag_j)){
                    half_i.push_back(loc_i);
                    half_j.push_back(loc_j);
                    h_half_to_full.push_back(e);
                    h_full_to_half[e] = n_half_candidates;
                    n_half_candidates++;
                } else {
                    h_full_to_half[e] = -1;
                }
            }
        }
        if (all_neigh_pairs > 0) {
            d_firstneigh_ptrs.upload_from(h_firstneigh_ptrs, all_neigh_pairs);
            d_full_to_half.upload_from(h_full_to_half, all_neigh_pairs);
        }
        // n_half_candidates = half_i.size();
        // 完成ij的half_pair对
        d_half_pair_i.grow_to(n_half_candidates, __FILE__, __LINE__);
        d_half_pair_j.grow_to(n_half_candidates, __FILE__, __LINE__);
        d_half_to_full.grow_to(n_half_candidates, __FILE__, __LINE__);
        d_active_pair_mask.grow_to(n_half_candidates, __FILE__, __LINE__);
        d_active_pair_ids.grow_to(n_half_candidates, __FILE__, __LINE__);
        d_half_pair_i.upload_from(half_i.data(), n_half_candidates);
        d_half_pair_j.upload_from(half_j.data(), n_half_candidates);
        d_half_to_full.upload_from(h_half_to_full.data(), n_half_candidates);

        // SAFE_CUDA_MEMCPY(d_firstneigh_ptrs.ptr,h_firstneigh_ptrs,
        //     (all_neigh_pairs) * sizeof(int),cudaMemcpyHostToDevice,f_check);
        DEBUG_LOG_COND((d_firstneigh_ptrs.ptr == NULL),"d_firstneigh_ptrs list not initialized");
        if (all_neigh_pairs >= 3) {
            DEBUG_LOG("d_firstneigh_ptrs list %d %d %d" ,h_firstneigh_ptrs[1],h_firstneigh_ptrs[2],h_firstneigh_ptrs[3]);
        }
        DEBUG_LOG("generate end d_firstneigh_ptrs");
        if (!init_flag) {init_flag = true;}
        last_neigh_lastcall_ = (long long)lmp->neighbor->lastcall;
    }
    ERR_COND((h_group_numneigh == nullptr),"h_group_numneigh null before use");
    LAMMPS_NS::tagint all_neigh_pairs = h_group_numneigh[group_count];
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
    // SAFE_CUDA_MEMCPY(d_x_flat.ptr,h_x_flat,((atom->nlocal + atom->nghost) * 3)*sizeof(double),cudaMemcpyHostToDevice, f_check);
    d_x_flat.upload_from(h_x_flat, ((atom->nlocal + atom->nghost) * 3));
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
    atom_all = (atom_all > N) ? atom_all : N;
    d_neigh_in_cutoff_r.grow_to(atom_all, __FILE__, __LINE__);
    d_neigh_in_cutoff_r.clear_async();
    d_neigh_in_switching.grow_to(atom_all, __FILE__, __LINE__);
    d_neigh_in_switching.clear_async();
    lmp->memory->grow(h_neigh_in_switching, atom_all, "STEIN_QL:h_stein_qlm");
    d_calculated_numneigh.grow_to(all_neigh_pairs, __FILE__, __LINE__);
    d_calculated_numneigh.clear_async();
    d_active_pair_ids.clear_async();
    d_active_pair_mask.clear_async();
    // d_is_pure_J.grow_to(atom_all, __FILE__, __LINE__);
    // d_is_pure_J.clear_async();
    DEBUG_LOG("release end");

    // =========================================================================
    // start kernel for calculate 
    // d_neigh_in_cutoff_r  : how many neigh atoms in cutoff_r (\sigma r_cut less than)
    // d_neigh_in_switching : sum of sigma(rij) for j in neigh(i)
    // =========================================================================
    DEBUG_LOG("box_lim x:%f y:%f z:%f max:%f" ,box_x,box_y,box_z,box_x+box_y+box_z );
    DEBUG_LOG("neigh finding .......");
    DEBUG_LOG("i will start a kernel");
    // kernel function will run
    cudaError_t launchErr = cudaGetLastError();
    cudaStream_t lmp_stream = 0;
    cudaDeviceSynchronize(); // waiting memory
    double cutoff_rsq = cutoff_r*cutoff_r;

    // cudaDeviceSynchronize(); //catch kernel done
    launchErr = cudaGetLastError();
    get_environment_Steinhardt_Q<<<block_num,d_block_size>>>
      ( my_r_SWfunc->params,
        group_count, 0,
        cutoff_r, cutoff_eps_r,
        // in
        d_group_indices.ptr, d_group_numneigh.ptr, d_firstneigh_ptrs.ptr, 
        d_x_flat.ptr, d_full_to_half.ptr,
        //   out
        d_neigh_in_cutoff_r.ptr, d_active_pair_mask.ptr, 
        d_neigh_in_switching.ptr, d_calculated_numneigh.ptr) ;
    DEBUG_LOG("env refresh out, kernel launched");

    cudaError_t syncErr = cudaDeviceSynchronize();
    ERR_COND((syncErr != cudaSuccess),"Kernel execution error: %s\n", cudaGetErrorString(syncErr));

    DEBUG_LOG("group_count=%d, atom_all=%d, all_neigh_pairs=%lld, group_count=%d", 
           group_count, atom_all, (long long)all_neigh_pairs, group_count);

    // 清空，全写0
    d_calculated_firstneigh_ptrs.grow_to(atom->nmax+1, __FILE__, __LINE__);
    d_calculated_firstneigh_ptrs.clear_async();
    lmp->memory->grow(h_calculated_firstneigh_ptrs, atom->nmax+1, "STEIN_LocalQL:h_calculated_firstneigh_ptrs");
    // a[n+1] = b[0]+...+b[n], a[0]=0
    d_neigh_in_cutoff_r.scan_to(d_calculated_firstneigh_ptrs, 
                            group_count, lmp_stream);
    d_active_pair_mask.flag_to(d_active_pair_ids, 
                            n_half_candidates, &n_active_pairs, lmp_stream);
    d_calculated_firstneigh_ptrs.download_to(h_calculated_firstneigh_ptrs, 
                            group_count+1, lmp_stream, __FILE__, __LINE__);
    cudaStreamSynchronize(lmp_stream);
    num_of_all_calc_fullpair = h_calculated_firstneigh_ptrs[group_count];

    DEBUG_LOG("num_of_all_calc_fullpair=%lld (from scan of %d elements), last raw d_neigh_in_cutoff_r[group_count]=?",
            (long long)num_of_all_calc_fullpair, group_count);

    syncErr = cudaDeviceSynchronize();
    ERR_COND((syncErr != cudaSuccess),"Kernel execution error: %s\n", cudaGetErrorString(syncErr));

    DEBUG_LOG("im out");
    DEBUG_LOG("neigh find finished");


    // return the array for neigh
    DEBUG_LOG("copy result array to cpu: group_dminneigh, neigh_in_cutoff_r, neigh_both_in_r_N");
    // DEBUG_LOG_COND((group_dminneigh == NULL),"group_dminneigh list not initialized");
    DEBUG_LOG_COND((neigh_in_cutoff_r == NULL),"group_dminneigh list not initialized");
    DEBUG_LOG_COND((neigh_both_in_r_N == NULL),"group_dminneigh list not initialized");
    lmp->memory->grow(neigh_in_cutoff_r, (group_count), "STEIN_QL:neigh_in_cutoff_r");
    d_neigh_in_cutoff_r.download_to(neigh_in_cutoff_r,group_count, lmp_stream, __FILE__, __LINE__);
    // SAFE_CUDA_MEMCPY(neigh_in_cutoff_r, d_neigh_in_cutoff_r.ptr,
    //   (group_count) * sizeof(int), cudaMemcpyDeviceToHost,f_check);
    lmp->memory->grow(h_neigh_in_switching, atom->nlocal + atom->nghost, "STEIN_QL:h_neigh_in_switching");
    d_neigh_in_switching.download_to(h_neigh_in_switching, atom->nlocal + atom->nghost, lmp_stream, __FILE__, __LINE__);
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



template <int L>
void MetaD_zqc::STEIN_QL<L>::environment(){
    DEBUG_LOG("last_update_step is %lld in %d, group_count=%d", (long long)my_env->last_update_step, L, my_env->group_count);
    const bool neigh_rebuilt_now =
        (lmp->update->ntimestep <= lmp->neighbor->lastcall) ||
        (lmp->update->ntimestep == 1) || !init_flag;
    const bool env_not_ready_this_step =
        (lmp->update->ntimestep > my_env->last_update_step);
    if (neigh_rebuilt_now || env_not_ready_this_step) {
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
    ERR_COND(my_env->neigh_id < 1, "STEIN_QL env has invalid neigh_id.");
    Fixmetad->neigh_hub.ensure(my_env->neigh_id);
    my_env->nlist = Fixmetad->neigh_hub.list(my_env->neigh_id);
    ERR_COND(my_env->nlist == nullptr, "STEIN_QL CV failed to find neighbor list now.");
    my_env->numneigh = my_env->nlist->numneigh;
    my_env->firstneigh = my_env->nlist->firstneigh;
    my_env->mask = lmp->atom->mask;
    
    compute_Q_peratoms();
}

template <int L>
void MetaD_zqc::STEIN_QL<L>::compute_Q_peratoms(){
    // =======接受邻居更新消息,进行与设备端通信===========
    // 仅 Neighbor 重建步 refresh；勿用 last_update_step 每步触发。
    const bool neigh_rebuilt_now =
        (lmp->update->ntimestep <= lmp->neighbor->lastcall) ||
        (lmp->update->ntimestep == 1) || !this->init_flag;
    const bool env_not_ready_this_step =
        (lmp->update->ntimestep > my_env->last_update_step);
    if (neigh_rebuilt_now || env_not_ready_this_step) {
        my_env->refresh_lmpbox();
        block_num = my_env->block_num;
        N = my_env->N;
        this->init_flag = true;
        DEBUG_LOG("refresh_lmpbox done, group_count=%d",my_env->group_count);
    }
    {
        int Threads_own_atoms = lmp->atom->nlocal + lmp->atom->nghost;
        lmp->memory->grow(stein_q, Threads_own_atoms, "metad:STEIN_locQL:cv_bound");
    }
    DEBUG_LOG("group_count=%lld",(long long)my_env->group_count);

    // 2. calculate atoms' environment
    DEBUG_LOG("environment function in, env_setNum is %s",env_setNum.c_str());
    environment();
    DEBUG_LOG("environment function out");

    // 3. calculate atoms' other things
    // steinhardt_param(Q_hybrid);
    steinhardt_param_calc(stein_q);
    
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
            ql_ave_local += Si * my_cv_SWfunc->f(Si);
        }
    }
    MPI_Allreduce(&ql_ave_local, &cv_value, 1, MPI_DOUBLE, MPI_SUM, lmp->world);
    DEBUG_LOG("group_count = %d, compute_cv_SW_FUNC = %g",group_count, cv_value);
    return cv_value;
}

template <int L>
void MetaD_zqc::STEIN_QL<L>::get_dcvdx_AVE(double cv_value, double *dcvdx){
    local_reduce_mode = 0;
    apply_get_dcvdx(cv_value, dcvdx, local_reduce_mode);
}

template <int L>
void MetaD_zqc::STEIN_QL<L>::get_dcvdx_SW_FUNC(double cv_value, double *dcvdx){
    local_reduce_mode = 1;
    apply_get_dcvdx(cv_value, dcvdx, local_reduce_mode);
}


template <int L>
void MetaD_zqc::STEIN_QL<L>::bias_force_AVE(double dVdcv){
    // pass
    local_reduce_mode = 0;
    apply_bias_force(dVdcv, local_reduce_mode);
}


template <int L>
void MetaD_zqc::STEIN_QL<L>::bias_force_SW_FUNC(double dVdcv){
    // pass
    local_reduce_mode = 1;
    apply_bias_force(dVdcv, local_reduce_mode);
}


template <int L>
void MetaD_zqc::STEIN_QL<L>::steinhardt_param_calc(double *stein_ql){
    double cutoff_eps_r = my_env->cutoff_eps_r;
    int last_group_count = my_env->last_group_count;
    int group_count = my_env->group_count;
    int Threads_own_atoms = lmp->atom->nlocal + lmp->atom->nghost;
    LAMMPS_NS::tagint all_neigh_pairs = my_env->h_group_numneigh[group_count];
    // TODO: we can change the cuda stream to lammps stream, 
    // but we need to make sure that the stream is synchronized before we copy data back to host. 
    // For now, we will use the default stream.
    cudaStream_t lammps_stream = 0; // Assuming you want to use the default stream. Adjust if you have a specific stream.
    // in class protect
    // result array
    // every q has <2*L + 1> qlm, with complex we will times 2
    // double *h_stein_qlm = new double [group_count*(L + 1)*2];
    // for the further concentrate we need to calculate qlm*Neigh, with comple

    d_stein_Ylm.grow_to((all_neigh_pairs*(L + 1)*2), __FILE__, __LINE__);

    // SAFE_CUDA_FREE(d_stein_ql);
    // SAFE_CUDA_MALLOC(&d_stein_ql, Threads_own_atoms*sizeof(double), f_check);
    d_stein_ql.grow_to(Threads_own_atoms, __FILE__, __LINE__);
    d_stein_ql.clear_async();
    d_stein_qlm.grow_to((Threads_own_atoms*(L + 1)*2), __FILE__, __LINE__);
    d_stein_qlm.clear_async();
    d_a_virial.grow_to((Threads_own_atoms*6), __FILE__, __LINE__);
    d_a_virial.clear_async();

    DEBUG_LOG("i will start a kernel of ql");
    cudaDeviceSynchronize(); // waiting memory
    call_steinhardt_cv_ql_i_kernel();
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

    // prepare for forward
    lmp->memory->grow(stein_ql, (Threads_own_atoms), "STEIN_QL:stein_ql");
    lmp->memory->grow(h_stein_qlm, (Threads_own_atoms)*num_elements, "STEIN_QL:h_stein_qlm");
    d_stein_ql.download_to(stein_ql, (Threads_own_atoms), 0, __FILE__, __LINE__);
    d_stein_qlm.download_to(h_stein_qlm, (Threads_own_atoms)*num_elements, 0, __FILE__, __LINE__);
}


template <int L>
void MetaD_zqc::STEIN_QL<L>::apply_get_dcvdx(double cv_value, double *dcvdx, int mode){
    
    int group_count = my_env->group_count;
    int Threads_own_atoms = lmp->atom->nlocal+lmp->atom->nghost;
    int last_group_count = my_env->last_group_count;
    size_t datalen = 0;
    

    // 都是GPU计算的数组
    // d_stein_qlm.grow_to(((Threads_own_atoms)*(L + 1)*2), __FILE__, __LINE__);

    datalen = (Threads_own_atoms * (6));
    lmp->memory->grow(h_a_virial, datalen, "STEIN_QL:h_a_virial");
    d_a_virial.grow_to(datalen, __FILE__, __LINE__);
    d_a_virial.clear_async();

    datalen = (Threads_own_atoms*3);
    lmp->memory->grow(h_dcvdx, datalen, "STEIN_QL:h_dcvdx");
    d_dcvdx.grow_to(datalen, __FILE__, __LINE__);
    d_dcvdx.clear_async();

    datalen = (group_count*3*2);
    lmp->memory->grow(h_dYlm_dr, datalen, "STEIN_QL:h_dYlm_dr");
    d_dYlm_dr.grow_to(datalen, __FILE__, __LINE__);
    d_dYlm_dr.clear_async();

    my_env->d_neigh_in_switching.download_to(my_env->h_neigh_in_switching, Threads_own_atoms, 0, __FILE__, __LINE__);


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

    d_stein_qlm.upload_from(h_stein_qlm, ((Threads_own_atoms)*(L + 1)*2));
    d_stein_ql.upload_from(stein_q, Threads_own_atoms);
    my_env->d_neigh_in_switching.upload_from(my_env->h_neigh_in_switching, Threads_own_atoms);
    // SAFE_CUDA_MEMCPY(d_stein_qlm.ptr, h_stein_qlm, ((Threads_own_atoms)*(L + 1)*2)*sizeof(double), cudaMemcpyHostToDevice,f_check);
    // SAFE_CUDA_MEMCPY(d_stein_ql.ptr, stein_q, Threads_own_atoms*sizeof(double), cudaMemcpyHostToDevice,f_check);
    // SAFE_CUDA_MEMCPY(my_env->d_neigh_both_in_r_N.ptr, my_env->neigh_both_in_r_N, Threads_own_atoms*sizeof(int), cudaMemcpyHostToDevice,f_check);



    // dcv_steinhardt_param_calc_kernel_q4(
    //     file, cutoff_Natoms, group_count, groupbit,
    //     mask, h_group_indices, calculated_numneigh,
    //     neigh_both_in_r_N, group_dminneigh,
    //     h_stein_qlm, h_stein_Ylm, stein_q,
    //     h_dYlm_dr, h_dcvdx);
    DEBUG_LOG("i will start a kernel of ql");
    cudaDeviceSynchronize(); // waiting memory
    if (mode==0){
        call_steinhardt_dcv_AVE_kernel();
    } else if (mode==1){
        call_steinhardt_dcv_SW_FUNC_kernel();
    }
    cudaDeviceSynchronize(); // waiting memory
    DEBUG_LOG("i am out");

    d_dcvdx.download_to(h_dcvdx, (Threads_own_atoms*3), 0, __FILE__, __LINE__);
    d_a_virial.download_to(h_a_virial, (Threads_own_atoms*6), 0, __FILE__, __LINE__);
    // cudaMemcpy(h_dcvdx, d_dcvdx.ptr, (group_count*3)*sizeof(double), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_a_virial, d_a_virial.ptr, ((lmp->atom->nlocal+lmp->atom->nghost)*6)*sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize(); // waiting memory
    DEBUG_LOG("1");

    comm_mode=true;
    lmp->comm->reverse_comm(Fixmetad);
    comm_mode=false;
    DEBUG_LOG("[Rank:%d][After Comm] h_dcvdx[0] = %f, ptr = %p\n",lmp->comm->me, h_a_virial[0], (void*)h_a_virial);
}


template <int L>
void MetaD_zqc::STEIN_QL<L>::apply_bias_force(double dVdcv, int mode){
    // pass
    // DEBUG_LOG("MetaD_zqc::STEIN_QL<L>::bias_force_SW_FUNC");
    double **f = lmp->atom->f;
    double **x = lmp->atom->x;
    int c_tag;
    // DEBUG_LOG("MetaD_zqc::STEIN_QL<L>::bias_force_SW_FUNC");
    if (mode==0){
        this->get_dcvdx_AVE(cv_value, h_dcvdx);
    } else if (mode==1){
        this->get_dcvdx_SW_FUNC(cv_value, h_dcvdx);
    }
    // DEBUG_LOG("cv_value = %g, dVdcv = %g, dcvdx = %g, %g, %g",cv_value, dVdcv, dcvdx[0], dcvdx[1], dcvdx[2]);
    // DEBUG_LOG("fx0,fy0,fz0  = %.6f, %.6f, %.6f", f[c_tag][0], f[c_tag][1], f[c_tag][2]);
    for (int c_atom=0; c_atom<(lmp->atom->nlocal); c_atom++){
        DEBUG_LOG("dcvdx, dcvdy, dcvdz  = %g, %g, %g", h_dcvdx[c_atom*3 + 0], h_dcvdx[c_atom*3 + 1], h_dcvdx[c_atom*3 + 2]);
        DEBUG_LOG("dVdcv  = %g", dVdcv);
        c_tag = c_atom;
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
        // virial
        #pragma unroll
        for (int i = 0; i < 6; i++) {
            Fixmetad->a_virial[c_tag*6 + i] += h_a_virial[ c_atom*6 + i]*dVdcv;
        }
    }

    // MPI_Allreduce(&v, &virial, 6, MPI_DOUBLE, MPI_SUM, lmp->world);
    DEBUG_LOG("post_force_r_end");
}


template <int L>
void MetaD_zqc::STEIN_QL<L>::summary(FILE* f){}


template <int L>
void MetaD_zqc::STEIN_QL<L>::call_steinhardt_cv_ql_i_kernel(){
    ERR_COND((my_env == nullptr),"my_env is NULL! Cannot launch kernel.");
    steinhardt_cv_kernel<L> <<<block_num,d_block_size>>>(
        (my_env)->my_r_SWfunc->params,
        (my_env->group_count),  (my_env->cutoff_r), (my_env->cutoff_eps_r), 
        (my_env->d_group_indices.ptr),
        (my_env->d_neigh_in_cutoff_r.ptr), 
        (my_env->d_group_numneigh.ptr),
        (my_env->d_calculated_firstneigh_ptrs.ptr),
        (my_env->d_calculated_numneigh.ptr),
        (my_env->d_x_flat.ptr),
        (my_env->d_neigh_in_switching.ptr),
        d_stein_qlm.ptr, d_stein_Ylm.ptr,
        d_stein_ql.ptr) ;
}


template <int L>
void MetaD_zqc::STEIN_QL<L>::call_steinhardt_dcv_AVE_kernel(){
    if (all_count <= 0) return;
    // LINE has f(q)=1 and df(q)=0; scale makes the effective f(q)=1/N_global.
    const MetaD_zqc::SwitchFunctionRequest sw_params_q{
        MetaD_zqc::LINE, 0.0, 0.0, 0.0, 0, 0};
    call_steinhardt_dcv_kernel(sw_params_q, 1.0 / static_cast<double>(all_count));
}

template <int L>
void MetaD_zqc::STEIN_QL<L>::call_steinhardt_dcv_SW_FUNC_kernel(){
    ERR_COND(my_cv_SWfunc == nullptr, "STEIN_QL SW_FUNC has no switching function.");
    call_steinhardt_dcv_kernel(my_cv_SWfunc->params, 1.0);
}

template <int L>
void MetaD_zqc::STEIN_QL<L>::call_steinhardt_dcv_kernel(
    const MetaD_zqc::SwitchFunctionRequest& sw_params_q, double q_weight_scale){
    if (my_env->n_active_pairs == 0) return;
    const int temp_block_num = (my_env->n_active_pairs + d_block_size - 1) / d_block_size;
    steinhardt_dcv_kernel<L> <<<temp_block_num, d_block_size>>>(
        my_env->my_r_SWfunc->params, sw_params_q, q_weight_scale,
        my_env->n_active_pairs, my_env->groupbit,
        my_env->d_mask.ptr,
        my_env->d_active_pair_ids.ptr,
        my_env->d_half_pair_i.ptr, my_env->d_half_pair_j.ptr,
        my_env->d_x_flat.ptr,
        my_env->d_neigh_in_cutoff_r.ptr, my_env->d_neigh_in_switching.ptr,
        d_stein_ql.ptr, d_stein_qlm.ptr,
        d_dYlm_dr.ptr, d_dcvdx.ptr, d_a_virial.ptr);
}


template <int L>
int MetaD_zqc::STEIN_QL<L>::get_comm_forward_bytes(){ 
    // need to communicate for each atom in the list
    // qlm[2*(L+1) ] and ql (double value) and Neigh_Nb (int value) and d_neigh_in_switching (int value)
    return num_elements +1 +1; // qlm + ql + d_neigh_in_switching
}

template <int L>
int MetaD_zqc::STEIN_QL<L>::pack_comm_forward_ubuf(int n, int *list, double *u_buf, int slot_offset, int comm_forward) {
    if (comm_mode){
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

            u_buf[m + cycle_offset*i + num_elements + 1] = my_env->h_neigh_in_switching[j];
        }
    }
    return (num_elements +1 +1);
}

template <int L>
void MetaD_zqc::STEIN_QL<L>::unpack_comm_forward_ubuf(int n, int first, double *u_buf, int slot_offset, int comm_forward) {
    
    if (comm_mode){
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

            // my_env->neigh_both_in_r_N[i] = (int) ubuf(u_buf[ m+ cycle_offset*(i-first) + num_elements +1]).i;

            my_env->h_neigh_in_switching[i] = u_buf[m + cycle_offset*(i-first) + num_elements + 1];
        }
    }
}


template <int L>
int MetaD_zqc::STEIN_QL<L>::get_comm_reverse_bytes(){ 
    int virial_num = 6;
    int dcvdx_num = 3;
    return virial_num + dcvdx_num;
}

template <int L>
int MetaD_zqc::STEIN_QL<L>::pack_comm_reverse_ubuf(int n, int first, 
                        double *u_buf, int slot_offset, int comm_reverse) {
    int virial_num = 6;
    int dcvdx_num = 3;
    if (!comm_mode){
        return virial_num + dcvdx_num;
    }
    // reverse_comm 在 get_dcvdx 里调用；若 h_a_virial 未分配则空指针解引用 → (nil) segfault
    if (h_a_virial == nullptr) {
        return virial_num + dcvdx_num;
    }
    int m = slot_offset; 
    // 参数名历史原因叫 comm_forward，实际传入的是 Fix::comm_reverse（每原子总槽位数）
    int cycle_offset = comm_reverse;

    for (int i = 0; i < n; i++) {
        #pragma unroll
        for (int ddx=0; ddx< virial_num ;ddx++){
            u_buf[m + cycle_offset*i + ddx] = h_a_virial[(i+first)*virial_num + ddx];
        }
        #pragma unroll
        for (int ddx=0; ddx< dcvdx_num ;ddx++){
            u_buf[m + cycle_offset*i + virial_num + ddx] = h_dcvdx[(i+first)*dcvdx_num + ddx];
        }
    }
    
    return virial_num + dcvdx_num;
}

template <int L>
void MetaD_zqc::STEIN_QL<L>::unpack_comm_reverse_ubuf(int n, int *list, 
                        double *u_buf, int slot_offset, int comm_reverse) {
    int virial_num = 6;
    int dcvdx_num = 3;
    if (!comm_mode || h_a_virial == nullptr){
        return;
    }
    int loctag;
    int m = slot_offset; 
    int cycle_offset = comm_reverse;
    
    // 将 ghost 上的 dcvdx 累加回对应的本地 owned 原子
    for (int i = 0; i < n; i++) {
        loctag = list[i];
        #pragma unroll
        for (int ddx=0; ddx< virial_num ;ddx++){
            h_a_virial[loctag* virial_num  + ddx] += u_buf[m + cycle_offset*i + ddx];
        }
        #pragma unroll
        for (int ddx=0; ddx<dcvdx_num;ddx++){
            h_dcvdx[loctag*dcvdx_num + ddx] += u_buf[m + cycle_offset*i + virial_num + ddx];
        }
    }
}


template <int L>
double* MetaD_zqc::STEIN_QL<L>::get_peratom_ptr(const std::string &prop_name) {
    if (prop_name == "stein_q") return stein_q;

    int axis;
    if      (prop_name == "dcvdx_x") axis = 0;
    else if (prop_name == "dcvdx_y") axis = 1;
    else if (prop_name == "dcvdx_z") axis = 2;
    else return nullptr;

    if (h_dcvdx == nullptr) return nullptr;

    double *&component = (axis == 0) ? h_dcvdx_x
                       : (axis == 1) ? h_dcvdx_y : h_dcvdx_z;
    lmp->memory->grow(component, lmp->atom->nmax, "STEIN_QL:dcvdx_component");

    // Extract the requested component for every owned atom, in local-index order.
    const int nlocal = lmp->atom->nlocal;
    for (int i = 0; i < nlocal; ++i) {
        component[i] = h_dcvdx[3*i + axis];
    }
    return component;
}

__global__ void get_environment_Steinhardt_Q(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, int start_idx,
    double cutoff_r, double cut_sigma_eps,
    // in
    int *d_group_indices, 
    LAMMPS_NS::tagint *d_group_numneigh,
    int *d_firstneigh_ptrs, double *d_x_flat,
    LAMMPS_NS::tagint *d_full_to_half,
    // out
    int *d_neigh_in_cutoff_r, int *d_active_pair_mask,
    double *d_neigh_in_switching,
    LAMMPS_NS::tagint *d_calculated_numneigh){
        
    #define sw_f(r) (MetaD_zqc::SwitchFunction::f(sw_params_rij, (r)))
    // get_environment_Steinhardt_Q in GPU
    int c_atom = blockIdx.x * blockDim.x + threadIdx.x;
    if(c_atom<calc_count){
        // double r2,temp_r2,temp_x,temp_y,temp_z,neigh_x,neigh_y,neigh_z;
        // double delt_x,delt_y,delt_z;
        int c_atom_calctag = c_atom+start_idx;
        int c_atom_loctag = d_group_indices[c_atom];
        int temp_tag;
        d_neigh_in_cutoff_r[c_atom_calctag] = 0;
        // c_glob_tag = h_tag[c_atom_loctag];
        double c_x = d_x_flat[c_atom_loctag*3];
        double c_y = d_x_flat[c_atom_loctag*3+1];
        double c_z = d_x_flat[c_atom_loctag*3+2];
        double sum_of_sigma_CNatoms = 0;
        int sum_of_numneigh = 0;
        int max_ii=0;
        //find curtoff_Natoms neigh
        int start_neigh = d_group_numneigh[c_atom];
        for (int neigh_atom=start_neigh; 
                neigh_atom<d_group_numneigh[c_atom+1]; 
                neigh_atom++){
            int neigh_loctag = d_firstneigh_ptrs[neigh_atom];
            double r2,sigma_r,r;
            double temp_x,temp_y,temp_z;
            double neigh_x,neigh_y,neigh_z;
            double delt_x,delt_y,delt_z;
            // if (neigh_loctag < 0 ) continue;
            // int n_glob_tag = h_tag[neigh_loctag];
            neigh_x = d_x_flat[neigh_loctag*3];
            neigh_y = d_x_flat[neigh_loctag*3+1];
            neigh_z = d_x_flat[neigh_loctag*3+2];
            delt_x = (neigh_x - c_x);
            delt_y = (neigh_y - c_y);
            delt_z = (neigh_z - c_z);
            r2 = delt_x*delt_x + delt_y*delt_y + delt_z*delt_z;
            r = sqrt(r2);
            sigma_r = sw_f(r);
            if ((sigma_r < cut_sigma_eps)) continue;
            // sigma_r >= cut_sigma_eps
            // 将通过筛选的原子压缩到一起，方便后续的线程数计算
            d_calculated_numneigh[start_neigh + sum_of_numneigh] = neigh_loctag;
            // 计算 sum_of_sigma_CNatoms
            sum_of_sigma_CNatoms += sigma_r;
            sum_of_numneigh ++;
            int half_idx = d_full_to_half[neigh_atom];
            if (half_idx != -1){
                d_active_pair_mask[half_idx] = 1;
            }
        }
        d_neigh_in_cutoff_r[c_atom_calctag] = sum_of_numneigh;
        d_neigh_in_switching[c_atom_loctag] = sum_of_sigma_CNatoms;
    }
    #undef sw_f
}



template <int L>
__global__ void steinhardt_cv_kernel(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, double cutoff_r, double cutoff_eps,
    int *d_group_indices,
    int *d_neigh_in_cutoff_r,
    LAMMPS_NS::tagint *d_group_numneigh,
    LAMMPS_NS::tagint *d_calculated_firstneigh_ptrs,
    LAMMPS_NS::tagint *d_calculated_numneigh,
    double *d_x_flat,
    double *d_neigh_in_switching,
    double *d_stein_qlm, double *d_stein_Ylm, double *d_stein_ql) {

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
    double NFb_i_check = d_neigh_in_switching[c_atom_loctag];
    if (neigh_num == 0 || NFb_i_check < 1e-12) return;
    // because we multiply a swfunction, so it need to devide by its weight sum
    double inv_neigh = 1.0 / (double)NFb_i_check;

    // 在寄存器（栈）上初始化局部数组用于累加，避免频繁读写全局显存
    double local_qlm[lm_size] = {0.0};
    double c_x = d_x_flat[c_atom_loctag*3];
    double c_y = d_x_flat[c_atom_loctag*3+1];
    double c_z = d_x_flat[c_atom_loctag*3+2];

    LAMMPS_NS::tagint neigh_base = d_group_numneigh[c_atom_calctag];
    LAMMPS_NS::tagint neigh_pair_base = d_calculated_firstneigh_ptrs[c_atom_calctag];

    double qlm_value_weight = 0.0;

    for (LAMMPS_NS::tagint neigh_atom = 0; neigh_atom < neigh_num; neigh_atom++) {
        LAMMPS_NS::tagint neigh_loctag = d_calculated_numneigh[neigh_base + neigh_atom];
        // Ylm 只与原子位置有关，所以可以按照group直接访问不需要扩大数组,所以用c_atom而不是c_atom_tag
        stein_Ylm_base_id = (neigh_pair_base+neigh_atom)*lm_size;
        double local_Ylm[lm_size] = {0.0};
        
        double neigh_x = d_x_flat[neigh_loctag*3];
        double neigh_y = d_x_flat[neigh_loctag*3+1];
        double neigh_z = d_x_flat[neigh_loctag*3+2];
        double delt_x = (neigh_x - c_x);
        double delt_y = (neigh_y - c_y);
        double delt_z = (neigh_z - c_z);
        double r2 = delt_x*delt_x + delt_y*delt_y + delt_z*delt_z;
        double r = sqrt(r2);
        double r_weight = sw_f(r);

        qlm_value_weight += r_weight;

        // Compute unweighted Ylm directly from the unit bond direction.
        const double inv_r = 1.0 / r;
        compute_Ylm_unit<L>(delt_x * inv_r, delt_y * inv_r, delt_z * inv_r, local_Ylm);
        #pragma unroll
        for (int m = 0; m < lm_size; ++m) {
            local_qlm[m] += r_weight * local_Ylm[m];
        }
        cuda::std::memcpy(&d_stein_Ylm[stein_Ylm_base_id], &local_Ylm, 
                            sizeof(double)*lm_size);
    }

    // --- 循环外归一化与写回全局显存 ---
    
    #pragma unroll
    for (int i = 0; i < lm_size; i++) {
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

    d_stein_ql[c_atom_loctag] = sqrt(ql_sq * 12.56637061435917295385/double(2*L + 1));
}
template __global__ void steinhardt_cv_kernel<3>(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, double cutoff_r, double cutoff_eps,
    int *d_group_indices,
    int *d_neigh_in_cutoff_r,
    LAMMPS_NS::tagint *d_group_numneigh,
    LAMMPS_NS::tagint *d_calculated_firstneigh_ptrs,
    LAMMPS_NS::tagint *d_calculated_numneigh,
    double *d_x_flat,
    double *d_neigh_in_switching,
    double *d_stein_qlm, double *d_stein_Ylm, double *d_stein_ql);
template __global__ void steinhardt_cv_kernel<4>(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, double cutoff_r, double cutoff_eps,
    int *d_group_indices,
    int *d_neigh_in_cutoff_r,
    LAMMPS_NS::tagint *d_group_numneigh,
    LAMMPS_NS::tagint *d_calculated_firstneigh_ptrs,
    LAMMPS_NS::tagint *d_calculated_numneigh,
    double *d_x_flat,
    double *d_neigh_in_switching,
    double *d_stein_qlm, double *d_stein_Ylm, double *d_stein_ql);
template __global__ void steinhardt_cv_kernel<6>(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, double cutoff_r, double cutoff_eps,
    int *d_group_indices,
    int *d_neigh_in_cutoff_r,
    LAMMPS_NS::tagint *d_group_numneigh,
    LAMMPS_NS::tagint *d_calculated_firstneigh_ptrs,
    LAMMPS_NS::tagint *d_calculated_numneigh,
    double *d_x_flat,
    double *d_neigh_in_switching,
    double *d_stein_qlm, double *d_stein_Ylm, double *d_stein_ql);


template <int L>
__global__ void steinhardt_dcv_kernel(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    MetaD_zqc::SwitchFunctionRequest sw_params_q,
    double q_weight_scale,
    LAMMPS_NS::tagint pair_all, int groupbit,
    int *d_mask,
    LAMMPS_NS::tagint *d_active_pair_ids,
    LAMMPS_NS::tagint *d_half_pair_i, LAMMPS_NS::tagint *d_half_pair_j,
    double *d_x_flat,
    int *d_neigh_in_cutoff_r, double *d_neigh_in_switching,
    double *d_stein_ql, double *d_stein_qlm,
    double *d_dYlm_dr, double *d_dcvdx, double *d_a_virial) {
    
    int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_id >= pair_all) return;
    
    int pair_tag = d_active_pair_ids[pair_id];

    #define sw_f_r(r) (MetaD_zqc::SwitchFunction::f(sw_params_rij, (r)))
    #define sw_df_r(r) (MetaD_zqc::SwitchFunction::df(sw_params_rij, (r)))
    #define sw_f_Q(x) (q_weight_scale * MetaD_zqc::SwitchFunction::f(sw_params_q, (x)))
    #define sw_df_Q(x) (q_weight_scale * MetaD_zqc::SwitchFunction::df(sw_params_q, (x)))

    double pre_ij_ji = (L%2==0)? 1.0 : -1.0; // (-1)^L
    constexpr int lm_size = (L + 1) * 2 ;

    LAMMPS_NS::tagint c_atom_loctag = d_half_pair_i[pair_tag];
    LAMMPS_NS::tagint neigh_loctag = d_half_pair_j[pair_tag];

    double scale = (4 * PI) / ((2 * L + 1));
    
    double c_x = d_x_flat[c_atom_loctag*3+0];
    double c_y = d_x_flat[c_atom_loctag*3+1];
    double c_z = d_x_flat[c_atom_loctag*3+2];
    double neigh_x = d_x_flat[neigh_loctag*3+0];
    double neigh_y = d_x_flat[neigh_loctag*3+1];
    double neigh_z = d_x_flat[neigh_loctag*3+2];
    double dx = (neigh_x - c_x);
    double dy = (neigh_y - c_y);
    double dz = (neigh_z - c_z);
    double r2 = dx*dx + dy*dy + dz*dz;
    double r = sqrt(r2);
    double r_weight = sw_f_r(r);

    double Factor_Y, Factor_Ydx, Factor_Ydy, Factor_Ydz;
    double tdx_r, tdx_i, tdy_r, tdy_i, tdz_r, tdz_i;
    

    // 【完全通用】框架：近邻、数组清零、寻址逻辑
        
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

    double catom_ql_timesN, neigh_ql_timesN;
    if ((d_mask[c_atom_loctag]&groupbit) == 0) {
        catom_ql_timesN = 0.0;
    } else {
        double qlc = d_stein_ql[c_atom_loctag];
        if (qlc < 1e-12) {
            catom_ql_timesN = 0.0;
        } else {
            catom_ql_timesN = scale * 1/qlc * 1/d_neigh_in_switching[c_atom_loctag]\
                                    * (sw_f_Q(qlc) + qlc * sw_df_Q(qlc));
        }
    }
    if ((d_mask[neigh_loctag]&groupbit) == 0) {
        neigh_ql_timesN = 0.0;
    } else {
        double qln = d_stein_ql[neigh_loctag];
        if (qln < 1e-12) {
            neigh_ql_timesN = 0.0;
        } else {
            neigh_ql_timesN = scale * 1/qln * 1/d_neigh_in_switching[neigh_loctag]\
                                    * (sw_f_Q(qln) + qln * sw_df_Q(qln));
        }
    }

    double local_Ylm[lm_size];
    const double inv_r = 1.0 / r;
    compute_Ylm_unit<L>(dx * inv_r, dy * inv_r, dz * inv_r, local_Ylm);
    // LAMMPS_NS::tagint stein_Ylm_base_id = d_half_to_full[pair_tag]*lm_size;
    LAMMPS_NS::tagint stein_qlm_base_id = c_atom_loctag * lm_size;
    LAMMPS_NS::tagint stein_qlm_neigh_id = neigh_loctag * lm_size;

    double sum_of_sigma = 0;
    double a,b;
    a = catom_ql_timesN * sw_df_r(r);
    b = pre_ij_ji * neigh_ql_timesN * sw_df_r(r);
    double Aqlmi = d_stein_qlm[stein_qlm_base_id + 0 +0];
    double Bqlmi = d_stein_qlm[stein_qlm_base_id + 0 +1];
    double Aqlmj = d_stein_qlm[stein_qlm_neigh_id + 0 +0];
    double Bqlmj = d_stein_qlm[stein_qlm_neigh_id + 0 +1];
    double Aylmrij = local_Ylm[0 +0];
    double Bylmrij = local_Ylm[0 +1];
    sum_of_sigma += a * (Aqlmi * (Aylmrij - Aqlmi)+ Bqlmi * (Bylmrij - Bqlmi));
    sum_of_sigma += b * (Aqlmj * (Aylmrij - pre_ij_ji * Aqlmj)+ Bqlmj * (Bylmrij - pre_ij_ji * Bqlmj));
    #pragma unroll
    for (int i = 1; i <= L; i++) {
        Aqlmi = d_stein_qlm[stein_qlm_base_id + 2*i +0];
        Bqlmi = d_stein_qlm[stein_qlm_base_id + 2*i +1];
        Aqlmj = d_stein_qlm[stein_qlm_neigh_id + 2*i +0];
        Bqlmj = d_stein_qlm[stein_qlm_neigh_id + 2*i +1];
        Aylmrij = local_Ylm[2*i +0];
        Bylmrij = local_Ylm[2*i +1];
        sum_of_sigma += 2* a * (Aqlmi * (Aylmrij - Aqlmi)+ Bqlmi * (Bylmrij - Bqlmi));
        sum_of_sigma += 2* b * (Aqlmj * (Aylmrij - pre_ij_ji * Aqlmj)+ Bqlmj * (Bylmrij - pre_ij_ji * Bqlmj));
    }
    double local_Ylmdr[3*2] = {0.0};
    catom_ql_timesN = catom_ql_timesN * r_weight;
    neigh_ql_timesN = neigh_ql_timesN * r_weight;

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
            d_stein_qlm, local_Ylmdr
            // &d_dYlm_dr[c_atom_calctag * 3 * 2]
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
            d_stein_qlm, local_Ylmdr
            // &d_dYlm_dr[c_atom_calctag * 3 * 2]
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
            d_stein_qlm, local_Ylmdr
            // &d_dYlm_dr[c_atom_calctag * 3 * 2]
        );
    }
    

    // f. is dcv/dx
    double f[3] = {0.0};
    f[0] = (local_Ylmdr[ 0*2 + 0] + local_Ylmdr[ 0*2 + 1] + sum_of_sigma*dx/r);
    f[1] = (local_Ylmdr[ 1*2 + 0] + local_Ylmdr[ 1*2 + 1] + sum_of_sigma*dy/r);
    f[2] = (local_Ylmdr[ 2*2 + 0] + local_Ylmdr[ 2*2 + 1] + sum_of_sigma*dz/r);
    // if (isnan(fx) || isnan(fy) || isnan(fz)) {
    //     // 只有崩成 NaN 的线程才会触发打印，不影响整体速度
    //     printf("[NaN Detected] c_atom_calctag = %d, neigh_tag = %d, Neigh_Nb=%d, d_stein_ql[neigh_tag]=%g, d_stein_qlm[stein_qlm_neigh_id + 0]=%g r = %f, dx = %f, dy = %f, dz = %f, sin_theta = %f\n", 
    //             c_atom_calctag, neigh_tag, Neigh_Nb, d_stein_ql[neigh_tag],d_stein_qlm[stein_qlm_neigh_id + 0], r, dx, dy, dz, sin_theta);
    // }
    
    double tmpvirial[6] = {0.0};
    tmpvirial[ 0 ] = dx*f[0]; // vxx
    tmpvirial[ 1 ] = dy*f[1]; // vyy
    tmpvirial[ 2 ] = dz*f[2]; // vzz
    tmpvirial[ 3 ] = dx*f[1]; // vxy
    tmpvirial[ 4 ] = dx*f[2]; // vxz
    tmpvirial[ 5 ] = dy*f[2]; // vyz
    #pragma unroll
    for (int i = 0; i < 6; i++) {
        atomicAdd(&d_a_virial[6*c_atom_loctag + i],-0.5*tmpvirial[i]);
        atomicAdd(&d_a_virial[6*neigh_loctag + i], -0.5*tmpvirial[i]);
        // d_a_virial[6*c_atom_loctag + i] += tmpvirial[i];
    }

    // 【完全通用】最后的总偏导汇总
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        atomicAdd(&d_dcvdx[c_atom_loctag * 3 + i], -f[i]);
        atomicAdd(&d_dcvdx[neigh_loctag * 3 + i], f[i]);
    }
}
template __global__ void steinhardt_dcv_kernel<3>(    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    MetaD_zqc::SwitchFunctionRequest sw_params_q,
    double q_weight_scale,
    LAMMPS_NS::tagint pair_all, int groupbit,
    int *d_mask,
    LAMMPS_NS::tagint *d_active_pair_ids,
    LAMMPS_NS::tagint *d_half_pair_i, LAMMPS_NS::tagint *d_half_pair_j,
    double *d_x_flat,
    int *d_neigh_in_cutoff_r, double *d_neigh_in_switching,
    double *d_stein_ql, double *d_stein_qlm,
    double *d_dYlm_dr, double *d_dcvdx, double *d_a_virial);
template __global__ void steinhardt_dcv_kernel<4>(    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    MetaD_zqc::SwitchFunctionRequest sw_params_q,
    double q_weight_scale,
    LAMMPS_NS::tagint pair_all, int groupbit,
    int *d_mask,
    LAMMPS_NS::tagint *d_active_pair_ids,
    LAMMPS_NS::tagint *d_half_pair_i, LAMMPS_NS::tagint *d_half_pair_j,
    double *d_x_flat,
    int *d_neigh_in_cutoff_r, double *d_neigh_in_switching,
    double *d_stein_ql, double *d_stein_qlm,
    double *d_dYlm_dr, double *d_dcvdx, double *d_a_virial);
template __global__ void steinhardt_dcv_kernel<6>(    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    MetaD_zqc::SwitchFunctionRequest sw_params_q,
    double q_weight_scale,
    LAMMPS_NS::tagint pair_all, int groupbit,
    int *d_mask,
    LAMMPS_NS::tagint *d_active_pair_ids,
    LAMMPS_NS::tagint *d_half_pair_i, LAMMPS_NS::tagint *d_half_pair_j,
    double *d_x_flat,
    int *d_neigh_in_cutoff_r, double *d_neigh_in_switching,
    double *d_stein_ql, double *d_stein_qlm,
    double *d_dYlm_dr, double *d_dcvdx, double *d_a_virial);


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