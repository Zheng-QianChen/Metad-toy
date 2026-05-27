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

std::map<std::string, MetaD_zqc::Stru_fact_env*> MetaD_zqc::Stru_fact_env::env_pool;

MetaD_zqc::CV* MetaD_zqc::Stru_factor::create(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::FixMetadynamics *Fixmetad, 
                                            int narg, char **arg, int &i, FILE *f_check){
    DEBUG_LOG("In STRU_FACTOR settings");
    LAMMPS_NS::Error *error = lmp->error;

    std::string cal_name = arg[i];

    MetaD_zqc::StruFactorRequest req;

    req.cal_name = cal_name;
    // 原子环境分析-初始设置
    // Usage: STRU_FACTOR <group>
    ERR_COND(i + 1 >= narg, "Error: STRU_FACTOR command requires at least \"STRU_FACTOR <group> \".");
    req.group_name = arg[i+1];
    req.group_id = lmp->group->find(req.group_name);
    ERR_COND(req.group_id == -1, "Error: group name %s not found.", req.group_name);
    //参数有效性
    // default values
    req.cutoff_r = 8.0;
    req.d_block_size = 128;
    int iarg=2 + i;
    printf("im in Stru_factor::create, with group_name=%s, group_id=%d\n", req.group_name, req.group_id);
    while (iarg < narg) {
        if (strcmp(arg[iarg], "cutoff_r") == 0) {
            ERR_COND((iarg + 1 >= narg) ,"Error: \'cutoff_r\' keyword requires a value");
            req.cutoff_r = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
            iarg += 2;
        } else if (strcmp(arg[iarg], "q_factor") == 0) {
            ERR_COND((iarg + 1 >= narg), "Error: \'q_factor\' keyword requires an integer");
            req.q_factor = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
            ERR_COND(req.q_factor <= 0, "Error: \'q_factor\' must be > 0");
            iarg += 2;
        } else if (strcmp(arg[iarg], "d_block_size") == 0) {
            ERR_COND((iarg + 1 >= narg), "Error: \'d_block_size\' keyword requires an integer");
            req.d_block_size = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
            ERR_COND(req.d_block_size <= 0, "Error: \'d_block_size\' must be > 0");
            iarg += 2;
        } else if (strcmp(arg[iarg], "Chemical") == 0) {
            req.use_chemical_lock = true;
            iarg += 1;
        } else if (strcmp(arg[iarg], "Chem_ctarget") == 0) {
            ERR_COND((iarg + 1 >= narg), "Error: \'Chem_ctarget\' keyword requires a number");
            req.c_target = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
            iarg += 2;
        } else if (strcmp(arg[iarg], "sigma") == 0) {
            ERR_COND((iarg + 1 >= narg), "Error: \'sigma\' keyword requires a number");
            req.sigma = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
            iarg += 2;
        } else {
            break;
        }
    }

    // We need full neighbor list to get cuda run faster
    NeighRequest *full_request;
    full_request = lmp->neighbor->add_request(Fixmetad, NeighConst::REQ_FULL);
    full_request->set_id(2);

    if (req.use_chemical_lock) {
        LOG("Logging: set STRU_FACTOR as group_name=%s q_factor=%g cutoff_r=%f d_block_size=%d.\n         Chemical lock is ON, with c_target=%g and sigma=%g.",
            req.group_name, req.q_factor, req.cutoff_r, req.d_block_size,
            req.c_target, req.sigma);
        // create Structure factor CV
        // env for CV
        // MetaD_zqc::Stru_fact_chem_env *temp_env = MetaD_zqc::Stru_fact_env::get_or_create(lmp, 
        //                         f_check, Fixmetad, req);
        MetaD_zqc::Stru_fact_chem_env *temp_env = static_cast<MetaD_zqc::Stru_fact_chem_env*>(
                                    MetaD_zqc::Stru_fact_env::get_or_create(lmp, f_check, Fixmetad, req)
                                );
        DEBUG_LOG("Stru_fact_chem_env is %p", temp_env);
        std::string env_setNum = temp_env->get_env_key();
        i = iarg;
        // return Stru_fact cv
        return new MetaD_zqc::Stru_factor_chem(lmp, Fixmetad, f_check, 
                                env_setNum, req.group_id, temp_env,
                                req.q_factor, req.d_block_size);
    } else {
        LOG("Logging: set STRU_FACTOR as group_name=%s q_factor=%g cutoff_r=%f d_block_size=%d.",
                            req.group_name, req.q_factor, req.cutoff_r, req.d_block_size);
        // create Structure factor CV
        // env for CV
        MetaD_zqc::Stru_fact_env *temp_env = MetaD_zqc::Stru_fact_env::get_or_create(lmp, 
                                f_check, Fixmetad, req);
        DEBUG_LOG("Stru_fact_env is %p", temp_env);
        std::string env_setNum = temp_env->get_env_key();
        i = iarg;
        // return Stru_fact cv
        return new MetaD_zqc::Stru_factor(lmp, Fixmetad, f_check, 
                                env_setNum, req.group_id, temp_env,
                                req.q_factor, req.d_block_size);
    }
}

MetaD_zqc::Stru_fact_env* MetaD_zqc::Stru_fact_env::get_or_create(LAMMPS_NS::LAMMPS *lmp, FILE *f_check,
                                            LAMMPS_NS::FixMetadynamics *Fixmetad,
                                            StruFactorRequest req) {
    printf("In Stru_fact_env::get_or_create with group_id=%d, cutoff_r=%g, use_chemical_lock=%d, c_target=%g, sigma=%g\n",
            req.group_id, req.cutoff_r, req.use_chemical_lock, req.c_target, req.sigma);
    if (!req.use_chemical_lock){
        // 1. generate a unique key for the environment based on its parameters
        std::string key = std::to_string(req.group_id) + "_" + std::to_string(req.cutoff_r)
                            + "_";
        // 2. check if the envioronment already exist in the pool
        if (env_pool.count(key)) {
            return env_pool[key]; // if exits, return the existing envioronment
        }
        // 3. new envioronment and store it in the pool if not exist
        MetaD_zqc::Stru_fact_env *new_env = new Stru_fact_env(lmp, f_check, Fixmetad,
                                                    req.group_id, req.cutoff_r);
        env_pool[key] = new_env; // store the new envioronment in the pool
        return new_env;
    } else {
        // 1. generate a unique key for the environment based on its parameters
        std::string key = std::to_string(req.group_id) + "_" + std::to_string(req.cutoff_r)
                            + "_" + std::to_string(req.use_chemical_lock)
                            + "_" + std::to_string(req.c_target) + "_" + std::to_string(req.sigma);
        // 2. check if the envioronment already exist in the pool
        if (env_pool.count(key)) {
            return env_pool[key]; // if exits, return the existing envioronment
        }
        // 3. new envioronment and store it in the pool if not exist
        MetaD_zqc::Stru_fact_chem_env *new_env = new Stru_fact_chem_env(lmp, f_check, Fixmetad,
                                                    req.group_id, req.cutoff_r, req.c_target, req.sigma);
        env_pool[key] = new_env; // store the new envioronment in the pool
        return new_env;
    }
}

std::string MetaD_zqc::Stru_fact_env::get_env_key(){
    std::string key = std::to_string(this->group_id) + "_" + std::to_string(this->cutoff_r);
    return key;
}

void MetaD_zqc::Stru_fact_env::clear_pool() {
    for (auto& pair : env_pool) delete pair.second;
    env_pool.clear();
}

MetaD_zqc::Stru_fact_env::Stru_fact_env(LAMMPS_NS::LAMMPS *lmp, FILE *f_check,
             LAMMPS_NS::FixMetadynamics *Fixmetad, int group_id, double cutoff_r)
    : lmp(lmp),
      f_check(f_check),
      Fixmetad(Fixmetad),
      group_id(group_id),
      cutoff_r(cutoff_r)
{
    error = lmp->error;

    pbc_x = (lmp->domain->xperiodic == 1);
    pbc_y = (lmp->domain->yperiodic == 1);
    pbc_z = (lmp->domain->zperiodic == 1);
    // 这里可以添加一些初始化代码，例如分配内存、设置默认值等
    DEBUG_LOG("Stru_fact_env initialized with cutoff_r=%g", cutoff_r);

    // const char *group_name = arg[1];
    groupbit = lmp->group->bitmask[group_id]; // 关键：存储原子组位掩码
    // group_dminneigh = new double [2]; //inintial
    // neigh_in_cutoff_r = new int [2]; //inintial
    lmp->memory->create(group_dminneigh, 0, "metad:Stru_fact_env:group_dminneigh");
    lmp->memory->create(neigh_in_cutoff_r, 0, "metad:Stru_fact_env:neigh_in_cutoff_r");
    init_flag = false;

    // comment name
    d_group_numneigh.set_name("d_group_numneigh");
    d_x_flat.set_name("d_x_flat");
    d_mask.set_name("d_mask");
    d_group_indices.set_name("d_group_indices");
    d_firstneigh_ptrs.set_name("d_firstneigh_ptrs");
    d_group_dminneigh.set_name("d_group_dminneigh");
    d_neigh_in_cutoff_r.set_name("d_neigh_in_cutoff_r");
    d_neigh_both_in_r_N.set_name("d_neigh_both_in_r_N");
    d_calculated_numneigh.set_name("d_calculated_numneigh");
}

MetaD_zqc::Stru_factor::Stru_factor(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check, 
                             std::string env_setNum, int group_id,
                             MetaD_zqc::Stru_fact_env* my_env,
                             double q_factor, int d_block_size)
                        : CV(lmp, f_check),
                            Fixmetad(Fixmetad),
                            env_setNum(env_setNum),
                            q_factor(q_factor),
                        //   group_id(group_id),
                            my_env(my_env),
                            d_block_size(d_block_size){
    // my_averager = new MetaD_zqc::CUBAverager();
    my_averager = new MetaD_zqc::KahanAverager();
    DEBUG_LOG("Logging: New a Stru_factor file, will generate %d lines in GPU,\n     with cutoff_r=%g",
                d_block_size, my_env->cutoff_r);
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
    lmp->memory->create(h_stru_factor, 0, "metad:Stru_factor:h_stru_factor");
    lmp->memory->create(h_dcvdx, 0, "metad:Stru_factor:h_dcvdx");
    error = lmp->error;

    
    // comment name
    d_stru_factor.set_name("d_stru_factor");
    d_dcvdx.set_name("d_dcvdx");
}

MetaD_zqc::Stru_fact_env::~Stru_fact_env(){
    atoms = nullptr;
    // release all alloc
    nlist = nullptr;
    lmp->memory->destroy(h_group_numneigh);
    numneigh = nullptr;
    firstneigh = nullptr;
    mask = nullptr;
    lmp->memory->destroy(h_x_flat);
    lmp->memory->destroy(h_group_indices);
    lmp->memory->destroy(h_firstneigh_ptrs);
    lmp->memory->destroy(group_dminneigh);
    lmp->memory->destroy(neigh_in_cutoff_r);
    lmp->memory->destroy(neigh_both_in_r_N);
    lmp->memory->destroy(calculated_numneigh);
    // the GpuBuffer will automatically release its memory, 
    // so we don't need to manually free it here
}


MetaD_zqc::Stru_factor::~Stru_factor(){
    lmp->memory->destroy(h_stru_factor);
    lmp->memory->destroy(h_dcvdx);
    // the GpuBuffer will automatically release its memory, 
    // so we don't need to manually free it here
}


void MetaD_zqc::Stru_fact_env::refresh_lmpbox(){
    // clear the h_group_indices
    atom = lmp->atom;
    mask = (atom)->mask;     // 原子组掩码

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

    d_mask.grow_to(((atom)->nlocal+(atom)->nghost), f_check, __FILE__, __LINE__);
    SAFE_CUDA_MEMCPY((d_mask.ptr),(mask),(((atom)->nlocal+(atom)->nghost))*sizeof(int),cudaMemcpyHostToDevice,f_check);

    // set up nvidia thread number
    block_num = ((group_count) + d_block_size - 1)/d_block_size;
    N = d_block_size*block_num;
    LOG_COND((((box_x)<2*(cutoff_r))||((box_y)<2*(cutoff_r))||((box_z)<2*(cutoff_r))),"Warning: box < cutoff_r, please check your system !");
}

void MetaD_zqc::Stru_fact_env::get_env(){
    DEBUG_LOG("im in get_env, current step is %lld, last_update_step is %lld", (long long)lmp->update->ntimestep, (long long)this->last_update_step);
    size_t datalen = 0;
    atom = lmp->atom;
    // =======更新一下邻居列表位置防止报错=========
    nlist = Fixmetad->listfull;
    ERR_COND((nlist == nullptr),"STRU_FACTOR CV failed to find neighbor list now.");
    // =======防止lammps运行过程体积更改==========
    ERR_COND((lmp->domain == NULL),"domain list not initialized");
    box_x = (pbc_x) ? lmp->domain->xprd : INFINITY;
    box_y = (pbc_y) ? lmp->domain->yprd : INFINITY;
    box_z = (pbc_z) ? lmp->domain->zprd : INFINITY;

    // utilize different envioronment set
    // such as neighbor list, atom position, box size, to get the local structure information for each atom in the group
    if ((lmp->update->ntimestep > lmp->neighbor->lastcall)&&(lmp->update->ntimestep != 1)&&((numneigh != nullptr))&&(init_flag)){
        DEBUG_LOG("we skip rebuild in envioronment when %lld.", (long long)lmp->neighbor->lastcall);
    } else {
        // =========================================================================
        // neighbour list and its copy to devise
        // h_group_indices / d_group_indices: where the group atoms in locals' tag
        // =========================================================================
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
        lmp->memory->grow(h_group_numneigh, datalen, "STRU_FACTOR:h_group_numneigh");
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
        lmp->memory->grow(h_firstneigh_ptrs, h_group_numneigh[group_count], "STRU_FACTOR:h_firstneigh_ptrs");
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
    DEBUG_LOG("d_x_flat=%p",d_x_flat.ptr);
    lmp->memory->grow(h_x_flat, (atom->nlocal + atom->nghost) * 3, "STRU_FACTOR:h_x_flat");
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
    ERR_COND((h_x == NULL),"h_x list not initialized");
    ERR_COND((h_x_flat == NULL),"h_x_flat list not initialized");
    ERR_COND((d_x_flat.ptr == NULL),"d_x_flat list not initialized");
    DEBUG_LOG("d_x_flat Allocated at: %p", d_x_flat.ptr);
    cudaDeviceSynchronize(); // waiting memory

    // =========================================================================
    // create output address
    // d_group_dminneigh : [dx, dy, dz, rsq] * (sum of whole pairs)
    // d_neigh_in_cutoff_r : count of c_atoms' neighbours that satisfied cutoff_r
    // =========================================================================
    DEBUG_LOG("release end");
    // beacause we cant foresee how many atoms will be in with a c_atoms neighbor
    // so we just allocate a large memory for the result, which is group_count*cutoff_Natoms*4 for d_group_dminneigh and group_count*cutoff_Natoms for d_calculated_numneigh 
    d_group_dminneigh.grow_to(h_group_numneigh[group_count]*4, f_check, __FILE__, __LINE__);
    d_neigh_in_cutoff_r.grow_to(N, f_check, __FILE__, __LINE__);
    d_calculated_numneigh.grow_to(h_group_numneigh[group_count]*2, f_check, __FILE__, __LINE__);

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
    get_envioronment_Strufactor<<<block_num,d_block_size>>>
      (cutoff_rsq, box_x, box_y, box_z, 
      group_count, d_group_indices.ptr, d_group_numneigh.ptr, d_firstneigh_ptrs.ptr, d_x_flat.ptr,
      d_group_dminneigh.ptr, d_neigh_in_cutoff_r.ptr, d_calculated_numneigh.ptr) ;
    DEBUG_LOG("env refresh out, kernel launched");
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
    // DEBUG_LOG((group_dminneigh == NULL),"group_dminneigh list not initialized");
    // DEBUG_LOG((neigh_in_cutoff_r == NULL),"neigh_in_cutoff_r list not initialized");
    // delete[] group_dminneigh;
    // group_dminneigh = new double [h_group_numneigh[group_count]*4];
    lmp->memory->grow(group_dminneigh, (h_group_numneigh[group_count]*4), "STRU_FACTOR:group_dminneigh");
    SAFE_CUDA_MEMCPY(group_dminneigh, d_group_dminneigh.ptr,
      (h_group_numneigh[group_count]*4) * sizeof(double), cudaMemcpyDeviceToHost,f_check);
    // delete[] neigh_in_cutoff_r;
    // neigh_in_cutoff_r = new int [group_count];
    lmp->memory->grow(neigh_in_cutoff_r, (group_count), "STRU_FACTOR:neigh_in_cutoff_r");
    SAFE_CUDA_MEMCPY(neigh_in_cutoff_r, d_neigh_in_cutoff_r.ptr,
      (group_count) * sizeof(int), cudaMemcpyDeviceToHost,f_check);
    // delete[] calculated_numneigh;
    // calculated_numneigh = new LAMMPS_NS::tagint [h_group_numneigh[group_count]];
    lmp->memory->grow(calculated_numneigh, (h_group_numneigh[group_count]), "STRU_FACTOR:calculated_numneigh");
    SAFE_CUDA_MEMCPY(calculated_numneigh, d_calculated_numneigh.ptr,
      (h_group_numneigh[group_count]) * sizeof(LAMMPS_NS::tagint), cudaMemcpyDeviceToHost,f_check);
    cudaDeviceSynchronize(); //catch kernel done
    DEBUG_LOG("copy end");
    DEBUG_LOG("group_dminneigh Allocated at: %p", group_dminneigh);
    this->last_update_step = lmp->update->ntimestep;
}

void MetaD_zqc::Stru_factor::envioronment(){
    DEBUG_LOG("last_update_step is %lld, group_count=%d", (long long)my_env->last_update_step, my_env->group_count);
    if (lmp->update->ntimestep > my_env->last_update_step){
        my_env->get_env();
    }
    // DEBUG_LOG("envioronment function in, env_setNum is %s, get_env done",env_setNum);
    DEBUG_LOG("last_update_step is %lld, group_count=%d", (long long)my_env->last_update_step, my_env->group_count);
}

auto MetaD_zqc::Stru_factor::set_CV_calculate(std::string func_name) -> CV_Calculation {
    if (func_name == "AVE") {
        return static_cast<CV_Calculation>(&Stru_factor::compute_cv_AVE);
    } else if (func_name == "FILTER_SUM") {
        // return static_cast<CV_Calculation>(&Stru_factor::compute_cv_FILTER_SUM);
    }
    return nullptr;
}

auto MetaD_zqc::Stru_factor::set_CV_bias_force(std::string func_name) -> CV_BiasForce {
    if (func_name == "AVE") {
        return static_cast<CV_BiasForce>(&Stru_factor::bias_force_AVE);
    } else if (func_name == "FILTER_SUM") {
        // return static_cast<CV_Calculation>(&Stru_factor::bias_force_FILTER_SUM);
    }
    return nullptr;
}

void MetaD_zqc::Stru_factor::base_calc(){
    compute_stru_factor_peratoms();
}

double MetaD_zqc::Stru_factor::compute_cv_AVE(){
    DEBUG_LOG("im in compute_cv_AVE.");
    int group_count = my_env->group_count;
    int Threads_own_atoms = lmp->atom->nlocal;
    DEBUG_LOG("group_count = %d",group_count);
    double sf_ave_local=0;
    DEBUG_LOG_COND((h_stru_factor == NULL),"h_stru_factor list not initialized");
    if (group_count != 0) {
        my_averager->compute(Threads_own_atoms, all_count, h_stru_factor, 
            lmp->atom->mask, my_env->groupbit, sf_ave_local);
    }
    MPI_Allreduce(&sf_ave_local, &cv_value, 1, MPI_DOUBLE, MPI_SUM, lmp->world);
    DEBUG_LOG("group_count = %d, compute_cv_AVE = %g",group_count, cv_value);
    return cv_value;
}

// double MetaD_zqc::Stru_factor::compute_cv_FILTER_SUM(){
//     DEBUG_LOG("Entering compute_cv_FILTER_SUM.");
//     int group_count = my_env->group_count;
//     double filter_sum_local = 0.0;
    
//     // 参数建议：s_0 是区分液固的阈值（论文中的 1.25），d_0 是平滑过渡的宽度
//     double s_0 = 1.25; 
//     double d_0 = 0.1;  

//     if (group_count != 0 && h_stru_factor != NULL) {
//         for (int i = 0; i < group_count; ++i) {
//             double s_i = h_stru_factor[i];
            
//             // 使用平滑阶跃函数：f(s) = 1 / (1 + exp(-(s - s_0) / d_0))
//             // 这样当 s_i 增加时，贡献度平滑地从 0 变到 1
//             double exp_term = std::exp(-(s_i - s_0) / d_0);
//             double filter_val = 1.0 / (1.0 + exp_term);
            
//             filter_sum_local += filter_val;
//         }
//     }

//     // 汇总所有进程的固态原子数估计值
//     MPI_Allreduce(&filter_sum_local, &cv_value, 1, MPI_DOUBLE, MPI_SUM, lmp->world);
    
//     DEBUG_LOG("FILTER_SUM (Estimated solid atoms): %g", cv_value);
//     return cv_value;
// }

void MetaD_zqc::Stru_factor::compute_stru_factor_peratoms(){
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
        // h_stru_factor for all aim atoms
        lmp->memory->grow(h_stru_factor, my_env->group_count, "metad:Stru_factor:cv_bound");
        DEBUG_LOG("d_block_size is %d, block_num is %d",d_block_size, block_num);
    }
    DEBUG_LOG("group_count=%lld",(long long)my_env->group_count);

    // 2. calculate atoms' envioronment
    DEBUG_LOG("envioronment function in, env_setNum is %s",env_setNum.c_str());
    envioronment();
    DEBUG_LOG("envioronment function out");

    // 3. calculate atoms' other things
    sf_param_calc(h_stru_factor);

    DEBUG_LOG_COND((my_env->group_dminneigh == NULL),"group_dminneigh list not initialized");
    DEBUG_LOG("group_dminneigh Allocated at: %p", my_env->group_dminneigh);
    
    // 输出group中每个原子的sf值
    DEBUG_RUN(for(int c_atom=0;c_atom<my_env->group_count;c_atom++)
                {
                    DEBUG_LOG("Stru_factor[%lld] = %f",(long long)c_atom,h_stru_factor[c_atom]);
                });
    DEBUG_LOG("post_force function end");
}


void MetaD_zqc::Stru_factor::bias_force_AVE(double dVdcv){
    // pass
    DEBUG_LOG("MetaD_zqc::Stru_factor::bias_force_AVE");
    double **f = lmp->atom->f;
    double **x = lmp->atom->x;
    int c_tag;
    double sumForce[3] = {0.0, 0.0, 0.0};
    DEBUG_LOG("MetaD_zqc::Stru_factor::bias_force_AVE");
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
             error->all(FLERR, "Stru_factor CV error: force is infinity, check your system or cv_value.");
        }
        f[c_tag][0] -= dVdcv*h_dcvdx[c_atom*3 + 0];
        f[c_tag][1] -= dVdcv*h_dcvdx[c_atom*3 + 1];
        f[c_tag][2] -= dVdcv*h_dcvdx[c_atom*3 + 2];
        DEBUG_LOG("fx,fy,fz  = %g, %g, %g", f[c_tag][0], f[c_tag][1], f[c_tag][2]);
        sumForce[0] += dVdcv*h_dcvdx[c_atom*3 + 0];
        sumForce[1] += dVdcv*h_dcvdx[c_atom*3 + 1];
        sumForce[2] += dVdcv*h_dcvdx[c_atom*3 + 2];
    }
    double sumforce_mod=sqrt(POW2(sumForce[0])+POW2(sumForce[1])+POW2(sumForce[2]));
    LOG_COND(sumforce_mod>1e-10,"Warning: bias force will make the system flow. sumforce_mod=%g",sumforce_mod);
    DEBUG_LOG("sumForce=%g",sumforce_mod);
    DEBUG_LOG("post_force_r_end");
}


void MetaD_zqc::Stru_factor::get_dcvdx_AVE(double cv_value, double *dcvdx){
    int group_count = my_env->group_count;
    int last_group_count = my_env->last_group_count;
    size_t datalen = 0;
    

    // DEBUG_RUN(
    datalen = group_count;
    lmp->memory->grow(h_stru_factor, datalen, "Stru_factor:h_stru_factor");
    SAFE_CUDA_MEMCPY(h_stru_factor, d_stru_factor.ptr, datalen*sizeof(double), cudaMemcpyDeviceToHost,f_check);


    datalen = (group_count*3);
    lmp->memory->grow(h_dcvdx, datalen, "Stru_factor:h_dcvdx");
    d_dcvdx.grow_to(datalen, f_check, __FILE__, __LINE__);
    SAFE_CUDA_MEMCPY(d_dcvdx.ptr,h_dcvdx, datalen*sizeof(double),cudaMemcpyHostToDevice,f_check);

    DEBUG_LOG("i will start a kernel of ql");
    cudaDeviceSynchronize(); // waiting memory
    call_structure_factor_dcv_AVE_kernel();
    cudaDeviceSynchronize(); // waiting memory
    DEBUG_LOG("i am out");

    cudaMemcpy(h_dcvdx, d_dcvdx.ptr, (group_count*3)*sizeof(double), cudaMemcpyDeviceToHost);
    // SAFE_CUDA_MEMCPY(h_dcvdx, d_dcvdx,
    //   (group_count*3)*sizeof(double), cudaMemcpyDeviceToHost, file);
    cudaDeviceSynchronize(); // waiting memory
}


void MetaD_zqc::Stru_factor::sf_param_calc(double *h_stru_factor){
    int last_group_count = my_env->last_group_count;
    int group_count = my_env->group_count;

    d_stru_factor.grow_to(group_count, f_check, __FILE__, __LINE__);

    DEBUG_LOG("i will start a kernel of ql");
    cudaDeviceSynchronize(); // waiting memory
    // TODO:
    call_structure_factor_cv_kernel();
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

    SAFE_CUDA_MEMCPY(h_stru_factor, d_stru_factor.ptr,
      (group_count) * sizeof(double), cudaMemcpyDeviceToHost,f_check);

}

void MetaD_zqc::Stru_factor::summary(FILE* f){}

void MetaD_zqc::Stru_factor::call_structure_factor_cv_kernel(){
    structure_factor_cv_kernel<<<block_num,d_block_size>>>(
        (my_env->group_count), q_factor, POW2(my_env->cutoff_r),
        (my_env->d_group_numneigh.ptr), 
        (my_env->d_group_dminneigh.ptr), 
        (my_env->d_neigh_in_cutoff_r.ptr), 
        d_stru_factor.ptr);
}

void MetaD_zqc::Stru_factor::call_structure_factor_dcv_AVE_kernel(){
    // printf("[Rank:%d]d_stein_Ylm is located in %p\n",lmp->comm->me,d_stein_Ylm.ptr);
    structure_factor_dcv_AVE_kernel<<<block_num,d_block_size>>>(
        (my_env->group_count), (my_env->groupbit), all_count, POW2(my_env->cutoff_r),
        (my_env->d_mask.ptr), (my_env->d_group_indices.ptr), 
        (my_env->d_calculated_numneigh.ptr), (my_env->d_group_numneigh.ptr),
        (my_env->d_neigh_in_cutoff_r.ptr), (my_env->d_group_dminneigh.ptr),
        q_factor, d_stru_factor.ptr, 
        d_dcvdx.ptr);
}

__global__ void get_envioronment_Strufactor(double cutoff_rsq,
    double box_x, double box_y, double box_z,
    int group_count, int *d_group_indices, LAMMPS_NS::tagint *d_group_numneigh,
    int *d_firstneigh_ptrs, double *d_x_flat,
    double *d_group_dminneigh, int *d_neigh_in_cutoff_r,
    LAMMPS_NS::tagint *d_calculated_numneigh){
    // get_envioronment in GPU
    int c_atom = blockIdx.x * blockDim.x + threadIdx.x;
    if(c_atom<group_count){
        double r2,temp_r2,temp_x,temp_y,temp_z,neigh_x,neigh_y,neigh_z;
        double delt_x,delt_y,delt_z;
        int c_atom_tag = d_group_indices[c_atom];
        int temp_tag = 0;
        d_neigh_in_cutoff_r[c_atom] = 0;
        // c_glob_tag = h_tag[c_atom_tag];
        double c_x = d_x_flat[c_atom_tag*3];
        double c_y = d_x_flat[c_atom_tag*3+1];
        double c_z = d_x_flat[c_atom_tag*3+2];
        int max_ii;
        // DEBUG_LOG("now im in %d, c_atom_tag=%d, cx,cy,cz:%f,%f,%f",c_atom,c_atom_tag,c_x,c_y,c_z);
        double max_r2 = (box_x+box_y+box_z)*(box_x+box_y+box_z);
        int neigh_min, neigh_max;
        neigh_min = d_group_numneigh[c_atom];
        neigh_max = d_group_numneigh[c_atom+1];
        for (int i=neigh_min;i<neigh_max;i++){
            d_group_dminneigh[i*4]=-1;
            d_calculated_numneigh[i] = -1;
        }
        //find curtoff_Natoms neigh
        for (int neigh_atom = neigh_min; neigh_atom < neigh_max; neigh_atom++){
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
            d_group_dminneigh[neigh_min*4 + temp_tag*4 + 0 ] = delt_x;
            d_group_dminneigh[neigh_min*4 + temp_tag*4 + 1 ] = delt_y;
            d_group_dminneigh[neigh_min*4 + temp_tag*4 + 2 ] = delt_z;
            d_group_dminneigh[neigh_min*4 + temp_tag*4 + 3 ] = r2;
            d_calculated_numneigh[neigh_min + temp_tag] = n_local_tag;
            temp_tag++;
        }
        d_neigh_in_cutoff_r[c_atom] = temp_tag;
    }
}

__global__ void structure_factor_cv_kernel(
        int group_count, double q_factor, double cutoff_rsq,
        LAMMPS_NS::tagint *d_group_numneigh,
        double *d_group_dminneigh, int *d_neigh_in_cutoff_r, 
        double *d_stru_factor){

    int c_atom = blockIdx.x * blockDim.x + threadIdx.x;
    double r_on = 0.8*cutoff_rsq;
    double s = 1.0;
    // double ds = 0.0;
    if(c_atom<group_count){
        int neigh_min, neigh_max;
        double theta;
        double sin_theta, cos_theta;
        neigh_min = d_group_numneigh[c_atom];
        neigh_max = d_group_numneigh[c_atom] + d_neigh_in_cutoff_r[c_atom];
        double sf_value = 0;
        d_stru_factor[c_atom] = 0;
        for (int neigh_atom = neigh_min; neigh_atom < neigh_max; neigh_atom++){
            double delt_x, delt_y, delt_z, r2, r;
            delt_x = d_group_dminneigh[neigh_atom*4 + 0];
            delt_y = d_group_dminneigh[neigh_atom*4 + 1];
            delt_z = d_group_dminneigh[neigh_atom*4 + 2];
            r2 = d_group_dminneigh[neigh_atom*4 + 3];
            r      = sqrt(r2);
            theta  = q_factor*r;
            sincos(theta, &sin_theta, &cos_theta);
            if (r2 > r_on){
                s = 1.0 - POW3((r2-r_on)/(cutoff_rsq-r_on));
            } else {
                s = 1.0;
                // ds = 0.0;
            }
            d_stru_factor[c_atom] += sin_theta/theta*s;
        }
        // d_stru_factor[c_atom] /= (double)(d_neigh_in_cutoff_r[c_atom]);
        d_stru_factor[c_atom] += 1.0;
    }
}

// __global__ void structure_factor_cv_Filter(
//         int group_count, double q_factor_filter,
//         LAMMPS_NS::tagint *d_group_numneigh,
//         double *d_stru_factor){

//     int c_atom = blockIdx.x * blockDim.x + threadIdx.x;
//     double r_on = 0.8*cutoff_rsq;
//     double s = 1.0;
//     // double ds = 0.0;
//     if(c_atom<group_count){
//         d_stru_factor[c_atom] = q_factor_filter;
//     }
// }

__global__ void structure_factor_dcv_AVE_kernel(
        int group_count, int groupbit, int all_count, double cutoff_rsq,
        int *d_mask, LAMMPS_NS::tagint *d_group_indices, 
        LAMMPS_NS::tagint *d_calculated_numneigh,
        LAMMPS_NS::tagint *d_group_numneigh,
        int *d_neigh_in_cutoff_r, double *d_group_dminneigh,
        double q_factor, double *d_stru_factor, 
        double *d_dcvdx){
    
    double r_on = 0.8*cutoff_rsq;
    double s = 1.0;
    double ds = 0.0;
    // devise version=============
    int c_atom = blockIdx.x * blockDim.x + threadIdx.x;
    if(c_atom<group_count){
    // host version===============
    // for (int c_atom=0; c_atom<group_count; c_atom++){
        int neigh_tag, neigh_Nb;
        int neigh_min, neigh_max;
        double neigh_q6_timesN, catom_q4_timesN;
        double dx, dy, dz, r2, r;
        double NeighInGroupWeight, temp, Stru_fact;
        double theta, sin_theta, cos_theta;
        // double sin_6phi, cos_6phi;
        int Stru_fact_base_id, Stru_fact_neigh_id, Neigh_Nb;
        neigh_min = d_group_numneigh[c_atom];
        neigh_max = d_group_numneigh[c_atom] + d_neigh_in_cutoff_r[c_atom];
        int neigh_num = d_neigh_in_cutoff_r[c_atom];
        for(int i=0; i<3; i++){
            // from 0 to l, both re_part and im_part
            d_dcvdx[c_atom*3 + i] = 0;
        }
        if (neigh_num == 0) {
            return;
        }
        for(int neigh_atom=neigh_min; neigh_atom<neigh_max; neigh_atom++){
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
            } else {
                s = 1.0;
            }
            temp = (NeighInGroupWeight * q_factor/ all_count) \
                        *(cos_theta/theta - sin_theta/POW2(theta)) *s;
            d_dcvdx[c_atom*3 + 0] -= (temp)*dx/r;
            d_dcvdx[c_atom*3 + 1] -= (temp)*dy/r;
            d_dcvdx[c_atom*3 + 2] -= (temp)*dz/r;
        }
    }

}

REGISTER_CV("STRU_FACTOR", MetaD_zqc::Stru_factor::create);