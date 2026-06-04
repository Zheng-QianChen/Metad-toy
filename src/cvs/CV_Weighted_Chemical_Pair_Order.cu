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
#include "CV_Weighted_Chemical_Pair_Order.h"

#include <cuda_runtime.h>
#include <cstring>
#include <algorithm>
#include <cmath>


using namespace LAMMPS_NS;

MetaD_zqc::CV* MetaD_zqc::Weighted_chem_pair::create(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::FixMetadynamics *Fixmetad, 
                                            int narg, char **arg, int &i, FILE *f_check){
    DEBUG_LOG("In Weighted_chem_pair settings");
    LAMMPS_NS::Error *error = lmp->error;

    std::string cal_name = arg[i];

    MetaD_zqc::StruFactorRequest req;

    req.cal_name = cal_name;
    // 原子环境分析-初始设置
    // Usage: CHEM_PAIR <group>
    ERR_COND(i + 1 >= narg, "Error: CHEM_PAIR command requires at least \"CHEM_PAIR <group> \".");
    req.group_name = arg[i+1];
    req.group_id = lmp->group->find(req.group_name);
    ERR_COND(req.group_id == -1, "Error: group name %s not found.", req.group_name);
    //参数有效性
    // default values
    req.cutoff_r = 8.0;
    req.d_block_size = 128;
    req.custom_weights.clear();
    int iarg=2 + i;
    printf("im in Weighted_chem_pair::create, with group_name=%s, group_id=%d\n", req.group_name, req.group_id);
    while (iarg < narg) {
        if (strcmp(arg[iarg], "cutoff_r") == 0) {
            ERR_COND((iarg + 1 >= narg) ,"Error: \'cutoff_r\' keyword requires a value");
            req.cutoff_r = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
            iarg += 2;
        } else if (strcmp(arg[iarg], "d_block_size") == 0) {
            ERR_COND((iarg + 1 >= narg), "Error: \'d_block_size\' keyword requires an integer");
            req.d_block_size = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
            ERR_COND(req.d_block_size <= 0, "Error: \'d_block_size\' must be > 0");
            iarg += 2;
        } else if (strcmp(arg[iarg], "SW_func") == 0) {
            ERR_COND((iarg + 1 >= narg), "Error: \'SW_func\' keyword requires a type (FERMI, TANH, RATIONAL)");
            req.use_sw_func = true;
            // 1. 解析开关函数类型
            std::string sw_type_str = arg[iarg + 1];
            if (sw_type_str == "FERMI") {
                req.sw_func_req.type = FERMI;
                // 设置该类型的默认值
                req.sw_func_req.r_0 = 1.0; 
                req.sw_func_req.alpha = 20.0;
            } else if (sw_type_str == "TANH") {
                req.sw_func_req.type = TANH_TYPE;
                req.sw_func_req.r_0 = 1.0;
                req.sw_func_req.alpha = 20.0;
            } else if (sw_type_str == "RATIONAL") {
                req.sw_func_req.type = RATIONAL;
                // 标准 PLUMED 默认值
                req.sw_func_req.r_0 = 1.25;
                req.sw_func_req.d_0 = 0.0;
                req.sw_func_req.n = 6;
                req.sw_func_req.m = 12;
            } else {
                error->all(FLERR, "Error: Unknown SW_func type. Choose from FERMI, TANH, RATIONAL.");
            }
            iarg += 2; // 消耗掉 "SW_func" 和 "TYPE"
            // 2. 内层循环：动态解析该开关函数的内部亚参数
            while (iarg < narg) {
                if (strcmp(arg[iarg], "r_0") == 0) {
                    ERR_COND((iarg + 1 >= narg), "Error: \'r_0\' requires a numeric value");
                    req.sw_func_req.r_0 = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
                    iarg += 2;
                } else if (strcmp(arg[iarg], "d_0") == 0) {
                    ERR_COND((iarg + 1 >= narg), "Error: \'d_0\' requires a numeric value");
                    req.sw_func_req.d_0 = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
                    iarg += 2;
                } else if (strcmp(arg[iarg], "alpha") == 0) {
                    ERR_COND((iarg + 1 >= narg), "Error: \'alpha\' requires a numeric value");
                    req.sw_func_req.alpha = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
                    iarg += 2;
                } else if (strcmp(arg[iarg], "n") == 0) {
                    ERR_COND((iarg + 1 >= narg), "Error: \'n\' requires an integer value");
                    req.sw_func_req.n = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
                    iarg += 2;
                } else if (strcmp(arg[iarg], "m") == 0) {
                    ERR_COND((iarg + 1 >= narg), "Error: \'m\' requires an integer value");
                    req.sw_func_req.m = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
                    iarg += 2;
                } else {
                    // 遇到不属于开关函数的参数（例如到了下一个主关键字 Chemical 或 cutoff_r），退出内层循环
                    break;
                }
            }
        } else if (strcmp(arg[iarg], "User_Chem_weight") == 0) {
            req.use_chemical_lock = true;
            iarg += 1;
        } else if (strcmp(arg[iarg], "Chem_ctarget") == 0) {
            ERR_COND((iarg + 1 >= narg), "Error: \'Chem_ctarget\' keyword requires a number");
            req.c_target = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
            iarg += 2;
        } else if (strcmp(arg[iarg], "Chem_sigma") == 0) {
            ERR_COND((iarg + 1 >= narg), "Error: \'sigma\' keyword requires a number");
            req.sigma = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
            iarg += 2;
        } else if (strcmp(arg[iarg], "Chem_map") == 0) {
            iarg += 1; // 跳过 "Chem_map" 关键字
            req.use_custom_weight = true;
            // 循环读取后面所有的 (type weight) 括号对
            while (iarg < narg && arg[iarg][0] == '(') {
                std::string pair_str = arg[iarg];
                
                // 健壮性处理：如果长字符串包含了空格分裂（某些系统传参导致的分离）
                // 或者形如 "(1" "5.0)" 的情况，为了安全，我们要求输入必须是紧凑的 "(1,1.0)" 或 "(1 1.0)"
                // 并在内部去除左右括号
                if (pair_str.front() == '(' && pair_str.back() == ')') {
                    pair_str = pair_str.substr(1, pair_str.size() - 2); // 掐头去尾
                    
                    // 兼容空格或逗号分隔
                    size_t sep = pair_str.find_first_of(" ,"); 
                    if (sep != std::string::npos) {
                        std::string type_s = pair_str.substr(0, sep);
                        std::string weight_s = pair_str.substr(sep + 1);
                        
                        // 去除可能残余的空格
                        int a_type = std::stoi(type_s);
                        double a_weight = std::stod(weight_s);
                        
                        req.custom_weights[a_type] = a_weight;
                    }
                } else {
                    // 如果 LAMMPS 把 (1 1.0) 拆成了两个 token: "(1" 和 "1.0)"
                    // 我们通过向前探测来组合它们
                    std::string part1 = arg[iarg];
                    ERR_COND(iarg + 1 >= narg, "Error: Invalid Chem_map format near %s", arg[iarg]);
                    std::string part2 = arg[iarg+1];
                    
                    if (part1.front() == '(' && part2.back() == ')') {
                        int a_type = std::stoi(part1.substr(1));
                        double a_weight = std::stod(part2.substr(0, part2.size() - 1));
                        req.custom_weights[a_type] = a_weight;
                        iarg += 1; // 额外消费一个参数
                    } else {
                        error->all(FLERR, "Error: Chem_map pairs must be formatted as (type weight)");
                    }
                }
                iarg += 1;
            }
        } else {
            break;
        }
    }
    
    // We need full neighbor list to get cuda run faster
    NeighRequest *full_request;
    full_request = lmp->neighbor->add_request(Fixmetad, NeighConst::REQ_FULL);
    full_request->set_id(2);

    MetaD_zqc::Weighted_chem_pair* Weighed_chem = nullptr;
    if (req.use_chemical_lock) {
        LOG("Logging: set CHEM_PAIR as group_name=%s cutoff_r=%f d_block_size=%d.\n         Chemical lock is ON, with c_target=%g and sigma=%g.",
            req.group_name, req.cutoff_r, req.d_block_size,
            req.c_target, req.sigma);
        // create Structure factor CV
        // env for CV
        MetaD_zqc::Stru_fact_chem_env *temp_env = static_cast<MetaD_zqc::Stru_fact_chem_env*>(
                                    MetaD_zqc::Stru_fact_env::get_or_create(lmp, f_check, Fixmetad, req)
                                );
        DEBUG_LOG("Stru_fact_chem_env is %p", temp_env);
        std::string env_setNum = temp_env->get_env_key();
        i = iarg;
        // return Stru_fact cv
        Weighed_chem = new MetaD_zqc::Weighted_chem_pair(lmp, Fixmetad, f_check, 
                                env_setNum, req.group_id, temp_env,
                                req.d_block_size);
    }

    if (req.use_sw_func){
        auto temp_sw = new MetaD_zqc::SwitchFunction(req.sw_func_req.type, req.sw_func_req.r_0, req.sw_func_req.d_0, 
                               req.sw_func_req.alpha, req.sw_func_req.n, req.sw_func_req.m);
        LOG("%s\n", temp_sw->get_summary_string().c_str());
        Weighed_chem->h_sw_func = temp_sw;
        Weighed_chem->use_sw_func = true;
        Weighed_chem->sw_params  = req.sw_func_req;
    }
    
    return Weighed_chem;
}


MetaD_zqc::Weighted_chem_pair::Weighted_chem_pair(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check, 
                             std::string env_setNum, int group_id,
                             MetaD_zqc::Stru_fact_chem_env* my_env,
                             int d_block_size)
                        : CV(lmp, f_check),
                            Fixmetad(Fixmetad),
                            env_setNum(env_setNum),
                            // q_factor(q_factor),
                            // group_id(group_id),
                            my_env(my_env),
                            d_block_size(d_block_size){
    // my_averager = new MetaD_zqc::CUBAverager();
    my_averager = new MetaD_zqc::KahanAverager();
    DEBUG_LOG("Logging: New a Weighted_chem_pair file, will generate %d lines in GPU,\n     with cutoff_r=%g",
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
    lmp->memory->create(h_chem_pair_r, 0, "metad:Weighted_chem_pair:h_chem_pair_r");
    lmp->memory->create(h_dcvdx, 0, "metad:Weighted_chem_pair:h_dcvdx");
    error = lmp->error;

    
    // comment name
    d_chem_pair_r.set_name("d_chem_pair_r");
    d_dcvdx.set_name("d_dcvdx");
}

MetaD_zqc::Weighted_chem_pair::~Weighted_chem_pair(){
    lmp->memory->destroy(h_chem_pair_r);
    lmp->memory->destroy(h_dcvdx);
    // the GpuBuffer will automatically release its memory, 
    // so we don't need to manually free it here
}


void MetaD_zqc::Weighted_chem_pair::environment(){
    DEBUG_LOG("last_update_step is %lld, group_count=%d", (long long)my_env->last_update_step, my_env->group_count);
    if (lmp->update->ntimestep > my_env->last_update_step){
        my_env->get_env();
    }
    // DEBUG_LOG("environment function in, env_setNum is %s, get_env done",env_setNum);
    DEBUG_LOG("last_update_step is %lld, group_count=%d", (long long)my_env->last_update_step, my_env->group_count);
}


auto MetaD_zqc::Weighted_chem_pair::set_CV_calculate(std::string func_name) -> CV_Calculation {
    if (func_name == "AVE") {
        return static_cast<CV_Calculation>(&Weighted_chem_pair::compute_cv_AVE);
    } else if (func_name == "FILTER_SUM") {
        // return static_cast<CV_Calculation>(&Weighted_chem_pair::compute_cv_FILTER_SUM);
    } else if (func_name == "COUNT") {
        return static_cast<CV_Calculation>(&Weighted_chem_pair::compute_cv_COUNT);
    }
    return nullptr;
}

auto MetaD_zqc::Weighted_chem_pair::set_CV_bias_force(std::string func_name) -> CV_BiasForce {
    if (func_name == "AVE") {
        return static_cast<CV_BiasForce>(&Weighted_chem_pair::bias_force_AVE);
    } else if (func_name == "FILTER_SUM") {
        // return static_cast<CV_Calculation>(&Weighted_chem_pair::bias_force_FILTER_SUM);
    } else if (func_name == "COUNT") {
        return static_cast<CV_BiasForce>(&Weighted_chem_pair::bias_force_COUNT);
    }
    return nullptr;
}

void MetaD_zqc::Weighted_chem_pair::base_calc(){
    compute_Weighted_chem_pair_peratoms();
}

double MetaD_zqc::Weighted_chem_pair::compute_cv_AVE(){
    DEBUG_LOG("im in compute_cv_AVE.");
    int group_count = my_env->group_count;
    int Threads_own_atoms = lmp->atom->nlocal;
    DEBUG_LOG("group_count = %d",group_count);
    double sf_ave_local=0;
    DEBUG_LOG_COND((h_chem_pair_r == NULL),"h_chem_pair_r list not initialized");
    if (group_count != 0) {
        my_averager->compute(Threads_own_atoms, all_count, h_chem_pair_r, 
            lmp->atom->mask, my_env->groupbit, sf_ave_local);
    }
    MPI_Allreduce(&sf_ave_local, &cv_value, 1, MPI_DOUBLE, MPI_SUM, lmp->world);
    DEBUG_LOG("group_count = %d, compute_cv_AVE = %g",group_count, cv_value);
    return cv_value;
}

double MetaD_zqc::Weighted_chem_pair::compute_cv_COUNT(){
    DEBUG_LOG("im in compute_cv_COUNT.");
    int group_count = my_env->group_count;
    DEBUG_LOG("group_count = %d",group_count);
    double sf_count_local=0;
    DEBUG_LOG_COND((h_chem_pair_r == NULL),"h_chem_pair_r list not initialized");
    if (group_count != 0) {
        for (int c_atom=0; c_atom<group_count; c_atom++){
            int c_tag = (my_env->h_group_indices)[c_atom];
            double Si = h_chem_pair_r[c_tag];
            sf_count_local += h_sw_func->f(sw_params, Si);
        }
    }
    MPI_Allreduce(&sf_count_local, &cv_value, 1, MPI_DOUBLE, MPI_SUM, lmp->world);
    DEBUG_LOG("group_count = %d, compute_cv_COUNT = %g",group_count, cv_value);
    return cv_value;
}

void MetaD_zqc::Weighted_chem_pair::compute_Weighted_chem_pair_peratoms(){
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
        // h_chem_pair_r for all aim atoms
        int Threads_own_atoms = lmp->atom->nlocal + lmp->atom->nghost;
        lmp->memory->grow(h_chem_pair_r, Threads_own_atoms, "metad:Weighted_chem_pair:cv_bound");
        DEBUG_LOG("d_block_size is %d, block_num is %d",d_block_size, block_num);
    }
    DEBUG_LOG("group_count=%lld",(long long)my_env->group_count);

    // 2. calculate atoms' environment
    DEBUG_LOG("environment function in, env_setNum is %s",env_setNum.c_str());
    environment();
    DEBUG_LOG("environment function out");

    // 3. calculate atoms' other things
    wcp_param_calc(h_chem_pair_r);

    DEBUG_LOG_COND((my_env->group_dminneigh == NULL),"group_dminneigh list not initialized");
    DEBUG_LOG("group_dminneigh Allocated at: %p", my_env->group_dminneigh);
    
    // 输出group中每个原子的sf值
    DEBUG_RUN(for(int c_atom=0;c_atom<my_env->group_count;c_atom++)
                {
                    DEBUG_LOG("Weighted_chem_pair[%lld] = %f",(long long)c_atom,
                    h_chem_pair_r[my_env->h_group_indices[c_atom]]);
                });
    DEBUG_LOG("post_force function end");
}


void MetaD_zqc::Weighted_chem_pair::bias_force_AVE(double dVdcv){
    // pass
    DEBUG_LOG("MetaD_zqc::Weighted_chem_pair::bias_force_AVE");
    double **f = lmp->atom->f;
    double **x = lmp->atom->x;
    int c_tag;
    double sumForce[3] = {0.0, 0.0, 0.0};
    DEBUG_LOG("MetaD_zqc::Weighted_chem_pair::bias_force_AVE");
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
             error->all(FLERR, "Weighted_chem_pair CV error: force is infinity, check your system or cv_value.");
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
    DEBUG_LOG_COND(sumforce_mod>1e-10,"Warning: bias force will make the system flow. sumforce_mod=%g",sumforce_mod);
    DEBUG_LOG("sumForce=%g",sumforce_mod);
    DEBUG_LOG("post_force_r_end");
}


void MetaD_zqc::Weighted_chem_pair::get_dcvdx_AVE(double cv_value, double *dcvdx){
    int group_count = my_env->group_count;
    int last_group_count = my_env->last_group_count;
    int Threads_own_atoms = lmp->atom->nlocal + lmp->atom->nghost;
    size_t datalen = 0;
    

    // DEBUG_RUN(
    datalen = Threads_own_atoms;
    lmp->memory->grow(h_chem_pair_r, datalen, "Weighted_chem_pair:h_chem_pair_r");
    SAFE_CUDA_MEMCPY(h_chem_pair_r, d_chem_pair_r.ptr, datalen*sizeof(double), cudaMemcpyDeviceToHost,f_check);


    datalen = (group_count*3);
    lmp->memory->grow(h_dcvdx, datalen, "Weighted_chem_pair:h_dcvdx");
    d_dcvdx.grow_to(datalen, f_check, __FILE__, __LINE__);
    SAFE_CUDA_MEMCPY(d_dcvdx.ptr,h_dcvdx, datalen*sizeof(double),cudaMemcpyHostToDevice,f_check);

    // sync Stein_qlm and stein_q with communication
    // then we can directly use the data in device to calculate dcvdx, 
    // without worrying about the data consistency between MPI processes.
    DEBUG_LOG("[Rank:%d][Before Comm] Weighted_chem_pair[0] = %f, ptr = %p\n",lmp->comm->me, h_chem_pair_r[0], (void*)h_chem_pair_r);
    cudaDeviceSynchronize(); // waiting memory
    MPI_Barrier(lmp->world); // ensure all processes reach this point before communication
    comm_mode=true;
    lmp->comm->forward_comm(Fixmetad);
    comm_mode=false;
    DEBUG_LOG("[Rank:%d][After Comm] Weighted_chem_pair[0] = %f, ptr = %p\n",lmp->comm->me, h_chem_pair_r[0], (void*)h_chem_pair_r);
    // for (int i=0; i<((Threads_own_atoms)); i++){
    //     printf("Weighted_chem_pair[%d] = %f\n", i, h_chem_pair_r[i]);
    // }


    DEBUG_LOG("i will start a kernel of ql");
    cudaDeviceSynchronize(); // waiting memory
    call_Weighted_chem_pair_dcv_AVE_kernel();
    cudaDeviceSynchronize(); // waiting memory
    DEBUG_LOG("i am out");

    cudaMemcpy(h_dcvdx, d_dcvdx.ptr, (group_count*3)*sizeof(double), cudaMemcpyDeviceToHost);
    // SAFE_CUDA_MEMCPY(h_dcvdx, d_dcvdx,
    //   (group_count*3)*sizeof(double), cudaMemcpyDeviceToHost, file);
    cudaDeviceSynchronize(); // waiting memory
}


void MetaD_zqc::Weighted_chem_pair::bias_force_COUNT(double dVdcv){
    // pass
    DEBUG_LOG("MetaD_zqc::Weighted_chem_pair::bias_force_COUNT");
    double **f = lmp->atom->f;
    double **x = lmp->atom->x;
    int c_tag;
    double sumForce[3] = {0.0, 0.0, 0.0};
    DEBUG_LOG("MetaD_zqc::Weighted_chem_pair::bias_force_COUNT");
    this->get_dcvdx_COUNT(cv_value, h_dcvdx);
    // DEBUG_LOG("cv_value = %g, dVdcv = %g, dcvdx = %g, %g, %g",cv_value, dVdcv, dcvdx[0], dcvdx[1], dcvdx[2]);
    // DEBUG_LOG("fx0,fy0,fz0  = %.6f, %.6f, %.6f", f[c_tag][0], f[c_tag][1], f[c_tag][2]);
    for (int c_atom=0; c_atom<(my_env->group_count); c_atom++){
        DEBUG_LOG("dcvdx, dcvdy, dcvdz  = %g, %g, %g", h_dcvdx[c_atom*3 + 0], h_dcvdx[c_atom*3 + 1], h_dcvdx[c_atom*3 + 2]);
        DEBUG_LOG("dVdcv  = %g", dVdcv);
        c_tag = (my_env->h_group_indices)[c_atom];
        DEBUG_LOG("fx0,fy0,fz0  = %g, %g, %g", f[c_tag][0], f[c_tag][1], f[c_tag][2]);
        if (isnan(f[c_tag][0])||isnan(f[c_tag][1])||isnan(f[c_tag][2])){
            printf("error: force is infinity, check your system or cv_value.\n");
             error->all(FLERR, "Weighted_chem_pair CV error: force is infinity, check your system or cv_value.");
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
    DEBUG_LOG_COND(sumforce_mod>1e-10,"Warning: bias force will make the system flow. sumforce_mod=%g",sumforce_mod);
    DEBUG_LOG("sumForce=%g",sumforce_mod);
    DEBUG_LOG("post_force_r_end");
}


void MetaD_zqc::Weighted_chem_pair::get_dcvdx_COUNT(double cv_value, double *dcvdx){
    int group_count = my_env->group_count;
    int last_group_count = my_env->last_group_count;
    int Threads_own_atoms = lmp->atom->nlocal + lmp->atom->nghost;
    size_t datalen = 0;
    

    // DEBUG_RUN(
    datalen = Threads_own_atoms;
    lmp->memory->grow(h_chem_pair_r, datalen, "Weighted_chem_pair:h_chem_pair_r");
    SAFE_CUDA_MEMCPY(h_chem_pair_r, d_chem_pair_r.ptr, datalen*sizeof(double), cudaMemcpyDeviceToHost,f_check);


    datalen = (group_count*3);
    lmp->memory->grow(h_dcvdx, datalen, "Weighted_chem_pair:h_dcvdx");
    d_dcvdx.grow_to(datalen, f_check, __FILE__, __LINE__);
    SAFE_CUDA_MEMCPY(d_dcvdx.ptr,h_dcvdx, datalen*sizeof(double),cudaMemcpyHostToDevice,f_check);

    // sync Stein_qlm and stein_q with communication
    // then we can directly use the data in device to calculate dcvdx, 
    // without worrying about the data consistency between MPI processes.
    DEBUG_LOG("[Rank:%d][Before Comm] Weighted_chem_pair[0] = %f, ptr = %p\n",lmp->comm->me, h_chem_pair_r[0], (void*)h_chem_pair_r);
    cudaDeviceSynchronize(); // waiting memory
    MPI_Barrier(lmp->world); // ensure all processes reach this point before communication
    comm_mode=true;
    lmp->comm->forward_comm(Fixmetad);
    comm_mode=false;
    DEBUG_LOG("[Rank:%d][After Comm] Weighted_chem_pair[0] = %f, ptr = %p\n",lmp->comm->me, h_chem_pair_r[0], (void*)h_chem_pair_r);
    // for (int i=0; i<((Threads_own_atoms)); i++){
    //     printf("Weighted_chem_pair[%d] = %f\n", i, h_chem_pair_r[i]);
    // }

    DEBUG_LOG("i will start a kernel of ql");
    cudaDeviceSynchronize(); // waiting memory
    call_Weighted_chem_pair_dcv_COUNT_kernel();
    cudaDeviceSynchronize(); // waiting memory
    DEBUG_LOG("i am out");

    cudaMemcpy(h_dcvdx, d_dcvdx.ptr, (group_count*3)*sizeof(double), cudaMemcpyDeviceToHost);
    // SAFE_CUDA_MEMCPY(h_dcvdx, d_dcvdx,
    //   (group_count*3)*sizeof(double), cudaMemcpyDeviceToHost, file);
    cudaDeviceSynchronize(); // waiting memory
}


void MetaD_zqc::Weighted_chem_pair::wcp_param_calc(double *h_chem_pair_r){
    int last_group_count = my_env->last_group_count;
    int group_count = my_env->group_count;
    int Threads_own_atoms = lmp->atom->nlocal + lmp->atom->nghost;
    cudaStream_t lammps_stream = 0; // Assuming you want to use the default stream. Adjust if you have a specific stream.

    d_chem_pair_r.grow_to(Threads_own_atoms, f_check, __FILE__, __LINE__);
    cudaMemsetAsync(d_chem_pair_r.ptr, 0, (Threads_own_atoms)*sizeof(double), lammps_stream);

    DEBUG_LOG("i will start a kernel of ql");
    cudaDeviceSynchronize(); // waiting memory
    // TODO:
    call_Weighted_chem_pair_cv_kernel();
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

    SAFE_CUDA_MEMCPY(h_chem_pair_r, d_chem_pair_r.ptr,
      (Threads_own_atoms) * sizeof(double), cudaMemcpyDeviceToHost,f_check);

}

void MetaD_zqc::Weighted_chem_pair::summary(FILE* f){}

int MetaD_zqc::Weighted_chem_pair::get_comm_forward_bytes(){ 
    return 1; // Weighted_chem_pair
}

int MetaD_zqc::Weighted_chem_pair::pack_comm_ubuf(int n, int *list, double *u_buf, int slot_offset, int comm_forward) {
    if (!comm_mode){
        return 1;
    }
    int m = slot_offset; 
    int cycle_offset = comm_forward;

    for (int i = 0; i < n; i++) {
        int j = list[i];
        u_buf[m + cycle_offset*i] = h_chem_pair_r[j];
    }
    
    return 1;
}

void MetaD_zqc::Weighted_chem_pair::unpack_comm_ubuf(int n, int first, double *u_buf, int slot_offset, int comm_forward) {
    if (!comm_mode){
        return;
    }

    int m = slot_offset; 
    int cycle_offset = comm_forward;
    
    // 从 first 开始，连续恢复 n 个 Ghost 原子的复合数据
    for (int i = first; i < first + n; i++) {
        h_chem_pair_r[i] = u_buf[ m+ cycle_offset*(i-first)];
    }
}


double* MetaD_zqc::Weighted_chem_pair::get_peratom_ptr(const std::string &prop_name) {
    if (prop_name == "chem_pair_r") {
        return h_chem_pair_r;
    }
    return nullptr;
}

void MetaD_zqc::Weighted_chem_pair::call_Weighted_chem_pair_cv_kernel(){
    Weighted_chem_pair_cv_kernel<<<block_num,d_block_size>>>(
        (my_env->group_count), POW2(my_env->cutoff_r),
        (my_env->d_group_numneigh.ptr), 
        (my_env->d_group_dminneigh.ptr), 
        (my_env->d_group_indices.ptr), 
        (my_env->d_firstneigh_ptrs.ptr),
        (my_env->d_neigh_in_cutoff_r.ptr), 
        (my_env->d_atom_types.ptr), (my_env->d_type_weights.ptr),
        d_chem_pair_r.ptr);
}

void MetaD_zqc::Weighted_chem_pair::call_Weighted_chem_pair_dcv_AVE_kernel(){
    // printf("[Rank:%d]d_stein_Ylm is located in %p\n",lmp->comm->me,d_stein_Ylm.ptr);
    Weighted_chem_pair_dcv_AVE_kernel<<<block_num,d_block_size>>>(
        (my_env->group_count), (my_env->groupbit), 
        all_count, POW2(my_env->cutoff_r),
        (my_env->d_mask.ptr), (my_env->d_group_indices.ptr), 
        (my_env->d_calculated_numneigh.ptr), (my_env->d_group_numneigh.ptr),
        (my_env->d_neigh_in_cutoff_r.ptr), (my_env->d_group_dminneigh.ptr),
        d_chem_pair_r.ptr, 
        (my_env->d_atom_types.ptr), (my_env->d_type_weights.ptr),
        d_dcvdx.ptr);
}

// TODO: 实现COUNT版本的dcvdx计算kernel
void MetaD_zqc::Weighted_chem_pair::call_Weighted_chem_pair_dcv_COUNT_kernel(){
    // printf("[Rank:%d]d_stein_Ylm is located in %p\n",lmp->comm->me,d_stein_Ylm.ptr);
    // Weighted_chem_pair_dcv_COUNT_kernel<<<block_num,d_block_size>>>(
    //     sw_params,
    //     (my_env->group_count), (my_env->groupbit), all_count, POW2(my_env->cutoff_r),
    //     (my_env->d_mask.ptr), (my_env->d_group_indices.ptr), 
    //     (my_env->d_calculated_numneigh.ptr), (my_env->d_group_numneigh.ptr),
    //     (my_env->d_neigh_in_cutoff_r.ptr), (my_env->d_group_dminneigh.ptr),
    //     d_chem_pair_r.ptr, 
    //     d_dcvdx.ptr);
}


__global__ void Weighted_chem_pair_cv_kernel(
        int group_count, double cutoff_rsq,
        LAMMPS_NS::tagint *d_group_numneigh,
        double *d_group_dminneigh, 
        LAMMPS_NS::tagint *d_group_indices,
        int *d_firstneigh_ptrs, 
        int *d_neigh_in_cutoff_r, 
        int *d_atom_types, double *d_type_weights,
        double *d_chem_pair_r){

    int c_atom = blockIdx.x * blockDim.x + threadIdx.x;
    double r_on = 0.8*cutoff_rsq;
    // double ds = 0.0;
    if(c_atom<group_count){
        int neigh_min, neigh_max;
        double chem_weight_catoms = d_type_weights[(int)d_atom_types[d_group_indices[c_atom]]];
        neigh_min = d_group_numneigh[c_atom];
        neigh_max = d_group_numneigh[c_atom] + d_neigh_in_cutoff_r[c_atom];
        d_chem_pair_r[c_atom] = 0;
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
            // theta  = q_factor*r;
            // sincos(theta, &sin_theta, &cos_theta);
            int n_local_tag = d_firstneigh_ptrs[neigh_atom];
            chem_weight = chem_weight_catoms*d_type_weights[(int)d_atom_types[n_local_tag]]; // 通过邻居原子类型获取化学权重
            double s = 1.0;
            if (r2 > r_on){
                s = 1.0 - POW3((r2-r_on)/(cutoff_rsq-r_on));
            }
            d_chem_pair_r[c_atom] += s*chem_weight;
        }
        // d_chem_pair_r[c_atom] /= (double)(d_neigh_in_cutoff_r[c_atom]);
        d_chem_pair_r[c_atom] += 1.0;
    }
}


__global__ void Weighted_chem_pair_dcv_AVE_kernel(
        int group_count, int groupbit, 
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
            // theta = q_factor*r;
            // sincos(theta, &sin_theta, &cos_theta);
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
            temp = (NeighInGroupWeight / all_count)*(ds); 
            dcvdx_local[0] -= chem_weight*(temp)*dx/r;
            dcvdx_local[1] -= chem_weight*(temp)*dy/r;
            dcvdx_local[2] -= chem_weight*(temp)*dz/r;
        }
        d_dcvdx[c_atom * 3 + 0] = dcvdx_local[0];
        d_dcvdx[c_atom * 3 + 1] = dcvdx_local[1];
        d_dcvdx[c_atom * 3 + 2] = dcvdx_local[2];
    }
}