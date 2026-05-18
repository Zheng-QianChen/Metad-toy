

#include "fix_crystallize.h"
#include "lammpsplugin.h"
#include "atom.h"
#include "comm.h"
#include "update.h"
#include "domain.h"
#include "memory.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
// #include "compute.h"
#include <cmath>
#include <cstdio>
#include "zqc_CVs.h"
#include "zqc_debug.h"
#include "zqc_DimSet.h"
#include "zqc_gaussian.h"
#include "zqc_mlcvs.h"

#include <cuda_runtime.h>
using namespace LAMMPS_NS;



FixMetadynamics::FixMetadynamics(LAMMPS *lmp, int narg, char **arg)
                                : Fix(lmp, narg, arg),
                                  pace(100),
                                  cv_dim(1), nbin_num(100){
    if (comm->me==0){
        f_check = fopen("metad_debug_logging.txt","w");
        LOG("New JOB STARTING WITH DEBUG MOD!");
        fclose(f_check);
    }
    MPI_Barrier(world);
    f_check = fopen("metad_debug_logging.txt","a");
    LOG("MPI_COMM_WORLD size: %d", comm->nprocs);
    pthread_t tid = pthread_self();
    LOG("Process Rank: %d | Total Processes: %d | Thread ID: %lu",
            comm->me, comm->nprocs, (unsigned long)tid);
    // Fix 构造函数的基本参数检查： fix ID group style args...
    // arg[0]=fix, arg[1]=ID, arg[2]=group, arg[3]=style (metad)
    // if (narg < 4) error->all(FLERR, "Fix metadynamics requires at least 1 argument after style name.");
    LOG("There are %d args", narg);
    // --- 核心参数解析：循环读取关键词/数值对 ---
    int i = 3; // 从第 4 个参数开始，即 style 名之后
    // std::vector<MetaD_zqc::SteinhardtRequest> steinh_requests;
    cv_configs = new MetaD_zqc::MetaDimensionManager();
    double *cv_bound;
    int *nbin;
    int Gaussian_Hill_type = 0;
    double sigma, height0, biasf, KB;
    sigma     = 0.05;
    height0   = 0.1;
    biasf     = 10.0;
    KB        = 0.025852;
    int continue_from_file=false;
    int WellT_bool=false;
    // Gaussian_Hill_type={
    // 0: 均匀网格
    // 1: 稀疏网格
    // 2: KDTree
    // }
    while (i < narg) {
        LOG("Im in arg loop");
        if (strcmp(arg[i], "GAUSSIAN") == 0) {
            ERR_COND(i + 3 >= narg, "Error: GAUSSIAN command requires 3 arguments: sigma, height, biasf.");
            sigma   = utils::numeric(FLERR, arg[i+1], false, lmp);
            height0 = utils::numeric(FLERR, arg[i+2], false, lmp);
            biasf   = utils::numeric(FLERR, arg[i+3], false, lmp);
            i += 4;

            // double R_SI = 8.314462618;
            // height0 = KB*height0/R_SI;

            LOG("Logging: set GAUSSIAN sigma, height, biasf: %g %g %g.", sigma, height0, biasf);
            LOG("Logging: attention, we use Joules/moles as the height's units, so if the lammps settings is not real, there will be a units transform.\n\
             If you set CV's bounds in physicals, this units will follows your lammps units settings.");
        } else if (strcmp(arg[i], "PACE") == 0) {
            ERR_COND(i + 1 >= narg, "Error: PACE command requires 1 argument: integer timesteps.");
            pace = utils::inumeric(FLERR, arg[i+1], false, lmp);
            i += 2;
            LOG("Logging: pace = %d.",pace);
        } else if (strcmp(arg[i], "RECORD") == 0) {
            ERR_COND(i + 1 >= narg, "Error: RECORD command requires 1 argument: integer timesteps.");
            int iarg=1 + i;
            while (iarg < narg) {
                if (strcmp(arg[iarg], "FILE_NAME") == 0) {
                    ERR_COND((iarg + 1 >= narg) ,"Error: \'FILE_NAME\' keyword requires a value");
                    rec_file_name = arg[iarg+1];
                    iarg += 2;
                }
                else if (strcmp(arg[iarg], "REC_PACE") == 0) {
                    ERR_COND((iarg + 1 >= narg), "Error: \'REC_PACE\' keyword requires an integer");
                    rec_pace = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
                    iarg += 2;
                }
                else {
                  break;
                }
            }
            LOG("Logging: rec_file = %s, rec_pace = %d.",rec_file_name.c_str(), rec_pace);
            rec_file = fopen(rec_file_name.c_str(),"w+");
            i = iarg;
        } else if (strcmp(arg[i], "CV_dim") == 0) {
            DEBUG_LOG("In CV_dim settings");
            ERR_COND(i + 1 >= narg, "Error: PACE command requires 1 argument: integer timesteps.");
            cv_dim = utils::inumeric(FLERR, arg[i+1], false, lmp);
            memory->create(nbin, cv_dim, "metad:nbin_size");
            memory->create(cv_bound, cv_dim*2, "metad:GaussianHill:cv_bound");
            i += 2;
        } else if (strcmp(arg[i], "CAL") == 0){
          ERR_COND(strcmp(arg[i+1], "NAME") != 0, "Error: CAL requires NAME keyword.");
          std::string cal_name = arg[i+2];
          std::string type = arg[i+3];
          LOG("Dueling with %s",cal_name.c_str());
          i += 3;
          cal_registry[cal_name] = MetaD_zqc::CVFactory::create(type, lmp, this, narg, arg, i, f_check);
          // if (strcmp(arg[i], "DISTANCE") == 0) {
          //     DEBUG_LOG("In DISTANCE settings");
          //     // DISTANCE 1 2 -> cv_values: 1-2 距离
          //     ERR_COND(i + 2 >= narg, "Error: DISTANCE command requires 2 atom IDs.");
          //     int id1   = utils::inumeric(FLERR, arg[i+1], false, lmp);
          //     int id2   = utils::inumeric(FLERR, arg[i+2], false, lmp);
          //     cal_registry[cal_name] = new MetaD_zqc::Distance(lmp, id1-1, id2-1, f_check);
          //     // DEBUG_LOG("debug: %d %d", id1, id2);
          //     i += 3;
          // } else if (strcmp(arg[i], "STEINH") == 0) {
          //     DEBUG_LOG("In STEINH settings");
          //     MetaD_zqc::SteinhardtRequest req;
          //     req.cal_name = cal_name;
          //     // 原子环境分析-初始设置
          //     // Usage: STEINH <Q/L> <4/6/8/12> <group>
          //     ERR_COND(i + 3 >= narg, "Error: STEINH command requires \"STEINH <Q/L> <4/6/8/12> <group> \".");
          //     req.Q_type_str = arg[i+1];
          //     req.Q_num   = utils::inumeric(FLERR, arg[i+2], false, lmp);
          //     req.group_name = arg[i+3];
          //     req.group_id = lmp->group->find(req.group_name);
          //     ERR_COND(req.group_id == -1, "Error: Steinhardt group name %s not found.", req.group_name);
          //     //参数有效性
          //     ERR_COND((req.Q_num != 3 && req.Q_num != 4 && req.Q_num != 6 && req.Q_num != 8 && req.Q_num != 12),"Error: Steinhardt order L must be 3, 4, 6, 8, or 12.");
          //     ERR_COND((strcmp(req.Q_type_str, "Q") != 0 && strcmp(req.Q_type_str, "L") != 0), "Error: Steinhardt type must be 'Q' (local) or 'L' (global).");
          //     // 进阶设置
          //     // default values
          //     req.cutoff_r = 4.0;
          //     req.cutoff_Natoms = 12;
          //     req.d_block_size = 128;
          //     int iarg=4 + i;
          //     while (iarg < narg) {
          //         if (strcmp(arg[iarg], "cutoff_r") == 0) {
          //             ERR_COND((iarg + 1 >= narg) ,"Error: \'cutoff_r\' keyword requires a value");
          //             req.cutoff_r = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
          //             iarg += 2;
          //         }
          //         else if (strcmp(arg[iarg], "cutoff_Natoms") == 0) {
          //             ERR_COND((iarg + 1 >= narg), "Error: \'cutoff_Natoms\' keyword requires an integer");
          //             req.cutoff_Natoms = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
          //             iarg += 2;
          //         }
          //         else if (strcmp(arg[iarg], "d_block_size") == 0) {
          //             ERR_COND((iarg + 1 >= narg), "Error: \'d_block_size\' keyword requires an integer");
          //             req.d_block_size = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
          //             ERR_COND(req.d_block_size <= 0, "Error: \'d_block_size\' must be > 0");
          //             iarg += 2;
          //         }
          //         else {
          //           break;
          //         }
          //     }
          //     LOG("Logging: set STEINH as Q_type_str=%s Q_num=%d group_name=%s cutoff_r=%f cutoff_Natoms=%d d_block_size=%d.",
          //                       req.Q_type_str, req.Q_num, req.group_name, req.cutoff_r, req.cutoff_Natoms, req.d_block_size);
          //     NeighRequest *full_request;
          //     full_request = neighbor->add_request(this, NeighConst::REQ_FULL);
          //     full_request->set_id(2);
          //     steinh_requests.push_back(req);
          //     // // 创建 CV 对象
          //     // TODO: 需要处理相同envs的合并问题
          //     MetaD_zqc::Steinhardt_env *temp_env = MetaD_zqc::Steinhardt_env::get_or_create(lmp, 
          //                           f_check, this, req.group_id, req.cutoff_r, req.cutoff_Natoms);
          //     DEBUG_LOG("Steinhardt_env is %p", temp_env);
          //     std::string env_setNum = temp_env->get_env_key();
          //     cal_registry[cal_name]= MetaD_zqc::create_steinhardt_cv(lmp, this, f_check, 
          //                           env_setNum, req.group_id, req.Q_num, temp_env, req.Q_type_str,
          //                           req.cutoff_r, req.cutoff_Natoms, req.d_block_size);
          //     i = iarg;
          // }
      } else if (strcmp(arg[i], "SYMBOL") == 0) {
            // SYMBOL v1 Q6.AVE -> SYMBOL <symbol_name> <cv_method>
            ERR_COND(i + 2 >= narg, "Error: DIM command requires 2 arguments: SYMBOL <symbol_name> <cv_method>.");
            // "Q4.mean"
            DEBUG_LOG("168L:%s", arg[i+2]);
            std::string target = arg[i+2];
            std::string target_name;
            std::string target_func;
            size_t dot_pos = target.find(".");
            if (dot_pos != std::string::npos) {
                target_name = target.substr(0, dot_pos);
                target_func = target.substr(dot_pos + 1);
            } else {
                target_name = target;
                target_func = "value"; 
            }
            DEBUG_LOG("VAL=%s, METHODS=%s",target_name.c_str(),target_func.c_str());
            // add_symbol(const std::string& name, CV* ptr, const std::string& func_name);
            cv_configs->add_symbol(arg[i+1], cal_registry[target_name], target_func.c_str());
            DEBUG_LOG("1");
            i += 3;
      } else if (strcmp(arg[i], "DIM") == 0) {
            // DIM 1 0 40 1600 (v1+v2)/2 -> DIM index, lower_bound, upper_bound, bins, cv_expr
            ERR_COND(i + 5 >= narg, "Error: DIM command requires 5 arguments: index, lower, upper, bins, cv_expr.");
            int dim_idx = utils::inumeric(FLERR, arg[i+1], false, lmp) - 1; // 0-based
            ERR_COND(dim_idx >= cv_dim, "Error: DIM index out of range or unsupported dimension.");
            cv_bound[dim_idx * 2] = utils::numeric(FLERR, arg[i+2], false, lmp); // lower
            cv_bound[dim_idx * 2 + 1] = utils::numeric(FLERR, arg[i+3], false, lmp); // upper
            int nbin_num = utils::inumeric(FLERR, arg[i+4], false, lmp); // 将 bins 数量赋给 nbin_num (仅用于单维度)
            nbin[dim_idx] = nbin_num;
            LOG("Logging: cv_compute=%d bound set at [%g,%g], total grid is %d.",dim_idx+1, cv_bound[dim_idx * 2], cv_bound[dim_idx * 2 +1], nbin_num);
            LOG("Logging: cv_compute=%d bound set expr as %s", dim_idx+1, arg[i+5]);
            // reg_expression(int dim_idx, const std::string& expr_str);
            cv_configs->reg_expression(dim_idx, arg[i+5]);
            i += 6;
        } else if (strcmp(arg[i], "Gaussian_Hill_type") == 0) {
            ERR_COND(i + 1 >= narg, "Error: Gaussian_Hill_type command requires 1 argument: 0(均匀网格) or 1(稀疏网格) or 2(KDTree).");
            Gaussian_Hill_type = (utils::inumeric(FLERR, arg[i+1], false, lmp) != 0);
            i += 2;
        } else if (strcmp(arg[i], "METAD_RESTART") == 0) {
            ERR_COND(i + 1 >= narg, "Error: METAD_RESTART command requires 1 argument: 0 or 1.");
            continue_from_file = (utils::inumeric(FLERR, arg[i+1], false, lmp) != 0);
            i += 2;
        } else if (strcmp(arg[i], "WT") == 0) {
            WellT_bool = (utils::inumeric(FLERR, arg[i+1], false, lmp) != 0);
            i += 2;
        } else {
            ERR_COND(1, "Error: Unknown keyword in fix metadynamics command: %s", arg[i]);
            break;
        }
    }
    LOG("init calc set end, start to define Gaussian.");

    // 定义Gaussian
    if (Gaussian_Hill_type==0){
      // p_gaussian = new MetaD_zqc::GH_t0_uniformGrid<cv_dim>(lmp, f_check,
      //                       cv_dim, sigma, height0, biasf,
      //                       WellT_bool, cv_bound, nbin);
      if (cv_dim == 1) {
        p_gaussian = new MetaD_zqc::GH_t0_uniformGrid<1>(lmp, f_check,
                            cv_dim, sigma, height0, biasf,
                            continue_from_file, WellT_bool, cv_bound, nbin);
      } else if (cv_dim == 2) {
        p_gaussian = new MetaD_zqc::GH_t0_uniformGrid<2>(lmp, f_check,
                            cv_dim, sigma, height0, biasf,
                            continue_from_file, WellT_bool, cv_bound, nbin);
      } else if (cv_dim == 3) {
        p_gaussian = new MetaD_zqc::GH_t0_uniformGrid<3>(lmp, f_check,
                            cv_dim, sigma, height0, biasf,
                            continue_from_file, WellT_bool, cv_bound, nbin);
      } else {
        error->all(FLERR, "Only 1D-3D are supported for optimized grid.");
      }
    } else if (Gaussian_Hill_type==1){
      if (cv_dim == 1) {
        p_gaussian = new MetaD_zqc::GH_t1_sparseHash<1>(lmp, f_check,
                            cv_dim, sigma, height0, biasf,
                            continue_from_file, WellT_bool, cv_bound, nbin);
      } else if (cv_dim == 2) {
        p_gaussian = new MetaD_zqc::GH_t1_sparseHash<2>(lmp, f_check,
                            cv_dim, sigma, height0, biasf,
                            continue_from_file, WellT_bool, cv_bound, nbin);
      } else if (cv_dim == 3) {
        p_gaussian = new MetaD_zqc::GH_t1_sparseHash<3>(lmp, f_check,
                            cv_dim, sigma, height0, biasf,
                            continue_from_file, WellT_bool, cv_bound, nbin);
    } else if (Gaussian_Hill_type==2){
        p_gaussian = new MetaD_zqc::GH_t0_uniformGrid<1>(lmp, f_check,
                            cv_dim, sigma, height0, biasf,
                            continue_from_file, WellT_bool, cv_bound, nbin);
    }
    }
    // p_gaussian->dim = cv_dim;
    // p_gaussian->sigma = sigma;
    // p_gaussian->height0 = height0;
    // p_gaussian->biasf = biasf;
    // p_gaussian->KB = lmp->force->boltz;
    // p_gaussian->WellT_bool = WellT_bool;
    // p_gaussian->cv_bound = cv_bound;
    
    
    // 输出文件
    first_run = true;
    LOG("Fix init end.");

    // // 设置执行时机
    // force_integrate = 1;
    // vflag = 1;
    // extscalar = 0;
    // extvector = 0;
}

FixMetadynamics::~FixMetadynamics() {
  memory->destroy(cv_values);
  memory->destroy(cv_history);
  memory->destroy(dVdcvs);
  for (auto const& pair : cal_registry) {
      delete pair.second; // 这里会调用 CV 及其派生类的析构函数
  }
  cal_registry.clear(); // 清空 map 容器
  // 2. 释放 cv_configs
  if (cv_configs) {
      delete cv_configs;
      cv_configs = nullptr;
  }
  // memory->destroy(cv_bound);
  // memory->destroy(nbin);
  MetaD_zqc::Steinhardt_env::clear_pool();
}

int FixMetadynamics::setmask() {
  return FixConst::POST_FORCE;
}

void FixMetadynamics::init() {
  // if (!atom->tag) error->all(FLERR, "Requires atom style with per-atom positions");
  if (first_run) {
    p_gaussian->init_set_mode();
    memory->create(cv_values, cv_dim, "metad:cv_values");
    memory->create(cv_history, cv_dim, "metad:cv_history");
    memory->create(dVdcvs, cv_dim, "metad:dVdcvs"); // 修正：dVdcv 应该是其梯度
    for (int k = 0; k < cv_dim; ++k) {
        cv_values[k] = 0.0;
        cv_history[k] = 0.0;
        dVdcvs[k] = 0.0;
    }
    if (cv_dim == 1){
      // double *cv_values[2]={0,0};
      // cv_values[0] = cv_compute[0]->compute_cv();
      cv_values[0] = 0.0;
      first_run = false;}
    if (cv_dim == 2){
      // double *cv_values[2]={0,0};
      // cv_values[0] = cv_compute[0]->compute_cv();
      // cv_values[1] = cv_compute[1]->compute_cv();
      cv_values[0] = 0.0;
      cv_values[1] = 0.0;
      first_run = false;}
    // 续算
  }
}

void FixMetadynamics::init_list(int id, NeighList *ptr)
{
  if (id == 1)
    listhalf = ptr;
  else if (id == 2)
    listfull = ptr;
}

// 3. 真正偏置力（解析梯度，比数值差分快）
void FixMetadynamics::post_force(int) {
  DEBUG_LOG("post_force start");
  // -----calculate cv_compute and add cv_history-----
  for (auto const& pair : cal_registry) {
    const std::string& name = pair.first;
    MetaD_zqc::CV* obj = pair.second;
    DEBUG_LOG("base_calc of %s is start", pair.first.c_str());
    obj->base_calc(); 
    DEBUG_LOG("base_calc of %s is end", pair.first.c_str());
  }
  cv_configs->compute_total_cv();
  for (int ii=0; ii<cv_dim; ii++){
    cv_values[ii] = cv_configs->compute_dim_cv(ii);
    cv_history[ii] += cv_values[ii];
    DEBUG_LOG("cv_compute[%d] = %g, cv_history[%d] = %g", ii, cv_values[ii], ii, cv_history[ii]);
  }
  // -----if pace, then add_hill-----
  if ((pace!=0)&&(update->ntimestep % pace == 0)) {
    for(int ii=0; ii<cv_dim; ii++){
      // cv_history[ii] = cv_history[ii]/pace;
      cv_history[ii] = cv_values[ii];
    }
    DEBUG_LOG("enter add_hill");
    add_hill(cv_history);
    DEBUG_LOG("coming out from add_hill");
    for(int ii=0; ii<cv_dim; ii++){
      cv_history[ii] = 0.0;
    }
  }
  // calculate grad of grid
  p_gaussian->get_dVdcv(cv_values, dVdcvs);
  // DEBUG_LOG("cv_compute = %g, dVdcv = %.g",cv_values[0], dVdcvs[0]);
  // printf("cv_compute = %g, dVdcv = %.g\n",cv_values[0], dVdcvs[0]);
  for(int ii=0; ii<cv_dim; ii++){
    DEBUG_LOG("dVdcv[%d] = %.g", ii, dVdcvs[ii]);
    // (base_cv[ii]->*cv_biasforce[ii])(dVdcvs[ii]);
    cv_configs->distribute_dim_bias_force(ii, dVdcvs[ii]);
  }
  DEBUG_LOG("post_force_end");
}

void FixMetadynamics::add_hill(double* cv_values){
    p_gaussian->add_hill(cv_values);
}

// 5. 网格梯度（中心差分）
void FixMetadynamics::get_dVdcv(double *cv_values,
                                    double *dVdcvs) {
  p_gaussian->get_dVdcv(cv_values, dVdcvs);
}

// 工厂函数：创建FixZeroForce对象
static Fix *fix_metad(LAMMPS *lmp, int narg, char **arg) {
    return new FixMetadynamics(lmp, narg, arg);
}

// 插件注册函数（必须命名为 lammpsplugin_init）
extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc) {
    lammpsplugin_t plugin;
    lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc)regfunc;

    // 注册Fix类型插件
    // plugin.version = LAMMPS_VERSION;
    plugin.style = "fix";                 // 插件类型为fix
    plugin.name = "metad";            // 插件名称
    plugin.info = "Metad plugin v1.0";
    plugin.author = "ZQC";
    plugin.creator.v2 = (lammpsplugin_factory2 *)fix_metad; // v2工厂函数
    plugin.handle = handle;
    (*register_plugin)(&plugin, lmp);     // 注册插件
}