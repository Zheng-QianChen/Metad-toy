#include <cmath>
#include <cstdio>

#include <cuda_runtime.h>

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
#include "compute.h"

#include "fix_crystallize.h"
#include "compute_MetaDToy.h"
#include "zqc_debug.h"
#include "zqc_DimSet.h"
#include "zqc_gaussian.h"
#include "zqc_mlcvs.h"
#include "zqc_switch_function.h"

using namespace LAMMPS_NS;



FixMetadynamics::FixMetadynamics(LAMMPS *lmp, int narg, char **arg)
                                : Fix(lmp, narg, arg),
                                  pace(100),
                                  cv_dim(1), nbin_num(100),
                                  bias_energy(0.0),
                                  f_before_bias(nullptr),
                                  max_f_before_bias(0),
                                  p_gaussian(nullptr),
                                  cv_values(nullptr),
                                  cv_history(nullptr),
                                  dVdcvs(nullptr),
                                  cv_configs(nullptr){
    neigh_hub.bind(lmp, this);
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
    double *cv_bound = nullptr;
    int *nbin = nullptr;
    int Gaussian_Hill_type = 0;
    double sigma, height0, biasf, KB, current_temp;
    sigma     = 0.05;
    height0   = 0.1;
    biasf     = 10.0;
    KB        = 0.025852;
    current_temp = 300.0;
    int continue_from_file=false;
    int WellT_bool=false;
    // Gaussian_Hill_type={
    // 0: 均匀网格
    // 1: 稀疏网格
    // 2: KDTree
    // }
    // Check dim configuration:
    // bool *has_dim_configured = nullptr;
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

            std::vector<bool> has_dim_configured(cv_dim, false);

            i += 2;
        } else if (strcmp(arg[i], "TEMP") == 0) {
            ERR_COND(i + 1 >= narg, "Error: TEMP command requires 1 argument: target temperature.");
            current_temp = utils::numeric(FLERR, arg[i+1], false, lmp);
            i += 2;
            LOG("Logging: current_temp = %g (用于Well-Tempered公式)", current_temp);
        } else if (strcmp(arg[i], "CAL") == 0){
          ERR_COND(strcmp(arg[i+1], "NAME") != 0, "Error: CAL requires NAME keyword.");
          std::string cal_name = arg[i+2];
          std::string type = arg[i+3];
          LOG("Dueling with %s",cal_name.c_str());
          i += 3;
          if (strcmp(type.c_str(), "SW_FUNC") == 0) {
            if (sw_registry.find(cal_name) != sw_registry.end()) {
              error->all(FLERR, "Error: CAL name '%s' is duplicated.", cal_name.c_str());
            } else {
               LOG("Creating switch function %s", cal_name.c_str());
            }
            // SW_FUNC 也将由 CAL 指定
            sw_registry[cal_name] = MetaD_zqc::SwitchFunction::create(lmp, this, f_check, narg, arg, i);
          } else {
            if (cal_registry.find(cal_name) != cal_registry.end()) {
              error->all(FLERR, "Error: CAL name '%s' is duplicated.", cal_name.c_str());
            } else {
               LOG("Creating switch function %s", cal_name.c_str());
            }
            // 通过工厂方法创建 CV 对象，并注册到 cal_registry 中
            auto new_cv = MetaD_zqc::CVFactory::create(type, lmp, this, f_check, narg, arg, i);
            if (new_cv == nullptr) {
              error->all(FLERR, "Metad-toy Error: Unknown or unregistered object type '%s' specified in CAL '%s'. "
                                "Please register it in CVFactory or check your input script.",
                        type.c_str(), cal_name.c_str());
            }
            cal_registry[cal_name] = new_cv;
          }
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
            ERR_COND(dim_idx > cv_dim, "Error: DIM index out of range or unsupported dimension.");
            cv_bound[dim_idx * 2] = utils::numeric(FLERR, arg[i+2], false, lmp); // lower
            cv_bound[dim_idx * 2 + 1] = utils::numeric(FLERR, arg[i+3], false, lmp); // upper
            int nbin_num = utils::inumeric(FLERR, arg[i+4], false, lmp); // 将 bins 数量赋给 nbin_num (仅用于单维度)
            nbin[dim_idx] = nbin_num;
            LOG("Logging: cv_compute=%d bound set at [%g,%g], total grid is %d.",dim_idx+1, cv_bound[dim_idx * 2], cv_bound[dim_idx * 2 +1], nbin_num);
            LOG("Logging: cv_compute=%d bound set expr as %s", dim_idx+1, arg[i+5]);
            // reg_expression(int dim_idx, const std::string& expr_str);
            cv_configs->reg_expression(dim_idx, arg[i+5]);
            // 【标记该维度已配置，后续检查是否所有维度都配置了表达式
            // has_dim_configured[dim_idx] = true;
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

    // for (int j = 0; j < cv_dim; ++j) {
    //     if (!has_dim_configured[j]) {
    //         error->all(FLERR, "Error: CV_dim is set to %d, but DIM %d is missing in the input script!", cv_dim, j + 1);
    //     }
    // }

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
    // WT 公式用 p_gaussian->current_temp；TEMP 关键字必须写进去
    p_gaussian->current_temp = current_temp;
    // p_gaussian->rehash_thresh = sparse_rehash_thresh;
    
    
    // 输出文件
    first_run = true;
    LOG("Fix init end.");

    // // 设置执行时机
    // force_integrate = 1;
    // vflag = 1;
    // extscalar = 0;
    // extvector = 0;

    // 设置compute_scalar和compute_vector的返回值
    scalar_flag = 1;                 // f_metad -> 偏置势 V_b（见 compute_scalar）
    vector_flag = 1;
    size_vector = cv_dim;            // f_metad[1]..f_metad[cv_dim] -> CV 值
    extscalar = 0;                   // 强度量，不是跨进程求和的广延量
    extvector = 0;

    // NPT/能量账本：偏置力必须进 virial；偏置势可进 PE（fix_modify energy）
    // 默认打开，与 fix plumed 一致；可用 fix_modify <id> energy no / virial no 关闭
    energy_global_flag = 1;
    virial_global_flag = 1;
    thermo_energy = 1;
    thermo_virial = 1;
}

FixMetadynamics::~FixMetadynamics() {
  memory->destroy(cv_values);
  memory->destroy(cv_history);
  memory->destroy(dVdcvs);
  memory->destroy(f_before_bias);
  cv_values = cv_history = dVdcvs = nullptr;
  f_before_bias = nullptr;
  for (auto const& pair : cal_registry) {
      delete pair.second; // 这里会调用 CV 及其派生类的析构函数
  }
  cal_registry.clear(); // 清空 map 容器
  // 2. 释放 cv_configs
  if (cv_configs) {
      delete cv_configs;
      cv_configs = nullptr;
  }
  // 3. 释放高斯网格（拥有 cv_bound/nbin 等）
  if (p_gaussian) {
      delete p_gaussian;
      p_gaussian = nullptr;
  }
  for (auto const& pair : sw_registry) {
      delete pair.second;
  }
  sw_registry.clear();
  if (rec_file) {
      fclose(rec_file);
      rec_file = nullptr;
  }
  if (f_check) {
      fclose(f_check);
      f_check = nullptr;
  }
}

int FixMetadynamics::setmask() {
  return FixConst::POST_FORCE;
}

void FixMetadynamics::init() {
  // if (!atom->tag) error->all(FLERR, "Requires atom style with per-atom positions");
  // Every System init (run / write_restart): LAMMPS drops NeighRequest; re-add then
  // drop cached ensure state. Do NOT Neighbor::build() / ago here (corrupts MEAM).
  neigh_hub.rerequest_all();
  neigh_hub.invalidate_all();

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

    // 确认通讯
    comm_forward=get_comm_forward_bytes();
    comm_reverse=get_comm_reverse_bytes();
  }
}

void FixMetadynamics::init_list(int id, NeighList *ptr)
{
  neigh_hub.on_init_list(id, ptr);
}

// 3. 真正偏置力（解析梯度，比数值差分快）
void FixMetadynamics::post_force(int vflag) {
  DEBUG_LOG("post_force start");

  // Occasional/perpetual Fix lists: ensure before any CV touches numneigh.
  neigh_hub.ensure_all();

  // 为 virial 记账准备：只累加“本 fix 新加上的力”
  v_init(vflag);
  double **f = atom->f;
  double **x = atom->x;
  imageint *image = atom->image;
  int nlocal = atom->nlocal;
  if (nlocal > max_f_before_bias) {
    memory->destroy(f_before_bias);
    max_f_before_bias = atom->nmax;
    memory->create(f_before_bias, max_f_before_bias, 3, "metad:f_before_bias");
  }
  for (int i = 0; i < nlocal; i++) {
    f_before_bias[i][0] = f[i][0];
    f_before_bias[i][1] = f[i][1];
    f_before_bias[i][2] = f[i][2];
  }

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
  // 偏置势（与梯度同一套插值族）；仅 rank0 算再广播，与 get_dVdcv 一致
  if (comm->me == 0) {
    bias_energy = p_gaussian->get_bias_energy(cv_values);
  }
  MPI_Bcast(&bias_energy, 1, MPI_DOUBLE, 0, world);

  // DEBUG_LOG("cv_compute = %g, dVdcv = %.g",cv_values[0], dVdcvs[0]);
  // printf("cv_compute = %g, dVdcv = %.g\n",cv_values[0], dVdcvs[0]);
  for(int ii=0; ii<cv_dim; ii++){
    DEBUG_LOG("dVdcv[%d] = %.g", ii, dVdcvs[ii]);
    // (base_cv[ii]->*cv_biasforce[ii])(dVdcvs[ii]);
    cv_configs->distribute_dim_bias_force(ii, dVdcvs[ii]);
  }

  // 用 Δf = f_after - f_before 做 r⊗F 维里；自动覆盖 STEINH / WEIGHT_CHEM / DISTANCE 等全部 CV
  if (vflag && thermo_virial) {
    double v[6], unwrap[3];
    for (int i = 0; i < nlocal; i++) {
      double dfx = f[i][0] - f_before_bias[i][0];
      double dfy = f[i][1] - f_before_bias[i][1];
      double dfz = f[i][2] - f_before_bias[i][2];
      if (dfx == 0.0 && dfy == 0.0 && dfz == 0.0) continue;
      domain->unmap(x[i], image[i], unwrap);
      v[0] = dfx * unwrap[0];
      v[1] = dfy * unwrap[1];
      v[2] = dfz * unwrap[2];
      v[3] = dfx * unwrap[1];
      v[4] = dfx * unwrap[2];
      v[5] = dfy * unwrap[2];
      v_tally(i, v);
    }
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

double FixMetadynamics::compute_scalar() {
    // 与 fix plumed 一致：scalar = 偏置势，供 thermo PE / fix_modify energy
    // CV 请用 f_metad[1] ... f_metad[cv_dim]
    return bias_energy;
}

double FixMetadynamics::compute_vector(int n) {
    // n 是 0-based 索引 (LAMMPS 内部约定)
    if (n < 0 || n >= cv_dim) return 0.0;
    return cv_values[n];
}

int FixMetadynamics::get_comm_forward_bytes() {
    int total_size = 0;
    for (auto const& pair : cal_registry) {
        MetaD_zqc::CV* obj = pair.second;
        // LocalQL 等 CV 可能继承了非零的 get_comm_forward_bytes，
        // 但 need_forward_comm()==false，不能计入 Fix::comm_forward。
        if (obj->need_forward_comm() && obj->get_comm_forward_bytes()) {
            total_size += obj->get_comm_forward_bytes();
        }
    }
    LOG("communicate: total_size=%d",total_size);
    return total_size;
}

int FixMetadynamics::get_comm_reverse_bytes() {
    int total_size = 0;
    for (auto const& pair : cal_registry) {
        MetaD_zqc::CV* obj = pair.second;
        if (obj->need_reverse_comm() && obj->get_comm_reverse_bytes()) {
            total_size += obj->get_comm_reverse_bytes();
        }
    }
    return total_size;
}

int FixMetadynamics::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/) {
    int slot_offset = 0; // 偏移量计数器，单位是“槽位(Slot)”，1个Slot = 1个double
    int cycle_offset = comm_forward;
    DEBUG_LOG("Pack forward comm start");

    for (auto const& pair : cal_registry) {
        MetaD_zqc::CV* obj = pair.second;
        if (obj->need_forward_comm()) {
            // 每个 obj 传入 ubuf 总线，并在它专属的 slot_offset 位置开始平铺写入
            // 子 obj 内部写入了多少个 slot，就返回多少，累加给 slot_offset
            slot_offset += obj->pack_comm_forward_ubuf(n, list, buf, slot_offset, comm_forward);
        }
    }
    
    int expected_total = n * this->comm_forward; // 官方大管家认为你应该打包的总 double 数
    DEBUG_LOG("Rank:%d,Pack forward comm: n=%d, comm_forward=%d, total packed=%d", lmp->comm->me,n, this->comm_forward, n*slot_offset);

    return slot_offset*n;
}

void FixMetadynamics::unpack_forward_comm(int n, int first, double *buf) {
    int slot_offset = 0;
    int cycle_offset = comm_forward;
    DEBUG_LOG("UNPack forward comm start");

    for (auto const& pair : cal_registry) {
        MetaD_zqc::CV* obj = pair.second;
        if (obj->need_forward_comm()) {
            // 一模一样地把 ubuf 总线和起始槽位偏移量交还给子 CV 解包
            obj->unpack_comm_forward_ubuf(n, first, buf, slot_offset, comm_forward);
            
            // 每一个 CV 占用的总槽位数 = 原子数 n * 单个原子需要的槽位数
            slot_offset +=  obj->get_comm_forward_bytes();
        }
    }
    int expected_total = n * this->comm_forward; // 基类预期解包的总 double 数
    DEBUG_LOG("Rank:%d,Unpack forward comm: n=%d, comm_forward=%d, total unpacked=%d", lmp->comm->me, n, this->comm_forward, slot_offset*n);
}

int FixMetadynamics::pack_reverse_comm(int n, int first, double *buf) {
    // 这里的 n 是本次通信的 ghost 原子总数
    // 注意：LAMMPS reverse buffer 步长是 Fix::comm_reverse，不是 comm_forward。
    // 之前误传 comm_forward 会导致 LocalQL(comm_forward 继承非零、comm_reverse=3)
    // 在 np>=2 时按过大步长写越界，最终表现为 lost atoms。
    int slot_offset = 0;
    for (auto const& pair : cal_registry) {
        MetaD_zqc::CV* obj = pair.second;
        if (obj->need_reverse_comm()) {
            // 将 ghost 原子计算的梯度 pack 进 buf
            // pack_comm_reverse_ubuf 返回的是每个原子写入的 double 个数
            int n_doubles = obj->pack_comm_reverse_ubuf(n, first, buf, slot_offset, comm_reverse);
            slot_offset += n_doubles;
        }
    }
    return slot_offset * n;
}

void FixMetadynamics::unpack_reverse_comm(int n, int *list, double *buf) {
    // 这里的 n 是本次通信接收的 ghost 原子数
    // list 是 ghost 到本地原子的映射索引
    int slot_offset = 0;
    for (auto const& pair : cal_registry) {
        MetaD_zqc::CV* obj = pair.second;
        if (obj->need_reverse_comm()) {
            // 将 buf 中的梯度累加到对应本地原子的 dcvdx 上
            obj->unpack_comm_reverse_ubuf(n, list, buf, slot_offset, comm_reverse);
            slot_offset += obj->get_comm_reverse_bytes();
        }
    }
}

void * FixMetadynamics::extract(const char *key, int &dim){
  std::string keystr(key);
  // 处理每原子数据请求
  if (keystr.rfind("colvar_peratom:", 0) == 0) {
    std::string full_query = keystr.substr(15); // 拿到 "Q4.stein_q"
    
    // 🔍 寻找点号位置进行切片
    size_t dot_pos = full_query.find('.');
    if (dot_pos == std::string::npos) return nullptr;
    
    std::string cv_instance_name = full_query.substr(0, dot_pos); // "Q4"
    std::string prop_name = full_query.substr(dot_pos + 1);       // "stein_q"
    
    // 从你 Fix 的自注册 Map 容器中路由找到 Q4 实例
    if (cal_registry.find(cv_instance_name) != cal_registry.end()) {
      dim = 1;
      // 调用该 CV 内部由属性名驱动的公共接口（或者让各个 CV 自己解析属性）
      return cal_registry[cv_instance_name]->get_peratom_ptr(prop_name); 
    }
  }
  return nullptr;
}

MetaD_zqc::SwitchFunction* FixMetadynamics::get_switching_function(const std::string& name) const {
  auto it = sw_registry.find(name);
  if (it != sw_registry.end()) {
    return it->second; 
  }
  return nullptr;
}

// 工厂函数：创建FixZeroForce对象
static Fix *fix_metad(LAMMPS *lmp, int narg, char **arg) {
    return new FixMetadynamics(lmp, narg, arg);
}
static Compute *compute_MetaD_toy(LAMMPS *lmp, int narg, char **arg, FILE* f_check) {
    return new ComputeMetaDToy(lmp, narg, arg, f_check);
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

    // 注册compute类型的插件
    plugin.style = "compute";             // 🚨 切换插件类型为 compute
    plugin.name = "metad/atom";           // 🚨 设定样式名（用户在 in 脚本里调用的样式）
    plugin.info = "Metadynamics CV Diagnostic Bus Probe v1.0";
    plugin.author = "ZQC";
    plugin.creator.v2 = (lammpsplugin_factory2 *)compute_MetaD_toy; // 🚨 绑定 Compute 的工厂函数
    plugin.handle = handle;               // 共享相同的句柄
    (*register_plugin)(&plugin, lmp);     // 再次调用注册，将其挂载到 LAMMPS 核心总线上
}