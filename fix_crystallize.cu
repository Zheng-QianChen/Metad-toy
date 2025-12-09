

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

#include <cuda_runtime.h>
using namespace LAMMPS_NS;


FixMetadynamics::FixMetadynamics(LAMMPS *lmp, int narg, char **arg)
    : Fix(lmp, narg, arg),
      sigma(0.05), height0(0.1), biasf(10.0), kBT(0.025852), pace(100),
      cv_dim(1), nbin_num(100), continue_from_file(false), WellT_bool(false),
      bias_grid(nullptr), f_hills(nullptr)
{
    if (comm->me==0) {
        f_check = fopen("a.txt","w");
    }
    // Fix 构造函数的基本参数检查： fix ID group style args...
    // arg[0]=fix, arg[1]=ID, arg[2]=group, arg[3]=style (metad)
    // if (narg < 4) error->all(FLERR, "Fix metadynamics requires at least 1 argument after style name.");
    LOG("There are %d args", narg);
    // --- 核心参数解析：循环读取关键词/数值对 ---
    int i = 3; // 从第 4 个参数开始，即 style 名之后
    int cv_count = 0; // 用于检查 DISTANCE/ANGLE 等 cv_values 的定义数量
    KB = lmp->force->boltz;
    while (i < narg) {
        LOG("Im in arg loop");
        if (strcmp(arg[i], "GAUSSIAN") == 0) {
            ERR_COND(i + 3 >= narg, "Error: GAUSSIAN command requires 3 arguments: sigma, height, biasf.");
            sigma   = utils::numeric(FLERR, arg[i+1], false, lmp);
            height0 = utils::numeric(FLERR, arg[i+2], false, lmp);
            biasf   = utils::numeric(FLERR, arg[i+3], false, lmp);
            i += 4;

            double R_SI = 8.314462618;
            height0 = KB*height0/R_SI;

            LOG("Logging: set GAUSSIAN sigma, height, biasf: %g %g %g.", sigma, height0, biasf);
            LOG("Logging: attention, we use Joules/moles as the height's units, so if the lammps settings is not real, there will be a units transform.\n\
             If you set CV's bounds in physicals, this units will follows your lammps units settings.");
        } else if (strcmp(arg[i], "PACE") == 0) {
            ERR_COND(i + 1 >= narg, "Error: PACE command requires 1 argument: integer timesteps.");
            pace = utils::inumeric(FLERR, arg[i+1], false, lmp);
            i += 2;
            LOG("Logging: pace = %d.",pace);
        } else if (strcmp(arg[i], "CV_dim") == 0) {
            DEBUG_LOG("In CV_dim settings");
            ERR_COND(i + 1 >= narg, "Error: PACE command requires 1 argument: integer timesteps.");
            cv_dim = utils::inumeric(FLERR, arg[i+1], false, lmp);
            memory->create(nbin, cv_dim, "metad:nbin_size");
            memory->create(cvspace_loc, cv_dim, "metad:cvspace_loc");
            // memory->create(cv, cv_dim, "metad:cv");
            memory->create(cv_bound, cv_dim*2, "metad:cv_bound");
            i += 2;
        } else if (strcmp(arg[i], "DISTANCE") == 0) {
            DEBUG_LOG("In DISTANCE settings");
            // DISTANCE 1 2 -> cv_values: 1-2 距离
            ERR_COND(i + 2 >= narg, "Error: DISTANCE command requires 2 atom IDs.");
            int id1   = utils::inumeric(FLERR, arg[i+1], false, lmp);
            int id2   = utils::inumeric(FLERR, arg[i+2], false, lmp);
            cv.push_back(new MetaD_zqc::Distance(lmp, id1-1, id2-1, f_check));
            cv_count++;
            ERR_COND(cv_count>cv_dim, "Error: cv_count > cv_dim.");
            DEBUG_LOG("debug: %d %d", id1, id2);
            DEBUG_RUN(cv[0]->summary(f_check));
            i += 3;
        } else if (strcmp(arg[i], "STEINH") == 0) {
            DEBUG_LOG("In STEINH settings");
            // 原子环境分析-初始设置
            // Usage: STEINH <Q/L> <4/6/8/12> <group>
            ERR_COND(i + 3 >= narg, "Error: STEINH command requires \"STEINH <Q/L> <4/6/8/12> <group> \".");
            char *Q_type_str   = arg[i+1];
            int Q_num   = utils::inumeric(FLERR, arg[i+2], false, lmp);
            char *group_name = arg[i+3];
            int group_id = lmp->group->find(group_name);
            ERR_COND(group_id == -1, "Error: Steinhardt group name %s not found.", group_name);
            //参数有效性
            ERR_COND((Q_num != 4 && Q_num != 6 && Q_num != 8 && Q_num != 12),"Error: Steinhardt order L must be 4, 6, 8, or 12.");
            ERR_COND((strcmp(Q_type_str, "Q") != 0 && strcmp(Q_type_str, "L") != 0), "Error: Steinhardt type must be 'Q' (local) or 'L' (global).");
            // 进阶设置
            // default values
            double cutoff_r = 4.0;
            int cutoff_Natoms = 12;
            int d_block_size = 128;
            int iarg=4 + i;
            while (iarg < narg) {
                if (strcmp(arg[iarg], "cutoff_r") == 0) {
                    ERR_COND((iarg + 1 >= narg) ,"Error: \'cutoff_r\' keyword requires a value");
                    cutoff_r = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
                    iarg += 2;
                }
                else if (strcmp(arg[iarg], "cutoff_Natoms") == 0) {
                    ERR_COND((iarg + 1 >= narg), "Error: \'cutoff_Natoms\' keyword requires an integer");
                    cutoff_Natoms = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
                    iarg += 2;
                }
                else if (strcmp(arg[iarg], "d_block_size") == 0) {
                    ERR_COND((iarg + 1 >= narg), "Error: \'d_block_size\' keyword requires an integer");
                    d_block_size = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
                    ERR_COND(d_block_size <= 0, "Error: \'d_block_size\' must be > 0");
                    iarg += 2;
                }
                else {
                  break;
                }
            }
            LOG("Logging: set STEINH as Q_type_str=%s Q_num=%d group_name=%s cutoff_r=%f cutoff_Natoms=%d d_block_size=%d.",
                              Q_type_str, Q_num, group_name, cutoff_r, cutoff_Natoms, d_block_size);
            NeighRequest *full_request;
            full_request = neighbor->add_request(this, NeighConst::REQ_FULL);
            full_request->set_id(2);
            // 创建 CV 对象
            cv.push_back(MetaD_zqc::create_steinhardt_cv(lmp, this, f_check, group_id, Q_num, Q_type_str, cutoff_r, cutoff_Natoms, d_block_size)); 
            cv_count++;
            ERR_COND(cv_count > cv_dim, "Error: cv_count > cv_dim.");
            i = iarg;
        } else if (strcmp(arg[i], "DIM") == 0) {
            // DIM 1 0 40 1600 -> DIM index, lower_bound, upper_bound, bins
            ERR_COND(i + 4 >= narg, "Error: DIM command requires 4 arguments: index, lower, upper, bins.");
            int dim_index = utils::inumeric(FLERR, arg[i+1], false, lmp) - 1; // 0-based
            ERR_COND(dim_index >= cv_dim, "Error: DIM index out of range or unsupported dimension.");
            cv_bound[dim_index * 2] = utils::numeric(FLERR, arg[i+2], false, lmp); // lower
            cv_bound[dim_index * 2 + 1] = utils::numeric(FLERR, arg[i+3], false, lmp); // upper
            int nbin_num = utils::inumeric(FLERR, arg[i+4], false, lmp); // 将 bins 数量赋给 nbin_num (仅用于单维度)
            nbin[dim_index] = nbin_num;
            LOG("Logging: cv=%d bound set at [%g,%g], total grid is %d.",dim_index+1, cv_bound[dim_index * 2], cv_bound[dim_index * 2 +1], nbin_num);
            i += 5;
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
    
    // 输出文件
    first_run = true;
    DEBUG_LOG("Fix init end.");
    
    // // 设置执行时机
    // force_integrate = 1;
    // vflag = 1;
    // extscalar = 0;
    // extvector = 0;
}

FixMetadynamics::~FixMetadynamics() {
  memory->destroy(bias_grid);
  memory->destroy(nbin);
  memory->destroy(cvspace_loc);
  memory->destroy(cv_bound);
  memory->destroy(cv_values);
  memory->destroy(cv_history);
  memory->destroy(dVdcvs);
  if (comm->me==0 && f_hills) {
    fclose(f_hills);
    fclose(f_check);
  }
}

int FixMetadynamics::setmask() {
  return FixConst::POST_FORCE;
}

void FixMetadynamics::init() {
//   if (!atom->tag) error->all(FLERR, "Requires atom style with per-atom positions");
  if (first_run) {
    // 配置邻居
    
    // 分配网格
    grid_size = 1;
    for (int k = 0; k < cv_dim; ++k) {
        grid_size *= nbin[k];
    }
    memory->create(bias_grid, grid_size, "metad:bias_grid");
    for (bigint k = 0; k < grid_size; ++k) {
        bias_grid[k] = 0.0;
    }
    memory->create(dcv, cv_dim, "metad:dcv");
    for (int k = 0; k < cv_dim; ++k) {
        dcv[k] = (cv_bound[2*k+1]-cv_bound[2*k])/nbin[k];
        LOG("dcv[k] = %g", dcv[k]);
    }
    DEBUG_LOG("Fix init end.");

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
      // cv_values[0] = cv[0]->compute_cv();
      cv_values[0] = 0.0;
      first_run = false;}
    if (cv_dim == 2){
      // double *cv_values[2]={0,0};
      // cv_values[0] = cv[0]->compute_cv();
      // cv_values[1] = cv[1]->compute_cv();
      cv_values[0] = 0.0;
      cv_values[1] = 0.0;
      first_run = false;}
    // 续算
    if (continue_from_file){
      if (comm->me == 0) {
          // 尝试打开 HILLS 文件进行读取
          FILE *f_read = fopen("HILLS", "r");
            if (f_read) {
                // 1. 读取 HILLS 文件并重建 bias_grid
                char line[1024];
                long long step;
                double h, s;
                long long current_timestep = 0;
                // 跳过头文件
                DEBUG_LOG("restart hills");
                if (fgets(line, sizeof(line), f_read) == NULL) {
                    // 如果文件为空，则视为新文件
                } else {
                    // 循环读取每一行数据
                    while (fscanf(f_read, "%lld %lf %lf %lf %lf\n", 
                                  &step, &cv_values[0], &cv_values[1], &h, &s)==5) {
                      fprintf(f_check, "%lld %lf %lf %lf %lf\n",step, cv_values[0], cv_values[1],h,s);
                        get_cvspace_loc(cv_values, cvspace_loc);
                        add_hill(cv_values, h);
                        // current_timestep = step;
                    }
                    fprintf(f_check, "%lld %lf %lf %lf %lf\n", step, cv_values[0], cv_values[1], h, s);
                    fflush(f_check);
                }
                fclose(f_read);
                DEBUG_LOG("restart hills end");
                // 2. 重新打开 HILLS 文件，使用 "a" (追加) 模式
                f_hills = fopen("HILLS", "a");
                // if (f_hills == NULL) {
                //     error->all(FLERR, "Cannot open HILLS file for appending.");
                // }
                // 3. 将 current_timestep 广播给所有进程 (如果需要)
                MPI_Bcast(&current_timestep, 1, MPI_LONG_LONG, 0, world); // 如果需要同步 step
          } else {
              // --- 未找到 HILLS 文件，创建新文件 ---
              f_hills = fopen("HILLS", "w");
              // if (f_hills == NULL) {
              //     error->all(FLERR, "Cannot open HILLS file for writing.");
              // }
              if (cv_dim==2){fprintf(f_hills, "# step cv1 cv2 height sigma\n");}
              if (cv_dim==1){fprintf(f_hills, "# step cv_values height sigma\n");}
          }
      } // end if comm->me == 0
    }
    else{
      f_hills = fopen("HILLS", "w");
      if (cv_dim==2){fprintf(f_hills, "# step cv1 cv2 height sigma\n");}
      if (cv_dim==1){fprintf(f_hills, "# step cv_values height sigma\n");}
    }
    MPI_Bcast(&bias_grid[0], grid_size, MPI_DOUBLE, 0, world);
  }

}

/* helper: 计算高斯 */
static inline double gauss(int dim, double* dx, double s) {
  if (dim==1){
    return exp(-0.5*(dx[0]*dx[0])/(s*s));
  } else if (dim==2){
    return exp(-0.5*(dx[0]*dx[0]+dx[1]*dx[1])/(s*s));
  } else {
    double paw=0.0;
    for (int i=0; i<dim; i++){
      paw += dx[i]*dx[i];
    }
    return exp(-0.5*(paw)/(s*s));
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
  // -----calculate cv and add cv_history-----
  for(int ii=0; ii<cv_dim; ii++){
    cv_values[ii] = cv[ii]->compute_cv();
    cv_history[ii] += cv_values[ii];
  }
  // -----if pace, then add_hill-----
  if ((update->ntimestep % pace == 0)&&(pace!=0)) {
    for(int ii=0; ii<cv_dim; ii++){
      // cv_history[ii] = cv_history[ii]/pace;
      cv_history[ii] = cv_values[ii];
    }
    get_cvspace_loc(cv_history, cvspace_loc);
    double Vbias = 0.0;
    if (WellT_bool){
      Vbias = get_total_bias(cvspace_loc);
    }
    double current_temp;
    current_temp = 300.0;
    double w = height0 * exp(-(Vbias)/(current_temp*KB*(biasf-1.0)));
    if (comm->me==0) {
      fprintf(f_hills, "%ld", update->ntimestep);
      for (int ii = 0; ii < cv_dim; ii++) {
          fprintf(f_hills, " %.16g", cv_values[ii]);
      }
      fprintf(f_hills, " %.16g %.16g\n", w, sigma);
      fflush(f_hills);
    }
    add_hill(cv_values, w);
    MPI_Bcast(&bias_grid[0], grid_size, MPI_DOUBLE, 0, world);
    for(int ii=0; ii<cv_dim; ii++){
      cv_history[ii] = 0.0;
    }
  }
  // calculate grad of grid
  get_cvspace_loc(cv_values, cvspace_loc);
  grid_gradient(cvspace_loc, dVdcvs);
  DEBUG_LOG("cv_value = %g, dVdcv = %.g",cv_values[0], dVdcvs[0]);
  for(int ii=0; ii<cv_dim; ii++){
    DEBUG_LOG("dVdcv[%d] = %.g", ii, dVdcvs[ii]);
    cv[ii]->bias_force(dVdcvs[ii]);
  }
  // post_force_r(cv_values[0], dVdcvs[0]);
  DEBUG_LOG("post_force_end");
}

void FixMetadynamics::get_cvspace_loc(double* cv_values, int* cvspace_loc){
  for(int ii=0; ii<cv_dim; ii++){
    // DEBUG_LOG("cv_values[%d] = %g, cvbound=[%g,%g], grid=%d",ii,cv_values[ii],cv_bound[ii*2+1],cv_bound[ii*2], nbin[ii]);
    cvspace_loc[ii] = static_cast<int>(((cv_values[ii]-cv_bound[ii*2])/(cv_bound[ii*2+1]-cv_bound[ii*2]))*nbin[ii]);
    cvspace_loc[ii] = (cvspace_loc[ii]<1)?1:(cvspace_loc[ii]>=nbin[ii]-1)?nbin[ii]-2:cvspace_loc[ii];
    // DEBUG_LOG("cvspace_loc[%d] = %d",ii,cvspace_loc[ii]);
  }
}

double FixMetadynamics::get_total_bias(int* cvspace_loc){
  if (cv_dim==1){
    return bias_grid[cvspace_loc[0]];
  } else if (cv_dim==2){
    return bias_grid[cvspace_loc[0]*nbin[0] + cvspace_loc[1]];
  } else {
    long long index = 0;
    long long stride = 1;
    for (int i = 0; i < cv_dim; ++i) {
        index += (long long)cvspace_loc[i] * stride;
        stride *= nbin[i]; 
    }
    return bias_grid[index];
  }
}

// 4. 把高斯累加到网格
void FixMetadynamics::add_hill(double *cv_values, double w) {
  double* delta_x=new double[cv_dim];
  if (cv_dim==1){
    for(long long g=0; g<grid_size;g++){
      // delta_x[0] =(cvspace_loc[0]- g)*dcv[0];
      delta_x[0] = cv_bound[0] + (g+0.5)*(cv_bound[1]-cv_bound[0])/nbin[0];
      delta_x[0] = cv_values[0]-delta_x[0];
      bias_grid[g] += w * gauss(1, delta_x,sigma);
      // DEBUG_LOG_COND((bias_grid[g]>1e-6),"init cvspace_loc=%d, bias_grid[%d]=%g",cvspace_loc[0],g,bias_grid[g]);
    }
  } else if (cv_dim==2){
    double xc, yc;
    int i,j;
    for (long long g=0;g<grid_size;g++){
      i = g/nbin[0];
      j = g%nbin[0];
      delta_x[0] = cv_bound[0] + (i+0.5)*(cv_bound[1]-cv_bound[0])/nbin[0];
      delta_x[0] = cv_values[0]-delta_x[0];
      delta_x[1] = cv_bound[2] + (j+0.5)*(cv_bound[3]-cv_bound[2])/nbin[1];
      delta_x[1] = cv_values[1]-delta_x[1];
      bias_grid[g] += w * gauss(2, delta_x, sigma);
    }
  } else {
    for (long long g = 0; g < grid_size; g++) {
        long long temp_g = g;
        // 采用行主序 (Row-Major Order)：CV0 变化最慢，CV(N-1) 变化最快。
        for (int i = 0; i < cv_dim; i++) {
            int loc_i = temp_g % nbin[cv_dim - 1 - i];
            double cv_min = cv_bound[2 * i];
            double cv_max = cv_bound[2 * i + 1];
            double bin_width = dcv[i];
            double x_center = cv_min + (loc_i + 0.5) * bin_width;
            delta_x[i] = cv_values[i] - x_center;
            temp_g /= nbin[cv_dim - 1 - i];
        }
        bias_grid[g] += w * gauss(cv_dim, delta_x, sigma);
    }
  }
  delete[] delta_x;
  DEBUG_LOG("add_hill_end\n");
}

// 5. 网格梯度（中心差分）
void FixMetadynamics::grid_gradient(int *cvspace_loc,
                                    double *dVdcvs) {
  DEBUG_LOG("grid_gradient");
  int *cvspace_loc_p = new int[cv_dim];
  int *cvspace_loc_m = new int[cv_dim];
  if (cv_dim==1){
    cvspace_loc_p[0] = cvspace_loc[0]+1;
    cvspace_loc_m[0] = cvspace_loc[0]-1;
    DEBUG_LOG("cvspace_loc_id p,m ; dcv = %d, %d; %lf",cvspace_loc_p[0],cvspace_loc_m[0],dcv[0]);
    DEBUG_LOG("cvspace_loc p, m = %g %g",get_total_bias(cvspace_loc_p), get_total_bias(cvspace_loc_m));
    dVdcvs[0] = (get_total_bias(cvspace_loc_p)-get_total_bias(cvspace_loc_m))/(2*dcv[0]);
    DEBUG_LOG("dVdcvs[0] = %g",dVdcvs[0]);
  } else if (cv_dim==2){
    cvspace_loc_p[0] = cvspace_loc[0]+1;
    cvspace_loc_m[0] = cvspace_loc[0]-1;
    cvspace_loc_p[1] = cvspace_loc[1];
    cvspace_loc_m[1] = cvspace_loc[1];
    dVdcvs[0] = (get_total_bias(cvspace_loc_p)-get_total_bias(cvspace_loc_m))/(2*dcv[0]);
    cvspace_loc_p[0] = cvspace_loc[0];
    cvspace_loc_m[0] = cvspace_loc[0];
    cvspace_loc_p[1] = cvspace_loc[1]+1;
    cvspace_loc_m[1] = cvspace_loc[1]-1;
    dVdcvs[1] = (get_total_bias(cvspace_loc_p)-get_total_bias(cvspace_loc_m))/(2*dcv[1]);
    // DEBUG_LOG("%f %f %f %f %f %f\n",bias_grid[(i+1)*nbin[0] + j], bias_grid[(i-1)*nbin[0] + j],bias_grid[i*nbin[0] + j+1],bias_grid[i*nbin[0] + j-1], dVdcv[0], dVdcv[1]);
  }
  delete[] cvspace_loc_p;
  delete[] cvspace_loc_m;
  DEBUG_LOG("dVdcvs[0] = %g",dVdcvs[0]);
  DEBUG_LOG("grid_gradient_end");
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