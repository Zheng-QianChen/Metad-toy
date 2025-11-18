

#include "fix_crystallize.h"
#include "lammpsplugin.h"  // 解决 lammpsplugin_t / LAMMPS_VERSION
#include "atom.h"
#include "comm.h"          // 解决 “incomplete type Comm”
#include "update.h"
#include "domain.h"
#include "memory.h"
#include "error.h"
// #include "compute.h"
#include <cmath>
#include <cstdio>

#include <cuda_runtime.h>
using namespace LAMMPS_NS;

static void all_reduce_cv(double *cv, double *cv_history, LAMMPS *lmp, FixMetadynamics *metad);
// LAMMPS_NS::Compute *compute_temp;


FixMetadynamics::FixMetadynamics(LAMMPS *lmp, int narg, char **arg)
  : Fix(lmp, narg, arg),
    sigma(0.05), height0(0.1), biasf(10.0), kBT(0.025852), pace(100),
    cv_dim(1), nbin_num(100), continue_from_file(false),
    // nadd(0), cv_hist(), height_hist(),
    // cv_bound[0](-1.0), cv_bound[1](1.0), cv_bound[2](-1.0), cv_bound[2](1.0),
    // nbin1(200), nbin2(200),
    bias_grid(nullptr), f_hills(nullptr)
{
  // 简单解析参数，示例： metad id group metad sigma height biasf pace
//   if (narg < 3) error->all(FLERR, "Usage: fix ID group metad sigma height biasf pace");
  sigma   = utils::numeric(FLERR, arg[3], false, lmp);
  height0 = utils::numeric(FLERR, arg[4], false, lmp);
  biasf   = utils::numeric(FLERR, arg[5], false, lmp);
  pace    = utils::inumeric(FLERR, arg[6], false, lmp);
  cv_dim  = utils::inumeric(FLERR, arg[7], false, lmp);
  nbin_num  = utils::inumeric(FLERR, arg[8], false, lmp);
  continue_from_file = utils::inumeric(FLERR, arg[9], false, lmp);

  memory->create(nbin, cv_dim, "metad:nbin_size");
  for (int i = 0; i < cv_dim; ++i) {
    nbin[i] = nbin_num;
  }

  // 分配网格
  grid_size = 1;
  for (int k = 0; k < cv_dim; ++k) {
      // 网格总大小等于所有维度 bin 数量的乘积
      grid_size *= nbin[k];
  }
  memory->create(bias_grid, grid_size, "metad:bias_grid");
  for (bigint i = 0; i < grid_size; ++i) {
      bias_grid[i] = 0.0;
  }

  memory->create(cv, cv_dim, "metad:cv");
  memory->create(cv_history, cv_dim, "metad:cv_history");
  memory->create(dVdcv, cv_dim, "metad:cv_history");
  for (int i = 0; i < cv_dim; ++i) {
      cv[i] = 0.0;
      cv_history[i] = 0.0;
      dVdcv[i] = 0.0;
  }

  if (comm->me==0) {
    f_check = fopen("a.txt","w");
  }
  // 输出文件
  first_run = true;
}

FixMetadynamics::~FixMetadynamics() {
  memory->destroy(bias_grid);
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
    if (cv_dim == 1){
      // double *cv[2]={0,0};
      memory->create(cv_bound, cv_dim*1, "metad:cv_bound");
      all_reduce_cv(cv, cv_history, lmp, this);            // 见下
      // for(int i=0; i<cv_dim; i++){
      //   cv_bound[i*2 + 0] = cv[i] - 20.0;
      //   cv_bound[i*2 + 1] = cv[i] + 20.0;
      // }
      cv_bound[0] = cv[0] - 0.0; cv_bound[1] = cv[0] + 40.0;
      first_run = false;}
    if (cv_dim == 2){
      // double *cv[2]={0,0};
      memory->create(cv_bound, cv_dim*2, "metad:cv_bound");
      all_reduce_cv(cv, cv_history, lmp, this);            // 见下
      // for(int i=0; i<cv_dim; i++){
      //   cv_bound[i*2 + 0] = cv[i] - 20.0;
      //   cv_bound[i*2 + 1] = cv[i] + 20.0;
      // }
      cv_bound[0] = cv[0] - 0.0; cv_bound[1] = cv[0] + 40.0;
      cv_bound[2] = cv[1] - 0.0; cv_bound[2] = cv[1] + 40.0;
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
          fprintf(f_check, "82\n");fflush(f_check);
                if (fgets(line, sizeof(line), f_read) == NULL) {
                    // 如果文件为空，则视为新文件
                } else {
          fprintf(f_check, "86\n");fflush(f_check);
                    // 循环读取每一行数据
                    while (fscanf(f_read, "%lld %lf %lf %lf %lf\n", 
                                  &step, &cv[0], &cv[1], &h, &s)==5) {
                      fprintf(f_check, "%lld %lf %lf %lf %lf\n",step, cv[0], cv[1],h,s);fflush(f_check);
                        add_hill(cv, h);
                        // current_timestep = step;
                    }
                    fprintf(f_check, "%lld %lf %lf %lf %lf\n", step, cv[0], cv[1], h, s);
                    fflush(f_check);
                }
                fclose(f_read);
                fprintf(f_check, "95\n");fflush(f_check);
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
              if (cv_dim==1){fprintf(f_hills, "# step cv height sigma\n");}
          }
      } // end if comm->me == 0
    }
    else{
      f_hills = fopen("HILLS", "w");
      if (cv_dim==2){fprintf(f_hills, "# step cv1 cv2 height sigma\n");}
      if (cv_dim==1){fprintf(f_hills, "# step cv height sigma\n");}
    }
    MPI_Bcast(&bias_grid[0], grid_size, MPI_DOUBLE, 0, world);
  }
}

// 归约 CV 到所有节点
static void all_reduce_cv(double *cv, double *cv_history, LAMMPS *lmp, FixMetadynamics *metad) {
  // double buf[2] = {0,0};
  // int nlocal = lmp->atom->nlocal;
  // for (int i=0;i<nlocal;i++) { buf[0]+=x[i][0]; buf[1]+=x[i][1]; }
  // MPI_Allreduce(MPI_IN_PLACE, buf, 2, MPI_DOUBLE, MPI_SUM, lmp->world);
  // int natoms = lmp->atom->natoms;
  // cv1 = buf[0]/natoms;
  // cv2 = buf[1]/natoms;
  double **x = lmp->atom->x;
  double dx,dy,dz,xbox,ybox,zbox;
  dx = x[1][0] - x[0][0];
  dy = x[1][1] - x[0][1];
  dz = x[1][2] - x[0][2];
  xbox = lmp->domain->xprd; // 使用 xprd 获取 X 周期性长度
  ybox = lmp->domain->yprd; // 使用 yprd 获取 Y 周期性长度
  zbox = lmp->domain->zprd; // 使用 zprd 获取 Z 周期性长度
  if (dx > xbox/2) {
      dx -= xbox;
  } else if (dx < -xbox/2) {
      dx += xbox;
  }
  if (dy > ybox/2) {
      dy -= ybox;
  } else if (dy < -ybox/2) {
      dy += ybox;
  }
  if (dz > zbox/2) {
      dz -= zbox;
  } else if (dz < -zbox/2) {
      dz += zbox;
  }

  if (metad->get_cv_dim()==2){
    // cv[0] = 2*(dx+dy+dz)/(xbox + ybox + zbox);
    cv[0] = 10;
    cv[1] = sqrt(dx*dx + dy*dy + dz*dz);
    cv_history[0] += cv[0];
    cv_history[1] += cv[1];
  }
  if (metad->get_cv_dim()==1){
    cv[0] = sqrt(dx*dx + dy*dy + dz*dz);
    cv_history[0] += cv[0];
  }
  // FILE *f=fopen("b.txt","a+");
  // fprintf(f,"%lf %lf\n",cv[0],cv_history[0]);
  // fclose(f);

}

/* helper: 计算高斯 */
static inline double gauss(double dx, double dy, double s) {
  return exp(-0.5*(dx*dx+dy*dy)/(s*s));
}

void FixMetadynamics::checkmax(double *cv, double *cv_max){
  if(cv_dim==2){
    // 检查边界
    bool resized = false;
    double new_min1 = cv_bound[0], new_max1 = cv_bound[1];
    double new_min2 = cv_bound[2], new_max2 = cv_bound[2];
    
    // 设置一个缓冲距离 (buffer)，例如 sigma 的 5 倍
    double buffer = 5.0 * sigma; 
    
    // 检查 CV1
    if (cv[0] < cv_bound[0] + buffer) {
        new_min1 = cv[0] - buffer;
        resized = true;
    } else if (cv[0] > cv_bound[1] - buffer) {
        new_max1 = cv[0] + buffer;
        resized = true;
    }

    // 检查 CV2
    if (cv[1] < cv_bound[2] + buffer) {
        new_min2 = cv[1] - buffer;
        resized = true;
    } else if (cv[1] > cv_bound[2] - buffer) {
        new_max2 = cv[1] + buffer;
        resized = true;
    }
  }
}

int FixMetadynamics::get_cv_dim() const {
        return cv_dim;
    }

// 3. 真正偏置力（解析梯度，比数值差分快）
void FixMetadynamics::post_force(int) {
  // fprintf(f_check, "post_force\n");fflush(f_check);
  if (cv_dim==1){
    all_reduce_cv(cv, cv_history, lmp, this);        // 归约
    if ((update->ntimestep % pace == 0)&&(pace!=0)) {
      cv_history[0] = cv[0];
      // cv_history[0] = cv_history[0]/pace;
      int i = static_cast<int>((cv_history[0]-cv_bound[0])/(cv_bound[1]-cv_bound[0])*nbin[0]);
      i = (i<1)?1:(i>=nbin[0]-1)?nbin[0]-2:i;
      // double Vbias = bias_grid[i];
      double Vbias = 0.0;
      double current_temp;
      current_temp = 300.0;
      double w = height0 * exp(-(Vbias)/(current_temp*(8.617e-5)*(biasf-1.0)));
      // fprintf(f_check, "%ld %.16g %.16g\n", update->ntimestep, Vbias, w);
      // fflush(f_check);
      if (comm->me==0) {
        fprintf(f_hills, "%ld %.16g %.16g %.16g\n", update->ntimestep, cv[0], w, sigma);
        fflush(f_hills);
      }
      add_hill(cv_history, w);             // 本地叠加
      MPI_Bcast(&bias_grid[0], grid_size, MPI_DOUBLE, 0, world);
      cv_history[0] = 0;
    }
    // 计算网格梯度
    grid_gradient(cv, dVdcv); // 见下
    // post_force_cus(cv[0], dVdcv[0]);
    post_force_r(cv[0], dVdcv[0]);
  }
  if (cv_dim==2){
    all_reduce_cv(cv, cv_history, lmp, this);        // 归约
    if ((update->ntimestep % pace == 0)&&(pace!=0)) {
    // if (update->ntimestep % pace == 0) {
      cv_history[0] = cv[0];
      cv_history[1] = cv[1];
      // cv_history[0] = cv_history[0]/pace;
      // cv_history[1] = cv_history[1]/pace;
      int i = static_cast<int>(((cv_history[0])-cv_bound[0])/(cv_bound[1]-cv_bound[0])*nbin[0]);
      int j = static_cast<int>(((cv_history[1])-cv_bound[2])/(cv_bound[3]-cv_bound[2])*nbin[1]);
      i = (i<1)?1:(i>=nbin[0]-1)?nbin[0]-2:i;
      j = (j<1)?1:(j>=nbin[1]-1)?nbin[1]-2:j;
      // double Vbias = bias_grid[i*nbin[0] + j];
      double Vbias = 0.0;
      double current_temp;
      current_temp = 1000.0;
      double w = height0 * exp(-(Vbias)/(current_temp*(1.38E-13)*(biasf-1.0)));
        // fprintf(f_hills, "w=%.12f, Vbias=%.12f\n", w, Vbias);
      if (comm->me==0) {
        fprintf(f_hills, "%ld %.16g %.16g %.16g %.16g\n", update->ntimestep, cv[0], cv[1], w, sigma);
        fflush(f_hills);
      }
      // add_hill(cv_history, w);             // 本地叠加
      add_hill(cv, w);             // 本地叠加
      MPI_Bcast(&bias_grid[0], grid_size, MPI_DOUBLE, 0, world);
      cv_history[0] = 0;
      cv_history[1] = 0;
    }

    // 计算网格梯度
    grid_gradient(cv, dVdcv); // 见下
    // post_force_cus(cv[0], dVdcv[0]);
    post_force_r(cv[1], dVdcv[1]);
  }
  // fprintf(f_check, "post_force_end\n");fflush(f_check);
}

void FixMetadynamics::post_force_cus(double cv, double dVdcv) {
    // fprintf(f_check, "post_force_cus\n");fflush(f_check);
    double **f = atom->f;
    int nlocal = atom->nlocal;
    double **x = lmp->atom->x;
    double *dcvdx = new double[3];
    get_dcvdx(cv, dcvdx);
    for (int i=0;i<nlocal;i++){
      f[i][0] -= dVdcv*dcvdx[0];
      f[i][1] -= dVdcv*dcvdx[1];
      f[i][2] -= dVdcv*dcvdx[2];
    }
    // fprintf(f_check, "post_force_cus_end\n");fflush(f_check);
}

void FixMetadynamics::post_force_r(double cv, double dVdcv) {
    // fprintf(f_check, "post_force_r\n");fflush(f_check);
    double **f = atom->f;
    int nlocal = atom->nlocal;
    double **x = lmp->atom->x;
    double *dcvdx = new double[3];
    get_dcvdx(cv, dcvdx);
    // 两个原子，dx具有方向性，dcvdx不相等哦！
    // for (int i=0;i<nlocal;i++){
    //   // fprintf(f_check, "dcvdx,dcvdy,dcvdz = %.6f, %.6f, %.6f \n", dcvdx[0], dcvdx[1], dcvdx[2]);
    //   // fprintf(f_check, "dVdcv = %.6f\n", dVdcv);
    //   // fprintf(f_check, "f0x,f0y,f0z = %.6f, %.6f, %.6f \n", f[i][0], f[i][1], f[i][2]);
    //   // f[i][0] -= dVdcv*dcvdx[i][0];
    //   // f[i][1] -= dVdcv*dcvdx[i][1];
    //   // f[i][2] -= dVdcv*dcvdx[i][2];
    //   // fprintf(f_check, "fx,fy,fz  = %.6f, %.6f, %.6f \n", f[i][0], f[i][1], f[i][2]);
    //   // fflush(f_check);
    // }
    fprintf(f_check, "dx, dy, dz  = %.6f, %.6f, %.6f \n", x[1][0]-x[0][0], x[1][1]-x[0][1], x[1][2]-x[0][2]);
    fprintf(f_check, "fx0,fy0,fz0  = %.6f, %.6f, %.6f \n", f[0][0], f[0][1], f[0][2]);
    f[0][0] += dVdcv*dcvdx[0];
    f[0][1] += dVdcv*dcvdx[1];
    f[0][2] += dVdcv*dcvdx[2];
    f[1][0] -= dVdcv*dcvdx[0];
    f[1][1] -= dVdcv*dcvdx[1];
    f[1][2] -= dVdcv*dcvdx[2];
    fprintf(f_check, "fx,fy,fz  = %.6f, %.6f, %.6f \n", f[0][0], f[0][1], f[0][2]);
    fflush(f_check);
    delete[] dcvdx;
    // fprintf(f_check, "post_force_r_end\n");fflush(f_check);
}

// 4. 把高斯累加到网格
void FixMetadynamics::add_hill(double *cv, double w) {
  // fprintf(f_check, "add_hill\n");fflush(f_check);
  if (cv_dim==1){
    double xc;
    for(int g=0; g<grid_size;g++){
      xc = cv_bound[0] + (g+0.5)*(cv_bound[1]-cv_bound[0])/nbin[0];
      bias_grid[g] += w*gauss(cv[0]-xc,0,sigma);
    }
  }
  if (cv_dim==2){
    double xc, yc;
    int i,j;
    for (int g=0;g<grid_size;g++){
      i = g/nbin[0];
      j = g%nbin[0];
      xc = cv_bound[0] + (i+0.5)*(cv_bound[1]-cv_bound[0])/nbin[0];
      yc = cv_bound[2] + (j+0.5)*(cv_bound[3]-cv_bound[2])/nbin[1];
      bias_grid[g] += w * gauss(cv[0]-xc, cv[1]-yc, sigma);
    }
  }
  // fprintf(f_check, "add_hill_end\n");fflush(f_check);
}

// 5. 网格梯度（中心差分）
void FixMetadynamics::grid_gradient(double *cv,
                                    double *dVdcv) {
  // fprintf(f_check, "grid_gradient\n");fflush(f_check);
  if (cv_dim==1){
    // fprintf(f_check, "%lf\n",cv_bound[0]);fflush(f_check);
    // fprintf(f_check, "%lf %lf %lf %lf\n",cv[0],cv_bound[0],cv_bound[1],nbin[0]);fflush(f_check);
    int i = static_cast<int>((cv[0]-cv_bound[0])/(cv_bound[1]-cv_bound[0])*nbin[0]);
    i = (i<1)?1:(i>=nbin[0]-1)?nbin[0]-2:i;
    double dx = (cv_bound[1]-cv_bound[0])/nbin[0];
    dVdcv[0] = (bias_grid[i+1]-bias_grid[i-1])/(2*dx);
  }
  if (cv_dim==2){
    int i = static_cast<int>((cv[0]-cv_bound[0])/(cv_bound[1]-cv_bound[0])*nbin[0]);
    int j = static_cast<int>((cv[1]-cv_bound[2])/(cv_bound[3]-cv_bound[2])*nbin[1]);
    i = (i<1)?1:(i>=nbin[0]-1)?nbin[0]-2:i;
    j = (j<1)?1:(j>=nbin[1]-1)?nbin[1]-2:j;
    double dx = (cv_bound[1]-cv_bound[0])/nbin[0];
    double dy = (cv_bound[3]-cv_bound[2])/nbin[1];
    dVdcv[0] = (bias_grid[(i+1)*nbin[0] + j]-bias_grid[(i-1)*nbin[0] + j])/(2*dx);
    dVdcv[1] = (bias_grid[i*nbin[0] + j+1]-bias_grid[i*nbin[0] + j-1])/(2*dy);
    // fprintf(f_hills, "%f %f %f %f %f %f\n",bias_grid[(i+1)*nbin[0] + j], bias_grid[(i-1)*nbin[0] + j],bias_grid[i*nbin[0] + j+1],bias_grid[i*nbin[0] + j-1], dVdcv[0], dVdcv[1]);
  }
  // fprintf(f_check, "grid_gradient_end\n");fflush(f_check);
}

void FixMetadynamics::get_dcvdx(double cv, double *dcvdx){
  // fprintf(f_check, "get_dcvdx\n");fflush(f_check);
  double **x = lmp->atom->x;
  double dx,dy,dz,xbox,ybox,zbox;
  dx = x[1][0] - x[0][0];
  dy = x[1][1] - x[0][1];
  dz = x[1][2] - x[0][2];
  xbox = lmp->domain->xprd; // 使用 xprd 获取 X 周期性长度
  ybox = lmp->domain->yprd; // 使用 yprd 获取 Y 周期性长度
  zbox = lmp->domain->zprd; // 使用 zprd 获取 Z 周期性长度
  if (dx > xbox/2) {
      dx -= xbox;
  } else if (dx < -xbox/2) {
      dx += xbox;
  }
  if (dy > ybox/2) {
      dy -= ybox;
  } else if (dy < -ybox/2) {
      dy += ybox;
  }
  if (dz > zbox/2) {
      dz -= zbox;
  } else if (dz < -zbox/2) {
      dz += zbox;
  }
  // dcvdx = dr/dx = dx/r
  dcvdx[0] = dx/cv;
  dcvdx[1] = dy/cv;
  dcvdx[2] = dz/cv;
  // fprintf(f_check, "get_dcvdx_end\n");fflush(f_check);
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