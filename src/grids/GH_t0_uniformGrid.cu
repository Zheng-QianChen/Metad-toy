#include "lammps.h"
#include "memory.h"
#include "comm.h"
#include "update.h"
#include "error.h"
#include "force.h"
#include "fix_crystallize.h"
#include "zqc_gaussian.h"
#include "zqc_debug.h"

// static inline double gauss_calc(int dim, double* dx, double s);


MetaD_zqc::Gaussian_Hill_Base::Gaussian_Hill_Base(LAMMPS_NS::LAMMPS *lmp,
                            FILE* f_check,
                            int cv_dim,
                            double sigma, double height0, double biasf,
                            int continue_from_file, int WellT_bool):
                            lmp(lmp),
                            f_hills(nullptr),
                            f_check(f_check),
                            cv_dim(cv_dim),
                            sigma(sigma),
                            height0(height0),
                            biasf(biasf),
                            KB(KB),
                            continue_from_file(continue_from_file),
                            WellT_bool(WellT_bool){
}

MetaD_zqc::Gaussian_Hill_Base::~Gaussian_Hill_Base(){
    if (f_hills) {
        fclose(f_hills);
        // fclose(f_check);
    }
}

template<int D>
MetaD_zqc::GH_t0_uniformGrid<D>::GH_t0_uniformGrid(LAMMPS_NS::LAMMPS *lmp,
                            FILE* f_check,
                            int cv_dim,
                            double sigma, double height0, double biasf,
                            int continue_from_file, int WellT_bool,
                            double *cv_bound, int *nbin):
                            Gaussian_Hill_Base(lmp, f_check,
                                cv_dim, sigma, height0, biasf,
                                continue_from_file, WellT_bool){
    // ?
    this->cv_bound = cv_bound;
    this->nbin = nbin;
    lmp->memory->create(delta_x, cv_dim, "metad:gaussian:delta_x");
}

template<int D>
MetaD_zqc::GH_t0_uniformGrid<D>::~GH_t0_uniformGrid(){
    lmp->memory->destroy(cv_bound);
    lmp->memory->destroy(nbin);
    lmp->memory->destroy(cvspace_loc);
    lmp->memory->destroy(dcv);
    lmp->memory->destroy(bias_grid);
    lmp->memory->destroy(delta_x);
    lmp->memory->destroy(index_radius);
    lmp->memory->destroy(lower);
    lmp->memory->destroy(upper);
}

template<int D>
void MetaD_zqc::GH_t0_uniformGrid<D>::init_set_mode(){
    this->KB = lmp->force->boltz;
    lmp->memory->create(cvspace_loc, cv_dim, "metad:cvspace_loc");
    // memory->create(cv_compute, cv_dim, "metad:cv_compute");

    // 分配网格
    grid_size = 1;
    for (int k = 0; k < cv_dim; ++k) {grid_size *= nbin[k];}
    lmp->memory->create(bias_grid, grid_size, "metad:bias_grid");
    for (LAMMPS_NS::bigint k = 0; k < grid_size; ++k) {bias_grid[k] = 0.0;}
    // GH_t0_uniformGrid<D>::precompute_index_radius();
    lmp->memory->create(index_radius, cv_dim, "metad:index_radius");
    lmp->memory->create(dcv, cv_dim, "metad:dcv");
    lmp->memory->create(lower, cv_dim, "metad:lower");
    lmp->memory->create(upper, cv_dim, "metad:upper");
    for (int k = 0; k < cv_dim; ++k) {
        dcv[k] = (cv_bound[2*k+1]-cv_bound[2*k])/nbin[k];
        LOG("dcv[k] = %g", dcv[k]);
        index_radius[k] = static_cast<int>(ceil(4.0 * sigma / dcv[k])) + 1;
        printf("Precompute index_radius[%d] = %d\n", k, index_radius[k]);
    }
    this->init_hills();

    DEBUG_LOG("Fix init end.");
}

template<int D>
void MetaD_zqc::GH_t0_uniformGrid<D>::init_hills(){
    int me = lmp->comm->me;
    int nprocs = lmp->comm->nprocs;
    long long total_size = 0;

    // 1. 获取文件总大小并广播 (避免所有进程同时访问元数据服务器)
    if (me == 0) {
        if (continue_from_file) {
            FILE *f_test = fopen("HILLS", "r");
            if (f_test) {
                fseek(f_test, 0, SEEK_END);
                total_size = ftell(f_test);
                fclose(f_test);
            }
        }
    }
    MPI_Bcast(&total_size, 1, MPI_LONG_LONG, 0, lmp->world);

    // 2. 如果文件存在且需要续接，则进入并行读取逻辑
    if (total_size > 0 && continue_from_file) {
        long long my_start_offset = me * (total_size / nprocs);
        long long my_end_offset = (me == nprocs - 1) ? total_size : (me + 1) * (total_size / nprocs);

        FILE *f_read = fopen("HILLS", "r");
        if (f_read) {
            fseek(f_read, my_start_offset, SEEK_SET);
            char line[1024];
            
            // 对齐逻辑：非 0 进程丢弃第一行（由前一个进程读过界来处理）
            if (me != 0) {
                if (!fgets(line, sizeof(line), f_read)) { /* 到达末尾 */ }
            }

            std::vector<double> current_cvs(cv_dim);
            // 只要起始位置在自己的物理区间内，就继续读取完整行
            while (ftell(f_read) < my_end_offset) {
                if (!fgets(line, sizeof(line), f_read)) break;
                if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;

                char *ptr = line;
                char *next_ptr;
                // 解析 Step
                strtoll(ptr, &next_ptr, 10);
                ptr = next_ptr;

                // 解析 CVs
                for (int d = 0; d < cv_dim; ++d) {
                    current_cvs[d] = strtod(ptr, &next_ptr);
                    ptr = next_ptr;
                }
                // 解析 Height (h)
                double h = strtod(ptr, &next_ptr);
                double s = strtod(ptr, &next_ptr);
                // sigma (s) 在此处根据需要解析，若只更新 grid 则解析到 h 即可

                // 累加到本地 bias_grid
                add_to_grid(current_cvs.data(), h, s);
            }
            fclose(f_read);
        }

        // 3. 规约通信：将所有进程本地计算的 bias_grid 累加到 rank 0
        if (nprocs > 1) {
            // 使用 MPI_IN_PLACE 减少 rank 0 的额外内存开销
            MPI_Reduce(me == 0 ? MPI_IN_PLACE : &bias_grid[0], 
                       &bias_grid[0], grid_size, MPI_DOUBLE, MPI_SUM, 0, lmp->world);
        }
    }

    // 4. Rank 0 负责文件句柄的后续管理
    if (me == 0) {
        if (total_size > 0 && continue_from_file) {
            f_hills = fopen("HILLS", "a");
            DEBUG_LOG("restart hills parallel end");
        } else {
            f_hills = fopen("HILLS", "w");
            if (cv_dim == 2) fprintf(f_hills, "# step cv1 cv2 height sigma\n");
            if (cv_dim == 1) fprintf(f_hills, "# step cv_values height sigma\n");
        }

        if (f_hills == NULL) {
            lmp->error->one(FLERR, "Cannot open HILLS file for writing/appending.");
        }
    }

    // 5. 根据你的要求，这里不再进行 Bcast
    // 注意：如果其他进程的 Kernel 需要读取最新的 bias_grid 副本，请务必恢复广播
    // MPI_Bcast(&bias_grid[0], grid_size, MPI_DOUBLE, 0, lmp->world);
    MPI_Barrier(lmp->world);
}

template<int D>
void MetaD_zqc::GH_t0_uniformGrid<D>::add_hill(double *cv_history){
    if (lmp->comm->me==0) {
        get_cvspace_loc(cv_history, cvspace_loc);
        double Vbias = 0.0;
        if (WellT_bool){
          Vbias = get_total_bias(cvspace_loc);
        }
        double w = height0 * exp(-(Vbias)/(current_temp*KB*(biasf-1.0)));
        write_hill(cv_history, w);
        add_to_grid(cv_history, w, this->sigma);
    }
    // MPI_Barrier(lmp->world);
    // MPI_Bcast(&bias_grid[0], grid_size, MPI_DOUBLE, 0, lmp->world);
}

template<int D>
void MetaD_zqc::GH_t0_uniformGrid<D>::write_hill(double *cv_values, double w){
    if (lmp->comm->me==0) {
        fprintf(f_hills, "%ld", lmp->update->ntimestep);
        for (int ii = 0; ii < cv_dim; ii++) {
            fprintf(f_hills, " %.16g", cv_values[ii]);
        }
        fprintf(f_hills, " %.16g %.16g\n", w, this->sigma);
        fflush(f_hills);
    }
}

// =============================================================================
// add_to_grid
// =============================================================================
// 4. 把高斯累加到网格
template<>
void MetaD_zqc::GH_t0_uniformGrid<1>::add_to_grid(double *cv_values, double w, double sig) {
  int center_idx = static_cast<int>((cv_values[0] - cv_bound[0]) / dcv[0]);
  lower[0] = std::max(0, center_idx - index_radius[0]);
  upper[0] = std::min((int)nbin[0] - 1, center_idx + index_radius[0]);
  for(long long g=lower[0]; g<=upper[0];g++){
      delta_x[0] = cv_bound[0] + (g+0.5)*(cv_bound[1]-cv_bound[0])/nbin[0];
      delta_x[0] = cv_values[0]-delta_x[0];
      bias_grid[g] += w * gauss_calc(1, delta_x, sig);
      // DEBUG_LOG_COND((bias_grid[g]>1e-6),"init cvspace_loc=%d, bias_grid[%d]=%g",cvspace_loc[0],g,bias_grid[g]);
      // if (bias_grid[g]>1e-6){
      //     printf("init cvspace_loc=%d, bias_grid[%d]=%g\n",cvspace_loc[0],g,bias_grid[g]);
      // }
  }
  DEBUG_LOG("add_hill_end");
}

template<>
void MetaD_zqc::GH_t0_uniformGrid<2>::add_to_grid(double *cv_values, double w, double sig) {
  double xc, yc;
  for (int k=0; k<cv_dim; k++){
    int center_idx = static_cast<int>((cv_values[k] - cv_bound[2*k]) / dcv[k]);
    lower[k] = std::max(0, center_idx - index_radius[k]);
    upper[k] = std::min((int)nbin[k] - 1, center_idx + index_radius[k]);
  }
  long long i,j,g;
  for (long long i=lower[0];i<=upper[0];i++){
    for (long long j=lower[1];j<=upper[1];j++){
      g = i * nbin[1] + j;
      delta_x[0] = cv_bound[0] + (i+0.5)*(cv_bound[1]-cv_bound[0])/nbin[0];
      delta_x[0] = cv_values[0]-delta_x[0];
      delta_x[1] = cv_bound[2] + (j+0.5)*(cv_bound[3]-cv_bound[2])/nbin[1];
      delta_x[1] = cv_values[1]-delta_x[1];
      bias_grid[g] += w * gauss_calc(2, delta_x, sig);
    }
  }
  DEBUG_LOG("add_hill_end"); 
}

template<>
void MetaD_zqc::GH_t0_uniformGrid<3>::add_to_grid(double *cv_values, double w, double sig) {
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
      bias_grid[g] += w * gauss_calc(cv_dim, delta_x, sig);
  }
  DEBUG_LOG("add_hill_end");
}

// =============================================================================
// get_dVdcv
// =============================================================================
template<>
void MetaD_zqc::GH_t0_uniformGrid<1>::get_dVdcv(double *cv_values,
                                    double *dVdcvs) {
    if (lmp->comm->me==0) {
        DEBUG_LOG("get_dVdcv");
        // printf("cv_values=%g\n",cv_values[0]);
        get_cvspace_loc(cv_values, cvspace_loc);
        int i = cvspace_loc[0];
        // 计算原子相对于网格点 i 的偏移量 [0, 1]
        double x = (cv_values[0] - (cv_bound[0] + i * dcv[0])) / dcv[0];
        // 确保 x 在插值公式中不因边界截断而产生错误的斜率
        if (x < 0.0) x = 0.0; if (x > 1.0) x = 1.0;
        double p0 = bias_grid[i - 1];
        double p1 = bias_grid[i];
        double p2 = bias_grid[i + 1];
        double p3 = bias_grid[i + 2];
        dVdcvs[0] = ((-0.5 * p0 + 0.5 * p2) + 
                    x * (p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3) + 
                    1.5 * x * x * (-p0 + 3.0 * p1 - 3.0 * p2 + p3)) / dcv[0];
        DEBUG_LOG("i,p0,p1,p2,p3 = %d %g %g %g %g, dVdcv[0]=%g", i, p0, p1, p2, p3,dVdcvs[0]);
        // printf("i,p0,p1,p2,p3 = %d %g %g %g %g, dVdcv[0]=%g\n", i, p0, p1, p2, p3,dVdcvs[0]);
        DEBUG_LOG("dVdcvs[0] = %g",dVdcvs[0]);
        if (isnan(dVdcvs[0])) { // 检查 NaN
            printf("Warning: dVdcvs[%d] is NaN, setting to 0\n", 0);
            dVdcvs[0] = 0.0;
        }
        DEBUG_LOG("grid_gradient_end");
    }
    MPI_Bcast(dVdcvs, cv_dim, MPI_DOUBLE, 0, lmp->world);
}

template<>
void MetaD_zqc::GH_t0_uniformGrid<2>::get_dVdcv(double *cv_values,
                                    double *dVdcvs) {
    if (lmp->comm->me==0) {
        get_cvspace_loc(cv_values, cvspace_loc);
        int i = cvspace_loc[0];
        int j = cvspace_loc[1]; 
        // 2. 计算相对于左下角网格点的偏移量 dx, dy (范围 [0, 1])
        double dx = (cv_values[0] - (cv_bound[0] + i * dcv[0])) / dcv[0];
        double dy = (cv_values[1] - (cv_bound[2] + j * dcv[1])) / dcv[1];
        // 边界保护
        if (dx < 0.0) dx = 0.0; if (dx > 1.0) dx = 1.0;
        if (dy < 0.0) dy = 0.0; if (dy > 1.0) dy = 1.0;
        // 3. 获取周边 4 个点的偏置势值
        // 布局： p01(i, j+1) -- p11(i+1, j+1)
        //        |                |
        //       p00(i, j)   -- p10(i+1, j)
        double p00 = bias_grid[i * nbin[1] + j];         // (i, j)
        double p10 = bias_grid[(i + 1) * nbin[1] + j];   // (i+1, j)
        double p01 = bias_grid[i * nbin[1] + (j + 1)];   // (i, j+1)
        double p11 = bias_grid[(i + 1) * nbin[1] + (j + 1)]; // (i+1, j+1)
        // 4. 双线性插值求偏导 (dV/dCV)
        // V(dx, dy) = p00(1-dx)(1-dy) + p10*dx*(1-dy) + p01*(1-dx)*dy + p11*dx*dy
        // dV/ddx = [ (p10 - p00)(1-dy) + (p11 - p01)dy ] / dcv[0]
        dVdcvs[0] = ((p10 - p00) * (1.0 - dy) + (p11 - p01) * dy) / dcv[0];
        // dV/ddy = [ (p01 - p00)(1-dx) + (p11 - p10)dx ] / dcv[1]
        dVdcvs[1] = ((p01 - p00) * (1.0 - dx) + (p11 - p10) * dx) / dcv[1];
        DEBUG_LOG("2D Gradient: i=%d, j=%d, dx=%f, dy=%f, dVdcv[0]=%g, dVdcv[1]=%g", 
                i, j, dx, dy, dVdcvs[0], dVdcvs[1]);
        // printf("2D Gradient: i=%d, j=%d, dx=%f, dy=%f, dVdcv[0]=%g, dVdcv[1]=%g\n", 
        //           i, j, dx, dy, dVdcvs[0], dVdcvs[1]);
        DEBUG_LOG("dVdcvs[0] = %g",dVdcvs[0]);
        if (isnan(dVdcvs[0]) || isnan(dVdcvs[1])) { // 检查 NaN
            printf("Warning: dVdcvs[%d] is NaN, setting to 0\n", 0);
            dVdcvs[0] = 0.0;
            dVdcvs[1] = 0.0;
        }
        DEBUG_LOG("grid_gradient_end");
    }
    MPI_Bcast(dVdcvs, cv_dim, MPI_DOUBLE, 0, lmp->world);
}

template<>
void MetaD_zqc::GH_t0_uniformGrid<3>::get_dVdcv(double *cv_values,
                                    double *dVdcvs) {
    if (lmp->comm->me==0) {
        DEBUG_LOG("grid_gradient_end");
    }
    MPI_Bcast(dVdcvs, cv_dim, MPI_DOUBLE, 0, lmp->world);
}


// =============================================================================
// get_cvspace_loc
// =============================================================================
template<int D>
void MetaD_zqc::GH_t0_uniformGrid<D>::get_cvspace_loc(double* cv_values, 
                                                      int* cvspace_loc){
  for(int ii=0; ii<cv_dim; ii++){
    // DEBUG_LOG("cv_values[%d] = %g, cvbound=[%g,%g], grid=%d",ii,cv_values[ii],cv_bound[ii*2+1],cv_bound[ii*2], nbin[ii]);
    int loc = static_cast<int>(((cv_values[ii]-cv_bound[ii*2])/(cv_bound[ii*2+1]-cv_bound[ii*2]))*nbin[ii]);
    if (loc < 1) {loc = 1;}
    if (loc > nbin[ii] - 3) {loc = nbin[ii] - 3;}
    cvspace_loc[ii] = (loc<1)?1:(loc>=nbin[ii]-1)?nbin[ii]-2:loc;
    // DEBUG_LOG("cvspace_loc[%d] = %d",ii,cvspace_loc[ii]);
    // printf("cvspace_loc[%d] = %d\n",ii,cvspace_loc[ii]);
  }
}



// =============================================================================
// get_total_bias
// =============================================================================
template<>
double MetaD_zqc::GH_t0_uniformGrid<1>::get_total_bias(int* cvspace_loc){
  return bias_grid[cvspace_loc[0]];
}

template<>
double MetaD_zqc::GH_t0_uniformGrid<2>::get_total_bias(int* cvspace_loc){
  return bias_grid[cvspace_loc[0]*nbin[1] + cvspace_loc[1]];
}

template<>
double MetaD_zqc::GH_t0_uniformGrid<3>::get_total_bias(int* cvspace_loc){
  long long index = 0;
  long long stride = 1;
  for (int i = 0; i < cv_dim; ++i) {
      index += (long long)cvspace_loc[i+1] * stride;
      stride *= nbin[i]; 
  }
  return bias_grid[index];
}

// Catmull-Rom 一维求值（与 get_dVdcv 的样条族一致）
static inline double catmull_rom_value(double p0, double p1, double p2, double p3, double x) {
  return p1 + 0.5 * x * (
      (p2 - p0) +
      x * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) +
      x * x * (-p0 + 3.0 * p1 - 3.0 * p2 + p3)
  );
}

template<>
double MetaD_zqc::GH_t0_uniformGrid<1>::get_bias_energy(double *cv_values){
  get_cvspace_loc(cv_values, cvspace_loc);
  int i = cvspace_loc[0];
  double x = (cv_values[0] - (cv_bound[0] + i * dcv[0])) / dcv[0];
  if (x < 0.0) x = 0.0; if (x > 1.0) x = 1.0;
  return catmull_rom_value(bias_grid[i - 1], bias_grid[i], bias_grid[i + 1], bias_grid[i + 2], x);
}

template<>
double MetaD_zqc::GH_t0_uniformGrid<2>::get_bias_energy(double *cv_values){
  // 与现有 2D 梯度一致：双线性四角插值
  get_cvspace_loc(cv_values, cvspace_loc);
  int i = cvspace_loc[0];
  int j = cvspace_loc[1];
  double dx = (cv_values[0] - (cv_bound[0] + i * dcv[0])) / dcv[0];
  double dy = (cv_values[1] - (cv_bound[2] + j * dcv[1])) / dcv[1];
  if (dx < 0.0) dx = 0.0; if (dx > 1.0) dx = 1.0;
  if (dy < 0.0) dy = 0.0; if (dy > 1.0) dy = 1.0;
  double p00 = bias_grid[i * nbin[1] + j];
  double p10 = bias_grid[(i + 1) * nbin[1] + j];
  double p01 = bias_grid[i * nbin[1] + (j + 1)];
  double p11 = bias_grid[(i + 1) * nbin[1] + (j + 1)];
  double v0 = p00 * (1.0 - dx) + p10 * dx;
  double v1 = p01 * (1.0 - dx) + p11 * dx;
  return v0 * (1.0 - dy) + v1 * dy;
}

template<>
double MetaD_zqc::GH_t0_uniformGrid<3>::get_bias_energy(double *cv_values){
  // 3D：取最近网格点（与均匀网格 3D 梯度实现复杂度对齐的保守近似）
  get_cvspace_loc(cv_values, cvspace_loc);
  return get_total_bias(cvspace_loc);
}


// =============================================================================
// gauss_calc
// =============================================================================
/* helper: 计算高斯 */
template<>
double MetaD_zqc::GH_t0_uniformGrid<1>::gauss_calc(int dim, double* dx, double s) {
  return exp(-0.5*(dx[0]*dx[0])/(s*s));
}
  
template<>
double MetaD_zqc::GH_t0_uniformGrid<2>::gauss_calc(int dim, double* dx, double s) {
  return exp(-0.5*(dx[0]*dx[0]+dx[1]*dx[1])/(s*s));
}
  
template<>
double MetaD_zqc::GH_t0_uniformGrid<3>::gauss_calc(int dim, double* dx, double s) {
  double paw=0.0;
  for (int i=0; i<dim; i++){
    paw += dx[i]*dx[i];
  }
  return exp(-0.5*(paw)/(s*s));
}
