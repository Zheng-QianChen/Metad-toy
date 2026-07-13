#include "lammps.h"
#include "memory.h"
#include "comm.h"
#include "update.h"
#include "error.h"
#include "force.h"
#include "fix_crystallize.h"
#include "zqc_gaussian.h"
#include "zqc_debug.h"

// double MetaD_zqc::GH_t1_sparseHash<D>::get_bias_at(const CoordKey& k) {
//     auto it = bias_hash.find(k);
//     return (it == bias_hash.end()) ? 0.0 : it->second;
// }

template<int D>
MetaD_zqc::GH_t1_sparseHash<D>::GH_t1_sparseHash(LAMMPS_NS::LAMMPS *lmp,
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
}

template<int D>
MetaD_zqc::GH_t1_sparseHash<D>::~GH_t1_sparseHash(){
    lmp->memory->destroy(cv_bound);
    lmp->memory->destroy(nbin);
    lmp->memory->destroy(cvspace_loc);
    lmp->memory->destroy(dcv);
    // lmp->memory->destroy(bias_grid);
    // lmp->memory->destroy(delta_x);
    lmp->memory->destroy(index_radius);
    // lmp->memory->destroy(lower);
    // lmp->memory->destroy(upper);
}

template<int D>
void MetaD_zqc::GH_t1_sparseHash<D>::init_set_mode(){
    this->KB = lmp->force->boltz;
    lmp->memory->create(cvspace_loc, cv_dim, "metad:cvspace_loc");
    // memory->create(cv_compute, cv_dim, "metad:cv_compute");

    // 分配网格
    lmp->memory->create(index_radius, cv_dim, "metad:index_radius");
    lmp->memory->create(dcv, cv_dim, "metad:dcv");
    // var delta_x 用于存储当前 CV 与网格点中心的偏移量，维度为 cv_dim
    // lmp->memory->create(delta_x, cv_dim, "metad:gaussian:delta_x");
    // lmp->memory->create(lower, cv_dim, "metad:lower");
    // lmp->memory->create(upper, cv_dim, "metad:upper");
    for (int k = 0; k < cv_dim; ++k) {
        dcv[k] = (cv_bound[2*k+1]-cv_bound[2*k])/nbin[k];
        LOG("dcv[k] = %g", dcv[k]);
        index_radius[k] = static_cast<int>(ceil(4.0 * sigma / dcv[k])) + 1;
        printf("Precompute index_radius[%d] = %d\n", k, index_radius[k]);
    }
    this->io_hills();

    DEBUG_LOG("Fix init end.");
}

template<int D>
void MetaD_zqc::GH_t1_sparseHash<D>::io_hills(){
    // TODO:可以使用MPI的规约通信改进当前文件存取
    printf("io_hills start on rank %d\n", lmp->comm->me);
    if (lmp->comm->me == 0) {
      if (continue_from_file){
          // 尝试打开 HILLS 文件进行读取
          FILE *f_read = fopen("HILLS", "r");
          printf("checking HILLS\n");
            if (f_read) {
                printf("reading HILLS\n");
                // 1. 读取 HILLS 文件并重建 bias_grid
                char line[1024];
                std::string format = "%lld";
                for (int d = 0; d < cv_dim; ++d) format += " %lf";
                format += " %lf %lf\n";
                std::vector<double> current_cvs(cv_dim);
                DEBUG_LOG("restart hills");
                if (fgets(line, sizeof(line), f_read)){
                  double h, s;
                  long long current_timestep = 0;
                  long long step;
                  while (fgets(line, sizeof(line), f_read)) {
                      // fscanf 的参数需要手动根据指针位置传入，这里利用数组地址特性
                      // 我们需要：&step, &cv[0], &cv[1]... , &h, &s
                      // 由于 fscanf 不直接支持数组指针展开，这里建议使用通用解析逻辑：
                      if (line[0] == '#' || line[0] == '\n') continue; // 跳过注释或空行
                      char *ptr = line;
                      char *next_ptr;
                      // 解析 Step
                      step = strtoll(ptr, &next_ptr, 10);
                      ptr = next_ptr;
                      // 解析 CVs
                      for (int d = 0; d < cv_dim; ++d) {
                          current_cvs[d] = strtod(ptr, &next_ptr);
                          ptr = next_ptr;
                      }
                      // 解析 h 和 s
                      h = strtod(ptr, &next_ptr);
                      ptr = next_ptr;
                      s = strtod(ptr, &next_ptr);
                      // 4. 更新 Grid
                      // 将读取的 CV 复制到类成员 cv_values 中供 add_hill 使用
                    //   for (int d = 0; d < cv_dim; ++d) cv_values[d] = current_cvs[d];
                      
                      get_cvspace_loc(current_cvs.data(), cvspace_loc);
                      add_to_grid(current_cvs.data(), h);
                  }
                }
                fclose(f_read);
                DEBUG_LOG("restart hills end");
                // 2. 重新打开 HILLS 文件，使用 "a" (追加) 模式
                f_hills = fopen("HILLS", "a");
                if (f_hills == NULL) {
                    lmp->error->all(FLERR, "Cannot open HILLS file for appending.");
                }
          } else {
              // --- 未找到 HILLS 文件，创建新文件 ---
              f_hills = fopen("HILLS", "w");
              if (f_hills == NULL) {
                  lmp->error->all(FLERR, "Cannot open HILLS file for writing.");
              }
              if (cv_dim==2){fprintf(f_hills, "# step cv1 cv2 height sigma\n");}
              if (cv_dim==1){fprintf(f_hills, "# step cv_values height sigma\n");}
          }
      } else{
          f_hills = fopen("HILLS", "w");
          if (cv_dim==2){fprintf(f_hills, "# step cv1 cv2 height sigma\n");}
          if (cv_dim==1){fprintf(f_hills, "# step cv_values height sigma\n");}
      } // end of continue_from_file
    } // end of comm->me==0
    MPI_Barrier(lmp->world);
    GridHashBcast();
}

template<int D>
void MetaD_zqc::GH_t1_sparseHash<D>::GridHashBcast(){
  // 其他rank不需要查询bias_hash，因此不需要完整的bias_hash数据结构。
  // 我们只需要在rank 0上维护bias_hash，
  // 并在每次更新后广播必要的信息（如新添加的hill位置和权重）给其他rank即可。
  // if (lmp->comm->me == 0) {
  //     int num_entries = bias_hash.size();
  //     MPI_Bcast(&num_entries, 1, MPI_INT, 0, lmp->world);
  //     std::vector<Entry> buffer;
  //     for (auto const& [key, val] : bias_hash) {
  //         buffer.push_back({key, val});
  //     }
  //     MPI_Bcast(buffer.data(), num_entries * sizeof(Entry), MPI_BYTE, 0, lmp->world);
  // } else {
  //     // 其他 Rank 接收并重建
  //     int num_entries;
  //     MPI_Bcast(&num_entries, 1, MPI_INT, 0, lmp->world);
  //     std::vector<Entry> buffer(num_entries);
  //     MPI_Bcast(buffer.data(), num_entries * sizeof(Entry), MPI_BYTE, 0, lmp->world);
      
  //     for (auto& e : buffer) bias_hash[e.k] = e.v;
  // }
}

template<int D>
void MetaD_zqc::GH_t1_sparseHash<D>::add_hill(double *cv_history){
    if (lmp->comm->me==0) {
        get_cvspace_loc(cv_history, cvspace_loc);
        double Vbias = 0.0;
        if (WellT_bool){
          Vbias = get_total_bias(cvspace_loc);
        }
        double w = height0 * exp(-(Vbias)/(current_temp*KB*(biasf-1.0)));
        write_hill(cv_history, w);
        add_to_grid(cv_history, w);
    }
    MPI_Barrier(lmp->world);
    GridHashBcast();
}

template<int D>
void MetaD_zqc::GH_t1_sparseHash<D>::write_hill(double *cv_values, double w){
    if (lmp->comm->me==0) {
        fprintf(f_hills, "%ld", lmp->update->ntimestep);
        for (int ii = 0; ii < cv_dim; ii++) {
            fprintf(f_hills, " %.16g", cv_values[ii]);
        }
        fprintf(f_hills, " %.16g %.16g\n", w, sigma);
        fflush(f_hills);
    }
}

// =============================================================================
// add_to_grid
// =============================================================================
// 4. 把高斯累加到网格
template<int D>
void MetaD_zqc::GH_t1_sparseHash<D>::add_to_grid(double *cv_values, double w) {
  CoordKey key;
  int lower[cv_dim], upper[cv_dim];
  double xc, yc;
  double delta_x[cv_dim]; // 存储当前 CV 与网格点中心的偏移量
  for (int k=0; k<cv_dim; k++){
    int center_idx = static_cast<int>((cv_values[k] - cv_bound[2*k]) / dcv[k]);
    lower[k] = std::max(0, center_idx - index_radius[k]);
    upper[k] = std::min((int)nbin[k] - 1, center_idx + index_radius[k]);
  }
  recursive_add2grid(0, key, delta_x, w, cv_values);
  DEBUG_LOG("add_hill_end");
}

template<int D>
void MetaD_zqc::GH_t1_sparseHash<D>::recursive_add2grid(int dim, CoordKey& key, 
                                                    double* delta_x,
                                                    double w, double* cv_values) {
  // 基准情况：已经确定了所有维度的坐标
    if (dim == cv_dim) {
        // 此时 key 已经填满了各个维度的索引，dx_array 填满了偏移量
        bias_hash[key] += w * gauss_calc(cv_dim, delta_x, sigma);
        return;
    }
    int lower[cv_dim], upper[cv_dim];
    // 递归分支：遍历当前维度的 [lower, upper] 范围
    for (int i = lower[dim]; i <= upper[dim]; ++i) {
        key.c[dim] = i; // 设置当前维度的网格坐标
        
        // 计算当前维度网格中心点到 CV 的距离
        double grid_center = cv_bound[2 * dim] + (i + 0.5) * dcv[dim];
        delta_x[dim] = cv_values[dim] - grid_center;

        // 进入下一维度
        recursive_add2grid(dim + 1, key, delta_x, w, cv_values);
    }
}

// =============================================================================
// get_dVdcv
// =============================================================================
template<int D>
void MetaD_zqc::GH_t1_sparseHash<D>::get_dVdcv(double *cv_values, double *dVdcvs) {
    // 只有主进程计算哈希表，避免所有进程都去查 Map 导致内存开销
    if (lmp->comm->me == 0) {
        int base_coord[D];
        double frac[D];

        // 1. 坐标预处理
        for (int d = 0; d < D; ++d) {
            double f_idx = (cv_values[d] - cv_bound[2*d]) / dcv[d];
            base_coord[d] = static_cast<int>(floor(f_idx));
            frac[d] = f_idx - base_coord[d];

            // 样条插值需要预留足够邻居，base_coord 至少从 1 开始
            if (base_coord[d] < 1) { base_coord[d] = 1; frac[d] = 0.0; }
            if (base_coord[d] >= nbin[d] - 2) { base_coord[d] = nbin[d] - 3; frac[d] = 1.0; }
        }

        // 2. 循环 D 个维度，计算每个方向的偏导
        for (int k = 0; k < D; ++k) {
            dVdcvs[k] = mixed_recursive_logic(0, k, base_coord, frac);
            if (isnan(dVdcvs[k])) { // 检查 NaN
                printf("Warning: dVdcvs[%d] is NaN, setting to 0\n", k);
                dVdcvs[k] = 0.0;
            }
        }
        // printf("dVdcvs computed on rank 0: %g\n", dVdcvs[0]);
    }

    // 3. 必须广播结果！确保所有进程的原子受力完全一致
    MPI_Bcast(dVdcvs, D, MPI_DOUBLE, 0, lmp->world);
}


template<int D>
double MetaD_zqc::GH_t1_sparseHash<D>::mixed_recursive_logic(int current_dim, int target_dim, int* coord, double* frac) {
    // 基准情况：所有维度已确定位置，从哈希表取值
    if (current_dim == D) {
        return get_total_bias(coord);
    }
    double p[4];
    if (current_dim == target_dim) {
        // --- 样条求导维度：采样 4 个点 (p0, p1, p2, p3) ---
        int origin = coord[current_dim];
        for (int i = -1; i <= 2; ++i) {
            coord[current_dim] = origin + i;
            // 边界检查：如果超出 nbin 范围，样条插值通常取边界值
            if (coord[current_dim] < 0) coord[current_dim] = 0;
            if (coord[current_dim] >= nbin[current_dim]) coord[current_dim] = nbin[current_dim] - 1;
            
            p[i+1] = mixed_recursive_logic(current_dim + 1, target_dim, coord, frac);
        }
        coord[current_dim] = origin; // 恢复现场

        // 执行 Catmull-Rom 样条求导公式
        double x = frac[current_dim];
        return ((-0.5 * p[0] + 0.5 * p[2]) + 
                x * (p[0] - 2.5 * p[1] + 2.0 * p[2] - 0.5 * p[3]) + 
                1.5 * x * x * (-p[0] + 3.0 * p[1] - 3.0 * p[2] + p[3])) / dcv[current_dim];

    } else {// --- 非求导维度：由线性插值改为 Catmull-Rom 样条插值 ---
        int origin = coord[current_dim];
        double x = frac[current_dim];

        for (int i = -1; i <= 2; ++i) {
            int temp_coord = origin + i;
            // 边界处理
            if (temp_coord < 0) temp_coord = 0;
            if (temp_coord >= nbin[current_dim]) temp_coord = nbin[current_dim] - 1;
            
            coord[current_dim] = temp_coord;
            p[i+1] = mixed_recursive_logic(current_dim + 1, target_dim, coord, frac);
        }
        coord[current_dim] = origin; // 恢复现场

        // Catmull-Rom 样条求值公式 (不是求导公式)
        return p[1] + 0.5 * x * (
            (p[2] - p[0]) + 
            x * (2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3]) + 
            x * x * (-p[0] + 3.0 * p[1] - 3.0 * p[2] + p[3])
        );
    }
}


// =============================================================================
// get_cvspace_loc
// =============================================================================
template<int D>
void MetaD_zqc::GH_t1_sparseHash<D>::get_cvspace_loc(double* cv_values, 
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
template<int D>
double MetaD_zqc::GH_t1_sparseHash<D>::get_total_bias(int* cvspace_loc){
  CoordKey local_key;
  for (int i = 0; i < cv_dim; ++i) {
      local_key.c[i] = cvspace_loc[i];
  }
  auto it = bias_hash.find(local_key);
  if (it == bias_hash.end()) return 0.0;
  return it->second;
}

template<int D>
double MetaD_zqc::GH_t1_sparseHash<D>::get_bias_energy(double *cv_values){
  // 与 get_dVdcv 同一套样条：target_dim=-1 表示所有维做求值（非求导）
  int base_coord[3];
  double frac[3];
  for (int d = 0; d < D; ++d) {
      double f_idx = (cv_values[d] - cv_bound[2*d]) / dcv[d];
      base_coord[d] = static_cast<int>(floor(f_idx));
      frac[d] = f_idx - base_coord[d];
      if (base_coord[d] < 1) { base_coord[d] = 1; frac[d] = 0.0; }
      if (base_coord[d] >= nbin[d] - 2) { base_coord[d] = nbin[d] - 3; frac[d] = 1.0; }
  }
  return mixed_recursive_logic(0, -1, base_coord, frac);
}


// =============================================================================
// gauss_calc
// =============================================================================
/* helper: 计算高斯 */
template<>
double MetaD_zqc::GH_t1_sparseHash<1>::gauss_calc(int dim, double* dx, double s) {
  return exp(-0.5*(dx[0]*dx[0])/(s*s));
}
  
template<>
double MetaD_zqc::GH_t1_sparseHash<2>::gauss_calc(int dim, double* dx, double s) {
  return exp(-0.5*(dx[0]*dx[0]+dx[1]*dx[1])/(s*s));
}
  
template<>
double MetaD_zqc::GH_t1_sparseHash<3>::gauss_calc(int dim, double* dx, double s) {
  double paw=0.0;
  for (int i=0; i<dim; i++){
    paw += dx[i]*dx[i];
  }
  return exp(-0.5*(paw)/(s*s));
}
