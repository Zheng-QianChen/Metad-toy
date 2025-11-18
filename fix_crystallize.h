#ifndef LMP_FIX_METADYNAMICS_H
#define LMP_FIX_METADYNAMICS_H
// #ifndef _GLIBCXX_USE_CXX11_ABI
// #error "_GLIBCXX_USE_CXX11_ABI not defined"
// #endif
// #if _GLIBCXX_USE_CXX11_ABI != 1
// #error "_GLIBCXX_USE_CXX11_ABI != 1"
// #endif

#include "fix.h"

namespace LAMMPS_NS {

class FixMetadynamics : public Fix {
 public:
  FixMetadynamics(class LAMMPS *, int, char **);
  ~FixMetadynamics() override;
  int setmask() override;
  void init() override;
  void post_force(int) override;
  void post_force_cus(double cv, double dVdcv);
  void post_force_r(double cv,double dVdcv);
  void add_hill(double *, double);
  void grid_gradient(double *, double *);
  void checkmax(double *cv, double *cv_max);
  void get_dcvdx(double cv, double *dcvdx);
  int get_cv_dim() const;
 private:
  int icv1, icv2;            // 原子在 x,y 方向的索引
  double sigma, height0, biasf, kBT;
  int pace;
  int nadd;                  // 已添加的高斯数
  bool first_run;
  std::vector<double> cv_hist;
  std::vector<double> height_hist;
  // double cv1_min, cv1_max, cv2_min, cv2_max;
  int cv_dim,grid_size,nbin_num;
  int* nbin;
  double *bias_grid, *cv_bound, *cv, *cv_history, *dVdcv;
  int continue_from_file;
  FILE *file;
  FILE *f_hills, *f_check;
};

}
#endif