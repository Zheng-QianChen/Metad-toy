#ifndef LMP_FIX_METADYNAMICS_H
#define LMP_FIX_METADYNAMICS_H
// #ifndef _GLIBCXX_USE_CXX11_ABI
// #error "_GLIBCXX_USE_CXX11_ABI not defined"
// #endif
// #if _GLIBCXX_USE_CXX11_ABI != 1
// #error "_GLIBCXX_USE_CXX11_ABI != 1"
// #endif

#include "fix.h"

namespace MetaD_zqc {
  class CV {
  public:
      virtual ~CV() {}
      virtual double compute_cv() = 0; // 计算 CV 值
      virtual void compute_grad(double dVdcv) = 0; // 计算梯度
      virtual void summary(FILE* f) = 0;
      virtual void get_dcvdx(double cv_value, double *dcvdx) = 0;
  };
}

namespace MetaD_zqc {
    class Distance : public CV {
    private:
        FILE* f_check;
        LAMMPS_NS::bigint atom_id1, atom_id2;
        LAMMPS_NS::LAMMPS *lmp;
        double cv_value;
        double *dcvdx;
        double dVdcv;
        bool pbc_x, pbc_y, pbc_z;
        double box_x, box_y, box_z;
        double dx, dy, dz;
    public:
        Distance(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::bigint id1, LAMMPS_NS::bigint id2, FILE *f_check);
        ~Distance() override;
        double compute_cv() override;
        void compute_grad(double dVdcv) override;
        void summary(FILE* f) override;
        void get_dcvdx(double cv_value, double *dcvdx) override;
        void delta_x();
    };
}

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
    void grid_gradient(int *, double *);
    void checkmax(double *cv, double *cv_max);
    void get_dcvdx(double cv, double *dcvdx);
    int get_cv_dim() const;
    double get_cvspace_loc(double* , int* );
    double get_total_bias(int* );
  private:
    double sigma, height0, biasf, kBT;
    int pace;
    bool first_run;
    double KB;
    std::vector<double> cv_hist;
    std::vector<double> height_hist;
    // double cv1_min, cv1_max, cv2_min, cv2_max;
    int cv_dim,grid_size,nbin_num;
    int *nbin, *cvspace_loc;
    double *bias_grid, *cv_bound, *cv_values, *cv_history, *dVdcvs, *dcv;
    std::vector<MetaD_zqc::CV*> cv;
    int continue_from_file, WellT_bool;
    // FILE *file;
    FILE *f_hills, *f_check;
  };
}


#endif