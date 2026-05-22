#pragma once  // 必须添加这一行
#ifndef LMP_FIX_METADYNAMICS_H
#define LMP_FIX_METADYNAMICS_H
// #ifndef _GLIBCXX_USE_CXX11_ABI
// #error "_GLIBCXX_USE_CXX11_ABI not defined"
// #endif
// #if _GLIBCXX_USE_CXX11_ABI != 1
// #error "_GLIBCXX_USE_CXX11_ABI != 1"
// #endif

#include "fix.h"

#include <map> 
#include <string>
#include <vector>

namespace LAMMPS_NS {
    class FixMetadynamics;
}

namespace MetaD_zqc {
  class CV {
    protected:
      FILE *f_check;
      LAMMPS_NS::LAMMPS *lmp;
      double cv_value;
      double *dcvdx;
      double dVdcv;
    public:
      CV(LAMMPS_NS::LAMMPS *lmp, FILE* f_check) 
          : lmp(lmp), f_check(f_check) {}
      virtual ~CV() {
          if (dcvdx != nullptr) {
              delete[] dcvdx; 
              dcvdx = nullptr;
          }
      }
      typedef double (CV::*CV_Calculation)();
      typedef void (CV::*CV_BiasForce)(double);
      virtual CV_Calculation set_CV_calculate(std::string func_name) = 0;
      virtual CV_BiasForce set_CV_bias_force(std::string func_name) = 0;
      virtual void base_calc() = 0; // 计算 CV 值
      // virtual double compute_cv() = 0; // 计算 CV 值
      // virtual void bias_force(double dVdcv) = 0; // 计算梯度
      virtual void summary(FILE* f) = 0;
      // virtual void get_dcvdx(double cv_value, double *dcvdx) = 0;
      virtual bool need_forward_comm(){ return false; } // 是否需要跨进程同步 Ghost 属性
      virtual int get_comm_forward_bytes(){ return 0; } // 每个原子需要同步多少个 bytes
      virtual int get_comm_reverse_bytes(){ return 0; } // 每个原子需要同步多少个 bytes
      virtual int pack_comm_ubuf(int n, int *list, double *u_buf, int slot_offset) { return 0; } // 具体 CV 自己的打包逻辑
      virtual void unpack_comm_ubuf(int n, int first, double *u_buf, int slot_offset) {} // 具体 CV 自己的解包逻辑
  };

  class MetaDimensionManager;

  class Gaussian_Hill_Base;
  // class GH_t0_uniformGrid;

  class CVFactory {
    typedef CV* (*CreatorFunc)(LAMMPS_NS::LAMMPS*, LAMMPS_NS::FixMetadynamics *,
                   int, char**, int&, FILE*);
    private:
      // 用静态方法包裹 map，确保初始化顺序安全
      static std::map<std::string, CreatorFunc>& get_registry();
    public:
      CVFactory() {};
      CVFactory(const CVFactory&) = delete;
      CVFactory& operator=(const CVFactory&) = delete;
      static void register_cv(std::string name, CreatorFunc func);
      static CV* create(std::string name, LAMMPS_NS::LAMMPS* lmp, LAMMPS_NS::FixMetadynamics *Fixmetad, 
                        int narg, char** arg, int &i, FILE *f_check);
  };
}

namespace LAMMPS_NS {
  class FixMetadynamics : public Fix {
  public:
    FixMetadynamics(class LAMMPS *, int, char **);
    ~FixMetadynamics() override;
    int setmask() override;
    void init() override;
    void init_list(int id, NeighList *ptr) override;
    int get_comm_forward_bytes();
    int get_comm_reverse_bytes();
    int pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/) override;
    void unpack_forward_comm(int n, int first, double *buf) override;
    void post_force(int) override;
    void add_hill(double *);
    void checkmax(double *cv, double *cv_max);
    void get_dVdcv(double *cv_values, double *dVdcvs);
    int get_cv_dim() const;
    void get_cvspace_loc(double* , int* );
    double get_total_bias(int* );
    NeighList *listhalf, *listfull;
  private:
    // double sigma, height0, biasf, kBT;
    // double KB;
    // int WellT_bool;
    // double *cv_bound, *dcv;
    // int *nbin, *cvspace_loc;
    // int grid_size;
    // int continue_from_file;
    // double *bias_grid;
    int cv_dim,nbin_num;
    int comm_forward,comm_reverse;
    MetaD_zqc::Gaussian_Hill_Base *p_gaussian;
    int pace,rec_pace;
    bool first_run;
    // double cv1_min, cv1_max, cv2_min, cv2_max;
    double *cv_values, *cv_history, *dVdcvs;
    struct DimConfig {
      std::string name;
      std::string func;
    };
    std::map<std::string, MetaD_zqc::CV*> cal_registry;
    MetaD_zqc::MetaDimensionManager *cv_configs;
    // FILE *file;
    // FILE *f_hills;
    FILE *f_check=NULL;
    std::string rec_file_name;
    FILE *rec_file=NULL;
  };
}


#endif