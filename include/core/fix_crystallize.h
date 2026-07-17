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
#include "neigh_hub.h"

#include <map> 
#include <string>
#include <vector>

namespace LAMMPS_NS {
    class FixMetadynamics;
}

namespace MetaD_zqc {
  template<typename T> struct GpuBuffer;

  class CV_info {
    protected:
      FILE *f_check;
      LAMMPS_NS::LAMMPS *lmp;
      LAMMPS_NS::Error *error = nullptr;
      LAMMPS_NS::FixMetadynamics *Fixmetad = nullptr;
    public:
      CV_info(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check)
          : lmp(lmp), f_check(f_check), Fixmetad(Fixmetad) {error=lmp->error; }
      ~CV_info(){};

      // GPU buffer manager
      template<typename T>
      void register_buffer(GpuBuffer<T>& buf, const char* name, int size = 0) {
          error = lmp->error;
          buf.set_name(name, this->f_check, this->error, this->lmp);
          if (size > 0) {
              buf.grow_to(size, __FILE__, __LINE__);
          }
      }
  };

  class CV {
    protected:
      FILE *f_check;
      LAMMPS_NS::LAMMPS *lmp;
      LAMMPS_NS::Error *error = nullptr;
      LAMMPS_NS::FixMetadynamics *Fixmetad = nullptr;
      bool comm_mode=false;               // 当前正在处于哪种通信状态
      double cv_value;
      double *dcvdx;
      double dVdcv;
    public:
      CV(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check)
          : lmp(lmp), f_check(f_check), Fixmetad(Fixmetad) {error=lmp->error; }
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

      // GPU buffer manager
      template<typename T>
      void register_buffer(GpuBuffer<T>& buf, const char* name, int size = 0) {
          buf.set_name(name, f_check, error, lmp);
          if (size > 0) {
              buf.grow_to(size, __FILE__, __LINE__);
          }
      }

      // 通信
      virtual bool need_forward_comm(){ return false; } // 是否需要跨进程同步 Ghost 属性
      virtual int get_comm_forward_bytes(){ return 0; } // 每个原子需要同步多少个 bytes
      virtual int pack_comm_forward_ubuf(int n, int *list, double *u_buf, int slot_offset, int comm_forward) { return 0; } // 具体 CV 自己的打包逻辑
      virtual void unpack_comm_forward_ubuf(int n, int first, double *u_buf, int slot_offset, int comm_forward) {} // 具体 CV 自己的解包逻辑
      
      virtual bool need_reverse_comm(){ return false; } // 是否需要跨进程同步 Ghost 属性
      virtual int get_comm_reverse_bytes(){ return 0; } // 每个原子需要同步多少个 bytes
      virtual int pack_comm_reverse_ubuf(int n, int first, double *u_buf, int slot_offset, int comm_forward) { return 0; } // 具体 CV 自己的打包逻辑
      virtual void unpack_comm_reverse_ubuf(int n, int *list, double *u_buf, int slot_offset, int comm_forward) {} // 具体 CV 自己的解包逻辑

      // 返回该 CV 是否有“每个原子”的数据
      virtual bool has_per_atom_data() { return false; }
      virtual double* get_per_atom_data() { return nullptr; }
      virtual std::string get_per_atom_name() { return ""; }

      // 返回该 CV 是否有“整个体系”的标量数据
      virtual bool has_global_data() { return false; }
      virtual double get_global_data() { return 0.0; }
      virtual std::string get_global_name() { return ""; }

      // Compute接口
      virtual double* get_peratom_ptr(const std::string &prop_name) { return nullptr; }
  };

  class MetaDimensionManager;

  class Gaussian_Hill_Base;
  // class GH_t0_uniformGrid;

  class SwitchFunction;

  class CVFactory {
    typedef CV* (*CreatorFunc)(LAMMPS_NS::LAMMPS*,
                    LAMMPS_NS::FixMetadynamics *, FILE*, 
                   int, char**, int&);
    private:
      // 用静态方法包裹 map，确保初始化顺序安全
      static std::map<std::string, CreatorFunc>& get_registry();
    public:
      CVFactory() {};
      CVFactory(const CVFactory&) = delete;
      CVFactory& operator=(const CVFactory&) = delete;
      static void register_cv(std::string name, CreatorFunc func);
      static CV* create(std::string name, LAMMPS_NS::LAMMPS* lmp, 
                        LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check, 
                        int narg, char** arg, int &i);
  };
}

namespace LAMMPS_NS {
  class FixMetadynamics : public Fix {
  public:
    MetaD_zqc::NeighHub neigh_hub;
    std::map<std::string, MetaD_zqc::SwitchFunction*> sw_registry;

    // return values for compute_scalar and compute_vector
    double compute_scalar() override;
    double compute_vector(int n) override;

    // set
    FixMetadynamics(class LAMMPS *, int, char **);
    ~FixMetadynamics() override;
    int setmask() override;
    void init() override;
    void init_list(int id, NeighList *ptr) override;

    // communication
    int get_comm_forward_bytes();
    int get_comm_reverse_bytes();
    int pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/) override;
    void unpack_forward_comm(int n, int first, double *buf) override;
    int pack_reverse_comm(int n, int first, double *buf) override;
    void unpack_reverse_comm(int n, int *list, double *buf) override;

    // calculation
    void post_force(int) override;
    void add_hill(double *);
    void get_dVdcv(double *cv_values, double *dVdcvs);
    int get_cv_dim() const;
    void get_cvspace_loc(double* , int* );
    double get_total_bias(int* );
    void *extract(const char *key, int &dim);

    // get_parameters
    MetaD_zqc::SwitchFunction* get_switching_function(const std::string& name) const;
  private:
    int cv_dim,nbin_num;
    MetaD_zqc::Gaussian_Hill_Base *p_gaussian;
    int pace,rec_pace;
    bool first_run;
    // double cv1_min, cv1_max, cv2_min, cv2_max;
    double *cv_values, *cv_history, *dVdcvs;
    double bias_energy;          // 当前步偏置势 V_b，供 thermo energy
    double **f_before_bias;      // post_force 前原子力快照，用于 virial = r⊗Δf
    int max_f_before_bias;
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