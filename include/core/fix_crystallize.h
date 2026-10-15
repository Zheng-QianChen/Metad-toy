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

#include <memory>
#include <stdexcept>
#include <utility>
#include <map> 
#include <string>
#include <vector>

namespace LAMMPS_NS {
    class FixMetadynamics;
}

namespace MetaD_zqc {
  template<typename T> struct GpuBuffer;
  class CVAssembly;
  class CVGraphExecution;

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
      double cv_value = 0.0;
      double *dcvdx = nullptr;
      double dVdcv = 0.0;
      std::shared_ptr<const CVAssembly> assembly_; // 外部筛好的 owned/ghost 编号和 owned Leaf 写入位置
      // double virial[6]={0.0};
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

      inline void set_cv_value(double value) noexcept {cv_value = value;}

      // GPU buffer manager
      template<typename T>
      void register_buffer(GpuBuffer<T>& buf, const char* name, int size = 0) {
          buf.set_name(name, f_check, error, lmp);
          if (size > 0) {
              buf.grow_to(size, __FILE__, __LINE__);
          }
      }

      // 由管理层绑定装配表；装配前先完成 base_calc()，目标显存由 Leaf/调用方持有。
      virtual void bind_assembly(std::shared_ptr<const CVAssembly> assembly) {
          assembly_ = std::move(assembly);
      }
      // 负值表示尚未实现原始值/梯度接口；Graph 在运行前拒绝不支持的 CV。
      virtual int graph_groupbit() const { return -1; }
      // Graph 每轮先准备全部输入，再开始 base_calc；共享环境不能在逐个计算时失效。
      virtual void prepare_graph_evaluation() {}
      const CVAssembly* assembly() const { return assembly_.get(); }
      virtual void assemble_values(GpuBuffer<double>&) {
          throw std::logic_error("CV: value assembly is not implemented");
      }
      // 输入为 Leaf 排列的 dV/dcv_i；各 rank 按相同 CV 顺序调用，包含空组 rank。
      virtual void apply_adjoint(const GpuBuffer<double>&) {
          throw std::logic_error("CV: raw adjoint is not implemented");
      }

      // 通信宽度为每原子的 double 数；0 表示该方向无需通信，各 rank 使用同一宽度。
      virtual int get_comm_forward_width() const { return 0; }
      virtual int get_comm_reverse_width() const { return 0; }
      // pack 返回每原子写入的 double 数，由 Fix 校验后换算为总发送长度。
      // forward 将 owned 属性写入 ghost；reverse 将 ghost 贡献累加回 owner。
      virtual int pack_comm_forward_ubuf(int n, int *list, double *buf, int offset, int stride) { return 0; }
      virtual void unpack_comm_forward_ubuf(int n, int first, double *buf, int offset, int stride) {}
      virtual int pack_comm_reverse_ubuf(int n, int first, double *buf, int offset, int stride) { return 0; }
      virtual void unpack_comm_reverse_ubuf(int n, int *list, double *buf, int offset, int stride) {}

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
    void setup(int vflag) override;
    void init_list(int id, NeighList *ptr) override;

    // communication
    // 所有 rank 按相同 CV 顺序调用，包含本地中心组为空的 rank。
    void communicate_forward(MetaD_zqc::CV& cv);
    void communicate_reverse(MetaD_zqc::CV& cv);
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
    double *a_virial;
  private:
    enum class CommDirection { None, Forward, Reverse };
    // 一次阻塞通信期间，由四个 LAMMPS 回调共同借用当前 CV。
    struct CVCommunication {
      MetaD_zqc::CV* cv = nullptr;
      int width = 0;
      CommDirection direction = CommDirection::None;
    } active_comm;
    void init_cv_communication();
    void communicate(MetaD_zqc::CV& cv, CommDirection direction);
    int comm_buffer_size(int n, CommDirection direction) const;
    void evaluate_bias(int vflag, bool advance_history);
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
    std::unique_ptr<MetaD_zqc::CVGraphExecution> graph_execution;
    // FILE *file;
    // FILE *f_hills;
    FILE *f_check=NULL;
    std::string rec_file_name;
    FILE *rec_file=NULL;
  };
}


#endif