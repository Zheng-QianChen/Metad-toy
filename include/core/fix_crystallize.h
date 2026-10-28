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
#include "../utils/zqc_engine.h"
#include "../utils/zqc_cv.h"
#include "zqc_lammps_diagnostics.h"

#include <memory>
#include <set>
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


  class MetaDimensionManager;

  class BiasSession;
  class FrozenSharedBias;
  // class GH_t0_uniformGrid;

  class SwitchFunction;


}

namespace LAMMPS_NS {
  class FixMetadynamics : public Fix {
  public:
    MetaD_zqc::NeighHub neigh_hub;
    // 适配器持有一轮 CV 求值共享的数组暂存和邻居请求，寿命覆盖所有 CV。
    std::unique_ptr<MetaD_zqc::EngineAdapter> engine_adapter;
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
    void setup_post_neighbor() override;
    void post_neighbor() override;
    void min_post_neighbor() override;

    // communication
    // 所有 rank 按相同 CV 顺序调用，包含本地中心组为空的 rank。
    void communicate_forward(MetaD_zqc::AtomCommunication& payload);
    void communicate_reverse(MetaD_zqc::AtomCommunication& payload);
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
    // 一次阻塞通信期间，由四个 LAMMPS 回调共同借用当前数据包。
    struct CVCommunication {
      MetaD_zqc::AtomCommunication* payload = nullptr;
      int width = 0;
      CommDirection direction = CommDirection::None;
    } active_comm;
    void init_cv_communication();
    void communicate(MetaD_zqc::AtomCommunication& payload, CommDirection direction);
    int comm_buffer_size(int n, CommDirection direction) const;
    void evaluate_bias(int vflag, bool advance_history);
    int cv_dim,nbin_num;
    MetaD_zqc::BiasSession *p_gaussian;
    std::unique_ptr<MetaD_zqc::FrozenSharedBias> shared_frozen_bias;
    MetaD_zqc::SelectionId shared_frozen_selection = MetaD_zqc::all_atoms;
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
    // compute::init 先于 fix::init：每次运行收集当前存在的输出请求，随后替换活动集合。
    std::set<MetaD_zqc::CV*> pending_output_cvs, output_cvs;
    std::map<std::string,std::vector<double>> peratom_output; // compute 的 Host 连续列
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
