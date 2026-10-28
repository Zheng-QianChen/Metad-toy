#pragma once
#include "zqc_engine.h"
#include <cstdio>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace MetaD_zqc {
template<class T> struct GpuBuffer;
class CVAssembly;
class SwitchFunction;

// 构建 CV 时借用的上下文。引擎/径向开关由上层管理，CV 保存所需引用和 Logger 副本。
// log_file 仅用于现有 summary(FILE*) 文本接口；诊断消息通过 logger 回传。
struct CVContext {
    EngineAdapter& engine;
    std::map<std::string,SwitchFunction*>& switches;
    Logger logger;
    FILE* log_file = nullptr;
    SwitchFunction* switching_function(const std::string& name) const {
        const auto found = switches.find(name);
        if (found == switches.end()) throw std::invalid_argument("Unknown radial switch: "+name);
        return found->second;
    }
};
  class CV_info {
    protected:
      FILE *f_check;
      EngineAdapter& engine;
      Logger logger;
    public:
      explicit CV_info(CVContext& context)
          : f_check(context.log_file), engine(context.engine), logger(context.logger) {}
      ~CV_info(){};

      // GPU buffer manager
      template<typename T>
      void register_buffer(GpuBuffer<T>& buf, const char* name, int size = 0) {
          buf.set_name(name, logger);
          if (size > 0) {
              buf.grow_to(size, __FILE__, __LINE__);
          }
      }
  };

  class CV : public AtomCommunication {
    protected:
      FILE *f_check;
      EngineAdapter& engine;
      Logger logger;
      double cv_value = 0.0;
      double *dcvdx = nullptr;
      double dVdcv = 0.0;
      std::shared_ptr<const CVAssembly> assembly_; // 外部筛好的 owned/ghost 编号和 owned Leaf 写入位置
      // double virial[6]={0.0};
    public:
      explicit CV(CVContext& context)
          : f_check(context.log_file), engine(context.engine), logger(context.logger) {}
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
      // 最近已提交工作的完成事件；装配后的 Leaf 由 Graph 在自己的流上等待。
      virtual ViewReadiness device_readiness() const { return {}; }
      virtual void base_calc() = 0; // 计算 CV 值
      // virtual double compute_cv() = 0; // 计算 CV 值
      // virtual void bias_force(double dVdcv) = 0; // 计算梯度
      virtual void summary(FILE* f) = 0;
      // virtual void get_dcvdx(double cv_value, double *dcvdx) = 0;

      inline void set_cv_value(double value) noexcept {cv_value = value;}

      // GPU buffer manager
      template<typename T>
      void register_buffer(GpuBuffer<T>& buf, const char* name, int size = 0) {
          buf.set_name(name, logger);
          if (size > 0) {
              buf.grow_to(size, __FILE__, __LINE__);
          }
      }

      // 由管理层绑定装配表；base_calc/assemble_values 顺序提交到环境流。
      // 目标显存由 Leaf/调用方持有；装配后取得 device_readiness()，在消费流上等待。
      virtual void bind_assembly(std::shared_ptr<const CVAssembly> assembly) {
          assembly_ = std::move(assembly);
      }
      // -2 表示未实现原始值接口；all_atoms=-1 是有效选择集。
      virtual SelectionId graph_selection() const { return -2; }
      // Graph 每轮先准备全部输入，再开始 base_calc；共享环境不能在逐个计算时失效。
      virtual void prepare_graph_evaluation() {}
      const CVAssembly* assembly() const { return assembly_.get(); }
      virtual void assemble_values(ArrayView<double>) {
          throw std::logic_error("CV: value assembly is not implemented");
      }
      // 输入为 Leaf 排列的 dV/dcv_i；各 rank 按相同 CV 顺序调用，包含空组 rank。
      virtual void apply_adjoint(ArrayView<const double>) {
          throw std::logic_error("CV: raw adjoint is not implemented");
      }

      // 全局标量 CV 在 base_calc 中完成跨 rank 汇总；Graph 的 Host Leaf 保存同一数值。
      virtual bool supports_global_graph() const { return false; }
      virtual double graph_global_value() const {
          throw std::logic_error("CV: global graph value is not implemented");
      }
      // seed 为每个 rank 上相同的 dV/dcv；CV 将力和 virial 写入各自 owned 原子。
      virtual void apply_global_adjoint(double seed) {
          throw std::logic_error("CV: global graph adjoint is not implemented");
      }

      // 返回该 CV 是否有“每个原子”的数据
      virtual bool has_per_atom_data() { return false; }
      virtual double* get_per_atom_data() { return nullptr; }
      virtual std::string get_per_atom_name() { return ""; }

      // 返回该 CV 是否有“整个体系”的标量数据
      virtual bool has_global_data() { return false; }
      virtual double get_global_data() { return 0.0; }
      virtual std::string get_global_name() { return ""; }

      // Compute接口
      // 返回只读逐原子诊断视图，行号为 localid；所有者为当前 CV，读取方执行等待钩子。
      virtual ArrayView<const double> peratom_values(const std::string& name) {
          throw std::invalid_argument("CV: unknown per-atom property "+name);
      }
  };


class CVFactory {
    using CreatorFunc = CV* (*)(CVContext&, int, char**, int&);
    static std::map<std::string,CreatorFunc>& get_registry();
public:
    static void register_cv(std::string name, CreatorFunc creator);
    static CV* create(std::string name, CVContext&, int, char**, int&);
};
} // namespace MetaD_zqc
