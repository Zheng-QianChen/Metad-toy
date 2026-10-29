#pragma once
#include "zqc_hill_bias.h"
#include "zqc_bias_grid.h"
#include <functional>
#include <fstream>
#include <memory>
#include <string>

namespace MetaD_zqc {
enum class BiasBackend { Auto, Grid, SparseGrid, Hills, PBMetaD };
struct BiasSettings {
    BiasBackend backend = BiasBackend::Auto;
    HillBiasSettings hills;
    std::vector<GridAxis> grid_axes;
    std::int64_t pace = 100;
    std::string history_path = "HILLS";
    bool restart = false;
    unsigned history_workers = 1;
    bool pb_sum_normalization = false; // MEAN 与 PLUMED 2.10 实际求值一致。
    PBKernel pb_kernel = PBKernel::Plumed210;
};
// 宿主提供单位已统一的参数和 root 广播。回调捕获对象必须覆盖 session 生命周期。
// 单进程 GPUMD 可保留默认 root=true、空广播；MPI 由各宿主绑定自己的 communicator。
struct BiasHost {
    bool root = true;
    std::function<void(double*,std::size_t)> broadcast;
    std::function<void(const std::string&)> log;
};
// initialize 对应宿主 init；advance 仅在真实积分步调用；evaluate 也可用于 setup/compute。
// 返回包布局 [V,dV/ds0,...]；异常交给宿主转换为自身报错机制。
class BiasSession {
public:
    BiasSession(BiasSettings settings, BiasHost host = {});
    void initialize(std::int64_t current_step);
    bool advance(std::int64_t step, const double* coordinates);
    double evaluate(const double* coordinates, double* gradient = nullptr);
    // 主流程以外的固定偏势测试入口；同一步的多个网格 hill 保留原有语义。
    void deposit_now(std::int64_t step, const double* coordinates);
    BiasBackend backend() const { return settings_.backend; }
    std::size_t dimension() const { return history_.dimension(); }
    const HillBias& history() const { return history_; }
    static const char* name(BiasBackend backend);
private:
    BiasSettings settings_;
    BiasHost host_;
    HillBias history_;
    std::unique_ptr<BiasGrid> grid_;
    std::vector<double> packet_;
    std::ofstream output_;
    bool initialized_ = false;
    bool legacy_history_ = false;
    void load_legacy(std::istream&);
    void write_record();
    double local_evaluate(const double*,double*);
};
} // namespace MetaD_zqc

