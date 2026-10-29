#pragma once
#include "zqc_cv.h"
#include "zqc_cuda_execution.cuh"
#include "zqc_gpu_buffer.cuh"
#include "zqc_switch_function_device.cuh"

namespace MetaD_zqc {
// Soft coordination or a Gaussian coordination score. Global reductions belong
// to Graph. RNN diagnostics are deliberately not exposed as differentiable leaves.
class LocalGeom : public CV {
public:
    static CV* create(CVContext&, int, char**, int&);
    LocalGeom(CVContext&, SelectionId, double, double, int, SwitchFunctionRequest,
              bool, double, double);
    SelectionId graph_selection() const override { return selection_; }
    CV_Calculation set_CV_calculate(std::string) override;
    CV_BiasForce set_CV_bias_force(std::string) override;
    void summary(FILE*) override;
    void bind_assembly(std::shared_ptr<const CVAssembly>) override;
    void prepare_graph_evaluation() override { ready_ = false; }
    void base_calc() override;
    void assemble_values(ArrayView<double>) override;
    void apply_adjoint(ArrayView<const double>) override;
    ViewReadiness device_readiness() const override { return execution_.ready(); }
    ArrayView<const double> peratom_values(const std::string&) override;
private:
    SelectionId selection_;
    int neighbor_handle_, neighbor_type_, total_ = 0;
    double cutoff_, epsilon_, center_, sigma_;
    bool tet_, ready_ = false;
    SwitchFunctionRequest radial_;
    std::uint64_t evaluation_ = 0, checked_generation_ = 0;
    const CVAssembly* checked_assembly_ = nullptr;
    NeighborListView neighbors_;
    ArrayView<const double> positions_;
    ArrayView<const int> types_;
    CudaExecutionStream execution_;
    GpuBuffer<double> values_, coordination_, rnn_, valid_, adjoints_, forces_, virials_;
    std::vector<double> diagnostic_;
    void check_values() const;
};
}
