#pragma once
#include "zqc_cv.h"
#include "zqc_distance.h"

namespace MetaD_zqc {
// 两个全局原子 ID 的最小镜像距离；值及位移在所有 rank 上相同。
// atom_ids_ 为稳定 AtomId；计算状态由 DistanceCalculation 持有，仅借用本轮 EngineAdapter。
class Distance : public CV {
private:
    AtomId atom_ids_[2];
    DistanceCalculation calculation_;
public:
    static CV* create(CVContext& context, int narg, char** arg, int& index);
    Distance(CVContext& context, AtomId first_id, AtomId second_id);
    CV_Calculation set_CV_calculate(std::string method) override;
    CV_BiasForce set_CV_bias_force(std::string method) override;
    void base_calc() override;
    double compute_cv();
    void bias_force(double seed);
    void summary(FILE*) override;

    bool supports_global_graph() const override { return true; }
    double graph_global_value() const override { return cv_value; }
    void apply_global_adjoint(double seed) override { bias_force(seed); }
};
}
