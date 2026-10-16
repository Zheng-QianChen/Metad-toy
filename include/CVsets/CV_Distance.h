#pragma once
#include "fix_crystallize.h"

namespace MetaD_zqc {
// 两个全局原子 ID 的最小镜像距离；值及位移在所有 rank 上相同。
class Distance : public CV {
private:
    LAMMPS_NS::tagint atom_ids_[2];
    int owned_indices_[2] = {-1, -1}; // 本轮 owner 的本地下标，其他 rank 为 -1
    double displacement_[3] = {};   // 第二颗原子相对第一颗原子的最小镜像位移
public:
    static CV* create(LAMMPS_NS::LAMMPS*, LAMMPS_NS::FixMetadynamics*, FILE*, int, char**, int&);
    Distance(LAMMPS_NS::LAMMPS*, LAMMPS_NS::FixMetadynamics*, FILE*,
             LAMMPS_NS::tagint first_id, LAMMPS_NS::tagint second_id);
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
