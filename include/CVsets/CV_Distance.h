#pragma once  // 必须添加这一行

#include "fix_crystallize.h"
#include "lammps.h"
#include "pair.h"
#include "neigh_request.h"
#include "neigh_list.h"
#include "zqc_CVs_tools.h"

namespace MetaD_zqc {
    // atoms distance
    class Distance : public CV {
    private:
        LAMMPS_NS::bigint atom_id1, atom_id2;
        bool pbc_x, pbc_y, pbc_z;
        double box_x, box_y, box_z;
        double dx, dy, dz;
    public:
        static CV* create(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::FixMetadynamics *Fixmetad,
                         FILE *f_check, int narg, char **arg, int &i);
        Distance(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check, LAMMPS_NS::bigint id1, LAMMPS_NS::bigint id2);
        ~Distance() override;
        CV_Calculation set_CV_calculate(std::string func_name) override;
        CV_BiasForce set_CV_bias_force(std::string func_name) override;
        void base_calc(); // 计算 CV 值
        double compute_cv();
        void bias_force(double dVdcv);
        void get_dcvdx(double cv_value, double *dcvdx);
        void summary(FILE* f) override;
        void delta_x();
    };
}