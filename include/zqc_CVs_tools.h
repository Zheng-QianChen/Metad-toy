#pragma once  // 必须添加这一行
#include "lammps.h"
#include "pair.h"

namespace MetaD_zqc {
    class Averager {
    public:
        virtual void compute(int n, double* arr, double &ave) = 0;
    };
    class KahanAverager : public Averager {
        void compute(int n, double* arr, double &ave) override;
    };
    class CUBAverager : public Averager {
    public:
        CUBAverager();  
        ~CUBAverager();
    private:
        double *d_sum;
        void compute(int n, double* arr, double &ave) override;
    };
}

__global__ void get_envioronment
(
    int cutoff_Natoms, double cutoff_rsq, double box_x, double box_y, double box_z,
    int group_count, int *d_group_indices, LAMMPS_NS::tagint *d_group_numneigh,
    int *d_firstneigh_ptrs, double *d_x_flat,
    double *d_group_dminneigh, int *d_neigh_in_cutoff_r, int *d_neigh_both_in_r_N,
    LAMMPS_NS::tagint *d_calculated_numneigh
);

__global__ void fix_crystallizes_kernel
(
    int cutoff_Natoms, double cutoff_rsq, double box_x, double box_y, double box_z,
    int group_count, int *d_group_indices, LAMMPS_NS::tagint *d_group_numneigh,
    int *d_firstneigh_ptrs, double *d_x_flat,
    double *d_group_dminneigh, int *d_neigh_in_cutoff_r, int *d_neigh_both_in_r_N,
    LAMMPS_NS::tagint *d_calculated_numneigh
);

__global__ void steinhardt_param_calc_kernel_q4(
    int group_count, int cutoff_Natoms,
    int *d_neigh_both_in_r_N, double *d_group_dminneigh,
    double *d_stein_qlm, double *d_stein_Ylm,
    double *d_stein_ql
);

__global__ void dcv_steinhardt_param_calc_kernel_q4(
    int cutoff_Natoms, 
    int group_count, int groupbit, int *d_mask,
    LAMMPS_NS::tagint *d_group_indices, LAMMPS_NS::tagint *calculated_numneigh, 
    int *d_neigh_both_in_r_N, double *d_group_dminneigh,
    double *d_stein_qlm, double *d_stein_Ylm, double *d_stein_ql,
    double *d_dYlm_dr,double *d_dcvdx
);

__global__ void steinhardt_param_calc_kernel_q6(
    int group_count, int cutoff_Natoms,
    int *d_neigh_both_in_r_N, double *d_group_dminneigh,
    double *d_stein_qlm, double *d_stein_Ylm,
    double *d_stein_ql
);

__global__ void dcv_steinhardt_param_calc_kernel_q6(
    int cutoff_Natoms, 
    int group_count, int groupbit, int *d_mask,
    LAMMPS_NS::tagint *d_group_indices, LAMMPS_NS::tagint *calculated_numneigh, 
    int *d_neigh_both_in_r_N, double *d_group_dminneigh,
    double *d_stein_qlm, double *d_stein_Ylm, double *d_stein_ql,
    double *d_dYlm_dr,double *d_dcvdx
);

