#pragma once  // 必须添加这一行
#include <cstring>
#include "lammps.h"
#include "pair.h"
#include "zqc_debug.h"

namespace MetaD_zqc {
    class Averager {
    public:
        virtual void compute(int n, int glob_N, double* arr, int* mask, int groupbit, double &ave) = 0;
    };
    class KahanAverager : public Averager {
        void compute(int n, int glob_N, double* arr, int* mask, int groupbit, double &ave) override;
    };
    class CUBAverager : public Averager {
    public:
        CUBAverager();  
        ~CUBAverager();
    private:
        double *d_sum;
        void compute(int n, int glob_N, double* arr, int* mask, int groupbit, double &ave) override;
    };
}

template <typename T>
struct GpuBuffer {
    char name[32];
    T* ptr = nullptr;
    size_t capacity = 0;
    // 构造函数初始化
    GpuBuffer() {
        std::strncpy(name, "Unknown", 31);
        name[31] = '\0';
    }
    // 提供一个设置名字的方法
    void set_name(const char* n) {
        if (n) {
            std::strncpy(name, n, 31);
            name[31] = '\0';
        }
    }
    void grow_to(size_t needed, FILE* f_check, const char* code_file, int code_line) {
        // if (true) {
        if (needed > capacity) {
            printf("DEBUG: GPU Buffer [%s] expanded from %zu to %zu\n", name, capacity, (size_t)(needed * 1.1) + 100);
            printf("in %s : %d\n", code_file, code_line);
            printf("the past location is %p\n", ptr);
            cudaError_t sync_err = cudaDeviceSynchronize();
            if (ptr) SAFE_CUDA_FREE_NOFILE(ptr, code_file, code_line);
            // 额外多给 10% 冗余，防止频繁 Malloc
            capacity = (size_t)(needed * 1.1) + 100;
            SAFE_CUDA_MALLOC_NOLMP(&ptr, capacity * sizeof(T), f_check);
            sync_err = cudaDeviceSynchronize();
            cudaMemset(ptr, 0, capacity * sizeof(T));
            printf("now location is %p\n", ptr);
        }
    }
    ~GpuBuffer() { if (ptr) SAFE_CUDA_FREE_NOFILE(ptr, __FILE__, __LINE__); }
};

__global__ void get_environment(
    int cutoff_Natoms, double cutoff_rsq, double box_x, double box_y, double box_z,
    int group_count, int *d_group_indices, LAMMPS_NS::tagint *d_group_numneigh,
    int *d_firstneigh_ptrs, double *d_x_flat,
    double *d_group_dminneigh, int *d_neigh_in_cutoff_r, int *d_neigh_both_in_r_N,
    LAMMPS_NS::tagint *d_calculated_numneigh
);

__global__ void fix_crystallizes_kernel(
    int cutoff_Natoms, double cutoff_rsq, double box_x, double box_y, double box_z,
    int group_count, int *d_group_indices, LAMMPS_NS::tagint *d_group_numneigh,
    int *d_firstneigh_ptrs, double *d_x_flat,
    double *d_group_dminneigh, int *d_neigh_in_cutoff_r, int *d_neigh_both_in_r_N,
    LAMMPS_NS::tagint *d_calculated_numneigh
);


template <int L>
__global__ void steinhardt_param_calc_kernel(
    int group_count, int cutoff_Natoms, int *d_group_indices,
    int *d_neigh_both_in_r_N, double *d_group_dminneigh,
    double *d_stein_qlm, double *d_stein_Ylm, double *d_stein_ql
);


template <int L>
__global__ void dcv_steinhardt_param_calc_kernel(
    int cutoff_Natoms, int group_count, int groupbit, int all_count, 
    int *d_mask, LAMMPS_NS::tagint *d_group_indices, LAMMPS_NS::tagint *calculated_numneigh, 
    int *d_neigh_both_in_r_N, double *d_group_dminneigh,
    double *d_stein_qlm, double *d_stein_Ylm, double *d_stein_ql,
    double *d_dYlm_dr, double *d_dcvdx
);


__global__ void steinhardt_param_calc_LOCAL_kernel(int group_count, int cutoff_Natoms,
                    int stein_l, int groupbit,
                    int *d_mask, LAMMPS_NS::tagint *d_group_indices,
                    LAMMPS_NS::tagint *calculated_numneigh, 
                    int *d_neigh_both_in_r_N,
                    double *d_stein_qlm, double *d_stein_LQlm,
                    double *d_stein_ql);