#pragma once  // 必须添加这一行

#include "fix_crystallize.h"
#include "lammps.h"
#include "pair.h"
#include "neigh_request.h"
#include "zqc_switch_function.h"
#include "CV_Stru_factor.h"


namespace MetaD_zqc {

    class Weighted_chem_pair: public CV{
        friend class Stru_fact_env;
    private:
        // FILE *f_check = nullptr;
        LAMMPS_NS::Error *error = nullptr;
        LAMMPS_NS::FixMetadynamics *Fixmetad = nullptr;
        MetaD_zqc::Stru_fact_chem_env* my_env; // 专有类型转换指针
        int d_block_size;         // use it to change the GPU set
        int GPU_number;
        int block_num;
        int neighbor_type = 0;
        LAMMPS_NS::tagint all_count;
        size_t N;
        double cv_value;

        std::string env_setNum;
        MetaD_zqc::Averager* my_averager;
        bool init_flag=false;

        bool use_sw_func = false;
        SwitchFunctionRequest sw_params;
        SwitchFunction* h_sw_func;

        // [chem_pair_r] = chem_pair_r per atoms
        double                                      *h_chem_pair_r = nullptr;
        GpuBuffer<double>                           d_chem_pair_r;
        // [dcvdx] = dcv/dx 
        double                                      *h_dcvdx = nullptr;
        GpuBuffer<double>                           d_dcvdx;
        // device local
    public:
        static CV* create(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::FixMetadynamics *Fixmetad,
                         int narg, char **arg, int &i, FILE *f_check);
        Weighted_chem_pair(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check,
                    std::string env_setNum, int group_id, MetaD_zqc::Stru_fact_chem_env* my_env,
                    int d_block_size);
        ~Weighted_chem_pair() override;
        CV_Calculation set_CV_calculate(std::string func_name) override;
        CV_BiasForce set_CV_bias_force(std::string func_name) override;

        void base_calc() override; // 计算 CV 值
        void compute_Weighted_chem_pair_peratoms();
        void wcp_param_calc(double *h_chem_pair_r);

        double compute_cv_AVE();
        void bias_force_AVE(double dVdcv);
        void get_dcvdx_AVE(double cv_value, double *dcvdx);
        double compute_cv_COUNT();
        void bias_force_COUNT(double dVdcv);
        void get_dcvdx_COUNT(double cv_value, double *dcvdx);
        
        void summary(FILE* f) override;
        void environment();

        bool need_forward_comm() override { return true; }
        int get_comm_forward_bytes() override;
        int pack_comm_ubuf(int n, int *list, double *u_buf, int slot_offset, int comm_forward) override;
        void unpack_comm_ubuf(int n, int first, double *u_buf, int slot_offset, int comm_forward) override;

        void call_Weighted_chem_pair_cv_kernel();
        void call_Weighted_chem_pair_dcv_COUNT_kernel();
        void call_Weighted_chem_pair_dcv_AVE_kernel();

        // compute
        double* get_peratom_ptr(const std::string &prop_name) override;
    };
}

__global__ void Weighted_chem_pair_dcv_AVE_kernel(
    int group_count, int groupbit, 
    int all_count, double cutoff_rsq,
    int *d_mask, LAMMPS_NS::tagint *d_group_indices, 
    LAMMPS_NS::tagint *d_calculated_numneigh,
    LAMMPS_NS::tagint *d_group_numneigh,
    int *d_neigh_in_cutoff_r, double *d_group_dminneigh,
    double *d_stru_factor, 
    int *d_atom_types, double *d_type_weights,
    double *d_dcvdx);

__global__ void Weighted_chem_pair_cv_kernel(
    int group_count, double cutoff_rsq,
    LAMMPS_NS::tagint *d_group_numneigh,
    double *d_group_dminneigh, 
    LAMMPS_NS::tagint *d_group_indices,
    int *d_firstneigh_ptrs, 
    int *d_neigh_in_cutoff_r, 
    int *d_atom_types, double *d_type_weights,
    double *d_chem_pair_r);