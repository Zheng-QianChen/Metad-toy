#pragma once  // 必须添加这一行

#include "lammps.h"
#include "pair.h"
#include "neigh_request.h"
#include "neigh_list.h"

#include "fix_crystallize.h"
#include "zqc_CVs_tools.h"
#include "zqc_switch_function.h"

#define PI 3.1415926535897932385
namespace MetaD_zqc {
    class Distance;
    class Steinhardt;
    class SwitchFunction;
    template <int L> class STEIN_QL;
    class Steinhardt_env;

    // Steinhardt local
    class Steinhardt : public CV {
    protected:
        double dx, dy, dz;
        double *my_qlm_data; 
        int num_elements; // 比如 (2l+1)*2
    private:
    public:
        static CV* create(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::FixMetadynamics *Fixmetad,
                             int narg, char **arg, int &i, FILE *f_check);
        Steinhardt(LAMMPS_NS::LAMMPS *lmp, FILE *f_check):CV(lmp, f_check){}
        virtual ~Steinhardt() override{};
        void summary(FILE* f) override = 0;
        virtual CV_Calculation set_CV_calculate(std::string func_name) = 0;
        virtual CV_BiasForce set_CV_bias_force(std::string func_name) = 0;
        virtual void base_calc() = 0; // 计算 CV 值
        // virtual void get_dcvdx(double cv_value, double *dcvdx) = 0;
    };

    struct SteinhardtRequest {
        int group_id;
        std::string cal_name;
        char *group_name;
        double cutoff_r;
        int cutoff_Natoms;
        int Q_num;
        char *Q_type_str; // 'Q' or 'L'
        int d_block_size;
        int original_arg_index; 
    };

    // Steinhardt environment
    class Steinhardt_env{
        friend class Steinhardt;
        template <int U> friend class STEIN_QL;
    protected:
        // use env_pool to save environment, 
        // avoid repeatly construct environment when multiple steinhardt CVs exist
        static std::map<std::string, Steinhardt_env*> env_pool;
        int ref_count = 0; // 全局静态计数
        LAMMPS_NS::bigint last_update_step = -1; // 避免同一步内重复计算 GPU Kernel

        // vars for environment calculate
        FILE *f_check = nullptr;
        LAMMPS_NS::LAMMPS *lmp = nullptr;
        LAMMPS_NS::Error *error = nullptr;
        LAMMPS_NS::FixMetadynamics *Fixmetad = nullptr;
        double cutoff_r;          // environment_cutoff radius
        int cutoff_Natoms;        // environment_cutoff natoms
        int last_group_count, group_count, group_id, groupbit;
        int init_flag=0;
        bool pbc_x, pbc_y, pbc_z;
        LAMMPS_NS::bigint                           *atoms;
        double box_x, box_y, box_z;
        size_t N;
        int d_block_size;         // use it to change the GPU set
        int GPU_number;
        int block_num;
        // [nlist] : full neighborlist
        LAMMPS_NS::NeighList                        *nlist = nullptr;
        // lammps imformation
        LAMMPS_NS::Atom                             *atom = nullptr;
        // [group_numneigh] : group neighbor, neighbors number for center atoms
        LAMMPS_NS::tagint                           *h_group_numneigh = nullptr;
        GpuBuffer<LAMMPS_NS::tagint>                d_group_numneigh;
        LAMMPS_NS::tagint                           *numneigh = nullptr;            // ptr to get the list->numneigh
        LAMMPS_NS::tagint                           **firstneigh = nullptr;         // ptr to get the list->firstneigh
        // [h_x_flat] : list for lammps atoms, 3*nlocal
        double                                      *h_x_flat = nullptr;
        GpuBuffer<double>                           d_x_flat;
        // [mask] : list for lammps each group id, 1-D = [nlocal]
        //          e.g. when use "group test id 1 1000 5000", find its groupid by "test"
        //               then we can find atoms in this group by use "mask[i] & groupid"
        // [group_indices] : group atoms tagint, 1-D = [atoms in group and also in local]
        int                                         *mask = nullptr;
        GpuBuffer<int>                              d_mask;
        LAMMPS_NS::tagint                           *h_group_indices = nullptr;
        GpuBuffer<LAMMPS_NS::tagint>                d_group_indices;
        // [firstneigh_ptrs]: group neighbor, each center atoms * neighbors localtag
        int                                         *h_firstneigh_ptrs = nullptr;
        GpuBuffer<int>                              d_firstneigh_ptrs;
        // [group_dminneigh] = [ delta x, delta y, delta z, r squared] * c_atoms * cutoff_N ]
        double                                      *group_dminneigh = nullptr;
        GpuBuffer<double>                           d_group_dminneigh;
        // [neigh_in_cutoff_r] : how many neigh's r less than set
        // [neigh_both_in_r_N] : how many neighs satisfied r and N
        // [calculated_numneigh] : local tagint of neighs, both in r and N
        //                         default tagint is -1, 1-D = [c_atoms*cutoff_N]
        int                                         *neigh_in_cutoff_r = nullptr;
        GpuBuffer<int>                              d_neigh_in_cutoff_r;
        int                                         *neigh_both_in_r_N = nullptr;
        GpuBuffer<int>                              d_neigh_both_in_r_N;
        LAMMPS_NS::tagint                           *calculated_numneigh = nullptr;
        GpuBuffer<LAMMPS_NS::tagint>                d_calculated_numneigh;
        
        Steinhardt_env(LAMMPS_NS::LAMMPS *lmp, FILE *f_check,
             LAMMPS_NS::FixMetadynamics *Fixmetad, int group_id,
             double cutoff_r, int cutoff_Natoms);
        void get_env();

    public:
        // 工厂函数：内部自动合并相同参数的环境
        static Steinhardt_env* get_or_create(LAMMPS_NS::LAMMPS *lmp, FILE *f_check, 
                                            LAMMPS_NS::FixMetadynamics *Fixmetad, 
                                            int group_id, double cutoff_r, int cutoff_Natoms);
        // Steinhardt* create_steinhardt_cv(LAMMPS_NS::LAMMPS *lmp, FILE *f_check,
        //      LAMMPS_NS::FixMetadynamics *Fixmetad, int group_id,
        //      double cutoff_r, int cutoff_Natoms);
        std::string get_env_key();

        // 用于在 Fix 析构时清理所有显存
        void register_env() { ref_count++; }
        void unregister_env() {
            ref_count--;
            if (ref_count <= 0) {
                // 1. 获取当前环境的唯一 Key (比如 "group_1_r_5.0")
                std::string key = this->get_env_key();
                // 2. 从全局复用池中将自己除名，防止后面的 CV 拿到已被销毁的僵尸指针
                auto it = env_pool.find(key);
                if (it != env_pool.end()) {
                    env_pool.erase(it);
                }
                // 这会自动触发真正的 ~Steinhardt_env() 析构函数并回收结构体自身的堆空间
                delete this; 
            }
        }

        ~Steinhardt_env();
        void refresh_lmpbox();
    };

    // Steinhardt local
    template <int L>
    class STEIN_QL : public Steinhardt {
    protected:
        // FILE *f_check = nullptr;
        LAMMPS_NS::Error *error = nullptr;
        LAMMPS_NS::FixMetadynamics *Fixmetad = nullptr;
        int init_flag=false;
        int stein_l=0;
        std::string env_setNum;
        MetaD_zqc::Steinhardt_env* my_env;
        
        MetaD_zqc::Averager* my_averager;
        MetaD_zqc::SwitchFunction* my_cv_SWfunc;
        MetaD_zqc::SwitchFunction* my_r_SWfunc;

        int d_block_size;         // use it to change the GPU set
        int GPU_number;
        int block_num;
        int neighbor_type = 0;
        LAMMPS_NS::tagint all_count;
        size_t N;
        double cv_value;
        // stein_ql in host is stored in steinq[i]
        double                                      *stein_q = nullptr;
        GpuBuffer<double>                           d_stein_ql;
        // [h_stein_Ylm] : Ylm for each c_atoms
        double                                      *h_stein_Ylm = nullptr;
        GpuBuffer<double>                           d_stein_Ylm;
        // double *Q_per_atoms_value = nullptr;
        // [dYlm_dr] = dcv/dx (complex add local)
        // [dcvdx] = dcv/dx
        double                                      *h_dYlm_dr = nullptr;
        GpuBuffer<double>                           d_dYlm_dr;
        double                                      *h_dcvdx = nullptr;
        GpuBuffer<double>                           d_dcvdx;
        // device local
        // stein_qlm stored in 
        double                                      *h_stein_qlm = nullptr;
        GpuBuffer<double>                           d_stein_qlm;
        double                                     *h_stein_LQlm = nullptr;
        GpuBuffer<double>                           d_stein_LQlm;
    public:
        using CV_Calculation = typename CV::CV_Calculation;
        using CV_BiasForce = typename CV::CV_BiasForce;
        STEIN_QL(LAMMPS_NS::LAMMPS *lmp,
                             LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check, 
                             std::string env_setNum, int group_id, int stein_l, 
                             MetaD_zqc::Steinhardt_env* my_env,
                             int d_block_size);
        ~STEIN_QL() override;
        CV_Calculation set_CV_calculate(std::string func_name) override;
        CV_BiasForce set_CV_bias_force(std::string func_name) override;
        // void cv_method();
        void base_calc() override;
        void compute_Q_peratoms();
        // void bias_force_LOC_AVE(double dVdcv);
        void summary(FILE* f) override;
        
        // AVE method
        double compute_cv_AVE();
        void bias_force_AVE(double dVdcv);
        void get_dcvdx_AVE(double cv_value, double *dcvdx);

        // SW_FUNC method
        double compute_cv_SW_FUNC();
        void bias_force_SW_FUNC(double dVdcv);
        void get_dcvdx_SW_FUNC(double cv_value, double *dcvdx);
        
        // void get_dcvdx_LOC_AVE(double cv_value, double *dcvdx);
        void steinhardt_param_calc(double *);
        void call_steinhardt_cv_AVE_kernel();
        void call_steinhardt_dcv_AVE_kernel();
        // void call_steinhardt_cv_SW_FUNC_kernel();
        void call_steinhardt_dcv_SW_FUNC_kernel();
        void environment();
        // communication for Ghost atoms
        virtual bool need_forward_comm() override { return true; }
        virtual int get_comm_forward_bytes() override;
        virtual int pack_comm_ubuf(int n, int *list, double *u_buf, int slot_offset, int comm_forward) override;
        virtual void unpack_comm_ubuf(int n, int first, double *u_buf, int slot_offset, int comm_forward) override;
        // compute
        virtual double* get_peratom_ptr(const std::string &prop_name) override;
    };

    // class STEIN_LocalQL_env : public Steinhardt_env{
    // };

    // // Steinhardt local
    // template <int L>
    // class STEIN_LocalQL : public STEIN_QL {
    // protected:
    // public:
    //     STEIN_LocalQL(LAMMPS_NS::LAMMPS *lmp,
    //                          LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check, 
    //                          std::string env_setNum, int group_id, int stein_l, 
    //                          MetaD_zqc::Steinhardt_env* my_env,
    //                          int d_block_size);
    //     ~STEIN_LocalQL() override;

    //     void compute_Q_peratoms();
    //     double compute_cv_AVE();
    //     void bias_force_AVE(double dVdcv);
    //     void get_dcvdx(double cv_value, double *dcvdx) override;
    //     void get_dcvdx_AVE(double cv_value, double *dcvdx);
    //     void summary(FILE* f) override;
    //     void call_steinhardt_cv_AVE_kernel();
    //     void call_steinhardt_dcv_AVE_kernel();

    //     // communication for Ghost atoms
    //     bool need_forward_comm() override { return true; }
    //     int get_comm_forward_bytes() override;
    //     int pack_comm_ubuf(int n, int *list, double *u_buf, int slot_offset, int comm_forward) override;
    //     void unpack_comm_ubuf(int n, int first, double *u_buf, int slot_offset, int comm_forward) override;
    //     // compute
    //     double* get_peratom_ptr(const std::string &prop_name) override;
    // };

    Steinhardt* create_steinhardt_cv(LAMMPS_NS::LAMMPS *lmp,
                                LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check,
                                std::string env_setNum, int group_id, int Q_num,
                                Steinhardt_env* my_env,
                                char *Q_type_str, double cutoff_r, int cutoff_Natoms, 
                                int d_block_size);

}


__global__ void get_environment_Steinhardt_Q(
    int cutoff_Natoms, double cutoff_rsq, double box_x, double box_y, double box_z,
    int group_count, int *d_group_indices, LAMMPS_NS::tagint *d_group_numneigh,
    int *d_firstneigh_ptrs, double *d_x_flat,
    double *d_group_dminneigh, int *d_neigh_in_cutoff_r, int *d_neigh_both_in_r_N,
    LAMMPS_NS::tagint *d_calculated_numneigh
);


template <int L>
__global__ void steinhardt_cv_AVE_kernel(
    int group_count, int cutoff_Natoms, int *d_group_indices,
    int *d_neigh_both_in_r_N, double *d_group_dminneigh,
    double *d_stein_qlm, double *d_stein_Ylm, double *d_stein_ql
);


template <int L>
__global__ void steinhardt_dcv_AVE_kernel(
    int cutoff_Natoms, int group_count, int groupbit, int all_count, 
    int *d_mask, LAMMPS_NS::tagint *d_group_indices, LAMMPS_NS::tagint *calculated_numneigh, 
    int *d_neigh_both_in_r_N, double *d_group_dminneigh,
    double *d_stein_qlm, double *d_stein_Ylm, double *d_stein_ql,
    double *d_dYlm_dr, double *d_dcvdx
);


template <int L>
__global__ void steinhardt_dcv_SW_FUNC_kernel(
    MetaD_zqc::SwitchFunctionRequest sw_params,
    int cutoff_Natoms, int group_count, int groupbit, int all_count, 
    int *d_mask, LAMMPS_NS::tagint *d_group_indices, LAMMPS_NS::tagint *calculated_numneigh, 
    int *d_neigh_both_in_r_N, double *d_group_dminneigh,
    double *d_stein_qlm, double *d_stein_Ylm, double *d_stein_ql,
    double *d_dYlm_dr, double *d_dcvdx
);


// __global__ void steinhardt_param_calc_LOCAL_kernel(int group_count, int cutoff_Natoms,
//                     int stein_l, int groupbit,
//                     int *d_mask, LAMMPS_NS::tagint *d_group_indices,
//                     LAMMPS_NS::tagint *calculated_numneigh, 
//                     int *d_neigh_both_in_r_N,
//                     double *d_stein_qlm, double *d_stein_LQlm,
//                     double *d_stein_ql);