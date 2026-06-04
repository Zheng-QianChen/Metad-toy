#pragma once  // 必须添加这一行

#include "fix_crystallize.h"
#include "lammps.h"
#include "pair.h"
#include "neigh_request.h"
#include "neigh_list.h"
#include "zqc_CVs_tools.h"

#define PI 3.1415926535897932385
namespace MetaD_zqc {
    class Distance;
    class Steinhardt;
    template <int L> class STEIN_QL;
    class Steinhardt_env;

    // atoms distance
    class Distance : public CV {
    private:
        LAMMPS_NS::bigint atom_id1, atom_id2;
        bool pbc_x, pbc_y, pbc_z;
        double box_x, box_y, box_z;
        double dx, dy, dz;
    public:
        static CV* create(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::FixMetadynamics *Fixmetad,
                         int narg, char **arg, int &i, FILE *f_check);
        Distance(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::bigint id1, LAMMPS_NS::bigint id2, FILE *f_check);
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
        virtual void get_dcvdx(double cv_value, double *dcvdx) = 0;
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
    private:
        // use env_pool to save environment, 
        // avoid repeatly construct environment when multiple steinhardt CVs exist
        static std::map<std::string, Steinhardt_env*> env_pool;
        int ref_count = 0; // 引用计数，用于管理生命周期
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
        static void clear_pool();

        ~Steinhardt_env();
        void refresh_lmpbox();
    };

    // Steinhardt local
    template <int L>
    class STEIN_QL : public Steinhardt {
    private:
        // FILE *f_check = nullptr;
        LAMMPS_NS::Error *error = nullptr;
        LAMMPS_NS::FixMetadynamics *Fixmetad = nullptr;
        int init_flag=false;
        int stein_l=0;
        std::string env_setNum;
        MetaD_zqc::Averager* my_averager;
        MetaD_zqc::Steinhardt_env* my_env;
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
        double compute_cv_AVE();
        double compute_cv_LOC_AVE();
        void bias_force_AVE(double dVdcv);
        void bias_force_LOC_AVE(double dVdcv);
        void summary(FILE* f) override;
        void get_dcvdx(double cv_value, double *dcvdx) override;
        void get_dcvdx_AVE(double cv_value, double *dcvdx);
        void get_dcvdx_LOC_AVE(double cv_value, double *dcvdx);
        void steinhardt_param_calc(double *);
        void get_numneigh_full_pair_ABANDON_();
        void call_steinhardt_cv_kernel();
        void call_steinhardt_dcv_kernel();
        void call_steinhardt_cv_LOCAL_kernel();
        void call_steinhardt_dcv_LOCAL_kernel();
        void environment();
        // communication for Ghost atoms
        bool need_forward_comm() override { return true; }
        int get_comm_forward_bytes() override;
        int pack_comm_ubuf(int n, int *list, double *u_buf, int slot_offset, int comm_forward) override;
        void unpack_comm_ubuf(int n, int first, double *u_buf, int slot_offset, int comm_forward) override;
        // compute
        double* get_peratom_ptr(const std::string &prop_name) override;
    };

    Steinhardt* create_steinhardt_cv(LAMMPS_NS::LAMMPS *lmp,
                                LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check,
                                std::string env_setNum, int group_id, int Q_num,
                                Steinhardt_env* my_env,
                                char *Q_type_str, double cutoff_r, int cutoff_Natoms, 
                                int d_block_size);

}