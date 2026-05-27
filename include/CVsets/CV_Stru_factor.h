#pragma once  // 必须添加这一行

#include "fix_crystallize.h"
#include "lammps.h"
#include "pair.h"
#include "neigh_request.h"
#include "neigh_list.h"
#include "zqc_CVs_tools.h"

namespace MetaD_zqc {
    struct StruFactorRequest;
    class Stru_fact_env;
    class Stru_fact_chem_env;
    class Stru_factor;
    class Stru_fact_chem;

    struct StruFactorRequest {
        int group_id;
        std::string cal_name;
        char *group_name;
        double cutoff_r;
        double q_factor;
        int d_block_size;
        int original_arg_index; 

        bool use_chemical_lock = false;
        double c_target = 0.0;
        double sigma = 1.0;

        bool use_custom_weight = false;
        std::map<int, double> custom_weights; // 如果认为哪种化学成分的原子相近，可以给出一个相似的map值
    };

    class Stru_fact_env{
        friend class Stru_factor;
        friend class Stru_fact_chem_env;
        friend class Stru_factor_chem;

    private:
        // use env_pool to save envioronment, 
        // avoid repeatly construct envioronment when multiple steinhardt CVs exist
        static std::map<std::string, Stru_fact_env*> env_pool;
        LAMMPS_NS::bigint last_update_step = -1; // 避免同一步内重复计算 GPU Kernel
        bool use_chemical_lock=false;

        // vars for envioronment calculate
        FILE *f_check = nullptr;
        LAMMPS_NS::LAMMPS *lmp = nullptr;
        LAMMPS_NS::Error *error = nullptr;
        LAMMPS_NS::FixMetadynamics *Fixmetad = nullptr;
        double q_factor;
        double cutoff_r;
        int last_group_count, group_count, group_id, groupbit;
        int init_flag=0;
        bool pbc_x, pbc_y, pbc_z;
        double box_x, box_y, box_z;
        LAMMPS_NS::bigint                           *atoms;
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
        // LAMMPS_NS::tagint                           *h_tag = nullptr;
        // GpuBuffer<LAMMPS_NS::tagint>                d_tag;
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
        
        void get_env();
        Stru_fact_env(LAMMPS_NS::LAMMPS *lmp, FILE *f_check,
             LAMMPS_NS::FixMetadynamics *Fixmetad, int group_id,
             double cutoff_r);
        public:
        // 工厂函数：内部自动合并相同参数的环境
        static Stru_fact_env* get_or_create(LAMMPS_NS::LAMMPS *lmp, FILE *f_check, 
                                            LAMMPS_NS::FixMetadynamics *Fixmetad, 
                                            StruFactorRequest req);
        // Stru_factor* create_Stru_fact(LAMMPS_NS::LAMMPS *lmp, FILE *f_check,
        //      LAMMPS_NS::FixMetadynamics *Fixmetad, int group_id,
        //      double cutoff_r);
        std::string get_env_key();
        // 用于在 Fix 析构时清理所有显存
        static void clear_pool();
        virtual ~Stru_fact_env();
        virtual void refresh_lmpbox();
    };

    class Stru_factor : public CV {
        friend class Stru_factor_chem;
    private:
        // FILE *f_check = nullptr;
        LAMMPS_NS::Error *error = nullptr;
        LAMMPS_NS::FixMetadynamics *Fixmetad = nullptr;
        int init_flag=false;
        std::string env_setNum;
        MetaD_zqc::Averager* my_averager;
        MetaD_zqc::Stru_fact_env* my_env;
        int d_block_size;         // use it to change the GPU set
        double q_factor;
        int GPU_number;
        int block_num;
        int neighbor_type = 0;
        LAMMPS_NS::tagint all_count;
        size_t N;
        double cv_value;
        // [stru_factor] = stru_factor per atoms
        double                                      *h_stru_factor = nullptr;
        GpuBuffer<double>                           d_stru_factor;
        // [dcvdx] = dcv/dx 
        double                                      *h_dcvdx = nullptr;
        GpuBuffer<double>                           d_dcvdx;
        // device local
    public:
        static CV* create(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::FixMetadynamics *Fixmetad,
                         int narg, char **arg, int &i, FILE *f_check);
        Stru_factor(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check,
                    std::string env_setNum, int group_id, MetaD_zqc::Stru_fact_env* my_env,
                    double q_factor, int d_block_size);
        ~Stru_factor() override;

        using CV_Calculation = typename CV::CV_Calculation;
        using CV_BiasForce = typename CV::CV_BiasForce;
        CV_Calculation set_CV_calculate(std::string func_name) override;
        CV_BiasForce set_CV_bias_force(std::string func_name) override;

        // void cv_method();
        void base_calc() override;
        virtual void call_structure_factor_cv_kernel();
        virtual void call_structure_factor_dcv_AVE_kernel();

        void compute_Q_peratoms();
        double compute_cv_AVE();
        void bias_force_AVE(double dVdcv);
        void summary(FILE* f) override;
        void get_dcvdx_AVE(double cv_value, double *dcvdx);
        void steinhardt_param_calc(double *);
        void get_numneigh_full_pair_ABANDON_();
        void compute_stru_factor_peratoms();
        void sf_param_calc(double *h_stru_factor);
        void envioronment();
    };

    class Stru_fact_chem_env : public Stru_fact_env {
        friend class Stru_factor_chem;
    private:
        double c_target;
        double sigma;
        // GPU 端常驻的高斯权重映射表（下标对应 LAMMPS 的原子 type）
        double                                      *h_type_weights = nullptr;
        GpuBuffer<double>                           d_type_weights;
        // [atom_types] : list for atom types, 1-D = [nlocal]
        // 注意：这个指针直接指向 LAMMPS 的 atom->type 数组，不需要在构造函数里分配和复制数据，只需要在 get_env() 里刷新一次即可
        int                                         *atom_types = nullptr;
        GpuBuffer<int>                              d_atom_types;

    public:
        // Stru_fact_chem_env(LAMMPS_NS::LAMMPS *lmp, FILE *f_check,
        //                    LAMMPS_NS::FixMetadynamics *Fixmetad, int group_id, 
        //                    double cutoff_r, double c_target, double sigma, 
        //                    const std::vector<double>& type_table);
        Stru_fact_chem_env(LAMMPS_NS::LAMMPS *lmp, FILE *f_check,
                           LAMMPS_NS::FixMetadynamics *Fixmetad, int group_id, 
                           double cutoff_r, double c_target, double sigma, 
                           const std::map<int, double>& custom_weights);
        ~Stru_fact_chem_env() override;
        void refresh_lmpbox() override;
    };

    class Stru_factor_chem : public Stru_factor {
    private:
        Stru_fact_chem_env* my_chem_env; // 专有类型转换指针
    public:
        Stru_factor_chem(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check,
                         std::string env_setNum, int group_id, MetaD_zqc::Stru_fact_chem_env* my_env,
                         double q_factor, int d_block_size);
        void call_structure_factor_cv_kernel() override;
        void call_structure_factor_dcv_AVE_kernel() override;
    };
}

__global__ void get_envioronment_Strufactor(double cutoff_rsq,
    double box_x, double box_y, double box_z,
    int group_count, int *d_group_indices, LAMMPS_NS::tagint *d_group_numneigh,
    int *d_firstneigh_ptrs, double *d_x_flat,
    double *d_group_dminneigh, int *d_neigh_in_cutoff_r,
    LAMMPS_NS::tagint *d_calculated_numneigh);

__global__ void structure_factor_cv_kernel(
        int group_count, double q_factor, double cutoff_rsq,
        LAMMPS_NS::tagint *d_group_numneigh,
        double *d_group_dminneigh, int *d_neigh_in_cutoff_r, 
        double *d_stru_factor);

__global__ void structure_factor_dcv_AVE_kernel(
        int group_count, int groupbit, int all_count, double cutoff_rsq,
        int *d_mask, LAMMPS_NS::tagint *d_group_indices, 
        LAMMPS_NS::tagint *d_calculated_numneigh,
        LAMMPS_NS::tagint *d_group_numneigh,
        int *d_neigh_in_cutoff_r, double *d_group_dminneigh,
        double q_factor, double *d_stru_factor, 
        double *d_dcvdx);

__global__ void structure_factor_chem_cv_kernel(
        int group_count, double q_factor, double cutoff_rsq,
        int *d_atom_types, double *d_gaussian_weights,
        LAMMPS_NS::tagint *d_group_numneigh,
        double *d_group_dminneigh, int *d_neigh_in_cutoff_r, 
        double *d_stru_factor);

__global__ void structure_factor_chem_dcv_AVE_kernel(
        int group_count, int groupbit, int all_count, double cutoff_rsq,
        int *d_mask, LAMMPS_NS::tagint *d_group_indices, 
        int *d_atom_types, double *d_gaussian_weights, // 传入化学锁信息
        LAMMPS_NS::tagint *d_calculated_numneigh,
        LAMMPS_NS::tagint *d_group_numneigh,
        int *d_neigh_in_cutoff_r, double *d_group_dminneigh,
        double q_factor, double *d_stru_factor, 
        double *d_dcvdx);

__global__ void cv_kernel_structure_factor_chem(
        int group_count, double q_factor, double cutoff_rsq,
        LAMMPS_NS::tagint *d_group_numneigh,
        double *d_group_dminneigh, 
        LAMMPS_NS::tagint *d_group_indices,
        int *d_firstneigh_ptrs, 
        int *d_neigh_in_cutoff_r, 
        int *d_atom_types, double *d_type_weights,
        double *d_stru_factor);

__global__ void dcv_AVE_kernel_structure_factor_chem(
        int group_count, double q_factor, int groupbit, 
        int all_count, double cutoff_rsq,
        int *d_mask, LAMMPS_NS::tagint *d_group_indices, 
        LAMMPS_NS::tagint *d_calculated_numneigh,
        LAMMPS_NS::tagint *d_group_numneigh,
        int *d_neigh_in_cutoff_r, double *d_group_dminneigh,
        double *d_stru_factor, 
        int *d_atom_types, double *d_type_weights,
        double *d_dcvdx);