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
                         FILE *f_check, int narg, char **arg, int &i);
        Steinhardt(LAMMPS_NS::LAMMPS *lmp, 
                    LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check)
                :CV(lmp, Fixmetad, f_check){}
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
        int Q_num;
        char *Q_type_str; // 'Q' or 'L'
        int d_block_size;
        int original_arg_index; 

        double cutoff_r;
        // only env
        int cutoff_Natoms;
        // only Local_env
        double cutoff_eps;

        SwitchFunction* SW_FUNC_r=nullptr;
        SwitchFunction* SW_FUNC_cv=nullptr;
    };

    // Steinhardt environment
    class Steinhardt_env : public CV_info{
        friend class Steinhardt;
        template <int U> friend class STEIN_QL;
        protected:
            // use env_pool to save environment, 
            // avoid repeatly construct environment when multiple steinhardt CVs exist
            static std::map<std::string, Steinhardt_env*> env_pool;
            int ref_count = 0; // 全局静态计数
            bool LOC_flag = false;
            LAMMPS_NS::bigint last_update_step = -1; // 避免同一步内重复计算 GPU Kernel
            
            MetaD_zqc::SwitchFunction* my_r_SWfunc;

            // vars for environment calculate
            // FILE *f_check = nullptr;
            // LAMMPS_NS::LAMMPS *lmp = nullptr;
            // LAMMPS_NS::Error *error = nullptr;
            // LAMMPS_NS::FixMetadynamics *Fixmetad = nullptr;
            double cutoff_r;          // environment_cutoff radius
            int cutoff_Natoms;        // environment_cutoff natoms
            int last_group_count, group_count, group_id, groupbit;
            int init_flag=false;
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

        public:
            virtual void get_env();
            Steinhardt_env(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::FixMetadynamics *Fixmetad, 
                FILE *f_check, int group_id,
                double cutoff_r, int cutoff_Natoms);
            // 工厂函数：内部自动合并相同参数的环境
            static Steinhardt_env* get_or_create(LAMMPS_NS::LAMMPS *lmp, 
                        LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check, 
                        SteinhardtRequest req);
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
            virtual void refresh_lmpbox();
    };

    // Steinhardt local
    template <int L>
    class STEIN_QL : public Steinhardt {
        protected:
            // FILE *f_check = nullptr;
            // LAMMPS_NS::Error *error = nullptr;
            // LAMMPS_NS::FixMetadynamics *Fixmetad = nullptr;
            int init_flag=false;
            int stein_l=0;
            std::string env_setNum;
            MetaD_zqc::Steinhardt_env* my_env;
            
            MetaD_zqc::Averager* my_averager;
            MetaD_zqc::SwitchFunction* my_cv_SWfunc;

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
            STEIN_QL(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::FixMetadynamics *Fixmetad, 
                                FILE *f_check, 
                                std::string env_setNum, int group_id, int stein_l, 
                                MetaD_zqc::Steinhardt_env* my_env,
                                int d_block_size);
            ~STEIN_QL() override;
            CV_Calculation set_CV_calculate(std::string func_name) override;
            CV_BiasForce set_CV_bias_force(std::string func_name) override;
            // void cv_method();
            void base_calc() override;
            virtual void compute_Q_peratoms();
            // void bias_force_LOC_AVE(double dVdcv);
            void summary(FILE* f) override;
            
            // AVE method
            virtual double compute_cv_AVE();
            virtual void bias_force_AVE(double dVdcv);
            virtual void get_dcvdx_AVE(double cv_value, double *dcvdx);

            // SW_FUNC method
            double compute_cv_SW_FUNC();
            void bias_force_SW_FUNC(double dVdcv);
            void get_dcvdx_SW_FUNC(double cv_value, double *dcvdx);
            
            // void get_dcvdx_LOC_AVE(double cv_value, double *dcvdx);
            virtual void steinhardt_param_calc(double *);
            void call_steinhardt_cv_AVE_kernel();
            void call_steinhardt_dcv_AVE_kernel();
            // void call_steinhardt_cv_SW_FUNC_kernel();
            void call_steinhardt_dcv_SW_FUNC_kernel();
            void environment();
            // communication for Ghost atoms
            virtual bool need_forward_comm() override { return true; }
            virtual int get_comm_forward_bytes() override;
            virtual int pack_comm_forward_ubuf(int n, int *list, double *u_buf, int slot_offset, int comm_forward) override;
            virtual void unpack_comm_forward_ubuf(int n, int first, double *u_buf, int slot_offset, int comm_forward) override;
            // compute
            virtual double* get_peratom_ptr(const std::string &prop_name) override;
    };
    template class MetaD_zqc::STEIN_QL<3>;
    template class MetaD_zqc::STEIN_QL<4>;
    template class MetaD_zqc::STEIN_QL<6>;


    // 这是一会要用的妙妙小工具
    // ---- 1. 状态标志位的掩码 (Flags) ----
    #define CALC_MASK_ACTIVE (1U << 24)  // 0x01000000
    #define CALC_MASK_IS_I   (1U << 25)  // 0x02000000
    #define CALC_MASK_IS_J   (1U << 26)  // 0x04000000
    #define CALC_MASK_IS_K   (1U << 27)  // 0x08000000
    // ---- 2. 计数器的位移与掩码 (Counters) ----
    #define CALC_SHIFT_I     16
    #define CALC_SHIFT_J     8
    #define CALC_SHIFT_K     0
    #define CALC_COUNTER_MASK 0xFFU      // 八位最大值 255

    class STEIN_LocalQL_env : public Steinhardt_env{
        template <int U> friend class STEIN_LocalQL;
        protected:
            // if sigma(r_ij) < cut_sigma_eps, this neighbor will be seen as kick off
            double                                      cutoff_eps_r = 1e-12;
            // [calc_tag] 1-D -> nall : local_tag -> h_calc_tag
            LAMMPS_NS::tagint                           *h_calc_tag = nullptr;
            GpuBuffer<LAMMPS_NS::tagint>                d_calc_tag;
            // [num_of_all_IJ_atoms] 1-D -> 1 : number of all atoms which is i or j
            // [d_pure_J_write_offset] 1-D -> 1 : how many atoms are j but not i
            // num_of_all_IJ_atoms = d_pure_J_write_offset + group_count
            LAMMPS_NS::tagint                           num_of_all_IJ_atoms;
            GpuBuffer<LAMMPS_NS::tagint>                d_pure_J_write_offset;
            // Mom [group_indices] : 1-D -> num_of_all_IJ_atoms : h_calc_tag -> local_tag
            // Mom [group_numneigh] : group neighbor, neighbors number for center atoms, (nall+1)
            // Mom [x_flat] : list for lammps atoms, 3*nlocal
            // Mom [mask] : list for lammps each group id, 1-D = [nlocal]
            //          e.g. when use "group test id 1 1000 5000", find its groupid by "test"
            //               then we can find atoms in this group by use "mask[i] & groupid"
            // Mom [firstneigh_ptrs]: group neighbor, each center atoms * neighbors localtag
            // Mom [calculated_numneigh] : local tagint of neighs, both in r and N
            // Mom [neigh_in_cutoff_r] 1-D -> (nall) : how many neigh's r less than set
            //                  [neigh_in_cutoff_r] = [12, 23, 0, 0, 14, ...]
            // [d_calculated_firstneigh_ptrs] 1-D -> (nall+1) : [0, 12, 12+23, 12+23, ...]
            // num_of_all_calc_fullpair = h_calculated_firstneigh_ptrs[num_of_all_IJ_atoms];
            LAMMPS_NS::tagint                           num_of_all_calc_fullpair;
            LAMMPS_NS::tagint                           *h_calculated_firstneigh_ptrs = nullptr;
            GpuBuffer<LAMMPS_NS::tagint>                d_calculated_firstneigh_ptrs;
            // [LQ_mask] : [32bit int mask] * nmax
            int                                         *h_LQ_mask = nullptr;
            GpuBuffer<int>                              d_LQ_mask;
            // [d_pure_J_offsets] : temp array which mask j and give idx
            // we dont need to quest in host    
            GpuBuffer<int>                              d_pure_J_offsets;
            // [neigh_in_switching] 1-D -> nmax (calctag): the sum of neigh_in_switching for calculated atoms
            double                                         *h_neigh_in_switching = nullptr;
            GpuBuffer<double>                              d_neigh_in_switching;
            
        public:
            STEIN_LocalQL_env(LAMMPS_NS::LAMMPS *lmp, 
                            LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check,
                            MetaD_zqc::SteinhardtRequest req);
            ~STEIN_LocalQL_env();

            void refresh_lmpbox() override;
            void get_env() override;
    };

    // Steinhardt local
    template <int L>
    class STEIN_LocalQL : public STEIN_QL<L> {
        protected:
        using CV::lmp;
        using CV::f_check;
        using CV::error;
        using CV::Fixmetad;
        using STEIN_QL<L>::my_averager;
        using STEIN_QL<L>::my_env;

        using STEIN_QL<L>::block_num;
        using STEIN_QL<L>::d_block_size;
        using STEIN_QL<L>::N;
        using STEIN_QL<L>::stein_l;
        using STEIN_QL<L>::comm_mode;

        using STEIN_QL<L>::all_count;

        using STEIN_QL<L>::my_cv_SWfunc;

        using STEIN_QL<L>::cv_value;

        STEIN_LocalQL_env* my_loc_env;

        // 如果采用切换函数，那么需要考虑一些切换函数的求和
        double                                          cvsum_in_switching;
        // Mom [d_stein_LQlm] 1-D -> nmax (calctag): Threads_own_atoms*(stein_l + 1)*2;
        GpuBuffer<double>                               d_stein_LQlm;
        GpuBuffer<double>                               d_dcvdx_rjk_prefix;
        // [sum_of_qlm_value_weights] 1-D -> nmax (calctag): Threads_own_atoms*(stein_l + 1)*2;
        GpuBuffer<double>                               sum_of_qlm_value_weights;
        // 全局规约后的对LQ的sw_sum
        double                                          global_ql_sum;
        double                                          global_sw_sum;
        // 由于我们未使用过stein_ql,所以这里其实可以重复使用stein_ql以保证继承更加顺畅
        using STEIN_QL<L>::env_setNum;
        // Mom [d_stein_ql] -> d_stein_LQl;
        using STEIN_QL<L>::stein_q;
        using STEIN_QL<L>::d_stein_ql;
        // Mom [stein_qlm] 1-D -> nmax*(2*(L+1)) (local_tag): stored each atoms' qlm
        using STEIN_QL<L>::h_stein_qlm;
        using STEIN_QL<L>::d_stein_qlm;
        using STEIN_QL<L>::h_dcvdx;
        using STEIN_QL<L>::d_dcvdx;
        using STEIN_QL<L>::h_stein_Ylm;
        using STEIN_QL<L>::d_stein_Ylm;
        using STEIN_QL<L>::h_dYlm_dr;
        using STEIN_QL<L>::d_dYlm_dr;
        // Mom [dYlm_dx]
        // 暂存所有的 dYlm/dx 1-D -> full_pair * 2*(l +1) *3(x,y,z)

        public:
        using CV::register_buffer;
        using STEIN_QL<L>::environment;

        STEIN_LocalQL(LAMMPS_NS::LAMMPS *lmp,
                             LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check, 
                             std::string env_setNum, int group_id,
                             MetaD_zqc::Steinhardt_env* my_env,
                             int d_block_size, MetaD_zqc::SteinhardtRequest req);
        ~STEIN_LocalQL();

        void summary(FILE* f) override;

        void compute_Q_peratoms() override;
        void steinhardt_param_calc(double *) override;
        double compute_cv_AVE() override;
        void bias_force_AVE(double dVdcv) override;
        // void get_dcvdx(double cv_value, double *dcvdx) override;
        void get_dcvdx_AVE(double cv_value, double *dcvdx) override;
        void call_steinhardt_Local_cv_AVE_kernel();
        void call_steinhardt_Local_dcv_AVE_kernel();

        // communication for Ghost atoms
        virtual bool need_reverse_comm() override {return true;}; // 是否需要跨进程同步 Ghost 属性
        virtual int get_comm_reverse_bytes() override; // 每个原子需要同步多少个 bytes
        virtual int pack_comm_reverse_ubuf(int n, int first, double *u_buf, 
                            int slot_offset, int comm_forward)  override; // 具体 CV 自己的打包逻辑
        virtual void unpack_comm_reverse_ubuf(int n, int *list, double *u_buf, 
                            int slot_offset, int comm_forward)  override; // 具体 CV 自己的解包逻辑
        virtual bool need_forward_comm() override { return false; }
        // int get_comm_forward_bytes() override;
        // int pack_comm_forward_ubuf(int n, int *list, double *u_buf, int slot_offset, int comm_forward) override;
        // void unpack_comm_forward_ubuf(int n, int first, double *u_buf, int slot_offset, int comm_forward) override;
        // // compute
        // double* get_peratom_ptr(const std::string &prop_name) override;
    };
    template class STEIN_LocalQL<3>;
    template class STEIN_LocalQL<4>;
    template class STEIN_LocalQL<6>;

    Steinhardt* create_steinhardt_cv(LAMMPS_NS::LAMMPS *lmp,
                                LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check,
                                std::string env_setNum, int group_id, int Q_num,
                                Steinhardt_env* my_env, 
                                SteinhardtRequest req);

}


// =============================================================================
// not LOCAL
// =============================================================================

__global__ void get_environment_Steinhardt_Q(
    int cutoff_Natoms, double cutoff_rsq, double box_x, double box_y, double box_z,
    int group_count, int *d_group_indices, LAMMPS_NS::tagint *d_group_numneigh,
    int *d_firstneigh_ptrs, double *d_x_flat,
    double *d_group_dminneigh, int *d_neigh_in_cutoff_r, int *d_neigh_both_in_r_N,
    LAMMPS_NS::tagint *d_calculated_numneigh
);


template <int L>
__global__ void steinhardt_cv_kernel(
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


// =============================================================================
// LOCAL
// =============================================================================
__global__ void get_environment_Steinhardt_LocalQ(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, int start_idx,
    double cutoff_r, double cut_sigma_eps,
    // in
    LAMMPS_NS::tagint *d_group_indices,
    LAMMPS_NS::tagint *d_calc_tag,
    LAMMPS_NS::tagint *d_group_numneigh,
    int *d_firstneigh_ptrs, double *d_x_flat,
    unsigned int current_pass_mask,   // 中心原子该打的标签 (例如：CALC_MASK_IS_I 或 CALC_MASK_IS_J)
    unsigned int neighbor_mask,       // 邻居原子该打的标签 (例如：CALC_MASK_IS_J 或 CALC_MASK_IS_K)
    unsigned int neighbor_inc_val,    // 邻居原子计数器自增的位移量 (例如：1U << CALC_SHIFT_I)
    // out
    int *d_neigh_in_cutoff_r, int *d_LQ_mask,
    double *d_neigh_in_switching,
    LAMMPS_NS::tagint *d_calculated_numneigh);

__global__ void get_environment_Steinhardt_LocalQ_promote_pure_K(
    int group_count,
    int atom_all, 
    const int *d_LQ_mask, 
    LAMMPS_NS::tagint *d_pure_J_write_offset,
    LAMMPS_NS::tagint *d_group_indices,
    LAMMPS_NS::tagint *d_calc_tag);

template <int L>
__global__ void steinhardt_Local_cv_get_qlm(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, double cutoff_r, double cutoff_eps,
    int *d_group_indices,
    int *d_neigh_in_cutoff_r,
    LAMMPS_NS::tagint *d_group_numneigh,
    LAMMPS_NS::tagint *d_calculated_firstneigh_ptrs,
    LAMMPS_NS::tagint *d_calculated_numneigh,
    double *d_x_flat,
    double *d_neigh_in_switching,
    double *d_stein_qlm, double *d_stein_Ylm);

template <int L>
__global__ void steinhardt_Local_cv_get_LQl(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, double cutoff_r, double cutoff_eps,
    int *d_group_indices,
    int *d_neigh_in_cutoff_r,
    LAMMPS_NS::tagint *d_group_numneigh,
    LAMMPS_NS::tagint *d_calculated_firstneigh_ptrs,
    LAMMPS_NS::tagint *d_calculated_numneigh,
    double *d_x_flat,
    double *d_neigh_in_switching,
    double *sum_of_qlm_value_weights,
    double *d_stein_qlm, 
    double *d_stein_LQlm, double *d_stein_ql);

template <int L>
__global__ void steinhardt_Local_dcv_AVE_ij_kernel(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    MetaD_zqc::SwitchFunctionRequest sw_params_LQ,
    double cv_value, double global_sw_sum,
    int group_count, 
    double *d_x_flat, double *d_neigh_in_switching,
    int *d_neigh_in_cutoff_r,
    LAMMPS_NS::tagint *d_calculated_firstneigh_ptrs,
    LAMMPS_NS::tagint *d_group_indices,
    LAMMPS_NS::tagint *d_group_numneigh,
    LAMMPS_NS::tagint *d_calculated_numneigh, 
    LAMMPS_NS::tagint *d_calc_tag, 
    double *sum_of_qlm_value_weights,
    double *d_dcvdx_rjk_prefix,
    double *d_stein_qlm, 
    double *d_stein_LQlm,
    double *d_stein_ql,
    double *d_dcvdx);

template <int L>
__global__ void steinhardt_Local_dcv_AVE_jk_kernel(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, double cutoff_r, double cutoff_eps,
    int *d_mask, 
    double *d_x_flat, double *d_neigh_in_switching,
    LAMMPS_NS::tagint *d_group_indices, 
    LAMMPS_NS::tagint *d_group_numneigh,
    LAMMPS_NS::tagint *d_calculated_firstneigh_ptrs,
    LAMMPS_NS::tagint *d_calculated_numneigh, 
    LAMMPS_NS::tagint *d_calc_tag, 
    int *d_neigh_in_cutoff_r, 
    double *d_dcvdx_rjk_prefix,
    double *d_stein_qlm, double *d_stein_Ylm, double *d_stein_ql,
    double *d_dYlm_dr, double *d_dcvdx);


// =========================================================================
// 🚀 1. 状态标志位操作宏 (Flags Operations) - 纯文本替换，全括号防御
// =========================================================================

// 检查状态（返回 bool 表达式）
#define MASK_IS_ACTIVE(mask) (((mask) & CALC_MASK_ACTIVE) != 0)
#define MASK_IS_I(mask)      (((mask) & CALC_MASK_IS_I)   != 0)
#define MASK_IS_J(mask)      (((mask) & CALC_MASK_IS_J)   != 0)
#define MASK_IS_K(mask)      (((mask) & CALC_MASK_IS_K)   != 0)

// 本地打标签（非并发安全，用于 CPU 或线程独占写入）
#define MASK_SET_ACTIVE(mask) ((mask) |= CALC_MASK_ACTIVE)
#define MASK_SET_I(mask)      ((mask) |= CALC_MASK_IS_I)
#define MASK_SET_J(mask)      ((mask) |= CALC_MASK_IS_J)
#define MASK_SET_K(mask)      ((mask) |= CALC_MASK_IS_K)

// 本地清除标签
#define MASK_CLEAR_ACTIVE(mask) ((mask) &= ~CALC_MASK_ACTIVE)

// =========================================================================
// 🚀 2. 计数器操作宏 (Counters Operations)
// =========================================================================

// 获取计数器的值（返回 int 表达式）
#define MASK_GET_COUNT_I(mask) (((mask) >> CALC_SHIFT_I) & CALC_COUNTER_MASK)
#define MASK_GET_COUNT_J(mask) (((mask) >> CALC_SHIFT_J) & CALC_COUNTER_MASK)
#define MASK_GET_COUNT_K(mask) (((mask) >> CALC_SHIFT_K) & CALC_COUNTER_MASK)

// =========================================================================
// 🚀 3. GPU 专用：硬件级原子并发操作宏（带 __CUDACC__ 保护）
// =========================================================================
#ifdef __CUDACC__

// 原子打标签（必须传入原始指针，例如 &d_LQ_mask[i]）
#define MASK_ATOMIC_SET_ACTIVE(mask_ptr) (atomicOr((unsigned int*)(mask_ptr), CALC_MASK_ACTIVE))
#define MASK_ATOMIC_SET_I(mask_ptr)      (atomicOr((unsigned int*)(mask_ptr), CALC_MASK_IS_I))
#define MASK_ATOMIC_SET_J(mask_ptr)      (atomicOr((unsigned int*)(mask_ptr), CALC_MASK_IS_J))
#define MASK_ATOMIC_SET_K(mask_ptr)      (atomicOr((unsigned int*)(mask_ptr), CALC_MASK_IS_K))
#define MASK_ATOMIC_SET_VAR(mask_ptr, target_mask) \
    (atomicOr((unsigned int*)(mask_ptr), CALC_MASK_ACTIVE | (target_mask)))

// 原子自增计数器
#define MASK_ATOMIC_INC_I(mask_ptr)      (atomicAdd((unsigned int*)(mask_ptr), (1U << CALC_SHIFT_I)))
#define MASK_ATOMIC_INC_J(mask_ptr)      (atomicAdd((unsigned int*)(mask_ptr), (1U << CALC_SHIFT_J)))
#define MASK_ATOMIC_INC_K(mask_ptr)      (atomicAdd((unsigned int*)(mask_ptr), (1U << CALC_SHIFT_K)))
#define MASK_ATOMIC_INC_VAR(mask_ptr, inc_val) \
    (atomicAdd((unsigned int*)(mask_ptr), (inc_val)))

#endif