#pragma once  // 必须添加这一行

#include "lammps.h"
#include "pair.h"
#include "neigh_request.h"
#include "neigh_list.h"

#include "fix_crystallize.h"
#include "zqc_CVs_tools.h"
#include "zqc_switch_function.h"
#include "zqc_cv_assembly.cuh"

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
        double cutoff_eps_r;

        SwitchFunction* SW_FUNC_r=nullptr;
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
            // Neighbor::lastcall when we last flattened firstneigh into GPU buffers
            long long neigh_lastcall_ = -1;
            // Q 的候选邻居缓存随 run、localid 布局和组表一起失效。
            unsigned long long cached_neigh_generation = 0;
            int cached_nlocal = -1, cached_nghost = -1;
            std::shared_ptr<const CVAssembly> atom_assembly; // owned 中心在前，组内 ghost 在后。
            bool graph_geometry_dirty = true; // 每轮 Graph 准备后，由首个共享 CV 更新几何。

            
            MetaD_zqc::SwitchFunction* my_r_SWfunc;

            // vars for environment calculate
            // FILE *f_check = nullptr;
            // LAMMPS_NS::LAMMPS *lmp = nullptr;
            // LAMMPS_NS::Error *error = nullptr;
            // LAMMPS_NS::FixMetadynamics *Fixmetad = nullptr;
            double                                      cutoff_r=0.0;   // environment_cutoff radius
            double                                      cutoff_eps_r = 1e-12;
            int last_group_count = 0, group_count = 0, group_id, groupbit;
            int init_flag=false;
            bool pbc_x, pbc_y, pbc_z;
            LAMMPS_NS::bigint                           *atoms;
            double box_x, box_y, box_z;
            size_t N;
            int d_block_size;         // use it to change the GPU set
            int GPU_number;
            int block_num;


            // [nlist] : full neighborlist (refreshed from hub each use)
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
            // [LQ_mask] : [32bit int mask] * nmax
            // [group_indices] : group atoms tagint, 1-D = [atoms in group and also in local]
            int                                         *mask = nullptr;
            GpuBuffer<int>                              d_mask;
            // int                                         *h_LQ_mask = nullptr;
            // GpuBuffer<int>                              d_LQ_mask;
            LAMMPS_NS::tagint                           *h_group_indices = nullptr;
            GpuBuffer<LAMMPS_NS::tagint>                d_group_indices;
            // [firstneigh_ptrs]: group neighbor, each center atoms * neighbors localtag
            int                                         *h_firstneigh_ptrs = nullptr;
            GpuBuffer<int>                              d_firstneigh_ptrs;
            // [neigh_in_switching] 1-D -> nmax (calctag): the sum of neigh_in_switching for calculated atoms
            double                                         *h_neigh_in_switching = nullptr;
            GpuBuffer<double>                              d_neigh_in_switching;
            // [d_calculated_firstneigh_ptrs] 1-D -> (nall+1) : [0, 12, 12+23, 12+23, ...]
            // num_of_all_calc_fullpair = h_calculated_firstneigh_ptrs[num_of_all_IJ_atoms];
            LAMMPS_NS::tagint                           num_of_all_calc_fullpair;
            LAMMPS_NS::tagint                           *h_calculated_firstneigh_ptrs = nullptr;
            GpuBuffer<LAMMPS_NS::tagint>                d_calculated_firstneigh_ptrs;
            // // [group_dminneigh] = [ delta x, delta y, delta z, r squared] * c_atoms * cutoff_N ]
            // double                                      *group_dminneigh = nullptr;
            // GpuBuffer<double>                           d_group_dminneigh;
            // [half_pair] : half pair for dcvdx
            // half_pair
            LAMMPS_NS::tagint                           n_half_candidates = 0;
            GpuBuffer<LAMMPS_NS::tagint>                d_half_pair_i;
            GpuBuffer<LAMMPS_NS::tagint>                d_half_pair_j;
            GpuBuffer<LAMMPS_NS::tagint>                d_half_to_full;
            LAMMPS_NS::tagint                           n_active_pairs = 0;
            GpuBuffer<int>                              d_active_pair_mask;
            GpuBuffer<LAMMPS_NS::tagint>                d_active_pair_ids;
            // full_pair
            LAMMPS_NS::tagint                           *h_full_to_half;
            GpuBuffer<LAMMPS_NS::tagint>                d_full_to_half;
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
            // NeighHub id on Fix; hot path: ensure + list(id)
            int neigh_id = -1;

            virtual void get_env();
            Steinhardt_env(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::FixMetadynamics *Fixmetad, 
                FILE *f_check, MetaD_zqc::SteinhardtRequest req);
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
            

            int d_block_size;         // use it to change the GPU set
            int GPU_number;
            int block_num;
            int neighbor_type = 0;
            size_t N;
            using CV::cv_value;
            // q_l 按 localid 存储，长度 nlocal+nghost；前向计算 owned，通信补齐 ghost。
            double                                      *stein_q = nullptr;
            GpuBuffer<double>                           d_stein_ql;
            // [h_stein_Ylm] : Ylm for each c_atoms
            double                                      *h_stein_Ylm = nullptr;
            GpuBuffer<double>                           d_stein_Ylm;
            // double *Q_per_atoms_value = nullptr;
            // 坐标梯度按 [localid*3+component] 存储。
            double                                      *h_dcvdx = nullptr;
            GpuBuffer<double>                           d_dcvdx;
            // device local
            // q_lm 按 [localid * 2*(L+1) + 2*m + real/imag] 存储，m=0..L。
            double                                      *h_stein_qlm = nullptr;
            GpuBuffer<double>                           d_stein_qlm;
            double                                     *h_stein_LQlm = nullptr;
            GpuBuffer<double>                           d_stein_LQlm;
            // [h_a_virial] = 6*[nlocal+nghost], local_tag : virial for all atoms
            double                                      *h_a_virial = nullptr;
            GpuBuffer<double>                           d_a_virial;

            double                                     *h_dcvdx_x = nullptr;
            double                                     *h_dcvdx_y = nullptr;
            double                                     *h_dcvdx_z = nullptr;
            int                                         dcvdx_flag = -1;

            // Graph 输入梯度先按 owned 中心排列提取，再按 localid 装入通信数组。
            GpuBuffer<double> d_center_adjoint; // [center]，group_count 个 owned dV/dq_i。
            GpuBuffer<double> d_atom_adjoint;   // [localid]，nlocal+nghost 项；通信后补齐 ghost。
            std::vector<double> h_center_adjoint; // [center]，长度 group_count。
            std::vector<double> h_atom_adjoint;   // [localid]，长度 nlocal+nghost，保存 dV/dq_i。
            bool graph_values_ready = false;
            bool graph_evaluation_prepared = false;
            LAMMPS_NS::bigint graph_value_step = -1;
            void check_graph_values() const;

            
        public:
            using CV_Calculation = typename CV::CV_Calculation;
            using CV_BiasForce = typename CV::CV_BiasForce;
            STEIN_QL(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::FixMetadynamics *Fixmetad, 
                                FILE *f_check, 
                                std::string env_setNum, int group_id, int stein_l, 
                                MetaD_zqc::Steinhardt_env* my_env,
                                int d_block_size);
            ~STEIN_QL() override;
            virtual CV_Calculation set_CV_calculate(std::string func_name) override;
            virtual CV_BiasForce set_CV_bias_force(std::string func_name) override;
            // void cv_method();
            void base_calc() override;
            // Q 与 Local Q 共用逐中心值装配；坐标反向由各自实现。
            int graph_groupbit() const override { return my_env->groupbit; }
            void bind_assembly(std::shared_ptr<const CVAssembly>) override;
            void prepare_graph_evaluation() override;
            void assemble_values(GpuBuffer<double>&) override;
            void apply_adjoint(const GpuBuffer<double>&) override;

            virtual void compute_Q_peratoms();
            // void bias_force_LOC_AVE(double dVdcv);
            void summary(FILE* f) override;


            virtual void steinhardt_param_calc(double *);
            void call_steinhardt_cv_ql_i_kernel();
            void environment();
            // communication for Ghost atoms
            virtual int get_comm_reverse_width() const override; // 每原子 double 数
            virtual int pack_comm_reverse_ubuf(int n, int first, double *u_buf, 
                                int offset, int stride)  override; // 具体 CV 自己的打包逻辑
            virtual void unpack_comm_reverse_ubuf(int n, int *list, double *u_buf, 
                                int offset, int stride)  override; // 具体 CV 自己的解包逻辑
            virtual int get_comm_forward_width() const override;
            virtual int pack_comm_forward_ubuf(int n, int *list, double *u_buf, int offset, int stride) override;
            virtual void unpack_comm_forward_ubuf(int n, int first, double *u_buf, int offset, int stride) override;
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
            // double                                      cutoff_eps_r = 1e-12;
            using Steinhardt_env::cutoff_eps_r;
            int                                         cached_nall_ = -1;
            long long                                   last_group_refresh_lastcall_ = -1;
            int                                         cached_nlocal_for_group_ = -1;
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
            // // [d_calculated_firstneigh_ptrs] 1-D -> (nall+1) : [0, 12, 12+23, 12+23, ...]
            // // num_of_all_calc_fullpair = h_calculated_firstneigh_ptrs[num_of_all_IJ_atoms];
            // LAMMPS_NS::tagint                           num_of_all_calc_fullpair;
            // LAMMPS_NS::tagint                           *h_calculated_firstneigh_ptrs = nullptr;
            // GpuBuffer<LAMMPS_NS::tagint>                d_calculated_firstneigh_ptrs;
            using Steinhardt_env::num_of_all_calc_fullpair;
            using Steinhardt_env::h_calculated_firstneigh_ptrs;
            using Steinhardt_env::d_calculated_firstneigh_ptrs;
            // [LQ_mask] : [32bit int mask] * nmax
            // using Steinhardt_env::h_LQ_mask;
            // using Steinhardt_env::d_LQ_mask;
            int                                         *h_LQ_mask = nullptr;
            GpuBuffer<int>                              d_LQ_mask;
            // [d_pure_J_offsets] : temp array which mask j and give idx
            // we dont need to quest in host    
            GpuBuffer<int>                              d_pure_J_offsets;
            // // [neigh_in_switching] 1-D -> nmax (calctag): the sum of neigh_in_switching for calculated atoms
            // double                                         *h_neigh_in_switching = nullptr;
            // GpuBuffer<double>                              d_neigh_in_switching;
            using Steinhardt_env::h_neigh_in_switching;
            using Steinhardt_env::d_neigh_in_switching;
            
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
        using STEIN_QL<L>::my_env;

        using STEIN_QL<L>::block_num;
        using STEIN_QL<L>::d_block_size;
        using STEIN_QL<L>::N;
        using STEIN_QL<L>::stein_l;



        STEIN_LocalQL_env* my_loc_env;
        // [localid][2*(L+1)]：owned I 的局域平均球谐分量，m=0..L，实部/虚部交错。
        GpuBuffer<double>                               d_stein_LQlm;
        // calctag 的前 group_count 项为 owned I，后面为补齐的 J（可含 ghost）。
        // [calctag][2*(L+1)]：所有 owned I 汇总到 q_lm 的梯度前因子。
        GpuBuffer<double>                               d_dcvdx_rjk_prefix;
        using CV::assembly_;
        using STEIN_QL<L>::d_center_adjoint;
        using STEIN_QL<L>::check_graph_values;
        void evaluate_coordinate_adjoint();
        void accumulate_bias_force();
        // [owned center][2*(L+1)]：外层径向权重乘邻居 q_lm 的累加值，供 I→J 反向使用。
        GpuBuffer<double>                               sum_of_qlm_value_weights;
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
        using STEIN_QL<L>::h_a_virial;
        using STEIN_QL<L>::d_a_virial;

        public:
        using CV::register_buffer;
        using STEIN_QL<L>::environment;

        STEIN_LocalQL(LAMMPS_NS::LAMMPS *lmp,
                             LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check, 
                             std::string env_setNum, int group_id,
                             MetaD_zqc::Steinhardt_env* my_env,
                             int d_block_size);
        ~STEIN_LocalQL();

        void summary(FILE* f) override;

        // Leaf 的 dV/dLQ_i 按 CVAssembly 的 owned 顺序收集，再传播 I→J→K。
        void apply_adjoint(const GpuBuffer<double>&) override;
        void compute_Q_peratoms() override;
        void steinhardt_param_calc(double *) override;
        void call_steinhardt_Local_forward_kernel();
        void call_steinhardt_Local_backward_kernel();

        using CV_Calculation = typename CV::CV_Calculation;
        using CV_BiasForce = typename CV::CV_BiasForce;
        CV_Calculation set_CV_calculate(std::string func_name) override;
        CV_BiasForce set_CV_bias_force(std::string func_name) override;

        // 使用两层 ghost halo 完成 I→J→K；reverse 包含六分量 virial 和三分量坐标梯度。
        int get_comm_forward_width() const override { return 0; }
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
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, int start_idx,
    double cutoff_r, double cut_sigma_eps,
    // in
    int *d_group_indices, 
    LAMMPS_NS::tagint *d_group_numneigh,
    int *d_firstneigh_ptrs, double *d_x_flat,
    LAMMPS_NS::tagint *d_full_to_half,
    // out
    int *d_neigh_in_cutoff_r, int *d_active_pair_mask,
    double *d_neigh_in_switching,
    LAMMPS_NS::tagint *d_calculated_numneigh);


template <int L>
__global__ void steinhardt_cv_kernel(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, double cutoff_r, double cutoff_eps,
    int *d_group_indices,
    int *d_neigh_in_cutoff_r,
    LAMMPS_NS::tagint *d_group_numneigh,
    LAMMPS_NS::tagint *d_calculated_firstneigh_ptrs,
    LAMMPS_NS::tagint *d_calculated_numneigh,
    double *d_x_flat,
    double *d_neigh_in_switching,
    double *d_stein_qlm, double *d_stein_Ylm, double *d_stein_ql);


// Graph 梯度按 localid 输入；每个半对汇总两端的力和 virial。
template <int L>
__global__ void steinhardt_dcv_kernel(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    LAMMPS_NS::tagint pair_all, int groupbit,
    int *d_mask,
    LAMMPS_NS::tagint *d_active_pair_ids,
    LAMMPS_NS::tagint *d_half_pair_i, LAMMPS_NS::tagint *d_half_pair_j,
    double *d_x_flat,
    double *d_neigh_in_switching,
    double *d_stein_ql, double *d_stein_qlm,
    double *d_dcvdx, double *d_a_virial,
    const double *atom_adjoint
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
__global__ void steinhardt_Local_backward_ij_kernel(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    const double *center_adjoint,
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
    double *d_dcvdx, double *d_a_virial);

template <int L>
__global__ void steinhardt_Local_backward_jk_kernel(
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
    double *d_dcvdx, double *d_a_virial);


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