#pragma once
#include "zqc_cuda_execution.cuh"
#include "zqc_gpu_buffer.cuh"
#include "zqc_engine.h"

namespace MetaD_zqc {

// Stru_factor/CSS 共用的邻居几何数据，由 Stru_fact_env 长期持有。
// GpuBuffer 持有设备内存；下列长度均指有效元素数，capacity 是可复用的分配容量。
// 引擎适配器提供中心和候选段；Stru_fact_env 在 GPU 筛选后填写有效邻居及几何。
//
// 三种下标需要区分：
//   k    : 压缩后的中心序号，范围 [0, group_count)，只包含 owned 中心；
//   i/j  : 本地原子下标 localid；i 为中心，j 可为 owned/ghost 或选择集外邻居；
//   pair : 展平邻居数组中的槽位，三维位移和距离平方共享同一个 pair。
// localid < owned_atom_count 为 owned；[owned, total) 为 ghost。
// i/j 使用 LocalAtomIndex；全局稳定原子编号另用 AtomId。
//
// 第 k 个中心的有效邻居槽位为：
//   first = d_group_numneigh[k]
//   [first, first + d_neigh_in_cutoff_r[k])
// d_group_numneigh[k+1]-first 是为该中心保留的槽位数，可能大于有效邻居数；
// 因此遍历使用 d_neigh_in_cutoff_r[k]，不读取该段未使用的尾部。
// 环境负责保证中心编号唯一、邻居不含中心自身、localid 已去除特殊邻居标志。
struct StruFactorData {
    int group_count = 0; // 有效 owned 中心数，与 GpuBuffer.capacity 的预留容量分开
    GpuBuffer<LocalAtomIndex> d_group_indices;       // [k] -> 中心 localid，长度 group_count。
    GpuBuffer<LocalAtomIndex> d_group_numneigh;      // [k] -> 候选段起点，长度 group_count+1，末项为总槽位数。
    GpuBuffer<int>              d_neigh_in_cutoff_r;   // [k] -> 筛选后的有效邻居数，长度 group_count。
    GpuBuffer<LocalAtomIndex> d_calculated_numneigh; // [pair] -> 邻居 localid，长度为总槽位数，读取每段有效前缀。
    GpuBuffer<double>           d_group_dminneigh;     // [4*pair+c]，长度 4*总槽位数，顺序 dx、dy、dz、r^2。
    // 位移为 r_neighbor-r_center，每个邻居的四个 double 连续存放。
    // 例：段起点 [0,4,7]、有效数量 [2,1] 表示两中心分别使用槽位 [0,1] 和 [4]。

    StruFactorData() = default;
    // 独立验证的定长分配入口，无需宿主引擎实例。
    // center_count 是中心数；pair_count 是展平邻居的槽位总数，可包含段内空余槽位。
    // 构造时分配并清零缓冲区，实际中心编号、偏移、邻居和几何数据由调用方填写。
    StruFactorData(int center_count, size_t pair_count);
    // 禁止复制显存所有者，避免 GpuBuffer 被浅拷贝后重复释放。
    StruFactorData(const StruFactorData&) = delete;
    StruFactorData& operator=(const StruFactorData&) = delete;

    // 为所有缓冲区绑定名称和通用诊断回调。
    void set_logger(Logger logger = {});
    // 设置 group_count 后，按需申请中心表。
    // 保留 GpuBuffer 的容量复用策略；真正扩容时旧内容不保留，需要重新写入。
    // group_count 个中心编号对应 group_count+1 个邻居偏移，最后一项为总槽位数。
    void grow_centers(const char* code_file, int code_line) {
        if (group_count < 0) throw std::invalid_argument("Stru_factor: negative center count");
        const size_t count = static_cast<size_t>(group_count);
        d_group_indices.grow_to(count, code_file, code_line);
        d_group_numneigh.grow_to(count+1, code_file, code_line);
    }
    // pair_count 为邻居槽位总数；padded_centers 为中心计数数组的申请长度，
    // 至少应覆盖 group_count；当前环境按有效中心数申请。
    // 扩容与数据更新在环境准备阶段完成，前向/反向启动函数不会调用本方法。
    void grow_neighbors(size_t pair_count, size_t padded_centers,
                        const char* code_file, int code_line) {
        if (pair_count > std::numeric_limits<size_t>::max()/4)
            throw std::length_error("Stru_factor: neighbor geometry size overflow");
        d_group_dminneigh.grow_to(4*pair_count, code_file, code_line);
        d_neigh_in_cutoff_r.grow_to(padded_centers, code_file, code_line);
        d_calculated_numneigh.grow_to(pair_count, code_file, code_line);
    }
};

// kernel 按值接收的只读视图：复制中心数及设备地址，数组布局与 StruFactorData 相同。
// 底层 GpuBuffer 由环境持有，并在 kernel 完成前保持有效；扩容后应重新取得视图。
struct StruFactorKernelView {
    int group_count;
    const LocalAtomIndex* d_group_indices;
    const LocalAtomIndex* d_group_numneigh;
    const int* d_neigh_in_cutoff_r;
    const LocalAtomIndex* d_calculated_numneigh;
    const double* d_group_dminneigh;
};
// 每次启动前读取当前 .ptr，避免 GpuBuffer 扩容后仍使用旧指针。
// 返回值只在本次启动中使用，不持有缓冲区，也不产生 CUDA 操作。
inline StruFactorKernelView kernel_view(const StruFactorData& data) {
    return {data.group_count, data.d_group_indices.ptr, data.d_group_numneigh.ptr,
            data.d_neigh_in_cutoff_r.ptr, data.d_calculated_numneigh.ptr,
            data.d_group_dminneigh.ptr};
}


// 从引擎无关候选表构建本轮几何。坐标借用 CUDA 的 AoS/SoA 视图。
// 当前入口要求邻居采用显式周期镜像坐标；后端负责声明这一约定。
// centers/offsets 复制到 data，筛选后的邻居与位移由 data 持有。
void strufactor_prepare_geometry(StruFactorData& data, const NeighborListView& neighbors,
                                 ArrayView<const double> coordinates, double cutoff,
                                 cudaStream_t stream = 0, int block_size = 128);

// 原始前向：每个线程处理一个 owned 中心，计算 S_i = 1 + sum_j phi(r_ij)。
// q 是正的波数，cutoff_rsq 是正的截断半径平方；phi 包含结构因子的径向平滑。
// 此处不做 AVE、全局求和或 CV 值的开关函数变换。
//
// values 是设备数组，按 localid 写入 values[d_group_indices[k]]，不是 values[k]。
// 数组长度必须覆盖全部中心 localid；只覆盖中心位置，其余位置由调用方初始化。
// 后续由 CVAssembly 把这些原子索引的值装配到连续的 Leaf 数组。
//
// data 按引用借用，启动时只提取当前设备指针，不复制/扩容/同步数组。
// group_count==0 时不启动 kernel；参数检查仍然执行。
// 启动检查只检查提交错误，不等待 GPU 完成；调用方负责 stream 依赖和内存生命周期，
// 并保证前向、反向之间坐标、邻居布局及相关参数一致。
void strufactor_forward(const StruFactorData& data, double q, double cutoff_rsq,
                        double* values, cudaStream_t stream = 0, int block_size = 128);
// 生产路径：values 按 localid 寻址，保留 item/component stride。
void strufactor_forward(const StruFactorData& data, double q, double cutoff_rsq,
                        ArrayView<double> values, cudaStream_t stream = 0, int block_size = 128);

// 原始反向：把图对 CV 值的导数转换为坐标力，F_j = -sum_i (dV/dS_i)*(dS_i/dr_j)。
// center_adjoint 是长度至少为 group_count 的设备标量数组，使用压缩中心序号 k；
// center_adjoint[k] 对应中心 i=d_group_indices[k] 的 dV/dS_i，不是 xyz 三分量梯度。
// 它由 CVAssembly 从 Leaf.adj 按中心顺序提取，已包含图中 AVE/开关函数/多输出的贡献。
// 本函数不重复乘 dV/dDIM、不除全局原子数，也不执行 MPI 通信。
//
// 输出均为累加（+=），使用 localid 排列，须覆盖 owned+ghost 个原子：
//   force [localid,3]：物理力 -dV/dr_j；
//   virial[localid,6]：逐原子 Symmetric6 virial，顺序 xx、yy、zz、xy、xz、yz。
// virial 可为空视图或指针 nullptr，此时跳过全部 virial 计算。需要独立结果时调用方先清零；
// 若要汇总多份贡献，也可在已有输出上继续累加。
// 选择集外邻居及 ghost 同样获得贡献；调用方完成 ghost 的 reverse communication，
// 再将结果交回宿主。此层不会判断邻居是否属于中心选择集。
// 异步执行和输入生命周期要求与前向相同。
void strufactor_backward(const StruFactorData& data, double q, double cutoff_rsq,
                         const double* center_adjoint, double* force, double* virial,
                         cudaStream_t stream = 0, int block_size = 128);
void strufactor_backward(const StruFactorData& data, double q, double cutoff_rsq,
                         ArrayView<const double> center_adjoint, ArrayView<double> force,
                         ArrayView<double> virial, cudaStream_t stream = 0, int block_size = 128);

// 化学加权前后向：S_i = 1 + sum_j w(type_i)*w(type_j)*phi(r_ij)。
// 保留常数自项 1；原子类型在本轮前后向之间保持固定。
// atom_types 按 localid 借用，可含非 1 的 item_stride；type_weights[type] 长度为
// atom_type_begin()+atom_type_count()，有效类型编号落在 [begin, begin+count)。
// 两个参数均借用设备数组；取类型时使用与 geometry 同槽位的筛选后邻居编号。
// values、center_adjoint、force、virial 的布局与上面的普通版本相同。
// 指针重载假定类型/值为连续 localid 排列，供既有独立 benchmark 使用。
void strufactor_chem_forward(const StruFactorData& data, double q, double cutoff_rsq,
                             const int* atom_types, const double* type_weights,
                             double* values, cudaStream_t stream = 0, int block_size = 128);
void strufactor_chem_forward(const StruFactorData& data, double q, double cutoff_rsq,
                             ArrayView<const int> atom_types, ArrayView<const double> type_weights,
                             ArrayView<double> values, cudaStream_t stream = 0, int block_size = 128);
void strufactor_chem_backward(const StruFactorData& data, double q, double cutoff_rsq,
                              const int* atom_types, const double* type_weights,
                              const double* center_adjoint, double* force, double* virial,
                              cudaStream_t stream = 0, int block_size = 128);
void strufactor_chem_backward(const StruFactorData& data, double q, double cutoff_rsq,
                              ArrayView<const int> atom_types, ArrayView<const double> type_weights,
                              ArrayView<const double> center_adjoint, ArrayView<double> force,
                              ArrayView<double> virial, cudaStream_t stream = 0, int block_size = 128);

} // namespace MetaD_zqc

// 完整插件的 CV 与环境接口。
#ifndef ZQC_STRUFACTOR_KERNELS_ONLY
#include "zqc_cv.h"
#include "zqc_cv_assembly.cuh"
#include "zqc_virial.h"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace MetaD_zqc {
struct StruFactorRequest {
    SelectionId selection = all_atoms;
    std::string cal_name;
    std::string selection_name; // 输入中的选择集名称，仅用于日志；运行时以 selection 为准。
    double cutoff_r = 8.0;
    double q_factor = 1.0;
    int d_block_size = 128;
    bool use_chemical_lock = false;
    double c_target = 0.0;
    double sigma = 1.0;
    bool use_custom_weight = false;
    // Chem_map 的键是引擎原生类型编号，落在 [atom_type_begin, begin+count)。
    std::map<int, double> custom_weights;
};

// 同选择集、同 cutoff 的 CV 共享中心表和邻居几何数据。
// 环境池按 EngineAdapter* 区分所有者；Fix 析构时 clear_pool(&engine) 释放本适配器环境。
// CV_info::engine 为长期引用，寿命覆盖借用本环境的全部 CV。
class Stru_fact_env : public CV_info {
public:
    static std::map<std::string, Stru_fact_env*> env_pool; // 共享环境的持有者，clear_pool() 统一释放。
    double cutoff_r;                        // 共享几何筛选半径，CSS 使用各通道 cutoff 的最大值。
    SelectionId selection_id_ = all_atoms;  // 中心选择集编号；-1 表示全部原子。
    std::string selection_name_;            // 构造时的选择集名称副本，供 summary 输出。
    int last_group_count = 0;               // 上次刷新前的 owned 中心数。
    CudaExecutionStream execution; // 环境和共享 scratch 的串行提交流。
    StruFactorData data;                    // 持有压缩中心、邻居段和几何的设备缓冲。
    // 共享本组装配表；取 local_ids() 前 group_count 项为中心。装配表由 Graph 侧持有。
    std::shared_ptr<const CVAssembly> atom_assembly;
    int d_block_size = 128, GPU_number = 0, block_num = 0; // 每 block 线程数、设备编号、中心 kernel 的 block 数。
    size_t N = 0;                           // 补齐到 block 的线程数：d_block_size*block_num。
    int neighbor_handle = -1; // 通用 NeighborManager 的请求句柄。
    // 与适配器求值轮次绑定；同 timestep 的重复 begin_evaluation 也会使几何失效。
    std::uint64_t geometry_evaluation = 0;

    Stru_fact_env(CVContext& context, SelectionId selection, std::string selection_name,
                  double cutoff_r);
    virtual ~Stru_fact_env();
    static Stru_fact_env* get_or_create(CVContext& context, StruFactorRequest req);
    // owner==nullptr 清空全部；否则只释放 &env->engine==owner 的条目。由 Fix/父层调用。
    static void clear_pool(EngineAdapter* owner = nullptr);
    std::string get_env_key() const;
    virtual void refresh_centers(); // 刷新 owned 中心表并核对 Leaf 装配顺序。
    void get_env();                 // 更新邻居位移、距离及有效数量。
};

class Stru_factor : public CV {
    friend class Stru_factor_chem;
    friend class Weighted_chem_pair;
private:
    Stru_fact_env* my_env; // 借用 env_pool 中的共享环境；不负责 delete。
    int d_block_size;      // 前向/反向中心 kernel 的每 block 线程数。
    double q_factor;      // sinc(q*r) 的波数，单位为距离单位的倒数。

    // 令 total=engine.total_atom_count()；以下长度均为有效元素数。
    // 原始值数组按 localid 存储，前向先清零，再覆盖 owned 中心位置。
    // Host 副本由本 CV 的 vector 持有；设备数组由各 GpuBuffer 持有，寿命随本对象。
    std::vector<double> h_stru_factor; // [localid]，逻辑长度 total；请求诊断时下载。
    GpuBuffer<double> d_stru_factor,   // [localid]，长度 total，保存 S_i=1+sum_j phi(r_ij)。
                      d_center_adjoint; // [k]，长度 group_count，从公共 Leaf 布局提取的 dV/dS_i。
    GpuBuffer<double> d_graph_force,  // [localid,3]，长度 3*total，xyz 逐原子 AoS。
                      d_graph_virial; // [localid,6]，长度 6*total，Symmetric6。
    // 反向输出每轮清零，提交给适配器完成 ghost reverse 和 owned 回写。
    bool raw_values_ready = false;         // 本轮原始值计算已提交到环境流。
    bool graph_evaluation_prepared = false; // 本轮图求值已完成共享环境的失效准备。
    std::uint64_t raw_value_evaluation = 0; // 最近一次 base_calc 的 evaluation_id。
    void check_graph_values() const;

public:
    static CV* create(CVContext& context, int narg, char** arg, int& i);
    Stru_factor(CVContext& context, Stru_fact_env* env, double q_factor, int block_size);
    ~Stru_factor() override;
    CV_Calculation set_CV_calculate(std::string) override;
    CV_BiasForce set_CV_bias_force(std::string) override;
    void summary(FILE*) override;

    ViewReadiness device_readiness() const override { return my_env->execution.ready(); }
    void base_calc() override;
    void bind_assembly(std::shared_ptr<const CVAssembly>) override;
    SelectionId graph_selection() const override { return my_env->selection_id_; }
    void prepare_graph_evaluation() override;
    // 调用方先为 Leaf 分配目标缓冲，再传入本轮视图；不可与源缓冲别名。
    void assemble_values(ArrayView<double>) override;
    void apply_adjoint(ArrayView<const double>) override;
    void environment();
    void compute_stru_factor_peratoms();
    virtual void call_structure_factor_cv_kernel();
    // 普通版与化学版选择各自的数值 kernel，共用装配、输出缓冲和 reverse 通信。
    virtual void call_structure_factor_adjoint_kernel(ArrayView<const double> center_adjoint,
                                                      ArrayView<double> atom_forces,
                                                      ArrayView<double> atom_virials);
    // 返回 Host、按 localid 排列、逻辑长度为 total_atom_count 的只读视图；所有者为本 CV。
    ArrayView<const double> peratom_values(const std::string&) override;
};

// 化学环境额外持有类型及其权重；共享键包含全部 Chem_map 配置。
// h_type_weights 由本环境 vector 持有；长度为 begin+count，下标为原生类型编号。
// d_type_weights 为其设备副本，构造时上传后不变。device_atom_types 每轮向引擎借用。
class Stru_fact_chem_env : public Stru_fact_env {
public:
    double c_target, sigma; // 未在 Chem_map 中指定的类型使用 exp(-0.5*((type-c_target)/sigma)^2)。
    std::vector<double> h_type_weights; // [type]，长度 begin+count；[0,begin) 为占位零。
    GpuBuffer<double> d_type_weights;   // 上述固定权重的设备副本，允许负权重。
    // 借用适配器本轮 owned+ghost 类型表；可能带非 1 的 item_stride，kernel 按视图寻址。
    ArrayView<const int> device_atom_types;
    Stru_fact_chem_env(CVContext& context, SelectionId selection, std::string selection_name,
                       double cutoff_r, double c_target, double sigma,
                       const std::map<int, double>& custom_weights);
    ~Stru_fact_chem_env() override;
    void refresh_centers() override;
};

// 化学版复用 Stru_factor 的 Graph 生命周期：原始值 -> Leaf -> 逐中心 adjoint -> 坐标力。
// AVE/MEAN 由图节点执行；前后向仅增加固定的元素对权重 w_i*w_j。
class Stru_factor_chem : public Stru_factor {
private:
    Stru_fact_chem_env* my_chem_env; // 借用 env_pool 中的化学环境，包含 owned+ghost 类型。
public:
    Stru_factor_chem(CVContext& context, Stru_fact_chem_env* env, double q_factor, int block_size);
    void call_structure_factor_cv_kernel() override;
    void call_structure_factor_adjoint_kernel(ArrayView<const double> center_adjoint,
                                              ArrayView<double> atom_forces,
                                              ArrayView<double> atom_virials) override;
};
} // namespace MetaD_zqc

#endif // ZQC_STRUFACTOR_KERNELS_ONLY
