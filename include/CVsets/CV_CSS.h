#pragma once
#include "CV_Stru_factor.h"
#include "zqc_switch_function_device.cuh"
#include <string>
#include <vector>

namespace MetaD_zqc {

// 一个通道统计一种邻居类型；同一类型可用不同径向开关定义多个统计范围。
// n_ic = sum_{j: type[j]==neighbor_type} radial(r_ij)，中心由 CSS 的 group 选择。
// 通道按输入 CHANNEL 的顺序保存，Host/GPU 均为连续的 CSSChannel 结构体数组。
struct CSSChannel {
    int neighbor_type;             // LAMMPS 类型编号；输入标签在 create() 中解析为编号。
    SwitchFunctionRequest radial;  // 按值保存的径向开关类型及参数，供每个匹配邻居求值。
    double target;                 // 目标平滑配位数，与 n_ic 使用相同单位。
    double sigma;                  // 配位数容差，评分使用 (n_ic-target)/sigma。
    double cutoff;                 // 距离截断半径，由径向函数和 cutoff_eps 求得。
};

// 前向先以“中心 × 通道”为线程统计配位数，再以中心为线程合成 CSS 评分。
// geometry 借用共享环境的设备数组；geometry.group_count 只计本 rank 的 owned 中心。
// total_atom_count = nlocal + nghost，localid 位于 [0,total_atom_count)。所有指针参数均指向设备。
// atom_types[localid]：长度 total_atom_count 的 LAMMPS 类型编号数组，覆盖 owned 和 ghost。
// channels[channel_index]：长度 channel_count 的通道结构体数组。
// css_values[localid]：长度 total_atom_count，每个 owned 中心写入一个 CSS_i。
// coordination_numbers[channel_index*total_atom_count + localid]：长度 channel_count*total_atom_count。
// 配位数采用通道优先排列：一个通道的全部原子槽位连续，再接下一个通道。
// 例如 total_atom_count=8，第 3 个通道、localid=5 的配位数位于 [2*8+5]。
// 调用方先清零输出，kernel 覆盖 owned 中心的位置，组外及 ghost 槽位保留零值。
// 两个 kernel 按 stream 顺序执行；输入和输出缓冲区须保持有效直到该流的任务完成。
void css_forward(StruFactorKernelView geometry, const int* atom_types, int total_atom_count,
                 const CSSChannel* channels, int channel_count,
                 double* css_values, double* coordination_numbers, cudaStream_t stream = 0);

// 反向输入使用本次前向的几何、类型、通道、评分及配位数。
// center_adjoint[center_index]：长度 geometry.group_count，保存图传入的 dV/dCSS_i。
// center_index 是压缩中心序号，通过 geometry.d_group_indices[center_index] 得到 localid。
// atom_forces[3*localid+component]：长度 3*total_atom_count，分量顺序 x、y、z，累加物理力。
// atom_virials[6*localid+component]：长度 6*total_atom_count，顺序 xx、yy、zz、xy、xz、yz。
// 两种输出均逐原子连续存储；调用方按需清零，atom_virials=nullptr 时只计算力。
// 每线程处理一个中心，先累加自身贡献，再写回；共享邻居的贡献使用 atomicAdd。
// owned、组外邻居和 ghost 都可接收力；CV 层在下载后通过 reverse 通信归还 ghost 贡献。
// 本入口将 kernel 提交到 stream，调用方按同一流顺序或显式依赖安排后续使用。
void css_backward(StruFactorKernelView geometry, const int* atom_types, int total_atom_count,
                  const CSSChannel* channels, int channel_count,
                  const double* css_values, const double* coordination_numbers, const double* center_adjoint,
                  double* atom_forces, double* atom_virials, cudaStream_t stream = 0);

#ifndef ZQC_CSS_KERNELS_ONLY
// CSS_i = exp[-1/2 sum_c ((n_ic-target_c)/sigma_c)^2]。
// 同一 CSS 中的所有中心共享一份目标环境；不同中心环境可分别定义 CSS，再由图组合。
class CSS : public CV {
public:
    static CV* create(LAMMPS_NS::LAMMPS*, LAMMPS_NS::FixMetadynamics*, FILE*,
                      int argument_count, char** arguments, int& argument_index);
    CSS(LAMMPS_NS::LAMMPS*, LAMMPS_NS::FixMetadynamics*, FILE*, Stru_fact_env*,
        std::vector<CSSChannel>, std::vector<std::string>, double cutoff_epsilon);
    CV_Calculation set_CV_calculate(std::string) override;
    CV_BiasForce set_CV_bias_force(std::string) override;
    void summary(FILE*) override;
    void bind_assembly(std::shared_ptr<const CVAssembly>) override;
    int graph_groupbit() const override { return environment_->groupbit; }
    // 每轮图求值前使缓存失效；首个使用共享环境的 CV 更新几何。
    void prepare_graph_evaluation() override;
    void base_calc() override;
    // localid 布局的值装配到公共 Leaf 布局；反向从该布局提取本组的中心种子。
    void assemble_values(GpuBuffer<double>&) override;
    void apply_adjoint(const GpuBuffer<double>&) override;
    // 返回按 localid 排列的 Host 数组：css/value 为评分，n1、n2、... 为各通道配位数。
    double* get_peratom_ptr(const std::string&) override;
    // 每原子通信 9 个 double，顺序为 3 分量力、6 分量 virial。
    int get_comm_reverse_width() const override { return 9; }
    int pack_comm_reverse_ubuf(int, int, double*, int, int) override;
    void unpack_comm_reverse_ubuf(int, int*, double*, int, int) override;

private:
    // 借用 env_pool 管理的环境；同组、同最大 cutoff 的 CV 共享中心和邻居几何。
    Stru_fact_env* environment_;
    std::vector<CSSChannel> channels_;     // Host 配置，顺序与输入 CHANNEL 一致，长度为通道数。
    std::vector<std::string> radial_names_; // 与 channels_ 一一对应的开关名称，供日志输出。
    double cutoff_eps_;                   // 径向函数尾部阈值，用于求各通道的 cutoff。
    GpuBuffer<CSSChannel> d_channels_;     // channels_ 的 GPU 副本，构造时上传。
    GpuBuffer<int> d_atom_types_;          // [localid]，长度 total_atom_count_，每轮从 atom->type 上传。

    // GpuBuffer 由本 CSS 对象持有；以下长度均为有效元素数，capacity 可更大。
    // 值和配位数在 base_calc() 中按当前 nlocal+nghost 分配并清零。
    GpuBuffer<double> d_css_values_,          // [localid]，长度 total_atom_count_，owned 中心的 CSS_i。
                      d_coordination_numbers_, // [channel_index*total_atom_count_+localid]，各通道的 n_ic。
                      d_center_adjoint_,     // [center_index]，长度 group_count，由 assembly_ 提取的 dV/dCSS_i。
                      d_atom_forces_,        // [3*localid+component]，长度 3*total_atom_count_，反向力。
                      d_atom_virials_;       // [6*localid+component]，长度 6*total_atom_count_，反向 virial。

    // Host 副本沿用对应设备数组的排列；诊断数组按需下载，力/virial 每轮反向后下载。
    // 诊断数组在零原子时保留一个占位元素，逻辑长度仍由 total_atom_count_ 决定。
    // reverse 通信把 ghost 贡献累加回 Host owner 槽位，随后写入 LAMMPS。
    std::vector<double> h_css_values_, h_coordination_numbers_, h_atom_forces_, h_atom_virials_;
    bool graph_evaluation_prepared_ = false, // 本轮已经执行 prepare，base_calc 可复用其他 CV 更新的环境。
         values_ready_ = false;             // 当前步前向已提交，后续设备读取按默认流顺序执行。
    bool host_css_values_ready_ = false,     // 本轮评分的 Host 诊断副本已下载。
         host_coordination_numbers_ready_ = false; // 本轮所有通道配位数的 Host 副本已下载。
    LAMMPS_NS::bigint value_step_ = -1;       // 最近一次 base_calc 的 timestep。
    int total_atom_count_ = 0;               // 本轮 nlocal+nghost，同时是配位数各通道之间的跨度。
    void check_values() const;               // 核对前向状态、timestep 及原子总数。
};
#endif
} // namespace MetaD_zqc
