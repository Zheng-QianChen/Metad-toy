#pragma once
#include "CV_Stru_factor.h"
#include "zqc_switch_function_device.cuh"
#include <cstdint>
#include <string>
#include <vector>

namespace MetaD_zqc {

// 一个通道统计一种邻居类型；同一类型可用不同径向开关定义多个统计范围。
// n_ic = sum_{j: type[j]==neighbor_type} radial(r_ij)，中心由 CSS 的选择集决定。
// 通道按输入 CHANNEL 的顺序保存，Host/GPU 均为连续的 CSSChannel 结构体数组。
struct CSSChannel {
    int neighbor_type;             // 引擎原生类型编号；输入标签在 create() 中经 atom_type_id 解析。
    SwitchFunctionRequest radial;  // 按值保存的径向开关类型及参数，供每个匹配邻居求值。
    double target;                 // 目标平滑配位数，与 n_ic 使用相同单位。
    double sigma;                  // 配位数容差，评分使用 (n_ic-target)/sigma。
    double cutoff;                 // 距离截断半径，由径向函数和 cutoff_eps 求得。
};

// 前向先以“中心 × 通道”为线程统计配位数，再以中心为线程合成 CSS 评分。
// geometry 借用共享环境的设备数组；geometry.group_count 只计本 rank 的 owned 中心。
// total_atom_count = owned+ghost，localid 位于 [0,total_atom_count)。
// atom_types 按 localid 借用，可含非 1 的 item_stride。
// channels[channel_index]：长度 channel_count 的通道结构体数组。
// css_values 按 localid 写入每个 owned 中心的 CSS_i。
// coordination_numbers 采用通道优先：channel_index 为慢变维，localid 为快变维；
// 逻辑下标为 channel_index*total_atom_count + localid。
// 调用方先清零输出，kernel 覆盖 owned 中心的位置，选择集外及 ghost 槽位保留零值。
// 两个 kernel 按 stream 顺序执行；输入和输出缓冲区须保持有效直到该流的任务完成。
void css_forward(StruFactorKernelView geometry, const int* atom_types, int total_atom_count,
                 const CSSChannel* channels, int channel_count,
                 double* css_values, double* coordination_numbers, cudaStream_t stream = 0);
void css_forward(StruFactorKernelView geometry, ArrayView<const int> atom_types, int total_atom_count,
                 const CSSChannel* channels, int channel_count,
                 ArrayView<double> css_values, ArrayView<double> coordination_numbers,
                 cudaStream_t stream = 0);

// 反向输入使用本次前向的几何、类型、通道、评分及配位数。
// center_adjoint[center_index]：长度 geometry.group_count，保存图传入的 dV/dCSS_i。
// center_index 是压缩中心序号，通过 geometry.d_group_indices[center_index] 得到 localid。
// atom_forces[localid,3]、atom_virials[localid,6]：累加 Symmetric6 力/virial。
// 两种输出均按视图 stride 寻址；调用方按需清零，virial 为空时只计算力。
// 每线程处理一个中心，先累加自身贡献，再写回；共享邻居的贡献使用 atomicAdd。
// owned、选择集外邻居和 ghost 都可接收力；CV 层通过 reverse 通信归还 ghost 贡献。
void css_backward(StruFactorKernelView geometry, const int* atom_types, int total_atom_count,
                  const CSSChannel* channels, int channel_count,
                  const double* css_values, const double* coordination_numbers, const double* center_adjoint,
                  double* atom_forces, double* atom_virials, cudaStream_t stream = 0);
void css_backward(StruFactorKernelView geometry, ArrayView<const int> atom_types, int total_atom_count,
                  const CSSChannel* channels, int channel_count,
                  ArrayView<const double> css_values, ArrayView<const double> coordination_numbers,
                  ArrayView<const double> center_adjoint, ArrayView<double> atom_forces,
                  ArrayView<double> atom_virials, cudaStream_t stream = 0);

#ifndef ZQC_CSS_KERNELS_ONLY
// CSS_i = exp[-1/2 sum_c ((n_ic-target_c)/sigma_c)^2]。
// 同一 CSS 中的所有中心共享一份目标环境；不同中心环境可分别定义 CSS，再由图组合。
class CSS : public CV {
public:
    static CV* create(CVContext& context, int argument_count, char** arguments, int& argument_index);
    CSS(CVContext& context, Stru_fact_env* shared_environment,
        std::vector<CSSChannel> channels, std::vector<std::string> radial_names, double cutoff_epsilon);
    CV_Calculation set_CV_calculate(std::string) override;
    CV_BiasForce set_CV_bias_force(std::string) override;
    void summary(FILE*) override;
    void bind_assembly(std::shared_ptr<const CVAssembly>) override;
    SelectionId graph_selection() const override { return environment_->selection_id_; }
    // 每轮图求值前使缓存失效；首个使用共享环境的 CV 更新几何。
    void prepare_graph_evaluation() override;
    ViewReadiness device_readiness() const override { return environment_->execution.ready(); }
    void base_calc() override;
    // 调用方先分配 Leaf 目标缓冲再取视图；反向从该布局提取本组的中心种子。
    void assemble_values(ArrayView<double>) override;
    void apply_adjoint(ArrayView<const double>) override;
    // Host 只读诊断：css/value 为评分；n1、n2、... 为各通道配位数。
    // 通道视图逻辑长度为 total_atom_count_，item_stride=1；底层为通道优先大缓冲的一段。
    ArrayView<const double> peratom_values(const std::string&) override;

private:
    // 借用 env_pool 管理的环境；同选择集、同最大 cutoff 的 CV 共享中心和邻居几何。
    Stru_fact_env* environment_;
    std::vector<CSSChannel> channels_;     // Host 配置，顺序与输入 CHANNEL 一致，长度为通道数。
    std::vector<std::string> radial_names_; // 与 channels_ 一一对应的开关名称，供日志输出。
    double cutoff_eps_;                   // 径向函数尾部阈值，用于求各通道的 cutoff。
    GpuBuffer<CSSChannel> d_channels_;     // channels_ 的 GPU 副本，构造时上传。
    // 借用适配器本轮共享类型表，[localid] 覆盖 owned+ghost；可能带非单位 stride。
    ArrayView<const int> device_atom_types_;

    // GpuBuffer 由本 CSS 对象持有；以下长度均为有效元素数，capacity 可更大。
    // 值和配位数在 base_calc() 中按当前 total_atom_count 分配并清零。
    GpuBuffer<double> d_css_values_,          // [localid]，长度 total_atom_count_，owned 中心的 CSS_i。
                      d_coordination_numbers_, // [channel*total + localid]，各通道的 n_ic。
                      d_center_adjoint_,     // [center_index]，长度 group_count，由 assembly_ 提取的 dV/dCSS_i。
                      d_atom_forces_,        // [localid,3]，反向力。
                      d_atom_virials_;       // [localid,6]，Symmetric6 反向 virial。

    // Host 诊断副本沿用对应设备数组的排列，按需下载。力和 virial 的回写暂存由适配器持有。
    // 诊断数组在零原子时保留一个占位元素，逻辑长度仍由 total_atom_count_ 决定。
    std::vector<double> h_css_values_, h_coordination_numbers_;
    bool graph_evaluation_prepared_ = false, // 本轮已经执行 prepare，base_calc 可复用其他 CV 更新的环境。
         values_ready_ = false;             // 当前步前向已提交，后续读取在环境流或经事件依赖执行。
    bool host_css_values_ready_ = false,     // 本轮评分的 Host 诊断副本已下载。
         host_coordination_numbers_ready_ = false; // 本轮所有通道配位数的 Host 副本已下载。
    std::uint64_t value_evaluation_ = 0;     // 最近一次 base_calc 的 evaluation_id。
    int total_atom_count_ = 0;               // 本轮 owned+ghost，同时是配位数各通道之间的跨度。
    void check_values() const;               // 核对前向状态、evaluation_id 及原子总数。
};
#endif
} // namespace MetaD_zqc
