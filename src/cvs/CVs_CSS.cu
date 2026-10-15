#include "CV_CSS.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace MetaD_zqc {
namespace {
void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

// 每线程处理一个 (center_index, channel_index)，串行累加该中心符合类型的邻居。
// 线程编号按中心展开、通道变化最快；输出则按通道优先存储，便于逐通道诊断。
__global__ void count_kernel(StruFactorKernelView geometry, const int* atom_types, int total_atom_count,
                             const CSSChannel* channels, int channel_count, double* coordination_numbers) {
    const size_t c_channels_idx = static_cast<size_t>(blockIdx.x)*blockDim.x + threadIdx.x;
    const size_t c_channels_count = static_cast<size_t>(geometry.group_count)*channel_count;
    if (c_channels_idx >= c_channels_count) return;

    const int center_index = c_channels_idx / channel_count, channel_index = c_channels_idx % channel_count;
    const auto channel = channels[channel_index];
    const auto first_neighbor_index = geometry.d_group_numneigh[center_index];
    double coordination_number = 0.0;
    for (int neighbor_offset = 0; neighbor_offset < geometry.d_neigh_in_cutoff_r[center_index]; ++neighbor_offset) {
        const auto neighbor_index = first_neighbor_index + neighbor_offset,
            neighbor_local_id = geometry.d_calculated_numneigh[neighbor_index];
        if (atom_types[neighbor_local_id] != channel.neighbor_type) continue;
        const double distance_squared = geometry.d_group_dminneigh[4*neighbor_index+3];
        if (distance_squared > channel.cutoff*channel.cutoff) continue;
        coordination_number += SwitchFunctionMath::f(channel.radial, sqrt(distance_squared));
    }
    coordination_numbers[static_cast<size_t>(channel_index)*total_atom_count+geometry.d_group_indices[center_index]] = coordination_number;
}

// 一个中心的全部通道配位数到齐后，计算 CSS_i = exp(-0.5*sum_c deviation_c^2)。
// 前面的 count_kernel 与本 kernel 提交到同一流，保证依赖顺序。
__global__ void score_kernel(StruFactorKernelView geometry, int total_atom_count, const CSSChannel* channels,
                             int channel_count, const double* coordination_numbers, double* css_values) {
    const int center_index = blockIdx.x*blockDim.x + threadIdx.x;
    if (center_index >= geometry.group_count) return;
    const int center_local_id = geometry.d_group_indices[center_index];
    double mismatch_squared = 0.0;
    for (int channel_index = 0; channel_index < channel_count; ++channel_index) {
        const double normalized_deviation = (coordination_numbers[static_cast<size_t>(channel_index)*total_atom_count+center_local_id]-channels[channel_index].target)/channels[channel_index].sigma;
        mismatch_squared += normalized_deviation*normalized_deviation;
    }
    css_values[center_local_id] = exp(-0.5*mismatch_squared);
}

// 一线程对应一个 owned 中心：遍历邻居，并把命中通道的 dCSS_i/dr_ij 合并。
// 中心的力/virial 在寄存器中累加，邻居贡献通过 atomicAdd 写入各自 localid 槽位。
__global__ void backward_kernel(StruFactorKernelView geometry, const int* atom_types, int total_atom_count,
                                const CSSChannel* channels, int channel_count, const double* css_values,
                                const double* coordination_numbers, const double* center_adjoint,
                                double* atom_forces, double* atom_virials) {
    const int center_index = blockIdx.x*blockDim.x + threadIdx.x;
    if (center_index >= geometry.group_count) return;
    const int center_local_id = geometry.d_group_indices[center_index];
    if (center_adjoint[center_index] == 0.0 || css_values[center_local_id] == 0.0) return;
    double center_force[3] = {}, center_virial[6] = {};
    const auto first_neighbor_index = geometry.d_group_numneigh[center_index];
    for (int neighbor_offset = 0; neighbor_offset < geometry.d_neigh_in_cutoff_r[center_index]; ++neighbor_offset) {
        const auto neighbor_index = first_neighbor_index+neighbor_offset,
            neighbor_local_id = geometry.d_calculated_numneigh[neighbor_index];
        const double* neighbor_geometry = geometry.d_group_dminneigh+4*neighbor_index;
        const double neighbor_distance = sqrt(neighbor_geometry[3]);
        if (neighbor_distance == 0.0) continue;
        // 同一个邻居可能属于同类型的多个径向通道，共享本次读取的距离和位移。
        double score_distance_derivative = 0.0;
        for (int channel_index = 0; channel_index < channel_count; ++channel_index) {
            const auto channel = channels[channel_index];
            if (atom_types[neighbor_local_id] != channel.neighbor_type || neighbor_distance > channel.cutoff) continue;
            // dCSS/dn_c = -CSS * (n_c-target_c)/sigma_c^2。
            const double score_coordination_derivative = -css_values[center_local_id]*(coordination_numbers[static_cast<size_t>(channel_index)*total_atom_count+center_local_id]-channel.target)
                                 /channel.sigma/channel.sigma;
            score_distance_derivative += score_coordination_derivative*SwitchFunctionMath::df(channel.radial,
                neighbor_distance);
        }
        // 链式法则：force_scale=(dV/dCSS_i)*(dCSS_i/dr_ij)/r_ij。
        // neighbor_geometry 保存 r_j-r_i 及距离平方，所以 F_i=+force_scale*位移，F_j=-F_i。
        const double force_scale = center_adjoint[center_index]*score_distance_derivative/neighbor_distance;
        double pair_force[3];
        for (int component = 0; component < 3; ++component) {
            pair_force[component] = force_scale*neighbor_geometry[component];
            center_force[component] += pair_force[component];
            atomicAdd(atom_forces+3*neighbor_local_id+component,-pair_force[component]);
        }
        // 此有向项的总 virial 为 -(r_j-r_i) tensor F_i，两个端点各分配一半。
        // 六分量顺序与 LAMMPS 输出保持一致：xx、yy、zz、xy、xz、yz。
        if (atom_virials) {
            const double half_pair_virial[6] = {-.5*neighbor_geometry[0]*pair_force[0],
                -.5*neighbor_geometry[1]*pair_force[1],-.5*neighbor_geometry[2]*pair_force[2],
                                    -.5*neighbor_geometry[0]*pair_force[1],-.5*neighbor_geometry[0]*pair_force[2],
                                        -.5*neighbor_geometry[1]*pair_force[2]};
            for (int component = 0; component < 6; ++component) {
                center_virial[component] += half_pair_virial[component];
                atomicAdd(atom_virials+6*neighbor_local_id+component,half_pair_virial[component]);
            }
        }
    }
    // 当前中心也可能是其他中心的邻居，合并本线程的累计贡献时仍使用原子加。
    for (int component = 0; component < 3; ++component) atomicAdd(atom_forces+3*center_local_id+component,
        center_force[component]);
    if (atom_virials)
        for (int component = 0; component < 6; ++component) atomicAdd(atom_virials+6*center_local_id+component,
            center_virial[component]);
}

void check_arguments(StruFactorKernelView geometry, const int* atom_types, int total_atom_count,
                     const CSSChannel* channels, int channel_count, const double* css_values,
                         const double* coordination_numbers) {
    if (geometry.group_count < 0 || total_atom_count < geometry.group_count || channel_count <= 0 || !channels)
        throw std::invalid_argument("CSS: invalid kernel dimensions/channels");
    if (geometry.group_count && (!atom_types || !css_values || !coordination_numbers || !geometry.d_group_indices ||
                         !geometry.d_group_numneigh || !geometry.d_neigh_in_cutoff_r))
        throw std::invalid_argument("CSS: missing kernel data");
    if (static_cast<size_t>(geometry.group_count)*channel_count > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::length_error("CSS: too many center/channel pairs");
}
} // namespace

// 启动配位数和评分两个阶段；缓冲区已由调用方按当前原子数准备好。
void css_forward(StruFactorKernelView geometry, const int* atom_types, int total_atom_count,
                 const CSSChannel* channels, int channel_count, double* css_values, double* coordination_numbers,
                     cudaStream_t stream) {
    check_arguments(geometry,atom_types,total_atom_count,channels,channel_count,css_values,coordination_numbers);
    if (!geometry.group_count) return;
    const size_t c_channels_count = static_cast<size_t>(geometry.group_count)*channel_count;
    count_kernel<<<(c_channels_count+127)/128,128,0,stream>>>(geometry,atom_types,total_atom_count,channels,
        channel_count,coordination_numbers);
    check_cuda(cudaGetLastError());
    score_kernel<<<(geometry.group_count-1)/128+1,128,0,stream>>>(geometry,total_atom_count,channels,channel_count,
        coordination_numbers,css_values);
    check_cuda(cudaGetLastError());
}

// 启动中心反向 kernel，输出累加到调用方提供的力/virial 缓冲区。
void css_backward(StruFactorKernelView geometry, const int* atom_types, int total_atom_count,
                  const CSSChannel* channels, int channel_count, const double* css_values,
                      const double* coordination_numbers,
                  const double* center_adjoint, double* atom_forces, double* atom_virials, cudaStream_t stream) {
    check_arguments(geometry,atom_types,total_atom_count,channels,channel_count,css_values,coordination_numbers);
    if (!geometry.group_count) return;
    if (!center_adjoint || !atom_forces) throw std::invalid_argument("CSS: missing adjoint/force");
    backward_kernel<<<(geometry.group_count-1)/128+1,128,0,stream>>>(
        geometry,atom_types,total_atom_count,channels,channel_count,css_values,coordination_numbers,center_adjoint,
            atom_forces,atom_virials);
    check_cuda(cudaGetLastError());
}
} // namespace MetaD_zqc

#ifndef ZQC_CSS_KERNELS_ONLY
#include "atom.h"
#include "comm.h"
#include "group.h"
#include "neighbor.h"
#include "update.h"
#include "utils.h"
#include "zqc_cv_assembly.cuh"

namespace MetaD_zqc {
CV* CSS::create(LAMMPS_NS::LAMMPS* lmp, LAMMPS_NS::FixMetadynamics* metad_fix, FILE* log_file,
                int argument_count, char** arguments, int& argument_index) {
    auto* error = lmp->error;
    ERR_COND(argument_index+1 >= argument_count,"CSS requires a center group");
    const int group_id = lmp->group->find(arguments[argument_index+1]);
    ERR_COND(group_id < 0,"CSS: unknown center group");
    argument_index += 2;
    std::vector<CSSChannel> channels;
    std::vector<std::string> radial_names;
    double cutoff_epsilon = 1e-12;
    while (argument_index < argument_count) {
        const std::string key = arguments[argument_index];
        if (key == "CHANNEL") {
            // CHANNEL <邻居类型编号或标签> <径向开关名> <目标配位数> <容差>。
            // 例：CHANNEL 1 first 12 2.0，统计 type 1，目标配位数为 12，容差为 2。
            // 在 fix 命令前定义 labelmap atom 1 Fe 2 "Fe3+" 后，也可写：
            // CHANNEL Fe first 12 2.0 或 CHANNEL "Fe3+" first 6 1.0。
            // 标签在读取配置时转换为整数类型编号；单个原子的 atom ID 由 group id 选择。
            ERR_COND(argument_index+4 >= argument_count,"CSS CHANNEL requires: type radial_switch target sigma");
            CSSChannel channel{};
            channel.neighbor_type = LAMMPS_NS::utils::expand_type_int(
                FLERR,arguments[argument_index+1],LAMMPS_NS::Atom::ATOM,lmp);
            const auto radial_switch_entry = metad_fix->sw_registry.find(arguments[argument_index+2]);
            ERR_COND(radial_switch_entry == metad_fix->sw_registry.end(),"CSS: unknown radial switch");
            channel.radial = radial_switch_entry->second->params;
            channel.target = LAMMPS_NS::utils::numeric(FLERR,arguments[argument_index+3],false,lmp);
            channel.sigma = LAMMPS_NS::utils::numeric(FLERR,arguments[argument_index+4],false,lmp);
            ERR_COND(channel.neighbor_type < 1 || channel.neighbor_type > lmp->atom->ntypes,
                     "CSS: neighbor type is outside 1..ntypes");
            ERR_COND(!std::isfinite(channel.target) || channel.target < 0 ||
                     !std::isfinite(channel.sigma) || channel.sigma <= 0 ||
                     !std::isfinite(1.0/channel.sigma/channel.sigma),
                     "CSS: target must be finite and nonnegative; sigma must be positive and finite");
            ERR_COND(channel.radial.type != FERMI && channel.radial.type != TANH_TYPE &&
                     channel.radial.type != RATIONAL,"CSS: radial switch must be FERMI, TANH or RATIONAL");
            channels.push_back(channel);
            radial_names.emplace_back(arguments[argument_index+2]);
            argument_index += 5;
        } else if (key == "cutoff_eps") {
            ERR_COND(argument_index+1 >= argument_count,"CSS cutoff_eps requires a value");
            cutoff_epsilon = LAMMPS_NS::utils::numeric(FLERR,arguments[argument_index+1],false,lmp);
            argument_index += 2;
        } else break;
    }
    ERR_COND(channels.empty(),"CSS requires at least one CHANNEL");
    double max_cutoff_radius = 0.0;
    for (auto& channel : channels) {
        const auto support = SwitchFunction::query_radial_support(channel.radial,cutoff_epsilon);
        ERR_COND(!support.has_radius(),
            "CSS: radial switch has invalid finite support; check parameters and cutoff_eps");
        channel.cutoff = support.radius;
        ERR_COND(!std::isfinite(channel.cutoff*channel.cutoff),"CSS: radial cutoff is too large");
        max_cutoff_radius = std::max(max_cutoff_radius,channel.cutoff);
    }
    // 复用同组同 cutoff 的中心表、候选邻居和 GPU 几何筛选。
    StruFactorRequest environment_request{};
    environment_request.group_id = group_id;
    environment_request.cutoff_r = max_cutoff_radius;
    auto* shared_environment = Stru_fact_env::get_or_create(lmp,metad_fix,log_file,environment_request);
    NeighSpec neighbor_request;
    neighbor_request.full = 1; neighbor_request.ghost = 0; neighbor_request.use_cutoff = 1; neighbor_request.cutoff = max_cutoff_radius;
    shared_environment->neigh_id = metad_fix->neigh_hub.get_or_create(neighbor_request);
    lmp->comm->cutghostuser = std::max(lmp->comm->cutghostuser,max_cutoff_radius+lmp->neighbor->skin);
    return new CSS(lmp,metad_fix,log_file,shared_environment,std::move(channels),std::move(radial_names),
        cutoff_epsilon);
}

CSS::CSS(LAMMPS_NS::LAMMPS* lmp, LAMMPS_NS::FixMetadynamics* metad_fix, FILE* log_file,
    Stru_fact_env* shared_environment,
         std::vector<CSSChannel> channels, std::vector<std::string> radial_names, double cutoff_epsilon)
    : CV(lmp,metad_fix,log_file), environment_(shared_environment), channels_(std::move(channels)),
      radial_names_(std::move(radial_names)), cutoff_eps_(cutoff_epsilon) {
    register_buffer(d_channels_,"css_channels"); register_buffer(d_atom_types_,"css_types");
    register_buffer(d_css_values_,"css_values"); register_buffer(d_coordination_numbers_,"css_counts");
    register_buffer(d_center_adjoint_,"css_adjoint"); register_buffer(d_atom_forces_,"css_force");
    register_buffer(d_atom_virials_,"css_virial");
    // 通道参数在构造时固定，按输入顺序将结构体数组一次上传到设备。
    d_channels_.grow_to(channels_.size(),__FILE__,__LINE__);
    check_cuda(cudaMemcpy(d_channels_.ptr,channels_.data(),channels_.size()*sizeof(CSSChannel),cudaMemcpyHostToDevice));
}

CV::CV_Calculation CSS::set_CV_calculate(std::string) {
    throw std::runtime_error("CSS: use EXECUTOR GRAPH with RAW or AVE");
}
CV::CV_BiasForce CSS::set_CV_bias_force(std::string) {
    throw std::runtime_error("CSS: use EXECUTOR GRAPH for adjoint propagation");
}
void CSS::summary(FILE* output_file) {
    if (!output_file) return;
    std::fprintf(output_file,"CSS: center group=%d, channels=%zu, cutoff_eps=%g\n",
                 environment_->group_id,channels_.size(),cutoff_eps_);
    for (size_t channel_index = 0; channel_index < channels_.size(); ++channel_index) {
        const auto& channel = channels_[channel_index];
        std::fprintf(output_file,"  channel %zu: neighbor type=%d radial=%s target=%g sigma=%g cutoff=%g\n",
                     channel_index+1,channel.neighbor_type,radial_names_[channel_index].c_str(),channel.target,
                         channel.sigma,channel.cutoff);
    }
}
void CSS::bind_assembly(std::shared_ptr<const CVAssembly> assembly) {
    if (assembly && (assembly->groupbit() != environment_->groupbit ||
                     assembly->nlocal() != lmp->atom->nlocal || assembly->nghost() != lmp->atom->nghost))
        throw std::invalid_argument("CSS: incompatible atom assembly");
    environment_->atom_assembly = assembly;
    environment_->last_update_step = -1;
    values_ready_ = false;
    CV::bind_assembly(std::move(assembly));
}
// Graph 先对所有 CV 做准备，再依次前向；共享环境由首个使用者更新一次。
void CSS::prepare_graph_evaluation() {
    values_ready_ = false;
    graph_evaluation_prepared_ = true;
    environment_->last_update_step = -1;
}
void CSS::base_calc() {
    values_ready_ = false;
    host_css_values_ready_ = host_coordination_numbers_ready_ = false;
    if (!graph_evaluation_prepared_) environment_->last_update_step = -1;
    graph_evaluation_prepared_ = false;
    if (environment_->last_update_step != lmp->update->ntimestep) {
        environment_->refresh_lmpbox();
        environment_->get_env();
    }
    // total_atom_count_ 同时决定 localid 数组长度及每个配位数通道的跨度。
    // 前向只覆盖 owned 中心，因此先将所有原子槽位清零，再保存本轮评分和配位数。
    total_atom_count_ = lmp->atom->nlocal+lmp->atom->nghost;
    const size_t coordination_buffer_size = static_cast<size_t>(total_atom_count_)*channels_.size();
    d_atom_types_.grow_to(total_atom_count_,__FILE__,__LINE__);
    d_css_values_.grow_to(total_atom_count_,__FILE__,__LINE__);
    d_coordination_numbers_.grow_to(coordination_buffer_size,__FILE__,__LINE__);
    if (total_atom_count_) {
        check_cuda(cudaMemcpy(d_atom_types_.ptr,lmp->atom->type,total_atom_count_*sizeof(int),cudaMemcpyHostToDevice));
        check_cuda(cudaMemset(d_css_values_.ptr,0,total_atom_count_*sizeof(double)));
        check_cuda(cudaMemset(d_coordination_numbers_.ptr,0,coordination_buffer_size*sizeof(double)));
    }
    css_forward(kernel_view(environment_->data),d_atom_types_.ptr,total_atom_count_,d_channels_.ptr,
                channels_.size(),d_css_values_.ptr,d_coordination_numbers_.ptr);
    value_step_ = lmp->update->ntimestep;
    values_ready_ = true;
}
void CSS::check_values() const {
    if (!values_ready_ || value_step_ != lmp->update->ntimestep || total_atom_count_ != lmp->atom->nlocal+lmp->atom->nghost)
        throw std::logic_error("CSS: base_calc is required for the current atom layout");
}
// 将本组 owned 中心的 localid 值写到公共 Leaf 的 leaf_slots，供融合图读取。
void CSS::assemble_values(GpuBuffer<double>& leaf_values) {
    check_values();
    if (!assembly_) throw std::logic_error("CSS: missing atom assembly");
    assembly_->assemble_values(d_css_values_,leaf_values);
}
void CSS::apply_adjoint(const GpuBuffer<double>& leaf_adjoint) {
    check_values();
    if (!assembly_ || assembly_->nlocal() != lmp->atom->nlocal ||
        assembly_->nghost() != lmp->atom->nghost || assembly_->group_count() != environment_->data.group_count)
        throw std::logic_error("CSS: stale atom assembly");
    // 公共 Leaf 可容纳多个 CV 组；先按本组装配表提取紧凑排列的中心种子。
    d_center_adjoint_.grow_to(assembly_->group_count(),__FILE__,__LINE__);
    assembly_->gather_adjoint(leaf_adjoint,d_center_adjoint_);
    const size_t total_atom_count = total_atom_count_;
    d_atom_forces_.grow_to(3*total_atom_count,__FILE__,__LINE__); d_atom_virials_.grow_to(6*total_atom_count,
        __FILE__,__LINE__);
    h_atom_forces_.resize(3*total_atom_count); h_atom_virials_.resize(6*total_atom_count);
    if (total_atom_count) {
        check_cuda(cudaMemset(d_atom_forces_.ptr,0,3*total_atom_count*sizeof(double)));
        check_cuda(cudaMemset(d_atom_virials_.ptr,0,6*total_atom_count*sizeof(double)));
    }
    css_backward(kernel_view(environment_->data),d_atom_types_.ptr,total_atom_count_,d_channels_.ptr,channels_.size(),
                 d_css_values_.ptr,d_coordination_numbers_.ptr,d_center_adjoint_.ptr,d_atom_forces_.ptr,
                     d_atom_virials_.ptr);
    if (total_atom_count) {
        check_cuda(cudaMemcpy(h_atom_forces_.data(),d_atom_forces_.ptr,3*total_atom_count*sizeof(double),
            cudaMemcpyDeviceToHost));
        check_cuda(cudaMemcpy(h_atom_virials_.data(),d_atom_virials_.ptr,6*total_atom_count*sizeof(double),
            cudaMemcpyDeviceToHost));
    }
    // Host 副本收齐 ghost 的反向贡献后，将 owned 原子的力/virial 累加到 LAMMPS。
    Fixmetad->communicate_reverse(*this);
    for (int atom_local_id = 0; atom_local_id < lmp->atom->nlocal; ++atom_local_id) {
        for (int component = 0; component < 3; ++component) lmp->atom->f[atom_local_id][component] += h_atom_forces_[3*atom_local_id+component];
        if (Fixmetad->a_virial)
            for (int component = 0; component < 6; ++component) Fixmetad->a_virial[6*atom_local_id+component] += h_atom_virials_[6*atom_local_id+component];
    }
}
// 每个 ghost 按 [Fx,Fy,Fz,Wxx,Wyy,Wzz,Wxy,Wxz,Wyz] 打包；offset/stride 以 double 为单位。
int CSS::pack_comm_reverse_ubuf(int atom_count, int first_ghost_local_id, double* buffer, int offset, int stride) {
    for (int packed_atom_index = 0; packed_atom_index < atom_count; ++packed_atom_index) {
        for (int component = 0; component < 3; ++component) buffer[offset+packed_atom_index*stride+component] = h_atom_forces_[3*(first_ghost_local_id+packed_atom_index)+component];
        for (int component = 0; component < 6; ++component) buffer[offset+packed_atom_index*stride+3+component] = h_atom_virials_[6*(first_ghost_local_id+packed_atom_index)+component];
    }
    return 9;
}
// 接收表给出 owner 的 localid；来自不同 rank 的贡献累加到同一个 Host 槽位。
void CSS::unpack_comm_reverse_ubuf(int atom_count, int* owner_local_ids, double* buffer, int offset, int stride) {
    for (int packed_atom_index = 0; packed_atom_index < atom_count; ++packed_atom_index) {
        for (int component = 0; component < 3; ++component) h_atom_forces_[3*owner_local_ids[packed_atom_index]+component] += buffer[offset+packed_atom_index*stride+component];
        for (int component = 0; component < 6; ++component) h_atom_virials_[6*owner_local_ids[packed_atom_index]+component] += buffer[offset+packed_atom_index*stride+3+component];
    }
}
// 逐原子诊断按需下载 Host 副本，同一轮多次请求复用缓存；base_calc() 使缓存失效。
double* CSS::get_peratom_ptr(const std::string& property) {
    check_values();
    if (property == "css" || property == "value") {
        if (!host_css_values_ready_) {
            h_css_values_.resize(std::max(1,total_atom_count_));
            if (total_atom_count_) check_cuda(cudaMemcpy(h_css_values_.data(),d_css_values_.ptr,
                total_atom_count_*sizeof(double),cudaMemcpyDeviceToHost));
            host_css_values_ready_ = true;
        }
        return h_css_values_.data();
    }
    // n1、n2、... 对应声明顺序；仅在逐原子输出请求时下载通道配位数。
    if (property.size() > 1 && property[0] == 'n') {
        size_t parsed_length = 0;
        int channel_index;
        try { channel_index = std::stoi(property.substr(1),&parsed_length); }
        catch (const std::exception&) { return nullptr; }
        if (parsed_length != property.size()-1 || channel_index < 1 || channel_index > static_cast<int>(channels_.size())) return nullptr;
        --channel_index;
        if (!host_coordination_numbers_ready_) {
            h_coordination_numbers_.resize(std::max(size_t(1),static_cast<size_t>(total_atom_count_)*channels_.size()));
            if (total_atom_count_) check_cuda(cudaMemcpy(h_coordination_numbers_.data(),d_coordination_numbers_.ptr,
                total_atom_count_*channels_.size()*sizeof(double),cudaMemcpyDeviceToHost));
            host_coordination_numbers_ready_ = true;
        }
        return h_coordination_numbers_.data()+static_cast<size_t>(channel_index)*total_atom_count_;
    }
    return nullptr;
}
REGISTER_CV("CSS",MetaD_zqc::CSS::create);
} // namespace MetaD_zqc
#endif
