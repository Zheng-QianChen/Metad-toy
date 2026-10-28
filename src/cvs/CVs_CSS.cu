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

ArrayView<const int> unchecked_types(const int* data) {
    ArrayView<const int> view{};
    view.data = data;
    view.item_count = data ? 1 : 0;
    view.component_count = 1;
    view.item_stride = 1;
    view.component_stride = 1;
    view.storage_size = data ? std::numeric_limits<size_t>::max()/2 : 0;
    view.memory = MemorySpace::Cuda;
    return view;
}
ArrayView<double> unchecked_scalars(double* data) {
    ArrayView<double> view{};
    view.data = data;
    view.item_count = data ? 1 : 0;
    view.component_count = 1;
    view.item_stride = 1;
    view.component_stride = 1;
    view.storage_size = data ? std::numeric_limits<size_t>::max()/2 : 0;
    view.memory = MemorySpace::Cuda;
    return view;
}
ArrayView<const double> unchecked_scalars(const double* data) {
    ArrayView<const double> view{};
    view.data = data;
    view.item_count = data ? 1 : 0;
    view.component_count = 1;
    view.item_stride = 1;
    view.component_stride = 1;
    view.storage_size = data ? std::numeric_limits<size_t>::max()/2 : 0;
    view.memory = MemorySpace::Cuda;
    return view;
}
ArrayView<double> unchecked_xyz(double* data) {
    ArrayView<double> view{};
    view.data = data;
    view.item_count = data ? 1 : 0;
    view.component_count = 3;
    view.item_stride = 3;
    view.component_stride = 1;
    view.storage_size = data ? std::numeric_limits<size_t>::max()/2 : 0;
    view.memory = MemorySpace::Cuda;
    return view;
}
ArrayView<double> unchecked_virial6(double* data) {
    ArrayView<double> view{};
    view.data = data;
    view.item_count = data ? 1 : 0;
    view.component_count = 6;
    view.item_stride = 6;
    view.component_stride = 1;
    view.storage_size = data ? std::numeric_limits<size_t>::max()/2 : 0;
    view.memory = MemorySpace::Cuda;
    return view;
}
ArrayView<double> empty_virial() {
    ArrayView<double> view{};
    view.component_count = 6;
    view.item_stride = 6;
    view.component_stride = 1;
    view.memory = MemorySpace::Cuda;
    return view;
}

// 每线程处理一个 (center_index, channel_index)，串行累加该中心符合类型的邻居。
// 线程编号按中心展开、通道变化最快；输出则按通道优先存储，便于逐通道诊断。
__global__ void count_kernel(StruFactorKernelView geometry, ArrayView<const int> atom_types,
                             int total_atom_count, const CSSChannel* channels, int channel_count,
                             ArrayView<double> coordination_numbers) {
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
        if (atom_types(neighbor_local_id) != channel.neighbor_type) continue;
        const double distance_squared = geometry.d_group_dminneigh[4*neighbor_index+3];
        if (distance_squared > channel.cutoff*channel.cutoff) continue;
        coordination_number += SwitchFunctionMath::f(channel.radial, sqrt(distance_squared));
    }
    // 通道优先：item = channel*total + localid，component=0。
    coordination_numbers(static_cast<size_t>(channel_index)*total_atom_count +
                         geometry.d_group_indices[center_index]) = coordination_number;
}

__global__ void score_kernel(StruFactorKernelView geometry, int total_atom_count, const CSSChannel* channels,
                             int channel_count, ArrayView<const double> coordination_numbers,
                             ArrayView<double> css_values) {
    const int center_index = blockIdx.x*blockDim.x + threadIdx.x;
    if (center_index >= geometry.group_count) return;
    const int center_local_id = geometry.d_group_indices[center_index];
    double mismatch_squared = 0.0;
    for (int channel_index = 0; channel_index < channel_count; ++channel_index) {
        const double normalized_deviation =
            (coordination_numbers(static_cast<size_t>(channel_index)*total_atom_count+center_local_id)
             -channels[channel_index].target)/channels[channel_index].sigma;
        mismatch_squared += normalized_deviation*normalized_deviation;
    }
    css_values(center_local_id) = exp(-0.5*mismatch_squared);
}

__global__ void backward_kernel(StruFactorKernelView geometry, ArrayView<const int> atom_types,
                                int total_atom_count, const CSSChannel* channels, int channel_count,
                                ArrayView<const double> css_values,
                                ArrayView<const double> coordination_numbers,
                                ArrayView<const double> center_adjoint,
                                ArrayView<double> atom_forces, ArrayView<double> atom_virials) {
    const int center_index = blockIdx.x*blockDim.x + threadIdx.x;
    if (center_index >= geometry.group_count) return;
    const int center_local_id = geometry.d_group_indices[center_index];
    if (center_adjoint(center_index) == 0.0 || css_values(center_local_id) == 0.0) return;
    double center_force[3] = {}, center_virial[6] = {};
    const auto first_neighbor_index = geometry.d_group_numneigh[center_index];
    const bool write_virial = atom_virials.data != nullptr;
    for (int neighbor_offset = 0; neighbor_offset < geometry.d_neigh_in_cutoff_r[center_index]; ++neighbor_offset) {
        const auto neighbor_index = first_neighbor_index+neighbor_offset,
            neighbor_local_id = geometry.d_calculated_numneigh[neighbor_index];
        const double* neighbor_geometry = geometry.d_group_dminneigh+4*neighbor_index;
        const double neighbor_distance = sqrt(neighbor_geometry[3]);
        if (neighbor_distance == 0.0) continue;
        double score_distance_derivative = 0.0;
        for (int channel_index = 0; channel_index < channel_count; ++channel_index) {
            const auto channel = channels[channel_index];
            if (atom_types(neighbor_local_id) != channel.neighbor_type || neighbor_distance > channel.cutoff) continue;
            const double score_coordination_derivative =
                -css_values(center_local_id)*(coordination_numbers(static_cast<size_t>(channel_index)*total_atom_count+center_local_id)-channel.target)
                 /channel.sigma/channel.sigma;
            score_distance_derivative += score_coordination_derivative*SwitchFunctionMath::df(channel.radial,
                neighbor_distance);
        }
        const double force_scale = center_adjoint(center_index)*score_distance_derivative/neighbor_distance;
        double pair_force[3];
        for (int component = 0; component < 3; ++component) {
            pair_force[component] = force_scale*neighbor_geometry[component];
            center_force[component] += pair_force[component];
            atomicAdd(&atom_forces(neighbor_local_id, component), -pair_force[component]);
        }
        if (write_virial) {
            const double half_pair_virial[6] = {-.5*neighbor_geometry[0]*pair_force[0],
                -.5*neighbor_geometry[1]*pair_force[1],-.5*neighbor_geometry[2]*pair_force[2],
                                    -.5*neighbor_geometry[0]*pair_force[1],-.5*neighbor_geometry[0]*pair_force[2],
                                        -.5*neighbor_geometry[1]*pair_force[2]};
            for (int component = 0; component < 6; ++component) {
                center_virial[component] += half_pair_virial[component];
                atomicAdd(&atom_virials(neighbor_local_id, component), half_pair_virial[component]);
            }
        }
    }
    for (int component = 0; component < 3; ++component)
        atomicAdd(&atom_forces(center_local_id, component), center_force[component]);
    if (write_virial)
        for (int component = 0; component < 6; ++component)
            atomicAdd(&atom_virials(center_local_id, component), center_virial[component]);
}

void check_arguments(StruFactorKernelView geometry, ArrayView<const int> atom_types, int total_atom_count,
                     const CSSChannel* channels, int channel_count,
                     ArrayView<const double> css_values, ArrayView<const double> coordination_numbers) {
    if (geometry.group_count < 0 || total_atom_count < geometry.group_count || channel_count <= 0 || !channels)
        throw std::invalid_argument("CSS: invalid kernel dimensions/channels");
    if (geometry.group_count && (!atom_types.data || !css_values.data || !coordination_numbers.data ||
                                !geometry.d_group_indices || !geometry.d_group_numneigh ||
                                !geometry.d_neigh_in_cutoff_r))
        throw std::invalid_argument("CSS: missing kernel data");
    if (static_cast<size_t>(geometry.group_count)*channel_count > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::length_error("CSS: too many center/channel pairs");
}
} // namespace

void css_forward(StruFactorKernelView geometry, ArrayView<const int> atom_types, int total_atom_count,
                 const CSSChannel* channels, int channel_count, ArrayView<double> css_values,
                 ArrayView<double> coordination_numbers, cudaStream_t stream) {
    check_arguments(geometry, atom_types, total_atom_count, channels, channel_count, css_values,
                    coordination_numbers);
    if (!geometry.group_count) return;
    if (atom_types.component_count != 1 || atom_types.memory != MemorySpace::Cuda ||
        css_values.component_count != 1 || css_values.memory != MemorySpace::Cuda ||
        coordination_numbers.component_count != 1 || coordination_numbers.memory != MemorySpace::Cuda)
        throw std::invalid_argument("CSS: forward views must be Cuda scalars");
    atom_types.wait_on(stream); css_values.wait_on(stream); coordination_numbers.wait_on(stream);
    const size_t c_channels_count = static_cast<size_t>(geometry.group_count)*channel_count;
    count_kernel<<<(c_channels_count+127)/128,128,0,stream>>>(geometry,atom_types,total_atom_count,channels,
        channel_count,coordination_numbers);
    check_cuda(cudaGetLastError());
    score_kernel<<<(geometry.group_count-1)/128+1,128,0,stream>>>(geometry,total_atom_count,channels,channel_count,
        coordination_numbers,css_values);
    check_cuda(cudaGetLastError());
}

void css_forward(StruFactorKernelView geometry, const int* atom_types, int total_atom_count,
                 const CSSChannel* channels, int channel_count, double* css_values,
                 double* coordination_numbers, cudaStream_t stream) {
    css_forward(geometry, unchecked_types(atom_types), total_atom_count, channels, channel_count,
                unchecked_scalars(css_values), unchecked_scalars(coordination_numbers), stream);
}

void css_backward(StruFactorKernelView geometry, ArrayView<const int> atom_types, int total_atom_count,
                  const CSSChannel* channels, int channel_count, ArrayView<const double> css_values,
                  ArrayView<const double> coordination_numbers, ArrayView<const double> center_adjoint,
                  ArrayView<double> atom_forces, ArrayView<double> atom_virials, cudaStream_t stream) {
    check_arguments(geometry, atom_types, total_atom_count, channels, channel_count, css_values,
                    coordination_numbers);
    if (!geometry.group_count) return;
    if (!center_adjoint.data || !atom_forces.data) throw std::invalid_argument("CSS: missing adjoint/force");
    if (atom_types.component_count != 1 || atom_forces.component_count != 3 ||
        (atom_virials.data && atom_virials.component_count != 6))
        throw std::invalid_argument("CSS: backward view shapes are invalid");
    atom_types.wait_on(stream); css_values.wait_on(stream); coordination_numbers.wait_on(stream);
    center_adjoint.wait_on(stream); atom_forces.wait_on(stream); atom_virials.wait_on(stream);
    backward_kernel<<<(geometry.group_count-1)/128+1,128,0,stream>>>(
        geometry,atom_types,total_atom_count,channels,channel_count,css_values,coordination_numbers,center_adjoint,
            atom_forces,atom_virials);
    check_cuda(cudaGetLastError());
}

void css_backward(StruFactorKernelView geometry, const int* atom_types, int total_atom_count,
                  const CSSChannel* channels, int channel_count, const double* css_values,
                  const double* coordination_numbers, const double* center_adjoint,
                  double* atom_forces, double* atom_virials, cudaStream_t stream) {
    css_backward(geometry, unchecked_types(atom_types), total_atom_count, channels, channel_count,
                 unchecked_scalars(css_values), unchecked_scalars(coordination_numbers),
                 unchecked_scalars(center_adjoint), unchecked_xyz(atom_forces),
                 atom_virials ? unchecked_virial6(atom_virials) : empty_virial(), stream);
}
} // namespace MetaD_zqc

#ifndef ZQC_CSS_KERNELS_ONLY
#include "zqc_cv_assembly.cuh"
#include "zqc_cv_debug.h"
#include "zqc_cv_parse.h"
#include "zqc_switch_function.h"
#include "zqc_virial.h"

namespace MetaD_zqc {
CV* CSS::create(CVContext& context, int argument_count, char** arguments, int& argument_index) {
    auto logger = context.logger;
    ERR_COND(argument_index+1 >= argument_count,"CSS requires a center group");
    const std::string selection_name = arguments[argument_index+1];
    const SelectionId selection = context.engine.selection_id(selection_name);
    argument_index += 2;
    std::vector<CSSChannel> channels;
    std::vector<std::string> radial_names;
    double cutoff_epsilon = 1e-12;
    const int type_begin = context.engine.atom_type_begin();
    const int type_count = context.engine.atom_type_count();
    while (argument_index < argument_count) {
        const std::string key = arguments[argument_index];
        if (key == "CHANNEL") {
            // CHANNEL <邻居类型编号或标签> <径向开关名> <目标配位数> <容差>。
            // 例：CHANNEL 1 first 12 2.0，统计 type 1，目标配位数为 12，容差为 2。
            // 标签经 engine.atom_type_id 解析为原生类型编号。
            ERR_COND(argument_index+4 >= argument_count,"CSS CHANNEL requires: type radial_switch target sigma");
            CSSChannel channel{};
            channel.neighbor_type = context.engine.atom_type_id(arguments[argument_index+1]);
            auto* radial_switch = context.switching_function(arguments[argument_index+2]);
            channel.radial = radial_switch->params;
            channel.target = cv_number(arguments[argument_index+3]);
            channel.sigma = cv_number(arguments[argument_index+4]);
            ERR_COND(channel.neighbor_type < type_begin ||
                     channel.neighbor_type >= type_begin + type_count,
                     "CSS: neighbor type is outside the engine type range");
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
            cutoff_epsilon = cv_number(arguments[argument_index+1]);
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
    StruFactorRequest environment_request{};
    environment_request.selection = selection;
    environment_request.selection_name = selection_name;
    environment_request.cutoff_r = max_cutoff_radius;
    auto* shared_environment = Stru_fact_env::get_or_create(context, environment_request);
    return new CSS(context, shared_environment, std::move(channels), std::move(radial_names),
                   cutoff_epsilon);
}

CSS::CSS(CVContext& context, Stru_fact_env* shared_environment,
         std::vector<CSSChannel> channels, std::vector<std::string> radial_names, double cutoff_epsilon)
    : CV(context), environment_(shared_environment), channels_(std::move(channels)),
      radial_names_(std::move(radial_names)), cutoff_eps_(cutoff_epsilon) {
    register_buffer(d_channels_,"css_channels");
    register_buffer(d_css_values_,"css_values"); register_buffer(d_coordination_numbers_,"css_counts");
    register_buffer(d_center_adjoint_,"css_adjoint"); register_buffer(d_atom_forces_,"css_force");
    register_buffer(d_atom_virials_,"css_virial");
    d_channels_.grow_to(channels_.size(),__FILE__,__LINE__);
    d_channels_.upload_from(channels_.data(), channels_.size());
}

CV::CV_Calculation CSS::set_CV_calculate(std::string) {
    throw std::runtime_error("CSS: use EXECUTOR GRAPH with RAW or AVE");
}
CV::CV_BiasForce CSS::set_CV_bias_force(std::string) {
    throw std::runtime_error("CSS: use EXECUTOR GRAPH for adjoint propagation");
}
void CSS::summary(FILE* output_file) {
    if (!output_file) return;
    std::fprintf(output_file,"CSS: center selection=%s (%d), channels=%zu, cutoff_eps=%g\n",
                 environment_->selection_name_.c_str(), environment_->selection_id_,
                 channels_.size(),cutoff_eps_);
    for (size_t channel_index = 0; channel_index < channels_.size(); ++channel_index) {
        const auto& channel = channels_[channel_index];
        std::fprintf(output_file,"  channel %zu: neighbor type=%d radial=%s target=%g sigma=%g cutoff=%g\n",
                     channel_index+1,channel.neighbor_type,radial_names_[channel_index].c_str(),channel.target,
                         channel.sigma,channel.cutoff);
    }
}
void CSS::bind_assembly(std::shared_ptr<const CVAssembly> assembly) {
    if (assembly && (assembly->selection_id() != environment_->selection_id_ ||
                     assembly->nlocal() != engine.owned_atom_count() ||
                     assembly->nghost() != engine.total_atom_count()-engine.owned_atom_count()))
        throw std::invalid_argument("CSS: incompatible atom assembly");
    environment_->atom_assembly = assembly;
    environment_->geometry_evaluation = 0;
    values_ready_ = false;
    CV::bind_assembly(std::move(assembly));
}
void CSS::prepare_graph_evaluation() {
    values_ready_ = false;
    graph_evaluation_prepared_ = true;
    environment_->geometry_evaluation = 0;
}
void CSS::base_calc() {
    values_ready_ = false;
    host_css_values_ready_ = host_coordination_numbers_ready_ = false;
    if (!graph_evaluation_prepared_) environment_->geometry_evaluation = 0;
    graph_evaluation_prepared_ = false;
    if (environment_->geometry_evaluation != engine.evaluation_id()) {
        environment_->refresh_centers();
        environment_->get_env();
    }
    total_atom_count_ = engine.total_atom_count();
    const size_t coordination_buffer_size = static_cast<size_t>(total_atom_count_)*channels_.size();
    device_atom_types_ = engine.atom_types(MemorySpace::Cuda);
    d_css_values_.grow_to(total_atom_count_,__FILE__,__LINE__);
    d_coordination_numbers_.grow_to(coordination_buffer_size,__FILE__,__LINE__);
    if (total_atom_count_) {
        check_cuda(cudaMemsetAsync(d_css_values_.ptr,0,total_atom_count_*sizeof(double),environment_->execution.get()));
        check_cuda(cudaMemsetAsync(d_coordination_numbers_.ptr,0,coordination_buffer_size*sizeof(double),environment_->execution.get()));
    }
    css_forward(kernel_view(environment_->data), device_atom_types_, total_atom_count_,
                d_channels_.ptr, static_cast<int>(channels_.size()),
                d_css_values_.view(total_atom_count_),
                d_coordination_numbers_.view(coordination_buffer_size), environment_->execution.get());
    value_evaluation_ = engine.evaluation_id();
    values_ready_ = true;
}
void CSS::check_values() const {
    if (!values_ready_ || value_evaluation_ != engine.evaluation_id() ||
        total_atom_count_ != engine.total_atom_count())
        throw std::logic_error("CSS: base_calc is required for the current atom layout");
}
void CSS::assemble_values(ArrayView<double> leaf_values) {
    check_values();
    if (!assembly_) throw std::logic_error("CSS: missing atom assembly");
    assembly_->assemble_values(d_css_values_.view(total_atom_count_), leaf_values, environment_->execution.get());
}
void CSS::apply_adjoint(ArrayView<const double> leaf_adjoint) {
    check_values();
    if (!assembly_ || assembly_->nlocal() != engine.owned_atom_count() ||
        assembly_->nghost() != engine.total_atom_count()-engine.owned_atom_count() ||
        assembly_->group_count() != environment_->data.group_count)
        throw std::logic_error("CSS: stale atom assembly");
    d_center_adjoint_.grow_to(assembly_->group_count(),__FILE__,__LINE__);
    assembly_->gather_adjoint(leaf_adjoint, d_center_adjoint_.view(assembly_->group_count()), environment_->execution.get());
    const size_t total_atom_count = static_cast<size_t>(total_atom_count_);
    d_atom_forces_.grow_to(3*total_atom_count,__FILE__,__LINE__);
    d_atom_virials_.grow_to(6*total_atom_count,__FILE__,__LINE__);
    if (total_atom_count) {
        check_cuda(cudaMemsetAsync(d_atom_forces_.ptr,0,3*total_atom_count*sizeof(double),environment_->execution.get()));
        check_cuda(cudaMemsetAsync(d_atom_virials_.ptr,0,6*total_atom_count*sizeof(double),environment_->execution.get()));
    }
    auto forces = d_atom_forces_.view(total_atom_count, 3);
    auto virials = d_atom_virials_.view(total_atom_count, 6);
    css_backward(kernel_view(environment_->data), device_atom_types_, total_atom_count_,
                 d_channels_.ptr, static_cast<int>(channels_.size()),
                 d_css_values_.view(total_atom_count_),
                 d_coordination_numbers_.view(total_atom_count_*channels_.size()),
                 d_center_adjoint_.view(assembly_->group_count()), forces, virials, environment_->execution.get());
    forces.readiness = virials.readiness = device_readiness();
    engine.commit_force_virial(forces, {virials, VirialRepresentation::Symmetric6});
}

ArrayView<const double> CSS::peratom_values(const std::string& property) {
    check_values();
    if (property == "css" || property == "value") {
        if (!host_css_values_ready_) {
            h_css_values_.resize(std::max(1,total_atom_count_));
            if (total_atom_count_)
                d_css_values_.download_to(h_css_values_.data(), total_atom_count_, environment_->execution.get());
            host_css_values_ready_ = true;
        }
        return {h_css_values_.data(), static_cast<size_t>(total_atom_count_), 1, 1, 1,
                h_css_values_.size(), MemorySpace::Host};
    }
    // n1、n2、... 对应声明顺序；通道优先大缓冲中每个通道是一段连续 localid 切片。
    if (property.size() > 1 && property[0] == 'n') {
        size_t parsed_length = 0;
        int channel_index;
        try { channel_index = std::stoi(property.substr(1),&parsed_length); }
        catch (const std::exception&) { throw std::invalid_argument("CSS: invalid channel property "+property); }
        if (parsed_length != property.size()-1 || channel_index < 1 ||
            channel_index > static_cast<int>(channels_.size()))
            throw std::invalid_argument("CSS: invalid channel property "+property);
        --channel_index;
        if (!host_coordination_numbers_ready_) {
            const size_t full = static_cast<size_t>(std::max(1,total_atom_count_))*channels_.size();
            h_coordination_numbers_.resize(std::max(size_t(1), full));
            if (total_atom_count_)
                d_coordination_numbers_.download_to(
                    h_coordination_numbers_.data(),
                    static_cast<size_t>(total_atom_count_)*channels_.size(), environment_->execution.get());
            host_coordination_numbers_ready_ = true;
        }
        const size_t offset = static_cast<size_t>(channel_index)*static_cast<size_t>(total_atom_count_);
        // 逻辑长度 = total；storage_size 覆盖本通道切片及其后可达尾部，避免伪造更短缓冲。
        return {h_coordination_numbers_.data()+offset,
                static_cast<size_t>(total_atom_count_), 1, 1, 1,
                h_coordination_numbers_.size()-offset, MemorySpace::Host};
    }
    throw std::invalid_argument("CSS: unknown property "+property);
}
REGISTER_CV("CSS",MetaD_zqc::CSS::create);
} // namespace MetaD_zqc
#endif
