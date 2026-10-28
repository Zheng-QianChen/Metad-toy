#include "CV_Stru_factor.h"
#include "CV_Weighted_Chemical_Pair_Order.h"
#include <cmath>
#include <limits>
#include <stdexcept>

// 独立 kernel 测试通过同一源文件编译设备数据与前后向计算。
#ifndef ZQC_STRUFACTOR_KERNELS_ONLY
#include "zqc_cv_debug.h"
#include "zqc_cv_parse.h"
#include "zqc_cv_assembly.cuh"

#include <cuda_runtime.h>
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <sstream>

namespace {
void check_strufactor_cuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
}

std::map<std::string, MetaD_zqc::Stru_fact_env*> MetaD_zqc::Stru_fact_env::env_pool;

MetaD_zqc::CV* MetaD_zqc::Stru_factor::create(CVContext& context, int narg, char** arg, int& i) {
    auto logger = context.logger;
    DEBUG_LOG("In STRU_FACTOR settings");
    std::string cal_name = arg[i];

    MetaD_zqc::StruFactorRequest req{};
    req.q_factor = 1.0;
    req.cal_name = cal_name;
    ERR_COND(i + 1 >= narg, "Error: STRU_FACTOR command requires at least \"STRU_FACTOR <group> \".");
    req.selection_name = arg[i+1];
    req.selection = context.engine.selection_id(req.selection_name);
    req.cutoff_r = 8.0;
    req.d_block_size = 128;
    req.custom_weights.clear();
    int iarg = 2 + i;
    while (iarg < narg) {
        if (strcmp(arg[iarg], "cutoff_r") == 0) {
            ERR_COND((iarg + 1 >= narg), "Error: 'cutoff_r' keyword requires a value");
            req.cutoff_r = cv_number(arg[iarg + 1]);
            iarg += 2;
        } else if (strcmp(arg[iarg], "q_factor") == 0) {
            ERR_COND((iarg + 1 >= narg), "Error: 'q_factor' keyword requires a number");
            req.q_factor = cv_number(arg[iarg + 1]);
            ERR_COND(req.q_factor <= 0, "Error: 'q_factor' must be > 0");
            iarg += 2;
        } else if (strcmp(arg[iarg], "d_block_size") == 0) {
            ERR_COND((iarg + 1 >= narg), "Error: 'd_block_size' keyword requires an integer");
            req.d_block_size = cv_integer(arg[iarg + 1]);
            ERR_COND(req.d_block_size <= 0, "Error: 'd_block_size' must be > 0");
            iarg += 2;
        } else if (strcmp(arg[iarg], "SW_func") == 0) {
            throw std::runtime_error("STRU_FACTOR: express switching functions in Graph using the RAW input");
        } else if (strcmp(arg[iarg], "Chemical") == 0) {
            req.use_chemical_lock = true;
            iarg += 1;
        } else if (strcmp(arg[iarg], "Chem_ctarget") == 0) {
            ERR_COND((iarg + 1 >= narg), "Error: 'Chem_ctarget' keyword requires a number");
            req.c_target = cv_number(arg[iarg + 1]);
            iarg += 2;
        } else if (strcmp(arg[iarg], "Chem_sigma") == 0) {
            ERR_COND((iarg + 1 >= narg), "Error: 'sigma' keyword requires a number");
            req.sigma = cv_number(arg[iarg + 1]);
            iarg += 2;
        } else if (strcmp(arg[iarg], "Chem_map") == 0) {
            iarg += 1;
            req.use_custom_weight = true;
            while (iarg < narg && arg[iarg][0] == '(') {
                std::string pair_str = arg[iarg];
                if (pair_str.front() == '(' && pair_str.back() == ')') {
                    pair_str = pair_str.substr(1, pair_str.size() - 2);
                    size_t sep = pair_str.find_first_of(" ,");
                    if (sep != std::string::npos) {
                        std::string type_s = pair_str.substr(0, sep);
                        std::string weight_s = pair_str.substr(sep + 1);
                        const int a_type = context.engine.atom_type_id(type_s);
                        const double a_weight = cv_number(weight_s.c_str());
                        req.custom_weights[a_type] = a_weight;
                    }
                } else {
                    std::string part1 = arg[iarg];
                    ERR_COND(iarg + 1 >= narg, "Error: Invalid Chem_map format near %s", arg[iarg]);
                    std::string part2 = arg[iarg+1];
                    if (part1.front() == '(' && part2.back() == ')') {
                        const int a_type = context.engine.atom_type_id(part1.substr(1));
                        const double a_weight = cv_number(part2.substr(0, part2.size() - 1).c_str());
                        req.custom_weights[a_type] = a_weight;
                        iarg += 1;
                    } else {
                        throw std::runtime_error("Error: Chem_map pairs must be formatted as (type weight)");
                    }
                }
                iarg += 1;
            }
        } else {
            break;
        }
    }
    i = iarg;

    ERR_COND(!std::isfinite(req.q_factor) || req.q_factor <= 0,
             "STRU_FACTOR: q_factor must be finite and positive");
    ERR_COND(!std::isfinite(req.cutoff_r*req.cutoff_r) || req.cutoff_r <= 0,
             "STRU_FACTOR: cutoff_r must be finite and positive");
    ERR_COND(req.d_block_size <= 0 || req.d_block_size > 1024,
             "STRU_FACTOR: d_block_size must be in [1,1024]");

    MetaD_zqc::Stru_factor* struc_factor = nullptr;
    if (req.use_chemical_lock) {
        LOG("Logging: set STRU_FACTOR as group_name=%s q_factor=%g cutoff_r=%f d_block_size=%d.\n         Chemical lock is ON, with c_target=%g and sigma=%g.",
            req.selection_name.c_str(), req.q_factor, req.cutoff_r, req.d_block_size,
            req.c_target, req.sigma);
        auto* temp_env = static_cast<MetaD_zqc::Stru_fact_chem_env*>(
            MetaD_zqc::Stru_fact_env::get_or_create(context, req));
        struc_factor = new MetaD_zqc::Stru_factor_chem(context, temp_env, req.q_factor, req.d_block_size);
    } else {
        LOG("Logging: set STRU_FACTOR as group_name=%s q_factor=%g cutoff_r=%f d_block_size=%d.",
            req.selection_name.c_str(), req.q_factor, req.cutoff_r, req.d_block_size);
        MetaD_zqc::Stru_fact_env* temp_env = MetaD_zqc::Stru_fact_env::get_or_create(context, req);
        struc_factor = new MetaD_zqc::Stru_factor(context, temp_env, req.q_factor, req.d_block_size);
    }
    return struc_factor;
}

MetaD_zqc::Stru_fact_env* MetaD_zqc::Stru_fact_env::get_or_create(CVContext& context,
                                                                  StruFactorRequest req) {
    // 键含引擎地址、选择集、cutoff 与完整 Chem 配置，保证不同权重各自持有类型表。
    std::ostringstream key_stream;
    key_stream << std::setprecision(std::numeric_limits<double>::max_digits10)
               << &context.engine << '_' << req.selection << '_' << req.cutoff_r << '_'
               << req.use_chemical_lock;
    if (req.use_chemical_lock) {
        if (!std::isfinite(req.c_target) || !std::isfinite(req.sigma) || req.sigma <= 0)
            throw std::invalid_argument("STRU_FACTOR Chemical: Chem_ctarget must be finite and Chem_sigma positive and finite");
        const int type_begin = context.engine.atom_type_begin();
        const int type_count = context.engine.atom_type_count();
        key_stream << '_' << req.c_target << '_' << req.sigma;
        for (const auto& entry : req.custom_weights) {
            if (entry.first < type_begin || entry.first >= type_begin + type_count ||
                !std::isfinite(entry.second))
                throw std::invalid_argument("STRU_FACTOR Chemical: Chem_map requires valid atom types and finite weights");
            key_stream << '_' << entry.first << ':' << entry.second;
        }
    }
    const std::string key = key_stream.str();
    const auto found = env_pool.find(key);
    if (found != env_pool.end()) return found->second;
    Stru_fact_env* environment;
    if (req.use_chemical_lock)
        environment = new Stru_fact_chem_env(context, req.selection, req.selection_name,
                                             req.cutoff_r, req.c_target, req.sigma, req.custom_weights);
    else
        environment = new Stru_fact_env(context, req.selection, req.selection_name, req.cutoff_r);
    env_pool.emplace(key, environment);
    return environment;
}

std::string MetaD_zqc::Stru_fact_env::get_env_key() const {
    return std::to_string(selection_id_) + "_" + std::to_string(cutoff_r);
}

void MetaD_zqc::Stru_fact_env::clear_pool(EngineAdapter* owner) {
    for (auto entry = env_pool.begin(); entry != env_pool.end();) {
        if (!owner || &entry->second->engine == owner) {
            delete entry->second;
            entry = env_pool.erase(entry);
        } else ++entry;
    }
}

MetaD_zqc::Stru_fact_env::Stru_fact_env(CVContext& context, SelectionId selection,
                                        std::string selection_name, double cutoff_r)
    : CV_info(context), cutoff_r(cutoff_r), selection_id_(selection),
      selection_name_(std::move(selection_name)) {
    data.set_logger(logger);
    NeighborRequest request;
    request.cutoff = cutoff_r;
    request.centers = selection;
    neighbor_handle = engine.neighbors().request(request);
}

MetaD_zqc::Stru_factor::Stru_factor(CVContext& context, Stru_fact_env* env,
                                  double q, int block_size)
    : CV(context), my_env(env), d_block_size(block_size), q_factor(q) {
    my_env->d_block_size = block_size;
    check_strufactor_cuda(cudaGetDevice(&my_env->GPU_number));
    register_buffer(d_stru_factor, "d_stru_factor");
    register_buffer(d_center_adjoint, "sf_center_adjoint");
    register_buffer(d_graph_force, "sf_graph_force");
    register_buffer(d_graph_virial, "sf_graph_virial");
}

MetaD_zqc::Stru_fact_env::~Stru_fact_env() = default;
MetaD_zqc::Stru_factor::~Stru_factor() = default;

// 适配器返回的中心顺序必须与 Leaf 装配表一致，邻居可含选择集外原子和 ghost。
void MetaD_zqc::Stru_fact_env::refresh_centers() {
    const auto candidates = engine.neighbors().prepare(neighbor_handle);
    candidates.host_centers.wait_host();
    last_group_count = data.group_count;
    data.group_count = static_cast<int>(candidates.host_centers.item_count);
    if (atom_assembly) {
        if (atom_assembly->nlocal() != engine.owned_atom_count() ||
            atom_assembly->nghost() != engine.total_atom_count()-engine.owned_atom_count() ||
            atom_assembly->group_count() != data.group_count)
            throw std::logic_error("Stru_factor: neighbor centers differ from Leaf assembly");
        for (int center=0;center<data.group_count;++center)
            if(candidates.host_centers(center)!=atom_assembly->local_ids()[center])
                throw std::logic_error("Stru_factor: neighbor center order differs from Leaf assembly");
    }
    block_num = (data.group_count + d_block_size - 1) / d_block_size;
    N = static_cast<size_t>(d_block_size) * block_num;
}

void MetaD_zqc::Stru_fact_env::get_env() {
    const auto candidates = engine.neighbors().prepare(neighbor_handle);
    strufactor_prepare_geometry(data, candidates, engine.positions(MemorySpace::Cuda),
                                cutoff_r, execution.get(), d_block_size);
    geometry_evaluation = engine.evaluation_id();
}

void MetaD_zqc::Stru_factor::environment() {
    if (my_env->geometry_evaluation != engine.evaluation_id()) {
        my_env->refresh_centers();
        my_env->get_env();
    }
}

auto MetaD_zqc::Stru_factor::set_CV_calculate(std::string) -> CV_Calculation {
    throw std::runtime_error("STRU_FACTOR: use EXECUTOR GRAPH; reductions belong to the graph");
}

auto MetaD_zqc::Stru_factor::set_CV_bias_force(std::string) -> CV_BiasForce {
    throw std::runtime_error("STRU_FACTOR: use EXECUTOR GRAPH for adjoint propagation");
}

void MetaD_zqc::Stru_factor::bind_assembly(std::shared_ptr<const CVAssembly> assembly) {
    if (assembly && (assembly->selection_id() != my_env->selection_id_ ||
                     assembly->nlocal() != engine.owned_atom_count() ||
                     assembly->nghost() != engine.total_atom_count()-engine.owned_atom_count()))
        throw std::invalid_argument("Stru_factor: incompatible atom assembly");
    if (my_env->atom_assembly != assembly) {
        my_env->atom_assembly = assembly;
        my_env->geometry_evaluation = 0;
    }
    raw_values_ready = false;
    graph_evaluation_prepared = false;
    CV::bind_assembly(std::move(assembly));
}

void MetaD_zqc::Stru_factor::check_graph_values() const {
    if (!assembly_)
        throw std::logic_error("Stru_factor: bind an atom assembly before using the graph interface");
    if (!raw_values_ready || raw_value_evaluation != engine.evaluation_id())
        throw std::logic_error("Stru_factor: base_calc is required before assembly/backward");
    if (assembly_->nlocal() != engine.owned_atom_count() ||
        assembly_->nghost() != engine.total_atom_count()-engine.owned_atom_count() ||
        assembly_->group_count() != my_env->data.group_count)
        throw std::logic_error("Stru_factor: owned/ghost layout changed; rebuild and bind it first");
}

void MetaD_zqc::Stru_factor::assemble_values(ArrayView<double> values) {
    check_graph_values();
    const size_t total = static_cast<size_t>(engine.total_atom_count());
    assembly_->assemble_values(d_stru_factor.view(total), values, my_env->execution.get());
}

void MetaD_zqc::Stru_factor::prepare_graph_evaluation() {
    raw_values_ready = false;
    my_env->geometry_evaluation = 0;
    graph_evaluation_prepared = true;
}

void MetaD_zqc::Stru_factor::base_calc() {
    raw_values_ready = false;
    if (assembly_) {
        if (!graph_evaluation_prepared) my_env->geometry_evaluation = 0;
    }
    graph_evaluation_prepared = false;
    compute_stru_factor_peratoms();
    raw_value_evaluation = engine.evaluation_id();
    raw_values_ready = true;
}

void MetaD_zqc::Stru_factor::apply_adjoint(ArrayView<const double> leaf_adjoint) {
    check_graph_values();
    const size_t nall = static_cast<size_t>(engine.total_atom_count());
    d_center_adjoint.grow_to(my_env->data.group_count, __FILE__, __LINE__);
    assembly_->gather_adjoint(leaf_adjoint, d_center_adjoint.view(my_env->data.group_count), my_env->execution.get());
    d_graph_force.grow_to(3*nall, __FILE__, __LINE__);
    d_graph_virial.grow_to(6*nall, __FILE__, __LINE__);
    if (nall) {
        check_strufactor_cuda(cudaMemsetAsync(d_graph_force.ptr, 0, 3*nall*sizeof(double), my_env->execution.get()));
        check_strufactor_cuda(cudaMemsetAsync(d_graph_virial.ptr, 0, 6*nall*sizeof(double), my_env->execution.get()));
    }
    auto forces = d_graph_force.view(nall, 3);
    auto virials = d_graph_virial.view(nall, 6);
    call_structure_factor_adjoint_kernel(d_center_adjoint.view(my_env->data.group_count),
                                          forces, virials);
    forces.readiness = virials.readiness = device_readiness();
    engine.commit_force_virial(forces, {virials, VirialRepresentation::Symmetric6});
}

void MetaD_zqc::Stru_factor::compute_stru_factor_peratoms() {
    environment();
    const size_t nall = static_cast<size_t>(engine.total_atom_count());
    h_stru_factor.assign(nall, 0.0);
    d_stru_factor.grow_to(nall, __FILE__, __LINE__);
    if (nall) check_strufactor_cuda(cudaMemsetAsync(d_stru_factor.ptr, 0, nall*sizeof(double), my_env->execution.get()));
    if (my_env->data.group_count) call_structure_factor_cv_kernel();
}

void MetaD_zqc::Stru_factor::call_structure_factor_cv_kernel() {
    const size_t nall = static_cast<size_t>(engine.total_atom_count());
    strufactor_forward(my_env->data, q_factor, my_env->cutoff_r*my_env->cutoff_r,
                       d_stru_factor.view(nall), my_env->execution.get(), d_block_size);
}

void MetaD_zqc::Stru_factor::call_structure_factor_adjoint_kernel(
    ArrayView<const double> center_adjoint, ArrayView<double> atom_forces,
    ArrayView<double> atom_virials) {
    strufactor_backward(my_env->data, q_factor, my_env->cutoff_r*my_env->cutoff_r,
                        center_adjoint, atom_forces, atom_virials, my_env->execution.get(), d_block_size);
}

void MetaD_zqc::Stru_factor::summary(FILE*) {}

MetaD_zqc::ArrayView<const double> MetaD_zqc::Stru_factor::peratom_values(const std::string& name) {
    if (name != "stru_f") throw std::invalid_argument("Stru_factor: unknown property "+name);
    if (!raw_values_ready || raw_value_evaluation != engine.evaluation_id())
        throw std::logic_error("Stru_factor: base_calc is required before per-atom diagnostics");
    const size_t nall = static_cast<size_t>(engine.total_atom_count());
    if (nall) d_stru_factor.download_to(h_stru_factor.data(), nall, my_env->execution.get());
    return {h_stru_factor.data(), nall, 1, 1, 1, h_stru_factor.size(), MemorySpace::Host};
}

REGISTER_CV("STRU_FACTOR", MetaD_zqc::Stru_factor::create);
#endif // ZQC_STRUFACTOR_KERNELS_ONLY

namespace MetaD_zqc {
namespace {

// 一线程处理一个中心，将 cutoff 内的邻居紧排到自己的候选段开头。
// candidates 按 offsets 分段，coordinates 的 stride 描述实际存储；坐标已包含周期镜像。
// 输出 neighbors[slot] 和 geometry[4*slot+c] 共享槽位，counts[k] 记录每段有效前缀长度。
__global__ void get_environment_Strufactor(double cutoff_rsq, int group_count,
    const int* centers, const int* offsets, const int* candidates, ArrayView<const double> coordinates,
    double* geometry, int* counts, int* neighbors) {
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= group_count) return;
    const int center = centers[k];
    const int first = offsets[k], last = offsets[k+1];
    int count = 0;
    for (int p = first; p < last; ++p) {
        const int neighbor = candidates[p];
        const double dx = coordinates(neighbor, 0) - coordinates(center, 0);
        const double dy = coordinates(neighbor, 1) - coordinates(center, 1);
        const double dz = coordinates(neighbor, 2) - coordinates(center, 2);
        const double r2 = dx*dx + dy*dy + dz*dz;
        if (r2 > cutoff_rsq || r2 < 1e-12) continue;
        const int slot = first + count++;
        geometry[4*slot] = dx;
        geometry[4*slot+1] = dy;
        geometry[4*slot+2] = dz;
        geometry[4*slot+3] = r2;
        neighbors[slot] = neighbor;
    }
    counts[k] = count;
}

int checked_center_count(int count) {
    if (count < 0) throw std::invalid_argument("Stru_factor: negative center count");
    return count;
}
size_t geometry_size(size_t pair_count) {
    if (pair_count > std::numeric_limits<size_t>::max()/4)
        throw std::length_error("Stru_factor: neighbor geometry size overflow");
    return 4*pair_count;
}

struct RadialValue { double value, derivative_over_r; };

__device__ __forceinline__ RadialValue radial(double r2, double q, double cutoff_rsq);
template<bool UseChemicalWeights, bool UseStructureFactor = true>
__global__ void forward_kernel(StruFactorKernelView view, double q, double cutoff_rsq,
                               ArrayView<const int> atom_types, const double* type_weights,
                               ArrayView<double> values);
template<bool UseChemicalWeights, bool UseStructureFactor = true>
__global__ void backward_kernel(StruFactorKernelView view, double q, double cutoff_rsq,
                                ArrayView<const int> atom_types, const double* type_weights,
                                ArrayView<const double> adjoint, ArrayView<double> force,
                                ArrayView<double> virial);

void check_arguments(const StruFactorKernelView& view, double cutoff_rsq, int block_size) {
    if (block_size <= 0 || block_size > 1024 || view.group_count < 0 ||
        !std::isfinite(cutoff_rsq) || cutoff_rsq <= 0)
        throw std::invalid_argument("Stru_factor: invalid kernel parameters");
    if (view.group_count && (!view.d_group_indices || !view.d_group_numneigh || !view.d_neigh_in_cutoff_r))
        throw std::invalid_argument("Stru_factor: missing center data");
}
void check_wave_number(double q) {
    if (!std::isfinite(q) || q <= 0)
        throw std::invalid_argument("Stru_factor: q must be finite and positive");
}
void check_launch() {
    const auto status = cudaGetLastError();
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

void require_scalar_items(ArrayView<const double> view, const char* what) {
    view.validate();
    if (view.component_count != 1 || view.memory != MemorySpace::Cuda)
        throw std::invalid_argument(std::string("Stru_factor: ")+what+" must be Cuda scalars");
}
void require_scalar_items(ArrayView<double> view, const char* what) {
    require_scalar_items(static_cast<ArrayView<const double>>(view), what);
}
void require_types(ArrayView<const int> view) {
    view.validate();
    if (view.component_count != 1 || view.memory != MemorySpace::Cuda)
        throw std::invalid_argument("Stru_factor: atom_types must be Cuda scalars");
}
void require_xyz(ArrayView<double> view) {
    view.validate();
    if (view.component_count != 3 || view.memory != MemorySpace::Cuda)
        throw std::invalid_argument("Stru_factor: force must be Cuda [atom,3]");
}
void require_virial6(ArrayView<double> view) {
    if (!view.data) return;
    view.validate();
    if (view.component_count != 6 || view.memory != MemorySpace::Cuda)
        throw std::invalid_argument("Stru_factor: virial must be Cuda Symmetric6");
}
} // namespace

void strufactor_prepare_geometry(StruFactorData& data, const NeighborListView& candidates,
    ArrayView<const double> coordinates, double cutoff, cudaStream_t stream, int block_size) {
    coordinates.validate();
    coordinates.wait_on(stream);
    candidates.centers.wait_on(stream); candidates.offsets.wait_on(stream); candidates.neighbors.wait_on(stream);
    candidates.centers.validate(); candidates.offsets.validate(); candidates.neighbors.validate();
    if (block_size <= 0 || block_size > 1024 || !(cutoff > 0) || !std::isfinite(cutoff*cutoff) ||
        candidates.owned_atom_count < 0 || candidates.total_atom_count < candidates.owned_atom_count ||
        coordinates.component_count != 3 || coordinates.memory != MemorySpace::Cuda ||
        coordinates.item_count != static_cast<size_t>(candidates.total_atom_count) ||
        candidates.centers.memory != MemorySpace::Cuda || candidates.offsets.memory != MemorySpace::Cuda ||
        candidates.neighbors.memory != MemorySpace::Cuda ||
        candidates.centers.item_count > static_cast<size_t>(candidates.owned_atom_count) ||
        candidates.offsets.item_count != candidates.centers.item_count+1 ||
        candidates.centers.item_stride != 1 || candidates.offsets.item_stride != 1 || candidates.neighbors.item_stride != 1 ||
        candidates.centers.component_count != 1 || candidates.offsets.component_count != 1 || candidates.neighbors.component_count != 1 ||
        candidates.neighbors.item_count > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Stru_factor: invalid geometry input views");
    if (candidates.coordinates != NeighborCoordinates::PeriodicImages)
        throw std::invalid_argument("Stru_factor: geometry requires periodic-image coordinates");
    data.group_count = static_cast<int>(candidates.centers.item_count);
    if (!data.group_count) return;
    data.grow_centers(__FILE__, __LINE__);
    data.grow_neighbors(candidates.neighbors.item_count, data.group_count, __FILE__, __LINE__);
    auto check = [](cudaError_t status) {
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    };
    check(cudaMemcpyAsync(data.d_group_indices.ptr, candidates.centers.data,
                          data.group_count*sizeof(int), cudaMemcpyDeviceToDevice, stream));
    check(cudaMemcpyAsync(data.d_group_numneigh.ptr, candidates.offsets.data,
                          (static_cast<size_t>(data.group_count)+1)*sizeof(int), cudaMemcpyDeviceToDevice, stream));
    get_environment_Strufactor<<<(data.group_count-1)/block_size+1, block_size, 0, stream>>>(
        cutoff*cutoff, data.group_count, data.d_group_indices.ptr, data.d_group_numneigh.ptr,
        candidates.neighbors.data, coordinates, data.d_group_dminneigh.ptr,
        data.d_neigh_in_cutoff_r.ptr, data.d_calculated_numneigh.ptr);
    check(cudaGetLastError());
}

StruFactorData::StruFactorData(int center_count, size_t pair_count)
    : group_count(checked_center_count(center_count)),
      d_group_indices(group_count), d_group_numneigh(static_cast<size_t>(group_count)+1),
      d_neigh_in_cutoff_r(group_count), d_calculated_numneigh(pair_count),
      d_group_dminneigh(geometry_size(pair_count)) {
    set_logger();
}

void StruFactorData::set_logger(Logger logger) {
    d_group_indices.set_name("d_group_indices", logger);
    d_group_numneigh.set_name("d_group_numneigh", logger);
    d_neigh_in_cutoff_r.set_name("d_neigh_in_cutoff_r", logger);
    d_calculated_numneigh.set_name("d_calculated_numneigh", logger);
    d_group_dminneigh.set_name("d_group_dminneigh", logger);
}

void strufactor_forward(const StruFactorData& data, double q, double cutoff_rsq,
                        ArrayView<double> values, cudaStream_t stream, int block_size) {
    const auto view = kernel_view(data);
    check_wave_number(q);
    check_arguments(view,cutoff_rsq,block_size);
    if (!view.group_count) return;
    require_scalar_items(values, "values");
    values.wait_on(stream);
    forward_kernel<false><<<(view.group_count-1)/block_size+1,block_size,0,stream>>>(
        view,q,cutoff_rsq,{},nullptr,values);
    check_launch();
}
void strufactor_forward(const StruFactorData& data, double q, double cutoff_rsq,
                        double* values, cudaStream_t stream, int block_size) {
    if (!values && data.group_count)
        throw std::invalid_argument("Stru_factor: missing value buffer");
    strufactor_forward(data, q, cutoff_rsq, unchecked_scalars(values), stream, block_size);
}

void strufactor_backward(const StruFactorData& data, double q, double cutoff_rsq,
                         ArrayView<const double> center_adjoint, ArrayView<double> force,
                         ArrayView<double> virial, cudaStream_t stream, int block_size) {
    const auto view = kernel_view(data);
    check_wave_number(q);
    check_arguments(view,cutoff_rsq,block_size);
    if (!view.group_count) return;
    require_scalar_items(center_adjoint, "center_adjoint");
    center_adjoint.wait_on(stream);
    require_xyz(force);
    force.wait_on(stream);
    require_virial6(virial);
    virial.wait_on(stream);
    backward_kernel<false><<<(view.group_count-1)/block_size+1,block_size,0,stream>>>(
        view,q,cutoff_rsq,{},nullptr,center_adjoint,force,virial);
    check_launch();
}
void strufactor_backward(const StruFactorData& data, double q, double cutoff_rsq,
                         const double* center_adjoint, double* force, double* virial,
                         cudaStream_t stream, int block_size) {
    if ((!center_adjoint || !force) && data.group_count)
        throw std::invalid_argument("Stru_factor: missing adjoint/force buffer");
    strufactor_backward(data, q, cutoff_rsq, unchecked_scalars(center_adjoint),
                        unchecked_xyz(force), virial ? unchecked_virial6(virial) : empty_virial(),
                        stream, block_size);
}

void strufactor_chem_forward(const StruFactorData& data, double q, double cutoff_rsq,
                             ArrayView<const int> atom_types, ArrayView<const double> type_weights,
                             ArrayView<double> values, cudaStream_t stream, int block_size) {
    const auto view = kernel_view(data);
    check_wave_number(q);
    check_arguments(view,cutoff_rsq,block_size);
    if (!view.group_count) return;
    require_types(atom_types);
    atom_types.wait_on(stream);
    require_scalar_items(type_weights, "type_weights");
    type_weights.wait_on(stream);
    require_scalar_items(values, "values");
    values.wait_on(stream);
    forward_kernel<true><<<(view.group_count-1)/block_size+1,block_size,0,stream>>>(
        view,q,cutoff_rsq,atom_types,type_weights.data,values);
    check_launch();
}
void strufactor_chem_forward(const StruFactorData& data, double q, double cutoff_rsq,
                             const int* atom_types, const double* type_weights,
                             double* values, cudaStream_t stream, int block_size) {
    if ((!atom_types || !type_weights || !values) && data.group_count)
        throw std::invalid_argument("Stru_factor_chem: missing type/weight/value buffer");
    ArrayView<const double> weights{};
    weights.data = type_weights;
    weights.item_count = type_weights ? 1 : 0;
    weights.component_count = 1;
    weights.item_stride = 1;
    weights.component_stride = 1;
    weights.storage_size = type_weights ? std::numeric_limits<size_t>::max()/2 : 0;
    weights.memory = MemorySpace::Cuda;
    strufactor_chem_forward(data, q, cutoff_rsq, unchecked_types(atom_types), weights,
                            unchecked_scalars(values), stream, block_size);
}

void strufactor_chem_backward(const StruFactorData& data, double q, double cutoff_rsq,
                              ArrayView<const int> atom_types, ArrayView<const double> type_weights,
                              ArrayView<const double> center_adjoint, ArrayView<double> force,
                              ArrayView<double> virial, cudaStream_t stream, int block_size) {
    const auto view = kernel_view(data);
    check_wave_number(q);
    check_arguments(view,cutoff_rsq,block_size);
    if (!view.group_count) return;
    require_types(atom_types);
    atom_types.wait_on(stream);
    require_scalar_items(type_weights, "type_weights");
    type_weights.wait_on(stream);
    require_scalar_items(center_adjoint, "center_adjoint");
    center_adjoint.wait_on(stream);
    require_xyz(force);
    force.wait_on(stream);
    require_virial6(virial);
    virial.wait_on(stream);
    backward_kernel<true><<<(view.group_count-1)/block_size+1,block_size,0,stream>>>(
        view,q,cutoff_rsq,atom_types,type_weights.data,center_adjoint,force,virial);
    check_launch();
}
void strufactor_chem_backward(const StruFactorData& data, double q, double cutoff_rsq,
                              const int* atom_types, const double* type_weights,
                              const double* center_adjoint, double* force, double* virial,
                              cudaStream_t stream, int block_size) {
    if ((!atom_types || !type_weights || !center_adjoint || !force) && data.group_count)
        throw std::invalid_argument("Stru_factor_chem: missing type/weight/adjoint/force buffer");
    ArrayView<const double> weights{};
    weights.data = type_weights;
    weights.item_count = type_weights ? 1 : 0;
    weights.component_count = 1;
    weights.item_stride = 1;
    weights.component_stride = 1;
    weights.storage_size = type_weights ? std::numeric_limits<size_t>::max()/2 : 0;
    weights.memory = MemorySpace::Cuda;
    strufactor_chem_backward(data, q, cutoff_rsq, unchecked_types(atom_types), weights,
                             unchecked_scalars(center_adjoint), unchecked_xyz(force),
                             virial ? unchecked_virial6(virial) : empty_virial(), stream, block_size);
}

void weighted_chem_forward(const StruFactorData& data, double cutoff_rsq,
                           ArrayView<const int> atom_types, ArrayView<const double> type_weights,
                           ArrayView<double> values, cudaStream_t stream, int block_size) {
    const auto view = kernel_view(data);
    check_arguments(view,cutoff_rsq,block_size);
    if (!view.group_count) return;
    require_types(atom_types);
    atom_types.wait_on(stream);
    require_scalar_items(type_weights, "type_weights");
    type_weights.wait_on(stream);
    require_scalar_items(values, "values");
    values.wait_on(stream);
    forward_kernel<true,false><<<(view.group_count-1)/block_size+1,block_size,0,stream>>>(
        view,0.0,cutoff_rsq,atom_types,type_weights.data,values);
    check_launch();
}
void weighted_chem_forward(const StruFactorData& data, double cutoff_rsq,
                           const int* atom_types, const double* type_weights,
                           double* values, cudaStream_t stream, int block_size) {
    if ((!atom_types || !type_weights || !values) && data.group_count)
        throw std::invalid_argument("WEIGHT_CHEM: missing type/weight/value buffer");
    ArrayView<const double> weights{};
    weights.data = type_weights;
    weights.item_count = type_weights ? 1 : 0;
    weights.component_count = 1;
    weights.item_stride = 1;
    weights.component_stride = 1;
    weights.storage_size = type_weights ? std::numeric_limits<size_t>::max()/2 : 0;
    weights.memory = MemorySpace::Cuda;
    weighted_chem_forward(data, cutoff_rsq, unchecked_types(atom_types), weights,
                          unchecked_scalars(values), stream, block_size);
}
void weighted_chem_backward(const StruFactorData& data, double cutoff_rsq,
                            ArrayView<const int> atom_types, ArrayView<const double> type_weights,
                            ArrayView<const double> center_adjoint, ArrayView<double> force,
                            ArrayView<double> virial, cudaStream_t stream, int block_size) {
    const auto view = kernel_view(data);
    check_arguments(view,cutoff_rsq,block_size);
    if (!view.group_count) return;
    require_types(atom_types);
    atom_types.wait_on(stream);
    require_scalar_items(type_weights, "type_weights");
    type_weights.wait_on(stream);
    require_scalar_items(center_adjoint, "center_adjoint");
    center_adjoint.wait_on(stream);
    require_xyz(force);
    force.wait_on(stream);
    require_virial6(virial);
    virial.wait_on(stream);
    backward_kernel<true,false><<<(view.group_count-1)/block_size+1,block_size,0,stream>>>(
        view,0.0,cutoff_rsq,atom_types,type_weights.data,center_adjoint,force,virial);
    check_launch();
}
void weighted_chem_backward(const StruFactorData& data, double cutoff_rsq,
                            const int* atom_types, const double* type_weights,
                            const double* center_adjoint, double* force, double* virial,
                            cudaStream_t stream, int block_size) {
    if ((!atom_types || !type_weights || !center_adjoint || !force) && data.group_count)
        throw std::invalid_argument("WEIGHT_CHEM: missing type/weight/adjoint/force buffer");
    ArrayView<const double> weights{};
    weights.data = type_weights;
    weights.item_count = type_weights ? 1 : 0;
    weights.component_count = 1;
    weights.item_stride = 1;
    weights.component_stride = 1;
    weights.storage_size = type_weights ? std::numeric_limits<size_t>::max()/2 : 0;
    weights.memory = MemorySpace::Cuda;
    weighted_chem_backward(data, cutoff_rsq, unchecked_types(atom_types), weights,
                           unchecked_scalars(center_adjoint), unchecked_xyz(force),
                           virial ? unchecked_virial6(virial) : empty_virial(), stream, block_size);
}

namespace {

__device__ __forceinline__ RadialValue radial_window(double r2, double cutoff_rsq) {
    if (r2 > cutoff_rsq) return {0.0,0.0};
    const double onset = 0.8*cutoff_rsq;
    if (r2 <= onset) return {1.0,0.0};
    const double width = cutoff_rsq-onset;
    const double scaled_distance = (r2-onset)/width;
    return {1.0-scaled_distance*scaled_distance*scaled_distance,
            -6.0*scaled_distance*scaled_distance/width};
}

__device__ __forceinline__ RadialValue radial(double r2, double q, double cutoff_rsq) {
    if (r2 > cutoff_rsq) return {0.0, 0.0};
    const double x2 = q*q*r2;
    double sinc, dsinc_over_r;
    if (x2 < 1e-6) {
        sinc = 1.0 + x2*(-1.0/6.0 + x2*(1.0/120.0 - x2/5040.0));
        dsinc_over_r = q*q*(-1.0/3.0 + x2*(1.0/30.0 + x2*(-1.0/840.0 + x2/45360.0)));
    } else {
        const double x = sqrt(x2);
        double sn, cs;
        sincos(x, &sn, &cs);
        sinc = sn/x;
        dsinc_over_r = q*q*(cs-sinc)/x2;
    }
    const auto window = radial_window(r2,cutoff_rsq);
    return {sinc*window.value, dsinc_over_r*window.value + sinc*window.derivative_over_r};
}

template<bool UseStructureFactor>
__device__ __forceinline__ RadialValue pair_radial(double r2, double q, double cutoff_rsq) {
    if constexpr (UseStructureFactor) return radial(r2,q,cutoff_rsq);
    else return radial_window(r2,cutoff_rsq);
}

template<bool UseChemicalWeights, bool UseStructureFactor>
__global__ void forward_kernel(StruFactorKernelView view, double q, double cutoff_rsq,
                               ArrayView<const int> atom_types, const double* type_weights,
                               ArrayView<double> values) {
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= view.group_count) return;
    double value = 1.0;
    double center_weight = 1.0;
    if constexpr (UseChemicalWeights) center_weight = type_weights[atom_types(view.d_group_indices[k])];
    const auto first = view.d_group_numneigh[k];
    for (int t = 0; t < view.d_neigh_in_cutoff_r[k]; ++t) {
        const auto pair = first+t;
        double pair_weight = 1.0;
        if constexpr (UseChemicalWeights)
            pair_weight = center_weight*type_weights[atom_types(view.d_calculated_numneigh[pair])];
        value += pair_weight*pair_radial<UseStructureFactor>(view.d_group_dminneigh[4*pair+3], q, cutoff_rsq).value;
    }
    values(view.d_group_indices[k]) = value;
}

template<bool UseChemicalWeights, bool UseStructureFactor>
__global__ void backward_kernel(StruFactorKernelView view, double q, double cutoff_rsq,
                                ArrayView<const int> atom_types, const double* type_weights,
                                ArrayView<const double> adjoint, ArrayView<double> force,
                                ArrayView<double> virial) {
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= view.group_count) return;
    const double seed = adjoint(k);
    if (seed == 0.0) return;
    const auto center = view.d_group_indices[k];
    double center_weight = 1.0;
    if constexpr (UseChemicalWeights) center_weight = type_weights[atom_types(center)];
    double center_force[3] = {0.0,0.0,0.0};
    double center_virial[6] = {0.0,0.0,0.0,0.0,0.0,0.0};
    const auto first = view.d_group_numneigh[k];
    const bool write_virial = virial.data != nullptr;
    for (int t = 0; t < view.d_neigh_in_cutoff_r[k]; ++t) {
        const auto pair = first+t;
        const auto neighbor = view.d_calculated_numneigh[pair];
        const double* dr = view.d_group_dminneigh+4*pair;
        double pair_weight = 1.0;
        if constexpr (UseChemicalWeights) pair_weight = center_weight*type_weights[atom_types(neighbor)];
        const double scale = seed*pair_weight*pair_radial<UseStructureFactor>(dr[3],q,cutoff_rsq).derivative_over_r;
        double f[3];
        for (int c=0; c<3; ++c) {
            f[c] = scale*dr[c];
            center_force[c] += f[c];
            atomicAdd(&force(neighbor, c), -f[c]);
        }
        if (write_virial) {
            const double half[6] = {-.5*dr[0]*f[0], -.5*dr[1]*f[1], -.5*dr[2]*f[2],
                                    -.5*dr[0]*f[1], -.5*dr[0]*f[2], -.5*dr[1]*f[2]};
            for (int c=0; c<6; ++c) {
                center_virial[c] += half[c];
                atomicAdd(&virial(neighbor, c), half[c]);
            }
        }
    }
    for (int c=0; c<3; ++c) atomicAdd(&force(center, c), center_force[c]);
    if (write_virial)
        for (int c=0; c<6; ++c) atomicAdd(&virial(center, c), center_virial[c]);
}
} // namespace
} // namespace MetaD_zqc
