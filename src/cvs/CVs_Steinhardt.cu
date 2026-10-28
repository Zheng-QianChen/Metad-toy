#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <sstream>
#include <iomanip>
#include "CV_Steinhardt.h"
#include "zqc_cv_debug.h"
#include "CV_Steinhardt_math.h"
#include "zqc_cv_parse.h"
using namespace MetaD_zqc;
namespace {
__global__ void physical_force_from_gradient(const double* gradient, double* force, size_t count) {
    const size_t i = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i < count) force[i] = -gradient[i];
}
template<class T> void resize_host(std::vector<T>& storage, T*& alias, size_t count) {
    storage.resize(std::max<size_t>(count,1));
    alias=storage.data();
}
}
MetaD_zqc::CV* MetaD_zqc::Steinhardt::create(CVContext& context,
                                            int narg, char **arg, int &i){
    const auto logger = context.logger;

    std::string cal_name = arg[i];

    MetaD_zqc::SteinhardtRequest req{};
    req.cal_name = cal_name;
    // 原子环境分析-初始设置
    // Usage: STEINH <Q/L> <3/4/6> <group>
    ERR_COND(i + 3 >= narg, "Error: STEINH command requires \"STEINH <Q/L> <3/4/6> <group> \".");
    req.Q_type_str = arg[i+1];
    req.Q_num   = cv_integer(arg[i+2]);
    req.group_name = arg[i+3];
    req.group_id = context.engine.selection_id(req.group_name);
    ERR_COND(req.group_id < all_atoms, "Error: Steinhardt group name %s not found.", req.group_name);
    //参数有效性
    ERR_COND((req.Q_num != 3 && req.Q_num != 4 && req.Q_num != 6),"Error: Steinhardt order L must be 3, 4, or 6.");
    ERR_COND((strcmp(req.Q_type_str, "Q") != 0 && strcmp(req.Q_type_str, "L") != 0), "Error: Steinhardt type must be 'Q' (per-center) or 'L' (neighbor-averaged).");
    // 进阶设置
    // default values
    req.cutoff_r = 4.0;
    // req.cutoff_Natoms = 12;
    req.d_block_size = 128;
    req.cutoff_eps_r = 1e-6;

    std::string temp_name;

    int iarg=4 + i;
    while (iarg < narg) {
        if (strcmp(arg[iarg], "cutoff_r") == 0) {
            ERR_COND((iarg + 1 >= narg) ,"Error: \'cutoff_r\' keyword requires a value");
            req.cutoff_r = cv_number(arg[iarg + 1]);
            iarg += 2;
        // } else if (strcmp(arg[iarg], "cutoff_Natoms") == 0) {
        //     ERR_COND((iarg + 1 >= narg), "Error: \'cutoff_Natoms\' keyword requires an integer");
        //     req.cutoff_Natoms = cv_integer(arg[iarg + 1]);
        //     iarg += 2;
        } else if (strcmp(arg[iarg], "cutoff_eps_r") == 0) {
            ERR_COND((iarg + 1 >= narg), "Error: \'cutoff_eps_r\' keyword requires a value");
            req.cutoff_eps_r = cv_number(arg[iarg + 1]);
            iarg += 2;
        } else if (strcmp(arg[iarg], "d_block_size") == 0) {
            ERR_COND((iarg + 1 >= narg), "Error: \'d_block_size\' keyword requires an integer");
            req.d_block_size = cv_integer(arg[iarg + 1]);
            ERR_COND(req.d_block_size <= 0, "Error: \'d_block_size\' must be > 0");
            iarg += 2;
        } else if (strcmp(arg[iarg], "SW_FUNC_r") == 0) {
            ERR_COND((iarg + 1 >= narg), "Error: \'SW_FUNC_r\' keyword requires a value");
            temp_name = arg[iarg + 1];
            req.SW_FUNC_r = context.switching_function(temp_name);
            iarg += 2;
        } else if (strcmp(arg[iarg], "SW_FUNC_cv") == 0) {
            ERR_COND(1,
                "STEINH: SW_FUNC_cv has moved to Graph; use RAW inputs and a switching expression in DIM");
        } else {
            break;
        }
    }
    if (req.SW_FUNC_r == nullptr) {
        req.SW_FUNC_r = MetaD_zqc::SwitchFunction::get_default_step();
    }
    const auto radial_support = MetaD_zqc::SwitchFunction::query_radial_support(
        req.SW_FUNC_r->params, req.cutoff_eps_r);
    ERR_COND(radial_support.error != nullptr, "Error: STEINH SW_FUNC_r: %s",
             radial_support.error);
    if (radial_support.has_radius()) {
        LOG("Logging: STEINH radial support=%s cutoff_r=%g -> %g eps=%g (configured cutoff replaced)",
            radial_support.kind == MetaD_zqc::RadialSupportKind::Finite ? "finite" : "epsilon",
            req.cutoff_r, radial_support.radius, req.cutoff_eps_r);
        req.cutoff_r = radial_support.radius;
    } else {
        // Constant radial weights have no natural support. Preserve the existing
        // configured/default cutoff as an explicit hard boundary in the kernels.
        LOG("Logging: STEINH constant radial weight uses configured cutoff_r=%g", req.cutoff_r);
    }
    ERR_COND(!std::isfinite(req.cutoff_r) || !(req.cutoff_r > 0.0),
             "Error: STEINH effective radial cutoff must be finite and positive.");
    MetaD_zqc::Steinhardt_env *temp_env = MetaD_zqc::Steinhardt_env::get_or_create(context, req);
    DEBUG_LOG("Steinhardt_env is %p neigh_id=%d", temp_env, temp_env->neigh_id);
    std::string env_setNum = temp_env->get_env_key();
    i = iarg;
    return MetaD_zqc::create_steinhardt_cv(context, env_setNum, req.group_id, req.Q_num, temp_env, req);
}

MetaD_zqc::Steinhardt* MetaD_zqc::create_steinhardt_cv(CVContext& context,
                                std::string env_setNum, int group_id, int Q_num,
                                MetaD_zqc::Steinhardt_env* my_env,
                                MetaD_zqc::SteinhardtRequest req){
    if (strcmp(req.Q_type_str, "Q") == 0){
        if (Q_num==3){
            return new MetaD_zqc::STEIN_QL<3>(context, env_setNum, group_id, Q_num, my_env, req.d_block_size);
        } else if (Q_num==4){
            return new MetaD_zqc::STEIN_QL<4>(context, env_setNum, group_id, Q_num, my_env, req.d_block_size);
        } else if (Q_num==6){
            return new MetaD_zqc::STEIN_QL<6>(context, env_setNum, group_id, Q_num, my_env, req.d_block_size);
        }
    } else if (strcmp(req.Q_type_str,"L") == 0){
        if (Q_num==3){
            return new MetaD_zqc::STEIN_LocalQL<3>(context, env_setNum, group_id, my_env, req.d_block_size);
        } else if (Q_num==4){
            return new MetaD_zqc::STEIN_LocalQL<4>(context, env_setNum, group_id, my_env, req.d_block_size);
        } else if (Q_num==6){
            return new MetaD_zqc::STEIN_LocalQL<6>(context, env_setNum, group_id, my_env, req.d_block_size);
        }
    }
    return nullptr;
}

std::map<std::string, MetaD_zqc::Steinhardt_env*> MetaD_zqc::Steinhardt_env::env_pool;

namespace {
std::string steinhardt_env_key(const EngineAdapter* owner, int group_id, double cutoff_r, double cutoff_eps_r,
                               bool loc_flag, const MetaD_zqc::SwitchFunction* sw) {
    std::ostringstream oss;
    oss << owner << "_" << group_id << "_"
        << std::setprecision(17) << cutoff_r << "_"
        << cutoff_eps_r << "_"
        << (loc_flag ? 1 : 0) << "_";
    if (sw == nullptr) {
        oss << "nosw";
    } else {
        const MetaD_zqc::SwitchFunctionRequest& p = sw->params;
        oss << static_cast<int>(p.type) << "_"
            << p.r_0 << "_" << p.d_0 << "_" << p.alpha << "_"
            << p.n << "_" << p.m;
    }
    return oss.str();
}
}

MetaD_zqc::Steinhardt_env* MetaD_zqc::Steinhardt_env::get_or_create(CVContext& context,
                                            MetaD_zqc::SteinhardtRequest req
                                            ) {
    int group_id = req.group_id;
    double cutoff_r = req.cutoff_r;
    double cutoff_eps_r= req.cutoff_eps_r;
    bool LOC_flag = (strcmp(req.Q_type_str,"L") == 0);
    // Radial switch is part of the environment: different r0/d0/n/m/alpha must not share one env.
    std::string key = steinhardt_env_key(&context.engine, group_id, cutoff_r, cutoff_eps_r, LOC_flag, req.SW_FUNC_r);
    // 2. check if the environment already exist in the pool
    if (!(env_pool.count(key))) {
        if (LOC_flag){
            // 3. new environment and store it in the pool if not exist
            MetaD_zqc::STEIN_LocalQL_env *new_env = new MetaD_zqc::STEIN_LocalQL_env(context, req);
            env_pool[key] = new_env; // store the new environment in the pool
        } else {
            // 3. new environment and store it in the pool if not exist
            MetaD_zqc::Steinhardt_env *new_env = new MetaD_zqc::Steinhardt_env(context, req);
            env_pool[key] = new_env; // store the new environment in the pool
        }
    }
    env_pool[key]->register_env(); // increase reference count
    return env_pool[key]; // if exits, return the existing environment
}


std::string MetaD_zqc::Steinhardt_env::get_env_key(){
    return steinhardt_env_key(&engine, group_id, cutoff_r, cutoff_eps_r, LOC_flag, my_r_SWfunc);
}


Steinhardt_env::Steinhardt_env(CVContext& context, SteinhardtRequest request)
    : CV_info(context), my_r_SWfunc(request.SW_FUNC_r), cutoff_r(request.cutoff_r),
      cutoff_eps_r(request.cutoff_eps_r), group_id(request.group_id), d_block_size(request.d_block_size) {
    LOC_flag = std::strcmp(request.Q_type_str,"L")==0;
    // Local Q 需要所有 I/J 的完整行；两层 halo 由引擎根据 ghost_centers 准备。
    NeighborRequest neighbors;
    neighbors.cutoff=cutoff_r;
    neighbors.centers=LOC_flag ? all_atoms : group_id;
    neighbors.ghost_centers=LOC_flag;
    neighbors.host_csr=!LOC_flag;
    neigh_id=engine.neighbors().request(neighbors);
}

void Steinhardt_env::refresh_centers() {
    const auto members=engine.selection_members(group_id);
    members.local_ids.wait_host();
    const int total=engine.total_atom_count();
    centers_changed=!members.generation || members.generation!=center_generation ||
        group_count!=members.owned_count || center_storage.size()!=size_t(total);
    group_count=members.owned_count;
    if (centers_changed) {
        resize_host(center_storage,h_group_indices,total);
        for(int center=0;center<group_count;++center) h_group_indices[center]=members.local_ids(center);
        center_generation=members.generation;
    }
    positions=engine.positions(MemorySpace::Cuda);
    positions.validate();
    positions.wait_on(execution.get());
    ERR_COND(positions.memory!=MemorySpace::Cuda || positions.component_count!=3 || positions.item_count<size_t(total),
             "STEINH: invalid device coordinates");
    // Q 的 half-pair 反向需要两端的组成员资格；Local Q 使用自己的 I/J/K 标记。
    if (!LOC_flag) {
        selection_flags=engine.selection_flags(group_id,MemorySpace::Cuda);
        selection_flags.validate();
        selection_flags.wait_on(execution.get());
        ERR_COND(selection_flags.memory!=MemorySpace::Cuda || selection_flags.component_count!=1 || selection_flags.item_count<size_t(total),
                 "STEINH: invalid selection flags");
    }
    block_num=(group_count+d_block_size-1)/d_block_size;
}

void Steinhardt_env::get_env() {
    const auto candidates=engine.neighbors().prepare(neigh_id);
    candidates.host_centers.wait_host(); candidates.host_offsets.wait_host(); candidates.host_neighbors.wait_host();
    ERR_COND(candidates.coordinates!=NeighborCoordinates::PeriodicImages,
             "STEINH requires image-resolved neighbor coordinates");
    const int total=engine.total_atom_count();
    if(!init_flag || cached_neigh_generation!=candidates.topology_generation) {
        // 建立中心行到 CSR 的映射；支持 provider 的任意行顺序和 Host 步长。
        std::vector<int> row_by_atom(total,-1);
        for(size_t row=0;row<candidates.host_centers.item_count;++row)
            row_by_atom.at(candidates.host_centers(row))=int(row);
        resize_host(candidate_offset_storage,h_group_numneigh,group_count+1);
        h_group_numneigh[0]=0;
        candidate_neighbor_storage.clear();
        std::vector<int> full_to_half, half_i,half_j,half_to_full;
        const auto ids=engine.atom_ids();
        const auto flags=engine.selection_flags(group_id,MemorySpace::Host);
        ids.wait_host(); flags.wait_host();
        for(int center=0;center<group_count;++center) {
            const int first=h_group_indices[center], row=row_by_atom[first];
            ERR_COND(row<0,"STEINH: missing neighbor row for owned center");
            for(int entry=candidates.host_offsets(row);entry<candidates.host_offsets(row+1);++entry) {
                const int second=candidates.host_neighbors(entry);
                const int full=int(candidate_neighbor_storage.size());
                candidate_neighbor_storage.push_back(second);
                if(!flags(second) || ids(first)<=ids(second)) {
                    full_to_half.push_back(int(half_i.size()));
                    half_i.push_back(first);half_j.push_back(second);half_to_full.push_back(full);
                } else full_to_half.push_back(-1);
            }
            h_group_numneigh[center+1]=int(candidate_neighbor_storage.size());
        }
        h_firstneigh_ptrs=candidate_neighbor_storage.data();
        d_group_indices.grow_to(group_count,__FILE__,__LINE__);
        d_group_indices.upload_from(h_group_indices,group_count, execution.get());
        d_group_numneigh.grow_to(group_count+1,__FILE__,__LINE__);
        d_group_numneigh.upload_from(h_group_numneigh,group_count+1, execution.get());
        const size_t pairs=candidate_neighbor_storage.size();
        d_firstneigh_ptrs.grow_to(pairs,__FILE__,__LINE__);
        d_firstneigh_ptrs.upload_from(h_firstneigh_ptrs,pairs, execution.get());
        d_full_to_half.grow_to(pairs,__FILE__,__LINE__);
        d_full_to_half.upload_from(full_to_half.data(),pairs, execution.get());
        n_half_candidates=int(half_i.size());
        d_half_pair_i.grow_to(n_half_candidates,__FILE__,__LINE__);
        d_half_pair_j.grow_to(n_half_candidates,__FILE__,__LINE__);
        d_half_to_full.grow_to(n_half_candidates,__FILE__,__LINE__);
        d_active_pair_mask.grow_to(n_half_candidates,__FILE__,__LINE__);
        d_active_pair_ids.grow_to(n_half_candidates,__FILE__,__LINE__);
        d_half_pair_i.upload_from(half_i.data(),n_half_candidates, execution.get());
        d_half_pair_j.upload_from(half_j.data(),n_half_candidates, execution.get());
        d_half_to_full.upload_from(half_to_full.data(),n_half_candidates, execution.get());
        cached_neigh_generation=candidates.topology_generation;init_flag=true;
    }
    d_neigh_in_cutoff_r.grow_to(std::max(total,1),__FILE__,__LINE__);
    d_neigh_in_switching.grow_to(std::max(total,1),__FILE__,__LINE__);
    d_calculated_numneigh.grow_to(candidate_neighbor_storage.size(),__FILE__,__LINE__);
    d_neigh_in_cutoff_r.clear_async(execution.get());d_neigh_in_switching.clear_async(execution.get());d_active_pair_mask.clear_async(execution.get());
    if(group_count) get_environment_Steinhardt_Q<<<block_num,d_block_size,0,execution.get()>>>(
        my_r_SWfunc->params,group_count,0,cutoff_r,cutoff_eps_r,
        d_group_indices.ptr,d_group_numneigh.ptr,d_firstneigh_ptrs.ptr,positions,d_full_to_half.ptr,
        d_neigh_in_cutoff_r.ptr,d_active_pair_mask.ptr,d_neigh_in_switching.ptr,d_calculated_numneigh.ptr);
    CUDA_LAUNCH_CHECK();
    d_calculated_firstneigh_ptrs.grow_to(group_count+1,__FILE__,__LINE__);
    d_calculated_firstneigh_ptrs.clear_async(execution.get());
    if(group_count) d_neigh_in_cutoff_r.scan_to(d_calculated_firstneigh_ptrs,group_count, execution.get());
    n_active_pairs=0;
    if(n_half_candidates) d_active_pair_mask.flag_to(d_active_pair_ids,n_half_candidates,&n_active_pairs, execution.get());
    resize_host(active_offset_storage,h_calculated_firstneigh_ptrs,group_count+1);
    d_calculated_firstneigh_ptrs.download_to(h_calculated_firstneigh_ptrs,group_count+1, execution.get());
    num_of_all_calc_fullpair=h_calculated_firstneigh_ptrs[group_count];
    resize_host(weight_storage,h_neigh_in_switching,total);
    if(total) d_neigh_in_switching.download_to(h_neigh_in_switching,total, execution.get());
    geometry_evaluation=engine.evaluation_id();
}

template<int L> STEIN_QL<L>::STEIN_QL(CVContext& context,std::string key,int,int,
                                    Steinhardt_env* environment,int block_size)
    : Steinhardt(context),d_block_size(block_size),env_setNum(std::move(key)),my_env(environment) {
    num_elements=2*(L+1);
}
template<int L> STEIN_QL<L>::~STEIN_QL() { my_env->unregister_env(); }
// CV 基类保留标量回调槽；Q 的规约与值变换由 Graph 构建。
template <int L>
auto MetaD_zqc::STEIN_QL<L>::set_CV_calculate(std::string) -> CV_Calculation {
    throw std::runtime_error("STEINH Q requires EXECUTOR GRAPH; use Q.AVE or Q.RAW with AVE(expr)");
}

template <int L>
auto MetaD_zqc::STEIN_QL<L>::set_CV_bias_force(std::string) -> CV_BiasForce {
    throw std::runtime_error("STEINH Q requires EXECUTOR GRAPH; gradients are supplied by Graph");
}

template <int L>
void MetaD_zqc::STEIN_QL<L>::bind_assembly(std::shared_ptr<const CVAssembly> assembly) {
    if (!assembly || assembly->selection_id() != my_env->group_id ||
        assembly->nlocal() != engine.owned_atom_count() || assembly->nghost() != (engine.total_atom_count()-engine.owned_atom_count()))
        throw std::invalid_argument("STEINH: incompatible atom assembly");
    if (my_env->atom_assembly != assembly) {
        my_env->atom_assembly = assembly;
        my_env->init_flag = false; // mask/组成员变化后重建候选段和半对索引。
        my_env->graph_geometry_dirty = true;
    }
    graph_values_ready = false;
    CV::bind_assembly(std::move(assembly));
}

template <int L>
void MetaD_zqc::STEIN_QL<L>::prepare_graph_evaluation() {
    graph_values_ready = false;
    my_env->graph_geometry_dirty = true;
}

template <int L>
void MetaD_zqc::STEIN_QL<L>::check_graph_values() const {
    if (!assembly_ || !graph_values_ready || graph_value_evaluation != engine.evaluation_id())
        throw std::logic_error("STEINH: prepare and base_calc before assembly/backward");
    if (assembly_->nlocal() != engine.owned_atom_count() || assembly_->nghost() != (engine.total_atom_count()-engine.owned_atom_count()) ||
        assembly_->group_count() != my_env->group_count)
        throw std::logic_error("STEINH: rebuild atom assembly after localid layout changes");
}

template <int L>
void MetaD_zqc::STEIN_QL<L>::assemble_values(ArrayView<double> values) {
    check_graph_values();
    assembly_->assemble_values(d_stein_ql.view(engine.total_atom_count()), values, my_env->execution.get());
}

template <int L>
void MetaD_zqc::STEIN_QL<L>::apply_adjoint(ArrayView<const double> leaf_adjoint) {
    check_graph_values();
    const int center_count = assembly_->group_count();
    const int atom_count = engine.total_atom_count();
    d_center_adjoint.grow_to(center_count, __FILE__, __LINE__);
    assembly_->gather_adjoint(leaf_adjoint, d_center_adjoint.view(center_count), my_env->execution.get());
    h_center_adjoint.resize(center_count);
    h_atom_adjoint.assign(atom_count, 0.0);
    if (center_count) {
        d_center_adjoint.download_to(h_center_adjoint.data(), center_count, my_env->execution.get());
        for (int center = 0; center < center_count; ++center)
            h_atom_adjoint[assembly_->local_ids()[center]] = h_center_adjoint[center];
    }

    // 半对只在一端所在 rank 计算，先取得另一端的 qlm、q、邻居权重和及图梯度。
    // 空中心 rank 同样参加通信，接收或转发其 owned/ghost 原子的贡献。
    // MPI 即将访问 Host 球谐和值，只等待本环境的前向完成。
    if (atom_count) {
        d_stein_ql.download_to(stein_q, atom_count, my_env->execution.get());
        d_stein_qlm.download_to(h_stein_qlm, size_t(atom_count)*num_elements, my_env->execution.get());
    }
    engine.communicate_forward(*this);
    d_atom_adjoint.grow_to(atom_count, __FILE__, __LINE__);
    if (atom_count) {
        d_atom_adjoint.upload_from(h_atom_adjoint.data(), atom_count, my_env->execution.get());
        d_stein_qlm.upload_from(h_stein_qlm, size_t(atom_count)*num_elements, my_env->execution.get());
        d_stein_ql.upload_from(stein_q, atom_count, my_env->execution.get());
        my_env->d_neigh_in_switching.upload_from(my_env->h_neigh_in_switching, atom_count, my_env->execution.get());
    }

    // localid 连续存储：梯度 [3*localid+c]，完整 virial [9*localid+c]。
    // 梯度已含 dV/dq_i；提交时转成物理力，由引擎完成 ghost 归并及累加。
    d_dcvdx.grow_to(size_t(atom_count)*3, __FILE__, __LINE__);
    d_a_virial.grow_to(size_t(atom_count)*9, __FILE__, __LINE__);
    d_dcvdx.clear_async(my_env->execution.get());
    d_a_virial.clear_async(my_env->execution.get());
    resize_host(gradient_storage,h_dcvdx,size_t(atom_count)*3);
    resize_host(virial_storage,h_a_virial,size_t(atom_count)*9);
    if (my_env->n_active_pairs > 0) {
        const int pair_blocks = (my_env->n_active_pairs+d_block_size-1)/d_block_size;
        steinhardt_dcv_kernel<L><<<pair_blocks,d_block_size,0,my_env->execution.get()>>>(
            my_env->my_r_SWfunc->params,
            my_env->n_active_pairs, 1, my_env->selection_flags,
            my_env->d_active_pair_ids.ptr, my_env->d_half_pair_i.ptr, my_env->d_half_pair_j.ptr,
            my_env->positions,
            my_env->d_neigh_in_switching.ptr, d_stein_ql.ptr, d_stein_qlm.ptr,
            d_dcvdx.ptr, d_a_virial.ptr, d_atom_adjoint.ptr);
        const auto status = cudaGetLastError();
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    }
    commit_coordinate_adjoint();
}

template <int L>
void MetaD_zqc::STEIN_QL<L>::base_calc() {
    graph_values_ready=false;
    compute_Q_peratoms();
    graph_value_evaluation=engine.evaluation_id();
    graph_values_ready=true;
}

template <int L>
void MetaD_zqc::STEIN_QL<L>::compute_Q_peratoms() {
    if(my_env->graph_geometry_dirty || my_env->geometry_evaluation!=engine.evaluation_id()) {
        my_env->refresh_centers();my_env->get_env();my_env->graph_geometry_dirty=false;
    }
    block_num=(my_env->group_count+d_block_size-1)/d_block_size;
    steinhardt_param_calc(stein_q);
}

template <int L>
void MetaD_zqc::STEIN_QL<L>::steinhardt_param_calc(double*) {
    const size_t total=engine.total_atom_count(), harmonics=total*num_elements;
    d_stein_ql.grow_to(total,__FILE__,__LINE__);d_stein_ql.clear_async(my_env->execution.get());
    d_stein_qlm.grow_to(harmonics,__FILE__,__LINE__);d_stein_qlm.clear_async(my_env->execution.get());
    call_steinhardt_cv_ql_i_kernel();CUDA_LAUNCH_CHECK();
    resize_host(value_storage,stein_q,total);
    resize_host(harmonic_storage,h_stein_qlm,harmonics);

}


template <int L>
void MetaD_zqc::STEIN_QL<L>::summary(FILE* f){}


template <int L>
void MetaD_zqc::STEIN_QL<L>::call_steinhardt_cv_ql_i_kernel(){
    ERR_COND((my_env == nullptr),"my_env is NULL! Cannot launch kernel.");
    if (my_env->group_count == 0) return;
    steinhardt_cv_kernel<L> <<<block_num,d_block_size,0,my_env->execution.get()>>>(
        (my_env)->my_r_SWfunc->params,
        (my_env->group_count),  (my_env->cutoff_r), (my_env->cutoff_eps_r),
        (my_env->d_group_indices.ptr),
        (my_env->d_neigh_in_cutoff_r.ptr),
        (my_env->d_group_numneigh.ptr),
        (my_env->d_calculated_firstneigh_ptrs.ptr),
        (my_env->d_calculated_numneigh.ptr),
        (my_env->positions),
        (my_env->d_neigh_in_switching.ptr),
        d_stein_qlm.ptr, nullptr,
        d_stein_ql.ptr) ;
}


template <int L>
int MetaD_zqc::STEIN_QL<L>::get_comm_forward_width() const {
    // q_lm、q_l、径向权重和、dV/dq_i，按每个原子连续打包。
    return num_elements + 3;
}

template <int L>
int MetaD_zqc::STEIN_QL<L>::pack_comm_forward_ubuf(
    int n, int *list, double *buf, int offset, int stride) {
    if (n > 0 && (!h_stein_qlm || !stein_q || !my_env->h_neigh_in_switching ||
                  h_atom_adjoint.size() < size_t(engine.total_atom_count())))
        throw std::logic_error("STEIN_QL: forward communication data is not ready");
    for (int k = 0; k < n; ++k) {
        const int atom = list[k];
        const int base = offset + k*stride;
        for (int c = 0; c < num_elements; ++c)
            buf[base+c] = h_stein_qlm[atom*num_elements+c];
        buf[base+num_elements] = stein_q[atom];
        buf[base+num_elements+1] = my_env->h_neigh_in_switching[atom];
        buf[base+num_elements+2] = h_atom_adjoint[atom];
    }
    return get_comm_forward_width();
}

template <int L>
void MetaD_zqc::STEIN_QL<L>::unpack_comm_forward_ubuf(
    int n, int first, double *buf, int offset, int stride) {
    if (n > 0 && (!h_stein_qlm || !stein_q || !my_env->h_neigh_in_switching ||
                  h_atom_adjoint.size() < size_t(engine.total_atom_count())))
        throw std::logic_error("STEIN_QL: forward communication data is not ready");
    for (int k = 0; k < n; ++k) {
        const int atom = first + k;
        const int base = offset + k*stride;
        for (int c = 0; c < num_elements; ++c)
            h_stein_qlm[atom*num_elements+c] = buf[base+c];
        stein_q[atom] = buf[base+num_elements];
        my_env->h_neigh_in_switching[atom] = buf[base+num_elements+1];
        h_atom_adjoint[atom] = buf[base+num_elements+2];
    }
}

// 引擎提交物理力，compute 仍可读取正号坐标梯度。提交时由适配器归并 ghost。
// 梯度在 GPU 取负为力后直接提交 Cuda View，适配器返回后源缓冲可立即复用。
template<int L> void STEIN_QL<L>::commit_coordinate_adjoint() {
    output_gradient_ready=false;
    const size_t total=engine.total_atom_count();
    d_physical_force.grow_to(total*3, __FILE__, __LINE__);
    if (total)
        physical_force_from_gradient<<<(total*3+255)/256,256,0,my_env->execution.get()>>>(
            d_dcvdx.ptr, d_physical_force.ptr, total*3);
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    auto force = d_physical_force.view(total, 3);
    auto virial = d_a_virial.view(total, 9);
    force.readiness = virial.readiness = device_readiness();
    engine.commit_force_virial(force, {virial, VirialRepresentation::Full9});
}
template<int L> ArrayView<const double> STEIN_QL<L>::peratom_values(const std::string& property) {
    const size_t total=engine.total_atom_count();
    if(property=="stein_q") {
        if (total) d_stein_ql.download_to(stein_q,total,my_env->execution.get());
        return {stein_q,total,1,1,1,value_storage.size(),MemorySpace::Host};
    }
    int component=property=="dcvdx_x"?0:property=="dcvdx_y"?1:property=="dcvdx_z"?2:-1;
    if(component<0) throw std::invalid_argument("STEINH: unknown per-atom property "+property);
    if(!h_dcvdx) throw std::logic_error("STEINH: coordinate gradient is not ready");
    if(!output_gradient_ready) {
        if (total) d_dcvdx.download_to(h_dcvdx, total*3, my_env->execution.get());
        output_gradient=gradient_storage;
        engine.communicate_reverse(*this);
        output_gradient_ready=true;
    }
    if(!total) return {nullptr,0,1,3,1,0,MemorySpace::Host};
    return {output_gradient.data()+component,total,1,3,1,output_gradient.size()-component,MemorySpace::Host};
}
template<int L> int STEIN_QL<L>::pack_comm_reverse_ubuf(int count,int first,double* buffer,int offset,int stride) {
    for(int entry=0;entry<count;++entry)
        for(int component=0;component<3;++component)
            buffer[offset+entry*stride+component]=output_gradient[3*(first+entry)+component];
    return 3;
}
template<int L> void STEIN_QL<L>::unpack_comm_reverse_ubuf(int count,int* ids,double* buffer,int offset,int stride) {
    for(int entry=0;entry<count;++entry)
        for(int component=0;component<3;++component)
            output_gradient[3*ids[entry]+component]+=buffer[offset+entry*stride+component];
}
__global__ void get_environment_Steinhardt_Q(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, int start_idx,
    double cutoff_r, double cut_sigma_eps,
    // in
    int *d_group_indices,
    int *d_group_numneigh,
    int *d_firstneigh_ptrs, MetaD_zqc::ArrayView<const double> d_x_flat,
    int *d_full_to_half,
    // out
    int *d_neigh_in_cutoff_r, int *d_active_pair_mask,
    double *d_neigh_in_switching,
    int *d_calculated_numneigh){

    #define sw_f(r) (MetaD_zqc::SwitchFunction::f(sw_params_rij, (r)))
    // get_environment_Steinhardt_Q in GPU
    int c_atom = blockIdx.x * blockDim.x + threadIdx.x;
    if(c_atom<calc_count){
        // double r2,temp_r2,temp_x,temp_y,temp_z,neigh_x,neigh_y,neigh_z;
        // double delt_x,delt_y,delt_z;
        int c_atom_calctag = c_atom+start_idx;
        int c_atom_loctag = d_group_indices[c_atom];
        int temp_tag;
        d_neigh_in_cutoff_r[c_atom_calctag] = 0;
        // c_glob_tag = h_tag[c_atom_loctag];
        double c_x = d_x_flat(c_atom_loctag,0);
        double c_y = d_x_flat(c_atom_loctag,1);
        double c_z = d_x_flat(c_atom_loctag,2);
        double sum_of_sigma_CNatoms = 0;
        int sum_of_numneigh = 0;
        int max_ii=0;
        //find curtoff_Natoms neigh
        int start_neigh = d_group_numneigh[c_atom];
        for (int neigh_atom=start_neigh;
                neigh_atom<d_group_numneigh[c_atom+1];
                neigh_atom++){
            int neigh_loctag = d_firstneigh_ptrs[neigh_atom];
            double r2,sigma_r,r;
            double temp_x,temp_y,temp_z;
            double neigh_x,neigh_y,neigh_z;
            double delt_x,delt_y,delt_z;
            // if (neigh_loctag < 0 ) continue;
            // int n_glob_tag = h_tag[neigh_loctag];
            neigh_x = d_x_flat(neigh_loctag,0);
            neigh_y = d_x_flat(neigh_loctag,1);
            neigh_z = d_x_flat(neigh_loctag,2);
            delt_x = (neigh_x - c_x);
            delt_y = (neigh_y - c_y);
            delt_z = (neigh_z - c_z);
            r2 = delt_x*delt_x + delt_y*delt_y + delt_z*delt_z;
            if (!(r2 > 0.0)) continue;
            r = sqrt(r2);
            // Candidate skin is never part of the CV's effective neighborhood.
            if (r >= cutoff_r) continue;
            sigma_r = sw_f(r);
            if ((sigma_r < cut_sigma_eps)) continue;
            // sigma_r >= cut_sigma_eps
            // 将通过筛选的原子压缩到一起，方便后续的线程数计算
            d_calculated_numneigh[start_neigh + sum_of_numneigh] = neigh_loctag;
            // 计算 sum_of_sigma_CNatoms
            sum_of_sigma_CNatoms += sigma_r;
            sum_of_numneigh ++;
            int half_idx = d_full_to_half[neigh_atom];
            if (half_idx != -1){
                d_active_pair_mask[half_idx] = 1;
            }
        }
        d_neigh_in_cutoff_r[c_atom_calctag] = sum_of_numneigh;
        d_neigh_in_switching[c_atom_loctag] = sum_of_sigma_CNatoms;
    }
    #undef sw_f
}



template <int L>
__global__ void steinhardt_cv_kernel(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, double cutoff_r, double cutoff_eps,
    int *d_group_indices,
    int *d_neigh_in_cutoff_r,
    int *d_group_numneigh,
    int *d_calculated_firstneigh_ptrs,
    int *d_calculated_numneigh,
    MetaD_zqc::ArrayView<const double> d_x_flat,
    double *d_neigh_in_switching,
    double *d_stein_qlm, double *d_stein_Ylm, double *d_stein_ql) {

    int c_atom_calctag = blockIdx.x * blockDim.x + threadIdx.x;
    #define sw_f(r) (MetaD_zqc::SwitchFunction::f(sw_params_rij, (r)))
    if (c_atom_calctag >= calc_count) return;

    int c_atom_loctag = d_group_indices[c_atom_calctag]; // 当前原子在local原子列表中的标签
    constexpr int lm_size = (L + 1) * 2;

    // 【与导数核函数完美镜像】的近邻数读取与基础寻址逻辑
    int neigh_num = d_neigh_in_cutoff_r[c_atom_calctag];
    int stein_qlm_base_id = c_atom_loctag * lm_size; // qlm 按 local 原子索引存储，便于通信

    // 如果没有邻居，直接清零退出
    double NFb_i_check = d_neigh_in_switching[c_atom_loctag];
    if (neigh_num == 0 || NFb_i_check < 1e-12) return;
    // because we multiply a swfunction, so it need to devide by its weight sum
    double inv_neigh = 1.0 / (double)NFb_i_check;

    // 在寄存器（栈）上初始化局部数组用于累加，避免频繁读写全局显存
    double local_qlm[lm_size] = {0.0};
    double c_x = d_x_flat(c_atom_loctag,0);
    double c_y = d_x_flat(c_atom_loctag,1);
    double c_z = d_x_flat(c_atom_loctag,2);

    int neigh_base = d_group_numneigh[c_atom_calctag];

    double qlm_value_weight = 0.0;

    for (int neigh_atom = 0; neigh_atom < neigh_num; neigh_atom++) {
        int neigh_loctag = d_calculated_numneigh[neigh_base + neigh_atom];
        double local_Ylm[lm_size] = {0.0};

        double neigh_x = d_x_flat(neigh_loctag,0);
        double neigh_y = d_x_flat(neigh_loctag,1);
        double neigh_z = d_x_flat(neigh_loctag,2);
        double delt_x = (neigh_x - c_x);
        double delt_y = (neigh_y - c_y);
        double delt_z = (neigh_z - c_z);
        double r2 = delt_x*delt_x + delt_y*delt_y + delt_z*delt_z;
        double r = sqrt(r2);
        double r_weight = sw_f(r);

        qlm_value_weight += r_weight;

        // Compute unweighted Ylm directly from the unit bond direction.
        const double inv_r = 1.0 / r;
        compute_Ylm_unit<L>(delt_x * inv_r, delt_y * inv_r, delt_z * inv_r, local_Ylm);
        #pragma unroll
        for (int m = 0; m < lm_size; ++m) {
            local_qlm[m] += r_weight * local_Ylm[m];
        }
    }

    // --- 循环外归一化与写回全局显存 ---

    #pragma unroll
    for (int i = 0; i < lm_size; i++) {
        local_qlm[i] *= inv_neigh;
        d_stein_qlm[stein_qlm_base_id + i] = local_qlm[i];
    }

    double ql_sq = local_qlm[0] * local_qlm[0];

    // 3. 对应你原代码的第三步：从 m = 1 开始累加已经归一化的各项平方和
    #pragma unroll
    for (int i = 1; i <= L; i++) {
        double re_part = local_qlm[i * 2 + 0];
        double im_part = local_qlm[i * 2 + 1];
        ql_sq += 2.0 * (re_part * re_part + im_part * im_part);
    }

    d_stein_ql[c_atom_loctag] = sqrt(ql_sq * 12.56637061435917295385/double(2*L + 1));
}
template __global__ void steinhardt_cv_kernel<3>(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, double cutoff_r, double cutoff_eps,
    int *d_group_indices,
    int *d_neigh_in_cutoff_r,
    int *d_group_numneigh,
    int *d_calculated_firstneigh_ptrs,
    int *d_calculated_numneigh,
    MetaD_zqc::ArrayView<const double> d_x_flat,
    double *d_neigh_in_switching,
    double *d_stein_qlm, double *d_stein_Ylm, double *d_stein_ql);
template __global__ void steinhardt_cv_kernel<4>(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, double cutoff_r, double cutoff_eps,
    int *d_group_indices,
    int *d_neigh_in_cutoff_r,
    int *d_group_numneigh,
    int *d_calculated_firstneigh_ptrs,
    int *d_calculated_numneigh,
    MetaD_zqc::ArrayView<const double> d_x_flat,
    double *d_neigh_in_switching,
    double *d_stein_qlm, double *d_stein_Ylm, double *d_stein_ql);
template __global__ void steinhardt_cv_kernel<6>(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int calc_count, double cutoff_r, double cutoff_eps,
    int *d_group_indices,
    int *d_neigh_in_cutoff_r,
    int *d_group_numneigh,
    int *d_calculated_firstneigh_ptrs,
    int *d_calculated_numneigh,
    MetaD_zqc::ArrayView<const double> d_x_flat,
    double *d_neigh_in_switching,
    double *d_stein_qlm, double *d_stein_Ylm, double *d_stein_ql);


template <int L>
__global__ void steinhardt_dcv_kernel(
    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int pair_all, int groupbit,
    MetaD_zqc::ArrayView<const int> d_mask,
    int *d_active_pair_ids,
    int *d_half_pair_i, int *d_half_pair_j,
    MetaD_zqc::ArrayView<const double> d_x_flat,
    double *d_neigh_in_switching,
    double *d_stein_ql, double *d_stein_qlm,
    double *d_dcvdx, double *d_a_virial,
    const double *atom_adjoint) {

    int pair_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_id >= pair_all) return;

    int pair_tag = d_active_pair_ids[pair_id];

    #define sw_f_r(r) (MetaD_zqc::SwitchFunction::f(sw_params_rij, (r)))
    #define sw_df_r(r) (MetaD_zqc::SwitchFunction::df(sw_params_rij, (r)))

    double pre_ij_ji = (L%2==0)? 1.0 : -1.0; // (-1)^L
    constexpr int lm_size = (L + 1) * 2 ;

    int c_atom_loctag = d_half_pair_i[pair_tag];
    int neigh_loctag = d_half_pair_j[pair_tag];

    double scale = (4 * PI) / ((2 * L + 1));

    double c_x = d_x_flat(c_atom_loctag,0);
    double c_y = d_x_flat(c_atom_loctag,1);
    double c_z = d_x_flat(c_atom_loctag,2);
    double neigh_x = d_x_flat(neigh_loctag,0);
    double neigh_y = d_x_flat(neigh_loctag,1);
    double neigh_z = d_x_flat(neigh_loctag,2);
    double dx = (neigh_x - c_x);
    double dy = (neigh_y - c_y);
    double dz = (neigh_z - c_z);
    double r2 = dx*dx + dy*dy + dz*dz;
    double r = sqrt(r2);
    double r_weight = sw_f_r(r);

    const double inv_r = 1.0 / r;
    const double nx = dx * inv_r, ny = dy * inv_r, nz = dz * inv_r;

    double catom_ql_timesN, neigh_ql_timesN;
    if ((d_mask(c_atom_loctag)&groupbit) == 0) {
        catom_ql_timesN = 0.0;
    } else {
        double qlc = d_stein_ql[c_atom_loctag];
        if (qlc < 1e-12) {
            catom_ql_timesN = 0.0;
        } else {
            const double value_adjoint = atom_adjoint[c_atom_loctag];
            catom_ql_timesN = scale / qlc / d_neigh_in_switching[c_atom_loctag] * value_adjoint;
        }
    }
    if ((d_mask(neigh_loctag)&groupbit) == 0) {
        neigh_ql_timesN = 0.0;
    } else {
        double qln = d_stein_ql[neigh_loctag];
        if (qln < 1e-12) {
            neigh_ql_timesN = 0.0;
        } else {
            const double value_adjoint = atom_adjoint[neigh_loctag];
            neigh_ql_timesN = scale / qln / d_neigh_in_switching[neigh_loctag] * value_adjoint;
        }
    }

    double local_Ylm[lm_size];
    double local_Ylmdr[3*2];
    int stein_qlm_base_id = c_atom_loctag * lm_size;
    int stein_qlm_neigh_id = neigh_loctag * lm_size;
    // Share the unit-direction polynomial recurrence between Ylm and dYlm/dR.
    compute_Ylm_unit_and_gradient<L>(
        nx, ny, nz, inv_r, catom_ql_timesN * r_weight, neigh_ql_timesN * r_weight,
        stein_qlm_base_id, stein_qlm_neigh_id, d_stein_qlm, local_Ylm, local_Ylmdr);

    double sum_of_sigma = 0;
    double a,b;
    a = catom_ql_timesN * sw_df_r(r);
    b = pre_ij_ji * neigh_ql_timesN * sw_df_r(r);
    double Aqlmi = d_stein_qlm[stein_qlm_base_id + 0 +0];
    double Bqlmi = d_stein_qlm[stein_qlm_base_id + 0 +1];
    double Aqlmj = d_stein_qlm[stein_qlm_neigh_id + 0 +0];
    double Bqlmj = d_stein_qlm[stein_qlm_neigh_id + 0 +1];
    double Aylmrij = local_Ylm[0 +0];
    double Bylmrij = local_Ylm[0 +1];
    sum_of_sigma += a * (Aqlmi * (Aylmrij - Aqlmi)+ Bqlmi * (Bylmrij - Bqlmi));
    sum_of_sigma += b * (Aqlmj * (Aylmrij - pre_ij_ji * Aqlmj)+ Bqlmj * (Bylmrij - pre_ij_ji * Bqlmj));
    #pragma unroll
    for (int i = 1; i <= L; i++) {
        Aqlmi = d_stein_qlm[stein_qlm_base_id + 2*i +0];
        Bqlmi = d_stein_qlm[stein_qlm_base_id + 2*i +1];
        Aqlmj = d_stein_qlm[stein_qlm_neigh_id + 2*i +0];
        Bqlmj = d_stein_qlm[stein_qlm_neigh_id + 2*i +1];
        Aylmrij = local_Ylm[2*i +0];
        Bylmrij = local_Ylm[2*i +1];
        sum_of_sigma += 2* a * (Aqlmi * (Aylmrij - Aqlmi)+ Bqlmi * (Bylmrij - Bqlmi));
        sum_of_sigma += 2* b * (Aqlmj * (Aylmrij - pre_ij_ji * Aqlmj)+ Bqlmj * (Bylmrij - pre_ij_ji * Bqlmj));
    }
    // f. is dcv/dx
    double f[3] = {0.0};
    f[0] = (local_Ylmdr[ 0*2 + 0] + local_Ylmdr[ 0*2 + 1] + sum_of_sigma*nx);
    f[1] = (local_Ylmdr[ 1*2 + 0] + local_Ylmdr[ 1*2 + 1] + sum_of_sigma*ny);
    f[2] = (local_Ylmdr[ 2*2 + 0] + local_Ylmdr[ 2*2 + 1] + sum_of_sigma*nz);
    // if (isnan(fx) || isnan(fy) || isnan(fz)) {
    //     // 只有崩成 NaN 的线程才会触发打印，不影响整体速度
    //     printf("[NaN Detected] c_atom_calctag = %d, neigh_tag = %d, Neigh_Nb=%d, d_stein_ql[neigh_tag]=%g, d_stein_qlm[stein_qlm_neigh_id + 0]=%g r = %f, dx = %f, dy = %f, dz = %f, sin_theta = %f\n",
    //             c_atom_calctag, neigh_tag, Neigh_Nb, d_stein_ql[neigh_tag],d_stein_qlm[stein_qlm_neigh_id + 0], r, dx, dy, dz, sin_theta);
    // }

    double tmpvirial[9] = {0.0};
    tmpvirial[ 0 ] = dx*f[0]; // vxx
    tmpvirial[ 1 ] = dy*f[1]; // vyy
    tmpvirial[ 2 ] = dz*f[2]; // vzz
    tmpvirial[ 3 ] = dx*f[1]; // vxy
    tmpvirial[ 4 ] = dx*f[2]; // vxz
    tmpvirial[ 5 ] = dy*f[2]; // vyz
    tmpvirial[6]=dy*f[0]; tmpvirial[7]=dz*f[0]; tmpvirial[8]=dz*f[1];
    #pragma unroll
    for (int i = 0; i < 9; i++) {
        atomicAdd(&d_a_virial[9*c_atom_loctag + i],-0.5*tmpvirial[i]);
        atomicAdd(&d_a_virial[9*neigh_loctag + i], -0.5*tmpvirial[i]);
        // d_a_virial[9*c_atom_loctag + i] += tmpvirial[i];
    }

    // 【完全通用】最后的总偏导汇总
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        atomicAdd(&d_dcvdx[c_atom_loctag * 3 + i], -f[i]);
        atomicAdd(&d_dcvdx[neigh_loctag * 3 + i], f[i]);
    }
}
template __global__ void steinhardt_dcv_kernel<3>(    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int pair_all, int groupbit,
    MetaD_zqc::ArrayView<const int> d_mask,
    int *d_active_pair_ids,
    int *d_half_pair_i, int *d_half_pair_j,
    MetaD_zqc::ArrayView<const double> d_x_flat,
    double *d_neigh_in_switching,
    double *d_stein_ql, double *d_stein_qlm,
    double *d_dcvdx, double *d_a_virial, const double *atom_adjoint);
template __global__ void steinhardt_dcv_kernel<4>(    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int pair_all, int groupbit,
    MetaD_zqc::ArrayView<const int> d_mask,
    int *d_active_pair_ids,
    int *d_half_pair_i, int *d_half_pair_j,
    MetaD_zqc::ArrayView<const double> d_x_flat,
    double *d_neigh_in_switching,
    double *d_stein_ql, double *d_stein_qlm,
    double *d_dcvdx, double *d_a_virial, const double *atom_adjoint);
template __global__ void steinhardt_dcv_kernel<6>(    MetaD_zqc::SwitchFunctionRequest sw_params_rij,
    int pair_all, int groupbit,
    MetaD_zqc::ArrayView<const int> d_mask,
    int *d_active_pair_ids,
    int *d_half_pair_i, int *d_half_pair_j,
    MetaD_zqc::ArrayView<const double> d_x_flat,
    double *d_neigh_in_switching,
    double *d_stein_ql, double *d_stein_qlm,
    double *d_dcvdx, double *d_a_virial, const double *atom_adjoint);


// __global__ void steinhardt_param_calc_LOCAL_kernel(int group_count, int cutoff_Natoms,
//                     int stein_l, int groupbit,
//                     MetaD_zqc::ArrayView<const int> d_mask, int *d_group_indices,
//                     int *d_calculated_numneigh,
//                     int *d_neigh_both_in_r_N,
//                     double *d_stein_qlm, double *d_stein_LQlm,
//                     double *d_stein_ql){
//     int c_atom = blockIdx.x * blockDim.x + threadIdx.x;
//     if(c_atom<group_count){
//         // steinhardt_param_calc_kernel_q4
//         // int stein_l=4;
//         int neigh_num, neigh_tag;
//         double temp4pi_2lplus1;
//         temp4pi_2lplus1 = 12.5663706143591729538505735331/(2*stein_l+1);
//         neigh_num = d_neigh_both_in_r_N[c_atom];
//         int base_LQlm_neigh_id,LQlm_neigh_id;
//         base_LQlm_neigh_id=c_atom*(stein_l + 1)*2;
//         d_stein_ql[c_atom] = 0;
//         if (neigh_num == 0) {
//             return;
//         }
//         for(int i=0; i<(stein_l + 1)*2; i++){
//             d_stein_LQlm[base_LQlm_neigh_id + i] = d_stein_qlm[c_atom + i];
//         }
//         for(int neigh_atom=0; neigh_atom<neigh_num; neigh_atom++){
//             neigh_tag = d_calculated_numneigh[c_atom*cutoff_Natoms + neigh_atom];
//             if (d_mask(neigh_tag)&groupbit){
//                 // TODO: 此处有问题，因为邻居原子可能不在cvgroup中，此时是找不到它的LQlm的
//                 // 使用二分查找法找 neigh_tag 对应在 d_stein_ql 中的位置
//                 int left = 0;
//                 int right = group_count - 1;
//                 // neigh_q4_deN default is 0
//                 while (left <= right) {
//                     int mid = left + (right - left) / 2;
//                     if (d_group_indices[mid] == neigh_tag) {
//                         LQlm_neigh_id = mid * (stein_l + 1) * 2;
//                         for(int i=0; i<(stein_l + 1)*2; i++){
//                             d_stein_LQlm[base_LQlm_neigh_id + i] += d_stein_qlm[LQlm_neigh_id + i];
//                         }
//                         break;
//                     } else if (d_group_indices[mid] < neigh_tag) {
//                         left = mid + 1;
//                     } else {
//                         right = mid - 1;
//                     }
//                 }
//             }
//         }
//         int i=0;
//         d_stein_LQlm[base_LQlm_neigh_id + i] = d_stein_LQlm[base_LQlm_neigh_id + i] / (neigh_num+1);
//         d_stein_ql[c_atom] += d_stein_LQlm[base_LQlm_neigh_id + i] * d_stein_LQlm[base_LQlm_neigh_id + i];
//         i=1;
//         d_stein_LQlm[base_LQlm_neigh_id + i] = d_stein_LQlm[base_LQlm_neigh_id + i] / (neigh_num+1);
//         d_stein_ql[c_atom] += d_stein_LQlm[base_LQlm_neigh_id + i] * d_stein_LQlm[base_LQlm_neigh_id + i];
//         for(int i=2; i<(stein_l + 1)*2; i++){
//             d_stein_LQlm[base_LQlm_neigh_id + i] = d_stein_LQlm[base_LQlm_neigh_id + i] / (neigh_num+1);
//             d_stein_ql[c_atom] += 2*d_stein_LQlm[base_LQlm_neigh_id + i] * d_stein_LQlm[base_LQlm_neigh_id + i];
//         }
//         d_stein_ql[c_atom] = sqrt(d_stein_ql[c_atom]*temp4pi_2lplus1);
//     }
// }


// __global__ void dcv_steinhardt_param_calc_LOCAL_kernel(int group_count, int cutoff_Natoms,
//                     int stein_l, int groupbit,
//                     MetaD_zqc::ArrayView<const int> d_mask, int *d_group_indices,
//                     int *d_calculated_numneigh,
//                     int *d_neigh_both_in_r_N,
//                     double *d_stein_qlm, double *d_stein_LQlm,
//                     double *d_stein_ql){
//     int c_atom = blockIdx.x * blockDim.x + threadIdx.x;
//     if(c_atom<group_count){
//         // steinhardt_param_calc_kernel_q4
//         // int stein_l=4;
//         int neigh_num, neigh_tag;
//         double temp4pi_2lplus1;
//         temp4pi_2lplus1 = 12.5663706143591729538505735331/(2*stein_l+1);
//         neigh_num = d_neigh_both_in_r_N[c_atom];
//         int base_LQlm_neigh_id,LQlm_neigh_id;
//         base_LQlm_neigh_id=c_atom*(stein_l + 1)*2;
//         d_stein_ql[c_atom] = 0;
//         if (neigh_num == 0) {
//             return;
//         }
//         for(int i=0; i<(stein_l + 1)*2; i++){
//             d_stein_LQlm[base_LQlm_neigh_id + i] = d_stein_qlm[c_atom + i];
//         }
//         for(int neigh_atom=0; neigh_atom<neigh_num; neigh_atom++){
//             neigh_tag = d_calculated_numneigh[c_atom*cutoff_Natoms + neigh_atom];
//             if (d_mask(neigh_tag)&groupbit){
//                 // TODO: 此处有问题，因为邻居原子可能不在cvgroup中，此时是找不到它的LQlm的
//                 // 使用二分查找法找 neigh_tag 对应在 d_stein_ql 中的位置
//                 int left = 0;
//                 int right = group_count - 1;
//                 // neigh_q4_deN default is 0
//                 while (left <= right) {
//                     int mid = left + (right - left) / 2;
//                     if (d_group_indices[mid] == neigh_tag) {
//                         LQlm_neigh_id = mid * (stein_l + 1) * 2;
//                         for(int i=0; i<(stein_l + 1)*2; i++){
//                             d_stein_LQlm[base_LQlm_neigh_id + i] += d_stein_qlm[LQlm_neigh_id + i];
//                         }
//                         break;
//                     } else if (d_group_indices[mid] < neigh_tag) {
//                         left = mid + 1;
//                     } else {
//                         right = mid - 1;
//                     }
//                 }
//             }
//         }
//         int i=0;
//         d_stein_LQlm[base_LQlm_neigh_id + i] = d_stein_LQlm[base_LQlm_neigh_id + i] / (neigh_num+1);
//         d_stein_ql[c_atom] += d_stein_LQlm[base_LQlm_neigh_id + i] * d_stein_LQlm[base_LQlm_neigh_id + i];
//         i=1;
//         d_stein_LQlm[base_LQlm_neigh_id + i] = d_stein_LQlm[base_LQlm_neigh_id + i] / (neigh_num+1);
//         d_stein_ql[c_atom] += d_stein_LQlm[base_LQlm_neigh_id + i] * d_stein_LQlm[base_LQlm_neigh_id + i];
//         for(int i=2; i<(stein_l + 1)*2; i++){
//             d_stein_LQlm[base_LQlm_neigh_id + i] = d_stein_LQlm[base_LQlm_neigh_id + i] / (neigh_num+1);
//             d_stein_ql[c_atom] += 2*d_stein_LQlm[base_LQlm_neigh_id + i] * d_stein_LQlm[base_LQlm_neigh_id + i];
//         }
//         d_stein_ql[c_atom] = sqrt(d_stein_ql[c_atom]*temp4pi_2lplus1);
//     }
// }


REGISTER_CV("STEINH", MetaD_zqc::Steinhardt::create);
template class MetaD_zqc::STEIN_QL<3>;
template class MetaD_zqc::STEIN_QL<4>;
template class MetaD_zqc::STEIN_QL<6>;
