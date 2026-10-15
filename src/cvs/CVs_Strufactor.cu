#include "CV_Stru_factor.h"
#include "CV_Weighted_Chemical_Pair_Order.h"
#include <cmath>
#include <stdexcept>

// 独立 kernel 测试通过同一源文件编译设备数据与前后向计算。
#ifndef ZQC_STRUFACTOR_KERNELS_ONLY
#include "lammpsplugin.h"

#include "lammps.h"
#include "update.h"
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "command.h"
#include "domain.h"
#include "force.h"
#include "group.h"
#include "version.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"         // 完整定义Neighbor类
#include "neigh_list.h"        // 定义NeighList结构
#include "pair.h"

#include "fix_crystallize.h"
#include "zqc_debug.h"
#include "zqc_cv_assembly.cuh"
#include "zqc_switch_function.h"


#include <cuda_runtime.h>
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <sstream>

using namespace LAMMPS_NS;

namespace {
void check_strufactor_cuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
}

std::map<std::string, MetaD_zqc::Stru_fact_env*> MetaD_zqc::Stru_fact_env::env_pool;

MetaD_zqc::CV* MetaD_zqc::Stru_factor::create(LAMMPS_NS::LAMMPS *lmp, 
                                          LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check, 
                                            int narg, char **arg, int &i){
    DEBUG_LOG("In STRU_FACTOR settings");
    LAMMPS_NS::Error *error = lmp->error;

    std::string cal_name = arg[i];

    MetaD_zqc::StruFactorRequest req{};
    req.q_factor = 1.0;

    req.cal_name = cal_name;
    // 原子环境分析-初始设置
    // Usage: STRU_FACTOR <group>
    ERR_COND(i + 1 >= narg, "Error: STRU_FACTOR command requires at least \"STRU_FACTOR <group> \".");
    req.group_name = arg[i+1];
    req.group_id = lmp->group->find(req.group_name);
    ERR_COND(req.group_id == -1, "Error: group name %s not found.", req.group_name);
    //参数有效性
    // default values
    req.cutoff_r = 8.0;
    req.d_block_size = 128;
    req.custom_weights.clear();
    int iarg=2 + i;
    while (iarg < narg) {
        if (strcmp(arg[iarg], "cutoff_r") == 0) {
            ERR_COND((iarg + 1 >= narg) ,"Error: \'cutoff_r\' keyword requires a value");
            req.cutoff_r = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
            iarg += 2;
        } else if (strcmp(arg[iarg], "q_factor") == 0) {
            ERR_COND((iarg + 1 >= narg), "Error: \'q_factor\' keyword requires a number");
            req.q_factor = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
            ERR_COND(req.q_factor <= 0, "Error: \'q_factor\' must be > 0");
            iarg += 2;
        } else if (strcmp(arg[iarg], "d_block_size") == 0) {
            ERR_COND((iarg + 1 >= narg), "Error: \'d_block_size\' keyword requires an integer");
            req.d_block_size = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
            ERR_COND(req.d_block_size <= 0, "Error: \'d_block_size\' must be > 0");
            iarg += 2;
        } else if (strcmp(arg[iarg], "SW_func") == 0) {
            error->all(FLERR, "STRU_FACTOR: express switching functions in Graph using the RAW input");
        } else if (strcmp(arg[iarg], "Chemical") == 0) {
            req.use_chemical_lock = true;
            iarg += 1;
        } else if (strcmp(arg[iarg], "Chem_ctarget") == 0) {
            ERR_COND((iarg + 1 >= narg), "Error: \'Chem_ctarget\' keyword requires a number");
            req.c_target = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
            iarg += 2;
        } else if (strcmp(arg[iarg], "Chem_sigma") == 0) {
            ERR_COND((iarg + 1 >= narg), "Error: \'sigma\' keyword requires a number");
            req.sigma = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
            iarg += 2;
        } else if (strcmp(arg[iarg], "Chem_map") == 0) {
            iarg += 1; // 跳过 "Chem_map" 关键字
            req.use_custom_weight = true;
            // 循环读取后面所有的 (type weight) 括号对
            while (iarg < narg && arg[iarg][0] == '(') {
                std::string pair_str = arg[iarg];
                
                // 健壮性处理：如果长字符串包含了空格分裂（某些系统传参导致的分离）
                // 或者形如 "(1" "5.0)" 的情况，为了安全，我们要求输入必须是紧凑的 "(1,1.0)" 或 "(1 1.0)"
                // 并在内部去除左右括号
                if (pair_str.front() == '(' && pair_str.back() == ')') {
                    pair_str = pair_str.substr(1, pair_str.size() - 2); // 掐头去尾
                    
                    // 兼容空格或逗号分隔
                    size_t sep = pair_str.find_first_of(" ,"); 
                    if (sep != std::string::npos) {
                        std::string type_s = pair_str.substr(0, sep);
                        std::string weight_s = pair_str.substr(sep + 1);
                        
                        // 去除可能残余的空格
                        int a_type = std::stoi(type_s);
                        double a_weight = std::stod(weight_s);
                        
                        req.custom_weights[a_type] = a_weight;
                    }
                } else {
                    // 如果 LAMMPS 把 (1 1.0) 拆成了两个 token: "(1" 和 "1.0)"
                    // 我们通过向前探测来组合它们
                    std::string part1 = arg[iarg];
                    ERR_COND(iarg + 1 >= narg, "Error: Invalid Chem_map format near %s", arg[iarg]);
                    std::string part2 = arg[iarg+1];
                    
                    if (part1.front() == '(' && part2.back() == ')') {
                        int a_type = std::stoi(part1.substr(1));
                        double a_weight = std::stod(part2.substr(0, part2.size() - 1));
                        req.custom_weights[a_type] = a_weight;
                        iarg += 1; // 额外消费一个参数
                    } else {
                        error->all(FLERR, "Error: Chem_map pairs must be formatted as (type weight)");
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

    // 按 CV 半径请求完整邻居表，并使 ghost 坐标覆盖 cutoff 与 skin。
    MetaD_zqc::NeighSpec nspec;
    nspec.full = 1;
    nspec.ghost = 0;
    nspec.use_cutoff = 1;
    nspec.cutoff = req.cutoff_r;
    lmp->comm->cutghostuser = std::max(lmp->comm->cutghostuser, req.cutoff_r+lmp->neighbor->skin);
    const int neigh_id = Fixmetad->neigh_hub.get_or_create(nspec);

    MetaD_zqc::Stru_factor* struc_factor = nullptr;
    if (req.use_chemical_lock) {
        LOG("Logging: set STRU_FACTOR as group_name=%s q_factor=%g cutoff_r=%f d_block_size=%d.\n         Chemical lock is ON, with c_target=%g and sigma=%g.",
            req.group_name, req.q_factor, req.cutoff_r, req.d_block_size,
            req.c_target, req.sigma);
        MetaD_zqc::Stru_fact_chem_env *temp_env = static_cast<MetaD_zqc::Stru_fact_chem_env*>(
                                    MetaD_zqc::Stru_fact_env::get_or_create(lmp, Fixmetad, f_check, req)
                                );
        temp_env->neigh_id = neigh_id;
        DEBUG_LOG("Stru_fact_chem_env is %p neigh_id=%d", temp_env, neigh_id);
        i = iarg;
        struc_factor = new MetaD_zqc::Stru_factor_chem(lmp, Fixmetad, f_check, 
                                temp_env,
                                req.q_factor, req.d_block_size);
    } else {
        LOG("Logging: set STRU_FACTOR as group_name=%s q_factor=%g cutoff_r=%f d_block_size=%d.",
                            req.group_name, req.q_factor, req.cutoff_r, req.d_block_size);
        MetaD_zqc::Stru_fact_env *temp_env = MetaD_zqc::Stru_fact_env::get_or_create(lmp, 
                                    Fixmetad, f_check, req);
        temp_env->neigh_id = neigh_id;
        DEBUG_LOG("Stru_fact_env is %p neigh_id=%d", temp_env, neigh_id);
        struc_factor =  new MetaD_zqc::Stru_factor(lmp, Fixmetad, f_check, 
                                temp_env,
                                req.q_factor, req.d_block_size);
    }


    return struc_factor;
}

MetaD_zqc::Stru_fact_env* MetaD_zqc::Stru_fact_env::get_or_create(LAMMPS_NS::LAMMPS *lmp,
                                            LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check, 
                                            StruFactorRequest req) {
    // 使用完整浮点精度和有序 Chem_map，保证不同权重配置各自持有正确的类型表。
    std::ostringstream key_stream;
    key_stream << std::setprecision(std::numeric_limits<double>::max_digits10)
               << req.group_id << '_' << req.cutoff_r << '_' << req.use_chemical_lock;
    if (req.use_chemical_lock) {
        if (!std::isfinite(req.c_target) || !std::isfinite(req.sigma) || req.sigma <= 0)
            throw std::invalid_argument("STRU_FACTOR Chemical: Chem_ctarget must be finite and Chem_sigma positive and finite");
        key_stream << '_' << req.c_target << '_' << req.sigma;
        for (const auto& entry : req.custom_weights) {
            if (entry.first < 1 || entry.first > lmp->atom->ntypes || !std::isfinite(entry.second))
                throw std::invalid_argument("STRU_FACTOR Chemical: Chem_map requires valid atom types and finite weights");
            key_stream << '_' << entry.first << ':' << entry.second;
        }
    }
    const std::string key = key_stream.str();
    const auto found = env_pool.find(key);
    if (found != env_pool.end()) return found->second;
    Stru_fact_env* environment;
    if (req.use_chemical_lock)
        environment = new Stru_fact_chem_env(lmp, Fixmetad, f_check, req.group_id,
                                             req.cutoff_r, req.c_target, req.sigma, req.custom_weights);
    else
        environment = new Stru_fact_env(lmp, Fixmetad, f_check, req.group_id, req.cutoff_r);
    env_pool.emplace(key, environment);
    return environment;
}

std::string MetaD_zqc::Stru_fact_env::get_env_key(){
    std::string key = std::to_string(this->group_id) + "_" + std::to_string(this->cutoff_r);
    return key;
}

void MetaD_zqc::Stru_fact_env::clear_pool() {
    for (auto& pair : env_pool) delete pair.second;
    env_pool.clear();
}

MetaD_zqc::Stru_fact_env::Stru_fact_env(LAMMPS* lmp, FixMetadynamics* Fixmetad,
                                                    FILE* f_check, int group_id, double cutoff_r)
    : CV_info(lmp, Fixmetad, f_check), cutoff_r(cutoff_r), group_id(group_id),
      groupbit(lmp->group->bitmask[group_id]) {
    data.set_context(f_check, lmp->error, lmp);
    register_buffer(d_x_flat, "d_x_flat");
    register_buffer(d_firstneigh_ptrs, "d_firstneigh_ptrs");
}

MetaD_zqc::Stru_factor::Stru_factor(LAMMPS* lmp, FixMetadynamics* Fixmetad, FILE* f_check,
                                  Stru_fact_env* env,
                                  double q, int block_size)
    : CV(lmp, Fixmetad, f_check), my_env(env), d_block_size(block_size), q_factor(q) {
    my_env->d_block_size = block_size;
    check_strufactor_cuda(cudaGetDevice(&my_env->GPU_number));
    register_buffer(d_stru_factor, "d_stru_factor");
    register_buffer(d_center_adjoint, "sf_center_adjoint");
    register_buffer(d_graph_force, "sf_graph_force");
    register_buffer(d_graph_virial, "sf_graph_virial");
}

MetaD_zqc::Stru_fact_env::~Stru_fact_env() {
    lmp->memory->destroy(h_group_numneigh);
    lmp->memory->destroy(h_x_flat);
    lmp->memory->destroy(h_group_indices);
    lmp->memory->destroy(h_firstneigh_ptrs);
}

MetaD_zqc::Stru_factor::~Stru_factor() {
    lmp->memory->destroy(h_stru_factor);
}

// 更新“压缩中心序号 -> owned localid”映射；中心顺序与 CVAssembly 的前缀保持一致。
void MetaD_zqc::Stru_fact_env::refresh_lmpbox() {
    atom = lmp->atom;
    lmp->memory->grow(h_group_indices, atom->nlocal, "Stru_fact:h_group_indices");
    last_group_count = data.group_count;
    data.group_count = 0;
    if (atom_assembly) {
        if (atom_assembly->nlocal() != atom->nlocal || atom_assembly->nghost() != atom->nghost)
            throw std::logic_error("Stru_factor: stale atom assembly");
        // 装配表前 group_count 项是本 rank 的 owned 中心。
        data.group_count = atom_assembly->group_count();
        if (data.group_count)
            std::copy_n(atom_assembly->local_ids().begin(), data.group_count, h_group_indices);
    } else {
        for (int i = 0; i < atom->nlocal; ++i)
            if (atom->mask[i] & groupbit) h_group_indices[data.group_count++] = i;
    }
    block_num = (data.group_count + d_block_size - 1) / d_block_size;
    N = static_cast<size_t>(d_block_size) * block_num;
}

// 准备共享几何：候选邻居按中心分段上传，再在 GPU 将有效邻居压到各段开头。
// 分段起点保留候选容量，遍历有效数据时使用 d_neigh_in_cutoff_r 给出的数量。
void MetaD_zqc::Stru_fact_env::get_env() {
    atom = lmp->atom;
    if (neigh_id < 1) throw std::logic_error("STRU_FACTOR: invalid neighbor request");
    Fixmetad->neigh_hub.ensure(neigh_id);
    nlist = Fixmetad->neigh_hub.list(neigh_id);
    if (!nlist) throw std::runtime_error("STRU_FACTOR: neighbor list is unavailable");
    if (data.group_count == 0) {
        last_update_step = lmp->update->ntimestep;
        return;
    }

    // 按压缩中心顺序展平 LAMMPS 候选邻居，偏移末项记录总槽位数。
    data.grow_centers(__FILE__, __LINE__);
    check_strufactor_cuda(cudaMemcpy(data.d_group_indices.ptr, h_group_indices,
                                     data.group_count*sizeof(*h_group_indices), cudaMemcpyHostToDevice));
    lmp->memory->grow(h_group_numneigh, data.group_count+1, "STRU_FACTOR:h_group_numneigh");
    h_group_numneigh[0] = 0;
    for (int k = 0; k < data.group_count; ++k)
        h_group_numneigh[k+1] = h_group_numneigh[k] + nlist->numneigh[h_group_indices[k]];
    const size_t pairs = h_group_numneigh[data.group_count];
    check_strufactor_cuda(cudaMemcpy(data.d_group_numneigh.ptr, h_group_numneigh,
                                     (data.group_count+1)*sizeof(*h_group_numneigh), cudaMemcpyHostToDevice));
    lmp->memory->grow(h_firstneigh_ptrs, pairs, "STRU_FACTOR:h_firstneigh_ptrs");
    d_firstneigh_ptrs.grow_to(pairs, __FILE__, __LINE__);
    for (int k = 0; k < data.group_count; ++k) {
        const int count = nlist->numneigh[h_group_indices[k]];
        if (count) std::copy_n(nlist->firstneigh[h_group_indices[k]], count,
                               h_firstneigh_ptrs+h_group_numneigh[k]);
    }
    if (pairs) check_strufactor_cuda(cudaMemcpy(d_firstneigh_ptrs.ptr, h_firstneigh_ptrs,
                                                pairs*sizeof(int), cudaMemcpyHostToDevice));

    // 邻居 localid 直接索引 owned+ghost 坐标；周期镜像坐标由 LAMMPS 提供。
    const size_t nall = static_cast<size_t>(atom->nlocal) + atom->nghost;
    lmp->memory->grow(h_x_flat, 3*nall, "STRU_FACTOR:h_x_flat");
    for (size_t i = 0; i < nall; ++i) std::copy_n(atom->x[i], 3, h_x_flat+3*i);
    d_x_flat.grow_to(3*nall, __FILE__, __LINE__);
    check_strufactor_cuda(cudaMemcpy(d_x_flat.ptr, h_x_flat, 3*nall*sizeof(double), cudaMemcpyHostToDevice));
    data.grow_neighbors(pairs, data.group_count, __FILE__, __LINE__);
    get_environment_Strufactor<<<block_num,d_block_size>>>(cutoff_r*cutoff_r,
        data.group_count, data.d_group_indices.ptr, data.d_group_numneigh.ptr,
        d_firstneigh_ptrs.ptr, d_x_flat.ptr, data.d_group_dminneigh.ptr,
        data.d_neigh_in_cutoff_r.ptr, data.d_calculated_numneigh.ptr);
    check_strufactor_cuda(cudaGetLastError());
    // 后续前向、反向在默认流中复用本轮筛选后的几何数据。
    last_update_step = lmp->update->ntimestep;
}

void MetaD_zqc::Stru_factor::environment() {
    if (my_env->last_update_step != lmp->update->ntimestep) {
        my_env->refresh_lmpbox();
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
    if (assembly && (assembly->groupbit() != my_env->groupbit ||
                     assembly->nlocal() != lmp->atom->nlocal ||
                     assembly->nghost() != lmp->atom->nghost))
        throw std::invalid_argument("Stru_factor: incompatible atom assembly");
    if (my_env->atom_assembly != assembly) {
        my_env->atom_assembly = assembly;
        my_env->last_update_step = -1;
    }
    raw_values_ready = false;
    graph_evaluation_prepared = false;
    CV::bind_assembly(std::move(assembly));
}

void MetaD_zqc::Stru_factor::check_graph_values() const {
    if (!assembly_)
        throw std::logic_error("Stru_factor: bind an atom assembly before using the graph interface");
    if (!raw_values_ready || raw_value_step != lmp->update->ntimestep)
        throw std::logic_error("Stru_factor: base_calc is required before assembly/backward");
    if (assembly_->nlocal() != lmp->atom->nlocal || assembly_->nghost() != lmp->atom->nghost ||
        assembly_->group_count() != my_env->data.group_count)
        throw std::logic_error("Stru_factor: owned/ghost layout changed; rebuild and bind it first");
}

// 从 localid 排列的原始值提取本组 owned 中心，装配到公共 Leaf 的 leaf_slots。
void MetaD_zqc::Stru_factor::assemble_values(GpuBuffer<double>& values) {
    check_graph_values();
    assembly_->assemble_values(d_stru_factor, values);
}

void MetaD_zqc::Stru_factor::prepare_graph_evaluation() {
    // 每轮统一使共享环境失效，由首个使用者更新；支持同一步重复求值。
    raw_values_ready = false;
    my_env->last_update_step = -1;
    graph_evaluation_prepared = true;
}

void MetaD_zqc::Stru_factor::base_calc(){
    raw_values_ready = false;
    if (assembly_) {
        // 独立调用时刷新环境；Graph 调用使用 prepare 阶段的失效标记。
        if (!graph_evaluation_prepared) my_env->last_update_step = -1;
    }
    graph_evaluation_prepared = false;
    compute_stru_factor_peratoms();
    raw_value_step = lmp->update->ntimestep;
    raw_values_ready = true;
}

void MetaD_zqc::Stru_factor::apply_adjoint(const GpuBuffer<double>& leaf_adjoint) {
    check_graph_values();
    const size_t nall = static_cast<size_t>(lmp->atom->nlocal) + lmp->atom->nghost;
    // Leaf 的公共排列可覆盖多个原子组，反向先提取本组紧凑的 dV/dS_i 数组。
    d_center_adjoint.grow_to(my_env->data.group_count, __FILE__, __LINE__);
    assembly_->gather_adjoint(leaf_adjoint, d_center_adjoint);
    d_graph_force.grow_to(3*nall, __FILE__, __LINE__);
    d_graph_virial.grow_to(6*nall, __FILE__, __LINE__);
    h_graph_force.resize(3*nall);
    h_graph_virial.resize(6*nall);
    if (nall) {
        check_strufactor_cuda(cudaMemset(d_graph_force.ptr, 0, 3*nall*sizeof(double)));
        check_strufactor_cuda(cudaMemset(d_graph_virial.ptr, 0, 6*nall*sizeof(double)));
    }
    call_structure_factor_adjoint_kernel(d_center_adjoint.ptr, d_graph_force.ptr, d_graph_virial.ptr);
    if (nall) {
        check_strufactor_cuda(cudaMemcpy(h_graph_force.data(), d_graph_force.ptr,
                                        3*nall*sizeof(double), cudaMemcpyDeviceToHost));
        check_strufactor_cuda(cudaMemcpy(h_graph_virial.data(), d_graph_virial.ptr,
                                        6*nall*sizeof(double), cudaMemcpyDeviceToHost));
    }
    // ghost 的力/virial 回到 owner；空中心组也参加这轮通信。
    Fixmetad->communicate_reverse(*this);
    for (int atom = 0; atom < lmp->atom->nlocal; ++atom) {
        for (int c = 0; c < 3; ++c) lmp->atom->f[atom][c] += h_graph_force[3*atom+c];
        if (Fixmetad->a_virial)
            for (int c = 0; c < 6; ++c) Fixmetad->a_virial[6*atom+c] += h_graph_virial[6*atom+c];
    }
}

// 每个 ghost 打包 3 分量力和 6 分量 virial；offset、stride 的单位是 double。
int MetaD_zqc::Stru_factor::pack_comm_reverse_ubuf(int n, int first, double* buf, int offset, int stride) {
    for (int i = 0; i < n; ++i) {
        for (int c = 0; c < 3; ++c)
            buf[offset+i*stride+c] = h_graph_force[3*(first+i)+c];
        for (int c = 0; c < 6; ++c)
            buf[offset+i*stride+3+c] = h_graph_virial[6*(first+i)+c];
    }
    return 9;
}

// list 给出本 rank 的 owner localid，将接收贡献累加到 Host 力/virial 数组。
void MetaD_zqc::Stru_factor::unpack_comm_reverse_ubuf(int n, int* list, double* buf, int offset, int stride) {
    for (int i = 0; i < n; ++i) {
        for (int c = 0; c < 3; ++c) h_graph_force[3*list[i]+c] += buf[offset+i*stride+c];
        for (int c = 0; c < 6; ++c) h_graph_virial[6*list[i]+c] += buf[offset+i*stride+3+c];
    }
}

// 原始值按 owned+ghost 的 localid 分配：清零后计算 owned 中心，下载 Host 副本供逐原子输出。
void MetaD_zqc::Stru_factor::compute_stru_factor_peratoms() {
    environment();
    const size_t nall = static_cast<size_t>(lmp->atom->nlocal) + lmp->atom->nghost;
    lmp->memory->grow(h_stru_factor, nall, "Stru_factor:h_stru_factor");
    d_stru_factor.grow_to(nall, __FILE__, __LINE__);
    if (nall) check_strufactor_cuda(cudaMemset(d_stru_factor.ptr, 0, nall*sizeof(double)));
    if (my_env->data.group_count) call_structure_factor_cv_kernel();
    if (nall) check_strufactor_cuda(cudaMemcpy(h_stru_factor, d_stru_factor.ptr,
                                              nall*sizeof(double), cudaMemcpyDeviceToHost));
}

void MetaD_zqc::Stru_factor::call_structure_factor_cv_kernel() {
    strufactor_forward(my_env->data, q_factor, my_env->cutoff_r*my_env->cutoff_r,
                       d_stru_factor.ptr, 0, d_block_size);
}

void MetaD_zqc::Stru_factor::call_structure_factor_adjoint_kernel(
    const double* center_adjoint, double* atom_forces, double* atom_virials) {
    strufactor_backward(my_env->data, q_factor, my_env->cutoff_r*my_env->cutoff_r,
                        center_adjoint, atom_forces, atom_virials, 0, d_block_size);
}

void MetaD_zqc::Stru_factor::summary(FILE*) {}

double* MetaD_zqc::Stru_factor::get_peratom_ptr(const std::string& name) {
    return name == "stru_f" ? h_stru_factor : nullptr;
}

// 一线程处理一个中心，将 cutoff 内的邻居紧排到自己的候选段开头。
// candidates 按 offsets 分段，coordinates 按 [3*localid+c] 排列；LAMMPS 已提供周期镜像坐标。
// 输出 neighbors[slot] 和 geometry[4*slot+c] 共享槽位，counts[k] 记录每段有效前缀长度。
__global__ void get_environment_Strufactor(double cutoff_rsq, int group_count,
    int* centers, LAMMPS_NS::tagint* offsets, int* candidates, double* coordinates,
    double* geometry, int* counts, LAMMPS_NS::tagint* neighbors) {
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= group_count) return;
    const int center = centers[k];
    const int first = offsets[k], last = offsets[k+1];
    int count = 0;
    for (int p = first; p < last; ++p) {
        const int neighbor = candidates[p] & NEIGHMASK;
        const double dx = coordinates[3*neighbor] - coordinates[3*center];
        const double dy = coordinates[3*neighbor+1] - coordinates[3*center+1];
        const double dz = coordinates[3*neighbor+2] - coordinates[3*center+2];
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

REGISTER_CV("STRU_FACTOR", MetaD_zqc::Stru_factor::create);
#endif // ZQC_STRUFACTOR_KERNELS_ONLY

namespace MetaD_zqc {
namespace {

// 验证中心数和几何数组申请长度，防止负计数或容量乘法溢出。
int checked_center_count(int count) {
    if (count < 0) throw std::invalid_argument("Stru_factor: negative center count");
    return count;
}
size_t geometry_size(size_t pair_count) {
    if (pair_count > std::numeric_limits<size_t>::max()/4)
        throw std::length_error("Stru_factor: neighbor geometry size overflow");
    return 4*pair_count;
}

// value = phi(r)；derivative_over_r = (d phi / dr) / r。
// 后者直接乘位移向量即可得到坐标导数；r=0 处使用解析极限。
struct RadialValue { double value, derivative_over_r; };

// 先定义参数/返回类型，再声明后半部分的设备函数。
__device__ __forceinline__ RadialValue radial(double r2, double q, double cutoff_rsq);
template<bool UseChemicalWeights, bool UseStructureFactor = true>
__global__ void forward_kernel(StruFactorKernelView view, double q, double cutoff_rsq,
                               const int* atom_types, const double* type_weights, double* values);
template<bool UseChemicalWeights, bool UseStructureFactor = true>
__global__ void backward_kernel(StruFactorKernelView view, double q, double cutoff_rsq,
                                const int* atom_types, const double* type_weights,
                                const double* adjoint, double* force, double* virial);

// 检查主机可见的启动参数和中心指针。
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
// 报告 kernel 提交时的错误。
void check_launch() {
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
} // namespace

// 使用 GpuBuffer 的定长构造入口。
StruFactorData::StruFactorData(int center_count, size_t pair_count)
    : group_count(checked_center_count(center_count)),
      d_group_indices(group_count), d_group_numneigh(static_cast<size_t>(group_count)+1),
      d_neigh_in_cutoff_r(group_count), d_calculated_numneigh(pair_count),
      d_group_dminneigh(geometry_size(pair_count)) {
    set_context(nullptr, nullptr, nullptr);
}

// 所有共享数组在同一处绑定元数据。
void StruFactorData::set_context(FILE* f_check, LAMMPS_NS::Error* error, LAMMPS_NS::LAMMPS* lmp) {
    d_group_indices.set_name("d_group_indices", f_check, error, lmp);
    d_group_numneigh.set_name("d_group_numneigh", f_check, error, lmp);
    d_neigh_in_cutoff_r.set_name("d_neigh_in_cutoff_r", f_check, error, lmp);
    d_calculated_numneigh.set_name("d_calculated_numneigh", f_check, error, lmp);
    d_group_dminneigh.set_name("d_group_dminneigh", f_check, error, lmp);
}

// 主机启动入口：一个线程对应一个压缩中心 k；空中心组直接返回。
// values 按 localid 输出，初始化由调用方完成；计算按指定 stream 异步提交。
void strufactor_forward(const StruFactorData& data, double q, double cutoff_rsq,
                        double* values, cudaStream_t stream, int block_size) {
    const auto view = kernel_view(data);
    check_wave_number(q);
    check_arguments(view,cutoff_rsq,block_size);
    if (!view.group_count) return;
    if (!values) throw std::invalid_argument("Stru_factor: missing value buffer");
    forward_kernel<false><<<(view.group_count-1)/block_size+1,block_size,0,stream>>>(
        view,q,cutoff_rsq,nullptr,nullptr,values);
    check_launch();
}
// center_adjoint 按压缩中心顺序排列，force/virial 已由调用方初始化。
// kernel 累加 owned 和 ghost 的输出，CV 层在下载之后执行 reverse communication。
void strufactor_backward(const StruFactorData& data, double q, double cutoff_rsq,
                         const double* center_adjoint, double* force, double* virial,
                         cudaStream_t stream, int block_size) {
    const auto view = kernel_view(data);
    check_wave_number(q);
    check_arguments(view,cutoff_rsq,block_size);
    if (!view.group_count) return;
    if (!center_adjoint || !force)
        throw std::invalid_argument("Stru_factor: missing adjoint/force buffer");
    backward_kernel<false><<<(view.group_count-1)/block_size+1,block_size,0,stream>>>(
        view,q,cutoff_rsq,nullptr,nullptr,center_adjoint,force,virial);
    check_launch();
}


// 化学版选择编译期特化，在每个邻居项上乘固定的元素对权重。
void strufactor_chem_forward(const StruFactorData& data, double q, double cutoff_rsq,
                             const int* atom_types, const double* type_weights,
                             double* values, cudaStream_t stream, int block_size) {
    const auto view = kernel_view(data);
    check_wave_number(q);
    check_arguments(view,cutoff_rsq,block_size);
    if (!view.group_count) return;
    if (!atom_types || !type_weights || !values)
        throw std::invalid_argument("Stru_factor_chem: missing type/weight/value buffer");
    forward_kernel<true><<<(view.group_count-1)/block_size+1,block_size,0,stream>>>(
        view,q,cutoff_rsq,atom_types,type_weights,values);
    check_launch();
}

void strufactor_chem_backward(const StruFactorData& data, double q, double cutoff_rsq,
                              const int* atom_types, const double* type_weights,
                              const double* center_adjoint, double* force, double* virial,
                              cudaStream_t stream, int block_size) {
    const auto view = kernel_view(data);
    check_wave_number(q);
    check_arguments(view,cutoff_rsq,block_size);
    if (!view.group_count) return;
    if (!atom_types || !type_weights || !center_adjoint || !force)
        throw std::invalid_argument("Stru_factor_chem: missing type/weight/adjoint/force buffer");
    backward_kernel<true><<<(view.group_count-1)/block_size+1,block_size,0,stream>>>(
        view,q,cutoff_rsq,atom_types,type_weights,center_adjoint,force,virial);
    check_launch();
}

// WEIGHT_CHEM 复用两端受力及 virial 的遍历，只选择径向开关 s(r)。
void weighted_chem_forward(const StruFactorData& data, double cutoff_rsq,
                           const int* atom_types, const double* type_weights,
                           double* values, cudaStream_t stream, int block_size) {
    const auto view = kernel_view(data);
    check_arguments(view,cutoff_rsq,block_size);
    if (!view.group_count) return;
    if (!atom_types || !type_weights || !values)
        throw std::invalid_argument("WEIGHT_CHEM: missing type/weight/value buffer");
    forward_kernel<true,false><<<(view.group_count-1)/block_size+1,block_size,0,stream>>>(
        view,0.0,cutoff_rsq,atom_types,type_weights,values);
    check_launch();
}
void weighted_chem_backward(const StruFactorData& data, double cutoff_rsq,
                            const int* atom_types, const double* type_weights,
                            const double* center_adjoint, double* force, double* virial,
                            cudaStream_t stream, int block_size) {
    const auto view = kernel_view(data);
    check_arguments(view,cutoff_rsq,block_size);
    if (!view.group_count) return;
    if (!atom_types || !type_weights || !center_adjoint || !force)
        throw std::invalid_argument("WEIGHT_CHEM: missing type/weight/adjoint/force buffer");
    backward_kernel<true,false><<<(view.group_count-1)/block_size+1,block_size,0,stream>>>(
        view,0.0,cutoff_rsq,atom_types,type_weights,center_adjoint,force,virial);
    check_launch();
}

// 继续文件开头的匿名命名空间，使下面定义与前置声明属于同一作用域。
namespace {

// 共同的径向开关；WEIGHT_CHEM 直接使用 s，Stru_factor 再乘 sinc(qr)。
// 返回 s 与 (ds/dr)/r，反向直接乘位移向量；内侧平坦区的导数为零。
__device__ __forceinline__ RadialValue radial_window(double r2, double cutoff_rsq) {
    if (r2 > cutoff_rsq) return {0.0,0.0};
    const double onset = 0.8*cutoff_rsq;
    if (r2 <= onset) return {1.0,0.0};
    const double width = cutoff_rsq-onset;
    const double scaled_distance = (r2-onset)/width;
    return {1.0-scaled_distance*scaled_distance*scaled_distance,
            -6.0*scaled_distance*scaled_distance/width};
}

// 一个中心-邻居项：phi(r) = sinc(q*r) * s(r)，sinc(x) = sin(x)/x。
// 沿用原 Stru_factor 的径向平滑，阈值作用在 r^2 上：
//   r^2 <= 0.8*rc^2 : s = 1
//   0.8*rc^2 < r^2 <= rc^2 : s = 1-u^3，u=(r^2-0.8*rc^2)/(0.2*rc^2)
//   r^2 > rc^2 : phi = 0
// 此多项式保证 rc 处的值为零，但不保证截断处的一阶导数连续。
// 这里的径向平滑是 S_i 定义的一部分，与图中作用在 CV 值上的开关函数分开。
// 前向和反向共用本函数，返回 phi(r) 和 phi'(r)/r。
__device__ __forceinline__ RadialValue radial(double r2, double q, double cutoff_rsq) {
    if (r2 > cutoff_rsq) return {0.0, 0.0};
    const double x2 = q*q*r2;
    double sinc, dsinc_over_r;
    // 小 qr 时直接计算 cos(x)-sinc(x) 会消减；分别使用各自的泰勒展开。
    // r=0 时 sinc -> 1，(d sinc(qr)/dr)/r -> -q^2/3，无需除以 r。
    if (x2 < 1e-6) {
        sinc = 1.0 + x2*(-1.0/6.0 + x2*(1.0/120.0 - x2/5040.0));
        dsinc_over_r = q*q*(-1.0/3.0 + x2*(1.0/30.0 + x2*(-1.0/840.0 + x2/45360.0)));
    } else {
        const double x = sqrt(x2);
        double sn, cs;
        sincos(x, &sn, &cs);
        sinc = sn/x;
        // (d sinc(qr)/dr)/r = q^2 * (cos(x)-sinc(x))/x^2。
        dsinc_over_r = q*q*(cs-sinc)/x2;
    }
    const auto window = radial_window(r2,cutoff_rsq);
    // 乘积求导：phi'(r)/r = [(d sinc(qr)/dr)/r]*s + sinc(qr)*[(ds/dr)/r]。
    return {sinc*window.value, dsinc_over_r*window.value + sinc*window.derivative_over_r};
}

// 编译期选择径向定义，三种 CV 共用相同的中心/邻居索引与反向累加。
template<bool UseStructureFactor>
__device__ __forceinline__ RadialValue pair_radial(double r2, double q, double cutoff_rsq) {
    if constexpr (UseStructureFactor) return radial(r2,q,cutoff_rsq);
    else return radial_window(r2,cutoff_rsq);
}

// 原始结构因子 S_i = 1 + sum_j phi(r_ij)，化学版的每项另乘 w_i*w_j。
// 常数 1 沿用原定义，邻居表只提供中心以外的原子。
// 每线程独占一个中心的输出，在寄存器中完成邻居求和后写回。
// k 用于中心表和邻居段，values 使用转换后的原子 localid。
template<bool UseChemicalWeights, bool UseStructureFactor>
__global__ void forward_kernel(StruFactorKernelView view, double q, double cutoff_rsq,
                               const int* atom_types, const double* type_weights, double* values) {
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= view.group_count) return;
    double value = 1.0;
    double center_weight = 1.0;
    if constexpr (UseChemicalWeights) center_weight = type_weights[atom_types[view.d_group_indices[k]]];
    // 有效邻居编号和几何使用相同槽位，段尾可能保留未使用槽位。
    const auto first = view.d_group_numneigh[k];
    for (int t = 0; t < view.d_neigh_in_cutoff_r[k]; ++t) {
        const auto pair = first+t;
        double pair_weight = 1.0;
        if constexpr (UseChemicalWeights)
            pair_weight = center_weight*type_weights[atom_types[view.d_calculated_numneigh[pair]]];
        value += pair_weight*pair_radial<UseStructureFactor>(view.d_group_dminneigh[4*pair+3], q, cutoff_rsq).value;
    }
    // 按 LAMMPS 原子顺序写回；无邻居中心仍得到 1，组外位置不在这里清零。
    values[view.d_group_indices[k]] = value;
}

// 对中心 i 的每个有向邻居项应用链式法则，seed = dV/dS_i。
// 若 i、j 都是中心，i->j 与 j->i 分属 S_i 和 S_j，各使用自己的 seed；
// 因此无需再根据邻居是否在组内乘 2，也不在这里做 AVE 的归一化。
template<bool UseChemicalWeights, bool UseStructureFactor>
__global__ void backward_kernel(StruFactorKernelView view, double q, double cutoff_rsq,
                                const int* atom_types, const double* type_weights,
                                const double* adjoint, double* force, double* virial) {
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= view.group_count) return;
    // adjoint 使用压缩中心序号 k；force/virial 使用原子 localid。
    const double seed = adjoint[k];
    if (seed == 0.0) return;
    const auto center = view.d_group_indices[k];
    double center_weight = 1.0;
    if constexpr (UseChemicalWeights) center_weight = type_weights[atom_types[center]];
    // 本线程先累加自身中心的贡献，结束后统一写回，减少中心上的原子加次数。
    double center_force[3] = {0.0,0.0,0.0};
    double center_virial[6] = {0.0,0.0,0.0,0.0,0.0,0.0};
    const auto first = view.d_group_numneigh[k];
    for (int t = 0; t < view.d_neigh_in_cutoff_r[k]; ++t) {
        const auto pair = first+t;
        const auto neighbor = view.d_calculated_numneigh[pair];
        const double* dr = view.d_group_dminneigh+4*pair;
        // dr = r_j-r_i；dr[3] 保存 r^2，不是第四个坐标。
        // dS_i/dr_i = -phi'(r)*dr/r，物理力还需取负号，故：
        // F_i += seed*phi'(r)*dr/r，F_j -= seed*phi'(r)*dr/r。
        double pair_weight = 1.0;
        if constexpr (UseChemicalWeights) pair_weight = center_weight*type_weights[atom_types[neighbor]];
        // 类型权重固定，链式求导时与图传入的中心 seed 一起乘到径向导数上。
        const double scale = seed*pair_weight*pair_radial<UseStructureFactor>(dr[3],q,cutoff_rsq).derivative_over_r;
        double f[3];
        for (int c=0; c<3; ++c) {
            f[c] = scale*dr[c];
            center_force[c] += f[c];
            // 不同中心可能共享同一个邻居（包括 ghost），必须原子累加。
            atomicAdd(force+3*neighbor+c, -f[c]);
        }
        if (virial) {
            // 本有向项的总 virial：W = r_i tensor F_i + r_j tensor F_j
            //                           = -(r_j-r_i) tensor F_i。
            // 六分量顺序为 xx,yy,zz,xy,xz,yz；径向力与 dr 平行，张量对称。
            // 将每项的一半记到中心、另一半记到邻居，使逐原子求和等于总 W。
            // 这个 1/2 只用于 virial 分摊，不用于力或 S_i 的求和。
            const double half[6] = {-.5*dr[0]*f[0], -.5*dr[1]*f[1], -.5*dr[2]*f[2],
                                    -.5*dr[0]*f[1], -.5*dr[0]*f[2], -.5*dr[1]*f[2]};
            for (int c=0; c<6; ++c) {
                center_virial[c] += half[c];
                atomicAdd(virial+6*neighbor+c, half[c]);
            }
        }
    }
    // 中心也可能是其他线程的邻居，最后写回仍需 atomicAdd。
    // ghost 贡献暂存在本 rank 的输出中，后续由 CV 层反向通信汇总到 owner。
    for (int c=0; c<3; ++c) atomicAdd(force+3*center+c, center_force[c]);
    if (virial)
        for (int c=0; c<6; ++c) atomicAdd(virial+6*center+c, center_virial[c]);
}
} // namespace
} // namespace MetaD_zqc
