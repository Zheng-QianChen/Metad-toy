#include "CV_Weighted_Chemical_Pair_Order.h"
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "group.h"
#include "neighbor.h"
#include "zqc_debug.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>

using namespace LAMMPS_NS;

namespace MetaD_zqc {

CV* Weighted_chem_pair::create(LAMMPS* lmp, FixMetadynamics* Fixmetad, FILE* f_check,
                               int narg, char** arg, int& i) {
    auto* error = lmp->error;
    ERR_COND(i+1 >= narg, "WEIGHT_CHEM requires a center group");
    StruFactorRequest request{};
    request.cal_name = arg[i];
    request.group_name = arg[i+1];
    request.group_id = lmp->group->find(request.group_name);
    ERR_COND(request.group_id < 0, "WEIGHT_CHEM: group %s not found", request.group_name);
    request.cutoff_r = 8.0;
    request.d_block_size = 128;
    request.use_chemical_lock = true;
    int cursor = i+2;
    while (cursor < narg) {
        const std::string keyword = arg[cursor];
        if (keyword == "cutoff_r" || keyword == "Chem_ctarget" || keyword == "Chem_sigma") {
            ERR_COND(cursor+1 >= narg, "WEIGHT_CHEM: %s requires a number", arg[cursor]);
            const double value = utils::numeric(FLERR, arg[cursor+1], false, lmp);
            if (keyword == "cutoff_r") request.cutoff_r = value;
            else if (keyword == "Chem_ctarget") request.c_target = value;
            else request.sigma = value;
            cursor += 2;
        } else if (keyword == "d_block_size") {
            ERR_COND(cursor+1 >= narg, "WEIGHT_CHEM: d_block_size requires an integer");
            request.d_block_size = utils::inumeric(FLERR, arg[cursor+1], false, lmp);
            cursor += 2;
        } else if (keyword == "Chem_map") {
            ++cursor;
            ERR_COND(cursor >= narg || arg[cursor][0] != '(', "WEIGHT_CHEM: Chem_map requires (type,weight) pairs");
            // 支持 (1,1.0)、单个引号括起的 '(1 1.0)'，以及 LAMMPS 拆开的 (1 1.0)。
            while (cursor < narg && arg[cursor][0] == '(') {
                std::string mapping = arg[cursor++];
                if (mapping.back() != ')') {
                    ERR_COND(cursor >= narg, "WEIGHT_CHEM: incomplete Chem_map pair");
                    mapping += " ";
                    mapping += arg[cursor++];
                }
                ERR_COND(mapping.size() < 5 || mapping.back() != ')', "WEIGHT_CHEM: invalid Chem_map pair %s", mapping.c_str());
                std::string contents = mapping.substr(1,mapping.size()-2);
                std::replace(contents.begin(),contents.end(),',',' ');
                std::istringstream fields(contents);
                std::string type_text, weight_text, extra;
                ERR_COND(!(fields >> type_text >> weight_text) || bool(fields >> extra),
                         "WEIGHT_CHEM: Chem_map expects exactly one type and weight");
                const int atom_type = utils::inumeric(FLERR,type_text.c_str(),false,lmp);
                const double weight = utils::numeric(FLERR,weight_text.c_str(),false,lmp);
                request.custom_weights[atom_type] = weight;
            }
        } else if (keyword == "SW_func") {
            error->all(FLERR,"WEIGHT_CHEM: express switching functions in Graph using the RAW input");
        } else {
            break;
        }
    }
    i = cursor;
    const double cutoff_rsq = request.cutoff_r*request.cutoff_r;
    ERR_COND(!std::isfinite(cutoff_rsq) || cutoff_rsq <= 0 || request.cutoff_r <= 0,
             "WEIGHT_CHEM: cutoff_r must be finite and positive");
    ERR_COND(request.d_block_size <= 0 || request.d_block_size > 1024,
             "WEIGHT_CHEM: d_block_size must be in [1,1024]");
    ERR_COND(!std::isfinite(request.c_target) || !std::isfinite(request.sigma) || request.sigma <= 0,
             "WEIGHT_CHEM: Chem_ctarget must be finite and Chem_sigma positive and finite");
    for (const auto& entry : request.custom_weights)
        ERR_COND(entry.first < 1 || entry.first > lmp->atom->ntypes || !std::isfinite(entry.second),
                 "WEIGHT_CHEM: Chem_map requires valid atom types and finite weights");

    // CV 独立请求所需的邻居半径；ghost 坐标同时覆盖 cutoff 和 skin。
    NeighSpec neighbor_spec;
    neighbor_spec.full = 1;
    neighbor_spec.ghost = 0;
    neighbor_spec.use_cutoff = 1;
    neighbor_spec.cutoff = request.cutoff_r;
    lmp->comm->cutghostuser = std::max(lmp->comm->cutghostuser,request.cutoff_r+lmp->neighbor->skin);
    auto* environment = static_cast<Stru_fact_chem_env*>(Stru_fact_env::get_or_create(lmp,Fixmetad,f_check,request));
    environment->neigh_id = Fixmetad->neigh_hub.get_or_create(neighbor_spec);
    LOG("WEIGHT_CHEM: group=%s cutoff_r=%g d_block_size=%d Chem_ctarget=%g Chem_sigma=%g",
        request.group_name,request.cutoff_r,request.d_block_size,request.c_target,request.sigma);
    return new Weighted_chem_pair(lmp,Fixmetad,f_check,environment,request.d_block_size);
}

Weighted_chem_pair::Weighted_chem_pair(LAMMPS* lmp, FixMetadynamics* Fixmetad, FILE* f_check,
                                       Stru_fact_chem_env* environment, int block_size)
    : Stru_factor(lmp,Fixmetad,f_check,environment,0.0,block_size), my_chem_env(environment) {}

auto Weighted_chem_pair::set_CV_calculate(std::string) -> CV_Calculation {
    throw std::runtime_error("WEIGHT_CHEM: use EXECUTOR GRAPH; reductions belong to the graph");
}
auto Weighted_chem_pair::set_CV_bias_force(std::string) -> CV_BiasForce {
    throw std::runtime_error("WEIGHT_CHEM: use EXECUTOR GRAPH for adjoint propagation");
}

// W_i 对应结构因子的 q=0 极限，实际启动只计算径向开关的编译器特化。
void Weighted_chem_pair::call_structure_factor_cv_kernel() {
    weighted_chem_forward(my_env->data,my_env->cutoff_r*my_env->cutoff_r,
                          my_chem_env->d_atom_types.ptr,my_chem_env->d_type_weights.ptr,
                          d_stru_factor.ptr,0,d_block_size);
}
void Weighted_chem_pair::call_structure_factor_adjoint_kernel(
    const double* center_adjoint, double* atom_forces, double* atom_virials) {
    weighted_chem_backward(my_env->data,my_env->cutoff_r*my_env->cutoff_r,
                           my_chem_env->d_atom_types.ptr,my_chem_env->d_type_weights.ptr,
                           center_adjoint,atom_forces,atom_virials,0,d_block_size);
}

double* Weighted_chem_pair::get_peratom_ptr(const std::string& name) {
    return name == "chem_pair_r" ? h_stru_factor : nullptr;
}

} // namespace MetaD_zqc
REGISTER_CV("WEIGHT_CHEM", MetaD_zqc::Weighted_chem_pair::create);
