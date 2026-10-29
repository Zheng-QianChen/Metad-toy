#include "CV_Weighted_Chemical_Pair_Order.h"
#include "zqc_cv_debug.h"
#include "zqc_cv_parse.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>

namespace MetaD_zqc {

CV* Weighted_chem_pair::create(CVContext& context, int narg, char** arg, int& i) {
    auto logger = context.logger;
    ERR_COND(i+1 >= narg, "WEIGHT_CHEM requires a center group");
    StruFactorRequest request{};
    request.cal_name = arg[i];
    request.selection_name = arg[i+1];
    request.selection = context.engine.selection_id(request.selection_name);
    request.cutoff_r = 8.0;
    request.d_block_size = 128;
    request.use_chemical_lock = true;
    int cursor = i+2;
    while (cursor < narg) {
        const std::string keyword = arg[cursor];
        if (keyword == "cutoff_r" || keyword == "Chem_ctarget" || keyword == "Chem_sigma") {
            ERR_COND(cursor+1 >= narg, "WEIGHT_CHEM: %s requires a number", arg[cursor]);
            const double value = cv_number(arg[cursor+1]);
            if (keyword == "cutoff_r") request.cutoff_r = value;
            else if (keyword == "Chem_ctarget") request.c_target = value;
            else request.sigma = value;
            cursor += 2;
        } else if (keyword == "d_block_size") {
            ERR_COND(cursor+1 >= narg, "WEIGHT_CHEM: d_block_size requires an integer");
            request.d_block_size = cv_integer(arg[cursor+1]);
            cursor += 2;
        } else if (keyword == "Chem_map") {
            ++cursor;
            ERR_COND(cursor >= narg || arg[cursor][0] != '(', "WEIGHT_CHEM: Chem_map requires (type,weight) pairs");
            // 支持 (1,1.0)、单个引号括起的 '(1 1.0)'，以及输入拆开的 (1 1.0)。
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
                const int atom_type = context.engine.atom_type_id(type_text);
                const double weight = cv_number(weight_text.c_str());
                request.custom_weights[atom_type] = weight;
            }
        } else if (keyword == "SW_func") {
            throw std::runtime_error("WEIGHT_CHEM: express switching functions in Graph using the RAW input");
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
    const int type_begin = context.engine.atom_type_begin();
    const int type_count = context.engine.atom_type_count();
    for (const auto& entry : request.custom_weights)
        ERR_COND(entry.first < type_begin || entry.first >= type_begin + type_count ||
                 !std::isfinite(entry.second),
                 "WEIGHT_CHEM: Chem_map requires valid atom types and finite weights");

    auto* environment = static_cast<Stru_fact_chem_env*>(
        Stru_fact_env::get_or_create(context, request));
    LOG("WEIGHT_CHEM: group=%s cutoff_r=%g d_block_size=%d Chem_ctarget=%g Chem_sigma=%g",
        request.selection_name.c_str(),request.cutoff_r,request.d_block_size,request.c_target,request.sigma);
    return new Weighted_chem_pair(context, environment, request.d_block_size);
}

Weighted_chem_pair::Weighted_chem_pair(CVContext& context, Stru_fact_chem_env* environment,
                                       int block_size)
    : Stru_factor(context, environment, 0.0, block_size), my_chem_env(environment) {}

auto Weighted_chem_pair::set_CV_calculate(std::string) -> CV_Calculation {
    throw std::runtime_error("WEIGHT_CHEM: use EXECUTOR GRAPH; reductions belong to the graph");
}
auto Weighted_chem_pair::set_CV_bias_force(std::string) -> CV_BiasForce {
    throw std::runtime_error("WEIGHT_CHEM: use EXECUTOR GRAPH for adjoint propagation");
}

void Weighted_chem_pair::call_structure_factor_cv_kernel() {
    const size_t nall = static_cast<size_t>(engine.total_atom_count());
    const size_t weight_slots = my_chem_env->h_type_weights.size();
    weighted_chem_forward(my_env->data, my_env->cutoff_r*my_env->cutoff_r,
                          my_chem_env->device_atom_types,
                          my_chem_env->d_type_weights.view(weight_slots),
                          d_stru_factor.view(nall), my_env->execution.get(), d_block_size);
}
void Weighted_chem_pair::call_structure_factor_adjoint_kernel(
    ArrayView<const double> center_adjoint, ArrayView<double> atom_forces,
    ArrayView<double> atom_virials) {
    const size_t weight_slots = my_chem_env->h_type_weights.size();
    weighted_chem_backward(my_env->data, my_env->cutoff_r*my_env->cutoff_r,
                           my_chem_env->device_atom_types,
                           my_chem_env->d_type_weights.view(weight_slots),
                           center_adjoint, atom_forces, atom_virials, my_env->execution.get(), d_block_size);
}

ArrayView<const double> Weighted_chem_pair::peratom_values(const std::string& name) {
    if (name != "chem_pair_r") throw std::invalid_argument("WEIGHT_CHEM: unknown property "+name);
    return Stru_factor::peratom_values("stru_f");
}

} // namespace MetaD_zqc
REGISTER_CV("WEIGHT_CHEM", MetaD_zqc::Weighted_chem_pair::create);
