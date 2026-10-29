#include "CV_Distance.h"
#include "zqc_cv_debug.h"
#include "zqc_cv_parse.h"
#include <cmath>
#include <stdexcept>

namespace MetaD_zqc {
CV* Distance::create(CVContext& context, int narg, char** arg, int& index) {
    if (index + 2 >= narg)
        throw std::invalid_argument("DISTANCE requires two global atom IDs");
    const AtomId first = cv_atom_id(arg[index+1]);
    const AtomId second = cv_atom_id(arg[index+2]);
    index += 3;
    return new Distance(context, first, second);
}

Distance::Distance(CVContext& context, AtomId first, AtomId second)
    : CV(context), atom_ids_{first, second}, calculation_(first, second) {}

void Distance::summary(FILE* output) {
    if (!output) return;
    fprintf(output, "DISTANCE: global atom IDs %llu %llu"
                    "; minimum-image distance, Global scalar\n",
            static_cast<unsigned long long>(atom_ids_[0]),
            static_cast<unsigned long long>(atom_ids_[1]));
    fflush(output);
}

// 主流程开始 evaluation；计算对象经由适配器读取、规约并回写 owned 数据。
void Distance::base_calc() {
    cv_value = calculation_.forward(engine);
}
double Distance::compute_cv() { return cv_value; }
void Distance::bias_force(double seed) {
    calculation_.backward(engine, seed);
}

CV::CV_Calculation Distance::set_CV_calculate(std::string method) {
    if (method != "DISTANCE" && method != "RAW" && method != "value")
        throw std::runtime_error("DISTANCE: supported methods are DISTANCE/RAW/value");
    return static_cast<CV_Calculation>(&Distance::compute_cv);
}
CV::CV_BiasForce Distance::set_CV_bias_force(std::string method) {
    set_CV_calculate(method);
    return static_cast<CV_BiasForce>(&Distance::bias_force);
}
}
REGISTER_CV("DISTANCE", MetaD_zqc::Distance::create);
