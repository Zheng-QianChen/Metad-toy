// 测试桥接：通过 Fix 的公开接口读取偏置种子、放置固定 hill。
// CV 与 Graph 的实际前向/反向仍由 LAMMPS run 0 执行。
#include "lammps.h"
#include "error.h"
#include "modify.h"
#include "fix_crystallize.h"
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;
LAMMPS_NS::FixMetadynamics* find_fix(void* pointer) {
    auto* lammps = static_cast<LAMMPS_NS::LAMMPS*>(pointer);
    const int index = lammps->modify->find_fix("m");
    if (index < 0) throw std::runtime_error("finite difference: fix m missing");
    return static_cast<LAMMPS_NS::FixMetadynamics*>(lammps->modify->fix[index]);
}
}
extern "C" const char* fd_probe_error() { return last_error.c_str(); }
extern "C" int fd_bias_gradient(void* pointer, double* gradient) {
    try {
        auto* fix = find_fix(pointer);
        std::vector<double> values(fix->size_vector);
        for (int dimension = 0; dimension < fix->size_vector; ++dimension)
            values[dimension] = fix->compute_vector(dimension);
        fix->get_dVdcv(values.data(), gradient);
        return 0;
    } catch (const std::exception& error) { last_error = error.what(); return 1; }
}
extern "C" int fd_add_hill(void* pointer, double* center) {
    try { find_fix(pointer)->add_hill(center); return 0; }
    catch (const std::exception& error) { last_error = error.what(); return 1; }
}
