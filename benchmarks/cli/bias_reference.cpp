// Offline cross-check: evaluate the production C++ kernels with the test fixture.
// Arrays use row-major [hill, axis] and [point, axis] storage; no MD engine is needed.
#include "zqc_hill_bias.h"
#include <exception>
using namespace MetaD_zqc;

extern "C" int metadtoy_bias_reference(
    int dimension, int parallel, int plumed, double thermal_energy,
    const double* sigma, const double* period, const double* centers,
    const double* heights, int count, const double* points, int point_count,
    double* energies) {
    try {
        HillBiasSettings settings;
        settings.mode = parallel ? HillBiasMode::Parallel : HillBiasMode::Hills;
        settings.pb_kernel = plumed ? PBKernel::Plumed210 : PBKernel::SmoothGaussian;
        settings.thermal_energy = thermal_energy;
        for (int axis = 0; axis < dimension; ++axis)
            settings.axes.push_back({sigma[axis], period[axis]});
        HillBias bias(settings);
        for (int hill = 0; hill < count; ++hill)
            bias.append(hill, centers + hill * dimension,
                        heights + hill * (parallel ? dimension : 1));
        for (int point = 0; point < point_count; ++point)
            energies[point] = bias.evaluate(points + point * dimension);
        return 0;
    } catch (const std::exception&) {
        return 1;
    }
}
