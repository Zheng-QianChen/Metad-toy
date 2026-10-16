#include "CV_Distance.h"
#include "zqc_debug.h"
#include "atom.h"
#include "domain.h"
#include "error.h"
#include "utils.h"
#include <cmath>
#include <mpi.h>
#include <stdexcept>

namespace MetaD_zqc {
CV* Distance::create(LAMMPS_NS::LAMMPS* lmp, LAMMPS_NS::FixMetadynamics* fix,
                     FILE* output, int narg, char** arg, int& index) {
    if (index + 2 >= narg)
        lmp->error->all(FLERR, "DISTANCE requires two global atom IDs");
    const auto first = LAMMPS_NS::utils::tnumeric(FLERR, arg[index+1], false, lmp);
    const auto second = LAMMPS_NS::utils::tnumeric(FLERR, arg[index+2], false, lmp);
    index += 3;
    return new Distance(lmp, fix, output, first, second);
}

Distance::Distance(LAMMPS_NS::LAMMPS* lmp, LAMMPS_NS::FixMetadynamics* fix,
                   FILE* output, LAMMPS_NS::tagint first, LAMMPS_NS::tagint second)
    : CV(lmp, fix, output), atom_ids_{first, second} {
    if (first <= 0 || second <= 0 || first == second)
        error->all(FLERR, "DISTANCE requires two distinct positive atom IDs");
    if (!lmp->atom->tag_enable)
        error->all(FLERR, "DISTANCE requires atom IDs");
}

void Distance::summary(FILE* output) {
    fprintf(output, "DISTANCE: global atom IDs " TAGINT_FORMAT " " TAGINT_FORMAT
                    "; minimum-image distance, Global scalar\n", atom_ids_[0], atom_ids_[1]);
    fflush(output);
}

void Distance::base_calc() {
    // 两组 (x,y,z,owner_count) 一次规约。每步重新定位以跟随迁移与排序。
    double coordinates[8] = {};
    owned_indices_[0] = owned_indices_[1] = -1;
    for (int local = 0; local < lmp->atom->nlocal; ++local) {
        for (int endpoint = 0; endpoint < 2; ++endpoint) {
            if (lmp->atom->tag[local] != atom_ids_[endpoint]) continue;
            owned_indices_[endpoint] = local;
            for (int axis = 0; axis < 3; ++axis)
                coordinates[4*endpoint+axis] += lmp->atom->x[local][axis];
            coordinates[4*endpoint+3] += 1.0;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, coordinates, 8, MPI_DOUBLE, MPI_SUM, lmp->world);
    if (coordinates[3] != 1.0 || coordinates[7] != 1.0)
        error->all(FLERR, "DISTANCE: each atom ID must have exactly one owner");
    for (int axis = 0; axis < 3; ++axis)
        displacement_[axis] = coordinates[4+axis] - coordinates[axis];
    // LAMMPS 同时处理正交盒、倾斜盒及各方向周期性。
    lmp->domain->minimum_image(displacement_);
    cv_value = std::sqrt(displacement_[0]*displacement_[0] +
                         displacement_[1]*displacement_[1] +
                         displacement_[2]*displacement_[2]);
    if (!(cv_value > 0.0) || !std::isfinite(cv_value))
        error->all(FLERR, "DISTANCE: zero or non-finite separation has no defined gradient");
}

double Distance::compute_cv() { return cv_value; }

void Distance::bias_force(double seed) {
    const double scale = seed / cv_value;
    const double* delta = displacement_;
    // W = -dV/d(epsilon)，顺序为 xx, yy, zz, xy, xz, yz；两端各保存一半。
    const double pair_virial[6] = {
        -scale*delta[0]*delta[0], -scale*delta[1]*delta[1], -scale*delta[2]*delta[2],
        -scale*delta[0]*delta[1], -scale*delta[0]*delta[2], -scale*delta[1]*delta[2]};
    for (int endpoint = 0; endpoint < 2; ++endpoint) {
        const int local = owned_indices_[endpoint];
        if (local < 0) continue;
        const double sign = endpoint == 0 ? 1.0 : -1.0;
        for (int axis = 0; axis < 3; ++axis)
            lmp->atom->f[local][axis] += sign * scale * delta[axis];
        if (Fixmetad->a_virial)
            for (int component = 0; component < 6; ++component)
                Fixmetad->a_virial[6*local+component] += 0.5 * pair_virial[component];
    }
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
