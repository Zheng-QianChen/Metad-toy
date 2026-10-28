// 当前坐标逐对枚举真实二跳邻域，检查缓存候选 CSR 的完整性。
#include "lammps.h"
#include "atom.h"
#include "group.h"
#include "modify.h"
#include "neighbor.h"
#include "fix_crystallize.h"
#include "zqc_lammps_engine.h"
#include "bias_probe.cpp"
#include <algorithm>
#include <string>
#include <vector>
namespace {
thread_local std::string ghost_error;
MetaD_zqc::LammpsEngineAdapter& ghost_engine(LAMMPS_NS::LAMMPS* simulation) {
    auto* fix = dynamic_cast<LAMMPS_NS::FixMetadynamics*>(simulation->modify->get_fix_by_id("m"));
    if (!fix) throw std::runtime_error("fix m missing");
    return dynamic_cast<MetaD_zqc::LammpsEngineAdapter&>(*fix->engine_adapter);
}
}
extern "C" const char* ghost_cache_error() { return ghost_error.c_str(); }
extern "C" int ghost_cache_register(void* pointer, double cutoff, int selection) {
    try {
        MetaD_zqc::NeighborRequest request;
        request.cutoff = cutoff; request.ghost_centers = true;
        request.centers = request.neighbors = selection;
        return ghost_engine(static_cast<LAMMPS_NS::LAMMPS*>(pointer)).register_neighbors(request);
    } catch (const std::exception& error) { ghost_error = error.what(); return -1; }
}
extern "C" int ghost_cache_stats(void* pointer, unsigned long long* output) {
    try {
        auto* simulation = static_cast<LAMMPS_NS::LAMMPS*>(pointer);
        const auto stats = ghost_engine(simulation).sync_stats();
        output[0] = stats.neighbor_rebuilds; output[1] = stats.ghost_row_rebuilds;
        output[2] = stats.neighbor_uploads; output[3] = simulation->neighbor->lastcall;
        return 0;
    } catch (const std::exception& error) { ghost_error = error.what(); return 1; }
}
extern "C" int ghost_cache_verify(void* pointer, int handle, double radius, int selection,
                                  unsigned long long* output) {
    try {
        auto* simulation = static_cast<LAMMPS_NS::LAMMPS*>(pointer);
        const auto view = ghost_engine(simulation).prepare_neighbors(handle);
        auto* atom = simulation->atom;
        const int total = atom->nlocal + atom->nghost;
        const int group_mask = selection < 0 ? -1 : simulation->group->bitmask[selection];
        std::vector<std::vector<int>> rows(total);
        std::vector<unsigned char> required_ghost(total, 0), listed(total, 0);
        for (size_t row = 0; row < view.host_centers.item_count; ++row) {
            const int center = view.host_centers(row); listed[center] = 1;
            for (int entry = view.host_offsets(row); entry < view.host_offsets(row+1); ++entry)
                rows[center].push_back(view.host_neighbors(entry));
            std::sort(rows[center].begin(), rows[center].end());
        }
        std::fill_n(output, 3, 0);
        const auto check_row = [&](int center) {
            if (!listed[center]) throw std::runtime_error("required center row missing");
            for (int neighbor = 0; neighbor < total; ++neighbor) {
                if (neighbor == center || (selection >= 0 && !(atom->mask[neighbor]&group_mask))) continue;
                double distance_sq = 0;
                for (int axis = 0; axis < 3; ++axis) {
                    const double difference = atom->x[center][axis]-atom->x[neighbor][axis];
                    distance_sq += difference*difference;
                }
                if (distance_sq > radius*radius) continue;
                if (!std::binary_search(rows[center].begin(), rows[center].end(), neighbor))
                    throw std::runtime_error("missing physical edge " + std::to_string(atom->tag[center]) +
                                             " -> " + std::to_string(atom->tag[neighbor]));
                ++output[center >= atom->nlocal];
                if (center < atom->nlocal && neighbor >= atom->nlocal) required_ghost[neighbor] = 1;
                if (center >= atom->nlocal && atom->tag[center] == 2 && atom->tag[neighbor] == 3) ++output[2];
            }
        };
        for (int center = 0; center < atom->nlocal; ++center)
            if (selection < 0 || (atom->mask[center]&group_mask)) check_row(center);
        for (int center = atom->nlocal; center < total; ++center)
            if (required_ghost[center]) check_row(center);
        return 0;
    } catch (const std::exception& error) { ghost_error = error.what(); return 1; }
}

