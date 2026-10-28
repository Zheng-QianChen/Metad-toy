// 真实适配器的缓存契约测试。临时改动只发生在独立测试实例，并由 Restore 恢复。
// 检查相同长度下的 ID/成员变化，避免“数组长度没变所以缓存有效”的假阳性。
#include "lammps.h"
#include "atom.h"
#include "group.h"
#include "modify.h"
#include "fix_crystallize.h"
#include "zqc_lammps_engine.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <string>
#include <vector>
using namespace MetaD_zqc;
namespace {
thread_local std::string last_error;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
LammpsEngineAdapter& adapter(LAMMPS_NS::LAMMPS* simulation) {
    auto* fix = dynamic_cast<LAMMPS_NS::FixMetadynamics*>(simulation->modify->get_fix_by_id("m"));
    require(fix != nullptr, "fix m missing");
    return dynamic_cast<LammpsEngineAdapter&>(*fix->engine_adapter);
}
std::vector<int> read_flags(ArrayView<const int> view) {
    std::vector<int> result(view.item_count);
    if (!result.empty()) require(cudaMemcpy(result.data(), view.data, result.size()*sizeof(int),
                                           cudaMemcpyDeviceToHost) == cudaSuccess, "flag download failed");
    return result;
}
}
extern "C" const char* engine_cache_error() { return last_error.c_str(); }
extern "C" int engine_cache_stats(void* pointer, unsigned long long* output) {
    try {
        auto* simulation = static_cast<LAMMPS_NS::LAMMPS*>(pointer);
        const auto stats = adapter(simulation).sync_stats();
        const std::uint64_t counters[] = {stats.position_uploads, stats.type_uploads, stats.id_rebuilds,
            stats.selection_scans, stats.selection_flag_uploads, stats.neighbor_rebuilds, stats.neighbor_uploads};
        std::copy_n(counters, 7, output);
        return 0;
    } catch (const std::exception& error) { last_error = error.what(); return 1; }
}
extern "C" int engine_cache_check(void* pointer) {
    try {
        auto* simulation = static_cast<LAMMPS_NS::LAMMPS*>(pointer);
        auto& engine = adapter(simulation);
        auto* atom = simulation->atom;
        const int selection = simulation->group->find("type2");
        const int bit = simulation->group->bitmask[selection];
        const int total = atom->nlocal + atom->nghost;
        struct Restore {
            LAMMPS_NS::Atom* atom;
            int& dynamic;
            int old_dynamic;
            LammpsEngineAdapter& engine;
            std::vector<int> masks, types;
            std::vector<LAMMPS_NS::tagint> tags;
            ~Restore() {
                std::copy(masks.begin(), masks.end(), atom->mask);
                std::copy(types.begin(), types.end(), atom->type);
                std::copy(tags.begin(), tags.end(), atom->tag);
                dynamic = old_dynamic;
                engine.invalidate_topology();
            }
        } restore{atom, simulation->group->dynamic[selection], simulation->group->dynamic[selection], engine,
                  {atom->mask, atom->mask+total}, {atom->type, atom->type+total}, {atom->tag, atom->tag+total}};

        // 共享静态组只扫描一次；Host 成员请求不会触发组标记上传。
        const auto before = engine.sync_stats();
        engine.selection_members(selection);
        const auto prepared = engine.sync_stats();
        engine.begin_evaluation();
        engine.selection_members(selection);
        require(engine.sync_stats().selection_scans == prepared.selection_scans, "static selection rescanned");
        require(engine.sync_stats().selection_flag_uploads == before.selection_flag_uploads, "Host request uploaded flags");
        engine.selection_flags(selection, MemorySpace::Cuda);
        const auto uploaded = engine.sync_stats().selection_flag_uploads;
        engine.begin_evaluation();
        engine.selection_flags(selection, MemorySpace::Cuda);
        require(engine.sync_stats().selection_flag_uploads == uploaded, "unchanged flags uploaded again");

        // 动态组交换一个成员和一个非成员，成员数保持不变。
        int member = -1, nonmember = -1;
        for (int local = 0; local < total; ++local)
            if (atom->mask[local]&bit) member = local; else nonmember = local;
        require(member >= 0 && nonmember >= 0, "fixture requires both selection states");
        simulation->group->dynamic[selection] = 1;
        atom->mask[member] &= ~bit;
        atom->mask[nonmember] |= bit;
        engine.begin_evaluation();
        const auto changed = read_flags(engine.selection_flags(selection, MemorySpace::Cuda));
        for (int local = 0; local < total; ++local)
            require(changed[local] == bool(atom->mask[local]&bit), "dynamic flags stale");
        require(engine.sync_stats().selection_flag_uploads == uploaded+1, "changed flags were not uploaded");
        engine.begin_evaluation();
        engine.selection_flags(selection, MemorySpace::Cuda);
        require(engine.sync_stats().selection_flag_uploads == uploaded+1, "unchanged dynamic flags uploaded again");

        // 类型可能在邻居重建间隔内改变；每次请求比较内容，变化后才上传。
        engine.atom_types(MemorySpace::Cuda);
        const auto type_uploads = engine.sync_stats().type_uploads;
        engine.begin_evaluation();
        engine.atom_types(MemorySpace::Cuda);
        require(engine.sync_stats().type_uploads == type_uploads, "unchanged types uploaded again");
        atom->type[0] = atom->type[0] == 1 ? 2 : 1;
        engine.begin_evaluation();
        const auto changed_types = read_flags(engine.atom_types(MemorySpace::Cuda));
        require(changed_types[0] == atom->type[0], "changed type stale");

        // 同原子数、同 localid 整数 CSR，稳定 ID 的映射已改变。
        require(atom->nlocal >= 2, "fixture needs two owned atoms per rank");
        const AtomId first_id = atom->tag[0];
        engine.atom_ids();
        const auto ids_built = engine.sync_stats().id_rebuilds;
        engine.begin_evaluation();
        engine.atom_ids();
        require(engine.sync_stats().id_rebuilds == ids_built, "unchanged IDs rebuilt");
        const auto previous_topology = engine.prepare_neighbors(0).topology_generation;
        std::swap(atom->tag[0], atom->tag[1]);
        engine.invalidate_topology();
        engine.begin_evaluation();
        require(engine.atom_ids()(1) == first_id, "ID reorder not invalidated");
        double coordinate[4]; int local;
        engine.read_owned_atoms(&first_id, 1, coordinate, &local);
        require(local == 1 && coordinate[3] == 1, "owned ID map stale");
        require(engine.prepare_neighbors(0).topology_generation > previous_topology,
                "same integer CSR concealed changed identity");
        return 0;
    } catch (const std::exception& error) { last_error = error.what(); return 1; }
}
