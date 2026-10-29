// 向真实 LAMMPS 适配器提交可追踪的六/九分量数组，按稳定 ID 独立求和验证 reverse。
// 测试输入包含 owned 和 ghost；原始 CV 的物理导数另由 run.py 的有限差分检查。
#include "lammps.h"
#include "atom.h"
#include "modify.h"
#include "fix_crystallize.h"
#include "zqc_gpu_buffer.cuh"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <mpi.h>

using namespace MetaD_zqc;
namespace {
thread_local std::string last_error;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class Function> void rejects(Function action) {
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "invalid virial output was accepted");
}
}
extern "C" const char* engine_virial_error() { return last_error.c_str(); }
extern "C" int engine_virial_check(void* pointer, int full9, int soa, int cuda_memory, double* max_error) {
    try {
        auto* lmp = static_cast<LAMMPS_NS::LAMMPS*>(pointer);
        const int fix_index = lmp->modify->find_fix("m");
        require(fix_index >= 0, "fix m is required");
        auto* fix = static_cast<LAMMPS_NS::FixMetadynamics*>(lmp->modify->fix[fix_index]);
        auto& engine = *fix->engine_adapter;
        auto* atom = lmp->atom;
        const size_t owned = atom->nlocal, total = owned+atom->nghost;
        const size_t components = full9?9:6, slots = total+7;
        require(fix->a_virial != nullptr, "fixture must request virial tally");
        int rank, ghost_count = atom->nghost;
        MPI_Comm_rank(lmp->world,&rank);
        MPI_Allreduce(MPI_IN_PLACE,&ghost_count,1,MPI_INT,MPI_SUM,lmp->world);
        require(ghost_count > 0, "fixture must exercise ghost reverse");
        long long max_tag = 0;
        for (size_t local = 0; local < total; ++local)
            max_tag = std::max(max_tag,static_cast<long long>(atom->tag[local]));
        MPI_Allreduce(MPI_IN_PLACE,&max_tag,1,MPI_LONG_LONG,MPI_MAX,lmp->world);
        require(max_tag > 0 && max_tag < 100000, "probe expects a small fixture with positive IDs");
        std::vector<double> expected((max_tag+1)*9,0), baseline(owned*9);
        std::vector<double> forces(slots*3,-999), virials(slots*components,-777);
        const auto force_offset = [&](size_t local, size_t component) {
            return soa?local+slots*component:3*local+component;
        };
        const auto virial_offset = [&](size_t local, size_t component) {
            return soa?local+slots*component:components*local+component;
        };
        for (size_t local = 0; local < total; ++local) {
            const size_t tag = atom->tag[local];
            // 每一列值不同，Full9 的转置三项可识别；数值为小整数，方便精确比较。
            const double scale = 100*tag+10*rank+local;
            for (size_t component = 0; component < 3; ++component) {
                forces[force_offset(local,component)] = scale+component;
                expected[9*tag+component] += scale+component;
                if (local < owned) baseline[9*local+component] = atom->f[local][component];
            }
            for (size_t component = 0; component < components; ++component) {
                virials[virial_offset(local,component)] = scale+20+component;
                if (component < 6) {
                    expected[9*tag+3+component] += scale+20+component;
                    if (local < owned) baseline[9*local+3+component] = fix->a_virial[6*local+component];
                }
            }
        }
        // 参考值按全局 ID 汇总全部副本，与适配器使用的 LAMMPS reverse 路由独立。
        MPI_Allreduce(MPI_IN_PLACE,expected.data(),static_cast<int>(expected.size()),MPI_DOUBLE,MPI_SUM,lmp->world);
        GpuBuffer<double> device_forces, device_virials;
        const double* force_pointer = forces.data();
        const double* virial_pointer = virials.data();
        if (cuda_memory) {
            device_forces.grow_to(forces.size(),__FILE__,__LINE__);
            device_virials.grow_to(virials.size(),__FILE__,__LINE__);
            device_forces.upload_from(forces.data(),forces.size());
            device_virials.upload_from(virials.data(),virials.size());
            force_pointer = device_forces.ptr; virial_pointer = device_virials.ptr;
        }
        const auto memory = cuda_memory?MemorySpace::Cuda:MemorySpace::Host;
        ArrayView<const double> force_view{force_pointer,total,3,soa?1:size_t(3),soa?slots:1,forces.size(),memory};
        VirialView virial_view{{virial_pointer,total,components,soa?1:components,soa?slots:1,virials.size(),memory},
                              full9?VirialRepresentation::Full9:VirialRepresentation::Symmetric6};
        // 错误元数据在通信之前拒绝；所有 rank 执行相同检查。
        auto invalid = virial_view;
        invalid.representation = full9?VirialRepresentation::Symmetric6:VirialRepresentation::Full9;
        rejects([&] { engine.commit_force_virial(force_view,invalid); });
        invalid = virial_view; invalid.values.item_count = total?total-1:1;
        rejects([&] { engine.commit_force_virial(force_view,invalid); });
        engine.commit_force_virial(force_view,virial_view);
        const double single_force[3] = {2,3,5};
        const double single_virial[18] = {11,-1,13,-1,17,-1,19,-1,23,-1,29,-1,31,-1,37,-1,41,-1};
        VirialView single{{single_virial,1,components,1,2,18,MemorySpace::Host},virial_view.representation};
        double local_error = 0;
        for (size_t local = 0; local < owned; ++local) {
            auto invalid_single = single; invalid_single.values.item_count = 0;
            rejects([&] { engine.add_owned_force_virial(local,single_force,invalid_single); });
            invalid_single = single; invalid_single.values.item_count = 2;
            rejects([&] { engine.add_owned_force_virial(local,single_force,invalid_single); });
            invalid_single = single; invalid_single.values.memory = MemorySpace::Cuda;
            rejects([&] { engine.add_owned_force_virial(local,single_force,invalid_single); });
            engine.add_owned_force_virial(local,single_force,single);
            const size_t tag = atom->tag[local];
            for (int component = 0; component < 3; ++component)
                local_error = std::max(local_error,std::abs(atom->f[local][component]-baseline[9*local+component]-
                    expected[9*tag+component]-single_force[component]));
            for (int component = 0; component < 6; ++component)
                local_error = std::max(local_error,std::abs(fix->a_virial[6*local+component]-baseline[9*local+3+component]-
                    expected[9*tag+3+component]-single_virial[2*component]));
        }
        if (cuda_memory) {
            auto actual_forces = forces, actual_virials = virials;
            device_forces.download_to(actual_forces.data(),actual_forces.size());
            device_virials.download_to(actual_virials.data(),actual_virials.size());
            require(actual_forces == forces && actual_virials == virials, "CUDA source or padding was modified");
        } else {
            // 重新从原公式检查整个输入，包括保留槽位，验证 Host 借用输入保持原样。
            for (size_t local = 0; local < slots; ++local) {
                const double scale = local<total?100*atom->tag[local]+10*rank+local:0;
                for (size_t component = 0; component < 3; ++component)
                    require(forces[force_offset(local,component)] == (local<total?scale+component:-999), "Host force source changed");
                for (size_t component = 0; component < components; ++component)
                    require(virials[virial_offset(local,component)] == (local<total?scale+20+component:-777), "Host virial source changed");
            }
        }
        MPI_Allreduce(&local_error,max_error,1,MPI_DOUBLE,MPI_MAX,lmp->world);
        require(*max_error < 1e-8, "LAMMPS force/virial projection or reverse mismatch");
        return 0;
    } catch (const std::exception& error) { last_error = error.what(); return 1; }
}
