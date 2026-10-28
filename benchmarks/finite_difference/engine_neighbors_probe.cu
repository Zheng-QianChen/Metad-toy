// 真实 LAMMPS 邻居请求验证：Half 与全体 Full 有向表独立筛选的参考结果比较。
// 采用 E_pair=|r_ij|^2/2，覆盖 pair 数、能量、两端力、九分量总 virial 和 ghost reverse。
#include "lammps.h"
#include "atom.h"
#ifdef ZQC_LAMMPS_KOKKOS
#include "atom_kokkos.h"
#include "atom_masks.h"
#endif
#include "group.h"
#include "modify.h"
#include "fix_crystallize.h"
#include "zqc_lammps_engine.h"
#include <mpi.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>
using namespace MetaD_zqc;
namespace {
thread_local std::string last_error;
struct Case { int centers, neighbors, handle; };
std::vector<Case> cases;
int reference_handle;
constexpr double cutoff = 3.05;
// 探针直接检查 Host 力；Kokkos 适配器将贡献写到 Device，读前同步，
// 恢复基线后标记 Host 修改，保持下一次 commit 的 DualView 状态正确。
void host_force_access(LAMMPS_NS::LAMMPS* simulation, bool modified = false) {
#ifdef ZQC_LAMMPS_KOKKOS
    if (simulation->kokkos) {
        auto* atom = static_cast<LAMMPS_NS::AtomKokkos*>(simulation->atom);
        if (modified) atom->modified(LAMMPS_NS::Host, F_MASK);
        else atom->sync(LAMMPS_NS::Host, F_MASK);
    }
#endif
}
void require(bool value,const char* message) { if(!value) throw std::runtime_error(message); }
LAMMPS_NS::FixMetadynamics& fix(LAMMPS_NS::LAMMPS* simulation) {
    auto* result=dynamic_cast<LAMMPS_NS::FixMetadynamics*>(simulation->modify->get_fix_by_id("m"));
    require(result!=nullptr,"fix m missing"); return *result;
}
LammpsEngineAdapter& adapter(LAMMPS_NS::LAMMPS* simulation) {
    return dynamic_cast<LammpsEngineAdapter&>(*fix(simulation).engine_adapter);
}
bool member(LAMMPS_NS::LAMMPS* simulation,int selection,int local) {
    return selection==all_atoms || (simulation->atom->mask[local]&simulation->group->bitmask[selection]);
}
}
extern "C" const char* engine_neighbors_error() { return last_error.c_str(); }
extern "C" int engine_neighbors_register(void* pointer) {
    try {
        auto* simulation=static_cast<LAMMPS_NS::LAMMPS*>(pointer); auto& engine=adapter(simulation);
        NeighborRequest request; request.cutoff=cutoff;
        reference_handle=engine.neighbors().request(request);
        const int first=simulation->group->find("first"),second=simulation->group->find("second"),
                  overlap=simulation->group->find("overlap"),empty=simulation->group->find("empty");
        require(first>=0 && second>=0 && overlap>=0 && empty>=0,"test selections missing");
        cases={{all_atoms,all_atoms,0},{first,first,0},{first,second,0},{first,overlap,0},
               {first,all_atoms,0},{all_atoms,first,0},{empty,all_atoms,0},{empty,empty,0},
               {all_atoms,0,0}};
        for(auto& test:cases) {
            request.centers=test.centers; request.neighbors=test.neighbors; request.kind=NeighborKind::Half;
            test.handle=engine.neighbors().request(request);
        }
        request.ghost_centers=true;
        bool rejected=false;try { engine.neighbors().request(request); } catch(const std::invalid_argument&) { rejected=true; }
        require(rejected,"Half ghost environment accepted");
        return 0;
    } catch(const std::exception& error) { last_error=error.what();return 1; }
}
extern "C" int engine_neighbors_check(void* pointer,double* maximum_error) {
    try {
        auto* simulation=static_cast<LAMMPS_NS::LAMMPS*>(pointer);auto& engine=adapter(simulation);
        auto* atom=simulation->atom; const int owned=atom->nlocal,total=owned+atom->nghost;
        engine.begin_evaluation(); const auto full=engine.neighbors().prepare(reference_handle);
        require(full.kind==NeighborKind::Full && full.pair_ownership==NeighborPairOwnership::DirectedByCenter,"Full metadata");
        LAMMPS_NS::tagint largest_tag=0;
        for(int local=0;local<owned;++local) largest_tag=std::max(largest_tag,atom->tag[local]);
        long long largest=largest_tag;
        MPI_Allreduce(MPI_IN_PLACE,&largest,1,MPI_LONG_LONG,MPI_MAX,simulation->world);
        require(largest>0 && largest<100000,"small positive fixture IDs required");
        double local_error=0;
        const int rows[9]={0,1,2,0,0,1,1,2,2},columns[9]={0,1,2,1,2,2,0,0,1};
        for(const auto& test:cases) {
            const auto half=engine.neighbors().prepare(test.handle);
            require(half.kind==NeighborKind::Half && half.pair_ownership==NeighborPairOwnership::UniqueAcrossRanks,"Half metadata");
            // 新流从返回视图的事件获取依赖，验证候选上传不会依赖默认流隐式同步。
            cudaStream_t consumer;require(cudaStreamCreateWithFlags(&consumer,cudaStreamNonBlocking)==cudaSuccess,"consumer stream");
            half.neighbors.wait_on(consumer);half.offsets.wait_on(consumer);half.centers.wait_on(consumer);
            std::vector<int> copied(half.neighbors.item_count);
            if(!copied.empty()) require(cudaMemcpyAsync(copied.data(),half.neighbors.data,copied.size()*sizeof(int),
                cudaMemcpyDeviceToHost,consumer)==cudaSuccess,"neighbor download");
            require(cudaStreamSynchronize(consumer)==cudaSuccess,"consumer wait");cudaStreamDestroy(consumer);
            for(size_t entry=0;entry<copied.size();++entry) require(copied[entry]==half.host_neighbors(entry),"stale device neighbors");
            std::vector<double> expected((largest+1)*12,0),forces(total*3,0),virials(total*9,0);
            std::array<double,11> reference{},actual{}; // pair count, energy, W[9]
            const auto evaluate=[&](const NeighborListView& list,bool oracle) {
                const double weight=oracle?.5:1.;auto& totals=oracle?reference:actual;
                for(size_t row=0;row<list.host_centers.item_count;++row) {
                    const int center=list.host_centers(row);require(center<owned,"Half center must be owned");
                    for(int entry=list.host_offsets(row);entry<list.host_offsets(row+1);++entry) {
                        const int neighbor=list.host_neighbors(entry);
                        const bool forward=member(simulation,test.centers,center)&&member(simulation,test.neighbors,neighbor);
                        const bool backward=member(simulation,test.centers,neighbor)&&member(simulation,test.neighbors,center);
                        if(oracle && !(forward||backward)) continue;
                        if(!oracle) require(forward,"Half orientation violates group selection");
                        double delta[3],distance_sq=0;
                        for(int axis=0;axis<3;++axis) { delta[axis]=atom->x[neighbor][axis]-atom->x[center][axis];distance_sq+=delta[axis]*delta[axis]; }
                        if(distance_sq>=cutoff*cutoff) continue;
                        totals[0]+=weight;totals[1]+=.5*weight*distance_sq;
                        for(int endpoint=0;endpoint<2;++endpoint) {
                            const int local=endpoint?neighbor:center;const size_t tag=atom->tag[local];
                            for(int axis=0;axis<3;++axis) {
                                const double force=(endpoint?-weight:weight)*delta[axis];
                                if(oracle) expected[12*tag+axis]+=force;else forces[3*local+axis]+=force;
                            }
                            for(int component=0;component<9;++component) {
                                const double contribution=-.5*weight*delta[rows[component]]*delta[columns[component]];
                                if(oracle) expected[12*tag+3+component]+=contribution;else virials[9*local+component]+=contribution;
                                totals[2+component]+=contribution;
                            }
                        }
                    }
                }
            };
            evaluate(full,true);evaluate(half,false);
            MPI_Allreduce(MPI_IN_PLACE,expected.data(),expected.size(),MPI_DOUBLE,MPI_SUM,simulation->world);
            MPI_Allreduce(MPI_IN_PLACE,reference.data(),11,MPI_DOUBLE,MPI_SUM,simulation->world);
            MPI_Allreduce(MPI_IN_PLACE,actual.data(),11,MPI_DOUBLE,MPI_SUM,simulation->world);
            for(size_t component=0;component<actual.size();++component)
                local_error=std::max(local_error,std::abs(actual[component]-reference[component]));
            host_force_access(simulation);
            std::vector<double> baseline(owned*9);
            require(fix(simulation).a_virial!=nullptr,"fixture must request virial tally");
            for(int local=0;local<owned;++local) {
                for(int axis=0;axis<3;++axis) baseline[9*local+axis]=atom->f[local][axis];
                for(int component=0;component<6;++component) baseline[9*local+3+component]=fix(simulation).a_virial[6*local+component];
            }
            engine.commit_force_virial(make_aos_view<const double>(forces.data(),total,3,forces.size(),MemorySpace::Host),
                {make_aos_view<const double>(virials.data(),total,9,virials.size(),MemorySpace::Host),VirialRepresentation::Full9});
            host_force_access(simulation);
            for(int local=0;local<owned;++local) {
                const size_t tag=atom->tag[local];
                for(int axis=0;axis<3;++axis) {
                    local_error=std::max(local_error,std::abs(atom->f[local][axis]-baseline[9*local+axis]-expected[12*tag+axis]));
                    atom->f[local][axis]=baseline[9*local+axis];
                }
                for(int component=0;component<6;++component) {
                    local_error=std::max(local_error,std::abs(fix(simulation).a_virial[6*local+component]-baseline[9*local+3+component]-expected[12*tag+3+component]));
                    fix(simulation).a_virial[6*local+component]=baseline[9*local+3+component];
                }
            }
            host_force_access(simulation, true);
            // 缓存命中不重建/上传；下一次 run 0 由驱动检查失效重建后的相同物理结果。
            const auto stats=engine.sync_stats();engine.begin_evaluation();
            const auto cached=engine.neighbors().prepare(test.handle);
            require(cached.topology_generation==half.topology_generation &&
                engine.sync_stats().neighbor_rebuilds==stats.neighbor_rebuilds &&
                engine.sync_stats().neighbor_uploads==stats.neighbor_uploads,"Half cache miss without changes");
        }
        MPI_Allreduce(&local_error,maximum_error,1,MPI_DOUBLE,MPI_MAX,simulation->world);
        require(*maximum_error<2e-8,"Half/Full pair count, energy, force or virial mismatch");
        return 0;
    } catch(const std::exception& error) { last_error=error.what();return 1; }
}
