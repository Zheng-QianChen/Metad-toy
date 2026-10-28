// 真实适配器批次回归：独立按全局原子 ID 汇总各 ghost 副本的贡献。
// 同时检查延迟写回、取消、空批次、输入复用和一次 reverse；每个用例恢复原宿主力。
#include "lammps.h"
#include "atom.h"
#include "modify.h"
#include "fix_crystallize.h"
#include "zqc_lammps_engine.h"
#include "zqc_gpu_buffer.cuh"
#ifdef ZQC_LAMMPS_KOKKOS
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "kokkos.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <mpi.h>

using namespace MetaD_zqc;
namespace {
thread_local std::string last_error;
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template<class Function> void rejects(Function action) {
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "invalid batch lifecycle accepted");
}
LAMMPS_NS::FixMetadynamics& get_fix(LAMMPS_NS::LAMMPS* simulation, const char* name) {
    auto* fix=dynamic_cast<LAMMPS_NS::FixMetadynamics*>(simulation->modify->get_fix_by_id(name));
    require(fix != nullptr, "MetaD fix missing");
    return *fix;
}
LammpsEngineAdapter& get_engine(LAMMPS_NS::FixMetadynamics& fix) {
    return dynamic_cast<LammpsEngineAdapter&>(*fix.engine_adapter);
}
void host_force_access(LAMMPS_NS::LAMMPS* simulation, bool modified=false) {
#ifdef ZQC_LAMMPS_KOKKOS
    if (simulation->kokkos) {
        auto* atom=static_cast<LAMMPS_NS::AtomKokkos*>(simulation->atom);
        if (modified) atom->modified(LAMMPS_NS::Host,F_MASK);
        else atom->sync(LAMMPS_NS::Host,F_MASK);
    }
#endif
}
}
extern "C" const char* force_batch_error() { return last_error.c_str(); }
extern "C" int force_batch_stats(void* pointer, const char* name, std::uint64_t* output) {
    try {
        const auto stats=get_engine(get_fix(static_cast<LAMMPS_NS::LAMMPS*>(pointer),name)).sync_stats();
        output[0]=stats.force_virial_submissions;
        output[1]=stats.force_reverse_calls;
        output[2]=stats.force_virial_flushes;
        return 0;
    } catch (const std::exception& error) { last_error=error.what(); return 1; }
}
extern "C" int force_batch_check(void* pointer, const char* name, int full9, int soa,
                                  int cuda_memory, double* maximum_error) {
    try {
        auto* simulation=static_cast<LAMMPS_NS::LAMMPS*>(pointer);
        auto& fix=get_fix(simulation,name);
        auto& engine=get_engine(fix);
        auto* atom=simulation->atom;
        const size_t owned=atom->nlocal, total=owned+atom->nghost;
        const size_t components=full9?9:6, slots=total+3;
        require(fix.a_virial != nullptr,"fixture must request virial");
        int rank=0, ghosts=atom->nghost;
        MPI_Comm_rank(simulation->world,&rank);
        MPI_Allreduce(MPI_IN_PLACE,&ghosts,1,MPI_INT,MPI_SUM,simulation->world);
        require(ghosts>0,"fixture must contain ghost images");
        long long largest_id=0;
        for(size_t local=0;local<total;++local)
            largest_id=std::max(largest_id,static_cast<long long>(atom->tag[local]));
        MPI_Allreduce(MPI_IN_PLACE,&largest_id,1,MPI_LONG_LONG,MPI_MAX,simulation->world);
        require(largest_id>0 && largest_id<100000,"small fixture required");

        host_force_access(simulation);
        std::vector<double> baseline(owned*9), expected((largest_id+1)*9,0);
        for(size_t local=0;local<owned;++local) {
            for(int axis=0;axis<3;++axis) baseline[9*local+axis]=atom->f[local][axis];
            for(int axis=0;axis<6;++axis) baseline[9*local+3+axis]=fix.a_virial[6*local+axis];
        }
        // 即使断言失败，也恢复真实 MD 输出，取消尚未提交的公共缓冲。
        struct Restore {
            LAMMPS_NS::LAMMPS* simulation;
            LAMMPS_NS::FixMetadynamics& fix;
            LammpsEngineAdapter& engine;
            const std::vector<double>& baseline;
            ~Restore() {
                engine.cancel_force_virial_batch();
                for(int local=0;local<simulation->atom->nlocal;++local) {
                    for(int axis=0;axis<3;++axis)
                        simulation->atom->f[local][axis]=baseline[9*local+axis];
                    for(int axis=0;axis<6;++axis)
                        fix.a_virial[6*local+axis]=baseline[9*local+3+axis];
                }
                host_force_access(simulation,true);
            }
        } restore{simulation,fix,engine,baseline};
        auto unchanged=[&] {
            host_force_access(simulation);
            for(size_t local=0;local<owned;++local) {
                for(int axis=0;axis<3;++axis)
                    require(atom->f[local][axis]==baseline[9*local+axis],"force written before batch finish");
                for(int axis=0;axis<6;++axis)
                    require(fix.a_virial[6*local+axis]==baseline[9*local+3+axis],"virial written before batch finish");
            }
        };
        std::vector<double> forces(3*slots,-999), virials(components*slots,-777);
        const auto force_offset=[&](size_t local,size_t axis) { return soa?local+axis*slots:3*local+axis; };
        const auto virial_offset=[&](size_t local,size_t axis) { return soa?local+axis*slots:components*local+axis; };
        for(size_t local=0;local<total;++local) {
            const size_t id=atom->tag[local];
            const double scale=(id+local+3*rank)*.125;
            for(size_t axis=0;axis<3;++axis) {
                forces[force_offset(local,axis)]=scale+axis;
                expected[9*id+axis]+=2*(scale+axis);
            }
            for(size_t axis=0;axis<components;++axis) {
                virials[virial_offset(local,axis)]=scale+10+axis;
                if(axis<6) expected[9*id+3+axis]+=2*(scale+10+axis);
            }
        }
        GpuBuffer<double> device_forces,device_virials;
        struct Stream {
            cudaStream_t value=nullptr;
            Stream() { require(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking)==cudaSuccess,"create producer stream"); }
            ~Stream() { cudaStreamDestroy(value); }
        } producer;
        if(cuda_memory) {
            device_forces.grow_to(forces.size(),__FILE__,__LINE__);
            device_virials.grow_to(virials.size(),__FILE__,__LINE__);
            device_forces.upload_from(forces.data(),forces.size());
            device_virials.upload_from(virials.data(),virials.size());
        }
        const auto memory=cuda_memory?MemorySpace::Cuda:MemorySpace::Host;
        ArrayView<const double> force_view{cuda_memory?device_forces.ptr:forces.data(),total,3,
            soa?1:size_t(3),soa?slots:1,forces.size(),memory};
        VirialView virial_view{{cuda_memory?device_virials.ptr:virials.data(),total,components,
            soa?1:components,soa?slots:1,virials.size(),memory},
            full9?VirialRepresentation::Full9:VirialRepresentation::Symmetric6};
        if(cuda_memory) {
            device_forces.record_write(producer.value);
            device_virials.record_write(producer.value);
            force_view.readiness=device_forces.readiness();
            virial_view.values.readiness=device_virials.readiness();
        }

        const auto before_empty=engine.sync_stats();
        engine.begin_force_virial_batch();
        rejects([&]{ engine.begin_force_virial_batch(); });
        rejects([&]{ engine.begin_evaluation(); });
        rejects([&]{ engine.invalidate_topology(); });
        engine.finish_force_virial_batch();
        require(engine.sync_stats().force_virial_flushes==before_empty.force_virial_flushes,
                "empty batch flushed");
        rejects([&]{ engine.finish_force_virial_batch(); });
        engine.begin_force_virial_batch();
        engine.commit_force_virial(force_view,virial_view);
        unchanged();
        engine.cancel_force_virial_batch();
        require(engine.sync_stats().force_reverse_calls==before_empty.force_reverse_calls,
                "cancelled batch communicated");
        unchanged();

        const auto before=engine.sync_stats();
        engine.begin_force_virial_batch();
        auto invalid=virial_view;
        invalid.values.item_count=total+1;
        rejects([&]{ engine.commit_force_virial(force_view,invalid); });
        if(cuda_memory==2) {
            auto host_force=force_view;
            auto host_virial=virial_view;
            host_force.data=forces.data();host_force.memory=MemorySpace::Host;host_force.readiness={};
            host_virial.values.data=virials.data();host_virial.values.memory=MemorySpace::Host;host_virial.values.readiness={};
            engine.commit_force_virial(host_force,host_virial);
        } else engine.commit_force_virial(force_view,virial_view);
        engine.commit_force_virial(force_view,virial_view);
        const double single_force[3]={2,3,5};
        const double single_virial[9]={7,11,13,17,19,23,29,31,37};
        if(owned) {
            engine.add_owned_force_virial(0,single_force,
                {{single_virial,1,9,9,1,9,MemorySpace::Host},VirialRepresentation::Full9});
            for(int axis=0;axis<3;++axis) expected[9*atom->tag[0]+axis]+=single_force[axis];
            for(int axis=0;axis<6;++axis) expected[9*atom->tag[0]+3+axis]+=single_virial[axis];
        }
        unchanged();
        // 源数据允许在 finish 前复用，批次必须已经取得独立贡献。
        if(cuda_memory) {
            // No default-stream download/fence here: an independent producer can
            // overwrite its source immediately when commit returns.
            device_forces.clear_async(producer.value);
            device_virials.clear_async(producer.value);
        } else {
            std::fill(forces.begin(),forces.end(),0);
            std::fill(virials.begin(),virials.end(),0);
        }
        engine.finish_force_virial_batch();
        const auto after=engine.sync_stats();
        require(after.force_virial_submissions-before.force_virial_submissions==2,"submission count");
        require(after.force_reverse_calls-before.force_reverse_calls==1,"batch must reverse exactly once");
        require(after.force_virial_flushes-before.force_virial_flushes==1,"batch must write exactly once");

        MPI_Allreduce(MPI_IN_PLACE,expected.data(),expected.size(),MPI_DOUBLE,MPI_SUM,simulation->world);
        host_force_access(simulation);
        double error=0;
        for(size_t local=0;local<owned;++local) {
            const size_t id=atom->tag[local];
            for(int axis=0;axis<3;++axis)
                error=std::max(error,std::abs(atom->f[local][axis]-baseline[9*local+axis]-expected[9*id+axis]));
            for(int axis=0;axis<6;++axis)
                error=std::max(error,std::abs(fix.a_virial[6*local+axis]-baseline[9*local+3+axis]-expected[9*id+3+axis]));
        }
        MPI_Allreduce(&error,maximum_error,1,MPI_DOUBLE,MPI_MAX,simulation->world);
        require(*maximum_error<1e-8,"batch force/virial sum differs from ID oracle");
        return 0;
    } catch(const std::exception& error) { last_error=error.what(); return 1; }
}
