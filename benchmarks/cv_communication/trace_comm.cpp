// 记录实际 LAMMPS Fix 通信宽度及 MPI_Send 数据量，供协议回归使用。
#include "comm_brick.h"
#include "comm_tiled.h"
#include "fix.h"
#include <mpi.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <tuple>

namespace {
struct Counts { unsigned long long calls=0, sends=0, doubles=0; };
using Key = std::tuple<std::string, int, int>;
std::map<Key, Counts> counts;
thread_local Counts* active = nullptr;
unsigned long long barriers = 0;

template<class Comm>
void trace_call(const char* symbol, const char* mode, Comm* self,
                LAMMPS_NS::Fix* fix, int size, int capacity) {
    using Function = void (*)(Comm*, LAMMPS_NS::Fix*, int);
    auto original = reinterpret_cast<Function>(dlsym(RTLD_NEXT, symbol));
    if (!original) { std::fprintf(stderr, "%s\n", dlerror()); std::abort(); }
    auto* previous = active;
    if (std::strcmp(fix->style, "metad") == 0) {
        active = &counts[{mode, size ? size : capacity, capacity}];
        ++active->calls;
    }
    try { original(self, fix, size); }
    catch (...) { active=previous; throw; }
    active=previous;
}
}

namespace LAMMPS_NS {
void CommBrick::forward_comm(Fix* fix, int size) {
    trace_call("_ZN9LAMMPS_NS9CommBrick12forward_commEPNS_3FixEi", "forward", this, fix, size, fix->comm_forward);
}
void CommBrick::reverse_comm(Fix* fix, int size) {
    trace_call("_ZN9LAMMPS_NS9CommBrick12reverse_commEPNS_3FixEi", "reverse", this, fix, size, fix->comm_reverse);
}
void CommTiled::forward_comm(Fix* fix, int size) {
    trace_call("_ZN9LAMMPS_NS9CommTiled12forward_commEPNS_3FixEi", "forward", this, fix, size, fix->comm_forward);
}
void CommTiled::reverse_comm(Fix* fix, int size) {
    trace_call("_ZN9LAMMPS_NS9CommTiled12reverse_commEPNS_3FixEi", "reverse", this, fix, size, fix->comm_reverse);
}
}

extern "C" int MPI_Send(const void* buf, int count, MPI_Datatype datatype,
                         int dest, int tag, MPI_Comm comm) {
    if (active && datatype == MPI_DOUBLE) {
        ++active->sends;
        active->doubles += count;
    }
    return PMPI_Send(buf, count, datatype, dest, tag, comm);
}
extern "C" int MPI_Barrier(MPI_Comm comm) {
    ++barriers;
    return PMPI_Barrier(comm);
}
extern "C" int MPI_Finalize() {
    const char* prefix = std::getenv("ZQC_COMM_TRACE");
    if (prefix) {
        int rank=0; PMPI_Comm_rank(MPI_COMM_WORLD,&rank);
        const auto path = std::string(prefix)+".rank"+std::to_string(rank)+".json";
        FILE* f=std::fopen(path.c_str(),"w");
        if (!f) std::abort();
        std::fprintf(f,"{\"rank\":%d,\"barriers\":%llu,\"records\":[",rank,barriers);
        bool first=true;
        for (const auto& entry:counts) {
            const auto& [mode,width,capacity]=entry.first;
            const auto& c=entry.second;
            std::fprintf(f,"%s{\"direction\":\"%s\",\"width\":%d,\"capacity\":%d,\"calls\":%llu,\"sends\":%llu,\"doubles\":%llu}",
                first?"":",",mode.c_str(),width,capacity,c.calls,c.sends,c.doubles);
            first=false;
        }
        std::fprintf(f,"]}\n"); std::fclose(f);
    }
    return PMPI_Finalize();
}
