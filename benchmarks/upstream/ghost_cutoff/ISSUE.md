**Summary**

A fix requesting a full neighbor list with ghost-center rows and a custom cutoff of 5.0 can receive complete owned rows but truncated ghost rows when the pair-derived ghost cutoff is smaller. Increasing the communication cutoff is insufficient: the target atom is already present locally, but absent from the requested ghost row.

Could you confirm whether custom cutoffs are intended to apply to ghost-center rows? If this combination is unsupported, documenting/rejecting it would also help.

**LAMMPS Version and Platform**

Reproduced with LAMMPS 29 Aug 2024, Ubuntu 22.04 under WSL2, GCC 11, MPI, CPU neighbor style bin. The standalone reproducer below requires no CUDA or application-specific code. I also inspected current develop: NPair::copy_neighbor_info still overrides cutneighsq for cutoff_custom without overriding cutneighghostsq. I have not built the current develop binary.

**Expected Behavior**

For a requested full+ghost list with cutoff 5.0, a target at distance 4.4 should appear in the ghost-center row when both atoms are available in the local+ghost atom arrays. The request header describes cutoff as a “special cutoff distance for this list,” which led us to expect the same requested range for these rows.

**Actual Behavior**

The probe checks one specific center/target pair and its original periodic images. Communication cutoff is 10, skin is 0.35. The pair itself has zero energy and forces.

| pair_style zero cutoff | owned checked/missing | ghost checked/missing |
|---|---|---|
| 4.0 | 1 / 0 | 1 / 1 |
| 5.0 | 1 / 0 | 1 / 0 |

Both executions exit successfully. The target exists in the receiving rank's atom arrays in both cases.

**Steps to Reproduce**

Build the small fix plugin against the same LAMMPS headers/library as the executable:

```sh
mpicxx -std=c++17 -O2 -shared -fPIC \
  -I/path/lammps/src ghost_cutoff_probe.cpp \
  -L/path/lammps/lib -llammps -o probe.so

mpiexec -np 2 lmp -var paircut 4.0 -var plugin /absolute/probe.so -in in.ghost_cutoff
mpiexec -np 2 lmp -var paircut 5.0 -var plugin /absolute/probe.so -in in.ghost_cutoff
```

<details><summary>in.ghost_cutoff (three atoms)</summary>

```lammps
units metal
atom_style atomic
boundary p p p
processors 2 1 1
region box block 0 20 0 20 0 20
create_box 1 box
create_atoms 1 single 10.1 10 10
create_atoms 1 single 9.8 10 10
create_atoms 1 single 14.5 10 10
mass 1 1
pair_style zero ${paircut}
pair_coeff * *
neighbor 0.35 bin
neigh_modify every 1 delay 0 check yes
comm_modify cutoff 10
plugin load ${plugin}
fix probe all ghost/cutoff/probe
run 0

```
</details>

<details><summary>ghost_cutoff_probe.cpp</summary>

```cpp
// Standalone LAMMPS plugin reproducer: custom cutoff on a full list with ghost rows.
#include "fix.h"
#include "atom.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "neigh_list.h"
#include "lammpsplugin.h"
#include "version.h"
#include <cmath>
#include <cstdio>
using namespace LAMMPS_NS;
class GhostCutoffProbe final : public Fix {
    NeighList* list_=nullptr;
public:
    GhostCutoffProbe(LAMMPS* lmp,int narg,char** arg):Fix(lmp,narg,arg) {}
    int setmask() override {return 0;}
    void init() override {
        auto* request=neighbor->add_request(this,NeighConst::REQ_FULL|NeighConst::REQ_GHOST);
        request->set_cutoff(5.0);
    }
    void init_list(int,NeighList* list) override {list_=list;}
    void setup(int) override {
        int counts[4]={0,0,0,0}; // owned checked/missing, ghost checked/missing
        for (int row=0;row<list_->inum+list_->gnum;++row) {
            const int center=list_->ilist[row];
            if (atom->tag[center]!=1 || (std::abs(atom->x[center][0]-10.1)>1e-8 || std::abs(atom->x[center][1]-10)>1e-8 || std::abs(atom->x[center][2]-10)>1e-8)) continue;
            for (int target=0;target<atom->nlocal+atom->nghost;++target) {
                if (atom->tag[target]!=3 || (std::abs(atom->x[target][0]-14.5)>1e-8 || std::abs(atom->x[target][1]-10)>1e-8 || std::abs(atom->x[target][2]-10)>1e-8)) continue;
                double distance_sq=0;
                for(int axis=0;axis<3;++axis) {
                    const double delta=atom->x[center][axis]-atom->x[target][axis];
                    distance_sq+=delta*delta;
                }
                if(distance_sq>=25.0) continue;
                const int offset=center<atom->nlocal ? 0 : 2;
                ++counts[offset];
                bool found=false;
                for(int index=0;index<list_->numneigh[center];++index)
                    found|=(list_->firstneigh[center][index]&NEIGHMASK)==target;
                if(!found) ++counts[offset+1];
            }
        }
        int total[4]={0,0,0,0};
        MPI_Allreduce(counts,total,4,MPI_INT,MPI_SUM,world);
        if(comm->me==0)
            printf("CUSTOM_GHOST_PROBE owned_checked=%d owned_missing=%d ghost_checked=%d ghost_missing=%d\n",
                   total[0],total[1],total[2],total[3]);
    }
};
static Fix* create_probe(LAMMPS* lmp,int narg,char** arg) {return new GhostCutoffProbe(lmp,narg,arg);}
extern "C" void lammpsplugin_init(void* lmp,void* handle,void* register_function) {
    lammpsplugin_t plugin{};
    plugin.version=LAMMPS_VERSION;plugin.style="fix";plugin.name="ghost/cutoff/probe";
    plugin.info="Custom cutoff ghost-row reproducer";plugin.author="ZQC";
    plugin.creator.v2=reinterpret_cast<lammpsplugin_factory2*>(create_probe);plugin.handle=handle;
    reinterpret_cast<lammpsplugin_regfunc>(register_function)(&plugin,lmp);
}

```
</details>

**Code pointers / possible direction**

- [stable_29Aug2024 NPair::copy_neighbor_info](https://github.com/lammps/lammps/blob/stable_29Aug2024/src/npair.cpp#L104-L116) replaces the owned-row cutoff array with the custom value.
- [Neighbor custom cutoff validation](https://github.com/lammps/lammps/blob/stable_29Aug2024/src/neighbor.cpp#L854-L855) checks communication coverage.
- [NPairBinGhost](https://github.com/lammps/lammps/blob/stable_29Aug2024/src/npair_bin_ghost.cpp) uses cutneighghostsq for ghost centers.
- [NeighRequest cutoff field](https://github.com/lammps/lammps/blob/stable_29Aug2024/src/neigh_request.h#L86-L87).
- The corresponding override is still present in [current develop npair.cpp](https://github.com/lammps/lammps/blob/develop/src/npair.cpp), with the ghost array continuing to point to the Neighbor data.

If supporting this combination is appropriate, one possible direction is a request-local ghost-cutoff array using max(custom cutoff squared, existing per-type ghost cutoff squared). Using max rather than direct assignment would preserve a larger ghostneigh cutoff needed by a many-body potential. This would need review of request matching/copying and the other ghost-list builders.

I found #4437 concerning a different custom-cutoff/type-pair case; this report specifically concerns ghost-center rows. I am opening an issue before a PR to confirm the intended contract. If maintainers favor support, a regression test could request full+ghost with a custom cutoff and check that an available target within the requested radius appears in the ghost row.

