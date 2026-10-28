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
