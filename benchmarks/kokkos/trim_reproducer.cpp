// 独立 LAMMPS 邻居请求复现：只请求 Full 表，不计算或修改原子力。
#include "fix.h"
#include "lammps.h"
#include "lammpsplugin.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "utils.h"
#include "version.h"
#include "error.h"
using namespace LAMMPS_NS;
class FixFullCutoffProbe : public Fix {
    double cutoff_;
public:
    FixFullCutoffProbe(LAMMPS* lmp, int narg, char** arg) : Fix(lmp,narg,arg) {
        if (narg != 4) error->all(FLERR,"full/cutoff/probe requires a cutoff");
        cutoff_ = utils::numeric(FLERR,arg[3],false,lmp);
    }
    int setmask() override { return 0; }
    void init() override {
        auto* request = neighbor->add_request(this, NeighConst::REQ_FULL);
        request->set_cutoff(cutoff_);
    }
};
static Fix* create(LAMMPS* lmp,int narg,char** arg) {
    return new FixFullCutoffProbe(lmp,narg,arg);
}
extern "C" void lammpsplugin_init(void* lmp,void* handle,void* callback) {
    lammpsplugin_t plugin{};
    plugin.version=LAMMPS_VERSION;
    plugin.style="fix"; plugin.name="full/cutoff/probe";
    plugin.info="Full neighbor cutoff trim reproducer"; plugin.author="MetaD-toy";
    plugin.creator.v2=reinterpret_cast<lammpsplugin_factory2*>(create);
    plugin.handle=handle;
    reinterpret_cast<lammpsplugin_regfunc>(callback)(&plugin,lmp);
}
