#include "fix_crystallize.h"
#include "zqc_debug.h"

#include "error.h"
#include "comm.h"


std::map<std::string, MetaD_zqc::CVFactory::CreatorFunc>& MetaD_zqc::CVFactory::get_registry() {
    static std::map<std::string, CreatorFunc> registry;
    return registry;
}

void MetaD_zqc::CVFactory::register_cv(std::string name, CreatorFunc func) {
    get_registry()[name] = func;
}

MetaD_zqc::CV* MetaD_zqc::CVFactory::create(std::string name, LAMMPS_NS::LAMMPS* lmp,
                LAMMPS_NS::FixMetadynamics *Fixmetad, 
                int narg, char** arg, int &i, FILE *f_check) {
    if (get_registry().count(name)) {
        return get_registry()[name](lmp, Fixmetad, narg, arg, i, f_check);
    } else {
        LOG("Failed to create this instance: name=%s\n", name.c_str());
        for (auto const& [key, val] : get_registry()) {
            LOG("  We Allowed Registered key: [%s]\n", key.c_str());
        }
        LAMMPS_NS::Error *error = lmp->error;
        ERR_COND((1), "Failed to create this instance: name=%s\n", name.c_str());
    }
    return nullptr; // not found
}