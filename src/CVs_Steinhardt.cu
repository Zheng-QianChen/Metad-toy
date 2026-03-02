#include "fix_crystallize.h"

#include "lammpsplugin.h"

#include "lammps.h"
#include "update.h"
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "command.h"
#include "domain.h"
#include "force.h"
#include "group.h"
#include "version.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"         // 完整定义Neighbor类
#include "neigh_list.h"        // 定义NeighList结构
#include "pair.h"

#include "zqc_debug.h"
#include "zqc_CVs.h"

#include <cuda_runtime.h>
#include <cstring>
#include <algorithm>

using namespace LAMMPS_NS;

MetaD_zqc::Steinhardt::~Steinhardt() {
    if (atoms != nullptr) {
        delete[] atoms; // 假设 atoms 是用 new[] 分配的
    }
}
    
    
MetaD_zqc::Steinhardt* MetaD_zqc::create_steinhardt_cv(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check, int group_id, int Q_num, 
                                char *Q_type_str, double cutoff_r, int cutoff_Natoms, 
                                int d_block_size) 
{
    if (strcmp(Q_type_str, "Q") == 0){
        if (Q_num==4){
            return new MetaD_zqc::STEIN_Q4(lmp, Fixmetad, f_check, group_id, cutoff_r, cutoff_Natoms, d_block_size);
        } else if (Q_num==6){
            return new MetaD_zqc::STEIN_Q6(lmp, Fixmetad, f_check, group_id, cutoff_r, cutoff_Natoms, d_block_size);
        }
    } else if (strcmp(Q_type_str,"L")){
        if (Q_num==4){
            // return new STEIN_LQ4(lmp, f_check, group_id, cutoff_r, cutoff_Natoms, d_block_size);
        } else if (Q_num==6){
            // return new STEIN_LQ6(lmp, f_check, group_id, cutoff_r, cutoff_Natoms, d_block_size);
        }
    }
    return nullptr;
}