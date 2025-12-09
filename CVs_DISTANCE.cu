#include "fix_crystallize.h"
#include "zqc_CVs.h"
#include "zqc_debug.h"

#include "lammpsplugin.h"
#include "atom.h"
#include "comm.h"
#include "update.h"
#include "domain.h"
#include "memory.h"
#include "error.h"
#include "force.h"
#include "group.h"



MetaD_zqc::Distance::~Distance(){
  delete[] dcvdx;
  // delete[] dVdcv;
}

MetaD_zqc::Distance::Distance(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::bigint id1, LAMMPS_NS::bigint id2, FILE* f_check):CV(lmp, f_check), atom_id1(id1), atom_id2(id2) {
    // DEBUG_COND_LOG(lmp->domain == nullptr, "Domain not initialized when creating Distance CV.");
    LAMMPS_NS::Domain *domain = lmp->domain;
    // double **x = lmp->atom->x;
    // DEBUG_LOG("Debug: Distance for lmp-> x[0][0]: %f",x[0][0]);
    pbc_x = (domain->xperiodic == 1);
    pbc_y = (domain->yperiodic == 1);
    pbc_z = (domain->zperiodic == 1);
    DEBUG_LOG("Debug:The pbc settings in 3 axis is (0 for non-p, 1 for periodic): x-%d y-%d z-%d",pbc_x, pbc_y, pbc_z);
    dcvdx = new double[3];
}

void MetaD_zqc::Distance::summary(FILE *f){
  fprintf(f, "CV summary: %d, %d\n", this->atom_id1, this->atom_id2);
  fflush(f);
}


void MetaD_zqc::Distance::delta_x(){
  double **x = lmp->atom->x;
  double xbox,ybox,zbox;
  dx = x[atom_id2][0] - x[atom_id1][0];
  dy = x[atom_id2][1] - x[atom_id1][1];
  dz = x[atom_id2][2] - x[atom_id1][2];
  if(pbc_x){
    xbox = lmp->domain->xprd;
    if (dx > xbox/2) {
        dx -= xbox;
    } else if (dx < -xbox/2) {
        dx += xbox;
    }
  }
  if(pbc_y){
    ybox = lmp->domain->yprd;
    if (dy > ybox/2) {
        dy -= ybox;
    } else if (dy < -ybox/2) {
        dy += ybox;
    }
  }
  if(pbc_z){
    zbox = lmp->domain->zprd;
    if (dz > zbox/2) {
        dz -= zbox;
    } else if (dz < -zbox/2) {
        dz += zbox;
    }
  }
  // DEBUG_LOG("xbox: %g, %g, %g",xbox,ybox,zbox);
  DEBUG_LOG("atom[0]: %g, %g, %g",x[atom_id1][0],x[atom_id1][1],x[atom_id1][2]);
  DEBUG_LOG("atom[1]: %g, %g, %g",x[atom_id2][0],x[atom_id2][1],x[atom_id2][2]);
  DEBUG_LOG("PBC dx, dy, dz  = %.6f, %.6f, %.6f", dx, dy, dz);
}

// 归约 cv_values 到所有节点
double MetaD_zqc::Distance::compute_cv() {
    this->delta_x();
    cv_value = sqrt(dx*dx + dy*dy + dz*dz);
    return cv_value;
}

void MetaD_zqc::Distance::bias_force(double dVdcv) {
    DEBUG_LOG("MetaD_zqc::Distance::bias_force");
    double **f = lmp->atom->f;
    double **x = lmp->atom->x;
    this->get_dcvdx(cv_value, dcvdx);
    DEBUG_LOG("cv_value = %g, dVdcv = %g, dcvdx = %g, %g, %g",cv_value, dVdcv, dcvdx[0], dcvdx[1], dcvdx[2]);
    DEBUG_LOG("dx, dy, dz  = %.6f, %.6f, %.6f", x[atom_id2][0]-x[atom_id1][0], x[atom_id2][1]-x[atom_id1][1], x[atom_id2][2]-x[atom_id1][2]);
    DEBUG_LOG("fx0,fy0,fz0  = %.6f, %.6f, %.6f", f[atom_id1][0], f[atom_id1][1], f[atom_id1][2]);
    if ((f[atom_id1][0] + f[atom_id1][1] + f[atom_id1][2]) > 1e-12) {
      f[atom_id1][0] += dVdcv*dcvdx[0];
      f[atom_id1][1] += dVdcv*dcvdx[1];
      f[atom_id1][2] += dVdcv*dcvdx[2];
      f[atom_id2][0] -= dVdcv*dcvdx[0];
      f[atom_id2][1] -= dVdcv*dcvdx[1];
      f[atom_id2][2] -= dVdcv*dcvdx[2];
    }
    DEBUG_LOG("fx,fy,fz  = %.6f, %.6f, %.6f", f[atom_id1][0], f[atom_id1][1], f[atom_id1][2]);
    DEBUG_LOG("post_force_r_end");
}

void MetaD_zqc::Distance::get_dcvdx(double value, double *dcvdx){
  DEBUG_LOG("get_dcvdx");
  double **x = lmp->atom->x;
  dcvdx[0] = dx/value;
  dcvdx[1] = dy/value;
  dcvdx[2] = dz/value;
  DEBUG_LOG("get_dcvdx_end");
}
