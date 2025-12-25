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

__global__ void fix_crystallizes_kernel
(
    int cutoff_Natoms, double cutoff_rsq, double box_x, double box_y, double box_z,
    int group_count, int *d_group_indices, LAMMPS_NS::tagint *d_group_numneigh,
    int *d_firstneigh_ptrs, double *d_x_flat,
    double *d_group_dminneigh, int *d_neigh_in_cutoff_r, int *d_neigh_both_in_r_N,
    LAMMPS_NS::tagint *d_calculated_numneigh
);
__global__ void steinhardt_param_calc_kernel_q4(
    int group_count, int cutoff_Natoms,
    int *d_neigh_both_in_r_N, double *d_group_dminneigh,
    double *d_stein_qlm, double *d_stein_Ylm,
    double *d_stein_ql
);
__global__ void steinhardt_param_calc_kernel_q6();

// void dcv_steinhardt_param_calc_kernel_q4(
//     FILE *f_check,
//     int cutoff_Natoms, 
//     int group_count, int groupbit, int *d_mask,
//     LAMMPS_NS::tagint *d_group_indices, LAMMPS_NS::tagint *calculated_numneigh, 
//     int *d_neigh_both_in_r_N, double *d_group_dminneigh,
//     double *d_stein_qlm, double *d_stein_Ylm, double *d_stein_ql,
//     double *d_dYlm_dr,double *d_dcvdx
// );
__global__ void dcv_steinhardt_param_calc_kernel_q4(
    int cutoff_Natoms, 
    int group_count, int groupbit, int *d_mask,
    LAMMPS_NS::tagint *d_group_indices, LAMMPS_NS::tagint *calculated_numneigh, 
    int *d_neigh_both_in_r_N, double *d_group_dminneigh,
    double *d_stein_qlm, double *d_stein_Ylm, double *d_stein_ql,
    double *d_dYlm_dr,double *d_dcvdx
);


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
            // return new STEIN_Q6(lmp, f_check, group_id, cutoff_r, cutoff_Natoms, d_block_size);
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


MetaD_zqc::STEIN_Q4::STEIN_Q4(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::FixMetadynamics *Fixmetad, FILE *f_check, int group_id, 
                             double cutoff_r, int cutoff_Natoms, int d_block_size)
    : Steinhardt(lmp, f_check),
      Fixmetad(Fixmetad),
      group_id(group_id),
      cutoff_r(cutoff_r),
      cutoff_Natoms(cutoff_Natoms), 
      d_block_size(d_block_size)
{
    pbc_x = (lmp->domain->xperiodic == 1);
    pbc_y = (lmp->domain->yperiodic == 1);
    pbc_z = (lmp->domain->zperiodic == 1);
    DEBUG_LOG("Logging: New a Stein_Q4 file, will generate %d lines in GPU,\n     with cutoff_r=%g, cutoff_Natoms=%d",
                d_block_size, cutoff_r, cutoff_Natoms);
    // gpu device settings
    cudaGetLastError(); // clear history error
    GPU_number = 0;
    cudaGetDevice(&GPU_number);
    DEBUG_LOG("GPU_number is %d",GPU_number);

    // const char *group_name = arg[1];
    groupbit = lmp->group->bitmask[group_id]; // 关键：存储原子组位掩码
    group_dminneigh = new double [2]; //inintial
    neigh_in_cutoff_r = new int [2]; //inintial
    neigh_both_in_r_N = new int [2]; //inintial
    // Q_per_atoms_value = new double [2]; //inintial
    stein_q = new double *[1];
    stein_q[0] = nullptr;
    file = f_check;
    error = lmp->error;

    // // default values
    // cutoff_r = 8;
    // cutoff_Natoms = 12;
    // d_block_size = 128;
}

MetaD_zqc::STEIN_Q4::~STEIN_Q4(){
    atoms = nullptr;
    // release all alloc
    delete[] group_dminneigh;
    delete[] neigh_in_cutoff_r;
    delete[] neigh_both_in_r_N;
    // delete[] Q_per_atoms_value;
}

double MetaD_zqc::STEIN_Q4::compute_cv(){
    // =======更新一下邻居列表位置防止报错=========
    nlist = Fixmetad->listfull;
    ERR_COND((nlist == nullptr),"STEIN_Q4 CV failed to find neighbor list now.");
    // DEBUG_LOG("rebuilds = %d", lmp->neighbor->lastcall);
    // DEBUG_LOG("now = %d", lmp->update->ntimestep);
    // DEBUG_LOG("rebuilds_fir = %p", nlist->firstneigh);
    // DEBUG_LOG("rebuilds_num = %p", nlist->numneigh);
    // =======防止lammps运行过程体积更改==========
    ERR_COND((lmp->domain == NULL),"domain list not initialized");
    box_x = (pbc_x) ? lmp->domain->xprd : INFINITY;
    box_y = (pbc_y) ? lmp->domain->yprd : INFINITY;
    box_z = (pbc_z) ? lmp->domain->zprd : INFINITY;
    // =======接受邻居更新消息,进行与设备端通信===========
    if ((lmp->update->ntimestep > lmp->neighbor->lastcall)&&(lmp->update->ntimestep != 1)){
        DEBUG_LOG("rebuilds = %d", lmp->neighbor->lastcall);
        DEBUG_LOG("now = %d", lmp->update->ntimestep);
        ERR_COND((h_group_indices == nullptr),"h_group_indices is nullptr.");
        DEBUG_LOG("h_group_indices=%p",h_group_indices);
    } else {
        // ===重建邻居列表后重新查找local中的目标原子=======
        // clear the h_group_indices
        delete[] h_group_indices;
        h_group_indices = nullptr;
        DEBUG_LOG("free h_group_indices");
        atom = lmp->atom;
        mask = atom->mask;     // 原子组掩码
        h_group_indices = new int [(atom->nlocal)];
        // group_count = how many aim atoms in local
        group_count = 0; // 当前local中有
        for (int i = 0; i < atom->nlocal; i++) {
            if (mask[i] & groupbit){
                h_group_indices[group_count] = i; // record local index
                group_count++;
            }
        }
        // set up nvidia thread number
        block_num = (group_count + d_block_size - 1)/d_block_size;
        N = d_block_size*block_num;
        // stein_q for all aim atoms
        delete[] stein_q[0];
        stein_q[0] = nullptr;
        stein_q[0] = new double[group_count];
        DEBUG_LOG("d_block_size is %d, block_num is %d",d_block_size, block_num);
        LOG_COND((group_count<cutoff_Natoms),"Warning: group_count < cutoff_Natoms, please check your system !");
        LOG_COND(((box_x<2*cutoff_r)||(box_y<2*cutoff_r)||(box_z<2*cutoff_r)),"Warning: box < cutoff_r, please check your system !");
    }

    // 2. calculate atoms' envioronment
    DEBUG_LOG("envioronment function in");
    envioronment();
    DEBUG_LOG("envioronment function out");

    // 3. calculate atoms' other things
    // steinhardt_param(Q_hybrid);
    steinhardt_param_calc(stein_q[0]);

    DEBUG_LOG_COND((group_dminneigh == NULL),"group_dminneigh list not initialized");
    DEBUG_LOG("group_dminneigh Allocated at: %p", group_dminneigh);
    
    // 输出邻居
    // for (int ii=0; ii<group_count; ii++){
    //     for (int jj=0; jj<1; jj++){
    //         fprintf(file, "c_atom_idx=%d,%d,%d : Nx:%f Ny:%f Nz:%f r2:%f\n", 
    //                 atom->tag[h_group_indices[ii]],
    //                 neigh_in_cutoff_r[ii],
    //                 neigh_both_in_r_N[ii],
    //                 group_dminneigh[ii*cutoff_Natoms*4 + jj*4 + 0],
    //                 group_dminneigh[ii*cutoff_Natoms*4 + jj*4 + 1],
    //                 group_dminneigh[ii*cutoff_Natoms*4 + jj*4 + 2],
    //                 group_dminneigh[ii*cutoff_Natoms*4 + jj*4 + 3]);
    //     }
    // }
    DEBUG_RUN(for(int c_atom=0;c_atom<group_count;c_atom++)
                {
                    DEBUG_LOG("stein_ql[%d] = %f",c_atom,stein_q[0][c_atom]);
                });

    DEBUG_LOG("post_force function end");

    if (1) {
        double q4_ave=0;
        for (int c_atom=0;c_atom<group_count;c_atom++){
            q4_ave += stein_q[0][c_atom];
        }
        q4_ave = q4_ave/group_count;
        cv_value = q4_ave;
        return cv_value;
    }
}

void MetaD_zqc::STEIN_Q4::bias_force(double dVdcv)
{
    // pass
    DEBUG_LOG("MetaD_zqc::STEIN_Q4::bias_force");
    double **f = lmp->atom->f;
    double **x = lmp->atom->x;
    int c_tag;
    this->get_dcvdx(cv_value, h_dcvdx);
    // DEBUG_LOG("cv_value = %g, dVdcv = %g, dcvdx = %g, %g, %g",cv_value, dVdcv, dcvdx[0], dcvdx[1], dcvdx[2]);
    // DEBUG_LOG("fx0,fy0,fz0  = %.6f, %.6f, %.6f", f[c_tag][0], f[c_tag][1], f[c_tag][2]);
    for (int c_atom=0; c_atom<group_count; c_atom++){
        DEBUG_LOG("dcvdx, dcvdy, dcvdz  = %g, %g, %g", h_dcvdx[c_atom*3 + 0], h_dcvdx[c_atom*3 + 1], h_dcvdx[c_atom*3 + 2]);
        DEBUG_LOG("dVdcv  = %g", dVdcv);
        c_tag = h_group_indices[c_atom];
        DEBUG_LOG("fx0,fy0,fz0  = %g, %g, %g", f[c_tag][0], f[c_tag][1], f[c_tag][2]);
        f[c_tag][0] -= dVdcv*h_dcvdx[c_atom*3 + 0];
        f[c_tag][1] -= dVdcv*h_dcvdx[c_atom*3 + 1];
        f[c_tag][2] -= dVdcv*h_dcvdx[c_atom*3 + 2];
        DEBUG_LOG("fx,fy,fz  = %g, %g, %g", f[c_tag][0], f[c_tag][1], f[c_tag][2]);
    }
    DEBUG_LOG("post_force_r_end");
}

void MetaD_zqc::STEIN_Q4::get_dcvdx(double cv_value, double *dcvdx)
{
    // DEBUG_RUN(
    delete[] h_stein_qlm;
    double *h_stein_qlm = (double *)malloc(group_count * (stein_l + 1) * 2 * sizeof(double));
    SAFE_CUDA_MEMCPY(h_stein_qlm, d_stein_qlm,
      (group_count*(stein_l + 1)*2)*sizeof(double), cudaMemcpyDeviceToHost,file);
    // );
    SAFE_CUDA_FREE(d_mask);
    SAFE_CUDA_MALLOC(&d_mask, (group_count)*sizeof(int), file);
    SAFE_CUDA_MEMCPY(d_mask,mask,(group_count)*sizeof(int),cudaMemcpyHostToDevice,file);

    delete[] h_dcvdx;
    h_dcvdx = nullptr;
    h_dcvdx = (double *)malloc((group_count*3)*sizeof(double));
    SAFE_CUDA_FREE(d_dcvdx);
    SAFE_CUDA_MALLOC(&d_dcvdx, (group_count*3)*sizeof(double), file);
    SAFE_CUDA_MEMCPY(d_dcvdx,h_dcvdx,(group_count*3)*sizeof(double),cudaMemcpyHostToDevice,file);

    delete[] h_dYlm_dr;
    h_dYlm_dr = nullptr;
    h_dYlm_dr = (double *)malloc((group_count*3*2)*sizeof(double));
    SAFE_CUDA_FREE(d_dYlm_dr);
    SAFE_CUDA_MALLOC(&d_dYlm_dr, (group_count)*3*2*sizeof(double), file);
    // SAFE_CUDA_MEMCPY(d_dYlm_dr,h_dYlm_dr,(group_count)*3*2*sizeof(double),cudaMemcpyHostToDevice,file);


    // dcv_steinhardt_param_calc_kernel_q4(
    //     file, cutoff_Natoms, group_count, groupbit,
    //     mask, h_group_indices, calculated_numneigh,
    //     neigh_both_in_r_N, group_dminneigh,
    //     h_stein_qlm, h_stein_Ylm, stein_q[0],
    //     h_dYlm_dr, h_dcvdx);
    DEBUG_LOG("i will start a kernel of q4");
    cudaDeviceSynchronize(); // waiting memory
    dcv_steinhardt_param_calc_kernel_q4<<<block_num,d_block_size>>>(
        cutoff_Natoms, group_count, groupbit,
        d_mask, d_group_indices, d_calculated_numneigh,
        d_neigh_both_in_r_N, d_group_dminneigh,
        d_stein_qlm, d_stein_Ylm,  d_stein_ql,
        d_dYlm_dr, d_dcvdx);
    cudaDeviceSynchronize(); // waiting memory
    DEBUG_LOG("i am out");

    cudaMemcpy(h_dcvdx, d_dcvdx, (group_count*3)*sizeof(double), cudaMemcpyDeviceToHost);
    // SAFE_CUDA_MEMCPY(h_dcvdx, d_dcvdx,
    //   (group_count*3)*sizeof(double), cudaMemcpyDeviceToHost, file);
    cudaDeviceSynchronize(); // waiting memory
    DEBUG_LOG("1");

}



void MetaD_zqc::STEIN_Q4::envioronment()
{
    DEBUG_LOG("1");
    if ((lmp->update->ntimestep > lmp->neighbor->lastcall)&&(lmp->update->ntimestep != 1)&&((numneigh != nullptr))){
        DEBUG_LOG("we skip rebuild in envioronment when %d.", lmp->neighbor->lastcall);
    } else {
        // =========================================================================
        // neighbour list and its copy to devise
        // h_group_indices / d_group_indices: where the group atoms in locals' tag
        // =========================================================================
        DEBUG_LOG("cutoff_Natoms is %d",cutoff_Natoms);
        DEBUG_LOG("cutoff_r is %f",cutoff_r);
        DEBUG_LOG("group_count is %d",group_count);
        // DEBUG_LOG("lastcall = %d", lmp->neighbor->lastcall);
        SAFE_CUDA_FREE(d_group_indices);
        // int *d_group_indices;
        SAFE_CUDA_MALLOC(&d_group_indices, (group_count)*sizeof(int), file);
        SAFE_CUDA_MEMCPY(d_group_indices,h_group_indices,(group_count)*sizeof(int),cudaMemcpyHostToDevice,file);
        // alloc
        DEBUG_LOG_COND((d_group_indices == NULL),"d_group_indices list not initialized");
        DEBUG_LOG("h_group_indices list %d" ,h_group_indices[0]);
        // =========================================================================
        // h_group_numneigh / d_group_numneigh :
        //      flatten index of the neighbour list. such as we have 20 neighbour
        //      for atom 1, then the list will be : [0, 20, ...]
        // =========================================================================
        // int *numneigh = nlist->numneigh;
        // int **firstneigh = nlist->firstneigh;
        numneigh = nlist->numneigh;
        firstneigh = nlist->firstneigh;
        DEBUG_LOG_COND((numneigh == NULL),"numneigh list not initialized");
        DEBUG_LOG_COND((firstneigh == NULL),"firstneigh list not initialized");
        // 2. creating number array of start num in different c_atom's neighbor
        // LAMMPS_NS::tagint *h_group_numneigh = new LAMMPS_NS::tagint[group_count + 1];
        // LAMMPS_NS::tagint *d_group_numneigh;
        delete[] h_group_numneigh;
        h_group_numneigh = nullptr;
        DEBUG_LOG("free h_group_numneigh");
        h_group_numneigh = new LAMMPS_NS::tagint[group_count + 1];
        SAFE_CUDA_FREE(d_group_numneigh);
        SAFE_CUDA_MALLOC(&d_group_numneigh, (group_count + 1) * sizeof(LAMMPS_NS::tagint), file); // 分配设备端邻居数目数组
        DEBUG_LOG_COND((h_group_numneigh == NULL),"h_group_numneigh list not initialized");
        // 3. 逐原子拷贝邻居列表数据到GPU
        DEBUG_LOG("group_count=%d" ,group_count);
        h_group_numneigh[0] = 0;
        for (int gr_i = 0; gr_i < group_count; gr_i++) {
            int i = h_group_indices[gr_i]; // 获取原子索引
            int jnum = numneigh[i]; // 邻居数量
            h_group_numneigh[gr_i+1] = h_group_numneigh[gr_i] + jnum;
            DEBUG_LOG("gr_i=%d, tag=%d , catom_id=%d, jnum=%d, sum=%d" ,gr_i, i,jnum,h_group_numneigh[gr_i+1]);
        }
        SAFE_CUDA_MEMCPY(d_group_numneigh,h_group_numneigh,(group_count + 1)*sizeof(LAMMPS_NS::tagint),cudaMemcpyHostToDevice,file);
        // =========================================================================
        // h_group_numneigh / d_group_numneigh :
        //      flatten index of the neighbour list. such as we have 20 neighbour
        //      for atom 1, then the list will be : [0, 20, ...]
        // h_firstneigh_ptrs / d_firstneigh_ptrs :
        //      flatten neighbour list
        // =========================================================================
        delete[] h_firstneigh_ptrs;
        h_firstneigh_ptrs = nullptr;
        DEBUG_LOG("free h_firstneigh_ptrs");
        SAFE_CUDA_FREE(d_firstneigh_ptrs);
        // int *h_firstneigh_ptrs = new int [h_group_numneigh[group_count]];
        // int *d_firstneigh_ptrs; // 设备端二级指针
        h_firstneigh_ptrs = new int [h_group_numneigh[group_count]];
        LAMMPS_NS::tagint ba_i;
        LAMMPS_NS::tagint nnumber;
        int i;
        SAFE_CUDA_MALLOC(&d_firstneigh_ptrs, (h_group_numneigh[group_count]) * sizeof(int),file); // 分配设备端指针数组
        DEBUG_LOG("generate d_firstneigh_ptrs, h_group_numneigh[group_count + 1]=%d",h_group_numneigh[group_count]);
        for (int gr_i = 0; gr_i < group_count; gr_i++) {
            i = h_group_indices[gr_i]; // 获取原子索引
            ba_i = h_group_numneigh[gr_i];
            nnumber = h_group_numneigh[gr_i+1]-h_group_numneigh[gr_i];
            DEBUG_LOG("h_group_numneigh=%d, num=%d" ,ba_i,nnumber);
            DEBUG_LOG("end of firstneigh[i]=%d,h_firstneigh_ptrs[h_group_numneigh[gr_i+1]-1] = %d",firstneigh[i][nnumber-1],h_firstneigh_ptrs[h_group_numneigh[gr_i+1]-1]);
            memcpy(&(h_firstneigh_ptrs[ba_i]),firstneigh[i],(nnumber)*sizeof(int));
            DEBUG_LOG("h_firstneigh_ptrs[h_group_numneigh[gr_i+1]-1] = %d",h_firstneigh_ptrs[h_group_numneigh[gr_i+1]-1]);
            if (gr_i==10){
                for (int i =ba_i; i<ba_i+nnumber ;i++){
                    DEBUG_LOG("%d,",atom->tag[h_firstneigh_ptrs[i]]);
                }
            }
        }
        SAFE_CUDA_MEMCPY(d_firstneigh_ptrs,h_firstneigh_ptrs,
            (h_group_numneigh[group_count]) * sizeof(int),cudaMemcpyHostToDevice,file);
        DEBUG_LOG_COND((d_firstneigh_ptrs == NULL),"d_firstneigh_ptrs list not initialized");
        DEBUG_LOG("d_firstneigh_ptrs list %d %d %d" ,h_firstneigh_ptrs[1],h_firstneigh_ptrs[2],h_firstneigh_ptrs[3]);
        DEBUG_LOG("generate end d_firstneigh_ptrs");
    }
    // =========================================================================
    // h_x / h_x_flat / d_x_flat :
    //      atoms coordinate position
    // =========================================================================
    // int *h_tag = atom->tag;       // 原子全局ID数组(主机)
    // int *d_tag;             // 设备端坐标二级指针
    double **h_x = atom->x;      // 原子坐标数组(主机)
    delete[] h_x_flat;
    h_x_flat = nullptr;
    DEBUG_LOG("free h_x_flat");
    DEBUG_LOG("d_x_flat=%p",d_x_flat);
    h_x_flat = new double [(atom->nlocal + atom->nghost) * 3];
    for (int i = 0; i < (atom->nlocal + atom->nghost); i++) {
        memcpy(&(h_x_flat[i*3]), h_x[i], 3*sizeof(double));
    }
    DEBUG_LOG("there are %d, h_x_flat[10]=%f",(atom->nlocal + atom->nghost),h_x_flat[10]);
    SAFE_CUDA_FREE(d_x_flat); 
    SAFE_CUDA_MALLOC(&d_x_flat, ((atom->nlocal + atom->nghost) * 3)*sizeof(double),file);
    SAFE_CUDA_MEMCPY(d_x_flat,h_x_flat,((atom->nlocal + atom->nghost) * 3)*sizeof(double),cudaMemcpyHostToDevice, file);
    // check the pointer
    // DEBUG_LOG("alloc h_x,h_tag.....");
    DEBUG_LOG_COND((h_x == NULL),"h_x list not initialized");
    DEBUG_LOG_COND((h_x_flat == NULL),"h_x_flat list not initialized");
    DEBUG_LOG_COND((d_x_flat == NULL),"d_x_flat list not initialized");
    DEBUG_LOG("d_x_flat Allocated at: %p", d_x_flat);
    cudaDeviceSynchronize(); // waiting memory

    // =========================================================================
    // create output address
    // d_group_dminneigh : (dx, dy, dz, r2) * pairs
    // d_neigh_in_cutoff_r : neighbour atoms that satisfied cutoff_r
    // d_neigh_both_in_r_N : neighbour atoms that satisfied both cutoff_r and N
    // =========================================================================
    DEBUG_LOG("release gpu");
    SAFE_CUDA_FREE(d_neigh_both_in_r_N);
    SAFE_CUDA_FREE(d_group_dminneigh);
    SAFE_CUDA_FREE(d_neigh_in_cutoff_r);
    SAFE_CUDA_FREE(d_calculated_numneigh);
    DEBUG_LOG("release end");
    // double *d_group_dminneigh;
    SAFE_CUDA_MALLOC(&d_group_dminneigh, (N*cutoff_Natoms*4)*sizeof(double),file);
    // int *d_neigh_in_cutoff_r;
    SAFE_CUDA_MALLOC(&d_neigh_in_cutoff_r, (N*4)*sizeof(int),file);
    // int *d_neigh_both_in_r_N;
    SAFE_CUDA_MALLOC(&d_neigh_both_in_r_N, (N)*sizeof(int),file);
    // double *d_calculated_numneigh;
    SAFE_CUDA_MALLOC(&d_calculated_numneigh, (N*cutoff_Natoms*sizeof(LAMMPS_NS::tagint)), file);

    // box_x=box_y=box_z=40.0;
    DEBUG_LOG("box_lim x:%f y:%f z:%f max:%f" ,box_x,box_y,box_z,box_x+box_y+box_z );
    DEBUG_LOG("neigh finding .......");
    DEBUG_LOG("i will start a kernel");
    // kernel function will run
    cudaError_t launchErr = cudaGetLastError();
    cudaDeviceSynchronize(); // waiting memory
    double cutoff_rsq = cutoff_r*cutoff_r;

    // cudaDeviceSynchronize(); //catch kernel done
    launchErr = cudaGetLastError();
    fix_crystallizes_kernel<<<block_num,d_block_size>>>
      ( cutoff_Natoms, cutoff_rsq, box_x, box_y, box_z, 
      group_count, d_group_indices, d_group_numneigh, d_firstneigh_ptrs, d_x_flat,
      d_group_dminneigh, d_neigh_in_cutoff_r, d_neigh_both_in_r_N, d_calculated_numneigh) ;
    // double *h_group_dminneigh = new double [group_count*cutoff_Natoms*4];
    // int *h_neigh_in_cutoff_r = new int [group_count];
    // int *h_neigh_both_in_r_N = new int [group_count];
    // int atomsnumber = (atom->nlocal + atom->nghost);
    // fix_crystallizes_kernel_temp
    //   ( cutoff_Natoms, cutoff_rsq, box_x, box_y, box_z, 
    //   group_count, h_group_indices, h_group_numneigh, h_firstneigh_ptrs, h_x_flat,
    //   h_group_dminneigh, h_neigh_in_cutoff_r, h_neigh_both_in_r_N,atomsnumber) ;
    cudaDeviceSynchronize(); //catch kernel done
    // cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        // 输出到您的文件
        fprintf(file, "CUDA Kernel launch failed: %s\n", cudaGetErrorString(launchErr));
        fflush(file);
        // 尝试输出到标准错误流 (确保在 LAMMPS 终端可见)
        fprintf(stderr, "LAMMPS CUDA ERROR: Kernel launch failed: %s\n", cudaGetErrorString(launchErr));
        error->all(FLERR, "Kernel launch failed. Check output for detailed CUDA error.");
    }
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        fprintf(file, "Kernel execution error: %s\n", cudaGetErrorString(syncErr));
        error->all(FLERR, "Kernel execution error\n");
    }
    DEBUG_LOG("im out");
    DEBUG_LOG("neigh find finished");



    // return the array for neigh
    DEBUG_LOG("copy result array to cpu: group_dminneigh, neigh_in_cutoff_r, neigh_both_in_r_N");
    DEBUG_LOG_COND((group_dminneigh == NULL),"group_dminneigh list not initialized");
    DEBUG_LOG_COND((neigh_in_cutoff_r == NULL),"group_dminneigh list not initialized");
    DEBUG_LOG_COND((neigh_both_in_r_N == NULL),"group_dminneigh list not initialized");
    delete[] group_dminneigh;
    group_dminneigh = new double [group_count*cutoff_Natoms*4];
    SAFE_CUDA_MEMCPY(group_dminneigh, d_group_dminneigh,
      (group_count*cutoff_Natoms*4) * sizeof(double), cudaMemcpyDeviceToHost,file);
    delete[] neigh_in_cutoff_r;
    neigh_in_cutoff_r = new int [group_count];
    SAFE_CUDA_MEMCPY(neigh_in_cutoff_r, d_neigh_in_cutoff_r,
      (group_count) * sizeof(int), cudaMemcpyDeviceToHost,file);
    delete[] neigh_both_in_r_N;
    neigh_both_in_r_N = new int [group_count];
    SAFE_CUDA_MEMCPY(neigh_both_in_r_N, d_neigh_both_in_r_N,
      (group_count) * sizeof(int), cudaMemcpyDeviceToHost,file);
    delete[] calculated_numneigh;
    calculated_numneigh = new LAMMPS_NS::tagint [group_count*cutoff_Natoms];
    SAFE_CUDA_MEMCPY(calculated_numneigh, d_calculated_numneigh,
      (group_count*cutoff_Natoms) * sizeof(LAMMPS_NS::tagint), cudaMemcpyDeviceToHost,file);
    cudaDeviceSynchronize(); //catch kernel done
    DEBUG_LOG("copy end");
    DEBUG_LOG("group_dminneigh Allocated at: %p", group_dminneigh);
    
}


void MetaD_zqc::STEIN_Q4::steinhardt_param_calc(double *stein_ql){
    // steinhardt_param_calc_kernel_q4
    int stein_l=4;
    // in class protect
    int *d_neigh_both_in_r_N;
    SAFE_CUDA_MALLOC(&d_neigh_both_in_r_N, (N*4)*sizeof(int),file);
    SAFE_CUDA_MEMCPY(d_neigh_both_in_r_N, neigh_both_in_r_N,
      (group_count) * sizeof(int), cudaMemcpyHostToDevice,file);
    double *d_group_dminneigh;
    SAFE_CUDA_MALLOC(&d_group_dminneigh, (N*cutoff_Natoms*4)*sizeof(double),file);
    SAFE_CUDA_MEMCPY(d_group_dminneigh, group_dminneigh,
      (group_count*cutoff_Natoms*4) * sizeof(double), cudaMemcpyHostToDevice,file);
    // result array
    // every q has <2*stein_l + 1> qlm, with complex we will times 2
    // double *h_stein_qlm = new double [group_count*(stein_l + 1)*2];
    // for the further concentrate we need to calculate qlm*Neigh, with comple
    delete[] h_stein_Ylm;
    h_stein_Ylm = new double [group_count*cutoff_Natoms*(stein_l + 1)*2];
    SAFE_CUDA_FREE(d_stein_ql);
    SAFE_CUDA_MALLOC(&d_stein_ql, group_count*sizeof(double), file);
    SAFE_CUDA_FREE(d_stein_qlm);
    SAFE_CUDA_MALLOC(&d_stein_qlm, (group_count*(stein_l + 1)*2)*sizeof(double), file);
    SAFE_CUDA_FREE(d_stein_Ylm);
    SAFE_CUDA_MALLOC(&d_stein_Ylm, (group_count*cutoff_Natoms*(stein_l + 1)*2)*sizeof(double), file);

    DEBUG_LOG("i will start a kernel of q4");
    cudaDeviceSynchronize(); // waiting memory
    steinhardt_param_calc_kernel_q4<<<block_num,d_block_size>>>(
        group_count, cutoff_Natoms,
        d_neigh_both_in_r_N, d_group_dminneigh,
        d_stein_qlm, d_stein_Ylm,
        d_stein_ql) ;
    
    cudaDeviceSynchronize(); //catch kernel done
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        fprintf(file, "Kernel launch failed: %s\n", cudaGetErrorString(launchErr));
        error->all(FLERR, "Kernel launch failed\n");
    }
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        fprintf(file, "Kernel execution error: %s\n", cudaGetErrorString(syncErr));
        error->all(FLERR, "Kernel execution error\n");
    }
    DEBUG_LOG("im out");
    DEBUG_LOG("q4 calculated find finished");

    cudaMemcpy(stein_ql, d_stein_ql, (group_count*(stein_l + 1)*2) * sizeof(double), cudaMemcpyDeviceToHost);
    SAFE_CUDA_MEMCPY(stein_ql, d_stein_ql,
      (group_count) * sizeof(double), cudaMemcpyDeviceToHost,file);
    SAFE_CUDA_MEMCPY(h_stein_Ylm, d_stein_Ylm,
      (group_count*cutoff_Natoms*(stein_l + 1)*2) * sizeof(double), cudaMemcpyDeviceToHost,file);

    DEBUG_LOG("release gpu");
    SAFE_CUDA_FREE(d_neigh_both_in_r_N);
    SAFE_CUDA_FREE(d_group_dminneigh);
    

}


void MetaD_zqc::STEIN_Q4::summary(FILE* f){}

__global__ void fix_crystallizes_kernel
(
    int cutoff_Natoms, double cutoff_rsq, double box_x, double box_y, double box_z,
    int group_count, int *d_group_indices, LAMMPS_NS::tagint *d_group_numneigh,
    int *d_firstneigh_ptrs, double *d_x_flat,
    double *d_group_dminneigh, int *d_neigh_in_cutoff_r, int *d_neigh_both_in_r_N,
    LAMMPS_NS::tagint *d_calculated_numneigh
)
{
    int c_atom = blockIdx.x * blockDim.x + threadIdx.x;
    if(c_atom<group_count){
        double r2,temp_r2,temp_x,temp_y,temp_z,neigh_x,neigh_y,neigh_z;
        double delt_x,delt_y,delt_z;
        int c_atom_tag = d_group_indices[c_atom];
        int temp_tag;
        d_neigh_in_cutoff_r[c_atom] = 0;
        // c_glob_tag = h_tag[c_atom_tag];
        double c_x = d_x_flat[c_atom_tag*3];
        double c_y = d_x_flat[c_atom_tag*3+1];
        double c_z = d_x_flat[c_atom_tag*3+2];
        int max_ii;
        // DEBUG_LOG("now im in %d, c_atom_tag=%d, cx,cy,cz:%f,%f,%f",c_atom,c_atom_tag,c_x,c_y,c_z);
        double max_r2 = (box_x+box_y+box_z)*(box_x+box_y+box_z);
        for (int i=0;i<cutoff_Natoms;i++){
            d_group_dminneigh[c_atom*4*cutoff_Natoms +i*4 + 3]=max_r2;
            d_calculated_numneigh[c_atom*cutoff_Natoms +i] = -1;
        }
        //find curtoff_Natoms neigh
        for (int neigh_atom=d_group_numneigh[c_atom]; neigh_atom<d_group_numneigh[c_atom+1]; neigh_atom++){
            int n_local_tag = d_firstneigh_ptrs[neigh_atom];
            // if (n_local_tag < 0 ) continue;
            // int n_glob_tag = h_tag[n_local_tag];
            neigh_x = d_x_flat[n_local_tag*3];
            neigh_y = d_x_flat[n_local_tag*3+1];
            neigh_z = d_x_flat[n_local_tag*3+2];
            delt_x = (neigh_x - c_x);
            delt_y = (neigh_y - c_y);
            delt_z = (neigh_z - c_z);
            if (delt_x > box_x/2) {
                delt_x -= box_x;
            } else if (delt_x < -box_x/2) {
                delt_x += box_x;
            }
            if (delt_y > box_y/2) {
                delt_y -= box_y;
            } else if (delt_y < -box_y/2) {
                delt_y += box_y;
            }
            if (delt_z > box_z/2) {
                delt_z -= box_z;
            } else if (delt_z < -box_z/2) {
                delt_z += box_z;
            }
            // DEBUG_LOG("c_atom_tag=%d, n_local_tag=%d, nx,ny,nz:%f,%f,%f",c_atom_tag,n_local_tag,delt_x,delt_y,delt_z);
            r2 = delt_x*delt_x + delt_y*delt_y + delt_z*delt_z;
            if ((r2 > cutoff_rsq )||(r2<1e-12)) continue;
            d_neigh_in_cutoff_r[c_atom]++;
            for (int ii=0; ii<cutoff_Natoms; ii++){
                if (d_group_dminneigh[c_atom*4*cutoff_Natoms + ii*4 + 3]>r2){
                    temp_x = d_group_dminneigh[c_atom*4*cutoff_Natoms + ii*4 + 0];
                    temp_y = d_group_dminneigh[c_atom*4*cutoff_Natoms + ii*4 + 1];
                    temp_z = d_group_dminneigh[c_atom*4*cutoff_Natoms + ii*4 + 2];
                    temp_r2 = d_group_dminneigh[c_atom*4*cutoff_Natoms + ii*4 + 3];
                    temp_tag = d_calculated_numneigh[c_atom*cutoff_Natoms + ii];
                    d_group_dminneigh[c_atom*4*cutoff_Natoms + ii*4 + 0] = delt_x;
                    d_group_dminneigh[c_atom*4*cutoff_Natoms + ii*4 + 1] = delt_y;
                    d_group_dminneigh[c_atom*4*cutoff_Natoms + ii*4 + 2] = delt_z;
                    d_group_dminneigh[c_atom*4*cutoff_Natoms + ii*4 + 3] = r2;
                    d_calculated_numneigh[c_atom*cutoff_Natoms + ii] = n_local_tag;
                    delt_x = temp_x;
                    delt_y = temp_y;
                    delt_z = temp_z;
                    r2 = temp_r2;
                    n_local_tag = temp_tag;
                }
            }
        }
        if (d_neigh_in_cutoff_r[c_atom]>=cutoff_Natoms){
            d_neigh_both_in_r_N[c_atom]=cutoff_Natoms;
        }
        else{
            d_neigh_both_in_r_N[c_atom]=d_neigh_in_cutoff_r[c_atom];
        }
    }
}



// void dcv_steinhardt_param_calc_kernel_q4(
//     FILE *f_check,
__global__ void dcv_steinhardt_param_calc_kernel_q4(
    int cutoff_Natoms, 
    int group_count, int groupbit, int *d_mask,
    LAMMPS_NS::tagint *d_group_indices, LAMMPS_NS::tagint *calculated_numneigh, 
    int *d_neigh_both_in_r_N, double *d_group_dminneigh,
    double *d_stein_qlm, double *d_stein_Ylm, double *d_stein_ql,
    double *d_dYlm_dr,double *d_dcvdx
)
{
    // devise version=============
    int c_atom = blockIdx.x * blockDim.x + threadIdx.x;
    if(c_atom<group_count){
    // host version===============
    // for (int c_atom=0; c_atom<group_count; c_atom++){
        int stein_l=4;
        int neigh_tag, neigh_Nb;
        double neigh_q4_timesN, catom_q4_timesN;
        double delt_x, delt_y, delt_z, r2, r;
        double theta, phi;
        double DthetaDx, DthetaDy, DthetaDz, DphiDx, DphiDy, DphiDz;
        double tempDtheta, tempDtheta_i, tempDtheta_r, tempDphi, tempDphi_i, tempDphi_r;
        int stein_qlm_base_id, stein_qlm_neigh_id, Neigh_Nb;
        int neigh_num = d_neigh_both_in_r_N[c_atom];
        catom_q4_timesN = 1/(d_stein_ql[c_atom]*neigh_num);
        for(int i=0; i<3; i++){
            // from 0 to l, both re_part and im_part
            d_dcvdx[c_atom*3 + i] = 0;
            d_dYlm_dr[c_atom*3*2 + i*2 + 0] = 0;
            d_dYlm_dr[c_atom*3*2 + i*2 + 1] = 0;
            // DEBUG_LOG("d_stein_qlm[%d] = %f + i* %f", stein_qlm_base_id + i + 1,d_stein_qlm[stein_qlm_base_id + i + 0],d_stein_qlm[stein_qlm_base_id + i + 1]);
        }
        for(int neigh_atom=0; neigh_atom<neigh_num; neigh_atom++){
            delt_x = d_group_dminneigh[c_atom*cutoff_Natoms*4 + neigh_atom*4 + 0];
            delt_y = d_group_dminneigh[c_atom*cutoff_Natoms*4 + neigh_atom*4 + 1];
            delt_z = d_group_dminneigh[c_atom*cutoff_Natoms*4 + neigh_atom*4 + 2];
            r2     = d_group_dminneigh[c_atom*cutoff_Natoms*4 + neigh_atom*4 + 3];
            r      = sqrt(r2);
            theta = asin(delt_z/r);
            phi = atan2(delt_y, delt_x);
            DthetaDx = (delt_x*delt_z)/(r2*sqrt(POW2(delt_x) + POW2(delt_y)));
            DthetaDy = (delt_y*delt_z)/(r2*sqrt(POW2(delt_x) + POW2(delt_y)));
            DthetaDz = sqrt(POW2(delt_x) + POW2(delt_y))/(r2);
            DphiDx = - (delt_y)/(POW2(delt_x) + POW2(delt_y));
            DphiDy = (delt_x)/(POW2(delt_x) + POW2(delt_y));
            DphiDz = 0;
            stein_qlm_base_id = c_atom*(stein_l + 1)*2;
            // 处理 neigh 与 cv-group 重合的部分
            neigh_tag = calculated_numneigh[c_atom*cutoff_Natoms + neigh_atom];
            // 判断当前 neigh atom 是否是 cv-group 的一个原子
            // 由于 Y,4,m(theta, phi) = Y,4,m(pi-theta,phi), 因此该邻居的Y与本原子的Y表达式相同
            Neigh_Nb = 0;
            neigh_q4_timesN = 0;
            stein_qlm_neigh_id=0;
            if (d_mask[neigh_tag]&groupbit){
                // 使用二分查找法找 neigh_tag 对应在 d_stein_ql 中的位置
                int left = 0;
                int right = group_count - 1;
                // neigh_q4_deN default is 0
                while (left <= right) {
                    int mid = left + (right - left) / 2;
                    if (d_group_indices[mid] == neigh_tag) {
                        Neigh_Nb = d_neigh_both_in_r_N[mid];
                        neigh_q4_timesN = 1/(d_stein_ql[mid]*Neigh_Nb);
                        stein_qlm_neigh_id = Neigh_Nb*cutoff_Natoms*(stein_l + 1)*2 + d_group_indices[c_atom]*(stein_l + 1)*2;
                        // DEBUG_LOG("mid=%d, stein_qlm_neigh_id=%d",mid,stein_qlm_neigh_id);
                        break;
                    } else if (d_group_indices[mid] < neigh_tag) {
                        left = mid + 1;
                    } else {
                        right = mid - 1;
                    }
                }
            }
            // d_dcvdx = [dcvdxc, dcvdyc, dcvdzc]*catoms --flatten
            // d_dYlm_dr = [dYlm_dx_re, dYlm_dx_im, dYlm_dy_re, dYlm_dy_im, dYlm_dz_re, dYlm_dz_im,]*catoms --flatten
            // Y,4,0
            // d Y,4,0 dx
            // 3/(16 Sqrt[\[Pi]])=0.10578554691520430380
            tempDtheta= 0.10578554691520430380*(60*cos(theta)*sin(theta)-140*sin(theta)*POW3(cos(theta)));
            tempDphi= 0 ;
            d_dYlm_dr[c_atom*3*2+0]+= (tempDtheta*DthetaDx + 0)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 0] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 0]) ;
            d_dYlm_dr[c_atom*3*2+1]+= 0 ;
            // d Y,4,0 dy
            d_dYlm_dr[c_atom*3*2+2]+= (tempDtheta*DthetaDy + 0)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 0] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 0]) ;
            d_dYlm_dr[c_atom*3*2+3]+= 0 ;
            // d Y,4,0 dz
            d_dYlm_dr[c_atom*3*2+4]+= (tempDtheta*DthetaDz + 0)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 0] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 0]) ;
            d_dYlm_dr[c_atom*3*2+5]+= 0 ;
            // Y,4,+-1
            // 3/32 Sqrt[5/\[Pi]] = 0.11827183696969500226
            tempDtheta= 2*0.11827183696969500226*(cos(2*theta) + 7*cos(4*theta));
            tempDtheta_r = tempDtheta*cos(phi);
            tempDtheta_i = tempDtheta*sin(phi);
            tempDphi= 0.11827183696969500226*(1 + 7*cos(2*theta))*sin(2*theta);
            tempDphi_r = tempDphi*sin(phi);
            tempDphi_i = -tempDphi*cos(phi);
            // d Y,4,1 dx
            d_dYlm_dr[c_atom*3*2+0]+= (tempDtheta_r*DthetaDx + tempDphi_r*DphiDx)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 2]) ;
            d_dYlm_dr[c_atom*3*2+1]+= (tempDtheta_i*DthetaDx + tempDphi_i*DphiDx)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 3]) ;
            // d Y,4,1 dy
            d_dYlm_dr[c_atom*3*2+2]+= (tempDtheta_r*DthetaDy + tempDphi_r*DphiDy)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 2]) ;
            d_dYlm_dr[c_atom*3*2+3]+= (tempDtheta_i*DthetaDy + tempDphi_i*DphiDy)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 3]) ;
            // d Y,4,1 dz
            d_dYlm_dr[c_atom*3*2+4]+= (tempDtheta_r*DthetaDz + tempDphi_r*DphiDz)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 2]) ;
            d_dYlm_dr[c_atom*3*2+5]+= (tempDtheta_i*DthetaDz + tempDphi_i*DphiDz)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 3]) ;
            // d Y,4,-1 dx
            d_dYlm_dr[c_atom*3*2+0]+= (-tempDtheta_r*DthetaDx - tempDphi_r*DphiDx)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 2]) ;
            d_dYlm_dr[c_atom*3*2+1]+= (tempDtheta_i*DthetaDx + tempDphi_i*DphiDx)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 3]) ;
            // d Y,4,-1 dy
            d_dYlm_dr[c_atom*3*2+2]+= (-tempDtheta_r*DthetaDy - tempDphi_r*DphiDy)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 2]) ;
            d_dYlm_dr[c_atom*3*2+3]+= (tempDtheta_i*DthetaDy + tempDphi_i*DphiDy)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 3]) ;
            // d Y,4,-1 dz
            d_dYlm_dr[c_atom*3*2+4]+= (-tempDtheta_r*DthetaDz - tempDphi_r*DphiDz)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 2]) ;
            d_dYlm_dr[c_atom*3*2+5]+= (tempDtheta_i*DthetaDz + tempDphi_i*DphiDz)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 3]) ;
            // Y,4,+-2
            // 3/8 Sqrt[5/(2 \[Pi])] = 0.33452327177864458398
            tempDtheta= 0.33452327177864458398*(-1 + 7*cos(2*theta))*sin(2*theta);
            tempDtheta_r = tempDtheta*cos(2*phi);
            tempDtheta_i = tempDtheta*sin(2*phi);
            tempDphi= 0.33452327177864458398*(5 + 7*cos(2*theta))*POW2(sin(theta));
            tempDphi_r = tempDphi*sin(2*phi);
            tempDphi_i = tempDphi*cos(2*phi);
            // d Y,4,2 dx
            d_dYlm_dr[c_atom*3*2+0]+= (tempDtheta_r*DthetaDx - tempDphi_r*DphiDx)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 4]) ;
            d_dYlm_dr[c_atom*3*2+1]+= (tempDtheta_i*DthetaDx + tempDphi_i*DphiDx)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 5]) ;
            // d Y,4,2 dy
            d_dYlm_dr[c_atom*3*2+2]+= (tempDtheta_r*DthetaDy - tempDphi_r*DphiDy)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 4]) ;
            d_dYlm_dr[c_atom*3*2+3]+= (tempDtheta_i*DthetaDy + tempDphi_i*DphiDy)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 5]) ;
            // d Y,4,2 dz
            d_dYlm_dr[c_atom*3*2+4]+= (tempDtheta_r*DthetaDz - tempDphi_r*DphiDz)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 4]) ;
            d_dYlm_dr[c_atom*3*2+5]+= (tempDtheta_i*DthetaDz + tempDphi_i*DphiDz)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 5]) ;
            // d Y,4,-2 dx
            d_dYlm_dr[c_atom*3*2+0]+= (tempDtheta_r*DthetaDx - tempDphi_r*DphiDx)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 4]) ;
            d_dYlm_dr[c_atom*3*2+1]+= (-tempDtheta_i*DthetaDx - tempDphi_i*DphiDx)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 5]) ;
            // d Y,4,-2 dy
            d_dYlm_dr[c_atom*3*2+2]+= (tempDtheta_r*DthetaDy - tempDphi_r*DphiDy)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 4]) ;
            d_dYlm_dr[c_atom*3*2+3]+= (-tempDtheta_i*DthetaDy - tempDphi_i*DphiDy)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 5]) ;
            // d Y,4,-2 dz
            d_dYlm_dr[c_atom*3*2+4]+= (tempDtheta_r*DthetaDz - tempDphi_r*DphiDz)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 4]) ;
            d_dYlm_dr[c_atom*3*2+5]+= (-tempDtheta_i*DthetaDz - tempDphi_i*DphiDz)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 5]) ;
            // Y,4,+-3
            // 3/8 Sqrt[35/\[Pi]] = 1.2516714708983522692
            tempDtheta= 1.2516714708983522692*(1 + 2*cos(2*theta))*POW2(sin(theta));
            tempDtheta_r = tempDtheta*cos(3*phi);
            tempDtheta_i = tempDtheta*sin(3*phi);
            tempDphi= 3*1.2516714708983522692*(cos(theta))*POW3(sin(theta));
            tempDphi_r = tempDphi*sin(3*phi);
            tempDphi_i = tempDphi*cos(3*phi);
            // d Y,4,3 dx
            d_dYlm_dr[c_atom*3*2+0]+= (-tempDtheta_r*DthetaDx + tempDphi_r*DphiDx)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 6]) ;
            d_dYlm_dr[c_atom*3*2+1]+= (-tempDtheta_i*DthetaDx - tempDphi_i*DphiDx)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 7]) ;
            // d Y,4,3 dy
            d_dYlm_dr[c_atom*3*2+2]+= (-tempDtheta_r*DthetaDy + tempDphi_r*DphiDy)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 6]) ;
            d_dYlm_dr[c_atom*3*2+3]+= (-tempDtheta_i*DthetaDy - tempDphi_i*DphiDy)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 7]) ;
            // d Y,4,3 dz
            d_dYlm_dr[c_atom*3*2+4]+= (-tempDtheta_r*DthetaDz + tempDphi_r*DphiDz)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 6]) ;
            d_dYlm_dr[c_atom*3*2+5]+= (-tempDtheta_i*DthetaDz - tempDphi_i*DphiDz)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 7]) ;
            // d Y,4,-3 dx
            d_dYlm_dr[c_atom*3*2+0]+= (tempDtheta_r*DthetaDx  -tempDphi_r*DphiDx)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 6]) ;
            d_dYlm_dr[c_atom*3*2+1]+= (-tempDtheta_i*DthetaDx - tempDphi_i*DphiDx)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 7]) ;
            // d Y,4,-3 dy
            d_dYlm_dr[c_atom*3*2+2]+= (tempDtheta_r*DthetaDy  -tempDphi_r*DphiDy)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 6]) ;
            d_dYlm_dr[c_atom*3*2+3]+= (-tempDtheta_i*DthetaDy - tempDphi_i*DphiDy)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 7]) ;
            // d Y,4,-3 dz
            d_dYlm_dr[c_atom*3*2+4]+= (tempDtheta_r*DthetaDz  -tempDphi_r*DphiDz)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 6]) ;
            d_dYlm_dr[c_atom*3*2+5]+= (-tempDtheta_i*DthetaDz - tempDphi_i*DphiDz)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 7]) ;
            // Y,4,+-4
            // 3/4 Sqrt[35/(2 \[Pi])] = 1.7701307697799305310
            tempDtheta= 1.7701307697799305310*(cos(theta))*POW3(sin(theta));
            tempDtheta_r = tempDtheta*cos(4*phi);
            tempDtheta_i = tempDtheta*sin(4*phi);
            tempDphi= 1.7701307697799305310*POW4(sin(theta));
            tempDphi_r = tempDphi*sin(4*phi);
            tempDphi_i = tempDphi*cos(4*phi);
            // d Y,4,4 dx
            d_dYlm_dr[c_atom*3*2+0]+= (tempDtheta_r*DthetaDx - tempDphi_r*DphiDx)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 8]) ;
            d_dYlm_dr[c_atom*3*2+1]+= (tempDtheta_i*DthetaDx + tempDphi_i*DphiDx)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 9]) ;
            // d Y,4,4 dy
            d_dYlm_dr[c_atom*3*2+2]+= (tempDtheta_r*DthetaDy - tempDphi_r*DphiDy)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 8]) ;
            d_dYlm_dr[c_atom*3*2+3]+= (tempDtheta_i*DthetaDy + tempDphi_i*DphiDy)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 9]) ;
            // d Y,4,4 dz
            d_dYlm_dr[c_atom*3*2+4]+= (tempDtheta_r*DthetaDz - tempDphi_r*DphiDz)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 8]) ;
            d_dYlm_dr[c_atom*3*2+5]+= (tempDtheta_i*DthetaDz + tempDphi_i*DphiDz)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 9]) ;
            // d Y,4,-4 dx
            d_dYlm_dr[c_atom*3*2+0]+= (tempDtheta_r*DthetaDx - tempDphi_r*DphiDx)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 8]) ;
            d_dYlm_dr[c_atom*3*2+1]+= (-tempDtheta_i*DthetaDx  -tempDphi_i*DphiDx)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 9]) ;
            // d Y,4,-4 dy
            d_dYlm_dr[c_atom*3*2+2]+= (tempDtheta_r*DthetaDy - tempDphi_r*DphiDy)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 8]) ;
            d_dYlm_dr[c_atom*3*2+3]+= (-tempDtheta_i*DthetaDy  -tempDphi_i*DphiDy)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 9]) ;
            // d Y,4,-4 dz
            d_dYlm_dr[c_atom*3*2+4]+= (tempDtheta_r*DthetaDz - tempDphi_r*DphiDz)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 8]) ;
            d_dYlm_dr[c_atom*3*2+5]+= (-tempDtheta_i*DthetaDz  -tempDphi_i*DphiDz)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 9]) ;
            // DEBUG_LOG("the neighbor has tag=%d is in c_atom=%d, group_indices[%d]=%d, the stein_q4=%g",neigh_tag,mid,mid,d_group_indices[mid],d_stein_ql[mid]);            
        }
        for (int i=0;i<3;i++){
            d_dcvdx[c_atom*3+i] = d_dYlm_dr[c_atom*3*2 + i*2 + 0] + d_dYlm_dr[c_atom*3*2 + i*2 + 1];
            d_dcvdx[c_atom*3+i] = -(d_dcvdx[c_atom*3+i]*2*PI)/(group_count*(2*stein_l+1));
        }
        // DEBUG_LOG("catom=%d, d_dcvdx[%d] dx, dy, dz = %g, %g, %g", c_atom, c_atom*3, d_dcvdx[c_atom*3+0], d_dcvdx[c_atom*3+1], d_dcvdx[c_atom*3+2]);
    }
}


__global__ void steinhardt_param_calc_kernel_q4(
    int group_count, int cutoff_Natoms,
    int *d_neigh_both_in_r_N, double *d_group_dminneigh,
    double *d_stein_qlm, double *d_stein_Ylm,
    double *d_stein_ql
){
    int c_atom = blockIdx.x * blockDim.x + threadIdx.x;
    int neigh_N = 0;
    if(c_atom<group_count){
        // steinhardt_param_calc_kernel_q4
        int stein_l=4;
        // init some we need
        int stein_Ylm_base_id, stein_qlm_base_id, neigh_num;
        double delt_x, delt_y, delt_z, r2, r;
        double theta, phi;
        double costheta, sintheta, cosphi, sinphi;
        double re_part, im_part;
        double temp_value, temp1;
        // d_stein_ql[c_atom] = 322;

        neigh_num = d_neigh_both_in_r_N[c_atom];
        // DEBUG_LOG("neigh_num of c_atom = %d",neigh_num);
        // q and qlm init
        stein_qlm_base_id = c_atom*(stein_l + 1)*2;
        d_stein_ql[c_atom] = 0;
        for(int i=0; i<(stein_l+1); i++){
            // from 0 to l, both re_part and im_part
            d_stein_qlm[stein_qlm_base_id + i*2 + 0] = 0;
            d_stein_qlm[stein_qlm_base_id + i*2 + 1] = 0;
            // DEBUG_LOG("d_stein_qlm[%d] = %f + i* %f", stein_qlm_base_id + i + 1,d_stein_qlm[stein_qlm_base_id + i + 0],d_stein_qlm[stein_qlm_base_id + i + 1]);
        }
        // start to calc
        for(int neigh_atom=0; neigh_atom<neigh_num; neigh_atom++){
            delt_x = d_group_dminneigh[c_atom*cutoff_Natoms*4 + neigh_atom*4 + 0];
            delt_y = d_group_dminneigh[c_atom*cutoff_Natoms*4 + neigh_atom*4 + 1];
            delt_z = d_group_dminneigh[c_atom*cutoff_Natoms*4 + neigh_atom*4 + 2];
            r2     = d_group_dminneigh[c_atom*cutoff_Natoms*4 + neigh_atom*4 + 3];
            r      = sqrt(r2);
            theta = asin(delt_z/r);
            phi = atan2(delt_y, delt_x);
            stein_Ylm_base_id = c_atom*cutoff_Natoms*(stein_l + 1)*2 + neigh_atom*(stein_l + 1)*2;
            // DEBUG_LOG("delt x,y,z ; r2, r = %f, %f, %f, %f, %f", delt_x, delt_y, delt_z, r2, r);
            // Y,4,0
            // 3/16*sqrt(1/(1*pi)) = 0.10578554691520430
            d_stein_Ylm[stein_Ylm_base_id + 0] = 0.10578554691520430*(3-30*cos(theta)*cos(theta)+35*cos(theta)*cos(theta)*cos(theta)*cos(theta));
            d_stein_Ylm[stein_Ylm_base_id + 1] = 0;
            // Y,4,1
            // Y,4,-1, -Re+Im
            // 3/8*sqrt(5/(1*pi)) = 0.47308734787878000
            temp_value = 0.47308734787878000*cos(theta)*(-3+7*cos(theta)*cos(theta))*sin(theta);
            d_stein_Ylm[stein_Ylm_base_id + 2] = - temp_value * cos(phi);
            d_stein_Ylm[stein_Ylm_base_id + 3] = - temp_value * sin(phi);
            // Y,4,2
            // Y,4,-2, Re-Im
            // 3/8*sqrt(5/(2*pi)) = 0.33452327177864458
            temp_value = 0.33452327177864458*(-1+7*cos(theta)*cos(theta))*sin(theta)*sin(theta);
            d_stein_Ylm[stein_Ylm_base_id + 4] = temp_value * cos(2*phi);
            d_stein_Ylm[stein_Ylm_base_id + 5] = temp_value * sin(2*phi);
            // Y,4,3
            // Y,4,-3, -Re+Im
            // 3/8*sqrt(35/(pi)) = 1.25167147089835227
            temp_value = 1.25167147089835227*cos(theta)*sin(theta)*sin(theta)*sin(theta);
            d_stein_Ylm[stein_Ylm_base_id + 6] = - cos(3*phi);
            d_stein_Ylm[stein_Ylm_base_id + 7] = - sin(3*phi);
            // Y,4,4
            // Y,4,-4, Re-Im
            // 3/16*sqrt(35/(2*pi)) = 0.44253269244498263
            temp_value = 0.44253269244498263*sin(theta)*sin(theta)*sin(theta)*sin(theta);
            d_stein_Ylm[stein_Ylm_base_id + 8] = cos(4*phi);
            d_stein_Ylm[stein_Ylm_base_id + 9] = sin(4*phi);
            // DEBUG_LOG("d_stein_Ylm 0, 1, 2, 3, 4 = %f, %f, %f, %f, %f",d_stein_Ylm[stein_Ylm_base_id + 0], d_stein_Ylm[stein_Ylm_base_id + 3], d_stein_Ylm[stein_Ylm_base_id + 4], d_stein_Ylm[stein_Ylm_base_id + 7], d_stein_Ylm[stein_Ylm_base_id + 8]);


            
            // q,4,0
            d_stein_qlm[stein_qlm_base_id + 0] += d_stein_Ylm[stein_Ylm_base_id + 0];
            d_stein_qlm[stein_qlm_base_id + 1] += d_stein_Ylm[stein_Ylm_base_id + 1];
            // q,4,1
            d_stein_qlm[stein_qlm_base_id + 2] += d_stein_Ylm[stein_Ylm_base_id + 2];
            d_stein_qlm[stein_qlm_base_id + 3] += d_stein_Ylm[stein_Ylm_base_id + 3];
            // q,4,2
            d_stein_qlm[stein_qlm_base_id + 4] += d_stein_Ylm[stein_Ylm_base_id + 4];
            d_stein_qlm[stein_qlm_base_id + 5] += d_stein_Ylm[stein_Ylm_base_id + 5];
            // q,4,3
            d_stein_qlm[stein_qlm_base_id + 6] += d_stein_Ylm[stein_Ylm_base_id + 6];
            d_stein_qlm[stein_qlm_base_id + 7] += d_stein_Ylm[stein_Ylm_base_id + 7];
            // q,4,4
            d_stein_qlm[stein_qlm_base_id + 8] += d_stein_Ylm[stein_Ylm_base_id + 8];
            d_stein_qlm[stein_qlm_base_id + 9] += d_stein_Ylm[stein_Ylm_base_id + 9];
        }
        // only calculate once devide for each Ylm
        for (int i=0;i<(stein_l + 1)*2;i++){
            // qlm = sum(Ylm)/N
            d_stein_qlm[stein_qlm_base_id + i] = d_stein_qlm[stein_qlm_base_id + i]/neigh_num;}
        // q init with q,l,0
        d_stein_ql[c_atom] = d_stein_qlm[stein_qlm_base_id + 0] * d_stein_qlm[stein_qlm_base_id + 0];
        // q add with q,l,m
        for (int i=1;i<(stein_l + 1);i++){
            re_part = d_stein_qlm[stein_qlm_base_id + i*2 + 0];
            im_part = d_stein_qlm[stein_qlm_base_id + i*2 + 1];
            d_stein_ql[c_atom] += 2*(re_part*re_part + im_part*im_part);
            // DEBUG_LOG("d_stein_qlm[%d] = %f + i* %f", stein_qlm_base_id + i + 1,d_stein_qlm[stein_qlm_base_id + i*2 + 0],d_stein_qlm[stein_qlm_base_id + i*2 + 1]);
        }
        // DEBUG_LOG("q4 of c_atom[%d] = %f",atom->tag[c_atom],d_stein_ql[c_atom]);
        // 4*pi/(2*1+1) = 4*pi/9 = 1.396263401595463
        d_stein_ql[c_atom] = 1.396263401595463*d_stein_ql[c_atom];
        d_stein_ql[c_atom] = sqrt(d_stein_ql[c_atom]);
    }
}
