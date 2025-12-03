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
    double *d_group_dminneigh, int *d_neigh_in_cutoff_r, int *d_neigh_both_in_r_N
);
__global__ void steinhardt_param_calc_kernel_q4(
    int group_count, int cutoff_Natoms,
    int *d_neigh_both_in_r_N, double *d_group_dminneigh,
    double *d_stein_qlm, double *d_stein_Ylm,
    double *d_stein_ql
);
__global__ void steinhardt_param_calc_kernel_q6();

void temp_steinhardt_param_calc_kernel_q4(
    FILE *f_check,
    int group_count, int cutoff_Natoms,
    int *d_neigh_both_in_r_N, double *d_group_dminneigh,
    double *d_stein_qlm, double *d_stein_Ylm,
    double *d_stein_ql
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
    box_x = (pbc_x) ? lmp->domain->xprd : INFINITY;
    box_y = (pbc_y) ? lmp->domain->yprd : INFINITY;
    box_z = (pbc_z) ? lmp->domain->zprd : INFINITY;
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
    Q_per_atoms_value = new double [2]; //inintial
    stein_q = new double *[1];
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
    delete[] Q_per_atoms_value;
}

double MetaD_zqc::STEIN_Q4::compute_cv(){

    if (nlist == nullptr){
        nlist = Fixmetad->listfull;
    }
    if (nlist == nullptr) {
        lmp->error->all(FLERR, "STEIN_Q4 CV failed to find neighbor list now.");
    }
    atom = lmp->atom;
    int *mask = atom->mask;     // 原子组掩码
    // DEBUG_LOG("post_force function in");
    // DEBUG_LOG("ITEM: TIMESTEP\n%ld\n", lmp->update->ntimestep);
    // DEBUG_LOG("ITEM: NUMBER OF ATOMS\n%d\n", atom->nlocal);
    // DEBUG_LOG("ITEM: BOX BOUNDS pp pp pp\n");
    // DEBUG_LOG("%f %f\n", lmp->domain->boxlo[0], lmp->domain->boxhi[0]);
    // DEBUG_LOG("%f %f\n", lmp->domain->boxlo[1], lmp->domain->boxhi[1]);
    // DEBUG_LOG("%f %f\n", lmp->domain->boxlo[2], lmp->domain->boxhi[2]);
    h_group_indices = new int [(atom->nlocal)];

    group_count = 0;
    // DEBUG_LOG("ITEM: ATOMS id type x y z\n");
    for (int i = 0; i < atom->nlocal; i++) {
        if (mask[i] & groupbit){
            h_group_indices[group_count] = i; // record local index
            // DEBUG_LOG("iD:%d local:%d T:%d %f %f %f", 
            //         atom->tag[i], i, atom->type[i],
            //         atom->x[i][0], atom->x[i][1], atom->x[i][2]);
            group_count++;
        }
    }
    // im done the groupcount
    block_num = (group_count + d_block_size - 1)/d_block_size;
    N = d_block_size*block_num;
    stein_q[0] = new double[group_count];
    DEBUG_LOG("d_block_size is %d, block_num is %d",d_block_size, block_num);
    // DEBUG_LOG_COND((pair == NULL),"pair list not initialized");

    // 2. calculate atoms' envioronment
    DEBUG_LOG("envioronment function in");
    envioronment();
    DEBUG_LOG("envioronment function out");

    // 3. calculate atoms' other things
    // steinhardt_param(Q_hybrid);
    steinhardt_param_calc(stein_q[0]);

    DEBUG_LOG_COND((group_dminneigh == NULL),"group_dminneigh list not initialized");
    DEBUG_LOG("group_dminneigh Allocated at: %p", group_dminneigh);
    
    for (int ii=0; ii<group_count; ii++){
        for (int jj=0; jj<1; jj++){
            fprintf(file, "c_atom_idx=%d,%d,%d : Nx:%f Ny:%f Nz:%f r2:%f\n", 
                    atom->tag[h_group_indices[ii]],
                    neigh_in_cutoff_r[ii],
                    neigh_both_in_r_N[ii],
                    group_dminneigh[ii*cutoff_Natoms*4 + jj*4 + 0],
                    group_dminneigh[ii*cutoff_Natoms*4 + jj*4 + 1],
                    group_dminneigh[ii*cutoff_Natoms*4 + jj*4 + 2],
                    group_dminneigh[ii*cutoff_Natoms*4 + jj*4 + 3]);
        }
    }

    DEBUG_LOG("post_force function end");
    delete[] h_group_indices;
    h_group_indices = nullptr;
    DEBUG_LOG("free h_group_indices");

    DEBUG_LOG("calculate domain....");
    DEBUG_LOG_COND((lmp->domain == NULL),"domain list not initialized");
}

void MetaD_zqc::STEIN_Q4::compute_grad(double dVdcv)
{
    // pass
    DEBUG_LOG("In STEIN_Q4 compute_grad");
}

void MetaD_zqc::STEIN_Q4::get_dcvdx(double cv_value, double *dcvdx)
{
    
}



void MetaD_zqc::STEIN_Q4::envioronment()
{
    // =========================================================================
    // neighbour list and its copy to devise
    // h_group_indices / d_group_indices: where the group atoms in locals' tag
    // =========================================================================
    DEBUG_LOG("cutoff_Natoms is %d",cutoff_Natoms);
    DEBUG_LOG("cutoff_r is %f",cutoff_r);
    DEBUG_LOG("group_count is %d",group_count);
    int *d_group_indices = nullptr;
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
    firstneigh = nlist->firstneigh;
    numneigh = nlist->numneigh;
    DEBUG_LOG_COND((numneigh == NULL),"numneigh list not initialized");
    DEBUG_LOG_COND((firstneigh == NULL),"firstneigh list not initialized");
    // 2. creating number array of start num in different c_atom's neighbor
    LAMMPS_NS::tagint *h_group_numneigh = new LAMMPS_NS::tagint[group_count + 1];
    LAMMPS_NS::tagint *d_group_numneigh;
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
    int *h_firstneigh_ptrs = new int [h_group_numneigh[group_count]];
    int *d_firstneigh_ptrs; // 设备端二级指针
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
    // delete[] h_firstneigh_ptrs;
    // h_firstneigh_ptrs = nullptr;
    DEBUG_LOG("free h_firstneigh_ptrs");
    // delete[] h_group_numneigh;
    // h_group_numneigh = nullptr;
    DEBUG_LOG("free h_group_numneigh");

    
    // =========================================================================
    // h_x / h_x_flat / d_x_flat :
    //      atoms coordinate position
    // =========================================================================
    // int *h_tag = atom->tag;       // 原子全局ID数组(主机)
    // int *d_tag;             // 设备端坐标二级指针
    double **h_x = atom->x;      // 原子坐标数组(主机)
    double *h_x_flat = new double [(atom->nlocal + atom->nghost) * 3];
    double *d_x_flat;             // 设备端坐标二级指针
    SAFE_CUDA_MALLOC(&d_x_flat, ((atom->nlocal + atom->nghost) * 3)*sizeof(double),file);
    for (int i = 0; i < (atom->nlocal + atom->nghost); i++) {
        memcpy(&(h_x_flat[i*3]), h_x[i], 3*sizeof(double));
    }
    DEBUG_LOG("there are %d, h_x_flat[10]=%f",(atom->nlocal + atom->nghost),h_x_flat[10]);
    SAFE_CUDA_MEMCPY(d_x_flat,h_x_flat,((atom->nlocal + atom->nghost) * 3)*sizeof(double),cudaMemcpyHostToDevice, file);
    // delete[] h_x_flat;
    // h_x_flat = nullptr;
    DEBUG_LOG("free h_x_flat");


    // check the pointer
    // DEBUG_LOG("alloc h_x,h_tag.....");
    DEBUG_LOG_COND((h_x == NULL),"h_x list not initialized");
    // DEBUG_LOG_COND((h_tag == NULL),"h_tag list not initialized");
    // DEBUG_LOG_COND((d_tag == NULL),"d_tag list not initialized");
    DEBUG_LOG_COND((h_x_flat == NULL),"h_x_flat list not initialized");
    DEBUG_LOG_COND((d_x_flat == NULL),"d_x_flat list not initialized");
    DEBUG_LOG("d_x_flat Allocated at: %p", d_x_flat);
    
    
    // =========================================================================
    // create output address
    // d_group_dminneigh : (dx, dy, dz, r2) * pairs
    // d_neigh_in_cutoff_r : neighbour atoms that satisfied cutoff_r
    // d_neigh_both_in_r_N : neighbour atoms that satisfied both cutoff_r and N
    // =========================================================================
    double *d_group_dminneigh;
    SAFE_CUDA_MALLOC(&d_group_dminneigh, (N*cutoff_Natoms*4)*sizeof(double),file);
    int *d_neigh_in_cutoff_r;
    SAFE_CUDA_MALLOC(&d_neigh_in_cutoff_r, (N*4)*sizeof(int),file);
    int *d_neigh_both_in_r_N;
    SAFE_CUDA_MALLOC(&d_neigh_both_in_r_N, (N*4)*sizeof(int),file);

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
      d_group_dminneigh, d_neigh_in_cutoff_r, d_neigh_both_in_r_N) ;
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
    // cudaError_t launchErr = cudaGetLastError();
    // if (launchErr != cudaSuccess) {
    //     fprintf(file, "Kernel launch failed: %s\n", cudaGetErrorString(launchErr));
    //     error->all(FLERR, "Kernel launch failed\n");
    // }
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
    cudaDeviceSynchronize(); //catch kernel done
    DEBUG_LOG("copy end");
    DEBUG_LOG("group_dminneigh Allocated at: %p", group_dminneigh);
    
    DEBUG_LOG("release gpu");
    SAFE_CUDA_FREE(d_group_indices);
    SAFE_CUDA_FREE(d_firstneigh_ptrs);
    SAFE_CUDA_FREE(d_group_numneigh);
    SAFE_CUDA_FREE(d_x_flat);
    SAFE_CUDA_FREE(d_group_dminneigh);
    SAFE_CUDA_FREE(d_neigh_in_cutoff_r);
    SAFE_CUDA_FREE(d_neigh_both_in_r_N);
    DEBUG_LOG("release end");
}


void MetaD_zqc::STEIN_Q4::get_numneigh_full_pair_ABANDON_()
{
    // 2. creating number array of start num in different c_atom's neighbor
    LAMMPS_NS::tagint *h_group_numneigh = new LAMMPS_NS::tagint[group_count + 1];
    LAMMPS_NS::tagint *d_group_numneigh;
    SAFE_CUDA_MALLOC(&d_group_numneigh, (group_count + 1) * sizeof(LAMMPS_NS::tagint), file); // 分配设备端邻居数目数组
    DEBUG_LOG_COND((h_group_numneigh == NULL),"h_group_numneigh list not initialized");

    // generate full neigh from half
    std::vector<std::vector<int>> full_neigh_temp(atom->nlocal+atom->nghost); 
    for (int i_local_tag = 0; i_local_tag < atom->nlocal; ++i_local_tag) {
        int jnum = numneigh[i_local_tag];
        for (int m = 0; m < jnum; m++) {
            int j_local_tag = firstneigh[i_local_tag][m];
            // if (j_local_tag < 0 || i_local_tag == j_local_tag) continue; 
            if (i_local_tag == j_local_tag) continue; 
            // i -> j
            full_neigh_temp[i_local_tag].push_back(j_local_tag);
            // j -> i
            full_neigh_temp[j_local_tag].push_back(i_local_tag);
        }
    }
    // 3. 对所有被修改的列表进行去重和排序
    for (int i = 0; i < atom->nlocal; ++i) {
        if (!full_neigh_temp[i].empty()) {
            std::sort(full_neigh_temp[i].begin(), full_neigh_temp[i].end());
            // 去重 (std::unique 要求列表已排序)
            full_neigh_temp[i].erase(
                std::unique(full_neigh_temp[i].begin(), full_neigh_temp[i].end()),
                full_neigh_temp[i].end()
            );
        }
    }

    // a. 计算目标原子组的 Full 邻居总数
    size_t total_full_size = 0;
    for (int gr_i = 0; gr_i < group_count; gr_i++) {
        int i_local_tag = h_group_indices[gr_i];
        total_full_size += full_neigh_temp[i_local_tag].size();
    }
    // b. 分配 Host 数组：大小恰好是 group_count
    int *h_firstneigh_ptrs = new int [total_full_size]; // Full 列表的扁平化数组
    // h_full_group_numneigh 大小为 group_count + 1 (用于存储 group_count 个原子的前缀和)
    size_t write_idx = 0;
    h_group_numneigh[0] = 0;
    // c. 填充 Host 数组（只遍历 group_count 个目标原子）
    for (int gr_i = 0; gr_i < group_count; gr_i++) {
        // 关键：i_local_tag 从 h_group_indices 中取出
        int i_local_tag = h_group_indices[gr_i]; 
        const auto& list_i = full_neigh_temp[i_local_tag]; // 从全局临时容器中取出目标列表
        // 拷贝邻居索引到扁平数组
        memcpy(&(h_firstneigh_ptrs[write_idx]), list_i.data(), list_i.size() * sizeof(int));
        // 更新前缀和： h_group_numneigh 只需要 group_count 个条目
        h_group_numneigh[gr_i + 1] = h_group_numneigh[gr_i] + list_i.size();
        write_idx += list_i.size();
        DEBUG_LOG("Full List: gr_i=%d, Full_jnum=%d, Full_sum=%d", 
                gr_i, (int)list_i.size(), h_group_numneigh[gr_i+1]);
    }
    int *d_firstneigh_ptrs; // 设备端二级指针
    SAFE_CUDA_MALLOC(&d_firstneigh_ptrs, (h_group_numneigh[group_count]) * sizeof(int),file); // 分配设备端指针数组
    DEBUG_LOG("generate d_firstneigh_ptrs, h_group_numneigh[group_count + 1]=%d",h_group_numneigh[group_count]);
    DEBUG_RUN(for (int gr_i = 0; gr_i < group_count; gr_i++) {
        int i_local_tag = h_group_indices[gr_i]; 
        int full_jnum = h_group_numneigh[gr_i+1] - h_group_numneigh[gr_i];
        int half_jnum = numneigh[i_local_tag]; 
        DEBUG_LOG("Comparison for gr_i=%d (Local Tag %d): Half Count=%d, Full Count=%d", 
                gr_i, i_local_tag, half_jnum, full_jnum);
        DEBUG_LOG("d_firstneigh_ptrs list %d" ,h_firstneigh_ptrs[gr_i]);
    });
    SAFE_CUDA_MEMCPY(d_group_numneigh,h_group_numneigh,(group_count + 1)*sizeof(LAMMPS_NS::tagint),cudaMemcpyHostToDevice,file);
    SAFE_CUDA_MEMCPY(d_firstneigh_ptrs,h_firstneigh_ptrs,
        (h_group_numneigh[group_count]) * sizeof(int),cudaMemcpyHostToDevice,file);
    DEBUG_LOG_COND((d_firstneigh_ptrs == NULL),"d_firstneigh_ptrs list not initialized");
    DEBUG_LOG("d_firstneigh_ptrs list %d %d %d" ,h_firstneigh_ptrs[1],h_firstneigh_ptrs[2],h_firstneigh_ptrs[3]);
    DEBUG_LOG("generate end d_firstneigh_ptrs");
    delete[] h_firstneigh_ptrs;
    h_firstneigh_ptrs = nullptr;
    DEBUG_LOG("free h_firstneigh_ptrs");
    delete[] h_group_numneigh;
    h_group_numneigh = nullptr;
    DEBUG_LOG("free h_group_numneigh");
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
    double *h_stein_Ylm = new double [group_count*cutoff_Natoms*(stein_l + 1)*2];
    double *d_stein_ql;
    SAFE_CUDA_MALLOC(&d_stein_ql, group_count*sizeof(double), file);
    double *d_stein_qlm;
    SAFE_CUDA_MALLOC(&d_stein_qlm, (group_count*(stein_l + 1)*2)*sizeof(double), file);
    double *d_stein_Ylm;
    SAFE_CUDA_MALLOC(&d_stein_Ylm, (group_count*cutoff_Natoms*(stein_l + 1)*2)*sizeof(double), file);

    DEBUG_LOG("i will start a kernel of q4");
    cudaDeviceSynchronize(); // waiting memory
    steinhardt_param_calc_kernel_q4<<<block_num,d_block_size>>>(
        group_count, cutoff_Natoms,
        d_neigh_both_in_r_N, d_group_dminneigh,
        d_stein_qlm, d_stein_Ylm,
        d_stein_ql) ;
    
    // double *h_stein_qlm = (double *)malloc(group_count * (stein_l + 1) * 2 * sizeof(double));
    // temp_steinhardt_param_calc_kernel_q4(
    //     file, group_count, cutoff_Natoms, cutoff_r*cutoff_r,
    //     neigh_both_in_r_N, group_dminneigh,
    //     h_stein_qlm, h_stein_Ylm,
    //     stein_ql) ;
    // delete[] h_stein_qlm;
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
    SAFE_CUDA_FREE(d_stein_ql);
    SAFE_CUDA_FREE(d_stein_qlm);
    SAFE_CUDA_FREE(d_stein_Ylm);
    

    for(int c_atom=0;c_atom<group_count;c_atom++)
    {
        DEBUG_LOG("stein_ql[%d] = %f",c_atom,stein_ql[c_atom]);
    }
    delete[] h_stein_Ylm;
}


void MetaD_zqc::STEIN_Q4::summary(FILE* f){}

// void MetaD_zqc::STEIN_Q4::fix_crystallizes_kernel_temp
// (
//     int cutoff_Natoms, double cutoff_rsq, double box_x, double box_y, double box_z,
//     int group_count, int *d_group_indices, LAMMPS_NS::tagint *d_group_numneigh,
//     int *d_firstneigh_ptrs, double *d_x_flat,
//     double *d_group_dminneigh, int *d_neigh_in_cutoff_r, int *d_neigh_both_in_r_N,
//     int atomsnumber
// )
// {
// for(int c_atom=0; c_atom<group_count; ++c_atom){
//     // int c_atom = blockIdx.x * blockDim.x + threadIdx.x;
//     if(c_atom<group_count){
//         double r2,temp_r2,temp_x,temp_y,temp_z,neigh_x,neigh_y,neigh_z;
//         double delt_x,delt_y,delt_z;
//         int c_atom_tag = d_group_indices[c_atom];
//         d_neigh_in_cutoff_r[c_atom] = 0;
//         // c_glob_tag = h_tag[c_atom_tag];
//         double c_x = d_x_flat[c_atom_tag*3];
//         double c_y = d_x_flat[c_atom_tag*3+1];
//         double c_z = d_x_flat[c_atom_tag*3+2];
//         double max_r2 = (box_x+box_y+box_z)*(box_x+box_y+box_z);
//         DEBUG_LOG("now im in %d, c_atom_tag=%d, cx,cy,cz:%f,%f,%f",c_atom,c_atom_tag,c_x,c_y,c_z);
//         for (int i=0;i<cutoff_Natoms;i++){
//             d_group_dminneigh[c_atom*4*cutoff_Natoms +i*4 + 3]=max_r2;
//         }
//         //find curtoff_Natoms neigh
//         for (int neigh_atom=d_group_numneigh[c_atom]; neigh_atom<d_group_numneigh[c_atom+1]; neigh_atom++){
//             int n_local_tag = d_firstneigh_ptrs[neigh_atom];

//         // DEBUG_LOG("all_atoms = %d",atomsnumber);
//         // for (int neigh_atom=0; neigh_atom<atomsnumber; neigh_atom++){
//         //     int n_local_tag = neigh_atom;

//             // if (n_local_tag < 0 ) continue;
//             // int n_glob_tag = h_tag[n_local_tag];
//             neigh_x = d_x_flat[n_local_tag*3+0];
//             neigh_y = d_x_flat[n_local_tag*3+1];
//             neigh_z = d_x_flat[n_local_tag*3+2];
//             delt_x = (neigh_x - c_x);
//             delt_y = (neigh_y - c_y);
//             delt_z = (neigh_z - c_z);
//             // if (delt_x > box_x/2) {
//             //     delt_x -= box_x;
//             // } else if (delt_x < -box_x/2) {
//             //     delt_x += box_x;
//             // }
//             // if (delt_y > box_y/2) {
//             //     delt_y -= box_y;
//             // } else if (delt_y < -box_y/2) {
//             //     delt_y += box_y;
//             // }
//             // if (delt_z > box_z/2) {
//             //     delt_z -= box_z;
//             // } else if (delt_z < -box_z/2) {
//             //     delt_z += box_z;
//             // }
//             lmp->domain->minimum_image(dx, dy, dz);
//             r2 = delt_x*delt_x + delt_y*delt_y + delt_z*delt_z;
//             DEBUG_LOG_COND((r2<10),"c_atom_tag=%d, n_local_tag=%d, nx,ny,nz:%f,%f,%f, r2:%g",c_atom_tag,n_local_tag,delt_x,delt_y,delt_z,r2);
//             if ((r2 > cutoff_rsq )||(r2<1e-12)) continue;
//             d_neigh_in_cutoff_r[c_atom]++;
//             for (int ii=0; ii<cutoff_Natoms; ii++){
//                 if (d_group_dminneigh[c_atom*4*cutoff_Natoms + ii*4 + 3]>r2){
//                     temp_x = d_group_dminneigh[c_atom*4*cutoff_Natoms + ii*4 + 0];
//                     temp_y = d_group_dminneigh[c_atom*4*cutoff_Natoms + ii*4 + 1];
//                     temp_z = d_group_dminneigh[c_atom*4*cutoff_Natoms + ii*4 + 2];
//                     temp_r2 = d_group_dminneigh[c_atom*4*cutoff_Natoms + ii*4 + 3];
//                     d_group_dminneigh[c_atom*4*cutoff_Natoms + ii*4 + 0] = delt_x;
//                     d_group_dminneigh[c_atom*4*cutoff_Natoms + ii*4 + 1] = delt_y;
//                     d_group_dminneigh[c_atom*4*cutoff_Natoms + ii*4 + 2] = delt_z;
//                     d_group_dminneigh[c_atom*4*cutoff_Natoms + ii*4 + 3] = r2;
//                     delt_x = temp_x;
//                     delt_y = temp_y;
//                     delt_z = temp_z;
//                     r2 = temp_r2;
//                 }
//             }
//         }
//         if (d_neigh_in_cutoff_r[c_atom]>=cutoff_Natoms){
//             d_neigh_both_in_r_N[c_atom]=cutoff_Natoms;
//         }
//         else{
//             d_neigh_both_in_r_N[c_atom]=d_neigh_in_cutoff_r[c_atom];
//         }
//     }
// }
// }

__global__ void fix_crystallizes_kernel
(
    int cutoff_Natoms, double cutoff_rsq, double box_x, double box_y, double box_z,
    int group_count, int *d_group_indices, LAMMPS_NS::tagint *d_group_numneigh,
    int *d_firstneigh_ptrs, double *d_x_flat,
    double *d_group_dminneigh, int *d_neigh_in_cutoff_r, int *d_neigh_both_in_r_N
)
{
    int c_atom = blockIdx.x * blockDim.x + threadIdx.x;
    if(c_atom<group_count){
        double r2,temp_r2,temp_x,temp_y,temp_z,neigh_x,neigh_y,neigh_z;
        double delt_x,delt_y,delt_z;
        int c_atom_tag = d_group_indices[c_atom];
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
                    d_group_dminneigh[c_atom*4*cutoff_Natoms + ii*4 + 0] = delt_x;
                    d_group_dminneigh[c_atom*4*cutoff_Natoms + ii*4 + 1] = delt_y;
                    d_group_dminneigh[c_atom*4*cutoff_Natoms + ii*4 + 2] = delt_z;
                    d_group_dminneigh[c_atom*4*cutoff_Natoms + ii*4 + 3] = r2;
                    delt_x = temp_x;
                    delt_y = temp_y;
                    delt_z = temp_z;
                    r2 = temp_r2;
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



void temp_steinhardt_param_calc_kernel_q4(
    FILE *f_check,
    int group_count, int cutoff_Natoms,
    int *d_neigh_both_in_r_N, double *d_group_dminneigh,
    double *d_stein_qlm, double *d_stein_Ylm,
    double *d_stein_ql
)
{
        // steinhardt_param_calc_kernel_q4
        int stein_l=4;
        // init some we need
        int stein_Ylm_base_id, stein_qlm_base_id, neigh_num;
        double delt_x, delt_y, delt_z, r2, r;
        double re_part, im_part;

    for(int c_atom=0;c_atom<group_count;c_atom++){
        int neigh_N = 0;
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
            stein_Ylm_base_id = c_atom*cutoff_Natoms*(stein_l + 1)*2 + neigh_atom*(stein_l + 1)*2;
            DEBUG_LOG("delt x,y,z ; r2, r = %f, %f, %f, %f, %f", delt_x, delt_y, delt_z, r2, r);
            // Y,4,0
            // 3/16*sqrt(1/(1*pi)) = 0.10578554691520430
            d_stein_Ylm[stein_Ylm_base_id + 0] = 0.10578554691520430/(r2*r2) * (35*delt_z*delt_z*delt_z*delt_z - 30*delt_z*delt_z*r2 + 3*r2*r2);
            d_stein_Ylm[stein_Ylm_base_id + 1] = 0;
            // Y,4,1
            // 3/8*sqrt(5/(*pi)) = 0.47308734787878000
            d_stein_Ylm[stein_Ylm_base_id + 2] = -0.47308734787878000/(r2*r2) * (7*delt_z*delt_z - 3*r2) * delt_z * delt_x;
            d_stein_Ylm[stein_Ylm_base_id + 3] = -0.47308734787878000/(r2*r2) * (7*delt_z*delt_z - 3*r2) * delt_z * delt_y;
            // Y,4,2
            // 3/8*sqrt(5/(2*pi)) = 0.33452327177864458
            d_stein_Ylm[stein_Ylm_base_id + 4] = 0.33452327177864458/(r2*r2) * (7*delt_z*delt_z - r2) * (delt_x*delt_x - delt_y*delt_y);
            d_stein_Ylm[stein_Ylm_base_id + 5] = 0.33452327177864458/(r2*r2) * (7*delt_z*delt_z - r2) * (2) * delt_x*delt_y;
            // Y,4,3
            // 3/8*sqrt(35/(pi)) = 1.25167147089835227
            d_stein_Ylm[stein_Ylm_base_id + 6] = -1.25167147089835227*delt_z/(r2*r2) * (delt_x*delt_x*delt_x - 3*delt_x*delt_y*delt_y);
            d_stein_Ylm[stein_Ylm_base_id + 7] = -1.25167147089835227*delt_z/(r2*r2) * (3*delt_x*delt_x*delt_y - delt_y*delt_y*delt_y);
            // Y,4,4
            // 3/16*sqrt(35/(2*pi)) = 0.44253269244498263
            d_stein_Ylm[stein_Ylm_base_id + 8] = 0.44253269244498263/(r2*r2) * (delt_x*delt_x*delt_x*delt_x + delt_y*delt_y*delt_y*delt_y - 6*delt_x*delt_x*delt_y*delt_y);
            d_stein_Ylm[stein_Ylm_base_id + 9] = 0.44253269244498263/(r2*r2) * (4*delt_x*delt_x*delt_x*delt_y - 4*delt_x*delt_y*delt_y*delt_y);
            // DEBUG_LOG("d_stein_Ylm 0, 1, 2, 3, 4 = %f, %f, %f, %f, %f",d_stein_Ylm[stein_Ylm_base_id + 0], d_stein_Ylm[stein_Ylm_base_id + 3], d_stein_Ylm[stein_Ylm_base_id + 4], d_stein_Ylm[stein_Ylm_base_id + 7], d_stein_Ylm[stein_Ylm_base_id + 8]);


            
            // q,4,0
            d_stein_qlm[stein_qlm_base_id + 0] += d_stein_Ylm[stein_Ylm_base_id + 0];
            d_stein_qlm[stein_qlm_base_id + 1] += 0;
            // q,4,1 + q,4,-1 = 2*(Y,4,1)
            d_stein_qlm[stein_qlm_base_id + 2] += d_stein_Ylm[stein_Ylm_base_id + 2];
            d_stein_qlm[stein_qlm_base_id + 3] += d_stein_Ylm[stein_Ylm_base_id + 3];
            // q,4,2 + q,4,-2
            d_stein_qlm[stein_qlm_base_id + 4] += d_stein_Ylm[stein_Ylm_base_id + 4];
            d_stein_qlm[stein_qlm_base_id + 5] += d_stein_Ylm[stein_Ylm_base_id + 5];
            // q,4,3 + q,4,-3
            d_stein_qlm[stein_qlm_base_id + 6] += d_stein_Ylm[stein_Ylm_base_id + 6];
            d_stein_qlm[stein_qlm_base_id + 7] += d_stein_Ylm[stein_Ylm_base_id + 7];
            // q,4,4 + q,4,-4
            d_stein_qlm[stein_qlm_base_id + 8] += d_stein_Ylm[stein_Ylm_base_id + 8];
            d_stein_qlm[stein_qlm_base_id + 9] += d_stein_Ylm[stein_Ylm_base_id + 9];
        }
        DEBUG_LOG("neigh_N = %d",neigh_N);
        for (int i=0;i<(stein_l + 1)*2;i++)
            // qlm = sum(Ylm)/N
            d_stein_qlm[stein_qlm_base_id + i] = d_stein_qlm[stein_qlm_base_id + i]/neigh_N;
        // q
        d_stein_ql[c_atom] = d_stein_qlm[stein_qlm_base_id + 0] * d_stein_qlm[stein_qlm_base_id + 0];
        DEBUG_LOG("d_stein_qlm[%d] = %f + i* %f", stein_qlm_base_id, d_stein_qlm[stein_qlm_base_id + 0],d_stein_qlm[stein_qlm_base_id + 1]);
        for (int i=1;i<(stein_l + 1);i++){
            re_part = d_stein_qlm[stein_qlm_base_id + i*2 + 0];
            im_part = d_stein_qlm[stein_qlm_base_id + i*2 + 1];
            d_stein_ql[c_atom] += 2*(re_part*re_part + im_part*im_part);
            DEBUG_LOG("d_stein_qlm[%d] = %f + i* %f", stein_qlm_base_id + i + 1,d_stein_qlm[stein_qlm_base_id + i*2 + 0],d_stein_qlm[stein_qlm_base_id + i*2 + 1]);
        }
        // DEBUG_LOG("q4 of c_atom[%d] = %f",atom->tag[c_atom],d_stein_ql[c_atom]);
        // 4*pi/(2*1+1) = 4*pi/9 = 1.396263401595463
        d_stein_ql[c_atom] = 1.396263401595463*d_stein_ql[c_atom];
        d_stein_ql[c_atom] = sqrt(d_stein_ql[c_atom]);
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
        double re_part, im_part;

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
            stein_Ylm_base_id = c_atom*cutoff_Natoms*(stein_l + 1)*2 + neigh_atom*(stein_l + 1)*2;
            // DEBUG_LOG("delt x,y,z ; r2, r = %f, %f, %f, %f, %f", delt_x, delt_y, delt_z, r2, r);
            // Y,4,0
            // 3/16*sqrt(1/(1*pi)) = 0.10578554691520430
            d_stein_Ylm[stein_Ylm_base_id + 0] = 0.10578554691520430/(r2*r2) * (35*delt_z*delt_z*delt_z*delt_z - 30*delt_z*delt_z*r2 + 3*r2*r2);
            d_stein_Ylm[stein_Ylm_base_id + 1] = 0;
            // Y,4,1
            // 3/8*sqrt(5/(*pi)) = 0.47308734787878000
            d_stein_Ylm[stein_Ylm_base_id + 2] = -0.47308734787878000/(r2*r2) * (7*delt_z*delt_z - 3*r2) * delt_z * delt_x;
            d_stein_Ylm[stein_Ylm_base_id + 3] = -0.47308734787878000/(r2*r2) * (7*delt_z*delt_z - 3*r2) * delt_z * delt_y;
            // Y,4,2
            // 3/8*sqrt(5/(2*pi)) = 0.33452327177864458
            d_stein_Ylm[stein_Ylm_base_id + 4] = 0.33452327177864458/(r2*r2) * (7*delt_z*delt_z - r2) * (delt_x*delt_x - delt_y*delt_y);
            d_stein_Ylm[stein_Ylm_base_id + 5] = 0.33452327177864458/(r2*r2) * (7*delt_z*delt_z - r2) * (2) * delt_x*delt_y;
            // Y,4,3
            // 3/8*sqrt(35/(pi)) = 1.25167147089835227
            d_stein_Ylm[stein_Ylm_base_id + 6] = -1.25167147089835227*delt_z/(r2*r2) * (delt_x*delt_x*delt_x - 3*delt_x*delt_y*delt_y);
            d_stein_Ylm[stein_Ylm_base_id + 7] = -1.25167147089835227*delt_z/(r2*r2) * (3*delt_x*delt_x*delt_y - delt_y*delt_y*delt_y);
            // Y,4,4
            // 3/16*sqrt(35/(2*pi)) = 0.44253269244498263
            d_stein_Ylm[stein_Ylm_base_id + 8] = 0.44253269244498263/(r2*r2) * (delt_x*delt_x*delt_x*delt_x + delt_y*delt_y*delt_y*delt_y - 6*delt_x*delt_x*delt_y*delt_y);
            d_stein_Ylm[stein_Ylm_base_id + 9] = 0.44253269244498263/(r2*r2) * (4*delt_x*delt_x*delt_x*delt_y - 4*delt_x*delt_y*delt_y*delt_y);
            // DEBUG_LOG("d_stein_Ylm 0, 1, 2, 3, 4 = %f, %f, %f, %f, %f",d_stein_Ylm[stein_Ylm_base_id + 0], d_stein_Ylm[stein_Ylm_base_id + 3], d_stein_Ylm[stein_Ylm_base_id + 4], d_stein_Ylm[stein_Ylm_base_id + 7], d_stein_Ylm[stein_Ylm_base_id + 8]);


            
            // q,4,0
            d_stein_qlm[stein_qlm_base_id + 0] += d_stein_Ylm[stein_Ylm_base_id + 0];
            d_stein_qlm[stein_qlm_base_id + 1] += 0;
            // q,4,1 + q,4,-1 = 2*(Y,4,1)
            d_stein_qlm[stein_qlm_base_id + 2] += d_stein_Ylm[stein_Ylm_base_id + 2];
            d_stein_qlm[stein_qlm_base_id + 3] += d_stein_Ylm[stein_Ylm_base_id + 3];
            // q,4,2 + q,4,-2
            d_stein_qlm[stein_qlm_base_id + 4] += d_stein_Ylm[stein_Ylm_base_id + 4];
            d_stein_qlm[stein_qlm_base_id + 5] += d_stein_Ylm[stein_Ylm_base_id + 5];
            // q,4,3 + q,4,-3
            d_stein_qlm[stein_qlm_base_id + 6] += d_stein_Ylm[stein_Ylm_base_id + 6];
            d_stein_qlm[stein_qlm_base_id + 7] += d_stein_Ylm[stein_Ylm_base_id + 7];
            // q,4,4 + q,4,-4
            d_stein_qlm[stein_qlm_base_id + 8] += d_stein_Ylm[stein_Ylm_base_id + 8];
            d_stein_qlm[stein_qlm_base_id + 9] += d_stein_Ylm[stein_Ylm_base_id + 9];
        }
        for (int i=0;i<(stein_l + 1)*2;i++)
            // qlm = sum(Ylm)/N
            d_stein_qlm[stein_qlm_base_id + i] = d_stein_qlm[stein_qlm_base_id + i]/neigh_num;
        // q
        d_stein_ql[c_atom] = d_stein_qlm[stein_qlm_base_id + 0] * d_stein_qlm[stein_qlm_base_id + 0];
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
