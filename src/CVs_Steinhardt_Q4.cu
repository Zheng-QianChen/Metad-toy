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

// direct_partial dx,dy,dz
// // void dcv_steinhardt_param_calc_kernel_q4(
// //     FILE *f_check,
// __global__ void dcv_steinhardt_param_calc_kernel_q4(
//     int cutoff_Natoms, 
//     int group_count, int groupbit, int *d_mask,
//     LAMMPS_NS::tagint *d_group_indices, LAMMPS_NS::tagint *calculated_numneigh, 
//     int *d_neigh_both_in_r_N, double *d_group_dminneigh,
//     double *d_stein_qlm, double *d_stein_Ylm, double *d_stein_ql,
//     double *d_dYlm_dr,double *d_dcvdx
// )
// {
//     // devise version=============
//     int c_atom = blockIdx.x * blockDim.x + threadIdx.x;
//     if(c_atom<group_count){
//     // host version===============
//     // for (int c_atom=0; c_atom<group_count; c_atom++){
//         int stein_l=4;
//         int neigh_tag, neigh_Nb;
//         double neigh_q4_timesN, catom_q4_timesN;
//         double dx, dy, dz, r2, r, rho;
//         // double theta, phi;
//         // double sin_theta, cos_theta, sin_2theta, cos_2theta;
//         // double sin_3theta, cos_3theta, sin_4theta, cos_4theta;
//         // double sin_phi, cos_phi, sin_2phi, cos_2phi;
//         // double sin_3phi, cos_3phi, sin_4phi, cos_4phi;
//         // double DthetaDx, DthetaDy, DthetaDz, DphiDx, DphiDy, DphiDz;
//         double temp;
//         double tDx, tDy, tDz, tDx_i, tDy_i,tDz_i;
//         int stein_qlm_base_id, stein_qlm_neigh_id, Neigh_Nb;
//         int neigh_num = d_neigh_both_in_r_N[c_atom];
//         catom_q4_timesN = 1.0/(d_stein_ql[c_atom]*neigh_num);
//         for(int i=0; i<3; i++){
//             // from 0 to l, both re_part and im_part
//             d_dcvdx[c_atom*3 + i] = 0;
//             d_dYlm_dr[c_atom*3*2 + i*2 + 0] = 0;
//             d_dYlm_dr[c_atom*3*2 + i*2 + 1] = 0;
//             // DEBUG_LOG("d_stein_qlm[%d] = %f + i* %f", stein_qlm_base_id + i + 1,d_stein_qlm[stein_qlm_base_id + i + 0],d_stein_qlm[stein_qlm_base_id + i + 1]);
//         }
//         for(int neigh_atom=0; neigh_atom<neigh_num; neigh_atom++){
//             dx = d_group_dminneigh[c_atom*cutoff_Natoms*4 + neigh_atom*4 + 0];
//             dy = d_group_dminneigh[c_atom*cutoff_Natoms*4 + neigh_atom*4 + 1];
//             dz = d_group_dminneigh[c_atom*cutoff_Natoms*4 + neigh_atom*4 + 2];
//             r2     = d_group_dminneigh[c_atom*cutoff_Natoms*4 + neigh_atom*4 + 3];
//             r      = sqrt(r2);
//             stein_qlm_base_id = c_atom*(stein_l + 1)*2;
//             // 处理 neigh 与 cv-group 重合的部分
//             neigh_tag = calculated_numneigh[c_atom*cutoff_Natoms + neigh_atom];
//             // 判断当前 neigh atom 是否是 cv-group 的一个原子
//             // 由于 Y,4,m(theta, phi) = Y,4,m(pi-theta,phi), 因此该邻居的Y与本原子的Y表达式相同
//             Neigh_Nb = 0;
//             neigh_q4_timesN = 0;
//             stein_qlm_neigh_id=0;
//             if (d_mask[neigh_tag]&groupbit){
//                 // 使用二分查找法找 neigh_tag 对应在 d_stein_ql 中的位置
//                 int left = 0;
//                 int right = group_count - 1;
//                 // neigh_q4_deN default is 0
//                 while (left <= right) {
//                     int mid = left + (right - left) / 2;
//                     if (d_group_indices[mid] == neigh_tag) {
//                         Neigh_Nb = d_neigh_both_in_r_N[mid];
//                         neigh_q4_timesN = 1.0/(d_stein_ql[mid]*Neigh_Nb);
//                         stein_qlm_neigh_id = Neigh_Nb*cutoff_Natoms*(stein_l + 1)*2 + d_group_indices[c_atom]*(stein_l + 1)*2;
//                         // DEBUG_LOG("mid=%d, stein_qlm_neigh_id=%d",mid,stein_qlm_neigh_id);
//                         break;
//                     } else if (d_group_indices[mid] < neigh_tag) {
//                         left = mid + 1;
//                     } else {
//                         right = mid - 1;
//                     }
//                 }
//             }
//             // d_dcvdx = [dcvdxc, dcvdyc, dcvdzc]*catoms --flatten
//             // d_dYlm_dr = [dYlm_dx_re, dYlm_dx_im, dYlm_dy_re, dYlm_dy_im, dYlm_dz_re, dYlm_dz_im,]*catoms --flatten
//             // Y,4,0
//             // d Y,4,0 dx
//             // 15/(4 Sqrt[\[Pi]])=2.1157109383040860760552979433529
//             temp = 2.1157109383040860760552979433529*((3*POW2(dx)+3*POW2(dy)-4*POW2(dz))*dz)/(POW3(r2));
//             tDx= temp*(dx*dz);
//             tDy= temp*(dy*dz);
//             tDz= temp*(-POW2(dx)-POW2(dy));
//             d_dYlm_dr[c_atom*3*2+0]+= (tDx)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 0] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 0]) ;
//             d_dYlm_dr[c_atom*3*2+1]+= 0 ;
//             // d Y,4,0 dy
//             d_dYlm_dr[c_atom*3*2+2]+= (tDy)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 0] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 0]) ;
//             d_dYlm_dr[c_atom*3*2+3]+= 0 ;
//             // d Y,4,0 dz
//             d_dYlm_dr[c_atom*3*2+4]+= (tDz)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 0] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 0]) ;
//             d_dYlm_dr[c_atom*3*2+5]+= 0 ;
            
//             // Y,4,+-1
//             // 3/32 Sqrt[5/\[Pi]] = 0.11827183696969500226
//             temp  = 0.47308734787878000904634053544357*(1)/(POW3(r2));
//             // tDx   = temp*(-((-21*POW2(dx)*POW2(dz)+POW2(dy)*POW2(dz)+3*POW4(dx)-3*POW4(dy)+4*POW4(dz))*dz));
//             tDx_i = temp*((-3*(POW2(dx)+POW2(dy))+11*POW2(dz))*dx*dy*dz);
//             // tDy   = temp*((-3*(POW2(dx)+POW2(dy))+11*POW2(dz))*dx*dy*dz);
//             tDy_i = temp*(-((POW2(dx)-21*POW2(dy))*POW3(dz))-4*POW5(dz)+3*(POW4(dx)-POW4(dy))*dz);
//             // tDz   = temp*((3*POW2(POW2(dx)+POW2(dy))-21*(POW2(dx)+POW2(dy))*POW2(dz)+4*POW4(dz))*dx);
//             tDz_i = temp*((3*POW2(POW2(dx)+POW2(dy))-21*(POW2(dx)+POW2(dy))*POW2(dz)+4*POW4(dz))*dy);
//             // d Y,4,1 dx
//             // d_dYlm_dr[c_atom*3*2+0]+= (tDx)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 2]) ;
//             // d_dYlm_dr[c_atom*3*2+1]+= (tDx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 3]) ;
//             // d Y,4,1 dy
//             // d_dYlm_dr[c_atom*3*2+2]+= (tDy)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 2]) ;
//             // d_dYlm_dr[c_atom*3*2+3]+= (tDy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 3]) ;
//             // d Y,4,1 dz
//             // d_dYlm_dr[c_atom*3*2+4]+= (tDz)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 2]) ;
//             // d_dYlm_dr[c_atom*3*2+5]+= (tDz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 3]) ;
//             // d Y,4,-1 dx
//             // d_dYlm_dr[c_atom*3*2+0]+= -(tDx)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 2]) ;
//             // d_dYlm_dr[c_atom*3*2+1]+= (tDx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 3]) ;
//             // d Y,4,-1 dy
//             // d_dYlm_dr[c_atom*3*2+2]+= -(tDy)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 2]) ;
//             // d_dYlm_dr[c_atom*3*2+3]+= (tDy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 3]) ;
//             // d Y,4,-1 dz
//             // d_dYlm_dr[c_atom*3*2+4]+= -(tDz)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 2]) ;
//             // d_dYlm_dr[c_atom*3*2+5]+= (tDz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 3]) ;
//             // d Y,4,1 dx + d Y,4,-1 dx
//             d_dYlm_dr[c_atom*3*2+0]+= 0 ;
//             d_dYlm_dr[c_atom*3*2+1]+= 2*(tDx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 3]) ;
//             // d Y,4,1 dx + d Y,4,-1 dy
//             d_dYlm_dr[c_atom*3*2+2]+= 0 ;
//             d_dYlm_dr[c_atom*3*2+3]+= 2*(tDy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 3]) ;
//             // d Y,4,1 dx + d Y,4,-1 dz
//             d_dYlm_dr[c_atom*3*2+4]+= 0 ;
//             d_dYlm_dr[c_atom*3*2+5]+= 2*(tDz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 3]) ;

//             // Y,4,+-2
//             // 3/2 Sqrt[5/2*\[Pi]] = 0.66904654355728916795211238971191
//             temp = 1.3380930871145783359042247794238*(1)/(POW3(r2));
//             tDx   = temp*(-((-9*POW2(dy)*POW2(dz)+POW2(dx)*(POW2(dy)+4*POW2(dz))+POW4(dy)-3*POW4(dz))*dx));
//             // tDx_i = temp*((-21*POW2(dx)*POW2(dz)+5*POW2(dy)*POW2(dz)+POW4(dx)-POW4(dy)+6*POW4(dz))*dy);
//             tDy   = temp*((POW2(dx)*(POW2(dy)-9*POW2(dz))+4*POW2(dy)*POW2(dz)+POW4(dx)-3*POW4(dz))*dy);
//             // tDy_i = temp*(((5*POW2(dx)-21*POW2(dy))*POW2(dz)-POW4(dx)+POW4(dy)+6*POW4(dz))*dx);
//             tDz   = temp*((4*(POW2(dx)+POW2(dy))-3*POW2(dz))*(dx-dy)*(dx+dy)*dz);
//             // tDz_i = temp*((4*(POW2(dx)+POW2(dy))-3*POW2(dz))*dx*dy*dz);
//             // d Y,4,2 dx
//             // d_dYlm_dr[c_atom*3*2+0]+= (tDx)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 4]) ;
//             // d_dYlm_dr[c_atom*3*2+1]+= (tDx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 5]) ;
//             // d Y,4,2 dy
//             // d_dYlm_dr[c_atom*3*2+2]+= (tDy)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 4]) ;
//             // d_dYlm_dr[c_atom*3*2+3]+= (tDy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 5]) ;
//             // d Y,4,2 dz
//             // d_dYlm_dr[c_atom*3*2+4]+= (tDz)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 4]) ;
//             // d_dYlm_dr[c_atom*3*2+5]+= (tDz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 5]) ;
//             // d Y,4,-2 dx
//             // d_dYlm_dr[c_atom*3*2+0]+= (tDx)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 4]) ;
//             // d_dYlm_dr[c_atom*3*2+1]+= -(tDx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 5]) ;
//             // d Y,4,-2 dy
//             // d_dYlm_dr[c_atom*3*2+2]+= (tDy)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 4]) ;
//             // d_dYlm_dr[c_atom*3*2+3]+= -(tDy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 5]) ;
//             // d Y,4,-2 dz
//             // d_dYlm_dr[c_atom*3*2+4]+= (tDz)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 4]) ;
//             // d_dYlm_dr[c_atom*3*2+5]+= -(tDz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 5]) ;
//             // d Y,4,2 dx + d Y,4,-2 dx
//             d_dYlm_dr[c_atom*3*2+0]+= 2*(tDx)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 4]) ;
//             d_dYlm_dr[c_atom*3*2+1]+= 0 ;
//             // d Y,4,2 dx + d Y,4,-2 dy
//             d_dYlm_dr[c_atom*3*2+2]+= 2*(tDy)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 4]) ;
//             d_dYlm_dr[c_atom*3*2+3]+= 0 ;
//             // d Y,4,2 dx + d Y,4,-2 dz
//             d_dYlm_dr[c_atom*3*2+4]+= 2*(tDz)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 4]) ;
//             d_dYlm_dr[c_atom*3*2+5]+= 0 ;

//             // Y,4,+-3
//             // 3/8 Sqrt[35/\[Pi]] = 1.2516714708983522692
//             temp  = 1.2516714708983522691732810472102*(1)/(POW3(r2));
//             // tDx   = temp*((3*POW2(dy)*(POW2(dy)+POW2(dz))-3*POW2(dx)*(4*POW2(dy)+POW2(dz))+POW4(dx))*dz);
//             tDx_i = temp*((3*POW2(dx)-5*POW2(dy)-3*POW2(dz))*dx*dy*dz);
//             // tDy   = temp*((5*POW2(dx)-3*POW2(dy)+3*POW2(dz))*dx*dy*dz);
//             tDy_i = temp*(-((-3*POW2(dy)*POW2(dz)+3*POW2(dx)*(-4*POW2(dy)+POW2(dz))+3*POW4(dx)+POW4(dy))*dz));
//             // tDz   = temp*(-((POW2(dx)-3*POW2(dy))*(POW2(dx)+POW2(dy)-3*POW2(dz))*dx));
//             tDz_i = temp*((-3*POW2(dx)+POW2(dy))*(POW2(dx)+POW2(dy)-3*POW2(dz))*dy);
//             // d Y,4,3 dx
//             // d_dYlm_dr[c_atom*3*2+0]+= (tDx)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 6]) ;
//             // d_dYlm_dr[c_atom*3*2+1]+= (tDx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 7]) ;
//             // d Y,4,3 dy
//             // d_dYlm_dr[c_atom*3*2+2]+= (tDy)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 6]) ;
//             // d_dYlm_dr[c_atom*3*2+3]+= (tDy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 7]) ;
//             // d Y,4,3 dz
//             // d_dYlm_dr[c_atom*3*2+4]+= (tDz)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 6]) ;
//             // d_dYlm_dr[c_atom*3*2+5]+= (tDz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 7]) ;
//             // d Y,4,-3 dx
//             // d_dYlm_dr[c_atom*3*2+0]+= -(tDx)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 6]) ;
//             // d_dYlm_dr[c_atom*3*2+1]+= (tDx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 7]) ;
//             // d Y,4,-3 dy
//             // d_dYlm_dr[c_atom*3*2+2]+= -(tDy)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 6]) ;
//             // d_dYlm_dr[c_atom*3*2+3]+= (tDy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 7]) ;
//             // d Y,4,-3 dz
//             // d_dYlm_dr[c_atom*3*2+4]+= -(tDz)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 6]) ;
//             // d_dYlm_dr[c_atom*3*2+5]+= (tDz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 7]) ;
//             // d Y,4,3 dx + d Y,4,-3 dx
//             d_dYlm_dr[c_atom*3*2+0]+= 0 ;
//             d_dYlm_dr[c_atom*3*2+1]+= 2*(tDx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 7]) ;
//             // d Y,4,3 dx + d Y,4,-3 dy
//             d_dYlm_dr[c_atom*3*2+2]+= 0 ;
//             d_dYlm_dr[c_atom*3*2+3]+= 2*(tDy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 7]) ;
//             // d Y,4,3 dx + d Y,4,-3 dz
//             d_dYlm_dr[c_atom*3*2+4]+= 0 ;
//             d_dYlm_dr[c_atom*3*2+5]+= 2*(tDz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 7]) ;
            
//             // Y,4,+-4
//             // 3/4 Sqrt[35/(2 \[Pi])] = 1.7701307697799305310
//             temp = 1.7701307697799305310368308326245*(1)/(POW3(r2));
//             tDx   = temp*((-3*POW2(dy)*POW2(dz)+POW2(dx)*(4*POW2(dy)+POW2(dz))-4*POW4(dy))*dx);
//             // tDx_i = temp*(-((POW2(dy)*(POW2(dy)+POW2(dz))-3*POW2(dx)*(2*POW2(dy)+POW2(dz))+POW4(dx))*dy));
//             tDy   = temp*((POW2(dx)*(4*POW2(dy)-3*POW2(dz))+POW2(dy)*POW2(dz)-4*POW4(dx))*dy);
//             // tDy_i = temp*((-3*POW2(dy)*POW2(dz)+POW2(dx)*(-6*POW2(dy)+POW2(dz))+POW4(dx)+POW4(dy))*dx);
//             tDz   = temp*(-((-6*POW2(dx)*POW2(dy)+POW4(dx)+POW4(dy))*dz));
//             // tDz_i = temp*((-POW2(dx)+POW2(dy))*dx*dy*dz);
//             // d Y,4,2 dx
//             // d_dYlm_dr[c_atom*3*2+0]+= (tDx)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 8]) ;
//             // d_dYlm_dr[c_atom*3*2+1]+= (tDx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 9]) ;
//             // d Y,4,2 dy
//             // d_dYlm_dr[c_atom*3*2+2]+= (tDy)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 8]) ;
//             // d_dYlm_dr[c_atom*3*2+3]+= (tDy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 9]) ;
//             // d Y,4,2 dz
//             // d_dYlm_dr[c_atom*3*2+4]+= (tDz)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 8]) ;
//             // d_dYlm_dr[c_atom*3*2+5]+= (tDz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 9]) ;
//             // d Y,4,-2 dx
//             // d_dYlm_dr[c_atom*3*2+0]+= (tDx)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 8]) ;
//             // d_dYlm_dr[c_atom*3*2+1]+= -(tDx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 9]) ;
//             // d Y,4,-2 dy
//             // d_dYlm_dr[c_atom*3*2+2]+= (tDy)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 8]) ;
//             // d_dYlm_dr[c_atom*3*2+3]+= -(tDy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 9]) ;
//             // d Y,4,-2 dz
//             // d_dYlm_dr[c_atom*3*2+4]+= (tDz)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 8]) ;
//             // d_dYlm_dr[c_atom*3*2+5]+= -(tDz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 9]) ;
//             // d Y,4,2 dx + d Y,4,-2 dx
//             d_dYlm_dr[c_atom*3*2+0]+= 2*(tDx)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 8]) ;
//             d_dYlm_dr[c_atom*3*2+1]+= 0 ;
//             // d Y,4,2 dx + d Y,4,-2 dy
//             d_dYlm_dr[c_atom*3*2+2]+= 2*(tDy)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 8]) ;
//             d_dYlm_dr[c_atom*3*2+3]+= 0 ;
//             // d Y,4,2 dx + d Y,4,-2 dz
//             d_dYlm_dr[c_atom*3*2+4]+= 2*(tDz)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_q4_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 8]) ;
//             d_dYlm_dr[c_atom*3*2+5]+= 0 ;
//         }
//         for (int i=0;i<3;i++){
//             d_dcvdx[c_atom*3+i] = d_dYlm_dr[c_atom*3*2 + i*2 + 0] + d_dYlm_dr[c_atom*3*2 + i*2 + 1];
//             d_dcvdx[c_atom*3+i] = -(d_dcvdx[c_atom*3+i]*2*PI)/(group_count*(2*stein_l+1));
//         }
//         // DEBUG_LOG("catom=%d, d_dcvdx[%d] dx, dy, dz = %g, %g, %g", c_atom, c_atom*3, d_dcvdx[c_atom*3+0], d_dcvdx[c_atom*3+1], d_dcvdx[c_atom*3+2]);
//     }
// }


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
        double neigh_q6_timesN, catom_q4_timesN;
        double dx, dy, dz, r2, r;
        double theta, phi;
        double sin_theta, cos_theta, sin_2theta, cos_2theta;
        double sin_3theta, cos_3theta, sin_4theta, cos_4theta;
        double sin_5theta, cos_5theta, sin_6theta, cos_6theta;
        double sin_phi, cos_phi, sin_2phi, cos_2phi;
        double sin_3phi, cos_3phi, sin_4phi, cos_4phi;
        double sin_5phi, cos_5phi;
        // double sin_6phi, cos_6phi;
        double Factor_Y, Factor_Ydx, Factor_Ydy, Factor_Ydz;
        double tdx_r, tdy_r, tdz_r, tdx_i, tdy_i,tdz_i;
        int stein_qlm_base_id, stein_qlm_neigh_id, Neigh_Nb;
        int neigh_num = d_neigh_both_in_r_N[c_atom];
        catom_q4_timesN = 1.0/(d_stein_ql[c_atom]*neigh_num);
        for(int i=0; i<3; i++){
            // from 0 to l, both re_part and im_part
            d_dcvdx[c_atom*3 + i] = 0;
            d_dYlm_dr[c_atom*3*2 + i*2 + 0] = 0;
            d_dYlm_dr[c_atom*3*2 + i*2 + 1] = 0;
            // DEBUG_LOG("d_stein_qlm[%d] = %f + i* %f", stein_qlm_base_id + i + 1,d_stein_qlm[stein_qlm_base_id + i + 0],d_stein_qlm[stein_qlm_base_id + i + 1]);
        }
        for(int neigh_atom=0; neigh_atom<neigh_num; neigh_atom++){
            dx = d_group_dminneigh[c_atom*cutoff_Natoms*4 + neigh_atom*4 + 0];
            dy = d_group_dminneigh[c_atom*cutoff_Natoms*4 + neigh_atom*4 + 1];
            dz = d_group_dminneigh[c_atom*cutoff_Natoms*4 + neigh_atom*4 + 2];
            r2     = d_group_dminneigh[c_atom*cutoff_Natoms*4 + neigh_atom*4 + 3];
            r      = sqrt(r2);
            theta = acos(dz/r);
            phi = atan2(dy, dx);
            sincos(theta, &sin_theta, &cos_theta);
            sincos(2*theta, &sin_2theta, &cos_2theta);
            sincos(3*theta, &sin_3theta, &cos_3theta);
            sincos(4*theta, &sin_4theta, &cos_4theta);
            sincos(5*theta, &sin_5theta, &cos_5theta);
            // sincos(6*theta, &sin_6theta, &cos_6theta);
            sincos(phi, &sin_phi, &cos_phi);
            sincos(2*phi, &sin_2phi, &cos_2phi);
            sincos(3*phi, &sin_3phi, &cos_3phi);
            sincos(4*phi, &sin_4phi, &cos_4phi);
            sincos(5*phi, &sin_5phi, &cos_5phi);
            // sincos(6*phi, &sin_6phi, &cos_6phi);
            stein_qlm_base_id = c_atom*(stein_l + 1)*2;
            // 处理 neigh 与 cv-group 重合的部分
            neigh_tag = calculated_numneigh[c_atom*cutoff_Natoms + neigh_atom];
            // 判断当前 neigh atom 是否是 cv-group 的一个原子
            // 由于 Y,6,m(theta, phi) = Y,6,m(pi-theta,phi), 因此该邻居的Y与本原子的Y表达式相同
            Neigh_Nb = 0;
            neigh_q6_timesN = 0;
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
                        neigh_q6_timesN = 1.0/(d_stein_ql[mid]*Neigh_Nb);
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
            Factor_Y = ((1.057855469152043*cos_theta*(1.+7.*cos_2theta)*sin_theta)/r);
            Factor_Ydx = -1.*cos_phi*cos_theta;
            tdx_r = 1.;
            tdx_i = 0;
            Factor_Ydy = -1.*cos_theta*sin_phi;
            tdy_r = 1.;
            tdy_i = 0;
            Factor_Ydz = sin_theta;
            tdz_r = 1.;
            tdz_i = 0;
            // d Y,4,0 dx
            d_dYlm_dr[c_atom*3*2+0]+= (tdx_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 0] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 0]) ;
            d_dYlm_dr[c_atom*3*2+1]+= 0 ;
            // d Y,4,0 dy
            d_dYlm_dr[c_atom*3*2+2]+= (tdy_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 0] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 0]) ;
            d_dYlm_dr[c_atom*3*2+3]+= 0 ;
            // d Y,4,0 dz
            d_dYlm_dr[c_atom*3*2+4]+= (tdz_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 0] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 0]) ;
            d_dYlm_dr[c_atom*3*2+5]+= 0 ;
            
            // Y,4,+-1
            Factor_Y = (0.23654367393939/r);
            Factor_Ydx = 2.*cos_theta;
            // tdx_r = 0.5*((-4.+3.*cos_2phi)*cos_2theta-7.*POW2(cos_phi)*cos_4theta-1.*POW2(sin_phi));
            tdx_i = (4.+7.*cos_2theta)*sin_2phi*POW2(sin_theta);
            Factor_Ydy = 2.*cos_theta;
            // tdy_r = (4.+7.*cos_2theta)*sin_2phi*POW2(sin_theta);
            tdy_i = 0.5*(-1.*POW2(cos_phi)-1.*(4.+3.*cos_2phi)*cos_2theta-7.*cos_4theta*POW2(sin_phi));
            Factor_Ydz = (cos_2theta+7.*cos_4theta)*sin_theta;
            // tdz_r = cos_phi;
            tdz_i = sin_phi;
            // d Y,4,1 dx
            // d_dYlm_dr[c_atom*3*2+0]+= (tdx_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 2]) ;
            // d_dYlm_dr[c_atom*3*2+1]+= (tdx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 3]) ;
            // d Y,4,1 dy
            // d_dYlm_dr[c_atom*3*2+2]+= (tdy_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 2]) ;
            // d_dYlm_dr[c_atom*3*2+3]+= (tdy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 3]) ;
            // d Y,4,1 dz
            // d_dYlm_dr[c_atom*3*2+4]+= (tdz_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 2]) ;
            // d_dYlm_dr[c_atom*3*2+5]+= (tdz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 3]) ;
            // d Y,4,-1 dx
            // d_dYlm_dr[c_atom*3*2+0]+= -(tdx_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 2]) ;
            // d_dYlm_dr[c_atom*3*2+1]+= (tdx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 3]) ;
            // d Y,4,-1 dy
            // d_dYlm_dr[c_atom*3*2+2]+= -(tdy_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 2]) ;
            // d_dYlm_dr[c_atom*3*2+3]+= (tdy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 3]) ;
            // d Y,4,-1 dz
            // d_dYlm_dr[c_atom*3*2+4]+= -(tdz_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 2]) ;
            // d_dYlm_dr[c_atom*3*2+5]+= (tdz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 3]) ;
            // d Y,4,1 dx + d Y,4,-1 dx
            d_dYlm_dr[c_atom*3*2+0]+= 0 ;
            d_dYlm_dr[c_atom*3*2+1]+= 2*(tdx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 3]) ;
            // d Y,4,1 dx + d Y,4,-1 dy
            d_dYlm_dr[c_atom*3*2+2]+= 0 ;
            d_dYlm_dr[c_atom*3*2+3]+= 2*(tdy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 3]) ;
            // d Y,4,1 dx + d Y,4,-1 dz
            d_dYlm_dr[c_atom*3*2+4]+= 0 ;
            d_dYlm_dr[c_atom*3*2+5]+= 2*(tdz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 3]) ;

            // Y,4,+-2
            Factor_Y = (0.3345232717786446/r);
            Factor_Ydx = 1.;
            tdx_r = (5.+7.*cos_2theta)*sin_phi*sin_2phi*sin_theta+cos_phi*cos_2phi*POW2(cos_theta)*(-9.*sin_theta+7.*sin_3theta);
            // tdx_i = 0.125*(-1.*sin_3phi*(8.*sin_theta+9.*sin_3theta)+sin_phi*(4.*sin_theta+19.*sin_3theta+28.*POW2(cos_phi)*sin_5theta));
            Factor_Ydy = sin_theta;
            tdy_r = -0.25*(15.+26.*cos_2theta+7.*cos_4theta)*sin_phi-1.*(6.+7.*cos_2theta)*sin_3phi*POW2(sin_theta);
            // tdy_i = cos_phi*(5.*POW2(cos_phi)+(6.+cos_2phi)*cos_2theta+7.*cos_4theta*POW2(sin_phi));
            Factor_Ydz = 2.*cos_theta*(-1.+7.*cos_2theta)*POW2(sin_theta);
            tdz_r = -1.*cos_2phi;
            // tdz_i = -2.*cos_phi*sin_phi;
            // d Y,4,2 dx
            // d_dYlm_dr[c_atom*3*2+0]+= (tdx_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 4]) ;
            // d_dYlm_dr[c_atom*3*2+1]+= (tdx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 5]) ;
            // d Y,4,2 dy
            // d_dYlm_dr[c_atom*3*2+2]+= (tdy_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 4]) ;
            // d_dYlm_dr[c_atom*3*2+3]+= (tdy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 5]) ;
            // d Y,4,2 dz
            // d_dYlm_dr[c_atom*3*2+4]+= (tdz_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 4]) ;
            // d_dYlm_dr[c_atom*3*2+5]+= (tdz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 5]) ;
            // d Y,4,-2 dx
            // d_dYlm_dr[c_atom*3*2+0]+= (tdx_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 4]) ;
            // d_dYlm_dr[c_atom*3*2+1]+= -(tdx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 5]) ;
            // d Y,4,-2 dy
            // d_dYlm_dr[c_atom*3*2+2]+= (tdy_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 4]) ;
            // d_dYlm_dr[c_atom*3*2+3]+= -(tdy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 5]) ;
            // d Y,4,-2 dz
            // d_dYlm_dr[c_atom*3*2+4]+= (tdz_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 4]) ;
            // d_dYlm_dr[c_atom*3*2+5]+= -(tdz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 5]) ;
            // d Y,4,2 dx + d Y,4,-2 dx
            d_dYlm_dr[c_atom*3*2+0]+= 2*(tdx_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 4]) ;
            d_dYlm_dr[c_atom*3*2+1]+= 0 ;
            // d Y,4,2 dx + d Y,4,-2 dy
            d_dYlm_dr[c_atom*3*2+2]+= 2*(tdy_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 4]) ;
            d_dYlm_dr[c_atom*3*2+3]+= 0 ;
            // d Y,4,2 dx + d Y,4,-2 dz
            d_dYlm_dr[c_atom*3*2+4]+= 2*(tdz_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 4]) ;
            d_dYlm_dr[c_atom*3*2+5]+= 0 ;

            // Y,4,+-3
            Factor_Y = ((1.251671470898352*POW2(sin_theta))/r);
            Factor_Ydx = cos_theta;
            // tdx_r = -2.*cos_2phi+cos_4phi+2.*POW2(cos_phi)*(1.-2.*cos_2phi)*cos_2theta;
            tdx_i = 3.*cos_3phi*sin_phi-1.*cos_phi*(1.+2.*cos_2theta)*sin_3phi;
            Factor_Ydy = cos_theta;
            // tdy_r = -1.*cos_3phi*(1.+2.*cos_2theta)*sin_phi+3.*cos_phi*sin_3phi;
            tdy_i = -2.*cos_2phi-1.*cos_4phi-2.*(1.+2.*cos_2phi)*cos_2theta*POW2(sin_phi);
            Factor_Ydz = sin_3theta;
            // tdz_r = cos_3phi;
            tdz_i = sin_3phi;
            // d Y,4,3 dx
            // d_dYlm_dr[c_atom*3*2+0]+= (tdx_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 6]) ;
            // d_dYlm_dr[c_atom*3*2+1]+= (tdx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 7]) ;
            // d Y,4,3 dy
            // d_dYlm_dr[c_atom*3*2+2]+= (tdy_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 6]) ;
            // d_dYlm_dr[c_atom*3*2+3]+= (tdy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 7]) ;
            // d Y,4,3 dz
            // d_dYlm_dr[c_atom*3*2+4]+= (tdz_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 6]) ;
            // d_dYlm_dr[c_atom*3*2+5]+= (tdz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 7]) ;
            // d Y,4,-3 dx
            // d_dYlm_dr[c_atom*3*2+0]+= -(tdx_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 6]) ;
            // d_dYlm_dr[c_atom*3*2+1]+= (tdx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 7]) ;
            // d Y,4,-3 dy
            // d_dYlm_dr[c_atom*3*2+2]+= -(tdy_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 6]) ;
            // d_dYlm_dr[c_atom*3*2+3]+= (tdy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 7]) ;
            // d Y,4,-3 dz
            // d_dYlm_dr[c_atom*3*2+4]+= -(tdz_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 6]) ;
            // d_dYlm_dr[c_atom*3*2+5]+= (tdz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 7]) ;
            // d Y,4,3 dx + d Y,4,-3 dx
            d_dYlm_dr[c_atom*3*2+0]+= 0 ;
            d_dYlm_dr[c_atom*3*2+1]+= 2*(tdx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 7]) ;
            // d Y,4,3 dx + d Y,4,-3 dy
            d_dYlm_dr[c_atom*3*2+2]+= 0 ;
            d_dYlm_dr[c_atom*3*2+3]+= 2*(tdy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 7]) ;
            // d Y,4,3 dx + d Y,4,-3 dz
            d_dYlm_dr[c_atom*3*2+4]+= 0 ;
            d_dYlm_dr[c_atom*3*2+5]+= 2*(tdz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 7]) ;
            
            // Y,4,+-4
            Factor_Y = ((1.770130769779931*POW3(sin_theta))/r);
            Factor_Ydx = 1.;
            tdx_r = cos_phi*cos_4phi*POW2(cos_theta)+sin_phi*sin_4phi;
            // tdx_i = -1.*cos_4phi*sin_phi+cos_phi*POW2(cos_theta)*sin_4phi;
            Factor_Ydy = 1.;
            tdy_r = cos_4phi*POW2(cos_theta)*sin_phi-1.*cos_phi*sin_4phi;
            // tdy_i = cos_phi*cos_4phi+POW2(cos_theta)*sin_phi*sin_4phi;
            Factor_Ydz = cos_theta*sin_theta;
            tdz_r = -1.*cos_4phi;
            // tdz_i = -1.*sin_4phi;
            // d Y,4,4 dx
            // d_dYlm_dr[c_atom*3*2+0]+= (tdx_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 8]) ;
            // d_dYlm_dr[c_atom*3*2+1]+= (tdx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 9]) ;
            // d Y,4,4 dy
            // d_dYlm_dr[c_atom*3*2+2]+= (tdy_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 8]) ;
            // d_dYlm_dr[c_atom*3*2+3]+= (tdy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 9]) ;
            // d Y,4,4 dz
            // d_dYlm_dr[c_atom*3*2+4]+= (tdz_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 8]) ;
            // d_dYlm_dr[c_atom*3*2+5]+= (tdz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 9]) ;
            // d Y,4,-4 dx
            // d_dYlm_dr[c_atom*3*2+0]+= (tdx_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 8]) ;
            // d_dYlm_dr[c_atom*3*2+1]+= -(tdx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 9]) ;
            // d Y,4,-4 dy
            // d_dYlm_dr[c_atom*3*2+2]+= (tdy_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 8]) ;
            // d_dYlm_dr[c_atom*3*2+3]+= -(tdy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 9]) ;
            // d Y,4,-4 dz
            // d_dYlm_dr[c_atom*3*2+4]+= (tdz_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 8]) ;
            // d_dYlm_dr[c_atom*3*2+5]+= -(tdz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 9]) ;
            // d Y,4,4 dx + d Y,4,-4 dx
            d_dYlm_dr[c_atom*3*2+0]+= 2*(tdx_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 8]) ;
            d_dYlm_dr[c_atom*3*2+1]+= 0 ;
            // d Y,4,4 dx + d Y,4,-4 dy
            d_dYlm_dr[c_atom*3*2+2]+= 2*(tdy_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 8]) ;
            d_dYlm_dr[c_atom*3*2+3]+= 0 ;
            // d Y,4,4 dx + d Y,4,-4 dz
            d_dYlm_dr[c_atom*3*2+4]+= 2*(tdz_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 8]) ;
            d_dYlm_dr[c_atom*3*2+5]+= 0 ;
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
        double sin_theta, cos_theta, sin_2theta, cos_2theta;
        double sin_3theta, cos_3theta, sin_4theta, cos_4theta;
        double sin_phi, cos_phi, sin_2phi, cos_2phi;
        double sin_3phi, cos_3phi, sin_4phi, cos_4phi;
        double re_part, im_part;
        double temp_value, temp4pi_2lplus1;
        // d_stein_ql[c_atom] = 322;
        // 4*pi/(2*1+1)
        // 4*pi = 12.5663706143591729538505735331
        temp4pi_2lplus1 = 12.5663706143591729538505735331/(2*stein_l+1);

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
            theta = acos(delt_z/r);
            phi = atan2(delt_y, delt_x);
            sincos(theta, &sin_theta, &cos_theta);
            sincos(2*theta, &sin_2theta, &cos_2theta);
            sincos(3*theta, &sin_3theta, &cos_3theta);
            sincos(4*theta, &sin_4theta, &cos_4theta);
            sincos(phi, &sin_phi, &cos_phi);
            sincos(2*phi, &sin_2phi, &cos_2phi);
            sincos(3*phi, &sin_3phi, &cos_3phi);
            sincos(4*phi, &sin_4phi, &cos_4phi);
            stein_Ylm_base_id = c_atom*cutoff_Natoms*(stein_l + 1)*2 + neigh_atom*(stein_l + 1)*2;
            // DEBUG_LOG("delt x,y,z ; r2, r = %f, %f, %f, %f, %f", delt_x, delt_y, delt_z, r2, r);
            // Y,4,0
            // 3/16*sqrt(1/(1*pi)) = 0.10578554691520430
            d_stein_Ylm[stein_Ylm_base_id + 0] = 0.10578554691520430*(3-30*POW2(cos_theta)+35*POW4(cos_theta));
            d_stein_Ylm[stein_Ylm_base_id + 1] = 0;
            // Y,4,1
            // Y,4,-1, -Re+Im
            // 3/8*sqrt(5/(1*pi)) = 0.47308734787878000
            temp_value = 0.47308734787878000*cos_theta*(-3+7*POW2(cos_theta))*sin_theta;
            d_stein_Ylm[stein_Ylm_base_id + 2] = - temp_value * cos_phi;
            d_stein_Ylm[stein_Ylm_base_id + 3] = - temp_value * sin_phi;
            // Y,4,2
            // Y,4,-2, Re-Im
            // 3/8*sqrt(5/(2*pi)) = 0.33452327177864458
            temp_value = 0.33452327177864458*(-1+7*POW2(cos_theta))*POW2(sin_theta);
            d_stein_Ylm[stein_Ylm_base_id + 4] = temp_value * cos_2phi;
            d_stein_Ylm[stein_Ylm_base_id + 5] = temp_value * sin_2phi;
            // Y,4,3
            // Y,4,-3, -Re+Im
            // 3/8*sqrt(35/(pi)) = 1.25167147089835227
            temp_value = 1.25167147089835227*cos_theta*POW3(sin_theta);
            d_stein_Ylm[stein_Ylm_base_id + 6] = - temp_value * cos_3phi;
            d_stein_Ylm[stein_Ylm_base_id + 7] = - temp_value * sin_3phi;
            // Y,4,4
            // Y,4,-4, Re-Im
            // 3/16*sqrt(35/(2*pi)) = 0.44253269244498263
            temp_value = 0.44253269244498263*POW4(sin_theta);
            d_stein_Ylm[stein_Ylm_base_id + 8] = temp_value * cos_4phi;
            d_stein_Ylm[stein_Ylm_base_id + 9] = temp_value * sin_4phi;
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
        d_stein_ql[c_atom] = d_stein_ql[c_atom]*temp4pi_2lplus1;
        d_stein_ql[c_atom] = sqrt(d_stein_ql[c_atom]);
    }
}
