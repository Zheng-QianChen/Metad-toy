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

// void dcv_steinhardt_param_calc_kernel_q6(
//     FILE *f_check,
__global__ void dcv_steinhardt_param_calc_kernel_q6(
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
        int stein_l=6;
        int neigh_tag, neigh_Nb;
        double neigh_q6_timesN, catom_q4_timesN;
        double dx, dy, dz, r2, r;
        double theta, phi;
        double sin_theta, cos_theta, sin_2theta, cos_2theta;
        double sin_3theta, cos_3theta, sin_4theta, cos_4theta;
        double sin_5theta, cos_5theta, sin_6theta, cos_6theta;
        double sin_phi, cos_phi, sin_2phi, cos_2phi;
        double sin_3phi, cos_3phi, sin_4phi, cos_4phi;
        double sin_5phi, cos_5phi, sin_6phi, cos_6phi;
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
            sincos(6*theta, &sin_6theta, &cos_6theta);
            sincos(phi, &sin_phi, &cos_phi);
            sincos(2*phi, &sin_2phi, &cos_2phi);
            sincos(3*phi, &sin_3phi, &cos_3phi);
            sincos(4*phi, &sin_4phi, &cos_4phi);
            sincos(5*phi, &sin_5phi, &cos_5phi);
            sincos(6*phi, &sin_6phi, &cos_6phi);
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
            // Y,6,0
            Factor_Y = (1.);
            Factor_Ydx = 1.;
            tdx_r = Factor_Y*Factor_Ydx*(-0.3337383119050492*cos_phi*POW2(cos_theta)*(19.+12.*cos_2theta+33.*cos_4theta)*sin_theta)/r;
            tdx_i = Factor_Y*Factor_Ydx*0;
            Factor_Ydy = 1.;
            tdy_r = Factor_Y*Factor_Ydy*(-0.3337383119050492*POW2(cos_theta)*(19.+12.*cos_2theta+33.*cos_4theta)*sin_phi*sin_theta)/r;
            tdy_i = Factor_Y*Factor_Ydy*0;
            Factor_Ydz = 1.;
            tdz_r = Factor_Y*Factor_Ydz*(0.3337383119050492*cos_theta*(19.+12.*cos_2theta+33.*cos_4theta)*POW2(sin_theta))/r;
            tdz_i = Factor_Y*Factor_Ydz*0;
            // d Y,6,0 dx
            d_dYlm_dr[c_atom*3*2+0]+= (tdx_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 0] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 0]) ;
            d_dYlm_dr[c_atom*3*2+1]+= 0 ;
            // d Y,6,0 dy
            d_dYlm_dr[c_atom*3*2+2]+= (tdy_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 0] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 0]) ;
            d_dYlm_dr[c_atom*3*2+3]+= 0 ;
            // d Y,6,0 dz
            d_dYlm_dr[c_atom*3*2+4]+= (tdz_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 0] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 0]) ;
            d_dYlm_dr[c_atom*3*2+5]+= 0 ;
            
            // Y,6,+-1
            Factor_Y = (1.);
            Factor_Ydx = 1.;
            // tdx_r = Factor_Y*Factor_Ydx*(-0.0257484697688213*cos_theta*(POW2(cos_phi)*(5.*cos_2theta+24.*cos_4theta+99.*cos_6theta)+2.*(19.+12.*cos_2theta+33.*cos_4theta)*POW2(sin_phi)))/r;
            tdx_i = Factor_Y*Factor_Ydx*(0.1029938790752852*cos_phi*cos_theta*(97.+156.*cos_2theta+99.*cos_4theta)*sin_phi*POW2(sin_theta))/r;
            Factor_Ydy = 1.;
            // tdy_r = Factor_Y*Factor_Ydy*(0.1029938790752852*cos_phi*cos_theta*(97.+156.*cos_2theta+99.*cos_4theta)*sin_phi*POW2(sin_theta))/r;
            tdy_i = Factor_Y*Factor_Ydy*(-0.0257484697688213*cos_theta*(POW2(cos_phi)*(38.+24.*cos_2theta+66.*cos_4theta)+(5.*cos_2theta+24.*cos_4theta+99.*cos_6theta)*POW2(sin_phi)))/r;
            Factor_Ydz = 1.;
            // tdz_r = Factor_Y*Factor_Ydz*(0.0257484697688213*cos_phi*(5.*cos_2theta+24.*cos_4theta+99.*cos_6theta)*sin_theta)/r;
            tdz_i = Factor_Y*Factor_Ydz*(0.0257484697688213*(5.*cos_2theta+24.*cos_4theta+99.*cos_6theta)*sin_phi*sin_theta)/r;
            // d Y,6,1 dx
            // d_dYlm_dr[c_atom*3*2+0]+= (tdx_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 2]) ;
            // d_dYlm_dr[c_atom*3*2+1]+= (tdx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 3]) ;
            // d Y,6,1 dy
            // d_dYlm_dr[c_atom*3*2+2]+= (tdy_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 2]) ;
            // d_dYlm_dr[c_atom*3*2+3]+= (tdy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 3]) ;
            // d Y,6,1 dz
            // d_dYlm_dr[c_atom*3*2+4]+= (tdz_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 2]) ;
            // d_dYlm_dr[c_atom*3*2+5]+= (tdz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 3]) ;
            // d Y,6,-1 dx
            // d_dYlm_dr[c_atom*3*2+0]+= -(tdx_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 2]) ;
            // d_dYlm_dr[c_atom*3*2+1]+= (tdx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 3]) ;
            // d Y,6,-1 dy
            // d_dYlm_dr[c_atom*3*2+2]+= -(tdy_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 2]) ;
            // d_dYlm_dr[c_atom*3*2+3]+= (tdy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 3]) ;
            // d Y,6,-1 dz
            // d_dYlm_dr[c_atom*3*2+4]+= -(tdz_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 2]) ;
            // d_dYlm_dr[c_atom*3*2+5]+= (tdz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 3]) ;
            // d Y,6,1 dx + d Y,6,-1 dx
            d_dYlm_dr[c_atom*3*2+0]+= 0 ;
            d_dYlm_dr[c_atom*3*2+1]+= 2*(tdx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 3]) ;
            // d Y,6,1 dx + d Y,6,-1 dy
            d_dYlm_dr[c_atom*3*2+2]+= 0 ;
            d_dYlm_dr[c_atom*3*2+3]+= 2*(tdy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 3]) ;
            // d Y,6,1 dx + d Y,6,-1 dz
            d_dYlm_dr[c_atom*3*2+4]+= 0 ;
            d_dYlm_dr[c_atom*3*2+5]+= 2*(tdz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 3]) ;

            // Y,6,+-2
            Factor_Y = (1.);
            Factor_Ydx = 1.;
            tdx_r = Factor_Y*Factor_Ydx*(0.08142381073346447*(cos_phi*cos_2phi*POW2(cos_theta)*(41.-12.*cos_2theta+99.*cos_4theta)+(35.+60.*cos_2theta+33.*cos_4theta)*sin_phi*sin_2phi)*sin_theta)/r;
            // tdx_i = Factor_Y*Factor_Ydx*(-0.08142381073346447*(cos_2phi*(35.+60.*cos_2theta+33.*cos_4theta)*sin_phi+cos_phi*POW2(cos_theta)*(-41.+12.*cos_2theta-99.*cos_4theta)*sin_2phi)*sin_theta)/r;
            Factor_Ydy = 1.;
            tdy_r = Factor_Y*Factor_Ydy*(-0.08142381073346447*(cos_2phi*POW2(cos_theta)*(-41.+12.*cos_2theta-99.*cos_4theta)*sin_phi+cos_phi*(35.+60.*cos_2theta+33.*cos_4theta)*sin_2phi)*sin_theta)/r;
            // tdy_i = Factor_Y*Factor_Ydy*(0.08142381073346447*(cos_phi*cos_2phi*(35.+60.*cos_2theta+33.*cos_4theta)+POW2(cos_theta)*(41.-12.*cos_2theta+99.*cos_4theta)*sin_phi*sin_2phi)*sin_theta)/r;
            Factor_Ydz = 1.;
            tdz_r = Factor_Y*Factor_Ydz*(-0.08142381073346447*cos_2phi*cos_theta*(41.-12.*cos_2theta+99.*cos_4theta)*POW2(sin_theta))/r;
            // tdz_i = Factor_Y*Factor_Ydz*(-0.08142381073346447*cos_theta*(41.-12.*cos_2theta+99.*cos_4theta)*sin_2phi*POW2(sin_theta))/r;
            // d Y,6,2 dx
            // d_dYlm_dr[c_atom*3*2+0]+= (tdx_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 4]) ;
            // d_dYlm_dr[c_atom*3*2+1]+= (tdx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 5]) ;
            // d Y,6,2 dy
            // d_dYlm_dr[c_atom*3*2+2]+= (tdy_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 4]) ;
            // d_dYlm_dr[c_atom*3*2+3]+= (tdy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 5]) ;
            // d Y,6,2 dz
            // d_dYlm_dr[c_atom*3*2+4]+= (tdz_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 4]) ;
            // d_dYlm_dr[c_atom*3*2+5]+= (tdz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 5]) ;
            // d Y,6,-2 dx
            // d_dYlm_dr[c_atom*3*2+0]+= (tdx_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 4]) ;
            // d_dYlm_dr[c_atom*3*2+1]+= -(tdx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 5]) ;
            // d Y,6,-2 dy
            // d_dYlm_dr[c_atom*3*2+2]+= (tdy_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 4]) ;
            // d_dYlm_dr[c_atom*3*2+3]+= -(tdy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 5]) ;
            // d Y,6,-2 dz
            // d_dYlm_dr[c_atom*3*2+4]+= (tdz_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 4]) ;
            // d_dYlm_dr[c_atom*3*2+5]+= -(tdz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 5]) ;
            // d Y,6,2 dx + d Y,6,-2 dx
            d_dYlm_dr[c_atom*3*2+0]+= 2*(tdx_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 4]) ;
            d_dYlm_dr[c_atom*3*2+1]+= 0 ;
            // d Y,6,2 dx + d Y,6,-2 dy
            d_dYlm_dr[c_atom*3*2+2]+= 2*(tdy_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 4]) ;
            d_dYlm_dr[c_atom*3*2+3]+= 0 ;
            // d Y,6,2 dx + d Y,6,-2 dz
            d_dYlm_dr[c_atom*3*2+4]+= 2*(tdz_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 4]) ;
            d_dYlm_dr[c_atom*3*2+5]+= 0 ;

            // Y,6,+-3
            Factor_Y = (1.);
            Factor_Ydx = 1.;
            // tdx_r = Factor_Y*Factor_Ydx*(0.1221357161001967*POW2(sin_theta)*(-2.*cos_2phi*cos_theta*(17.+36.*cos_2theta+11.*cos_4theta)+4.*cos_4phi*(25.*cos_theta+11.*cos_3theta)*POW2(sin_theta)))/r;
            tdx_i = Factor_Y*Factor_Ydx*(0.4885428644007868*cos_theta*(2.*cos_3phi*(5.+11.*cos_2theta)*sin_phi-1.*cos_phi*(7.+14.*cos_2theta+11.*cos_4theta)*sin_3phi)*POW2(sin_theta))/r;
            Factor_Ydy = 1.;
            // tdy_r = Factor_Y*Factor_Ydy*(-0.4885428644007868*cos_theta*(cos_3phi*(7.+14.*cos_2theta+11.*cos_4theta)*sin_phi-2.*cos_phi*(5.+11.*cos_2theta)*sin_3phi)*POW2(sin_theta))/r;
            tdy_i = Factor_Y*Factor_Ydy*(-0.1221357161001967*POW2(sin_theta)*(cos_2phi*(70.*cos_theta+47.*cos_3theta+11.*cos_5theta)+4.*cos_4phi*(25.*cos_theta+11.*cos_3theta)*POW2(sin_theta)))/r;
            Factor_Ydz = 1.;
            // tdz_r = Factor_Y*Factor_Ydz*(0.4885428644007868*cos_3phi*(7.+14.*cos_2theta+11.*cos_4theta)*POW3(sin_theta))/r;
            tdz_i = Factor_Y*Factor_Ydz*(0.4885428644007868*(7.+14.*cos_2theta+11.*cos_4theta)*sin_3phi*POW3(sin_theta))/r;
            // d Y,6,3 dx
            // d_dYlm_dr[c_atom*3*2+0]+= (tdx_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 6]) ;
            // d_dYlm_dr[c_atom*3*2+1]+= (tdx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 7]) ;
            // d Y,6,3 dy
            // d_dYlm_dr[c_atom*3*2+2]+= (tdy_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 6]) ;
            // d_dYlm_dr[c_atom*3*2+3]+= (tdy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 7]) ;
            // d Y,6,3 dz
            // d_dYlm_dr[c_atom*3*2+4]+= (tdz_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 6]) ;
            // d_dYlm_dr[c_atom*3*2+5]+= (tdz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 7]) ;
            // d Y,6,-3 dx
            // d_dYlm_dr[c_atom*3*2+0]+= -(tdx_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 6]) ;
            // d_dYlm_dr[c_atom*3*2+1]+= (tdx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 7]) ;
            // d Y,6,-3 dy
            // d_dYlm_dr[c_atom*3*2+2]+= -(tdy_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 6]) ;
            // d_dYlm_dr[c_atom*3*2+3]+= (tdy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 7]) ;
            // d Y,6,-3 dz
            // d_dYlm_dr[c_atom*3*2+4]+= -(tdz_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 6]) ;
            // d_dYlm_dr[c_atom*3*2+5]+= (tdz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 7]) ;
            // d Y,6,3 dx + d Y,6,-3 dx
            d_dYlm_dr[c_atom*3*2+0]+= 0 ;
            d_dYlm_dr[c_atom*3*2+1]+= 2*(tdx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 7]) ;
            // d Y,6,3 dx + d Y,6,-3 dy
            d_dYlm_dr[c_atom*3*2+2]+= 0 ;
            d_dYlm_dr[c_atom*3*2+3]+= 2*(tdy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 7]) ;
            // d Y,6,3 dx + d Y,6,-3 dz
            d_dYlm_dr[c_atom*3*2+4]+= 0 ;
            d_dYlm_dr[c_atom*3*2+5]+= 2*(tdz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 7]) ;
            
            // Y,6,+-4
            Factor_Y = (1.);
            Factor_Ydx = 1.;
            tdx_r = Factor_Y*Factor_Ydx*(0.356781262853998*(cos_phi*cos_4phi*POW2(cos_theta)*(7.+33.*cos_2theta)+2.*(9.+11.*cos_2theta)*sin_phi*sin_4phi)*POW3(sin_theta))/r;
            // tdx_i = Factor_Y*Factor_Ydx*(0.356781262853998*(-2.*cos_4phi*(9.+11.*cos_2theta)*sin_phi+cos_phi*POW2(cos_theta)*(7.+33.*cos_2theta)*sin_4phi)*POW3(sin_theta))/r;
            Factor_Ydy = 1.;
            tdy_r = Factor_Y*Factor_Ydy*(0.356781262853998*(cos_4phi*POW2(cos_theta)*(7.+33.*cos_2theta)*sin_phi-2.*cos_phi*(9.+11.*cos_2theta)*sin_4phi)*POW3(sin_theta))/r;
            // tdy_i = Factor_Y*Factor_Ydy*(0.356781262853998*(2.*cos_phi*cos_4phi*(9.+11.*cos_2theta)+POW2(cos_theta)*(7.+33.*cos_2theta)*sin_phi*sin_4phi)*POW3(sin_theta))/r;
            Factor_Ydz = 1.;
            tdz_r = Factor_Y*Factor_Ydz*(-0.178390631426999*cos_4phi*(47.*cos_theta+33.*cos_3theta)*POW4(sin_theta))/r;
            // tdz_i = Factor_Y*Factor_Ydz*(-0.356781262853998*cos_theta*(7.+33.*cos_2theta)*sin_4phi*POW4(sin_theta))/r;
            // d Y,6,4 dx
            // d_dYlm_dr[c_atom*3*2+0]+= (tdx_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 8]) ;
            // d_dYlm_dr[c_atom*3*2+1]+= (tdx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 9]) ;
            // d Y,6,4 dy
            // d_dYlm_dr[c_atom*3*2+2]+= (tdy_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 8]) ;
            // d_dYlm_dr[c_atom*3*2+3]+= (tdy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 9]) ;
            // d Y,6,4 dz
            // d_dYlm_dr[c_atom*3*2+4]+= (tdz_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 8]) ;
            // d_dYlm_dr[c_atom*3*2+5]+= (tdz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 9]) ;
            // d Y,6,-4 dx
            // d_dYlm_dr[c_atom*3*2+0]+= (tdx_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 8]) ;
            // d_dYlm_dr[c_atom*3*2+1]+= -(tdx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 9]) ;
            // d Y,6,-4 dy
            // d_dYlm_dr[c_atom*3*2+2]+= (tdy_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 8]) ;
            // d_dYlm_dr[c_atom*3*2+3]+= -(tdy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 9]) ;
            // d Y,6,-4 dz
            // d_dYlm_dr[c_atom*3*2+4]+= (tdz_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 8]) ;
            // d_dYlm_dr[c_atom*3*2+5]+= -(tdz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 9]) ;
            // d Y,6,4 dx + d Y,6,-4 dx
            d_dYlm_dr[c_atom*3*2+0]+= 2*(tdx_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 8]) ;
            d_dYlm_dr[c_atom*3*2+1]+= 0 ;
            // d Y,6,4 dx + d Y,6,-4 dy
            d_dYlm_dr[c_atom*3*2+2]+= 2*(tdy_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 8]) ;
            d_dYlm_dr[c_atom*3*2+3]+= 0 ;
            // d Y,6,4 dx + d Y,6,-4 dz
            d_dYlm_dr[c_atom*3*2+4]+= 2*(tdz_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 8]) ;
            d_dYlm_dr[c_atom*3*2+5]+= 0 ;

            // Y,6,+-5
            Factor_Y = (1.);
            Factor_Ydx = 1.;
            // tdx_r = Factor_Y*Factor_Ydx*(0.836726229050049*cos_theta*POW4(sin_theta)*(-1.*cos_4phi*(7.+3.*cos_2theta)+6.*cos_6phi*POW2(sin_theta)))/r;
            tdx_i = Factor_Y*Factor_Ydx*(1.673452458100098*cos_theta*(5.*cos_5phi*sin_phi-1.*cos_phi*(2.+3.*cos_2theta)*sin_5phi)*POW4(sin_theta))/r;
            Factor_Ydy = 1.;
            // tdy_r = Factor_Y*Factor_Ydy*(-1.673452458100098*cos_theta*(cos_5phi*(2.+3.*cos_2theta)*sin_phi-5.*cos_phi*sin_5phi)*POW4(sin_theta))/r;
            tdy_i = Factor_Y*Factor_Ydy*(-0.836726229050049*cos_theta*POW4(sin_theta)*(cos_4phi*(7.+3.*cos_2theta)+6.*cos_6phi*POW2(sin_theta)))/r;
            Factor_Ydz = 1.;
            // tdz_r = Factor_Y*Factor_Ydz*(1.673452458100098*cos_5phi*(2.+3.*cos_2theta)*POW5(sin_theta))/r;
            tdz_i = Factor_Y*Factor_Ydz*(1.673452458100098*(2.+3.*cos_2theta)*sin_5phi*POW5(sin_theta))/r;
            // d Y,6,5 dx
            // d_dYlm_dr[c_atom*3*2+0]+= (tdx_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 10] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 10]) ;
            // d_dYlm_dr[c_atom*3*2+1]+= (tdx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 11] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 11]) ;
            // d Y,6,5 dy
            // d_dYlm_dr[c_atom*3*2+2]+= (tdy_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 10] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 10]) ;
            // d_dYlm_dr[c_atom*3*2+3]+= (tdy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 11] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 11]) ;
            // d Y,6,5 dz
            // d_dYlm_dr[c_atom*3*2+4]+= (tdz_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 10] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 10]) ;
            // d_dYlm_dr[c_atom*3*2+5]+= (tdz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 11] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 11]) ;
            // d Y,6,-5 dx
            // d_dYlm_dr[c_atom*3*2+0]+= -(tdx_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 10] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 10]) ;
            // d_dYlm_dr[c_atom*3*2+1]+= (tdx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 11] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 11]) ;
            // d Y,6,-5 dy
            // d_dYlm_dr[c_atom*3*2+2]+= -(tdy_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 10] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 10]) ;
            // d_dYlm_dr[c_atom*3*2+3]+= (tdy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 11] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 11]) ;
            // d Y,6,-5 dz
            // d_dYlm_dr[c_atom*3*2+4]+= -(tdz_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 10] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 10]) ;
            // d_dYlm_dr[c_atom*3*2+5]+= (tdz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 11] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 11]) ;
            // d Y,6,5 dx + d Y,6,-5 dx
            d_dYlm_dr[c_atom*3*2+0]+= 0 ;
            d_dYlm_dr[c_atom*3*2+1]+= 2*(tdx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 11] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 11]) ;
            // d Y,6,5 dx + d Y,6,-5 dy
            d_dYlm_dr[c_atom*3*2+2]+= 0 ;
            d_dYlm_dr[c_atom*3*2+3]+= 2*(tdy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 11] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 11]) ;
            // d Y,6,5 dx + d Y,6,-5 dz
            d_dYlm_dr[c_atom*3*2+4]+= 0 ;
            d_dYlm_dr[c_atom*3*2+5]+= 2*(tdz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 11] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 11]) ;
            
            // Y,6,+-6
            Factor_Y = (0.03125);
            Factor_Ydx = 32.;
            tdx_r = Factor_Y*Factor_Ydx*(2.898504681480397*(cos_phi*cos_6phi*POW2(cos_theta)+sin_phi*sin_6phi)*POW5(sin_theta))/r;
            // tdx_i = Factor_Y*Factor_Ydx*(2.898504681480397*(-1.*cos_6phi*sin_phi+cos_phi*POW2(cos_theta)*sin_6phi)*POW5(sin_theta))/r;
            Factor_Ydy = 32.;
            tdy_r = Factor_Y*Factor_Ydy*(2.898504681480397*(cos_6phi*POW2(cos_theta)*sin_phi-1.*cos_phi*sin_6phi)*POW5(sin_theta))/r;
            // tdy_i = Factor_Y*Factor_Ydy*(2.898504681480397*(cos_phi*cos_6phi+POW2(cos_theta)*sin_phi*sin_6phi)*POW5(sin_theta))/r;
            Factor_Ydz = 92.75214980737272;
            tdz_r = Factor_Y*Factor_Ydz*(-1.*cos_6phi*cos_theta*POW6(sin_theta))/r;
            // tdz_i = Factor_Y*Factor_Ydz*(-1.*cos_theta*sin_6phi*POW6(sin_theta))/r;
            // d Y,6,6 dx
            // d_dYlm_dr[c_atom*3*2+0]+= (tdx_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 12] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 12]) ;
            // d_dYlm_dr[c_atom*3*2+1]+= (tdx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 13] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 13]) ;
            // d Y,6,6 dy
            // d_dYlm_dr[c_atom*3*2+2]+= (tdy_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 12] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 12]) ;
            // d_dYlm_dr[c_atom*3*2+3]+= (tdy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 13] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 13]) ;
            // d Y,6,6 dz
            // d_dYlm_dr[c_atom*3*2+4]+= (tdz_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 12] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 12]) ;
            // d_dYlm_dr[c_atom*3*2+5]+= (tdz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 13] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 13]) ;
            // d Y,6,-6 dx
            // d_dYlm_dr[c_atom*3*2+0]+= (tdx_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 12] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 12]) ;
            // d_dYlm_dr[c_atom*3*2+1]+= -(tdx_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 13] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 13]) ;
            // d Y,6,-6 dy
            // d_dYlm_dr[c_atom*3*2+2]+= (tdy_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 12] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 12]) ;
            // d_dYlm_dr[c_atom*3*2+3]+= -(tdy_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 13] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 13]) ;
            // d Y,6,-6 dz
            // d_dYlm_dr[c_atom*3*2+4]+= (tdz_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 12] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 12]) ;
            // d_dYlm_dr[c_atom*3*2+5]+= -(tdz_i)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 13] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 13]) ;
            // d Y,6,6 dx + d Y,6,-6 dx
            d_dYlm_dr[c_atom*3*2+0]+= 2*(tdx_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 12] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 12]) ;
            d_dYlm_dr[c_atom*3*2+1]+= 0 ;
            // d Y,6,6 dx + d Y,6,-6 dy
            d_dYlm_dr[c_atom*3*2+2]+= 2*(tdy_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 12] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 12]) ;
            d_dYlm_dr[c_atom*3*2+3]+= 0 ;
            // d Y,6,6 dx + d Y,6,-6 dz
            d_dYlm_dr[c_atom*3*2+4]+= 2*(tdz_r)*(catom_q4_timesN*2*d_stein_qlm[stein_qlm_base_id + 12] + neigh_q6_timesN*2*d_stein_Ylm[stein_qlm_neigh_id + 12]) ;
            d_dYlm_dr[c_atom*3*2+5]+= 0 ;
        }
        for (int i=0;i<3;i++){
            d_dcvdx[c_atom*3+i] = d_dYlm_dr[c_atom*3*2 + i*2 + 0] + d_dYlm_dr[c_atom*3*2 + i*2 + 1];
            d_dcvdx[c_atom*3+i] = -(d_dcvdx[c_atom*3+i]*2*PI)/(group_count*(2*stein_l+1));
        }
        // DEBUG_LOG("catom=%d, d_dcvdx[%d] dx, dy, dz = %g, %g, %g", c_atom, c_atom*3, d_dcvdx[c_atom*3+0], d_dcvdx[c_atom*3+1], d_dcvdx[c_atom*3+2]);
    }
}



// dx,dy,dz method
// __global__ void steinhardt_param_calc_kernel_q6(
//     int group_count, int cutoff_Natoms,
//     int *d_neigh_both_in_r_N, double *d_group_dminneigh,
//     double *d_stein_qlm, double *d_stein_Ylm,
//     double *d_stein_ql
// ){
//     int c_atom = blockIdx.x * blockDim.x + threadIdx.x;
//     int neigh_N = 0;
//     if(c_atom<group_count){
//         // steinhardt_param_calc_kernel_q6
//         int stein_l=6;
//         // init some we need
//         int stein_Ylm_base_id, stein_qlm_base_id, neigh_num;
//         double dx, dy, dz, r2, r;
//         double re_part, im_part;
//         double temp, temp4pi_2lplus1;
//         // d_stein_ql[c_atom] = 322;
//         // 4*pi/(2*1+1)
//         // 4*pi = 12.5663706143591729538505735331
//         temp4pi_2lplus1 = 12.5663706143591729538505735331/(2*stein_l+1);

//         neigh_num = d_neigh_both_in_r_N[c_atom];
//         // DEBUG_LOG("neigh_num of c_atom = %d",neigh_num);
//         // q and qlm init
//         stein_qlm_base_id = c_atom*(stein_l + 1)*2;
//         d_stein_ql[c_atom] = 0;
//         for(int i=0; i<(stein_l+1); i++){
//             // from 0 to l, both re_part and im_part
//             d_stein_qlm[stein_qlm_base_id + i*2 + 0] = 0;
//             d_stein_qlm[stein_qlm_base_id + i*2 + 1] = 0;
//             // DEBUG_LOG("d_stein_qlm[%d] = %f + i* %f", stein_qlm_base_id + i + 1,d_stein_qlm[stein_qlm_base_id + i + 0],d_stein_qlm[stein_qlm_base_id + i + 1]);
//         }
//         // start to calc
//         for(int neigh_atom=0; neigh_atom<neigh_num; neigh_atom++){
//             dx = d_group_dminneigh[c_atom*cutoff_Natoms*4 + neigh_atom*4 + 0];
//             dy = d_group_dminneigh[c_atom*cutoff_Natoms*4 + neigh_atom*4 + 1];
//             dz = d_group_dminneigh[c_atom*cutoff_Natoms*4 + neigh_atom*4 + 2];
//             r2     = d_group_dminneigh[c_atom*cutoff_Natoms*4 + neigh_atom*4 + 3];
//             r      = sqrt(r2);
//             stein_Ylm_base_id = c_atom*cutoff_Natoms*(stein_l + 1)*2 + neigh_atom*(stein_l + 1)*2;
//             // DEBUG_LOG("delt x,y,z ; r2, r = %f, %f, %f, %f, %f", dx, dy, dz, r2, r);
//             // Y,6,0
//             temp = 2.6699064952403938891787353045434*(-(POW2(dz)*(10*POW2(dx)*(POW2(dy)-2*POW2(dz))-20*POW2(dy)*POW2(dz)+5*POW4(dx)+5*POW4(dy)+8*POW4(dz))*dx))/(POW4(r2));
//             d_stein_Ylm[stein_Ylm_base_id + 0] = temp;
//             d_stein_Ylm[stein_Ylm_base_id + 1] = 0;
//             // Y,6,1
//             // Y,6,-1, -Re+Im
//             temp = 0.41197551630114077740374798507278*(dz)/(POW4(r2));
//             d_stein_Ylm[stein_Ylm_base_id + 2] = temp * (-5*(17*POW2(dx)-3*POW2(dy))*(POW2(dx)+POW2(dy))*POW2(dz)+4*(25*POW2(dx)+3*POW2(dy))*POW4(dz)-8*POW5(dz)+5*POW2(POW2(dx)+POW2(dy))*(dx-dy)*(dx+dy));
//             d_stein_Ylm[stein_Ylm_base_id + 3] = temp * ((5*POW2(POW2(dx)+POW2(dy))-50*(POW2(dx)+POW2(dy))*POW2(dz)+44*POW4(dz))*dx*dy);
//             // Y,6,2
//             // Y,6,-2, Re-Im
//             temp = 0.65139048586771573756679520789964*(1)/(POW4(r2));
//             d_stein_Ylm[stein_Ylm_base_id + 4] = temp * (dx*(2*POW2(POW2(dx)+POW2(dy))*POW2(dy)+(19*POW2(dx)-49*POW2(dy))*(POW2(dx)+POW2(dy))*POW2(dz)+16*POW5(dz)-64*POW4(dz)*(dx-dy)*(dx+dy)));
//             d_stein_Ylm[stein_Ylm_base_id + 5] = temp * (dy*((53*POW2(dx)-15*POW2(dy))*(POW2(dx)+POW2(dy))*POW2(dz)-128*POW2(dx)*POW4(dz)+16*POW5(dz)-POW2(POW2(dx)+POW2(dy))*(dx-dy)*(dx+dy)));
//             // Y,6,3
//             // Y,6,-3, -Re+Im
//             temp = 1.9541714576031472127003856236989*(dz)/(POW4(r2));
//             d_stein_Ylm[stein_Ylm_base_id + 6] = temp * ((11*POW2(dy)+13*POW2(dz))*POW4(dx)+5*POW2(dz)*POW4(dy)+POW2(dx)*(-54*POW2(dy)*POW2(dz)+9*POW4(dy)-8*POW4(dz))+8*POW2(dy)*POW4(dz)-POW5(dx)-3*POW5(dy));
//             d_stein_Ylm[stein_Ylm_base_id + 7] = temp * ((-14*POW2(dy)*POW2(dz)+2*POW2(dx)*(POW2(dy)+11*POW2(dz))-3*POW4(dx)+5*POW4(dy)-8*POW4(dz))*dx*dy);
//             // Y,6,4
//             // Y,6,-4, Re-Im
//             temp = 0.71356252570799605523573437949432*(1)/(POW4(r2));
//             d_stein_Ylm[stein_Ylm_base_id + 8] = temp * ((POW2(dz)*(150*POW2(dx)*POW2(dy)-13*POW4(dx)-85*POW4(dy))+8*POW2(dy)*(-POW4(dx)+POW4(dy))+20*(POW2(dx)-3*POW2(dy))*POW4(dz))*dx);
//             d_stein_Ylm[stein_Ylm_base_id + 9] = temp * ((POW2(dy)*(POW2(dy)-10*POW2(dz))*(POW2(dy)+POW2(dz))-5*(POW2(dy)+7*POW2(dz))*POW4(dx)-5*POW2(dx)*(-16*POW2(dy)*POW2(dz)+POW4(dy)-6*POW4(dz))+POW5(dx))*dy);
//             // Y,6,5
//             // Y,6,-5, Re-Im
//             temp = 1.673452458100097901242992085976*(dz)/(POW4(r2));
//             d_stein_Ylm[stein_Ylm_base_id + 10] = temp * (-35*POW2(dy)*POW4(dx)+55*POW2(dx)*POW4(dy)-5*POW2(dz)*(-6*POW2(dx)*POW2(dy)+POW4(dx)+POW4(dy))+POW5(dx)-5*POW5(dy));
//             d_stein_Ylm[stein_Ylm_base_id + 11] = temp * ((10*POW2(dy)*POW2(dz)-10*POW2(dx)*(3*POW2(dy)+POW2(dz))+5*POW4(dx)+13*POW4(dy))*dx*dy);
//             // Y,6,6
//             // Y,6,-6, Re-Im
//             temp = 2.8985046814803973618377667394124*(1)/(POW4(r2));
//             d_stein_Ylm[stein_Ylm_base_id + 12] = temp * ((-10*POW2(dx)*POW2(dy)*(2*POW2(dy)+POW2(dz))+(6*POW2(dy)+POW2(dz))*POW4(dx)+5*POW2(dz)*POW4(dy)+6*POW5(dy))*dx);
//             d_stein_Ylm[stein_Ylm_base_id + 13] = temp * ((-5*(3*POW2(dy)+POW2(dz))*POW4(dx)-(POW2(dy)+POW2(dz))*POW4(dy)+5*POW2(dx)*(2*POW2(dy)*POW2(dz)+3*POW4(dy))+POW5(dx))*dy);
//             // DEBUG_LOG("d_stein_Ylm 0, 1, 2, 3, 4 = %f, %f, %f, %f, %f",d_stein_Ylm[stein_Ylm_base_id + 0], d_stein_Ylm[stein_Ylm_base_id + 3], d_stein_Ylm[stein_Ylm_base_id + 4], d_stein_Ylm[stein_Ylm_base_id + 7], d_stein_Ylm[stein_Ylm_base_id + 8]);
            
//             // q,6,0
//             d_stein_qlm[stein_qlm_base_id + 0] += d_stein_Ylm[stein_Ylm_base_id + 0];
//             d_stein_qlm[stein_qlm_base_id + 1] += d_stein_Ylm[stein_Ylm_base_id + 1];
//             // q,6,1
//             d_stein_qlm[stein_qlm_base_id + 2] += d_stein_Ylm[stein_Ylm_base_id + 2];
//             d_stein_qlm[stein_qlm_base_id + 3] += d_stein_Ylm[stein_Ylm_base_id + 3];
//             // q,6,2
//             d_stein_qlm[stein_qlm_base_id + 4] += d_stein_Ylm[stein_Ylm_base_id + 4];
//             d_stein_qlm[stein_qlm_base_id + 5] += d_stein_Ylm[stein_Ylm_base_id + 5];
//             // q,6,3
//             d_stein_qlm[stein_qlm_base_id + 6] += d_stein_Ylm[stein_Ylm_base_id + 6];
//             d_stein_qlm[stein_qlm_base_id + 7] += d_stein_Ylm[stein_Ylm_base_id + 7];
//             // q,6,4
//             d_stein_qlm[stein_qlm_base_id + 8] += d_stein_Ylm[stein_Ylm_base_id + 8];
//             d_stein_qlm[stein_qlm_base_id + 9] += d_stein_Ylm[stein_Ylm_base_id + 9];
//             // q,6,5
//             d_stein_qlm[stein_qlm_base_id + 10] += d_stein_Ylm[stein_Ylm_base_id + 10];
//             d_stein_qlm[stein_qlm_base_id + 11] += d_stein_Ylm[stein_Ylm_base_id + 11];
//             // q,6,6
//             d_stein_qlm[stein_qlm_base_id + 12] += d_stein_Ylm[stein_Ylm_base_id + 12];
//             d_stein_qlm[stein_qlm_base_id + 13] += d_stein_Ylm[stein_Ylm_base_id + 13];
//         }
//         // only calculate once devide for each Ylm
//         for (int i=0;i<(stein_l + 1)*2;i++){
//             // qlm = sum(Ylm)/N
//             d_stein_qlm[stein_qlm_base_id + i] = d_stein_qlm[stein_qlm_base_id + i]/neigh_num;}
//         // q init with q,l,0
//         d_stein_ql[c_atom] = d_stein_qlm[stein_qlm_base_id + 0] * d_stein_qlm[stein_qlm_base_id + 0];
//         // q add with q,l,m
//         for (int i=1;i<(stein_l + 1);i++){
//             re_part = d_stein_qlm[stein_qlm_base_id + i*2 + 0];
//             im_part = d_stein_qlm[stein_qlm_base_id + i*2 + 1];
//             d_stein_ql[c_atom] += 2*(re_part*re_part + im_part*im_part);
//             // DEBUG_LOG("d_stein_qlm[%d] = %f + i* %f", stein_qlm_base_id + i + 1,d_stein_qlm[stein_qlm_base_id + i*2 + 0],d_stein_qlm[stein_qlm_base_id + i*2 + 1]);
//         }
//         // DEBUG_LOG("q4 of c_atom[%d] = %f",atom->tag[c_atom],d_stein_ql[c_atom]);
//         d_stein_ql[c_atom] = d_stein_ql[c_atom]*temp4pi_2lplus1;
//         d_stein_ql[c_atom] = sqrt(d_stein_ql[c_atom]);
//     }
// }


// theta, phi
__global__ void steinhardt_param_calc_kernel_q6(
    int group_count, int cutoff_Natoms,
    int *d_neigh_both_in_r_N, double *d_group_dminneigh,
    double *d_stein_qlm, double *d_stein_Ylm,
    double *d_stein_ql
){
    int c_atom = blockIdx.x * blockDim.x + threadIdx.x;
    int neigh_N = 0;
    if(c_atom<group_count){
        // steinhardt_param_calc_kernel_q6
        int stein_l=6;
        // init some we need
        int stein_Ylm_base_id, stein_qlm_base_id, neigh_num;
        double delt_x, delt_y, delt_z, r2, r;
        double theta, phi;
        double sin_theta, cos_theta, sin_2theta, cos_2theta;
        double sin_3theta, cos_3theta, sin_4theta, cos_4theta;
        double sin_5theta, cos_5theta, sin_6theta, cos_6theta;
        double sin_phi, cos_phi, sin_2phi, cos_2phi;
        double sin_3phi, cos_3phi, sin_4phi, cos_4phi;
        double sin_5phi, cos_5phi, sin_6phi, cos_6phi;
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
            sincos(5*theta, &sin_5theta, &cos_5theta);
            sincos(6*theta, &sin_6theta, &cos_6theta);
            sincos(phi, &sin_phi, &cos_phi);
            sincos(2*phi, &sin_2phi, &cos_2phi);
            sincos(3*phi, &sin_3phi, &cos_3phi);
            sincos(4*phi, &sin_4phi, &cos_4phi);
            sincos(5*phi, &sin_5phi, &cos_5phi);
            sincos(6*phi, &sin_6phi, &cos_6phi);
            stein_Ylm_base_id = c_atom*cutoff_Natoms*(stein_l + 1)*2 + neigh_atom*(stein_l + 1)*2;
            // DEBUG_LOG("delt x,y,z ; r2, r = %f, %f, %f, %f, %f", delt_x, delt_y, delt_z, r2, r);
            // Y,6,0
            temp_value = 0.001986537570863388*(50. + 105.*cos_2theta + 126.*cos_4theta + 231.*cos_6theta);
            d_stein_Ylm[stein_Ylm_base_id + 0] = temp_value;
            d_stein_Ylm[stein_Ylm_base_id + 1] = 0.0;

            // Y,6,1
            // Y,6,-1, -Re+Im
            temp_value = -0.01287423488441065*(5.*sin_2theta + 12.*sin_4theta + 33.*sin_6theta);
            d_stein_Ylm[stein_Ylm_base_id + 2] = temp_value * cos_phi;
            d_stein_Ylm[stein_Ylm_base_id + 3] = temp_value * sin_phi;

            // Y,6,2
            // Y,6,-2, -Re+Im
            temp_value = 0.04071190536673223*(35. + 60.*cos_2theta + 33.*cos_4theta)*POW2(sin_theta);
            d_stein_Ylm[stein_Ylm_base_id + 4] = temp_value * cos_2phi;
            d_stein_Ylm[stein_Ylm_base_id + 5] = temp_value * sin_2phi;

            // Y,6,3
            // Y,6,-3, -Re+Im
            temp_value = -0.3256952429338579*cos_theta*(5. + 11.*cos_2theta)*POW3(sin_theta);
            d_stein_Ylm[stein_Ylm_base_id + 6] = temp_value * cos_3phi;
            d_stein_Ylm[stein_Ylm_base_id + 7] = temp_value * sin_3phi;

            // Y,6,4
            // Y,6,-4, -Re+Im
            temp_value = 0.178390631426999*(9. + 11.*cos_2theta)*POW4(sin_theta);
            d_stein_Ylm[stein_Ylm_base_id + 8] = temp_value * cos_4phi;
            d_stein_Ylm[stein_Ylm_base_id + 9] = temp_value * sin_4phi;

            // Y,6,5
            // Y,6,-5, -Re+Im
            temp_value = -1.673452458100098*cos_theta*POW5(sin_theta);
            d_stein_Ylm[stein_Ylm_base_id + 10] = temp_value * cos_5phi;
            d_stein_Ylm[stein_Ylm_base_id + 11] = temp_value * sin_5phi;

            // Y,6,6
            // Y,6,-6, -Re+Im
            temp_value = 0.4830841135800662*POW6(sin_theta);
            d_stein_Ylm[stein_Ylm_base_id + 12] = temp_value * cos_6phi;
            d_stein_Ylm[stein_Ylm_base_id + 13] = temp_value * sin_6phi;


            // DEBUG_LOG("d_stein_Ylm 0, 1, 2, 3, 4 = %f, %f, %f, %f, %f",d_stein_Ylm[stein_Ylm_base_id + 0], d_stein_Ylm[stein_Ylm_base_id + 3], d_stein_Ylm[stein_Ylm_base_id + 4], d_stein_Ylm[stein_Ylm_base_id + 7], d_stein_Ylm[stein_Ylm_base_id + 8]);
            
            // q,6,0
            d_stein_qlm[stein_qlm_base_id + 0] += d_stein_Ylm[stein_Ylm_base_id + 0];
            d_stein_qlm[stein_qlm_base_id + 1] += d_stein_Ylm[stein_Ylm_base_id + 1];
            // q,6,1
            d_stein_qlm[stein_qlm_base_id + 2] += d_stein_Ylm[stein_Ylm_base_id + 2];
            d_stein_qlm[stein_qlm_base_id + 3] += d_stein_Ylm[stein_Ylm_base_id + 3];
            // q,6,2
            d_stein_qlm[stein_qlm_base_id + 4] += d_stein_Ylm[stein_Ylm_base_id + 4];
            d_stein_qlm[stein_qlm_base_id + 5] += d_stein_Ylm[stein_Ylm_base_id + 5];
            // q,6,3
            d_stein_qlm[stein_qlm_base_id + 6] += d_stein_Ylm[stein_Ylm_base_id + 6];
            d_stein_qlm[stein_qlm_base_id + 7] += d_stein_Ylm[stein_Ylm_base_id + 7];
            // q,6,4
            d_stein_qlm[stein_qlm_base_id + 8] += d_stein_Ylm[stein_Ylm_base_id + 8];
            d_stein_qlm[stein_qlm_base_id + 9] += d_stein_Ylm[stein_Ylm_base_id + 9];
            // q,6,5
            d_stein_qlm[stein_qlm_base_id + 10] += d_stein_Ylm[stein_Ylm_base_id + 10];
            d_stein_qlm[stein_qlm_base_id + 11] += d_stein_Ylm[stein_Ylm_base_id + 11];
            // q,6,6
            d_stein_qlm[stein_qlm_base_id + 12] += d_stein_Ylm[stein_Ylm_base_id + 12];
            d_stein_qlm[stein_qlm_base_id + 13] += d_stein_Ylm[stein_Ylm_base_id + 13];
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
