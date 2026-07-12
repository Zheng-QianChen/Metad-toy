// Ylm_equations.h
#pragma once



namespace MetaD_zqc {
    // 用于访问Ylmdx的语义化
    struct Ylm_Entry { 
        double dx_re, dx_im, dy_re, dy_im, dz_re, dz_im; 
        
        __device__ __forceinline__ void set(double dr, double di, double dyr, double dyi, double dzr, double dzi) {
            dx_re = dr; dx_im = di;
            dy_re = dyr; dy_im = dyi;
            dz_re = dzr; dz_im = dzi;
        }
    };

    // 模板布局容器
    template<int L> struct D_Ylm_Layout;

    // L=3 特化布局
    template<> struct alignas(16) D_Ylm_Layout<3> {
        Ylm_Entry Y30, Y31, Y32, Y33;
    };

    // L=4 特化布局
    template<> struct alignas(16) D_Ylm_Layout<4> {
        Ylm_Entry Y40, Y41, Y42, Y43, Y44; // 根据你的物理公式补充
    };

    // L=4 特化布局
    template<> struct alignas(16) D_Ylm_Layout<6> {
        Ylm_Entry Y60, Y61, Y62, Y63, Y64, Y65, Y66; // 根据你的物理公式补充
    };
}



// 专门处理 L=3 的所有 m 项合并公式（纯扁平）
__device__ __forceinline__ void compute_Ylm_gradient_L3(
    double r,
    double cos_theta, double sin_theta, double cos_phi, double sin_phi,
    double cos_2theta, double sin_2theta, double cos_2phi, double sin_2phi,
    double cos_3phi, double sin_3phi,
    double cos_4theta, double sin_4theta,
    double catom_ql_timesN, double neigh_ql_timesN,
    int stein_qlm_base_id, int stein_qlm_neigh_id,
    double *d_stein_qlm, double *d_dYlm_dr_cut) {

    double Factor_Y, Factor_Ydx, Factor_Ydy, Factor_Ydz;
    double tdx_r, tdx_i, tdy_r, tdy_i, tdz_r, tdz_i;
    // d_dcvdx = [dcvdxc, dcvdyc, dcvdzc]*catoms --flatten
    // d_dYlm_dr = [dYlm_dx_re, dYlm_dx_im, dYlm_dy_re, dYlm_dy_im, dYlm_dz_re, dYlm_dz_im,]*catoms --flatten
    // Y,3,0
    Factor_Y = (0.1399411247212933/r);
    Factor_Ydx = -1.*cos_phi*(6.*sin_2theta+5.*sin_4theta);
    tdx_r = Factor_Y*Factor_Ydx*(1.);
    // tdx_i = Factor_Y*Factor_Ydx*(0);
    Factor_Ydy = -1.*sin_phi*(6.*sin_2theta+5.*sin_4theta);
    tdy_r = Factor_Y*Factor_Ydy*(1.);
    // tdy_i = Factor_Y*Factor_Ydy*(0);
    Factor_Ydz = 4.*(3.+5.*cos_2theta)*POW2(sin_theta);
    tdz_r = Factor_Y*Factor_Ydz*(1.);
    // tdz_i = Factor_Y*Factor_Ydz*(0);
    // d Y,3,0 dx
    d_dYlm_dr_cut[0]+= (tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 0] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 0]) ;
    d_dYlm_dr_cut[1]+= 0 ;
    // d Y,3,0 dy
    d_dYlm_dr_cut[2]+= (tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 0] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 0]) ;
    d_dYlm_dr_cut[3]+= 0 ;
    // d Y,3,0 dz
    d_dYlm_dr_cut[4]+= (tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 0] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 0]) ;
    d_dYlm_dr_cut[5]+= 0 ;
    
    // Y,3,+-1
    Factor_Y = (0.08079504602853766/r);
    Factor_Ydx = 1.;
    // tdx_r = Factor_Y*Factor_Ydx*(-2.*(POW2(cos_phi)*POW2(cos_theta)*(-7.+15.*cos_2theta)+(3.+5.*cos_2theta)*POW2(sin_phi)));
    tdx_i = Factor_Y*Factor_Ydx*((13.+15.*cos_2theta)*sin_2phi*POW2(sin_theta));
    Factor_Ydy = 1.;
    // tdy_r = Factor_Y*Factor_Ydy*((13.+15.*cos_2theta)*sin_2phi*POW2(sin_theta));
    tdy_i = Factor_Y*Factor_Ydy*(-2.*(POW2(cos_phi)*(3.+5.*cos_2theta)+POW2(cos_theta)*(-7.+15.*cos_2theta)*POW2(sin_phi)));
    Factor_Ydz = (-7.+15.*cos_2theta)*sin_2theta;
    // tdz_r = Factor_Y*Factor_Ydz*(cos_phi);
    tdz_i = Factor_Y*Factor_Ydz*(sin_phi);
    // d Y,3,1 dx
    // d_dYlm_dr_cut[0]+= (tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 2]) ;
    // d_dYlm_dr_cut[1]+= (tdx_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 3]) ;
    // d Y,3,1 dy
    // d_dYlm_dr_cut[2]+= (tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 2]) ;
    // d_dYlm_dr_cut[3]+= (tdy_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 3]) ;
    // d Y,3,1 dz
    // d_dYlm_dr_cut[4]+= (tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 2]) ;
    // d_dYlm_dr_cut[5]+= (tdz_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 3]) ;
    // d Y,3,-1 dx
    // d_dYlm_dr_cut[0]+= -(tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 2]) ;
    // d_dYlm_dr_cut[1]+= (tdx_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 3]) ;
    // d Y,3,-1 dy
    // d_dYlm_dr_cut[2]+= -(tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 2]) ;
    // d_dYlm_dr_cut[3]+= (tdy_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 3]) ;
    // d Y,3,-1 dz
    // d_dYlm_dr_cut[4]+= -(tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 2]) ;
    // d_dYlm_dr_cut[5]+= (tdz_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 3]) ;
    // d Y,3,1 dx + d Y,3,-1 dx
    d_dYlm_dr_cut[0]+= 0 ;
    d_dYlm_dr_cut[1]+= 2*(tdx_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 3]) ;
    // d Y,3,1 dx + d Y,3,-1 dy
    d_dYlm_dr_cut[2]+= 0 ;
    d_dYlm_dr_cut[3]+= 2*(tdy_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 3]) ;
    // d Y,3,1 dx + d Y,3,-1 dz
    d_dYlm_dr_cut[4]+= 0 ;
    d_dYlm_dr_cut[5]+= 2*(tdz_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 3]) ;

    // Y,3,+-2
    Factor_Y = (0.2554963691083206/r);
    Factor_Ydx = sin_2theta;
    tdx_r = Factor_Y*Factor_Ydx*(cos_phi*(cos_2phi*(1.+3.*cos_2theta)+8.*POW2(sin_phi)));
    // tdx_i = Factor_Y*Factor_Ydx*(-4.*cos_2phi*sin_phi+cos_phi*(1.+3.*cos_2theta)*sin_2phi);
    // TODO: check this Factor_Ydy, mathmetica automatic generate in this line gave a wrong exper c-coding.
    // Factor_Ydy = Csc(phiB)*sin_2theta;
    Factor_Ydy = cos_phi*sin_2theta;
    tdy_r = Factor_Y*Factor_Ydy*(-2.*POW2(sin_phi)*(2.+3.*cos_2phi*POW2(sin_theta)));
    // tdy_i = Factor_Y*Factor_Ydy*(2.*cos_phi*(1.+3.*cos_2theta)*POW3(sin_phi)+sin_4phi);
    Factor_Ydz = 2.*(1.+3.*cos_2theta)*POW2(sin_theta);
    tdz_r = Factor_Y*Factor_Ydz*(-1.*cos_2phi);
    // tdz_i = Factor_Y*Factor_Ydz*(-sin_2phi);
    // tdz_i = Factor_Y*Factor_Ydz*(-2.*cos_phi*sin_phi);
    // d Y,3,2 dx
    // d_dYlm_dr_cut[0]+= (tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 4]) ;
    // d_dYlm_dr_cut[1]+= (tdx_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 5]) ;
    // d Y,3,2 dy
    // d_dYlm_dr_cut[2]+= (tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 4]) ;
    // d_dYlm_dr_cut[3]+= (tdy_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 5]) ;
    // d Y,3,2 dz
    // d_dYlm_dr_cut[4]+= (tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 4]) ;
    // d_dYlm_dr_cut[5]+= (tdz_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 5]) ;
    // d Y,3,-2 dx
    // d_dYlm_dr_cut[0]+= (tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 4]) ;
    // d_dYlm_dr_cut[1]+= -(tdx_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 5]) ;
    // d Y,3,-2 dy
    // d_dYlm_dr_cut[2]+= (tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 4]) ;
    // d_dYlm_dr_cut[3]+= -(tdy_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 5]) ;
    // d Y,3,-2 dz
    // d_dYlm_dr_cut[4]+= (tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 4]) ;
    // d_dYlm_dr_cut[5]+= -(tdz_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 5]) ;
    // d Y,3,2 dx + d Y,3,-2 dx
    d_dYlm_dr_cut[0]+= 2*(tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 4]) ;
    d_dYlm_dr_cut[1]+= 0 ;
    // d Y,3,2 dx + d Y,3,-2 dy
    d_dYlm_dr_cut[2]+= 2*(tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 4]) ;
    d_dYlm_dr_cut[3]+= 0 ;
    // d Y,3,2 dx + d Y,3,-2 dz
    d_dYlm_dr_cut[4]+= 2*(tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 4]) ;
    d_dYlm_dr_cut[5]+= 0 ;

    // Y,4,+-3
    Factor_Y = ((1.251671470898352*POW2(sin_theta))/r);
    Factor_Ydx = 1.;
    // tdx_r = Factor_Y*Factor_Ydx*(-1.*cos_phi*cos_3phi*POW2(cos_theta)-1.*sin_phi*sin_3phi);
    tdx_i = Factor_Y*Factor_Ydx*(cos_3phi*sin_phi-1.*cos_phi*POW2(cos_theta)*sin_3phi);
    Factor_Ydy = 1.;
    // tdy_r = Factor_Y*Factor_Ydy*(-1.*cos_3phi*POW2(cos_theta)*sin_phi+cos_phi*sin_3phi);
    tdy_i = Factor_Y*Factor_Ydy*(-1.*cos_phi*cos_3phi-1.*POW2(cos_theta)*sin_phi*sin_3phi);
    Factor_Ydz = cos_theta*sin_theta;
    // tdz_r = Factor_Y*Factor_Ydz*(cos_3phi);
    tdz_i = Factor_Y*Factor_Ydz*(sin_3phi);
    // d Y,4,3 dx
    // d_dYlm_dr_cut[0]+= (tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 6]) ;
    // d_dYlm_dr_cut[1]+= (tdx_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 7]) ;
    // d Y,4,3 dy
    // d_dYlm_dr_cut[2]+= (tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 6]) ;
    // d_dYlm_dr_cut[3]+= (tdy_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 7]) ;
    // d Y,4,3 dz
    // d_dYlm_dr_cut[4]+= (tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 6]) ;
    // d_dYlm_dr_cut[5]+= (tdz_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 7]) ;
    // d Y,4,-3 dx
    // d_dYlm_dr_cut[0]+= -(tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 6]) ;
    // d_dYlm_dr_cut[1]+= (tdx_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 7]) ;
    // d Y,4,-3 dy
    // d_dYlm_dr_cut[2]+= -(tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 6]) ;
    // d_dYlm_dr_cut[3]+= (tdy_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 7]) ;
    // d Y,4,-3 dz
    // d_dYlm_dr_cut[4]+= -(tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 6]) ;
    // d_dYlm_dr_cut[5]+= (tdz_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 7]) ;
    // d Y,4,3 dx + d Y,4,-3 dx
    d_dYlm_dr_cut[0]+= 0 ;
    d_dYlm_dr_cut[1]+= 2*(tdx_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 7]) ;
    // d Y,4,3 dx + d Y,4,-3 dy
    d_dYlm_dr_cut[2]+= 0 ;
    d_dYlm_dr_cut[3]+= 2*(tdy_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 7]) ;
    // d Y,4,3 dx + d Y,4,-3 dz
    d_dYlm_dr_cut[4]+= 0 ;
    d_dYlm_dr_cut[5]+= 2*(tdz_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] - neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 7]) ;
}




// 专门处理 L=4 的所有 m 项合并公式（纯扁平）
__device__ __forceinline__ void compute_Ylm_gradient_L4(
    double r,
    double cos_theta, double sin_theta, double cos_phi, double sin_phi,
    double cos_2theta, double sin_2theta, double cos_2phi, double sin_2phi,
    double cos_3theta, double sin_3theta, double cos_3phi, double sin_3phi,
    double cos_4theta, double sin_4theta, double cos_4phi, double sin_4phi,
    double cos_5theta, double sin_5theta, double cos_5phi, double sin_5phi,
    double cos_6theta, double sin_6theta, 
    double catom_ql_timesN, double neigh_ql_timesN,
    int stein_qlm_base_id, int stein_qlm_neigh_id,
    double *d_stein_qlm, double *d_dYlm_dr_cut) {
    double Factor_Y, Factor_Ydx, Factor_Ydy, Factor_Ydz;
    double tdx_r, tdx_i, tdy_r, tdy_i, tdz_r, tdz_i;

    // d_dcvdx = [dcvdxc, dcvdyc, dcvdzc]*catoms --flatten
    // d_dYlm_dr = [dYlm_dx_re, dYlm_dx_im, dYlm_dy_re, dYlm_dy_im, dYlm_dz_re, dYlm_dz_im,]*catoms --flatten
    // Y,4,0
    Factor_Y = ((1.057855469152043*cos_theta*(1.+7.*cos_2theta)*sin_theta)/r);
    Factor_Ydx = -1.*cos_phi*cos_theta;
    tdx_r = Factor_Y*Factor_Ydx*(1.);
    tdx_i = Factor_Y*Factor_Ydx*(0);
    Factor_Ydy = -1.*cos_theta*sin_phi;
    tdy_r = Factor_Y*Factor_Ydy*(1.);
    tdy_i = Factor_Y*Factor_Ydy*(0);
    Factor_Ydz = sin_theta;
    tdz_r = Factor_Y*Factor_Ydz*(1.);
    tdz_i = Factor_Y*Factor_Ydz*(0);
    // d Y,4,0 dx
    d_dYlm_dr_cut[0]+= (tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 0] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 0]) ;
    d_dYlm_dr_cut[1]+= 0 ;
    // d Y,4,0 dy
    d_dYlm_dr_cut[2]+= (tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 0] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 0]) ;
    d_dYlm_dr_cut[3]+= 0 ;
    // d Y,4,0 dz
    d_dYlm_dr_cut[4]+= (tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 0] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 0]) ;
    d_dYlm_dr_cut[5]+= 0 ;
    
    // Y,4,+-1
    Factor_Y = (0.23654367393939/r);
    Factor_Ydx = 2.*cos_theta;
    // tdx_r = Factor_Y*Factor_Ydx*(0.5*((-4.+3.*cos_2phi)*cos_2theta-7.*POW2(cos_phi)*cos_4theta-1.*POW2(sin_phi)));
    tdx_i = Factor_Y*Factor_Ydx*((4.+7.*cos_2theta)*sin_2phi*POW2(sin_theta));
    Factor_Ydy = 2.*cos_theta;
    // tdy_r = Factor_Y*Factor_Ydy*((4.+7.*cos_2theta)*sin_2phi*POW2(sin_theta));
    tdy_i = Factor_Y*Factor_Ydy*(0.5*(-1.*POW2(cos_phi)-1.*(4.+3.*cos_2phi)*cos_2theta-7.*cos_4theta*POW2(sin_phi)));
    Factor_Ydz = (cos_2theta+7.*cos_4theta)*sin_theta;
    // tdz_r = Factor_Y*Factor_Ydz*(cos_phi);
    tdz_i = Factor_Y*Factor_Ydz*(sin_phi);
    // d Y,4,1 dx
    // d_dYlm_dr_cut[0]+= (tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 2]) ;
    // d_dYlm_dr_cut[1]+= (tdx_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 3]) ;
    // d Y,4,1 dy
    // d_dYlm_dr_cut[2]+= (tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 2]) ;
    // d_dYlm_dr_cut[3]+= (tdy_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 3]) ;
    // d Y,4,1 dz
    // d_dYlm_dr_cut[4]+= (tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 2]) ;
    // d_dYlm_dr_cut[5]+= (tdz_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 3]) ;
    // d Y,4,-1 dx
    // d_dYlm_dr_cut[0]+= -(tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 2]) ;
    // d_dYlm_dr_cut[1]+= (tdx_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 3]) ;
    // d Y,4,-1 dy
    // d_dYlm_dr_cut[2]+= -(tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 2]) ;
    // d_dYlm_dr_cut[3]+= (tdy_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 3]) ;
    // d Y,4,-1 dz
    // d_dYlm_dr_cut[4]+= -(tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 2]) ;
    // d_dYlm_dr_cut[5]+= (tdz_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 3]) ;
    // d Y,4,1 dx + d Y,4,-1 dx
    d_dYlm_dr_cut[0]+= 0 ;
    d_dYlm_dr_cut[1]+= 2*(tdx_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 3]) ;
    // d Y,4,1 dx + d Y,4,-1 dy
    d_dYlm_dr_cut[2]+= 0 ;
    d_dYlm_dr_cut[3]+= 2*(tdy_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 3]) ;
    // d Y,4,1 dx + d Y,4,-1 dz
    d_dYlm_dr_cut[4]+= 0 ;
    d_dYlm_dr_cut[5]+= 2*(tdz_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 3]) ;

    // Y,4,+-2
    Factor_Y = (0.3345232717786446/r);
    Factor_Ydx = 1.;
    tdx_r = Factor_Y*Factor_Ydx*((5.+7.*cos_2theta)*sin_phi*sin_2phi*sin_theta+cos_phi*cos_2phi*POW2(cos_theta)*(-9.*sin_theta+7.*sin_3theta));
    // tdx_i = Factor_Y*Factor_Ydx*(0.125*(-1.*sin_3phi*(8.*sin_theta+9.*sin_3theta)+sin_phi*(4.*sin_theta+19.*sin_3theta+28.*POW2(cos_phi)*sin_5theta)));
    Factor_Ydy = sin_theta;
    tdy_r = Factor_Y*Factor_Ydy*(-0.25*(15.+26.*cos_2theta+7.*cos_4theta)*sin_phi-1.*(6.+7.*cos_2theta)*sin_3phi*POW2(sin_theta));
    // tdy_i = Factor_Y*Factor_Ydy*(cos_phi*(5.*POW2(cos_phi)+(6.+cos_2phi)*cos_2theta+7.*cos_4theta*POW2(sin_phi)));
    Factor_Ydz = 2.*cos_theta*(-1.+7.*cos_2theta)*POW2(sin_theta);
    tdz_r = Factor_Y*Factor_Ydz*(-1.*cos_2phi);
    // tdz_i = Factor_Y*Factor_Ydz*(-2.*cos_phi*sin_phi);
    // d Y,4,2 dx
    // d_dYlm_dr_cut[0]+= (tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 4]) ;
    // d_dYlm_dr_cut[1]+= (tdx_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 5]) ;
    // d Y,4,2 dy
    // d_dYlm_dr_cut[2]+= (tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 4]) ;
    // d_dYlm_dr_cut[3]+= (tdy_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 5]) ;
    // d Y,4,2 dz
    // d_dYlm_dr_cut[4]+= (tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 4]) ;
    // d_dYlm_dr_cut[5]+= (tdz_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 5]) ;
    // d Y,4,-2 dx
    // d_dYlm_dr_cut[0]+= (tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 4]) ;
    // d_dYlm_dr_cut[1]+= -(tdx_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 5]) ;
    // d Y,4,-2 dy
    // d_dYlm_dr_cut[2]+= (tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 4]) ;
    // d_dYlm_dr_cut[3]+= -(tdy_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 5]) ;
    // d Y,4,-2 dz
    // d_dYlm_dr_cut[4]+= (tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 4]) ;
    // d_dYlm_dr_cut[5]+= -(tdz_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 5]) ;
    // d Y,4,2 dx + d Y,4,-2 dx
    d_dYlm_dr_cut[0]+= 2*(tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 4]) ;
    d_dYlm_dr_cut[1]+= 0 ;
    // d Y,4,2 dx + d Y,4,-2 dy
    d_dYlm_dr_cut[2]+= 2*(tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 4]) ;
    d_dYlm_dr_cut[3]+= 0 ;
    // d Y,4,2 dx + d Y,4,-2 dz
    d_dYlm_dr_cut[4]+= 2*(tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 4]) ;
    d_dYlm_dr_cut[5]+= 0 ;

    // Y,4,+-3
    Factor_Y = ((1.251671470898352*POW2(sin_theta))/r);
    Factor_Ydx = cos_theta;
    // tdx_r = Factor_Y*Factor_Ydx*(-2.*cos_2phi+cos_4phi+2.*POW2(cos_phi)*(1.-2.*cos_2phi)*cos_2theta);
    tdx_i = Factor_Y*Factor_Ydx*(3.*cos_3phi*sin_phi-1.*cos_phi*(1.+2.*cos_2theta)*sin_3phi);
    Factor_Ydy = cos_theta;
    // tdy_r = Factor_Y*Factor_Ydy*(-1.*cos_3phi*(1.+2.*cos_2theta)*sin_phi+3.*cos_phi*sin_3phi);
    tdy_i = Factor_Y*Factor_Ydy*(-2.*cos_2phi-1.*cos_4phi-2.*(1.+2.*cos_2phi)*cos_2theta*POW2(sin_phi));
    Factor_Ydz = sin_3theta;
    // tdz_r = Factor_Y*Factor_Ydz*(cos_3phi);
    tdz_i = Factor_Y*Factor_Ydz*(sin_3phi);
    // d Y,4,3 dx
    // d_dYlm_dr_cut[0]+= (tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 6]) ;
    // d_dYlm_dr_cut[1]+= (tdx_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 7]) ;
    // d Y,4,3 dy
    // d_dYlm_dr_cut[2]+= (tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 6]) ;
    // d_dYlm_dr_cut[3]+= (tdy_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 7]) ;
    // d Y,4,3 dz
    // d_dYlm_dr_cut[4]+= (tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 6]) ;
    // d_dYlm_dr_cut[5]+= (tdz_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 7]) ;
    // d Y,4,-3 dx
    // d_dYlm_dr_cut[0]+= -(tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 6]) ;
    // d_dYlm_dr_cut[1]+= (tdx_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 7]) ;
    // d Y,4,-3 dy
    // d_dYlm_dr_cut[2]+= -(tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 6]) ;
    // d_dYlm_dr_cut[3]+= (tdy_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 7]) ;
    // d Y,4,-3 dz
    // d_dYlm_dr_cut[4]+= -(tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 6]) ;
    // d_dYlm_dr_cut[5]+= (tdz_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 7]) ;
    // d Y,4,3 dx + d Y,4,-3 dx
    d_dYlm_dr_cut[0]+= 0 ;
    d_dYlm_dr_cut[1]+= 2*(tdx_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 7]) ;
    // d Y,4,3 dx + d Y,4,-3 dy
    d_dYlm_dr_cut[2]+= 0 ;
    d_dYlm_dr_cut[3]+= 2*(tdy_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 7]) ;
    // d Y,4,3 dx + d Y,4,-3 dz
    d_dYlm_dr_cut[4]+= 0 ;
    d_dYlm_dr_cut[5]+= 2*(tdz_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 7]) ;
    
    // Y,4,+-4
    Factor_Y = ((1.770130769779931*POW3(sin_theta))/r);
    Factor_Ydx = 1.;
    tdx_r = Factor_Y*Factor_Ydx*(cos_phi*cos_4phi*POW2(cos_theta)+sin_phi*sin_4phi);
    // tdx_i = Factor_Y*Factor_Ydx*(-1.*cos_4phi*sin_phi+cos_phi*POW2(cos_theta)*sin_4phi);
    Factor_Ydy = 1.;
    tdy_r = Factor_Y*Factor_Ydy*(cos_4phi*POW2(cos_theta)*sin_phi-1.*cos_phi*sin_4phi);
    // tdy_i = Factor_Y*Factor_Ydy*(cos_phi*cos_4phi+POW2(cos_theta)*sin_phi*sin_4phi);
    Factor_Ydz = cos_theta*sin_theta;
    tdz_r = Factor_Y*Factor_Ydz*(-1.*cos_4phi);
    // tdz_i = Factor_Y*Factor_Ydz*(-1.*sin_4phi);
    // d Y,4,4 dx
    // d_dYlm_dr_cut[0]+= (tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 8]) ;
    // d_dYlm_dr_cut[1]+= (tdx_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 9]) ;
    // d Y,4,4 dy
    // d_dYlm_dr_cut[2]+= (tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 8]) ;
    // d_dYlm_dr_cut[3]+= (tdy_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 9]) ;
    // d Y,4,4 dz
    // d_dYlm_dr_cut[4]+= (tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 8]) ;
    // d_dYlm_dr_cut[5]+= (tdz_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 9]) ;
    // d Y,4,-4 dx
    // d_dYlm_dr_cut[0]+= (tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 8]) ;
    // d_dYlm_dr_cut[1]+= -(tdx_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 9]) ;
    // d Y,4,-4 dy
    // d_dYlm_dr_cut[2]+= (tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 8]) ;
    // d_dYlm_dr_cut[3]+= -(tdy_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 9]) ;
    // d Y,4,-4 dz
    // d_dYlm_dr_cut[4]+= (tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 8]) ;
    // d_dYlm_dr_cut[5]+= -(tdz_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 9]) ;
    // d Y,4,4 dx + d Y,4,-4 dx
    d_dYlm_dr_cut[0]+= 2*(tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 8]) ;
    d_dYlm_dr_cut[1]+= 0 ;
    // d Y,4,4 dx + d Y,4,-4 dy
    d_dYlm_dr_cut[2]+= 2*(tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 8]) ;
    d_dYlm_dr_cut[3]+= 0 ;
    // d Y,4,4 dx + d Y,4,-4 dz
    d_dYlm_dr_cut[4]+= 2*(tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 8]) ;
    d_dYlm_dr_cut[5]+= 0 ;
}

// 专门处理 L=6 的所有 m 项合并公式（纯扁平）
__device__ __forceinline__ void compute_Ylm_gradient_L6(
    double r,
    double cos_theta, double sin_theta, double cos_phi, double sin_phi,
    double cos_2theta, double sin_2theta, double cos_2phi, double sin_2phi,
    double cos_3theta, double sin_3theta, double cos_3phi, double sin_3phi,
    double cos_4theta, double sin_4theta, double cos_4phi, double sin_4phi,
    double cos_5theta, double sin_5theta, double cos_5phi, double sin_5phi,
    double cos_6theta, double sin_6theta, double cos_6phi, double sin_6phi,
    double catom_ql_timesN, double neigh_ql_timesN,
    int stein_qlm_base_id, int stein_qlm_neigh_id,
    double *d_stein_qlm, double *d_dYlm_dr_cut) {

    double Factor_Y, Factor_Ydx, Factor_Ydy, Factor_Ydz;
    double tdx_r, tdx_i, tdy_r, tdy_i, tdz_r, tdz_i;
    
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
    d_dYlm_dr_cut[0]+= (tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 0] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 0]) ;
    d_dYlm_dr_cut[1]+= 0 ;
    // d Y,6,0 dy
    d_dYlm_dr_cut[2]+= (tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 0] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 0]) ;
    d_dYlm_dr_cut[3]+= 0 ;
    // d Y,6,0 dz
    d_dYlm_dr_cut[4]+= (tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 0] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 0]) ;
    d_dYlm_dr_cut[5]+= 0 ;
    
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
    // d_dYlm_dr_cut[0]+= (tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 2]) ;
    // d_dYlm_dr_cut[1]+= (tdx_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 3]) ;
    // d Y,6,1 dy
    // d_dYlm_dr_cut[2]+= (tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 2]) ;
    // d_dYlm_dr_cut[3]+= (tdy_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 3]) ;
    // d Y,6,1 dz
    // d_dYlm_dr_cut[4]+= (tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 2]) ;
    // d_dYlm_dr_cut[5]+= (tdz_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 3]) ;
    // d Y,6,-1 dx
    // d_dYlm_dr_cut[0]+= -(tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 2]) ;
    // d_dYlm_dr_cut[1]+= (tdx_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 3]) ;
    // d Y,6,-1 dy
    // d_dYlm_dr_cut[2]+= -(tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 2]) ;
    // d_dYlm_dr_cut[3]+= (tdy_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 3]) ;
    // d Y,6,-1 dz
    // d_dYlm_dr_cut[4]+= -(tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 2] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 2]) ;
    // d_dYlm_dr_cut[5]+= (tdz_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 3]) ;
    // d Y,6,1 dx + d Y,6,-1 dx
    d_dYlm_dr_cut[0]+= 0 ;
    d_dYlm_dr_cut[1]+= 2*(tdx_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 3]) ;
    // d Y,6,1 dx + d Y,6,-1 dy
    d_dYlm_dr_cut[2]+= 0 ;
    d_dYlm_dr_cut[3]+= 2*(tdy_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 3]) ;
    // d Y,6,1 dx + d Y,6,-1 dz
    d_dYlm_dr_cut[4]+= 0 ;
    d_dYlm_dr_cut[5]+= 2*(tdz_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 3] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 3]) ;

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
    // d_dYlm_dr_cut[0]+= (tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 4]) ;
    // d_dYlm_dr_cut[1]+= (tdx_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 5]) ;
    // d Y,6,2 dy
    // d_dYlm_dr_cut[2]+= (tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 4]) ;
    // d_dYlm_dr_cut[3]+= (tdy_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 5]) ;
    // d Y,6,2 dz
    // d_dYlm_dr_cut[4]+= (tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 4]) ;
    // d_dYlm_dr_cut[5]+= (tdz_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 5]) ;
    // d Y,6,-2 dx
    // d_dYlm_dr_cut[0]+= (tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 4]) ;
    // d_dYlm_dr_cut[1]+= -(tdx_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 5]) ;
    // d Y,6,-2 dy
    // d_dYlm_dr_cut[2]+= (tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 4]) ;
    // d_dYlm_dr_cut[3]+= -(tdy_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 5]) ;
    // d Y,6,-2 dz
    // d_dYlm_dr_cut[4]+= (tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 4]) ;
    // d_dYlm_dr_cut[5]+= -(tdz_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 5] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 5]) ;
    // d Y,6,2 dx + d Y,6,-2 dx
    d_dYlm_dr_cut[0]+= 2*(tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 4]) ;
    d_dYlm_dr_cut[1]+= 0 ;
    // d Y,6,2 dx + d Y,6,-2 dy
    d_dYlm_dr_cut[2]+= 2*(tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 4]) ;
    d_dYlm_dr_cut[3]+= 0 ;
    // d Y,6,2 dx + d Y,6,-2 dz
    d_dYlm_dr_cut[4]+= 2*(tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 4] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 4]) ;
    d_dYlm_dr_cut[5]+= 0 ;

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
    // d_dYlm_dr_cut[0]+= (tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 6]) ;
    // d_dYlm_dr_cut[1]+= (tdx_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 7]) ;
    // d Y,6,3 dy
    // d_dYlm_dr_cut[2]+= (tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 6]) ;
    // d_dYlm_dr_cut[3]+= (tdy_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 7]) ;
    // d Y,6,3 dz
    // d_dYlm_dr_cut[4]+= (tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 6]) ;
    // d_dYlm_dr_cut[5]+= (tdz_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 7]) ;
    // d Y,6,-3 dx
    // d_dYlm_dr_cut[0]+= -(tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 6]) ;
    // d_dYlm_dr_cut[1]+= (tdx_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 7]) ;
    // d Y,6,-3 dy
    // d_dYlm_dr_cut[2]+= -(tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 6]) ;
    // d_dYlm_dr_cut[3]+= (tdy_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 7]) ;
    // d Y,6,-3 dz
    // d_dYlm_dr_cut[4]+= -(tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 6] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 6]) ;
    // d_dYlm_dr_cut[5]+= (tdz_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 7]) ;
    // d Y,6,3 dx + d Y,6,-3 dx
    d_dYlm_dr_cut[0]+= 0 ;
    d_dYlm_dr_cut[1]+= 2*(tdx_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 7]) ;
    // d Y,6,3 dx + d Y,6,-3 dy
    d_dYlm_dr_cut[2]+= 0 ;
    d_dYlm_dr_cut[3]+= 2*(tdy_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 7]) ;
    // d Y,6,3 dx + d Y,6,-3 dz
    d_dYlm_dr_cut[4]+= 0 ;
    d_dYlm_dr_cut[5]+= 2*(tdz_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 7] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 7]) ;
    
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
    // d_dYlm_dr_cut[0]+= (tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 8]) ;
    // d_dYlm_dr_cut[1]+= (tdx_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 9]) ;
    // d Y,6,4 dy
    // d_dYlm_dr_cut[2]+= (tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 8]) ;
    // d_dYlm_dr_cut[3]+= (tdy_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 9]) ;
    // d Y,6,4 dz
    // d_dYlm_dr_cut[4]+= (tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 8]) ;
    // d_dYlm_dr_cut[5]+= (tdz_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 9]) ;
    // d Y,6,-4 dx
    // d_dYlm_dr_cut[0]+= (tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 8]) ;
    // d_dYlm_dr_cut[1]+= -(tdx_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 9]) ;
    // d Y,6,-4 dy
    // d_dYlm_dr_cut[2]+= (tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 8]) ;
    // d_dYlm_dr_cut[3]+= -(tdy_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 9]) ;
    // d Y,6,-4 dz
    // d_dYlm_dr_cut[4]+= (tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 8]) ;
    // d_dYlm_dr_cut[5]+= -(tdz_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 9] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 9]) ;
    // d Y,6,4 dx + d Y,6,-4 dx
    d_dYlm_dr_cut[0]+= 2*(tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 8]) ;
    d_dYlm_dr_cut[1]+= 0 ;
    // d Y,6,4 dx + d Y,6,-4 dy
    d_dYlm_dr_cut[2]+= 2*(tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 8]) ;
    d_dYlm_dr_cut[3]+= 0 ;
    // d Y,6,4 dx + d Y,6,-4 dz
    d_dYlm_dr_cut[4]+= 2*(tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 8] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 8]) ;
    d_dYlm_dr_cut[5]+= 0 ;

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
    // d_dYlm_dr_cut[0]+= (tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 10] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 10]) ;
    // d_dYlm_dr_cut[1]+= (tdx_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 11] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 11]) ;
    // d Y,6,5 dy
    // d_dYlm_dr_cut[2]+= (tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 10] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 10]) ;
    // d_dYlm_dr_cut[3]+= (tdy_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 11] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 11]) ;
    // d Y,6,5 dz
    // d_dYlm_dr_cut[4]+= (tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 10] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 10]) ;
    // d_dYlm_dr_cut[5]+= (tdz_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 11] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 11]) ;
    // d Y,6,-5 dx
    // d_dYlm_dr_cut[0]+= -(tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 10] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 10]) ;
    // d_dYlm_dr_cut[1]+= (tdx_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 11] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 11]) ;
    // d Y,6,-5 dy
    // d_dYlm_dr_cut[2]+= -(tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 10] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 10]) ;
    // d_dYlm_dr_cut[3]+= (tdy_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 11] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 11]) ;
    // d Y,6,-5 dz
    // d_dYlm_dr_cut[4]+= -(tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 10] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 10]) ;
    // d_dYlm_dr_cut[5]+= (tdz_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 11] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 11]) ;
    // d Y,6,5 dx + d Y,6,-5 dx
    d_dYlm_dr_cut[0]+= 0 ;
    d_dYlm_dr_cut[1]+= 2*(tdx_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 11] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 11]) ;
    // d Y,6,5 dx + d Y,6,-5 dy
    d_dYlm_dr_cut[2]+= 0 ;
    d_dYlm_dr_cut[3]+= 2*(tdy_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 11] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 11]) ;
    // d Y,6,5 dx + d Y,6,-5 dz
    d_dYlm_dr_cut[4]+= 0 ;
    d_dYlm_dr_cut[5]+= 2*(tdz_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 11] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 11]) ;
    
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
    // d_dYlm_dr_cut[0]+= (tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 12] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 12]) ;
    // d_dYlm_dr_cut[1]+= (tdx_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 13] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 13]) ;
    // d Y,6,6 dy
    // d_dYlm_dr_cut[2]+= (tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 12] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 12]) ;
    // d_dYlm_dr_cut[3]+= (tdy_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 13] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 13]) ;
    // d Y,6,6 dz
    // d_dYlm_dr_cut[4]+= (tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 12] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 12]) ;
    // d_dYlm_dr_cut[5]+= (tdz_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 13] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 13]) ;
    // d Y,6,-6 dx
    // d_dYlm_dr_cut[0]+= (tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 12] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 12]) ;
    // d_dYlm_dr_cut[1]+= -(tdx_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 13] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 13]) ;
    // d Y,6,-6 dy
    // d_dYlm_dr_cut[2]+= (tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 12] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 12]) ;
    // d_dYlm_dr_cut[3]+= -(tdy_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 13] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 13]) ;
    // d Y,6,-6 dz
    // d_dYlm_dr_cut[4]+= (tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 12] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 12]) ;
    // d_dYlm_dr_cut[5]+= -(tdz_i)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 13] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 13]) ;
    // d Y,6,6 dx + d Y,6,-6 dx
    d_dYlm_dr_cut[0]+= 2*(tdx_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 12] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 12]) ;
    d_dYlm_dr_cut[1]+= 0 ;
    // d Y,6,6 dx + d Y,6,-6 dy
    d_dYlm_dr_cut[2]+= 2*(tdy_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 12] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 12]) ;
    d_dYlm_dr_cut[3]+= 0 ;
    // d Y,6,6 dx + d Y,6,-6 dz
    d_dYlm_dr_cut[4]+= 2*(tdz_r)*(catom_ql_timesN*2*d_stein_qlm[stein_qlm_base_id + 12] + neigh_ql_timesN*2*d_stein_qlm[stein_qlm_neigh_id + 12]) ;
    d_dYlm_dr_cut[5]+= 0 ;
}


// 专门处理 L=3 的所有 m 项合并公式（纯扁平）
__device__ __forceinline__ void compute_dYlmdx_gradient_L3(
    double r,
    double cos_theta, double sin_theta, double cos_phi, double sin_phi,
    double cos_2theta, double sin_2theta, double cos_2phi, double sin_2phi,
    double cos_3phi, double sin_3phi, double sin_4phi,
    double cos_4theta, double sin_4theta,
    MetaD_zqc::D_Ylm_Layout<3> *d_dYlm_dr_cut) {

    double Factor_Y, Factor_Ydx, Factor_Ydy, Factor_Ydz;
    double tdx_r, tdx_i, tdy_r, tdy_i, tdz_r, tdz_i;

    // auto* layout = reinterpret_cast<MetaD_zqc::D_Ylm_Layout<3>*>(d_dYlm_dr_cut);

    // d_dcvdx = [dcvdxc, dcvdyc, dcvdzc]*catoms --flatten
    // d_dYlm_dr = [dYlm_dx_re, dYlm_dx_im, dYlm_dy_re, dYlm_dy_im, dYlm_dz_re, dYlm_dz_im,]*catoms --flatten
    // Y,3,0
    Factor_Y = (0.1399411247212933/r);
    Factor_Ydx = -1.*cos_phi*(6.*sin_2theta+5.*sin_4theta);
    tdx_r = Factor_Y*Factor_Ydx*(1.);
    tdx_i = Factor_Y*Factor_Ydx*(0);
    Factor_Ydy = -1.*sin_phi*(6.*sin_2theta+5.*sin_4theta);
    tdy_r = Factor_Y*Factor_Ydy*(1.);
    tdy_i = Factor_Y*Factor_Ydy*(0);
    Factor_Ydz = 4.*(3.+5.*cos_2theta)*POW2(sin_theta);
    tdz_r = Factor_Y*Factor_Ydz*(1.);
    tdz_i = Factor_Y*Factor_Ydz*(0);
    // // d Y,3,0 dx
    // d_dYlm_dr_cut[0]= (tdx_r);
    // d_dYlm_dr_cut[1]= 0;
    // // d Y,3,0 dy
    // d_dYlm_dr_cut[2]= (tdy_r);
    // d_dYlm_dr_cut[3]= (tdy_i);
    // // d Y,3,0 dz
    // d_dYlm_dr_cut[4]= (tdz_r);
    // d_dYlm_dr_cut[5]= (tdz_i);
    d_dYlm_dr_cut->Y30.set(tdx_r, 0.0, tdy_r, 0.0, tdz_r, 0.0);
    
    // Y,3,+-1
    Factor_Y = (0.08079504602853766/r);
    Factor_Ydx = 1.;
    tdx_r = Factor_Y*Factor_Ydx*(-2.*(POW2(cos_phi)*POW2(cos_theta)*(-7.+15.*cos_2theta)+(3.+5.*cos_2theta)*POW2(sin_phi)));
    tdx_i = Factor_Y*Factor_Ydx*((13.+15.*cos_2theta)*sin_2phi*POW2(sin_theta));
    Factor_Ydy = 1.;
    tdy_r = Factor_Y*Factor_Ydy*((13.+15.*cos_2theta)*sin_2phi*POW2(sin_theta));
    tdy_i = Factor_Y*Factor_Ydy*(-2.*(POW2(cos_phi)*(3.+5.*cos_2theta)+POW2(cos_theta)*(-7.+15.*cos_2theta)*POW2(sin_phi)));
    Factor_Ydz = (-7.+15.*cos_2theta)*sin_2theta;
    tdz_r = Factor_Y*Factor_Ydz*(cos_phi);
    tdz_i = Factor_Y*Factor_Ydz*(sin_phi);
    // // d Y,3,1 dx
    // d_dYlm_dr_cut[6]= tdx_r;
    // d_dYlm_dr_cut[7]= tdx_i;
    // // d Y,3,1 dy
    // d_dYlm_dr_cut[8]= (tdy_r) ;
    // d_dYlm_dr_cut[9]= (tdy_i);
    // // d Y,3,1 dz
    // d_dYlm_dr_cut[10]= (tdz_r);
    // d_dYlm_dr_cut[11]= (tdz_i);
    d_dYlm_dr_cut->Y31.set(tdx_r, tdx_i, tdy_r, tdy_i, tdz_r, tdz_i);

    // Y,3,+-2
    Factor_Y = (0.2554963691083206/r);
    Factor_Ydx = sin_2theta;
    tdx_r = Factor_Y*Factor_Ydx*(cos_phi*(cos_2phi*(1.+3.*cos_2theta)+8.*POW2(sin_phi)));
    tdx_i = Factor_Y*Factor_Ydx*(-4.*cos_2phi*sin_phi+cos_phi*(1.+3.*cos_2theta)*sin_2phi);
    // TODO: check this Factor_Ydy, mathmetica automatic generate in this line gave a wrong exper c-coding.
    // Factor_Ydy = Csc(phiB)*sin_2theta;
    // Factor_Ydy = sin_2theta/sin_phi;
    // tdy_r = Factor_Y*Factor_Ydy*(-2.*POW2(sin_phi)*(2.+3.*cos_2phi*POW2(sin_theta)));
    // tdy_i = Factor_Y*Factor_Ydy*(2.*cos_phi*(1.+3.*cos_2theta)*POW3(sin_phi)+sin_4phi);
    // dy 分量: 用稳定版公式替换，不再依赖 Csc(phi)因为有除零风险
    {
        double C_Y32 = 1.021985476433282;   // 与 compute_Ylm_forward_L3 中 Y32 的系数一致
        tdy_r = (C_Y32/r) * sin_theta*cos_theta*sin_phi
                *(6.*POW2(sin_phi)*POW2(sin_theta) - 3.*POW2(sin_theta) - 2.);
        tdy_i = (2.*C_Y32/r) * sin_theta*cos_theta*cos_phi
                *(1. - 3.*POW2(sin_phi)*POW2(sin_theta));
    }
    Factor_Ydz = 2.*(1.+3.*cos_2theta)*POW2(sin_theta);
    tdz_r = Factor_Y*Factor_Ydz*(-1.*cos_2phi);
    tdz_i = Factor_Y*Factor_Ydz*(-1.*sin_2phi);
    // // d Y,3,2 dx
    // d_dYlm_dr_cut[12]= (tdx_r);
    // d_dYlm_dr_cut[13]= tdx_i;
    // // d Y,3,2 dy
    // d_dYlm_dr_cut[14]= (tdy_r);
    // d_dYlm_dr_cut[15]= (tdy_i);
    // // d Y,3,2 dz
    // d_dYlm_dr_cut[16]= (tdz_r);
    // d_dYlm_dr_cut[17]= (tdz_i);
    d_dYlm_dr_cut->Y32.set(tdx_r, tdx_i, tdy_r, tdy_i, tdz_r, tdz_i);

    // Y,3,+-3
    Factor_Y = ((1.251671470898352*POW2(sin_theta))/r);
    Factor_Ydx = 1.;
    tdx_r = Factor_Y*Factor_Ydx*(-1.*cos_phi*cos_3phi*POW2(cos_theta)-1.*sin_phi*sin_3phi);
    tdx_i = Factor_Y*Factor_Ydx*(cos_3phi*sin_phi-1.*cos_phi*POW2(cos_theta)*sin_3phi);
    Factor_Ydy = 1.;
    tdy_r = Factor_Y*Factor_Ydy*(-1.*cos_3phi*POW2(cos_theta)*sin_phi+cos_phi*sin_3phi);
    tdy_i = Factor_Y*Factor_Ydy*(-1.*cos_phi*cos_3phi-1.*POW2(cos_theta)*sin_phi*sin_3phi);
    Factor_Ydz = cos_theta*sin_theta;
    tdz_r = Factor_Y*Factor_Ydz*(cos_3phi);
    tdz_i = Factor_Y*Factor_Ydz*(sin_3phi);
    // // d Y,3,3 dx
    // d_dYlm_dr_cut[18]= (tdx_r);
    // d_dYlm_dr_cut[19]= (tdx_i);
    // // d Y,3,3 dy
    // d_dYlm_dr_cut[20]= (tdy_r);
    // d_dYlm_dr_cut[21]= (tdy_i);
    // // d Y,3,3 dz
    // d_dYlm_dr_cut[22]= (tdz_r);
    // d_dYlm_dr_cut[23]= (tdz_i);
    d_dYlm_dr_cut->Y33.set(tdx_r, tdx_i, tdy_r, tdy_i, tdz_r, tdz_i);
}




// 专门处理 L=4 的所有 m 项合并公式（纯扁平）
__device__ __forceinline__ void compute_dYlmdx_gradient_L4(
    double r,
    double cos_theta, double sin_theta, double cos_phi, double sin_phi,
    double cos_2theta, double sin_2theta, double cos_2phi, double sin_2phi,
    double cos_3theta, double sin_3theta, double cos_3phi, double sin_3phi,
    double cos_4theta, double sin_4theta, double cos_4phi, double sin_4phi,
    double cos_5theta, double sin_5theta, double cos_5phi, double sin_5phi,
    double cos_6theta, double sin_6theta, 
    MetaD_zqc::D_Ylm_Layout<4> *d_dYlm_dr_cut) {
    double Factor_Y, Factor_Ydx, Factor_Ydy, Factor_Ydz;
    double tdx_r, tdx_i, tdy_r, tdy_i, tdz_r, tdz_i;

    // auto* layout = reinterpret_cast<MetaD_zqc::D_Ylm_Layout<4>*>(d_dYlm_dr_cut);

    // d_dcvdx = [dcvdxc, dcvdyc, dcvdzc]*catoms --flatten
    // d_dYlm_dr = [dYlm_dx_re, dYlm_dx_im, dYlm_dy_re, dYlm_dy_im, dYlm_dz_re, dYlm_dz_im,]*catoms --flatten
    // Y,4,0
    Factor_Y = ((1.057855469152043*cos_theta*(1.+7.*cos_2theta)*sin_theta)/r);
    Factor_Ydx = -1.*cos_phi*cos_theta;
    tdx_r = Factor_Y*Factor_Ydx*(1.);
    tdx_i = 0.0;
    Factor_Ydy = -1.*cos_theta*sin_phi;
    tdy_r = Factor_Y*Factor_Ydy*(1.);
    tdy_i = 0.0;
    Factor_Ydz = sin_theta;
    tdz_r = Factor_Y*Factor_Ydz*(1.);
    tdz_i = 0.0;
    d_dYlm_dr_cut->Y40.set(tdx_r, tdx_i, tdy_r, tdy_i, tdz_r, tdz_i);
    
    // Y,4,+-1
    Factor_Y = (0.23654367393939/r);
    Factor_Ydx = 2.*cos_theta;
    tdx_r = Factor_Y*Factor_Ydx*(0.5*((-4.+3.*cos_2phi)*cos_2theta-7.*POW2(cos_phi)*cos_4theta-1.*POW2(sin_phi)));
    tdx_i = Factor_Y*Factor_Ydx*((4.+7.*cos_2theta)*sin_2phi*POW2(sin_theta));
    Factor_Ydy = 2.*cos_theta;
    tdy_r = Factor_Y*Factor_Ydy*((4.+7.*cos_2theta)*sin_2phi*POW2(sin_theta));
    tdy_i = Factor_Y*Factor_Ydy*(0.5*(-1.*POW2(cos_phi)-1.*(4.+3.*cos_2phi)*cos_2theta-7.*cos_4theta*POW2(sin_phi)));
    Factor_Ydz = (cos_2theta+7.*cos_4theta)*sin_theta;
    tdz_r = Factor_Y*Factor_Ydz*(cos_phi);
    tdz_i = Factor_Y*Factor_Ydz*(sin_phi);
    d_dYlm_dr_cut->Y41.set(tdx_r, tdx_i, tdy_r, tdy_i, tdz_r, tdz_i);

    // Y,4,+-2
    Factor_Y = (0.3345232717786446/r);
    Factor_Ydx = 1.;
    tdx_r = Factor_Y*Factor_Ydx*((5.+7.*cos_2theta)*sin_phi*sin_2phi*sin_theta+cos_phi*cos_2phi*POW2(cos_theta)*(-9.*sin_theta+7.*sin_3theta));
    tdx_i = Factor_Y*Factor_Ydx*(0.125*(-1.*sin_3phi*(8.*sin_theta+9.*sin_3theta)+sin_phi*(4.*sin_theta+19.*sin_3theta+28.*POW2(cos_phi)*sin_5theta)));
    Factor_Ydy = sin_theta;
    tdy_r = Factor_Y*Factor_Ydy*(-0.25*(15.+26.*cos_2theta+7.*cos_4theta)*sin_phi-1.*(6.+7.*cos_2theta)*sin_3phi*POW2(sin_theta));
    tdy_i = Factor_Y*Factor_Ydy*(cos_phi*(5.*POW2(cos_phi)+(6.+cos_2phi)*cos_2theta+7.*cos_4theta*POW2(sin_phi)));
    Factor_Ydz = 2.*cos_theta*(-1.+7.*cos_2theta)*POW2(sin_theta);
    tdz_r = Factor_Y*Factor_Ydz*(-1.*cos_2phi);
    tdz_i = Factor_Y*Factor_Ydz*(-2.*cos_phi*sin_phi);
    d_dYlm_dr_cut->Y42.set(tdx_r, tdx_i, tdy_r, tdy_i, tdz_r, tdz_i);

    // Y,4,+-3
    Factor_Y = ((1.251671470898352*POW2(sin_theta))/r);
    Factor_Ydx = cos_theta;
    tdx_r = Factor_Y*Factor_Ydx*(-2.*cos_2phi+cos_4phi+2.*POW2(cos_phi)*(1.-2.*cos_2phi)*cos_2theta);
    tdx_i = Factor_Y*Factor_Ydx*(3.*cos_3phi*sin_phi-1.*cos_phi*(1.+2.*cos_2theta)*sin_3phi);
    Factor_Ydy = cos_theta;
    tdy_r = Factor_Y*Factor_Ydy*(-1.*cos_3phi*(1.+2.*cos_2theta)*sin_phi+3.*cos_phi*sin_3phi);
    tdy_i = Factor_Y*Factor_Ydy*(-2.*cos_2phi-1.*cos_4phi-2.*(1.+2.*cos_2phi)*cos_2theta*POW2(sin_phi));
    Factor_Ydz = sin_3theta;
    tdz_r = Factor_Y*Factor_Ydz*(cos_3phi);
    tdz_i = Factor_Y*Factor_Ydz*(sin_3phi);
    d_dYlm_dr_cut->Y43.set(tdx_r, tdx_i, tdy_r, tdy_i, tdz_r, tdz_i);
    
    // Y,4,+-4
    Factor_Y = ((1.770130769779931*POW3(sin_theta))/r);
    Factor_Ydx = 1.;
    tdx_r = Factor_Y*Factor_Ydx*(cos_phi*cos_4phi*POW2(cos_theta)+sin_phi*sin_4phi);
    tdx_i = Factor_Y*Factor_Ydx*(-1.*cos_4phi*sin_phi+cos_phi*POW2(cos_theta)*sin_4phi);
    Factor_Ydy = 1.;
    tdy_r = Factor_Y*Factor_Ydy*(cos_4phi*POW2(cos_theta)*sin_phi-1.*cos_phi*sin_4phi);
    tdy_i = Factor_Y*Factor_Ydy*(cos_phi*cos_4phi+POW2(cos_theta)*sin_phi*sin_4phi);
    Factor_Ydz = cos_theta*sin_theta;
    tdz_r = Factor_Y*Factor_Ydz*(-1.*cos_4phi);
    tdz_i = Factor_Y*Factor_Ydz*(-1.*sin_4phi);
    d_dYlm_dr_cut->Y44.set(tdx_r, tdx_i, tdy_r, tdy_i, tdz_r, tdz_i);
}

// 专门处理 L=6 的所有 m 项合并公式（纯扁平）
__device__ __forceinline__ void compute_dYlmdx_gradient_L6(
    double r,
    double cos_theta, double sin_theta, double cos_phi, double sin_phi,
    double cos_2theta, double sin_2theta, double cos_2phi, double sin_2phi,
    double cos_3theta, double sin_3theta, double cos_3phi, double sin_3phi,
    double cos_4theta, double sin_4theta, double cos_4phi, double sin_4phi,
    double cos_5theta, double sin_5theta, double cos_5phi, double sin_5phi,
    double cos_6theta, double sin_6theta, double cos_6phi, double sin_6phi,
    MetaD_zqc::D_Ylm_Layout<6> *d_dYlm_dr_cut) {

    double Factor_Y, Factor_Ydx, Factor_Ydy, Factor_Ydz;
    double tdx_r, tdx_i, tdy_r, tdy_i, tdz_r, tdz_i;

    // auto* layout = reinterpret_cast<MetaD_zqc::D_Ylm_Layout<6>*>(d_dYlm_dr_cut);
    
    // d_dcvdx = [dcvdxc, dcvdyc, dcvdzc]*catoms --flatten
    // d_dYlm_dr = [dYlm_dx_re, dYlm_dx_im, dYlm_dy_re, dYlm_dy_im, dYlm_dz_re, dYlm_dz_im,]*catoms --flatten
    // Y,6,0
    Factor_Y = (1.);
    Factor_Ydx = 1.;
    tdx_r = Factor_Y*Factor_Ydx*(-0.3337383119050492*cos_phi*POW2(cos_theta)*(19.+12.*cos_2theta+33.*cos_4theta)*sin_theta)/r;
    // tdx_i = Factor_Y*Factor_Ydx*0;
    Factor_Ydy = 1.;
    tdy_r = Factor_Y*Factor_Ydy*(-0.3337383119050492*POW2(cos_theta)*(19.+12.*cos_2theta+33.*cos_4theta)*sin_phi*sin_theta)/r;
    // tdy_i = Factor_Y*Factor_Ydy*0;
    Factor_Ydz = 1.;
    tdz_r = Factor_Y*Factor_Ydz*(0.3337383119050492*cos_theta*(19.+12.*cos_2theta+33.*cos_4theta)*POW2(sin_theta))/r;
    // tdz_i = Factor_Y*Factor_Ydz*0;
    d_dYlm_dr_cut->Y60.set(tdx_r, 0.0, tdy_r, 0.0, tdz_r, 0.0);
    
    // Y,6,+-1
    Factor_Y = (1.);
    Factor_Ydx = 1.;
    tdx_r = Factor_Y*Factor_Ydx*(-0.0257484697688213*cos_theta*(POW2(cos_phi)*(5.*cos_2theta+24.*cos_4theta+99.*cos_6theta)+2.*(19.+12.*cos_2theta+33.*cos_4theta)*POW2(sin_phi)))/r;
    tdx_i = Factor_Y*Factor_Ydx*(0.1029938790752852*cos_phi*cos_theta*(97.+156.*cos_2theta+99.*cos_4theta)*sin_phi*POW2(sin_theta))/r;
    Factor_Ydy = 1.;
    tdy_r = Factor_Y*Factor_Ydy*(0.1029938790752852*cos_phi*cos_theta*(97.+156.*cos_2theta+99.*cos_4theta)*sin_phi*POW2(sin_theta))/r;
    tdy_i = Factor_Y*Factor_Ydy*(-0.0257484697688213*cos_theta*(POW2(cos_phi)*(38.+24.*cos_2theta+66.*cos_4theta)+(5.*cos_2theta+24.*cos_4theta+99.*cos_6theta)*POW2(sin_phi)))/r;
    Factor_Ydz = 1.;
    tdz_r = Factor_Y*Factor_Ydz*(0.0257484697688213*cos_phi*(5.*cos_2theta+24.*cos_4theta+99.*cos_6theta)*sin_theta)/r;
    tdz_i = Factor_Y*Factor_Ydz*(0.0257484697688213*(5.*cos_2theta+24.*cos_4theta+99.*cos_6theta)*sin_phi*sin_theta)/r;
    d_dYlm_dr_cut->Y61.set(tdx_r, tdx_i, tdy_r, tdy_i, tdz_r, tdz_i);

    // Y,6,+-2
    Factor_Y = (1.);
    Factor_Ydx = 1.;
    tdx_r = Factor_Y*Factor_Ydx*(0.08142381073346447*(cos_phi*cos_2phi*POW2(cos_theta)*(41.-12.*cos_2theta+99.*cos_4theta)+(35.+60.*cos_2theta+33.*cos_4theta)*sin_phi*sin_2phi)*sin_theta)/r;
    tdx_i = Factor_Y*Factor_Ydx*(-0.08142381073346447*(cos_2phi*(35.+60.*cos_2theta+33.*cos_4theta)*sin_phi+cos_phi*POW2(cos_theta)*(-41.+12.*cos_2theta-99.*cos_4theta)*sin_2phi)*sin_theta)/r;
    Factor_Ydy = 1.;
    tdy_r = Factor_Y*Factor_Ydy*(-0.08142381073346447*(cos_2phi*POW2(cos_theta)*(-41.+12.*cos_2theta-99.*cos_4theta)*sin_phi+cos_phi*(35.+60.*cos_2theta+33.*cos_4theta)*sin_2phi)*sin_theta)/r;
    tdy_i = Factor_Y*Factor_Ydy*(0.08142381073346447*(cos_phi*cos_2phi*(35.+60.*cos_2theta+33.*cos_4theta)+POW2(cos_theta)*(41.-12.*cos_2theta+99.*cos_4theta)*sin_phi*sin_2phi)*sin_theta)/r;
    Factor_Ydz = 1.;
    tdz_r = Factor_Y*Factor_Ydz*(-0.08142381073346447*cos_2phi*cos_theta*(41.-12.*cos_2theta+99.*cos_4theta)*POW2(sin_theta))/r;
    tdz_i = Factor_Y*Factor_Ydz*(-0.08142381073346447*cos_theta*(41.-12.*cos_2theta+99.*cos_4theta)*sin_2phi*POW2(sin_theta))/r;
    d_dYlm_dr_cut->Y62.set(tdx_r, tdx_i, tdy_r, tdy_i, tdz_r, tdz_i);

    // Y,6,+-3
    Factor_Y = (1.);
    Factor_Ydx = 1.;
    tdx_r = Factor_Y*Factor_Ydx*(0.1221357161001967*POW2(sin_theta)*(-2.*cos_2phi*cos_theta*(17.+36.*cos_2theta+11.*cos_4theta)+4.*cos_4phi*(25.*cos_theta+11.*cos_3theta)*POW2(sin_theta)))/r;
    tdx_i = Factor_Y*Factor_Ydx*(0.4885428644007868*cos_theta*(2.*cos_3phi*(5.+11.*cos_2theta)*sin_phi-1.*cos_phi*(7.+14.*cos_2theta+11.*cos_4theta)*sin_3phi)*POW2(sin_theta))/r;
    Factor_Ydy = 1.;
    tdy_r = Factor_Y*Factor_Ydy*(-0.4885428644007868*cos_theta*(cos_3phi*(7.+14.*cos_2theta+11.*cos_4theta)*sin_phi-2.*cos_phi*(5.+11.*cos_2theta)*sin_3phi)*POW2(sin_theta))/r;
    tdy_i = Factor_Y*Factor_Ydy*(-0.1221357161001967*POW2(sin_theta)*(cos_2phi*(70.*cos_theta+47.*cos_3theta+11.*cos_5theta)+4.*cos_4phi*(25.*cos_theta+11.*cos_3theta)*POW2(sin_theta)))/r;
    Factor_Ydz = 1.;
    tdz_r = Factor_Y*Factor_Ydz*(0.4885428644007868*cos_3phi*(7.+14.*cos_2theta+11.*cos_4theta)*POW3(sin_theta))/r;
    tdz_i = Factor_Y*Factor_Ydz*(0.4885428644007868*(7.+14.*cos_2theta+11.*cos_4theta)*sin_3phi*POW3(sin_theta))/r;
    d_dYlm_dr_cut->Y63.set(tdx_r, tdx_i, tdy_r, tdy_i, tdz_r, tdz_i);
    
    // Y,6,+-4
    Factor_Y = (1.);
    Factor_Ydx = 1.;
    tdx_r = Factor_Y*Factor_Ydx*(0.356781262853998*(cos_phi*cos_4phi*POW2(cos_theta)*(7.+33.*cos_2theta)+2.*(9.+11.*cos_2theta)*sin_phi*sin_4phi)*POW3(sin_theta))/r;
    tdx_i = Factor_Y*Factor_Ydx*(0.356781262853998*(-2.*cos_4phi*(9.+11.*cos_2theta)*sin_phi+cos_phi*POW2(cos_theta)*(7.+33.*cos_2theta)*sin_4phi)*POW3(sin_theta))/r;
    Factor_Ydy = 1.;
    tdy_r = Factor_Y*Factor_Ydy*(0.356781262853998*(cos_4phi*POW2(cos_theta)*(7.+33.*cos_2theta)*sin_phi-2.*cos_phi*(9.+11.*cos_2theta)*sin_4phi)*POW3(sin_theta))/r;
    tdy_i = Factor_Y*Factor_Ydy*(0.356781262853998*(2.*cos_phi*cos_4phi*(9.+11.*cos_2theta)+POW2(cos_theta)*(7.+33.*cos_2theta)*sin_phi*sin_4phi)*POW3(sin_theta))/r;
    Factor_Ydz = 1.;
    tdz_r = Factor_Y*Factor_Ydz*(-0.178390631426999*cos_4phi*(47.*cos_theta+33.*cos_3theta)*POW4(sin_theta))/r;
    tdz_i = Factor_Y*Factor_Ydz*(-0.356781262853998*cos_theta*(7.+33.*cos_2theta)*sin_4phi*POW4(sin_theta))/r;
    d_dYlm_dr_cut->Y64.set(tdx_r, tdx_i, tdy_r, tdy_i, tdz_r, tdz_i);

    // Y,6,+-5
    Factor_Y = (1.);
    Factor_Ydx = 1.;
    tdx_r = Factor_Y*Factor_Ydx*(0.836726229050049*cos_theta*POW4(sin_theta)*(-1.*cos_4phi*(7.+3.*cos_2theta)+6.*cos_6phi*POW2(sin_theta)))/r;
    tdx_i = Factor_Y*Factor_Ydx*(1.673452458100098*cos_theta*(5.*cos_5phi*sin_phi-1.*cos_phi*(2.+3.*cos_2theta)*sin_5phi)*POW4(sin_theta))/r;
    Factor_Ydy = 1.;
    tdy_r = Factor_Y*Factor_Ydy*(-1.673452458100098*cos_theta*(cos_5phi*(2.+3.*cos_2theta)*sin_phi-5.*cos_phi*sin_5phi)*POW4(sin_theta))/r;
    tdy_i = Factor_Y*Factor_Ydy*(-0.836726229050049*cos_theta*POW4(sin_theta)*(cos_4phi*(7.+3.*cos_2theta)+6.*cos_6phi*POW2(sin_theta)))/r;
    Factor_Ydz = 1.;
    tdz_r = Factor_Y*Factor_Ydz*(1.673452458100098*cos_5phi*(2.+3.*cos_2theta)*POW5(sin_theta))/r;
    tdz_i = Factor_Y*Factor_Ydz*(1.673452458100098*(2.+3.*cos_2theta)*sin_5phi*POW5(sin_theta))/r;
    d_dYlm_dr_cut->Y65.set(tdx_r, tdx_i, tdy_r, tdy_i, tdz_r, tdz_i);
    
    // Y,6,+-6
    Factor_Y = (0.03125);
    Factor_Ydx = 32.;
    tdx_r = Factor_Y*Factor_Ydx*(2.898504681480397*(cos_phi*cos_6phi*POW2(cos_theta)+sin_phi*sin_6phi)*POW5(sin_theta))/r;
    tdx_i = Factor_Y*Factor_Ydx*(2.898504681480397*(-1.*cos_6phi*sin_phi+cos_phi*POW2(cos_theta)*sin_6phi)*POW5(sin_theta))/r;
    Factor_Ydy = 32.;
    tdy_r = Factor_Y*Factor_Ydy*(2.898504681480397*(cos_6phi*POW2(cos_theta)*sin_phi-1.*cos_phi*sin_6phi)*POW5(sin_theta))/r;
    tdy_i = Factor_Y*Factor_Ydy*(2.898504681480397*(cos_phi*cos_6phi+POW2(cos_theta)*sin_phi*sin_6phi)*POW5(sin_theta))/r;
    Factor_Ydz = 92.75214980737272;
    tdz_r = Factor_Y*Factor_Ydz*(-1.*cos_6phi*cos_theta*POW6(sin_theta))/r;
    tdz_i = Factor_Y*Factor_Ydz*(-1.*cos_theta*sin_6phi*POW6(sin_theta))/r;
    d_dYlm_dr_cut->Y66.set(tdx_r, tdx_i, tdy_r, tdy_i, tdz_r, tdz_i);
}


__device__ __forceinline__ void compute_Ylm_forward_L3(
    double cos_theta, double sin_theta, double cos_phi, double sin_phi,
    double cos_2theta, double sin_2theta, double cos_2phi, double sin_2phi,
    double cos_3theta, double sin_3theta, double cos_3phi, double sin_3phi,
    double *d_stein_Ylm_cut) {
    double temp_value;
    // DEBUG_LOG("delt x,y,z ; r2, r = %f, %f, %f, %f, %f", delt_x, delt_y, delt_z, r2, r);
    // Y,3,0
    temp_value = 0.1865881662950577*cos_theta*(-1.+5.*cos_2theta);
    d_stein_Ylm_cut[0] = temp_value;
    d_stein_Ylm_cut[1] = 0.0;

    // Y,3,1
    // Y,3,-1, -Re+Im
    temp_value = -0.08079504602853766*(sin_theta+5.*sin_3theta);
    d_stein_Ylm_cut[2] = temp_value * cos_phi;
    d_stein_Ylm_cut[3] = temp_value * sin_phi;

    // Y,3,2
    // Y,3,-2, Re-Im
    temp_value = 1.021985476433282*cos_theta*POW2(sin_theta);
    d_stein_Ylm_cut[4] = temp_value * cos_2phi;
    d_stein_Ylm_cut[5] = temp_value * sin_2phi;

    // Y,3,3
    // Y,3,-3, -Re+Im
    temp_value = -0.4172238236327841*POW3(sin_theta);
    d_stein_Ylm_cut[6] = temp_value * cos_3phi;
    d_stein_Ylm_cut[7] = temp_value * sin_3phi;   
}


__device__ __forceinline__ void compute_qlm_forward_L3(
    double r_weight,
    double cos_theta, double sin_theta, double cos_phi, double sin_phi,
    double cos_2theta, double sin_2theta, double cos_2phi, double sin_2phi,
    double cos_3theta, double sin_3theta, double cos_3phi, double sin_3phi,
    double *d_stein_qlm_cut, double *d_stein_Ylm_cut) {

    compute_Ylm_forward_L3(
        cos_theta, sin_theta, cos_phi, sin_phi,
        cos_2theta, sin_2theta, cos_2phi, sin_2phi,
        cos_3theta, sin_3theta, cos_3phi, sin_3phi,
        &d_stein_Ylm_cut[0]
    );
    
    
    // q,3,0
    d_stein_qlm_cut[0] += r_weight*d_stein_Ylm_cut[0];
    d_stein_qlm_cut[1] += r_weight*d_stein_Ylm_cut[1];
    // q,3,1
    d_stein_qlm_cut[2] += r_weight*d_stein_Ylm_cut[2];
    d_stein_qlm_cut[3] += r_weight*d_stein_Ylm_cut[3];
    // q,3,2
    d_stein_qlm_cut[4] += r_weight*d_stein_Ylm_cut[4];
    d_stein_qlm_cut[5] += r_weight*d_stein_Ylm_cut[5];
    // q,3,3
    d_stein_qlm_cut[6] += r_weight*d_stein_Ylm_cut[6];
    d_stein_qlm_cut[7] += r_weight*d_stein_Ylm_cut[7];

}


__device__ __forceinline__ void compute_Ylm_forward_L4(
    double cos_theta, double sin_theta, double cos_phi, double sin_phi,
    double cos_2phi, double sin_2phi,
    double cos_3phi, double sin_3phi,
    double cos_4phi, double sin_4phi,
    double *d_stein_Ylm_cut) {
    
    double temp_value;
    // Y,4,0
    // 3/16*sqrt(1/(1*pi)) = 0.10578554691520430
    d_stein_Ylm_cut[0] = 0.10578554691520430*(3-30*POW2(cos_theta)+35*POW4(cos_theta));
    d_stein_Ylm_cut[1] = 0;
    // Y,4,1
    // Y,4,-1, -Re+Im
    // 3/8*sqrt(5/(1*pi)) = 0.47308734787878000
    temp_value = 0.47308734787878000*cos_theta*(-3+7*POW2(cos_theta))*sin_theta;
    d_stein_Ylm_cut[2] = - temp_value * cos_phi;
    d_stein_Ylm_cut[3] = - temp_value * sin_phi;
    // Y,4,2
    // Y,4,-2, Re-Im
    // 3/8*sqrt(5/(2*pi)) = 0.33452327177864458
    temp_value = 0.33452327177864458*(-1+7*POW2(cos_theta))*POW2(sin_theta);
    d_stein_Ylm_cut[4] = temp_value * cos_2phi;
    d_stein_Ylm_cut[5] = temp_value * sin_2phi;
    // Y,4,3
    // Y,4,-3, -Re+Im
    // 3/8*sqrt(35/(pi)) = 1.25167147089835227
    temp_value = 1.25167147089835227*cos_theta*POW3(sin_theta);
    d_stein_Ylm_cut[6] = - temp_value * cos_3phi;
    d_stein_Ylm_cut[7] = - temp_value * sin_3phi;
    // Y,4,4
    // Y,4,-4, Re-Im
    // 3/16*sqrt(35/(2*pi)) = 0.44253269244498263
    temp_value = 0.44253269244498263*POW4(sin_theta);
    d_stein_Ylm_cut[8] = temp_value * cos_4phi;
    d_stein_Ylm_cut[9] = temp_value * sin_4phi;
    // DEBUG_LOG("d_stein_Ylm_cut 0, 1, 2, 3, 4 = %f, %f, %f, %f, %f",d_stein_Ylm_cut[0], d_stein_Ylm_cut[3], d_stein_Ylm_cut[4], d_stein_Ylm_cut[7], d_stein_Ylm_cut[8]);
}

__device__ __forceinline__ void compute_qlm_forward_L4(
    double r_weight,
    double cos_theta, double sin_theta, double cos_phi, double sin_phi,
    double cos_2phi, double sin_2phi,
    double cos_3phi, double sin_3phi,
    double cos_4phi, double sin_4phi,
    double *d_stein_qlm_cut, double *d_stein_Ylm_cut) {

    compute_Ylm_forward_L4(
        cos_theta, sin_theta, cos_phi, sin_phi,
        cos_2phi, sin_2phi,
        cos_3phi, sin_3phi,
        cos_4phi, sin_4phi,
        &d_stein_Ylm_cut[0]
    );
    
    // q,4,0
    d_stein_qlm_cut[0] += r_weight*d_stein_Ylm_cut[0];
    d_stein_qlm_cut[1] += r_weight*d_stein_Ylm_cut[1];
    // q,4,1
    d_stein_qlm_cut[2] += r_weight*d_stein_Ylm_cut[2];
    d_stein_qlm_cut[3] += r_weight*d_stein_Ylm_cut[3];
    // q,4,2
    d_stein_qlm_cut[4] += r_weight*d_stein_Ylm_cut[4];
    d_stein_qlm_cut[5] += r_weight*d_stein_Ylm_cut[5];
    // q,4,3
    d_stein_qlm_cut[6] += r_weight*d_stein_Ylm_cut[6];
    d_stein_qlm_cut[7] += r_weight*d_stein_Ylm_cut[7];
    // q,4,4
    d_stein_qlm_cut[8] += r_weight*d_stein_Ylm_cut[8];
    d_stein_qlm_cut[9] += r_weight*d_stein_Ylm_cut[9];
}


__device__ __forceinline__ void compute_Ylm_forward_L6(
    double cos_theta, double sin_theta, double cos_phi, double sin_phi,
    double cos_2theta, double sin_2theta, double cos_2phi, double sin_2phi,
    double cos_3phi, double sin_3phi,
    double cos_4theta, double sin_4theta, double cos_4phi, double sin_4phi,
    double cos_5phi, double sin_5phi,
    double cos_6theta, double sin_6theta, double cos_6phi, double sin_6phi,
    double *d_stein_Ylm_cut) {

    double temp_value;
    // DEBUG_LOG("delt x,y,z ; r2, r = %f, %f, %f, %f, %f", delt_x, delt_y, delt_z, r2, r);
    // Y,6,0
    temp_value = 0.001986537570863388*(50. + 105.*cos_2theta + 126.*cos_4theta + 231.*cos_6theta);
    d_stein_Ylm_cut[0] = temp_value;
    d_stein_Ylm_cut[1] = 0.0;

    // Y,6,1
    // Y,6,-1, -Re+Im
    temp_value = -0.01287423488441065*(5.*sin_2theta + 12.*sin_4theta + 33.*sin_6theta);
    d_stein_Ylm_cut[2] = temp_value * cos_phi;
    d_stein_Ylm_cut[3] = temp_value * sin_phi;

    // Y,6,2
    // Y,6,-2, -Re+Im
    temp_value = 0.04071190536673223*(35. + 60.*cos_2theta + 33.*cos_4theta)*POW2(sin_theta);
    d_stein_Ylm_cut[4] = temp_value * cos_2phi;
    d_stein_Ylm_cut[5] = temp_value * sin_2phi;

    // Y,6,3
    // Y,6,-3, -Re+Im
    temp_value = -0.3256952429338579*cos_theta*(5. + 11.*cos_2theta)*POW3(sin_theta);
    d_stein_Ylm_cut[6] = temp_value * cos_3phi;
    d_stein_Ylm_cut[7] = temp_value * sin_3phi;

    // Y,6,4
    // Y,6,-4, -Re+Im
    temp_value = 0.178390631426999*(9. + 11.*cos_2theta)*POW4(sin_theta);
    d_stein_Ylm_cut[8] = temp_value * cos_4phi;
    d_stein_Ylm_cut[9] = temp_value * sin_4phi;

    // Y,6,5
    // Y,6,-5, -Re+Im
    temp_value = -1.673452458100098*cos_theta*POW5(sin_theta);
    d_stein_Ylm_cut[10] = temp_value * cos_5phi;
    d_stein_Ylm_cut[11] = temp_value * sin_5phi;

    // Y,6,6
    // Y,6,-6, -Re+Im
    temp_value = 0.4830841135800662*POW6(sin_theta);
    d_stein_Ylm_cut[12] = temp_value * cos_6phi;
    d_stein_Ylm_cut[13] = temp_value * sin_6phi;

    // DEBUG_LOG("d_stein_Ylm 0, 1, 2, 3, 4 = %f, %f, %f, %f, %f",d_stein_Ylm_cut[0], d_stein_Ylm_cut[3], d_stein_Ylm_cut[4], d_stein_Ylm_cut[7], d_stein_Ylm_cut[8]);
}


__device__ __forceinline__ void compute_qlm_forward_L6(
    double r_weight,
    double cos_theta, double sin_theta, double cos_phi, double sin_phi,
    double cos_2theta, double sin_2theta, double cos_2phi, double sin_2phi,
    double cos_3phi, double sin_3phi,
    double cos_4theta, double sin_4theta, double cos_4phi, double sin_4phi,
    double cos_5phi, double sin_5phi,
    double cos_6theta, double sin_6theta, double cos_6phi, double sin_6phi,
    double *d_stein_qlm_cut, double *d_stein_Ylm_cut) {
    
    compute_Ylm_forward_L6(
        cos_theta, sin_theta, cos_phi, sin_phi,
        cos_2theta, sin_2theta, cos_2phi, sin_2phi,
        cos_3phi, sin_3phi,
        cos_4theta, sin_4theta, cos_4phi, sin_4phi,
        cos_5phi, sin_5phi,
        cos_6theta, sin_6theta, cos_6phi, sin_6phi,
        &d_stein_Ylm_cut[0]
    );
    
    // q,6,0
    d_stein_qlm_cut[0] += r_weight*d_stein_Ylm_cut[0];
    d_stein_qlm_cut[1] += r_weight*d_stein_Ylm_cut[1];
    // q,6,1
    d_stein_qlm_cut[2] += r_weight*d_stein_Ylm_cut[2];
    d_stein_qlm_cut[3] += r_weight*d_stein_Ylm_cut[3];
    // q,6,2
    d_stein_qlm_cut[4] += r_weight*d_stein_Ylm_cut[4];
    d_stein_qlm_cut[5] += r_weight*d_stein_Ylm_cut[5];
    // q,6,3
    d_stein_qlm_cut[6] += r_weight*d_stein_Ylm_cut[6];
    d_stein_qlm_cut[7] += r_weight*d_stein_Ylm_cut[7];
    // q,6,4
    d_stein_qlm_cut[8] += r_weight*d_stein_Ylm_cut[8];
    d_stein_qlm_cut[9] += r_weight*d_stein_Ylm_cut[9];
    // q,6,5
    d_stein_qlm_cut[10] += r_weight*d_stein_Ylm_cut[10];
    d_stein_qlm_cut[11] += r_weight*d_stein_Ylm_cut[11];
    // q,6,6
    d_stein_qlm_cut[12] += r_weight*d_stein_Ylm_cut[12];
    d_stein_qlm_cut[13] += r_weight*d_stein_Ylm_cut[13];

}


