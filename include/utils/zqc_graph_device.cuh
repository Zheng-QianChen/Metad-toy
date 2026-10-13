#ifndef ZQC_GRAPH_DEVICE_CUH
#define ZQC_GRAPH_DEVICE_CUH

// 离线 nvcc 编译引入头文件；NVRTC 使用内置的 CUDA 类型与数学函数。
#ifndef __CUDACC_RTC__
#include <cuda_runtime.h>
#include <cmath>
#endif

// 乘法
__device__ __forceinline__
double zqc_graph_mul(double a, double b) { return a * b; }
__device__ __forceinline__
void zqc_graph_mul_bwd(double adj, double a, double b, double *a_adjv, double *b_adjv) { *a_adjv += adj*b; *b_adjv += adj*a; }

// 加法
__device__ __forceinline__
double zqc_graph_add(double a, double b) { return a + b; }
__device__ __forceinline__
void zqc_graph_add_bwd(double adj, double *a_adjv, double *b_adjv) { *a_adjv += adj; *b_adjv += adj; }

// 取负：反向累加到输入梯度，保留其他分支的贡献。
__device__ __forceinline__
double zqc_graph_neg(double a) { return -a; }
__device__ __forceinline__
void zqc_graph_neg_bwd(double adj, double *a_adjv) { *a_adjv -= adj; }

// 线性函数
__device__ __forceinline__
double zqc_graph_linear(double a, double k0, double b0) { return k0 * a + b0; }
__device__ __forceinline__
void zqc_graph_linear_bwd(double adj, double k0, double *a_adjv) { *a_adjv += adj*k0; }

// 指数函数
__device__ __forceinline__
double zqc_graph_exp(const double a) { return exp(a); }
__device__ __forceinline__
void zqc_graph_exp_bwd(const double adj, const double a, double *a_adjv) { *a_adjv += adj*exp(a);}

// Scalar formulas shared by graph codegen and the radial switching adapter.
// Keep definitions here so the embedded NVRTC source is self-contained.
__host__ __device__ __forceinline__ double zqc_graph_constant_one(double) { return 1.0; }
__host__ __device__ __forceinline__ double zqc_graph_constant_one_df(double) { return 0.0; }

__host__ __device__ __forceinline__ double zqc_graph_step_switch(double x, double r0) {
    return x < r0 ? 1.0 : 0.0;
}

// STEP retains the zero-derivative convention, including at the threshold.
__host__ __device__ __forceinline__ double zqc_graph_step_switch_df(double, double) {
    return 0.0;
}

__host__ __device__ __forceinline__ double zqc_graph_fermi(double x, double r0, double alpha) {
    const double z = alpha * (x - r0);
    if (z >= 0.0) return 1.0 / (1.0 + exp(-z));
    const double e = exp(z);
    return e / (1.0 + e);
}

__host__ __device__ __forceinline__ double zqc_graph_fermi_df(double x, double r0, double alpha) {
    const double e = exp(-fabs(alpha * (x - r0)));
    const double denom = 1.0 + e;
    return alpha * (e / (denom * denom));
}

__host__ __device__ __forceinline__ double zqc_graph_tanh_switch(double x, double r0, double alpha) {
    return 0.5 * (1.0 + tanh(alpha * (x - r0)));
}

__host__ __device__ __forceinline__ double zqc_graph_tanh_switch_df(double x, double r0, double alpha) {
    const double t = tanh(alpha * (x - r0));
    return 0.5 * alpha * (1.0 - t * t);
}

// Horner evaluation of the geometric sum and its derivative near x=1.
__host__ __device__ __forceinline__ void zqc_graph_rational_geometric(int k, double x, double& a, double& da) {
    a = 1.0; da = 0.0;
    for (int j = 1; j < k; ++j) {
        da = da * x + a;
        a = a * x + 1.0;
    }
}

__host__ __device__ __forceinline__ double zqc_graph_rational_switch(double value, double r0, int n, int m, double d0) {
    double x = (value - d0) / r0;
    // When both powers are even, the same removable limit also occurs at -1.
    if ((n % 2 == 0) && (m % 2 == 0)) x = fabs(x);
    if (2 * static_cast<long long>(n) == m) {
        double xn;
        if (n == 6) { const double x2=x*x; xn=x2*x2*x2; }
        else if (n == 4) { const double x2=x*x; xn=x2*x2; }
        else xn=pow(x,n);
        return 1.0 / (1.0 + xn);
    }
    if (n == 2 * static_cast<long long>(m)) return 1.0 + pow(x,m);
    if (fabs(x - 1.0) < 1e-4) {
        double a, da, b, db;
        zqc_graph_rational_geometric(n,x,a,da); zqc_graph_rational_geometric(m,x,b,db);
        return a / b;
    }
    return (1.0 - pow(x,n)) / (1.0 - pow(x,m));
}

__host__ __device__ __forceinline__ double zqc_graph_rational_switch_df(double value, double r0, int n, int m, double d0) {
    double x = (value - d0) / r0;
    double sign = 1.0;
    if ((n % 2 == 0) && (m % 2 == 0) && x < 0.0) { x=-x; sign=-1.0; }
    double deriv;
    if (2 * static_cast<long long>(n) == m) {
        double xn, xnm1;
        if (n == 6) { const double x2=x*x; xnm1=x2*x2*x; xn=xnm1*x; }
        else if (n == 4) { const double x2=x*x; xnm1=x2*x; xn=xnm1*x; }
        else { xnm1=pow(x,n-1); xn=xnm1*x; }
        const double denom=1.0+xn;
        deriv=-static_cast<double>(n)*xnm1/(denom*denom);
    } else if (n == 2 * static_cast<long long>(m)) {
        deriv=static_cast<double>(m)*pow(x,m-1);
    } else if (fabs(x - 1.0) < 1e-4) {
        double a, da, b, db;
        zqc_graph_rational_geometric(n,x,a,da); zqc_graph_rational_geometric(m,x,b,db);
        deriv=(da*b-a*db)/(b*b);
    } else {
        const double xn=pow(x,n), xm=pow(x,m);
        const double denom=1.0-xm;
        deriv=(-static_cast<double>(n)*pow(x,n-1)*denom
               +static_cast<double>(m)*pow(x,m-1)*(1.0-xn))/(denom*denom);
    }
    return sign * deriv / r0;
}

// Graph reverse-mode adapters accumulate into the input adjoint.
__device__ __forceinline__
void zqc_graph_constant_one_bwd(double adj, double a, double* a_adj) {
    *a_adj += adj * zqc_graph_constant_one_df(a);
}

__device__ __forceinline__
void zqc_graph_step_switch_bwd(double adj, double a, double r0, double* a_adj) {
    *a_adj += adj * zqc_graph_step_switch_df(a, r0);
}

__device__ __forceinline__
void zqc_graph_fermi_bwd(double adj, double a, double r0, double alpha, double* a_adj) {
    *a_adj += adj * zqc_graph_fermi_df(a, r0, alpha);
}

__device__ __forceinline__
void zqc_graph_tanh_switch_bwd(double adj, double a, double r0, double alpha, double* a_adj) {
    *a_adj += adj * zqc_graph_tanh_switch_df(a, r0, alpha);
}

__device__ __forceinline__
void zqc_graph_rational_switch_bwd(double adj, double a, double r0, int power_n, int power_m, double d0, double* a_adj) {
    *a_adj += adj * zqc_graph_rational_switch_df(a, r0, power_n, power_m, d0);
}

#endif // ZQC_GRAPH_DEVICE_CUH
