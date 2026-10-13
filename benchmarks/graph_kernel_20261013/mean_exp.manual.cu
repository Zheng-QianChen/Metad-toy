#ifndef __CUDACC_RTC__
#include <cuda_runtime.h>
#include <cmath>
#endif

extern "C" __global__ void cv_forward(int n,const double* a,const double* b,const double* c,const double* d,double* out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    out[i]=a[i]+b[i]*c[i]-exp(d[i]);
}
extern "C" __global__ void cv_backward(int n,const double* a,const double* b,const double* c,const double* d,double seed,double* ga,double* gb,double* gc,double* gd) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    ga[i]=seed;
    gb[i]=seed*c[i];
    gc[i]=seed*b[i];
    gd[i]=-seed*exp(d[i]);
}
