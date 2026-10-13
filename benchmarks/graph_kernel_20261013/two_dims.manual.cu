#ifndef __CUDACC_RTC__
#include <cuda_runtime.h>
#include <cmath>
#endif

extern "C" __global__ void cv_forward(int n,const double* a,const double* b,const double* c,double* y1,double* y2) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    y1[i]=a[i]*b[i];
    y2[i]=a[i]*c[i];
}
extern "C" __global__ void cv_backward(int n,const double* a,const double* b,const double* c,const double* s,const double* t,double* ga,double* gb,double* gc) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    ga[i]=s[i]*b[i]+t[i]*c[i];
    gb[i]=s[i]*a[i];
    gc[i]=t[i]*a[i];
}
