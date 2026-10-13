#ifndef ZQC_SWITCH_FUNCTION_DEVICE_CUH
#define ZQC_SWITCH_FUNCTION_DEVICE_CUH

// Parameter adaptation only; scalar formulas live in the embedded graph header.
#include "zqc_graph_device.cuh"

namespace MetaD_zqc {
enum SwitchType { LINE = 0, STEP = 1, FERMI = 2, TANH_TYPE = 3, RATIONAL = 4 };
struct SwitchFunctionRequest {
    SwitchType type;
    double r_0; // threshold or characteristic distance
    double d_0; // Rational input offset
    double alpha; // Fermi/Tanh slope
    int n;
    int m;
};

// Legacy parameter dispatch for radial switching; no mathematical formulas here.
struct SwitchFunctionMath {
    static __host__ __device__ __forceinline__ double f_step(const SwitchFunctionRequest& p, double x) {
        return ::zqc_graph_step_switch(x, p.r_0);
    }
    static __host__ __device__ __forceinline__ double df_step(const SwitchFunctionRequest& p, double x) {
        return ::zqc_graph_step_switch_df(x, p.r_0);
    }
    static __host__ __device__ __forceinline__ double f_fermi(const SwitchFunctionRequest& p, double x) {
        return ::zqc_graph_fermi(x, p.r_0, p.alpha);
    }
    static __host__ __device__ __forceinline__ double df_fermi(const SwitchFunctionRequest& p, double x) {
        return ::zqc_graph_fermi_df(x, p.r_0, p.alpha);
    }
    static __host__ __device__ __forceinline__ double f_tanh(const SwitchFunctionRequest& p, double x) {
        return ::zqc_graph_tanh_switch(x, p.r_0, p.alpha);
    }
    static __host__ __device__ __forceinline__ double df_tanh(const SwitchFunctionRequest& p, double x) {
        return ::zqc_graph_tanh_switch_df(x, p.r_0, p.alpha);
    }
    static __host__ __device__ __forceinline__ double f_rational(const SwitchFunctionRequest& p, double x) {
        return ::zqc_graph_rational_switch(x, p.r_0, p.n, p.m, p.d_0);
    }
    static __host__ __device__ __forceinline__ double df_rational(const SwitchFunctionRequest& p, double x) {
        return ::zqc_graph_rational_switch_df(x, p.r_0, p.n, p.m, p.d_0);
    }
    static __host__ __device__ __forceinline__ double f(const SwitchFunctionRequest& p, double x) {
        switch (p.type) {
            case LINE: return ::zqc_graph_constant_one(x);
            case STEP: return f_step(p,x);
            case FERMI: return f_fermi(p,x);
            case TANH_TYPE: return f_tanh(p,x);
            case RATIONAL: return f_rational(p,x);
            default: return 1.0;
        }
    }
    static __host__ __device__ __forceinline__ double df(const SwitchFunctionRequest& p, double x) {
        switch (p.type) {
            case LINE: return ::zqc_graph_constant_one_df(x);
            case STEP: return df_step(p,x);
            case FERMI: return df_fermi(p,x);
            case TANH_TYPE: return df_tanh(p,x);
            case RATIONAL: return df_rational(p,x);
            default: return 0.0;
        }
    }
};
} // namespace MetaD_zqc
#endif
