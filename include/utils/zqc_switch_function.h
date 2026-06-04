#ifndef SWITCH_FUNCTION_CUH
#define SWITCH_FUNCTION_CUH

#include <cuda_runtime.h>
#include <math.h>
#include <cmath>
#include "zqc_debug.h"

namespace MetaD_zqc {

    enum SwitchType {
        FERMI = 0,
        TANH_TYPE = 1,
        RATIONAL = 2  // 标准 PLUMED 经典有理函数
    };

    struct SwitchFunctionRequest {
        SwitchType type;
        double r_0;    // 基础特征距离/阈值（对应 Fermi 里的 q_bar 或 Rational 里的 r_0）
        double d_0;
        double alpha;  // 陡峭系数（针对 Fermi/Tanh）
        int n;         // 针对 Rational 的分子幂次（通常为 6）
        int m;         // 针对 Rational 的分母幂次（通常为 12）
    };

    class SwitchFunction {
    public:
        SwitchType type;
        double r_0;    // 基础特征距离/阈值（对应 Fermi 里的 q_bar 或 Rational 里的 r_0）
        double d_0;
        double alpha;  // 陡峭系数（针对 Fermi/Tanh）
        int n;         // 针对 Rational 的分子幂次（通常为 6）
        int m;         // 针对 Rational 的分母幂次（通常为 12）

        std::string get_summary_string() const {
            char buf[256];
            switch (type) {
                case FERMI:
                    std::snprintf(buf, sizeof(buf), "Type=FERMI, r_0=%g, alpha=%g", r_0, alpha);
                    break;
                case TANH_TYPE:
                    std::snprintf(buf, sizeof(buf), "Type=TANH, r_0=%g, alpha=%g", r_0, alpha);
                    break;
                case RATIONAL:
                    std::snprintf(buf, sizeof(buf), "Type=RATIONAL, r_0=%g, d_0=%g, n=%d, m=%d", r_0, d_0, n, m);
                    break;
                default:
                    std::snprintf(buf, sizeof(buf), "Type=Unknown");
                    break;
            }
            return std::string(buf);
        }

        // 默认构造函数
        SwitchFunction() 
            : type(RATIONAL), r_0(1.25), d_0(0.0), alpha(20.0), n(6), m(12) {}

        // 动态全参数构造函数
        SwitchFunction(SwitchType _type, double _r_0, double _d_0, 
                double _alpha, int _n = 6, int _m = 12)
            : type(_type), r_0(_r_0), d_0(_d_0), alpha(_alpha), n(_n), m(_m) {}
        
        // 🌟 1. 计算 f_sw(S_i)
        static __host__ __device__ __forceinline__ double f(const SwitchFunctionRequest& p, double S_i) {
            int n = p.n;
            int m = p.m;
            switch (p.type) {
                case FERMI: {
                    return 1.0 / (1.0 + exp(-p.alpha * (S_i - p.r_0)));
                }
                case TANH_TYPE: {
                    return 0.5 * (1.0 + tanh(p.alpha * (S_i - p.r_0)));
                }
                case RATIONAL: {
                    double x = (S_i-p.d_0) / p.r_0;
                    if (n == 6 && m == 12) {
                        // PLUMED 标准有理函数的优化版本：
                        double x2 = POW2(x);       // x^2
                        double x6 = POW3(x2);     // x^6
                        return 1.0 / (1.0 + x6); // (1-x^6)/(1-x^12)=1/(1+x^6)
                    } else if (n==4 && m==8) {
                        // 4-8 有理函数的优化版本：
                        double x2 = POW2(x);       // x^2
                        double x4 = POW2(x2);     // x^4
                        return 1.0 / (1.0 + x4); // (1-x^4)/(1-x^8)=1/(1+x^4)
                    } else if (2*n==m){
                        // m=2n 的特殊情况优化：
                        double x_n = std::pow(x, n);   // x^n
                        return 1.0 / (1.0 + x_n); // (1-x^n)/(1-x^(2m))=1/(1+x^n)
                    } else if (n==2*m){
                        // n=2m 的特殊情况优化：
                        double x_m = std::pow(x, m);   // x^m
                        return (1.0 + x_m); // (1-x^m)/(1-x^(2m))=1/(1+x^m)
                    }
                    return (1-std::pow(x, n))/(1-std::pow(x, m));
                }
                default: return 0.0;
            }
        }

        // 🌟 2. 计算解析导数 df_sw / dS_i
        static __host__ __device__ __forceinline__ double df(const SwitchFunctionRequest& p, double S_i) {
            int n = p.n;
            int m = p.m;
            switch (p.type) {
                case FERMI: {
                    double exp_term = exp(-p.alpha * (S_i - p.r_0));
                    double denom = 1.0 + exp_term;
                    return (p.alpha * exp_term) / (denom * denom);
                }
                case TANH_TYPE: {
                    double t = tanh(p.alpha * (S_i - p.r_0));
                    return 0.5 * p.alpha * (1.0 - t * t);
                }
                case RATIONAL: {
                    double x = (S_i-p.d_0) / p.r_0;
                    if (n == 6 && m == 12) {
                        // PLUMED 标准有理函数的优化版本：
                        double x2 = POW2(x);       // x^2
                        double x5 = POW2(x2)*x;     // x^5
                        double x6 = POW3(x2);     // x^6
                        return -n*x5 / POW2(1.0 + x6) / p.r_0; // (1-x^6)/(1-x^12)=1/(1+x^6)
                    } else if (n==4 && m==8) {
                        // 4-8 有理函数的优化版本：
                        double x2 = POW2(x);       // x^2
                        double x3 = x2 * x;        // x^3
                        double x4 = POW2(x2);     // x^4
                        return -(double)n*x3 / POW2(1.0 + x4) / p.r_0; // (1-x^4)/(1-x^8)=1/(1+x^4)
                    } else if (2*n==m){
                        // m=2n 的特殊情况优化：
                        double x_nm1 = std::pow(x, n-1);   // x^n
                        double x_n = std::pow(x, n);   // x^n
                        return -(double)n*x_nm1 / POW2(1.0 + x_n) / p.r_0; // (1-x^n)/(1-x^(2m))=1/(1+x^n)
                    } else if (n==2*m){
                        // n=2m 的特殊情况优化：
                        double x_mm1 = std::pow(x, m-1);   // x^n
                        return (double)m*x_mm1 / p.r_0; // (1-x^m)/(1-x^(2m))=1/(1+x^m)
                    }
                    double x_mm1 = std::pow(x, m-1);       // x^(m-1)
                    double x_m = x_mm1 * x;       // x^m

                    if (std::fabs(1.0 - x_m) < 1e-12) {
                        return -0.5 * (double)(n * (n - m)) / (double)m / p.r_0;
                    }

                    double x_nm1 = std::pow(x, n - 1); // x^(n-1)
                    double x_n = x_nm1 * x;       // x^n
                    return ((double)(n-m)*x_nm1*x_m + m*x_mm1 - n*x_nm1)/POW2(1-x_m) / p.r_0; // (m*x^(m-1) - n*x^(n-1) + (n-m)*x^(n+m-1)) / (1-x^m)^2
                }
                default: return 0.0;
            }
        }
    };
}

#endif