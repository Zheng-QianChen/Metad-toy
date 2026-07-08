#ifndef SWITCH_FUNCTION_CUH
#define SWITCH_FUNCTION_CUH

#include <cuda_runtime.h>
#include <math.h>
#include <cmath>

#include "error.h"
#include "comm.h"

#include "zqc_debug.h"
#include "fix_crystallize.h"

namespace MetaD_zqc {

    enum SwitchType {
        LINE = 0,
        STEP = 1,
        FERMI = 2,
        TANH_TYPE = 3,
        RATIONAL = 4  // 标准 PLUMED 经典有理函数
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
        // SwitchType type;
        // double r_0;    // 基础特征距离/阈值（对应 Fermi 里的 q_bar 或 Rational 里的 r_0）
        // double d_0;
        // double alpha;  // 陡峭系数（针对 Fermi/Tanh）
        // int n;         // 针对 Rational 的分子幂次（通常为 6）
        // int m;         // 针对 Rational 的分母幂次（通常为 12）
        SwitchFunctionRequest params;

        std::string get_summary_string() const {
            char buf[256];
            switch (params.type) {
                case LINE:
                    std::snprintf(buf, sizeof(buf), "Type=LINE, f=1, df=0");
                    break;
                case STEP:
                    std::snprintf(buf, sizeof(buf), "Type=STEP, r_0=%g, f=1[x < r_0];0[x > r_0], df=0", 
                                    params.r_0);
                    break;
                case FERMI:
                    std::snprintf(buf, sizeof(buf), "Type=FERMI, r_0=%g, alpha=%g", 
                                    params.r_0, params.alpha);
                    break;
                case TANH_TYPE:
                    std::snprintf(buf, sizeof(buf), "Type=TANH, r_0=%g, alpha=%g", 
                                    params.r_0, params.alpha);
                    break;
                case RATIONAL:
                    std::snprintf(buf, sizeof(buf), "Type=RATIONAL, r_0=%g, d_0=%g, n=%d, m=%d", 
                                    params.r_0, params.d_0, params.n, params.m);
                    break;
                default:
                    std::snprintf(buf, sizeof(buf), "Type=Unknown");
                    break;
            }
            return std::string(buf);
        }

        // 默认构造函数
        SwitchFunction() 
            : params({RATIONAL, 1.25, 0.0, 20.0, 6, 12}) {}

        // 动态全参数构造函数
        SwitchFunction(SwitchType _type, double _r_0, double _d_0, 
                double _alpha, int _n = 6, int _m = 12)
            : params({_type, _r_0, _d_0, _alpha, _n, _m}) {}

        static inline SwitchFunction* create(LAMMPS_NS::LAMMPS *lmp, LAMMPS_NS::FixMetadynamics *Fixmetad,
                         FILE *f_check, int narg, char **arg, int &i) {
            DEBUG_LOG("In SW_FUNC settings");
            LAMMPS_NS::Error *error = lmp->error;

            i++;

            // 1. 初始化一个默认请求参数包（由于放弃多态，直接在栈上初始化默认结构体）
            SwitchFunctionRequest req;
            req.type = RATIONAL;
            req.r_0 = 1.25;
            req.d_0 = 0.0;
            req.alpha = 20.0;
            req.n = 6;
            req.m = 12;

            // 🚨 此时 arg[i] 应该是开关函数子类型的关键字（例如：FERMI, TANH, RATIONAL）
            ERR_COND(i >= narg, "Error: SW_FUNC command requires a type (LINE, STEP, FERMI, TANH, RATIONAL).");
            std::string sw_type_str = arg[i];
            
            if (sw_type_str == "STEP") {
                req.type = STEP;
                req.r_0 = 1.0;
            } else if (sw_type_str == "FERMI") {
                req.type = FERMI;
                req.r_0 = 1.0; 
                req.alpha = 20.0;
            } else if (sw_type_str == "TANH") {
                req.type = TANH_TYPE;
                req.r_0 = 1.0;
                req.alpha = 20.0;
            } else if (sw_type_str == "RATIONAL") {
                req.type = RATIONAL;
                req.r_0 = 1.25;
                req.d_0 = 0.0;
                req.n = 6;
                req.m = 12;
            } else {
                ERR_COND(1, "Error: Unknown SW_func type. Choose from FERMI, TANH, RATIONAL.");
            }

            // 2. 仿照你的风格，使用 iarg 从 i+1 开始向后解析亚参数
            int iarg = i + 1;
            printf("im in SwitchFunction::create, with type=%s\n", sw_type_str.c_str());

            while (iarg < narg) {
                if (strcmp(arg[iarg], "r_0") == 0) {
                    ERR_COND((iarg + 1 >= narg), "Error: 'r_0' requires a numeric value");
                    req.r_0 = LAMMPS_NS::utils::numeric(FLERR, arg[iarg + 1], false, lmp);
                    iarg += 2;
                } else if (strcmp(arg[iarg], "d_0") == 0) {
                    ERR_COND((iarg + 1 >= narg), "Error: 'd_0' requires a numeric value");
                    req.d_0 = LAMMPS_NS::utils::numeric(FLERR, arg[iarg + 1], false, lmp);
                    iarg += 2;
                } else if (strcmp(arg[iarg], "alpha") == 0) {
                    ERR_COND((iarg + 1 >= narg), "Error: 'alpha' requires a numeric value");
                    req.alpha = LAMMPS_NS::utils::numeric(FLERR, arg[iarg + 1], false, lmp);
                    iarg += 2;
                } else if (strcmp(arg[iarg], "n") == 0) {
                    ERR_COND((iarg + 1 >= narg), "Error: 'n' requires an integer value");
                    req.n = LAMMPS_NS::utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
                    iarg += 2;
                } else if (strcmp(arg[iarg], "m") == 0) {
                    ERR_COND((iarg + 1 >= narg), "Error: 'm' requires an integer value");
                    req.m = LAMMPS_NS::utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
                    iarg += 2;
                } else {
                    // 遇到不属于当前开关函数的参数（例如到了下一个主关键字 CAL 等），退出当前解析
                    break;
                }
            }

            // 3. 🚨【核心】同步更新外层的参数索引指针 `i`，确保外层大循环不会乱
            i = iarg;

            LOG("[Metad-toy LOG] SW_FUNC parsed: %s, r_0=%g, d_0=%g, alpha=%g, n=%d, m=%d",
                        sw_type_str.c_str(), req.r_0, req.d_0, req.alpha, req.n, req.m);

            // 4. 通过动态全参数构造函数，new 出统一的计算类实例，返回并注册到 sw_registry 中
            return new MetaD_zqc::SwitchFunction(req.type, req.r_0, req.d_0, req.alpha, req.n, req.m);
        }

        __host__ __forceinline__ double f(double S_i) const {
            return SwitchFunction::f(this->params, S_i); 
        }
        
        // 🌟 1. 计算 f_sw(S_i)
        static __host__ __device__ __forceinline__ double f(const SwitchFunctionRequest& p, double S_i) {
            int n = p.n;
            int m = p.m;
            switch (p.type) {
                case LINE: {
                    return 1.0;
                }
                case STEP: {
                    return f_step(p, S_i);
                }
                case FERMI: {
                    return f_fermi(p, S_i);
                }
                case TANH_TYPE: {
                    return f_tanh(p, S_i);
                }
                case RATIONAL: {
                    return f_rational(p, S_i);
                }
                default: return 1.0;
            }
        }

        // 🌟 2. 计算解析导数 df_sw / dS_i
        static __host__ __device__ __forceinline__ double df(const SwitchFunctionRequest& p, double S_i) {
            int n = p.n;
            int m = p.m;
            switch (p.type) {
                case LINE: {
                    return 0.0;
                }
                case STEP: {
                    return df_step(p, S_i);
                }
                case FERMI: {
                    return df_fermi(p, S_i);
                }
                case TANH_TYPE: {
                    return df_tanh(p, S_i);
                }
                case RATIONAL: {
                    return df_rational(p, S_i);
                }
                default: return 0.0;
            }
        }

        static SwitchFunction* get_default_step() {
            static SwitchFunction instance(LINE, 0.0, 0.0, 0.0, 0, 0);
            return &instance;
        }

        static __host__ __device__ __forceinline__ double f_step(const SwitchFunctionRequest& p, double S_i) {
            if (S_i < p.r_0) {
                return 1.0;
            } else {
                return 0.0;
            }
        }

        static __host__ __device__ __forceinline__ double df_step(const SwitchFunctionRequest& p, double S_i) {
            return 0.0;
        }

        static __host__ __device__ __forceinline__ double f_fermi(const SwitchFunctionRequest& p, double S_i) {
            return 1.0 / (1.0 + exp(-p.alpha * (S_i - p.r_0)));
        }

        static __host__ __device__ __forceinline__ double df_fermi(const SwitchFunctionRequest& p, double S_i) {
            double exp_term = exp(-p.alpha * (S_i - p.r_0));
            double denom = 1.0 + exp_term;
            return (p.alpha * exp_term) / (denom * denom);
        }

        static __host__ __device__ __forceinline__ double f_tanh(const SwitchFunctionRequest& p, double S_i) {
            return 0.5 * (1.0 + tanh(p.alpha * (S_i - p.r_0)));
        }

        static __host__ __device__ __forceinline__ double df_tanh(const SwitchFunctionRequest& p, double S_i) {
            double t = tanh(p.alpha * (S_i - p.r_0));
            return 0.5 * p.alpha * (1.0 - t * t);
        }

        static __host__ __device__ __forceinline__ double f_rational(const SwitchFunctionRequest& p, double S_i) {
            int n = p.n;
            int m = p.m;
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

        static __host__ __device__ __forceinline__ double df_rational(const SwitchFunctionRequest& p, double S_i) {
            int n = p.n;
            int m = p.m;
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

        
        // 获得 f(r) = eps 的反解 r, 通过数值二分法求解
        // 数值二分法兜底 (适用于任意单调递减的开关函数, 包括通用n,m的RATIONAL)
        static __host__ inline double invert_bisect(const SwitchFunctionRequest& p, double target_eps) {
            double lo = 0.0, hi = 1.0;
            // 先找一个足够大的hi, 使f(hi) < target_eps
            while (f(p, hi) > target_eps) {
                hi *= 2.0;
                if (hi > 1e8) break;  // 保护, 防止死循环(比如STEP这种恒为1的类型不应该调这个函数)
            }
            // 标准二分, 假设f在[lo,hi]上单调递减
            for (int iter = 0; iter < 100; iter++) {
                double mid = 0.5*(lo+hi);
                if (f(p, mid) > target_eps) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            return 0.5*(lo+hi);
        }

        // 给定希望达到的截断精度eps, 反推出对应的距离r (即 f(r)=eps 的解)
        // 对每种类型, 优先用闭式解(快, 精确), 无法闭式求解时用数值二分兜底
        static __host__ inline double invert_for_eps(const SwitchFunctionRequest& p, double eps) {
            switch (p.type) {
                case STEP: {
                    // STEP恒为1, 没有衰减, 反推没有意义, 直接返回一个不可用的标记
                    return p.r_0;
                }
                case FERMI: {
                    return p.r_0 - std::log(1.0/eps - 1.0)/p.alpha;
                }
                case TANH_TYPE: {
                    return p.r_0 + std::atanh(2.0*eps - 1.0)/p.alpha;
                }
                case RATIONAL: {
                    if (p.n == 6 && p.m == 12) {
                        double x = std::pow(1.0/eps - 1.0, 1.0/6.0);
                        return x*p.r_0 + p.d_0;
                    } else if (p.n == 4 && p.m == 8) {
                        double x = std::pow(1.0/eps - 1.0, 1.0/4.0);
                        return x*p.r_0 + p.d_0;
                    } else if (2*p.n == p.m) {
                        double x = std::pow(1.0/eps - 1.0, 1.0/p.n);
                        return x*p.r_0 + p.d_0;
                    }
                    // 通用n,m情况, 没有简单闭式解, 用二分法兜底
                    return invert_bisect(p, eps);
                }
                default: return -1.0;
            }
        }
    };
}

#endif