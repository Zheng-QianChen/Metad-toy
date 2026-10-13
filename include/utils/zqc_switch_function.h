#ifndef SWITCH_FUNCTION_CUH
#define SWITCH_FUNCTION_CUH

#include <cuda_runtime.h>
#include <math.h>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "zqc_switch_function_device.cuh"

#include "error.h"
#include "comm.h"

#include "zqc_debug.h"
#include "lammps.h"
#include "utils.h"

namespace LAMMPS_NS { class FixMetadynamics; }

namespace MetaD_zqc {

    // Host-side metadata: a mathematical radius, never a neighbor-list skin.
    enum class RadialSupportKind { Finite, Epsilon, Unavailable };

    struct RadialSupport {
        RadialSupportKind kind;
        double radius;
        const char *error; // nullptr for a valid result, including manual-cutoff LINE.

        bool has_radius() const {
            return error == nullptr && kind != RadialSupportKind::Unavailable;
        }
    };

    class SwitchFunction : public SwitchFunctionMath {
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
        
        // Keep the existing static f(params,x), df(params,x) and f_*/df_* APIs.
        using SwitchFunctionMath::f;
        using SwitchFunctionMath::df;

        static SwitchFunction* get_default_step() {
            static SwitchFunction instance(LINE, 0.0, 0.0, 0.0, 0, 0);
            return &instance;
        }

        // Query radial support on r >= 0. Epsilon means a numerical tail
        // truncation, not exact compact support. LINE requires a caller cutoff.
        // RATIONAL currently accepts d_0 == 0 only: positive-offset plateau
        // semantics are not implemented in f()/df() and are a separate change.
        // 用来确定容差的。
        static __host__ inline RadialSupport query_radial_support(
                const SwitchFunctionRequest& p, double eps) {
            const auto unavailable = [](const char *error) {
                return RadialSupport{RadialSupportKind::Unavailable, 0.0, error};
            };
            if (!std::isfinite(eps) || !(eps > 0.0 && eps < 1.0))
                return unavailable("Radial cutoff epsilon must be finite and in (0,1).");

            double radius = 0.0;
            switch (p.type) {
                case LINE:
                    return unavailable(nullptr);
                case STEP:
                    if (!std::isfinite(p.r_0) || !(p.r_0 > 0.0))
                        return unavailable("Radial STEP requires a finite positive r_0.");
                    return {RadialSupportKind::Finite, p.r_0, nullptr};
                case FERMI:
                case TANH_TYPE: {
                    if (!std::isfinite(p.r_0) || !std::isfinite(p.alpha)
                            || !(p.alpha < 0.0))
                        return unavailable("Radial FERMI/TANH requires finite r_0 and negative alpha.");
                    const double log_odds = std::log1p(-eps) - std::log(eps);
                    const double factor = p.type == FERMI ? 1.0 : 0.5;
                    radius = p.r_0 - factor * log_odds / p.alpha;
                    break;
                }
                case RATIONAL: {
                    if (!std::isfinite(p.r_0) || !(p.r_0 > 0.0)
                            || !std::isfinite(p.d_0) || p.d_0 != 0.0
                            || !(p.n > 0 && p.m > p.n))
                        return unavailable("Radial RATIONAL requires r_0>0, d_0=0, and 0<n<m.");
                    double x = 0.0;
                    if (2LL * p.n == p.m) {
                        x = std::exp((std::log1p(-eps) - std::log(eps)) / p.n);
                    } else {
                        // Bracket the decreasing dimensionless function on [0,hi].
                        // This helper handles x=1 without the 0/0 in the generic
                        // value formula, and avoids overflow in the long tail.
                        double lo = 0.0, hi = 1.0;
                        while (radial_rational_weight(hi, p.n, p.m) > eps) {
                            if (hi > std::numeric_limits<double>::max() / 2.0)
                                return unavailable("Radial RATIONAL could not bracket a finite epsilon radius.");
                            hi *= 2.0;
                        }
                        for (int iter = 0; iter < 100; ++iter) {
                            const double mid = lo + 0.5 * (hi - lo);
                            if (mid == lo || mid == hi) break;
                            if (radial_rational_weight(mid, p.n, p.m) > eps)
                                lo = mid;
                            else
                                hi = mid;
                        }
                        x = hi;
                    }
                    radius = p.r_0 * x;
                    break;
                }
                default:
                    return unavailable("Unknown radial switching-function type.");
            }
            if (!std::isfinite(radius) || !(radius > 0.0))
                return unavailable("Radial epsilon cutoff must be finite and positive; check switch parameters and epsilon.");
            return {RadialSupportKind::Epsilon, radius, nullptr};
        }

        // Compatibility wrappers. Callers needing diagnostics should use the
        // metadata query; unavailable/invalid support never returns a fake radius.
        static __host__ inline double invert_for_eps(
                const SwitchFunctionRequest& p, double eps) {
            const auto support = query_radial_support(p, eps);
            return support.has_radius() ? support.radius
                : std::numeric_limits<double>::quiet_NaN();
        }

        static __host__ inline double invert_bisect(
                const SwitchFunctionRequest& p, double eps) {
            return invert_for_eps(p, eps);
        }

    private:
        static __host__ inline double radial_rational_weight(double x, int n, int m) {
            if (x == 0.0) return 1.0;
            if (x == 1.0) return static_cast<double>(n) / m;
            const double log_x = std::log(x);
            if (x < 1.0)
                return std::expm1(n * log_x) / std::expm1(m * log_x);
            return std::exp((static_cast<double>(n) - m) * log_x)
                * std::expm1(-n * log_x) / std::expm1(-m * log_x);
        }

    };
}

#endif