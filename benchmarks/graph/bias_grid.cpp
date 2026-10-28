// BiasGrid 独立回归：1/2/3D、dense/sparse、夹紧/周期、串并行重建、解析梯度。
#include "zqc_bias_grid.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using MetaD_zqc::BiasGrid;
using MetaD_zqc::GridAxis;

int g_failures = 0;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++g_failures;
    }
}

bool nearly_equal(double a, double b, double abs_tol, double rel_tol = 0.0) {
    const double diff = std::abs(a - b);
    const double scale = std::max(std::abs(a), std::abs(b));
    return diff <= abs_tol + rel_tol * scale;
}

constexpr double kPi = 3.14159265358979323846;

// 与实现相同的解析高斯列表能量（无网格插值），用于容差对照。
double exact_hill_energy(const std::vector<GridAxis>& axes,
                         const std::vector<double>& centers,
                         const std::vector<double>& heights,
                         const double* point) {
    const size_t dim = axes.size();
    const size_t count = heights.size();
    double energy = 0.0;
    for (size_t hill = 0; hill < count; ++hill) {
        double product = heights[hill];
        for (size_t axis = 0; axis < dim; ++axis) {
            const GridAxis& a = axes[axis];
            const double center = centers[hill * dim + axis];
            if (a.period > 0.0) {
                double delta = center - point[axis];
                delta -= a.period * std::nearbyint(delta / a.period);
                const double argument = kPi * delta / a.period;
                const double sine_half = std::sin(argument);
                const double scale = kPi * a.sigma / a.period;
                const double q = (sine_half / scale) * (sine_half / scale);
                product *= std::exp(-0.5 * q);
            } else {
                const double delta = center - point[axis];
                const double scaled = delta / a.sigma;
                product *= std::exp(-0.5 * scaled * scaled);
            }
        }
        energy += product;
    }
    return energy;
}

std::vector<double> bin_center_point(const std::vector<GridAxis>& axes,
                                     const std::vector<int>& indices) {
    std::vector<double> point(axes.size());
    for (size_t axis = 0; axis < axes.size(); ++axis) {
        const double spacing =
            (axes[axis].upper - axes[axis].lower) /
            static_cast<double>(axes[axis].bins);
        point[axis] =
            axes[axis].lower + (static_cast<double>(indices[axis]) + 0.5) * spacing;
    }
    return point;
}

void check_dense_sparse_agree(const std::vector<GridAxis>& axes,
                              const std::vector<double>& centers,
                              const std::vector<double>& heights,
                              const std::vector<double>& probes,
                              const char* label) {
    BiasGrid dense(axes, false);
    BiasGrid sparse(axes, true);
    const size_t dim = axes.size();
    for (size_t hill = 0; hill < heights.size(); ++hill) {
        dense.add(centers.data() + hill * dim, heights[hill]);
        sparse.add(centers.data() + hill * dim, heights[hill]);
    }
    const size_t probe_count = probes.size() / dim;
    for (size_t p = 0; p < probe_count; ++p) {
        double grad_d[3] = {0, 0, 0};
        double grad_s[3] = {0, 0, 0};
        const double* point = probes.data() + p * dim;
        const double ed = dense.evaluate(point, grad_d);
        const double es = sparse.evaluate(point, grad_s);
        expect(ed == es, std::string(label) + " energy dense==sparse");
        for (size_t axis = 0; axis < dim; ++axis) {
            expect(grad_d[axis] == grad_s[axis],
                   std::string(label) + " grad dense==sparse");
        }
    }
}

void check_finite_difference(BiasGrid& grid, const double* point, size_t dim,
                             const std::vector<bool>& periodic,
                             const char* label) {
    double analytic[3] = {0, 0, 0};
    grid.evaluate(point, analytic);
    const double step = 1e-5;
    for (size_t axis = 0; axis < dim; ++axis) {
        double plus[3], minus[3];
        for (size_t d = 0; d < dim; ++d) {
            plus[d] = point[d];
            minus[d] = point[d];
        }
        plus[axis] += step;
        minus[axis] -= step;
        const double ep = grid.evaluate(plus, nullptr);
        const double em = grid.evaluate(minus, nullptr);
        const double numeric = (ep - em) / (2.0 * step);
        // 夹紧区解析梯度应为 0；中心区与中心差分对齐
        if (std::abs(analytic[axis]) < 1e-14 && !periodic[axis]) {
            expect(std::abs(numeric) < 1e-6 ||
                       nearly_equal(analytic[axis], numeric, 1e-4, 1e-3),
                   std::string(label) + " clamped/near-zero grad");
        } else {
            expect(nearly_equal(analytic[axis], numeric, 1e-4, 5e-3),
                   std::string(label) + " FD axis " + std::to_string(axis) +
                       " ana=" + std::to_string(analytic[axis]) +
                       " num=" + std::to_string(numeric));
        }
    }
}

void check_rebuild_bitwise(const std::vector<GridAxis>& axes,
                           const std::vector<double>& centers,
                           const std::vector<double>& heights,
                           const char* label) {
    const size_t dim = axes.size();
    BiasGrid by_add(axes, false);
    for (size_t hill = 0; hill < heights.size(); ++hill) {
        by_add.add(centers.data() + hill * dim, heights[hill]);
    }
    BiasGrid serial(axes, false);
    serial.rebuild(centers.data(), heights.data(), heights.size(), 1);
    BiasGrid parallel(axes, false);
    parallel.rebuild(centers.data(), heights.data(), heights.size(), 4);

    BiasGrid sparse_serial(axes, true);
    sparse_serial.rebuild(centers.data(), heights.data(), heights.size(), 1);
    BiasGrid sparse_parallel(axes, true);
    sparse_parallel.rebuild(centers.data(), heights.data(), heights.size(), 4);

    // 在全部箱中心比较：CR@center 返回格点值，可检测位级一致
    std::vector<int> indices(dim, 0);
    auto advance = [&]() -> bool {
        for (size_t axis = dim; axis-- > 0;) {
            ++indices[axis];
            if (indices[axis] < axes[axis].bins) {
                return true;
            }
            indices[axis] = 0;
        }
        return false;
    };
    do {
        const std::vector<double> point = bin_center_point(axes, indices);
        const double a = by_add.evaluate(point.data());
        const double s = serial.evaluate(point.data());
        const double p = parallel.evaluate(point.data());
        const double ss = sparse_serial.evaluate(point.data());
        const double sp = sparse_parallel.evaluate(point.data());
        expect(a == s, std::string(label) + " add==serial rebuild");
        expect(s == p, std::string(label) + " serial==parallel rebuild");
        expect(ss == sp, std::string(label) + " sparse serial==parallel");
        expect(a == ss, std::string(label) + " dense==sparse rebuild");
    } while (advance());
}

void check_exact_list_tolerance(const std::vector<GridAxis>& axes,
                                const std::vector<double>& centers,
                                const std::vector<double>& heights,
                                const std::vector<double>& probes,
                                double abs_tol, const char* label) {
    BiasGrid grid(axes, false);
    const size_t dim = axes.size();
    grid.rebuild(centers.data(), heights.data(), heights.size(), 1);
    const size_t probe_count = probes.size() / dim;
    for (size_t p = 0; p < probe_count; ++p) {
        const double* point = probes.data() + p * dim;
        const double grid_e = grid.evaluate(point);
        const double exact =
            exact_hill_energy(axes, centers, heights, point);
        expect(nearly_equal(grid_e, exact, abs_tol, 0.05),
               std::string(label) + " vs exact list @" + std::to_string(p) +
                   " grid=" + std::to_string(grid_e) +
                   " exact=" + std::to_string(exact));
    }
}

void test_invalid_settings() {
    try {
        BiasGrid bad({GridAxis{0, 1, 0.1, 3}}, false);
        expect(false, "bins<4 should throw");
    } catch (const std::invalid_argument&) {
        expect(true, "bins<4 throws");
    }
    try {
        BiasGrid bad({GridAxis{0, 1, 0.1, 8, 0.5}}, false);
        expect(false, "period mismatch should throw");
    } catch (const std::invalid_argument&) {
        expect(true, "period mismatch throws");
    }
    try {
        BiasGrid bad(std::vector<GridAxis>{}, false);
        expect(false, "empty axes should throw");
    } catch (const std::invalid_argument&) {
        expect(true, "empty axes throws");
    }
    try {
        BiasGrid bad({GridAxis{0, 1, -0.1, 8}}, false);
        expect(false, "negative sigma should throw");
    } catch (const std::invalid_argument&) {
        expect(true, "negative sigma throws");
    }
    BiasGrid ok({GridAxis{0, 1, 0.1, 8}}, false);
    double nan_center = std::nan("");
    try {
        ok.add(&nan_center, 1.0);
        expect(false, "NaN center should throw");
    } catch (const std::invalid_argument&) {
        expect(true, "NaN center throws");
    }
}

void run_dimension_suite() {
    // 1D 非周期，不等观感 bins=11
    {
        std::vector<GridAxis> axes{{0.0, 2.0, 0.15, 11, 0.0}};
        std::vector<double> centers{0.7, 1.2, 0.4};
        std::vector<double> heights{1.0, 0.5, 0.8};
        std::vector<double> probes{0.7, 1.0, 0.05, 1.95};
        check_dense_sparse_agree(axes, centers, heights, probes, "1D");
        BiasGrid grid(axes, false);
        for (size_t h = 0; h < heights.size(); ++h) {
            grid.add(&centers[h], heights[h]);
        }
        const double interior = 1.0;
        check_finite_difference(grid, &interior, 1, {false}, "1D-interior");
        const double lo = 0.02;
        check_finite_difference(grid, &lo, 1, {false}, "1D-clamped-lo");
        const double hi = 1.98;
        check_finite_difference(grid, &hi, 1, {false}, "1D-clamped-hi");
        check_rebuild_bitwise(axes, centers, heights, "1D");
        check_exact_list_tolerance(axes, centers, heights, {1.0, 0.7}, 0.15,
                                   "1D-exact");
    }

    // 1D 周期
    {
        std::vector<GridAxis> axes{{0.0, 2.0 * kPi, 0.25, 16, 2.0 * kPi}};
        std::vector<double> centers{0.3, 5.5, 2.0};
        std::vector<double> heights{1.2, 0.7, 0.9};
        std::vector<double> probes{0.3, 6.0, 0.01, 2.0 * kPi - 0.01};
        check_dense_sparse_agree(axes, centers, heights, probes, "1D-periodic");
        BiasGrid grid(axes, false);
        grid.rebuild(centers.data(), heights.data(), heights.size(), 1);
        const double wrap_pt = 0.05;
        check_finite_difference(grid, &wrap_pt, 1, {true}, "1D-periodic-FD");
        check_rebuild_bitwise(axes, centers, heights, "1D-periodic");
        check_exact_list_tolerance(axes, centers, heights, {1.0, 5.5}, 0.2,
                                   "1D-periodic-exact");
    }

    // 2D 不等 bins
    {
        std::vector<GridAxis> axes{
            {0.0, 1.0, 0.12, 9, 0.0},
            {-1.0, 1.0, 0.18, 13, 0.0},
        };
        std::vector<double> centers{
            0.4, 0.1, 0.6, -0.3, 0.2, 0.5,
        };
        std::vector<double> heights{1.0, 0.6, 0.8};
        std::vector<double> probes{
            0.4, 0.1, 0.5, 0.0, 0.05, -0.9, 0.95, 0.9,
        };
        check_dense_sparse_agree(axes, centers, heights, probes, "2D");
        BiasGrid grid(axes, false);
        grid.rebuild(centers.data(), heights.data(), heights.size(), 2);
        const double interior[2] = {0.45, 0.05};
        check_finite_difference(grid, interior, 2, {false, false}, "2D-interior");
        const double edge[2] = {0.02, 0.0};
        check_finite_difference(grid, edge, 2, {false, false}, "2D-clamped");
        check_rebuild_bitwise(axes, centers, heights, "2D");
        check_exact_list_tolerance(axes, centers, heights, {0.4, 0.1, 0.5, 0.0},
                                   0.2, "2D-exact");
    }

    // 2D 一轴周期
    {
        std::vector<GridAxis> axes{
            {0.0, 2.0, 0.15, 10, 2.0},
            {0.0, 1.0, 0.12, 8, 0.0},
        };
        std::vector<double> centers{0.2, 0.4, 1.8, 0.6};
        std::vector<double> heights{1.0, 0.7};
        BiasGrid grid(axes, false);
        grid.rebuild(centers.data(), heights.data(), heights.size(), 1);
        const double pt[2] = {0.05, 0.5};
        check_finite_difference(grid, pt, 2, {true, false}, "2D-mixed-periodic");
        check_rebuild_bitwise(axes, centers, heights, "2D-mixed");
    }

    // 3D 不等 bins
    {
        std::vector<GridAxis> axes{
            {0.0, 1.0, 0.14, 7, 0.0},
            {0.0, 2.0, 0.16, 9, 0.0},
            {-0.5, 0.5, 0.11, 8, 0.0},
        };
        std::vector<double> centers{
            0.4, 1.0, 0.0, 0.3, 0.8, -0.1, 0.55, 1.2, 0.15,
        };
        std::vector<double> heights{1.0, 0.5, 0.75};
        std::vector<double> probes{
            0.4, 1.0, 0.0, 0.5, 1.0, 0.0, 0.05, 0.1, -0.45,
        };
        check_dense_sparse_agree(axes, centers, heights, probes, "3D");
        BiasGrid grid(axes, false);
        grid.rebuild(centers.data(), heights.data(), heights.size(), 3);
        const double interior[3] = {0.42, 1.05, 0.02};
        check_finite_difference(grid, interior, 3, {false, false, false},
                                "3D-interior");
        check_rebuild_bitwise(axes, centers, heights, "3D");
        check_exact_list_tolerance(axes, centers, heights,
                                   {0.4, 1.0, 0.0, 0.5, 1.0, 0.0}, 0.25,
                                   "3D-exact");
    }
}

}  // namespace

int main() {
    test_invalid_settings();
    run_dimension_suite();
    if (g_failures == 0) {
        std::cout << "bias_grid: all checks passed\n";
        return 0;
    }
    std::cerr << "bias_grid: " << g_failures << " failure(s)\n";
    return 1;
}
