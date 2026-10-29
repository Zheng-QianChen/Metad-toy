#include "zqc_bias_grid.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace MetaD_zqc {
namespace {

constexpr double kPi = 3.14159265358979323846;

inline bool is_finite_number(double value) {
    return std::isfinite(value) != 0;
}

// Catmull-Rom 三次样条（分数坐标 x∈[0,1]）及其对 x 的导数。
inline double catmull_rom_value(double p0, double p1, double p2, double p3, double x) {
    return p1 + 0.5 * x * (
        (p2 - p0) +
        x * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) +
        x * x * (-p0 + 3.0 * p1 - 3.0 * p2 + p3));
}

inline double catmull_rom_deriv(double p0, double p1, double p2, double p3, double x) {
    return 0.5 * (
        (p2 - p0) +
        2.0 * x * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) +
        3.0 * x * x * (-p0 + 3.0 * p1 - 3.0 * p2 + p3));
}

}  // namespace

BiasGrid::BiasGrid(std::vector<GridAxis> axes, bool sparse)
    : sparse_mode_(sparse), axes_(std::move(axes)) {
    validate_axes(axes_);
    dimension_ = axes_.size();
    init_layout();
    clear();
}

void BiasGrid::validate_axes(const std::vector<GridAxis>& axes) const {
    if (axes.empty() || axes.size() > kMaxDimension) {
        throw std::invalid_argument("BiasGrid: dimension must be 1..3");
    }
    std::uint64_t product = 1;
    for (size_t axis = 0; axis < axes.size(); ++axis) {
        const GridAxis& a = axes[axis];
        if (a.bins < 4) {
            throw std::invalid_argument("BiasGrid: bins must be >= 4");
        }
        if (!is_finite_number(a.lower) || !is_finite_number(a.upper) ||
            !is_finite_number(a.sigma) || !is_finite_number(a.period)) {
            throw std::invalid_argument("BiasGrid: axis bounds/sigma/period must be finite");
        }
        if (!(a.upper > a.lower)) {
            throw std::invalid_argument("BiasGrid: upper must be greater than lower");
        }
        if (!(a.sigma > 0.0)) {
            throw std::invalid_argument("BiasGrid: sigma must be positive");
        }
        const double length = a.upper - a.lower;
        if (!std::isfinite(length) || !std::isfinite(length/a.bins) || length/a.bins<=0)
            throw std::invalid_argument("BiasGrid: invalid axis spacing");
        if (a.period > 0.0) {
            if (std::abs(a.period - length) >
                1e-12 * std::max(1.0, std::abs(length))) {
                throw std::invalid_argument(
                    "BiasGrid: period>0 must match upper-lower");
            }
        } else if (a.period < 0.0) {
            throw std::invalid_argument("BiasGrid: period must be >= 0");
        }
        if (product > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) /
                          static_cast<std::uint64_t>(a.bins)) {
            throw std::overflow_error("BiasGrid: bins product overflows");
        }
        product *= static_cast<std::uint64_t>(a.bins);
    }
}

void BiasGrid::init_layout() {
    spacings_.resize(dimension_);
    strides_.resize(dimension_);
    periodic_flags_.resize(dimension_);
    for (size_t axis = 0; axis < dimension_; ++axis) {
        spacings_[axis] =
            (axes_[axis].upper - axes_[axis].lower) /
            static_cast<double>(axes_[axis].bins);
        periodic_flags_[axis] = axes_[axis].period > 0.0;
    }
    // 行主序：末维变化最快，stride[D-1]=1
    strides_[dimension_ - 1] = 1;
    for (size_t axis = dimension_ - 1; axis > 0; --axis) {
        strides_[axis - 1] =
            strides_[axis] * static_cast<std::int64_t>(axes_[axis].bins);
    }
    total_cells_ = 1;
    for (size_t axis = 0; axis < dimension_; ++axis) {
        total_cells_ *= static_cast<std::uint64_t>(axes_[axis].bins);
    }
    if (!sparse_mode_) {
        dense_values_.assign(static_cast<size_t>(total_cells_), 0.0);
    }
}

double BiasGrid::bin_center(size_t axis, int bin_index) const {
    // 箱中心：lower + (i+1/2)*Δ；不等 bins 时各轴独立 spacing。
    return axes_[axis].lower +
           (static_cast<double>(bin_index) + 0.5) * spacings_[axis];
}

std::uint64_t BiasGrid::flat_index(const int* multi_index) const {
    std::uint64_t flat = 0;
    for (size_t axis = 0; axis < dimension_; ++axis) {
        flat += static_cast<std::uint64_t>(multi_index[axis]) *
                static_cast<std::uint64_t>(strides_[axis]);
    }
    return flat;
}

void BiasGrid::multi_index_from_flat(std::uint64_t flat, int* multi_index) const {
    for (size_t axis = 0; axis < dimension_; ++axis) {
        const std::uint64_t stride =
            static_cast<std::uint64_t>(strides_[axis]);
        multi_index[axis] = static_cast<int>(flat / stride);
        flat %= stride;
    }
}

int BiasGrid::wrap_bin(int index, int bins) const {
    int wrapped = index % bins;
    if (wrapped < 0) {
        wrapped += bins;
    }
    return wrapped;
}

bool BiasGrid::axis_kernel(size_t axis, double center_coord, double cell_coord,
                           double& kernel_value) const {
    const GridAxis& a = axes_[axis];
    if (periodic_flags_[axis]) {
        // 周期余弦核：q = 2(1-cos(2πδ/L))/(2πσ/L)²
        // 稳定半角：1-cos θ = 2 sin²(θ/2)，θ=2πδ/L → q = [sin(πδ/L)/(πσ/L)]²
        const double length = a.period;
        const double delta = center_coord - cell_coord;
        if (!std::isfinite(delta)) throw std::overflow_error("BiasGrid: coordinate difference overflow");
        const double argument=kPi*(std::remainder(delta,length)/length);
        const double ratio=std::sin(argument)*(length/kPi)/a.sigma;
        const double q=ratio*ratio;
        kernel_value = std::exp(-0.5 * q);
        return true;
    }
    const double delta = center_coord - cell_coord;
    // 与 rebuild 完全相同的 6σ 谓词
    if (std::abs(delta) > kSigmaTruncate * a.sigma) {
        return false;
    }
    const double scaled = delta / a.sigma;
    kernel_value = std::exp(-0.5 * scaled * scaled);
    return true;
}

double BiasGrid::hill_on_cell(const double* center, double height,
                              const int* multi_index) const {
    double product = height;
    for (size_t axis = 0; axis < dimension_; ++axis) {
        double kernel = 0.0;
        if (!axis_kernel(axis, center[axis], bin_center(axis, multi_index[axis]),
                         kernel)) {
            return 0.0;
        }
        product *= kernel;
    }
    return product;
}

void BiasGrid::add_to_grid(std::uint64_t flat, double delta) {
    if (delta == 0.0) {
        return;
    }
    if (sparse_mode_) {
        sparse_values_[flat] += delta;
    } else {
        dense_values_[static_cast<size_t>(flat)] += delta;
    }
}

void BiasGrid::set_grid(std::uint64_t flat, double value) {
    if (sparse_mode_) {
        if (value == 0.0) {
            sparse_values_.erase(flat);
        } else {
            sparse_values_[flat] = value;
        }
    } else {
        dense_values_[static_cast<size_t>(flat)] = value;
    }
}

double BiasGrid::sample_grid(const int* multi_index) const {
    int wrapped[kMaxDimension];
    for (size_t axis = 0; axis < dimension_; ++axis) {
        int index = multi_index[axis];
        if (periodic_flags_[axis]) {
            index = wrap_bin(index, axes_[axis].bins);
        } else if (index < 0 || index >= axes_[axis].bins) {
            // 非周期越界：夹紧到边界箱（样条邻域）
            index = std::max(0, std::min(axes_[axis].bins - 1, index));
        }
        wrapped[axis] = index;
    }
    const std::uint64_t flat = flat_index(wrapped);
    if (sparse_mode_) {
        const auto found = sparse_values_.find(flat);
        return found == sparse_values_.end() ? 0.0 : found->second;
    }
    return dense_values_[static_cast<size_t>(flat)];
}

void BiasGrid::accumulate_hill_on_cell(const double* center, double height,
                                       std::uint64_t flat) {
    int multi_index[kMaxDimension];
    multi_index_from_flat(flat, multi_index);
    const double contribution = hill_on_cell(center, height, multi_index);
    add_to_grid(flat, contribution);
}

bool BiasGrid::deposition_bounds(const double* center,int* lower_bin,int* upper_bin) const {
    for (size_t axis=0;axis<dimension_;++axis) {
        lower_bin[axis]=0;upper_bin[axis]=axes_[axis].bins-1;
        if (periodic_flags_[axis]) continue;
        const double half_width=kSigmaTruncate*axes_[axis].sigma;
        const double first=(center[axis]-half_width-axes_[axis].lower)/spacings_[axis]-0.5;
        const double last=(center[axis]+half_width-axes_[axis].lower)/spacings_[axis]-0.5;
        if (first>upper_bin[axis]+1.0 || last<-1.0) return false;
        // 先限制到有效区间再转换为整数，并多取一格供精确的 6σ 谓词筛选。
        lower_bin[axis]=static_cast<int>(std::max(0.0,std::min(double(upper_bin[axis]),std::floor(first))));
        upper_bin[axis]=static_cast<int>(std::max(0.0,std::min(double(upper_bin[axis]),std::ceil(last))));
    }
    return true;
}
void BiasGrid::deposit_hill_serial(const double* center, double height) {
    int lower_bin[kMaxDimension],upper_bin[kMaxDimension];
    if (!deposition_bounds(center,lower_bin,upper_bin)) return;

    int multi_index[kMaxDimension];
    if (dimension_ == 1) {
        for (int i0 = lower_bin[0]; i0 <= upper_bin[0]; ++i0) {
            multi_index[0] = i0;
            add_to_grid(flat_index(multi_index),
                        hill_on_cell(center, height, multi_index));
        }
        return;
    }
    if (dimension_ == 2) {
        for (int i0 = lower_bin[0]; i0 <= upper_bin[0]; ++i0) {
            multi_index[0] = i0;
            for (int i1 = lower_bin[1]; i1 <= upper_bin[1]; ++i1) {
                multi_index[1] = i1;
                add_to_grid(flat_index(multi_index),
                            hill_on_cell(center, height, multi_index));
            }
        }
        return;
    }
    for (int i0 = lower_bin[0]; i0 <= upper_bin[0]; ++i0) {
        multi_index[0] = i0;
        for (int i1 = lower_bin[1]; i1 <= upper_bin[1]; ++i1) {
            multi_index[1] = i1;
            for (int i2 = lower_bin[2]; i2 <= upper_bin[2]; ++i2) {
                multi_index[2] = i2;
                add_to_grid(flat_index(multi_index),
                            hill_on_cell(center, height, multi_index));
            }
        }
    }
}

void BiasGrid::add(const double* center, double height) {
    if (center == nullptr) {
        throw std::invalid_argument("BiasGrid::add: null center");
    }
    if (!is_finite_number(height)) {
        throw std::invalid_argument("BiasGrid::add: height must be finite");
    }
    for (size_t axis = 0; axis < dimension_; ++axis) {
        if (!is_finite_number(center[axis])) {
            throw std::invalid_argument("BiasGrid::add: center must be finite");
        }
    }
    deposit_hill_serial(center, height);
}

void BiasGrid::clear() {
    if (sparse_mode_) {
        sparse_values_.clear();
    } else {
        std::fill(dense_values_.begin(), dense_values_.end(), 0.0);
    }
}

size_t BiasGrid::stored_points() const {
    if (sparse_mode_) {
        return sparse_values_.size();
    }
    return dense_values_.size();
}

double BiasGrid::tensor_recurse(size_t current_axis, int target_axis,
                                int* base_index,
                                const double* fraction) const {
    if (current_axis == dimension_) {
        return sample_grid(base_index);
    }
    double samples[4];
    const int origin = base_index[current_axis];
    for (int offset = -1; offset <= 2; ++offset) {
        base_index[current_axis] = origin + offset;
        samples[offset + 1] =
            tensor_recurse(current_axis + 1, target_axis, base_index, fraction);
    }
    base_index[current_axis] = origin;
    const double x = fraction[current_axis];
    if (static_cast<int>(current_axis) == target_axis) {
        return catmull_rom_deriv(samples[0], samples[1], samples[2], samples[3],
                                 x);
    }
    return catmull_rom_value(samples[0], samples[1], samples[2], samples[3], x);
}

double BiasGrid::interpolate_tensor(const double* point,
                                    double* gradient) const {
    int base_index[kMaxDimension] = {};
    double fraction[kMaxDimension] = {};
    bool axis_clamped[kMaxDimension] = {};

    for (size_t axis = 0; axis < dimension_; ++axis) {
        axis_clamped[axis] = false;
        double coordinate = point[axis];
        if (periodic_flags_[axis]) {
            const double length = axes_[axis].period;
            coordinate = axes_[axis].lower +
                         std::fmod(std::fmod(coordinate - axes_[axis].lower,
                                             length) +
                                       length,
                                   length);
        }
        // bin 中心定位：u = (x-lower)/Δ - 1/2
        const double continuous_index =
            (coordinate - axes_[axis].lower) / spacings_[axis] - 0.5;
        int index=1;
        double frac=0;
        if (!std::isfinite(continuous_index)) {
            if (periodic_flags_[axis]) throw std::overflow_error("BiasGrid: periodic coordinate overflow");
            index=coordinate<axes_[axis].lower ? 1 : axes_[axis].bins-3;
            frac=coordinate<axes_[axis].lower ? 0 : 1;
            axis_clamped[axis]=true;
        } else if (!periodic_flags_[axis] && continuous_index<1) {
            axis_clamped[axis]=true;
        } else if (!periodic_flags_[axis] && continuous_index>=axes_[axis].bins-2) {
            index=axes_[axis].bins-3;frac=1;axis_clamped[axis]=true;
        } else {
            index=static_cast<int>(std::floor(continuous_index));
            frac=continuous_index-index;
        }

        if (!periodic_flags_[axis]) {
            // 样条邻域 [i-1..i+2] 需 i∈[1,bins-3]；夹紧后该轴 dV/dx=0
            if (index < 1) {
                index = 1;
                frac = 0.0;
                axis_clamped[axis] = true;
            }
            if (index > axes_[axis].bins - 3) {
                index = axes_[axis].bins - 3;
                frac = 1.0;
                axis_clamped[axis] = true;
            }
        }
        base_index[axis] = index;
        fraction[axis] = frac;
    }

    const double energy = tensor_recurse(0, -1, base_index, fraction);
    if (gradient != nullptr) {
        for (size_t axis = 0; axis < dimension_; ++axis) {
            if (!periodic_flags_[axis] && axis_clamped[axis]) {
                gradient[axis] = 0.0;
                continue;
            }
            // dV/dx = (dV/dfrac) / spacing
            gradient[axis] = tensor_recurse(0, static_cast<int>(axis),
                                            base_index, fraction) /
                             spacings_[axis];
        }
    }
    return energy;
}

double BiasGrid::evaluate(const double* point, double* gradient) const {
    if (point == nullptr) {
        throw std::invalid_argument("BiasGrid::evaluate: null point");
    }
    for (size_t axis = 0; axis < dimension_; ++axis) {
        if (!is_finite_number(point[axis])) {
            throw std::invalid_argument(
                "BiasGrid::evaluate: point must be finite");
        }
    }
    const double energy=interpolate_tensor(point,gradient);
    if (!std::isfinite(energy)) throw std::overflow_error("BiasGrid: nonfinite energy");
    if (gradient) for(size_t axis=0;axis<dimension_;++axis)
        if(!std::isfinite(gradient[axis])) throw std::overflow_error("BiasGrid: nonfinite gradient");
    return energy;
}

void BiasGrid::rebuild_serial(const double* centers, const double* heights,
                              size_t count) {
    clear();
    for (size_t hill = 0; hill < count; ++hill) {
        deposit_hill_serial(centers + hill * dimension_, heights[hill]);
    }
}

void BiasGrid::rebuild_parallel(const double* centers, const double* heights,
                                size_t count, unsigned workers) {
    clear();
    const unsigned available=std::max(1u,std::thread::hardware_concurrency());
    const unsigned effective=std::min({workers,available,16u,static_cast<unsigned>(axes_[0].bins)});
    if (effective<2 || count<2 || total_cells_<64) {rebuild_serial(centers,heights,count);return;}
    std::vector<std::exception_ptr> errors(effective);
    std::vector<std::unordered_map<std::uint64_t,double>> local_maps(sparse_mode_ ? effective : 0);
    // 按第一轴切分互不重叠的 slab。每个 slab 仍按时间顺序遍历 hills，
    // 且只访问核支持范围；复杂度随被更新的格点数增长。
    auto fill_slab=[&](unsigned worker) {
        try {
            const int slab_begin=axes_[0].bins*static_cast<std::int64_t>(worker)/effective;
            const int slab_end=axes_[0].bins*static_cast<std::int64_t>(worker+1)/effective-1;
            int lower[kMaxDimension],upper[kMaxDimension],index[kMaxDimension];
            for(size_t hill=0;hill<count;++hill) {
                const double* center=centers+hill*dimension_;
                if(!deposition_bounds(center,lower,upper)) continue;
                lower[0]=std::max(lower[0],slab_begin);upper[0]=std::min(upper[0],slab_end);
                if(lower[0]>upper[0]) continue;
                std::copy_n(lower,dimension_,index);
                while(true) {
                    const auto flat=flat_index(index);
                    const double value=hill_on_cell(center,heights[hill],index);
                    if(value!=0) {
                        if(sparse_mode_) local_maps[worker][flat]+=value;
                        else dense_values_[flat]+=value;
                    }
                    size_t axis=dimension_;
                    while(axis>0) {
                        --axis;
                        if(index[axis]<upper[axis]) {++index[axis];break;}
                        index[axis]=lower[axis];
                    }
                    if(axis==0 && index[0]==lower[0]) break;
                }
            }
        } catch(...) {errors[worker]=std::current_exception();}
    };
    std::vector<std::thread> threads;
    try {for(unsigned worker=1;worker<effective;++worker) threads.emplace_back(fill_slab,worker);}
    catch(...) {for(auto& thread:threads) thread.join();throw;}
    fill_slab(0);
    for(auto& thread:threads) thread.join();
    for(const auto& error:errors) if(error) std::rethrow_exception(error);
    if(sparse_mode_) for(auto& values:local_maps) sparse_values_.merge(values);
}

void BiasGrid::rebuild(const double* centers, const double* heights,
                       size_t count, unsigned workers) {
    if (count > 0 && (centers == nullptr || heights == nullptr)) {
        throw std::invalid_argument("BiasGrid::rebuild: null centers/heights");
    }
    for (size_t hill = 0; hill < count; ++hill) {
        if (!is_finite_number(heights[hill])) {
            throw std::invalid_argument(
                "BiasGrid::rebuild: heights must be finite");
        }
        for (size_t axis = 0; axis < dimension_; ++axis) {
            if (!is_finite_number(centers[hill * dimension_ + axis])) {
                throw std::invalid_argument(
                    "BiasGrid::rebuild: centers must be finite");
            }
        }
    }
    if (workers <= 1) {
        rebuild_serial(centers, heights, count);
        return;
    }
    rebuild_parallel(centers, heights, count, workers);
}

}  // namespace MetaD_zqc
