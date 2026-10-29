#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace MetaD_zqc {

// 一维 CV 轴：区间 [lower, upper]、高斯标准差 sigma、分箱数 bins。
// period>0 表示周期轴，且必须等于 (upper-lower)；能量核改用稳定半角余弦形式。
struct GridAxis {
    double lower = 0.0;
    double upper = 0.0;
    double sigma = 0.0;
    int bins = 0;
    double period = 0.0;
};

// 运行时维数 D=1..3 的低维偏置网格缓存（纯宿主 C++17）。
// 存储：行主序展平下标；dense 用连续 vector，sparse 用 unordered_map。
// 插值：各轴 bin 中心上的 Catmull-Rom 张量积；非周期轴夹紧时该轴解析梯度强制为 0。
class BiasGrid {
public:
    explicit BiasGrid(std::vector<GridAxis> axes, bool sparse = false);

    size_t dimension() const { return dimension_; }

    // 将一座高斯山（高 height）累加到网格；非周期轴按每轴 |Δ|≤6σ 截断，周期轴扫全箱。
    void add(const double* center, double height);

    // Catmull-Rom 求能量；若 gradient≠nullptr 则写解析梯度（物理坐标单位）。
    double evaluate(const double* point, double* gradient = nullptr) const;

    void clear();

    // 用输入山列表确定性重建。workers=1 走串行；>1 时按输出格点分块并行，
    // 每格内仍按 hills 输入顺序累加，与反复 add 位级一致。
    void rebuild(const double* centers, const double* heights, size_t count,
                 unsigned workers = 1);

    // dense：总格点数；sparse：当前 map 条目数。
    size_t stored_points() const;

private:
    static constexpr size_t kMaxDimension = 3;
    static constexpr double kSigmaTruncate = 6.0;

    size_t dimension_ = 0;
    bool sparse_mode_ = false;

    // 轴参数（长度 = D）
    std::vector<GridAxis> axes_;
    std::vector<double> spacings_;   // (upper-lower)/bins
    std::vector<std::int64_t> strides_; // 行主序：末轴 stride=1
    std::vector<bool> periodic_flags_;

    std::uint64_t total_cells_ = 0;
    std::vector<double> dense_values_;
    std::unordered_map<std::uint64_t, double> sparse_values_;

    void validate_axes(const std::vector<GridAxis>& axes) const;
    void init_layout();

    double bin_center(size_t axis, int bin_index) const;
    std::uint64_t flat_index(const int* multi_index) const;
    void multi_index_from_flat(std::uint64_t flat, int* multi_index) const;

    // 单轴核：非周期 Gaussian；周期为 q=2(1-cos)/(2πσ/L)² 的稳定 sin(θ/2) 形式。
    // 若该轴非周期且 |Δ|>6σ，返回 false（整格贡献丢弃）。
    bool axis_kernel(size_t axis, double center_coord, double cell_coord,
                     double& kernel_value) const;

    double hill_on_cell(const double* center, double height,
                        const int* multi_index) const;

    void accumulate_hill_on_cell(const double* center, double height,
                                 std::uint64_t flat);

    void deposit_hill_serial(const double* center, double height);
    bool deposition_bounds(const double* center, int* lower, int* upper) const;

    double sample_grid(const int* multi_index) const;
    void set_grid(std::uint64_t flat, double value);
    void add_to_grid(std::uint64_t flat, double delta);

    // 共享张量积 Catmull-Rom：能量与梯度同一套节点采样。
    double interpolate_tensor(const double* point, double* gradient) const;
    double tensor_recurse(size_t current_axis, int target_axis, int* base_index,
                          const double* fraction) const;

    int wrap_bin(int index, int bins) const;

    void rebuild_serial(const double* centers, const double* heights, size_t count);
    void rebuild_parallel(const double* centers, const double* heights, size_t count,
                          unsigned workers);
};

}  // namespace MetaD_zqc
