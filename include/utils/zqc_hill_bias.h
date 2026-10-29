#pragma once

#include <cstdint>
#include <iosfwd>
#include <vector>

namespace MetaD_zqc {

// 每个轴的宽度采用该 CV 的单位；period=0 为非周期轴。
struct BiasAxis {
    double sigma = 0.05;
    double period = 0.0;
};
enum class HillBiasMode { Hills, Parallel };
// PLUMED 2.10 PBMetaD 源码：最小镜像距离、半平方距离 < 6.25 的高斯核。
// SmoothGaussian 使用光滑周期核，便于不需要 PLUMED 核语义的独立计算。
enum class PBKernel { SmoothGaussian, Plumed210 };
struct HillBiasSettings {
    HillBiasMode mode = HillBiasMode::Hills;
    std::vector<BiasAxis> axes;
    double initial_height = 0.1;
    double thermal_energy = 1.0; // k_B*T，采用宿主的能量单位。
    double bias_factor = 10.0;
    bool well_tempered = false;
    PBKernel pb_kernel = PBKernel::SmoothGaussian;
};

// 任意维的 Host 偏势历史。完整 hill 的 centers 按 [沉积次数][维度] 连续存储，
// heights 每次一个；PB 每次保存 D 个一维高度，共用同一组中心和步号。
// 所有核共享固定的轴参数。求值 O(H*D)，存储 O(H*D)。
class HillBias {
public:
    explicit HillBias(HillBiasSettings settings);
    const HillBiasSettings& settings() const { return settings_; }
    std::size_t dimension() const { return settings_.axes.size(); }
    std::size_t depositions() const { return steps_.size(); }
    std::int64_t last_deposition_step() const { return steps_.empty() ? -1 : steps_.back(); }

    // 返回 V，同时写入 D 个 dV/ds。gradient=nullptr 可只求能量。
    double evaluate(const double* coordinates, double* gradient = nullptr) const;
    // 调用者决定 PACE；同一步重复调用为 no-op，倒退的沉积步会报错。
    bool deposit(std::int64_t step, const double* coordinates, const double* old_energy = nullptr);
    // 装载已知高度或测试固定偏势；PB 高度长度 D，普通 hill 长度 1。
    void append(std::int64_t step, const double* coordinates, const double* heights, bool allow_same_step = false);
    const std::vector<double>& centers() const { return centers_; }
    const std::vector<double>& heights() const { return heights_; }
    const std::vector<std::int64_t>& steps() const { return steps_; }
    // 分块只改变遍历批次；块内和块间都保持历史顺序。
    void set_block_size(std::size_t size);
    std::size_t block_size() const { return block_size_; }

    // 版本化文本格式保存实际加入的高度。读取先验证完整文件，再替换历史。
    void write_header(std::ostream&) const;
    void write_last_deposition(std::ostream&) const;
    void load(std::istream&, unsigned workers = 1);

private:
    HillBiasSettings settings_;
    std::size_t block_size_ = 256;
    std::vector<double> centers_, heights_;
    std::vector<std::int64_t> steps_;
    std::size_t heights_per_deposition() const;
    void validate_coordinates(const double*) const;
    void components(const double*, std::vector<double>& values,
                    std::vector<double>& derivatives) const;
    double weights(const std::vector<double>& values, std::vector<double>& normalized) const;
};

} // namespace MetaD_zqc
