#include "zqc_hill_bias.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <istream>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <thread>
#include <exception>

namespace MetaD_zqc {
namespace {
constexpr double two_pi = 6.283185307179586476925286766559;
void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
struct KernelAxis { double quadratic, derivative; };
KernelAxis kernel_axis(double value, double center, const BiasAxis& axis) {
    const double delta = value-center;
    require(std::isfinite(delta), "Bias coordinate difference overflow");
    if (axis.period == 0) {
        const double scaled = delta/axis.sigma;
        return {scaled*scaled, scaled/axis.sigma};
    }
    // 用 sin(angle/2)^2 写周期核，避免小位移时 1-cos(angle) 相减损失精度。
    const double angle = two_pi*(std::remainder(delta,axis.period)/axis.period);
    const double length_scale = axis.period/two_pi;
    const double scaled = 2*std::sin(angle/2)*length_scale/axis.sigma;
    return {scaled*scaled, std::sin(angle)*(length_scale/axis.sigma)/axis.sigma};
}
void finite_result(double value) {
    if (!std::isfinite(value)) throw std::overflow_error("Nonfinite bias energy or derivative");
}
std::string read_line(std::istream& input) {
    std::string line;
    require(bool(std::getline(input,line)) && !input.eof(), "Incomplete bias history header");
    return line;
}
void no_extra(std::istringstream& row) {
    std::string extra;
    require(!(row>>extra), "Unexpected field in bias history");
}
}

HillBias::HillBias(HillBiasSettings settings) : settings_(std::move(settings)) {
    require(settings_.mode==HillBiasMode::Hills || settings_.mode==HillBiasMode::Parallel,
            "Unknown hill bias mode");
    require(!settings_.axes.empty(), "Bias requires at least one axis");
    for (const auto& axis : settings_.axes) {
        require(std::isfinite(axis.sigma) && axis.sigma>0, "Bias sigma must be finite and positive");
        require(std::isfinite(axis.period) && axis.period>=0, "Bias period must be finite and nonnegative");
    }
    require(std::isfinite(settings_.initial_height) && settings_.initial_height>0,
            "Bias height must be finite and positive");
    require(std::isfinite(settings_.thermal_energy) && settings_.thermal_energy>0,
            "Bias kBT must be finite and positive");
    require(std::isfinite(settings_.bias_factor) &&
            (!settings_.well_tempered || settings_.bias_factor>1), "WT bias factor must exceed one");
    if (settings_.well_tempered)
        require(std::isfinite(settings_.thermal_energy*(settings_.bias_factor-1)) &&
                settings_.thermal_energy*(settings_.bias_factor-1)>0,
                "WT thermal scale must be finite and positive");
}
std::size_t HillBias::heights_per_deposition() const {
    return settings_.mode==HillBiasMode::Parallel ? dimension() : 1;
}
void HillBias::validate_coordinates(const double* coordinates) const {
    require(coordinates!=nullptr, "Missing bias coordinates");
    for (std::size_t axis=0; axis<dimension(); ++axis)
        require(std::isfinite(coordinates[axis]), "Bias coordinates must be finite");
}
void HillBias::components(const double* coordinates, std::vector<double>& values,
                          std::vector<double>& derivatives) const {
    values.assign(dimension(),0.0);
    derivatives.assign(dimension(),0.0);
    for (std::size_t block=0; block<depositions(); block+=std::min(block_size_,depositions()-block))
      for (std::size_t hill=block; hill<block+std::min(block_size_,depositions()-block); ++hill)
        for (std::size_t axis=0; axis<dimension(); ++axis) {
            const std::size_t index=hill*dimension()+axis;
            KernelAxis kernel;
            if (settings_.pb_kernel==PBKernel::Plumed210) {
                const auto& configuration=settings_.axes[axis];
                double displacement=coordinates[axis]-centers_[index];
                require(std::isfinite(displacement),"Bias coordinate difference overflow");
                if (configuration.period>0) displacement=std::remainder(displacement,configuration.period);
                const double scaled=displacement/configuration.sigma;
                kernel={scaled*scaled,scaled/configuration.sigma};
                if (0.5*kernel.quadratic>=6.25) continue;
            } else kernel=kernel_axis(coordinates[axis],centers_[index],settings_.axes[axis]);
            const double contribution=heights_[index]*std::exp(-0.5*kernel.quadratic);
            if (contribution==0) continue; // 远处核下溢为零，其梯度也为零。
            values[axis]+=contribution;
            derivatives[axis]-=contribution*kernel.derivative;
        }
    for (std::size_t axis=0; axis<dimension(); ++axis) {
        finite_result(values[axis]);
        finite_result(derivatives[axis]);
    }
}
double HillBias::weights(const std::vector<double>& values, std::vector<double>& normalized) const {
    const double minimum=*std::min_element(values.begin(),values.end());
    normalized.resize(dimension());
    double partition=0;
    for (std::size_t axis=0; axis<dimension(); ++axis) {
        normalized[axis]=std::exp(-(values[axis]-minimum)/settings_.thermal_energy);
        partition+=normalized[axis];
    }
    for (auto& weight : normalized) weight/=partition;
    // mean 约定使空历史 V=0；与 sum 约定相差固定的 kBT*log(D)。
    return minimum-settings_.thermal_energy*std::log(partition/double(dimension()));
}
double HillBias::evaluate(const double* coordinates, double* gradient) const {
    validate_coordinates(coordinates);
    if (gradient) std::fill_n(gradient,dimension(),0.0);
    if (settings_.mode==HillBiasMode::Parallel) {
        std::vector<double> values, derivatives, conditional;
        components(coordinates,values,derivatives);
        const double energy=weights(values,conditional);
        finite_result(energy);
        if (gradient)
            for (std::size_t axis=0; axis<dimension(); ++axis) {
                gradient[axis]=conditional[axis]*derivatives[axis];
                finite_result(gradient[axis]);
            }
        return energy;
    }
    double energy=0;
    std::vector<double> derivative(dimension());
    for (std::size_t block=0; block<depositions(); block+=std::min(block_size_,depositions()-block))
      for (std::size_t hill=block; hill<block+std::min(block_size_,depositions()-block); ++hill) {
        double quadratic=0;
        for (std::size_t axis=0; axis<dimension(); ++axis) {
            const auto kernel=kernel_axis(coordinates[axis],centers_[hill*dimension()+axis],settings_.axes[axis]);
            quadratic+=kernel.quadratic;
            derivative[axis]=kernel.derivative;
        }
        const double contribution=heights_[hill]*std::exp(-0.5*quadratic);
        if (contribution==0) continue;
        energy+=contribution;
        if (gradient)
            for (std::size_t axis=0; axis<dimension(); ++axis)
                gradient[axis]-=contribution*derivative[axis];
    }
    finite_result(energy);
    if (gradient) for (std::size_t axis=0; axis<dimension(); ++axis) finite_result(gradient[axis]);
    return energy;
}
bool HillBias::deposit(std::int64_t step, const double* coordinates, const double* old_energy) {
    validate_coordinates(coordinates);
    require(step>=0 && step>=last_deposition_step(), "Bias deposition time moved backwards");
    if (step==last_deposition_step()) return false;
    std::vector<double> additions(heights_per_deposition(),settings_.initial_height);
    const double thermal_scale=settings_.thermal_energy*(settings_.bias_factor-1);
    if (settings_.mode==HillBiasMode::Parallel) {
        std::vector<double> values, derivatives, conditional;
        // 全部 V_j 与条件权重取自同一份旧历史，然后一次追加各维高度。
        components(coordinates,values,derivatives);
        weights(values,conditional);
        for (std::size_t axis=0; axis<dimension(); ++axis)
            additions[axis]*=conditional[axis]*
                (settings_.well_tempered ? std::exp(-values[axis]/thermal_scale) : 1.0);
    } else if (settings_.well_tempered) {
        const double energy=old_energy ? *old_energy : evaluate(coordinates);
        finite_result(energy);
        additions[0]*=std::exp(-energy/thermal_scale);
    }
    for (double height : additions) finite_result(height);
    append(step,coordinates,additions.data());
    return true;
}
void HillBias::set_block_size(std::size_t size) {
    require(size>0,"Bias block size must be positive");
    block_size_=size;
}
void HillBias::append(std::int64_t step, const double* coordinates, const double* heights, bool allow_same_step) {
    validate_coordinates(coordinates);
    require(heights!=nullptr,"Missing hill heights");
    require(step>=0 && (step>last_deposition_step() || (allow_same_step && step==last_deposition_step())),
            "Invalid bias append step");
    for (std::size_t axis=0;axis<heights_per_deposition();++axis)
        require(std::isfinite(heights[axis]) && heights[axis]>=0,"Invalid hill height");
    // 预留后再追加，内存分配失败不会留下半次 PB 沉积。
    require(dimension()<=centers_.max_size()-centers_.size(), "Bias history size overflow");
    require(heights_per_deposition()<=heights_.max_size()-heights_.size(), "Bias history size overflow");
    // 按几何容量增长，避免每次沉积重新复制全部历史。
    auto reserve_extra=[](auto& values, std::size_t count) {
        if (values.capacity()-values.size()>=count) return;
        const auto needed=values.size()+count;
        const auto growth=values.capacity()<=values.max_size()/2 ? values.capacity()*2 : values.max_size();
        values.reserve(std::max(needed,growth));
    };
    reserve_extra(centers_,dimension()); reserve_extra(heights_,heights_per_deposition()); reserve_extra(steps_,1);
    centers_.insert(centers_.end(),coordinates,coordinates+dimension());
    heights_.insert(heights_.end(),heights,heights+heights_per_deposition());
    steps_.push_back(step);
}
void HillBias::write_header(std::ostream& output) const {
    output<<std::setprecision(17);
    output<<"# METAD_TOY_BIAS 3 "<<(settings_.mode==HillBiasMode::Hills ? "HILLS" : "PBMETAD")
          <<" "<<dimension()<<"\n";
    output<<"# PARAMETERS "<<settings_.initial_height<<" "<<settings_.thermal_energy<<" "
          <<settings_.bias_factor<<" "<<int(settings_.well_tempered)<<" "
          <<(settings_.pb_kernel==PBKernel::Plumed210 ? "PLUMED_2_10" : "SMOOTH_GAUSSIAN")<<"\n";
    output<<"# SIGMA"; for (const auto& axis : settings_.axes) output<<" "<<axis.sigma;
    output<<"\n# PERIOD"; for (const auto& axis : settings_.axes) output<<" "<<axis.period;
    output<<"\n# FIELDS step";
    for (std::size_t axis=0; axis<dimension(); ++axis) output<<" s"<<axis+1;
    for (std::size_t axis=0; axis<heights_per_deposition(); ++axis) output<<" height"<<axis+1;
    output<<"\n";
    require(bool(output), "Failed writing bias history header");
}
void HillBias::write_last_deposition(std::ostream& output) const {
    require(!steps_.empty(), "No deposition to write");
    output<<std::setprecision(17)<<steps_.back();
    const auto hill=depositions()-1;
    for (std::size_t axis=0; axis<dimension(); ++axis) output<<" "<<centers_[hill*dimension()+axis];
    for (std::size_t axis=0; axis<heights_per_deposition(); ++axis)
        output<<" "<<heights_[hill*heights_per_deposition()+axis];
    output<<"\n";
    require(bool(output), "Failed writing bias deposition");
}
void HillBias::load(std::istream& input, unsigned workers) {
    // 同参数才能续写该历史；模式/周期核/宽度变化需创建新的偏势。
    // 17 位有效数字使本程序生成的参数可无损恢复。
    std::ostringstream expected_header;
    write_header(expected_header);
    std::istringstream expected(expected_header.str());
    bool old_header=false;
    for (int line=0; line<5; ++line) {
        std::string actual_line=read_line(input);
        if (line==0) {
            for (const std::string version : {"1","2"}) {
                const std::string prefix="# METAD_TOY_BIAS "+version+" ";
                if (actual_line.rfind(prefix,0)==0) {
                    old_header=true;
                    actual_line.replace(0,prefix.size(),"# METAD_TOY_BIAS 3 ");
                }
            }
        }
        // 早期文件未记录核类型，其数学含义固定为 SmoothGaussian。
        if (line==1 && old_header) actual_line+=" SMOOTH_GAUSSIAN";
        std::istringstream actual_row(actual_line), expected_row(read_line(expected));
        std::string actual, wanted;
        while (expected_row>>wanted) {
            require(bool(actual_row>>actual) && actual==wanted, "Bias history settings/header mismatch");
        }
        no_extra(actual_row);
    }
    HillBias restored(settings_);
    std::vector<std::string> records;
    std::string line;
    while (std::getline(input,line)) {
        require(!input.eof(),"Unterminated bias history record");
        if (!line.empty()) records.push_back(std::move(line));
    }
    require(!input.bad(),"Failed reading bias history");
    require(records.size()<=centers_.max_size()/dimension(),"History size overflow");
    restored.centers_.resize(records.size()*dimension());
    restored.heights_.resize(records.size()*heights_per_deposition());
    restored.steps_.resize(records.size());
    // 每个线程解析独立行区间，最终按文件顺序校验；不改变浮点表示或记录顺序。
    workers=std::max(1u,std::min(workers,16u));
    workers=std::min(workers,static_cast<unsigned>(records.size()/1024+1));
    std::vector<std::exception_ptr> errors(workers);
    auto parse=[&](unsigned worker) {
        try {
            for (std::size_t record=records.size()*worker/workers;record<records.size()*(worker+1)/workers;++record) {
                std::istringstream row(records[record]);
                auto& step=restored.steps_[record];
                require(bool(row>>step) && step>=0,"Invalid bias deposition step");
                for (std::size_t axis=0;axis<dimension();++axis) {
                    auto& value=restored.centers_[record*dimension()+axis];
                    require(bool(row>>value) && std::isfinite(value),"Invalid bias center record");
                }
                for (std::size_t axis=0;axis<heights_per_deposition();++axis) {
                    auto& value=restored.heights_[record*heights_per_deposition()+axis];
                    require(bool(row>>value) && std::isfinite(value) && value>=0,"Invalid bias height record");
                }
                no_extra(row);
            }
        } catch (...) {errors[worker]=std::current_exception();}
    };
    std::vector<std::thread> threads;
    try { for (unsigned worker=1;worker<workers;++worker) threads.emplace_back(parse,worker); }
    catch (...) {for (auto& thread:threads) thread.join();throw;}
    parse(0);for (auto& thread:threads) thread.join();
    for (const auto& error:errors) if (error) std::rethrow_exception(error);
    for (std::size_t record=1;record<records.size();++record)
        require(restored.steps_[record]>=restored.steps_[record-1],"Reversed deposition step");
    centers_.swap(restored.centers_); heights_.swap(restored.heights_); steps_.swap(restored.steps_);
}
} // namespace MetaD_zqc
