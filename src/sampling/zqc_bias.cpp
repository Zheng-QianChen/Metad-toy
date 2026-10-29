#include "zqc_bias.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace MetaD_zqc {
namespace {
BiasSettings normalize(BiasSettings settings) {
    if (settings.backend==BiasBackend::Auto)
        settings.backend=settings.hills.axes.size()>3 ? BiasBackend::PBMetaD : BiasBackend::Grid;
    settings.hills.mode=settings.backend==BiasBackend::PBMetaD ? HillBiasMode::Parallel : HillBiasMode::Hills;
    settings.hills.pb_kernel=settings.backend==BiasBackend::PBMetaD ? settings.pb_kernel : PBKernel::SmoothGaussian;
    if (settings.pace<0) throw std::invalid_argument("Bias pace must be nonnegative");
    if (settings.history_path.empty()) throw std::invalid_argument("Missing bias history path");
    if (!settings.history_workers || settings.history_workers>16)
        throw std::invalid_argument("Bias history_workers must be between 1 and 16");
    if (settings.backend==BiasBackend::Grid || settings.backend==BiasBackend::SparseGrid) {
        if (settings.grid_axes.size()!=settings.hills.axes.size() || settings.grid_axes.size()>3)
            throw std::invalid_argument("Grid requires 1-3 configured axes");
        for (std::size_t axis=0;axis<settings.grid_axes.size();++axis) {
            settings.grid_axes[axis].sigma=settings.hills.axes[axis].sigma;
            settings.grid_axes[axis].period=settings.hills.axes[axis].period;
        }
    }
    return settings;
}
}
const char* BiasSession::name(BiasBackend backend) {
    switch(backend) {
    case BiasBackend::Auto: return "AUTO";
    case BiasBackend::Grid: return "GRID";
    case BiasBackend::SparseGrid: return "SPARSE_GRID";
    case BiasBackend::Hills: return "HILLS";
    case BiasBackend::PBMetaD: return "PBMETAD";
    }
    throw std::invalid_argument("Unknown bias backend");
}
BiasSession::BiasSession(BiasSettings settings, BiasHost host)
    : settings_(normalize(std::move(settings))),host_(std::move(host)),
      history_(settings_.hills),packet_(dimension()+1,0.0) {
    if (host_.root && (settings_.backend==BiasBackend::Grid || settings_.backend==BiasBackend::SparseGrid))
        grid_=std::make_unique<BiasGrid>(settings_.grid_axes,settings_.backend==BiasBackend::SparseGrid);
}
void BiasSession::load_legacy(std::istream& input) {
    if (!grid_) throw std::invalid_argument("Legacy grid HILLS requires a grid backend");
    for (const auto& axis:settings_.hills.axes)
        if (axis.period!=0 || axis.sigma!=settings_.hills.axes[0].sigma)
            throw std::invalid_argument("Legacy HILLS requires uniform sigma and nonperiodic axes");
    std::string line;
    while (std::getline(input,line)) {
        if (input.eof()) throw std::invalid_argument("Unterminated legacy HILLS record");
        if (line.empty() || line[0]=='#') continue;
        std::istringstream row(line);
        std::int64_t step;double height,sigma;std::string extra;
        std::vector<double> center(dimension());
        if (!(row>>step)) throw std::invalid_argument("Invalid legacy HILLS step");
        for (auto& value:center) if (!(row>>value)) throw std::invalid_argument("Invalid legacy HILLS center");
        if (!(row>>height>>sigma) || (row>>extra) || sigma!=settings_.hills.axes[0].sigma)
            throw std::invalid_argument("Legacy HILLS sigma/fields mismatch");
        history_.append(step,center.data(),&height,true);
    }
    if (input.bad()) throw std::runtime_error("Failed reading legacy HILLS");
    legacy_history_=true;
}
void BiasSession::initialize(std::int64_t current_step) {
    if (initialized_) return;
    if (current_step<0) throw std::invalid_argument("Negative simulation step");
    if (host_.root) {
        const auto start=std::chrono::steady_clock::now();
        if (settings_.restart) {
            std::ifstream input(settings_.history_path);
            if (!input) throw std::runtime_error("Cannot open requested bias restart: "+settings_.history_path);
            std::string first_line;std::getline(input,first_line);input.clear();input.seekg(0);
            if (first_line.rfind("# METAD_TOY_BIAS ",0)==0) history_.load(input,settings_.history_workers);
            else load_legacy(input);
            if (history_.last_deposition_step()>current_step)
                throw std::invalid_argument("Bias history extends beyond the current simulation step");
            if (grid_) grid_->rebuild(history_.centers().data(),history_.heights().data(),
                                     history_.depositions(),settings_.history_workers);
        }
        output_.open(settings_.history_path,settings_.restart ? std::ios::app : std::ios::trunc);
        if (!output_) throw std::runtime_error("Cannot write bias history: "+settings_.history_path);
        if (!settings_.restart) history_.write_header(output_);
        output_.flush();
        if (!output_) throw std::runtime_error("Cannot flush bias history");
        if (host_.log) {
            std::ostringstream message;
            const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            message<<"[Bias] backend="<<name(backend())<<" dimensions="<<dimension()
                   <<" history="<<history_.depositions()<<" workers="<<settings_.history_workers
                   <<" initialize_s="<<seconds;
            if (backend()==BiasBackend::PBMetaD) message<<" kernel="<<(settings_.pb_kernel==PBKernel::Plumed210 ? "PLUMED_2_10" : "SMOOTH_GAUSSIAN")<<" normalization="<<(settings_.pb_sum_normalization ? "SUM" : "MEAN");
            host_.log(message.str());
        }
    }
    initialized_=true;
}
void BiasSession::write_record() {
    if (legacy_history_) {
        output_<<std::setprecision(17)<<history_.last_deposition_step();
        const auto offset=(history_.depositions()-1)*dimension();
        for (std::size_t axis=0;axis<dimension();++axis) output_<<" "<<history_.centers()[offset+axis];
        output_<<" "<<history_.heights().back()<<" "<<settings_.hills.axes[0].sigma<<"\n";
    } else history_.write_last_deposition(output_);
    output_.flush();
    if (!output_) throw std::runtime_error("Failed writing complete bias deposition");
}
void BiasSession::deposit_now(std::int64_t step, const double* coordinates) {
    if (!initialized_) throw std::logic_error("Bias deposition before initialization");
    if (!host_.root) return;
    if (grid_) {
        const double energy=grid_->evaluate(coordinates);
        double height=settings_.hills.initial_height;
        if (settings_.hills.well_tempered)
            height*=std::exp(-energy/(settings_.hills.thermal_energy*(settings_.hills.bias_factor-1)));
        history_.append(step,coordinates,&height,true);
        grid_->add(coordinates,height);
    } else if (!history_.deposit(step,coordinates)) return;
    write_record();
}
bool BiasSession::advance(std::int64_t step, const double* coordinates) {
    if (!initialized_) throw std::logic_error("Bias advance before initialization");
    if (!host_.root || !settings_.pace || step%settings_.pace) return false;
    if (step==history_.last_deposition_step()) return false;
    deposit_now(step,coordinates);
    return true;
}
double BiasSession::local_evaluate(const double* coordinates,double* gradient) {
    double energy=grid_ ? grid_->evaluate(coordinates,gradient) : history_.evaluate(coordinates,gradient);
    if (backend()==BiasBackend::PBMetaD && settings_.pb_sum_normalization)
        energy-=settings_.hills.thermal_energy*std::log(double(dimension()));
    return energy;
}
double BiasSession::evaluate(const double* coordinates,double* gradient) {
    if (!initialized_) throw std::logic_error("Bias evaluation before initialization");
    if (host_.root) packet_[0]=local_evaluate(coordinates,packet_.data()+1);
    if (host_.broadcast) host_.broadcast(packet_.data(),packet_.size());
    if (gradient) std::copy_n(packet_.data()+1,dimension(),gradient);
    return packet_[0];
}
} // namespace MetaD_zqc

