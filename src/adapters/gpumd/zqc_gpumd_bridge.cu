#include "zqc_gpumd_bridge.cuh"
#include "zqc_metadtoy_api.h"
#include "integrate/integrate.cuh"
#include "model/atom.cuh"
#include "utilities/error.cuh"
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {
void require(bool condition,const char* message) {
    if (!condition) throw std::runtime_error(message);
}
// 宿主负责把库返回的错误转发到 GPUMD；C ABI 内部已经捕获算法异常。
template<class Function> void gpumd_boundary(Function&& function) {
    try { function(); }
    catch (const std::exception& error) { PRINT_INPUT_ERROR(error.what()); }
}
std::vector<std::string> tokenize(std::string line) {
    line.erase(line.find('#')==std::string::npos ? line.size() : line.find('#'));
    std::istringstream input(line);
    std::vector<std::string> tokens;
    for (std::string word;input>>word;) tokens.push_back(word);
    return tokens;
}
ZqcFrameV1 frame_from(Atom& atom,Box& box,long long step) {
    require(box.cpu_h[1]==0 && box.cpu_h[2]==0 && box.cpu_h[3]==0 &&
            box.cpu_h[5]==0 && box.cpu_h[6]==0 && box.cpu_h[7]==0,
            "MetaD-toy requires an orthogonal box");
    const uint64_t count=static_cast<uint64_t>(atom.number_of_atoms);
    ZqcFrameV1 frame{};
    frame.size=sizeof(frame);
    frame.abi_version=ZQC_METADTOY_ABI_VERSION;
    // GPUMD 的原子数组为 SoA，同分量连续、相邻分量相隔 N 个元素。
    frame.positions={atom.position_per_atom.data(),count,3,1,count,atom.position_per_atom.size()};
    frame.forces={atom.force_per_atom.data(),count,3,1,count,atom.force_per_atom.size()};
    frame.virials={atom.virial_per_atom.data(),count,9,1,count,atom.virial_per_atom.size()};
    frame.potential={atom.potential_per_atom.data(),count,1,1,1,atom.potential_per_atom.size()};
    frame.types={atom.type.data(),count,1,1,1,atom.type.size()};
    frame.box_lengths[0]=box.cpu_h[0]; frame.box_lengths[1]=box.cpu_h[4]; frame.box_lengths[2]=box.cpu_h[8];
    frame.periodic[0]=box.pbc_x; frame.periodic[1]=box.pbc_y; frame.periodic[2]=box.pbc_z;
    frame.step=step;
    frame.stream=nullptr; // GPUMD v5.9.1 力计算和积分使用默认流。
    return frame;
}
}

struct MetaDToyGPUMDBridge::Impl {
    const ZqcApiV1* api=nullptr; // 函数表由共享库持有，随进程加载。
    void* handle=nullptr;       // 算法状态由 create/destroy 配对管理。
    char error[4096]{};

    explicit Impl(const std::string& filename) {
        api=zqc_metadtoy_get_api(ZQC_METADTOY_ABI_VERSION);
        require(api && api->size==sizeof(ZqcApiV1) && api->abi_version==ZQC_METADTOY_ABI_VERSION,
                "Incompatible libmetadtoy ABI: update the library or rebuild the GPUMD shim");
        ZqcHostV1 host{};
        host.size=sizeof(host);
        host.log=[](void*,const char* message) { std::printf("%s\n",message); };
        check(api->create(filename.c_str(),&host,&handle,error,sizeof(error)));
    }
    ~Impl() { if (handle) api->destroy(handle); }
    void check(int status) {
        if (status) throw std::runtime_error(std::string("MetaD-toy: ")+error);
    }
};
MetaDToyGPUMDBridge::MetaDToyGPUMDBridge(const std::string& filename) : impl_(new Impl(filename)) {}
MetaDToyGPUMDBridge::~MetaDToyGPUMDBridge()=default;

std::unique_ptr<MetaDToyGPUMDBridge> MetaDToyGPUMDBridge::create_if_requested(const std::string& filename) {
    std::unique_ptr<MetaDToyGPUMDBridge> result;
    gpumd_boundary([&] {
        std::ifstream input(filename);
        require(bool(input),"Cannot open enhanced sampling input");
        for (std::string line;std::getline(input,line);) {
            const auto tokens=tokenize(line);
            if (tokens.empty()) continue;
            if (tokens.size()==2 && tokens[0]=="backend" && tokens[1]=="metadtoy")
                result.reset(new MetaDToyGPUMDBridge(filename));
            break;
        }
    });
    return result;
}
void MetaDToyGPUMDBridge::pre_run(Integrate& integrate,Atom& atom,Box& box) {
    gpumd_boundary([&] {
        const auto type=integrate.get_type();
        require(type==EnsembleType::NVE || type==EnsembleType::NVT_NHC ||
                type==EnsembleType::NVT_BDP || type==EnsembleType::NVT_LAN,
                "Supported ensembles: NVE, constant-temperature NVT_NHC/NVT_BDP/NVT_LAN");
        require(integrate.get_fixed_group()<0 && integrate.get_move_group()<0,
                "Fixed/moving groups are not supported by this bridge");
        ZqcRunV1 run{sizeof(ZqcRunV1),type!=EnsembleType::NVE,integrate.get_temperature1()};
        if (run.has_thermostat)
            require(integrate.get_temperature1()==integrate.get_temperature2(),
                    "A constant target temperature is required");
        auto frame=frame_from(atom,box,0);
        impl_->check(impl_->api->pre_run(impl_->handle,&frame,&run,impl_->error,sizeof(impl_->error)));
    });
}
void MetaDToyGPUMDBridge::evaluate(long long step,bool integrating,Atom& atom,Box& box) {
    gpumd_boundary([&] {
        auto frame=frame_from(atom,box,step);
        impl_->check(impl_->api->evaluate(impl_->handle,&frame,integrating,impl_->error,sizeof(impl_->error)));
    });
}
void MetaDToyGPUMDBridge::end_of_step(long long step) {
    gpumd_boundary([&] { impl_->check(impl_->api->end_of_step(impl_->handle,step,impl_->error,sizeof(impl_->error))); });
}
void MetaDToyGPUMDBridge::post_run() {
    gpumd_boundary([&] { impl_->check(impl_->api->post_run(impl_->handle,impl_->error,sizeof(impl_->error))); });
}
