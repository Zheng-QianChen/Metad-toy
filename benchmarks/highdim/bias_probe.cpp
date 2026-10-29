// 测试桥接：Python 只采样和统计，偏势求值调用生产 C++ HillBias。
#include "zqc_hill_bias.h"
#include <stdexcept>
#include <memory>
#include <string>
using namespace MetaD_zqc;
static thread_local std::string failure;
extern "C" const char* bias_probe_error() {return failure.c_str();}
extern "C" void* bias_probe_create(int dimension,int parallel,const double* sigma,
                                   const double* centers,const double* heights,int hills) {
    try {
        HillBiasSettings settings;settings.mode=parallel ? HillBiasMode::Parallel : HillBiasMode::Hills;
        if(parallel) settings.pb_kernel=PBKernel::Plumed210;
        for(int axis=0;axis<dimension;++axis) settings.axes.push_back({sigma[axis],0});
        auto bias=std::make_unique<HillBias>(settings);
        for(int hill=0;hill<hills;++hill)
            bias->append(hill,centers+hill*dimension,heights+hill*(parallel ? dimension : 1));
        return bias.release();
    } catch(const std::exception& error) {failure=error.what();return nullptr;}
}
extern "C" void bias_probe_destroy(void* pointer) {delete static_cast<HillBias*>(pointer);}
extern "C" int bias_probe_evaluate(void* pointer,const double* points,int count,double* energies,double* gradients) {
    try {
        if(!pointer) throw std::invalid_argument("Missing bias");
        auto& bias=*static_cast<HillBias*>(pointer);
        for(int point=0;point<count;++point)
            energies[point]=bias.evaluate(points+point*bias.dimension(),
                                gradients ? gradients+point*bias.dimension() : nullptr);
        return 0;
    } catch(const std::exception& error) {failure=error.what();return 1;}
}

