// Pointwise bridge to the actual production CUDA evaluator. Each call compares
// a two-center configuration; multiplying MEAN by two also checks SUM semantics
// without adding a new aggregation mode to the production implementation.
#include "zqc_shared_bias.cuh"
#include "zqc_gpu_buffer.cuh"
#include <string>
#include <stdexcept>
using namespace MetaD_zqc;
namespace {
thread_local std::string last_error;
void check(cudaError_t status) {
    if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
}
extern "C" {
const char* shared_probability_cuda_error() {return last_error.c_str();}
int shared_probability_cuda_evaluate(const char* path,int count,const double* pairs,
                                    double* energies,double* derivatives) {
    try {
        FrozenSharedBias bias(path);
        GpuBuffer<double> coordinates; GpuBuffer<int> slots;
        coordinates.grow_to(2,__FILE__,__LINE__); slots.grow_to(2,__FILE__,__LINE__);
        const int indices[]={0,1};
        check(cudaMemcpy(slots.ptr,indices,sizeof(indices),cudaMemcpyHostToDevice));
        slots.record_write();
        for(int i=0;i<count;++i) {
            // The previous evaluation is consumed before this two-item buffer
            // is reused. Production ArrayView waiting still guards the read.
            check(cudaMemcpy(coordinates.ptr,pairs+2*i,2*sizeof(double),cudaMemcpyHostToDevice));
            coordinates.record_write();
            auto result=bias.evaluate(coordinates.view(2),slots.view(2));
            result.gradient.wait_host();
            check(cudaMemcpy(derivatives+2*i,result.gradient.data,2*sizeof(double),cudaMemcpyDeviceToHost));
            energies[i]=result.energy;
        }
        last_error.clear(); return 0;
    } catch(const std::exception& error) {last_error=error.what(); return -1;}
}
}
