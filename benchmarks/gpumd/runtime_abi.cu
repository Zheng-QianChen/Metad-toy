// 只包含公开 C ABI：检查错误回传、非默认流、借用数组和物理回写。
// 此测试不包含 GPUMD 或 MetaD-toy 内部 C++ 头文件。
#include "zqc_metadtoy_api.h"
#include <cuda_runtime.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

void require(bool value,const char* text) { if (!value) throw std::runtime_error(text); }
void cuda_check(cudaError_t status) { require(status==cudaSuccess,cudaGetErrorString(status)); }
int main(int argc,char** argv) {
    try {
        require(argc==2,"Expected a unique output prefix");
        const std::string prefix=argv[1], config=prefix+".in", colvar=prefix+".COLVAR";
        const auto api=zqc_metadtoy_get_api(1);
        require(api && api->size==sizeof(ZqcApiV1) && api->abi_version==1,"ABI table mismatch");
        require(!zqc_metadtoy_get_api(999),"Unknown ABI must be rejected");
        std::ofstream(config) << "backend metadtoy\nstart fresh\noutput "+colvar+" 1\n"
            "cv r distance atoms 0 1\nexpression r\nbias harmonic center 1 kappa 2\n";
        char error[512]{};
        std::string messages;
        ZqcHostV1 host{sizeof(ZqcHostV1),[](void* user,const char* message) {
            *static_cast<std::string*>(user)+=std::string(message)+"\n";
        },&messages};
        void* handle=nullptr;
        require(api->create("/definitely/missing/metadtoy.in",&host,&handle,error,sizeof(error))!=0 &&
                !handle && std::string(error).find("Cannot open")!=std::string::npos,"Error must cross C ABI");
        require(api->create(config.c_str(),&host,&handle,error,sizeof(error))==0,error);
        require(api->post_run(handle,error,sizeof(error))!=0,"post_run before pre_run must fail");
        ZqcRunV1 run{sizeof(ZqcRunV1),0,0};
        ZqcFrameV1 frame{};
        frame.size=sizeof(frame); frame.abi_version=999;
        require(api->pre_run(handle,&frame,&run,error,sizeof(error))!=0 &&
                std::string(error).find("ABI")!=std::string::npos,"Frame ABI must be checked");
        frame.abi_version=1;
        double *positions=nullptr,*forces=nullptr,*virials=nullptr,*potential=nullptr;
        cudaStream_t stream;
        cuda_check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        cuda_check(cudaMalloc(&positions,6*sizeof(double)));
        cuda_check(cudaMalloc(&forces,6*sizeof(double)));
        cuda_check(cudaMalloc(&virials,18*sizeof(double)));
        cuda_check(cudaMalloc(&potential,2*sizeof(double)));
        const double coordinates[]={0,1.4,0,0,0,0};
        cuda_check(cudaMemcpyAsync(positions,coordinates,sizeof(coordinates),cudaMemcpyHostToDevice,stream));
        cuda_check(cudaMemsetAsync(forces,0,6*sizeof(double),stream));
        cuda_check(cudaMemsetAsync(virials,0,18*sizeof(double),stream));
        cuda_check(cudaMemsetAsync(potential,0,2*sizeof(double),stream));
        frame.positions={positions,2,3,1,2,6}; frame.forces={forces,2,3,1,2,6};
        frame.virials={virials,2,9,1,2,18}; frame.potential={potential,2,1,1,1,2};
        for(int axis=0;axis<3;++axis) { frame.box_lengths[axis]=20; frame.periodic[axis]=1; }
        frame.stream=stream;
        require(api->pre_run(handle,&frame,&run,error,sizeof(error))==0,error);
        require(api->pre_run(handle,&frame,&run,error,sizeof(error))!=0,"Duplicate pre_run must fail");
        require(api->evaluate(handle,&frame,0,error,sizeof(error))==0,error);
        double result_forces[6],result_virials[18],result_potential[2];
        cuda_check(cudaMemcpyAsync(result_forces,forces,sizeof(result_forces),cudaMemcpyDeviceToHost,stream));
        cuda_check(cudaMemcpyAsync(result_virials,virials,sizeof(result_virials),cudaMemcpyDeviceToHost,stream));
        cuda_check(cudaMemcpyAsync(result_potential,potential,sizeof(result_potential),cudaMemcpyDeviceToHost,stream));
        cuda_check(cudaStreamSynchronize(stream));
        require(std::abs(result_forces[0]-.8)<1e-12 && std::abs(result_forces[1]+.8)<1e-12,"Force mismatch");
        require(std::abs(result_potential[0]+result_potential[1]-.16)<1e-12,"Energy mismatch");
        require(std::abs(result_virials[0]+result_virials[1]+1.12)<1e-12,"Virial mismatch");
        require(api->end_of_step(handle,5,error,sizeof(error))!=0,"Step mismatch must fail");
        require(api->post_run(handle,error,sizeof(error))==0,error);
        api->destroy(handle);
        require(messages.find("shared runtime ABI 1")!=std::string::npos,"Host logger was not called");
        cuda_check(cudaFree(positions)); cuda_check(cudaFree(forces));
        cuda_check(cudaFree(virials)); cuda_check(cudaFree(potential)); cuda_check(cudaStreamDestroy(stream));
        std::cout<<"PASS: C ABI, errors, logger, nondefault stream, DISTANCE force/virial\n";
    } catch(const std::exception& exception) { std::cerr<<exception.what()<<"\n";return 1; }
}
