// 本实验直接使用生产 WEIGHT_CHEM 和实际 Graph 生成的 NVRTC kernel。
#define ZQC_STRUFACTOR_KERNELS_ONLY
#include "../../src/cvs/CVs_Strufactor.cu"
#include "fused_kernels.cuh"
#include "zqc_graph.cuh"
#include <cuda.h>
#include <nvrtc.h>
#include <array>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <chrono>

using namespace MetaD_zqc;
using LAMMPS_NS::tagint;
using Position=std::array<double,3>;
cudaStream_t benchmark_stream=nullptr;
void check(cudaError_t error) { if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error)); }
void driver(CUresult result) {
    if(result!=CUDA_SUCCESS) { const char* error=nullptr;cuGetErrorString(result,&error);throw std::runtime_error(error?error:"CUDA driver error"); }
}
void rtc(nvrtcResult result) { if(result!=NVRTC_SUCCESS) throw std::runtime_error(nvrtcGetErrorString(result)); }
template<class T> void upload(GpuBuffer<T>& buffer,const std::vector<T>& values) {
    check(cudaMemcpy(buffer.ptr,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice));
}
std::vector<double> download(const GpuBuffer<double>& buffer,size_t count) {
    std::vector<double> values(count);check(cudaMemcpy(values.data(),buffer.ptr,count*sizeof(double),cudaMemcpyDeviceToHost));return values;
}

struct Module {
    CUmodule module=nullptr;
    CUfunction forward=nullptr,backward=nullptr;
    explicit Module(const std::string& directory) {
        GpuBuffer<double> values(1);
        Leaf input(values,1);
        Graph graph(Graph::Symbols{{"w",&input}});
        graph.parse_expr("fermi(w,-4,0.4)");graph.build();
        std::ofstream(directory+"/switch.graph.cu")<<graph.kernel_source;
        nvrtcProgram program=nullptr;
        rtc(nvrtcCreateProgram(&program,graph.kernel_source.c_str(),"switch.graph.cu",0,nullptr,nullptr));
        const char* options[]={"--std=c++17","--gpu-architecture=sm_89"};
        const auto start=std::chrono::steady_clock::now();
        const auto status=nvrtcCompileProgram(program,2,options);
        if(status!=NVRTC_SUCCESS) {
            size_t size=0;nvrtcGetProgramLogSize(program,&size);std::string log(size,'\0');nvrtcGetProgramLog(program,log.data());throw std::runtime_error(log);
        }
        size_t size=0;rtc(nvrtcGetCUBINSize(program,&size));std::vector<char> cubin(size);rtc(nvrtcGetCUBIN(program,cubin.data()));
        driver(cuModuleLoadData(&module,cubin.data()));driver(cuModuleGetFunction(&forward,module,"cv_forward"));driver(cuModuleGetFunction(&backward,module,"cv_backward"));
        rtc(nvrtcDestroyProgram(&program));
        std::cout<<"NVRTC compile/load outside timing: "<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()<<" ms\n";
    }
    ~Module(){ if(module)cuModuleUnload(module); }
};

struct Fixture {
    int atom_count,center_count;
    double cutoff;
    std::vector<tagint> centers,offsets,neighbors;
    std::vector<int> counts,types;
    std::vector<double> geometry;
    Fixture(int cells,int requested_centers,double cutoff_) : atom_count(4*cells*cells*cells),
        center_count(requested_centers?requested_centers:atom_count),cutoff(cutoff_) {
        const Position basis[4]={{0,0,0},{0,.5,.5},{.5,0,.5},{.5,.5,0}};
        auto wrap=[cells](int value){return (value%cells+cells)%cells;};
        auto local_id=[&](int x,int y,int z,int b){return ((wrap(z)*cells+wrap(y))*cells+wrap(x))*4+b;};
        for(int atom=0;atom<atom_count;++atom)types.push_back(1+(atom/4+atom%4)%3);
        const int radius=static_cast<int>(std::ceil(cutoff));
        // 邻居偏移只依赖晶格基元；中心 localid 使用紧凑前缀，便于隔离算子段的计时。
        struct Neighbor { int x,y,z,b;Position displacement;double squared; };
        std::vector<Neighbor> patterns[4];
        for(int b=0;b<4;++b)
            for(int z=-radius;z<=radius;++z)for(int y=-radius;y<=radius;++y)for(int x=-radius;x<=radius;++x)
                for(int neighbor_basis=0;neighbor_basis<4;++neighbor_basis) {
                    Position displacement={x+basis[neighbor_basis][0]-basis[b][0],y+basis[neighbor_basis][1]-basis[b][1],z+basis[neighbor_basis][2]-basis[b][2]};
                    double squared=std::inner_product(displacement.begin(),displacement.end(),displacement.begin(),0.0);
                    if(squared>1e-12 && squared<=cutoff*cutoff)patterns[b].push_back({x,y,z,neighbor_basis,displacement,squared});
                }
        for(int center=0;center<center_count;++center) {
            int cell=center/4;const int x=cell%cells;cell/=cells;const int y=cell%cells,z=cell/cells;
            centers.push_back(center);offsets.push_back(neighbors.size());counts.push_back(patterns[center%4].size());
            for(const auto& neighbor:patterns[center%4]) {
                neighbors.push_back(local_id(x+neighbor.x,y+neighbor.y,z+neighbor.z,neighbor.b));
                geometry.insert(geometry.end(),{neighbor.displacement[0],neighbor.displacement[1],neighbor.displacement[2],neighbor.squared});
            }
        }
        offsets.push_back(neighbors.size());
    }
};

struct Calculation {
    const Fixture& fixture;
    Module& module;
    StruFactorData data;
    GpuBuffer<int> types;
    GpuBuffer<double> weights,raw,separate_output,fused_raw,fused_output,seeds,adjoints,forces,virials;
    Calculation(const Fixture& fixture_,Module& module_) : fixture(fixture_),module(module_),data(fixture.center_count,fixture.neighbors.size()),
        types(fixture.atom_count),weights(4),raw(fixture.atom_count),separate_output(fixture.atom_count),fused_raw(fixture.atom_count),fused_output(fixture.atom_count),
        seeds(fixture.center_count),adjoints(fixture.center_count),forces(3*fixture.atom_count),virials(6*fixture.atom_count) {
        upload(data.d_group_indices,fixture.centers);upload(data.d_group_numneigh,fixture.offsets);upload(data.d_neigh_in_cutoff_r,fixture.counts);
        upload(data.d_calculated_numneigh,fixture.neighbors);upload(data.d_group_dminneigh,fixture.geometry);
        upload(types,fixture.types);upload(weights,std::vector<double>{0,1,-.7,.35});
        upload(seeds,std::vector<double>(fixture.center_count,.7/fixture.center_count));
    }
    void forward(int fused) {
        if(fused) fused_weighted_forward(data,fixture.cutoff*fixture.cutoff,types.ptr,weights.ptr,fused_raw.ptr,fused_output.ptr,benchmark_stream,128);
        else {
            weighted_chem_forward(data,fixture.cutoff*fixture.cutoff,types.ptr,weights.ptr,raw.ptr,benchmark_stream,128);
            int count=fixture.center_count;void* args[]={&count,&raw.ptr,&separate_output.ptr};
            driver(cuLaunchKernel(module.forward,(count+255)/256,1,1,256,1,1,0,benchmark_stream,args,nullptr));
        }
    }
    void backward(int fused) {
        check(cudaMemsetAsync(forces.ptr,0,3*fixture.atom_count*sizeof(double),benchmark_stream));
        check(cudaMemsetAsync(virials.ptr,0,6*fixture.atom_count*sizeof(double),benchmark_stream));
        if(fused) fused_weighted_backward(data,fixture.cutoff*fixture.cutoff,types.ptr,weights.ptr,fused_raw.ptr,seeds.ptr,forces.ptr,virials.ptr,benchmark_stream,128);
        else {
            int count=fixture.center_count;void* args[]={&count,&raw.ptr,&seeds.ptr,&adjoints.ptr};
            driver(cuLaunchKernel(module.backward,(count+255)/256,1,1,256,1,1,0,benchmark_stream,args,nullptr));
            weighted_chem_backward(data,fixture.cutoff*fixture.cutoff,types.ptr,weights.ptr,adjoints.ptr,forces.ptr,virials.ptr,benchmark_stream,128);
        }
    }
    void verify() {
        forward(0);backward(0);
        const auto values=download(separate_output,fixture.center_count),reference_forces=download(forces,3*fixture.atom_count),reference_virials=download(virials,6*fixture.atom_count);
        forward(1);backward(1);
        const auto fused_values=download(fused_output,fixture.center_count),actual_forces=download(forces,3*fixture.atom_count),actual_virials=download(virials,6*fixture.atom_count);
        auto error=[](const auto& a,const auto& b){double maximum=0;for(size_t i=0;i<a.size();++i){if(!std::isfinite(a[i]) || !std::isfinite(b[i]))throw std::runtime_error("nonfinite result");maximum=std::max(maximum,std::abs(a[i]-b[i]));}return maximum;};
        const double value_error=error(values,fused_values),force_error=error(reference_forces,actual_forces),virial_error=error(reference_virials,actual_virials);
        if(value_error>1e-13 || force_error>1e-12 || virial_error>1e-12)throw std::runtime_error("fused/separate mismatch");
        const auto maximum=std::max_element(reference_forces.begin(),reference_forces.end(),[](double a,double b){return std::abs(a)<std::abs(b);});
        if(maximum==reference_forces.end() || std::abs(*maximum)<1e-12)throw std::runtime_error("benchmark force is zero");
        std::cout<<"VALID atoms="<<fixture.atom_count<<" centers="<<fixture.center_count<<" neighbors="<<fixture.counts[0]<<" value_error="<<value_error<<" force_error="<<force_error<<" virial_error="<<virial_error<<'\n';
    }
};

struct Replay {
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    cudaEvent_t start,stop;
    static constexpr int batch=32;
    explicit Replay(const std::function<void()>& launch) {
        for(int i=0;i<8;++i)launch();check(cudaDeviceSynchronize());
        check(cudaStreamBeginCapture(benchmark_stream,cudaStreamCaptureModeGlobal));
        for(int i=0;i<batch;++i)launch();
        check(cudaStreamEndCapture(benchmark_stream,&graph));check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
        check(cudaEventCreate(&start));check(cudaEventCreate(&stop));
        check(cudaGraphLaunch(executable,benchmark_stream));check(cudaDeviceSynchronize());
    }
    double milliseconds(int repetitions) {
        check(cudaEventRecord(start,benchmark_stream));for(int i=0;i<repetitions;++i)check(cudaGraphLaunch(executable,benchmark_stream));check(cudaEventRecord(stop,benchmark_stream));check(cudaEventSynchronize(stop));
        float elapsed=0;check(cudaEventElapsedTime(&elapsed,start,stop));return elapsed;
    }
    ~Replay(){cudaGraphExecDestroy(executable);cudaGraphDestroy(graph);cudaEventDestroy(start);cudaEventDestroy(stop);}
};

int main(int argc,char** argv) {
    if(argc<2)return 2;
    try {
        const int rounds=argc>2?std::stoi(argv[2]):9;
        std::ofstream csv(std::string(argv[1])+"/micro.csv");csv<<"atoms,centers,neighbors,phase,round,separate_us,fused_us,speedup\n"<<std::setprecision(12);
        check(cudaStreamCreate(&benchmark_stream));
        Module module(argv[1]);std::mt19937 random(20261015);
        const std::array<std::pair<int,int>,4> sizes={{{4,0},{10,0},{20,0},{20,64}}};
        for(const auto& size:sizes)for(double cutoff:{.76,1.25,1.9}) {
            Fixture fixture(size.first,size.second,cutoff);Calculation calculation(fixture,module);calculation.verify();
            for(const std::string phase:{"forward","backward","pair"}) {
                auto operation=[&](int fused){if(phase!="backward")calculation.forward(fused);if(phase!="forward")calculation.backward(fused);};
                Replay separate([&]{operation(0);}),fused([&]{operation(1);});
                const int repetitions=std::max(1,std::min(20000,static_cast<int>(30.0/std::min(separate.milliseconds(1),fused.milliseconds(1)))));
                std::vector<double> ratios;
                for(int round=0;round<rounds;++round) {
                    double times[2];
                    if(random()%2){times[0]=separate.milliseconds(repetitions);times[1]=fused.milliseconds(repetitions);}
                    else {times[1]=fused.milliseconds(repetitions);times[0]=separate.milliseconds(repetitions);}
                    for(double& time:times)time=time*1000/(repetitions*Replay::batch);
                    csv<<fixture.atom_count<<','<<fixture.center_count<<','<<fixture.counts[0]<<','<<phase<<','<<round<<','<<times[0]<<','<<times[1]<<','<<times[0]/times[1]<<'\n';
                    ratios.push_back(times[0]/times[1]);
                }
                std::sort(ratios.begin(),ratios.end());
                std::cout<<"TIMING centers="<<fixture.center_count<<" neighbors="<<fixture.counts[0]<<" phase="<<phase<<" fused_speedup="<<ratios[rounds/2]<<std::endl;
            }
        }
        std::cout<<"SWITCH_MICRO_PASS\n";
    } catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
