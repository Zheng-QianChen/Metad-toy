#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
using Clock=std::chrono::steady_clock;
void ck(cudaError_t x){if(x!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(x));}
void dk(CUresult x){if(x!=CUDA_SUCCESS){const char*s=nullptr;cuGetErrorString(x,&s);throw std::runtime_error(s?s:"driver");}}
void nk(nvrtcResult x){if(x!=NVRTC_SUCCESS)throw std::runtime_error(nvrtcGetErrorString(x));}
std::string read_file(const std::string& path){std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error(path);return {std::istreambuf_iterator<char>(f),{}};}
double millis(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
struct Compiled{std::vector<char> cubin;double compile_ms,total_ms;};
Compiled compile(const std::string& source,const std::string& arch){
    auto total=Clock::now();nvrtcProgram p=nullptr;nk(nvrtcCreateProgram(&p,source.c_str(),"benchmark.cu",0,nullptr,nullptr));
    const char* options[]={arch.c_str(),"--std=c++17"};auto start=Clock::now();auto status=nvrtcCompileProgram(p,2,options);double ms=millis(start);
    if(status!=NVRTC_SUCCESS){size_t n;nvrtcGetProgramLogSize(p,&n);std::string log(n,'\0');nvrtcGetProgramLog(p,log.data());nvrtcDestroyProgram(&p);throw std::runtime_error(log);}
    size_t size;nk(nvrtcGetCUBINSize(p,&size));std::vector<char> bytes(size);nk(nvrtcGetCUBIN(p,bytes.data()));nk(nvrtcDestroyProgram(&p));return {std::move(bytes),ms,millis(total)};
}
struct Module{
    CUmodule mod=nullptr;CUfunction fwd,bwd;
    explicit Module(const void* bytes){dk(cuModuleLoadData(&mod,bytes));dk(cuModuleGetFunction(&fwd,mod,"cv_forward"));dk(cuModuleGetFunction(&bwd,mod,"cv_backward"));}
    ~Module(){if(mod)cuModuleUnload(mod);}
};
struct Buffers{
    std::array<double*,12> p{};
    explicit Buffers(int n){for(auto& q:p)ck(cudaMalloc(&q,size_t(n)*sizeof(double)));}
    ~Buffers(){for(auto q:p)cudaFree(q);}
};
__global__ void initialize(double*a,double*b,double*c,double*d,double*s,double*t,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    a[i]=.1+.01*(i%127);b[i]=-.3+.02*(i%71);c[i]=.7-.001*(i%97);d[i]=-.2+.005*(i%113);s[i]=.5+.1*(i%3);t[i]=-.25;
}
void launch(Module& m,bool mean,int phase,int n,Buffers& buf,cudaStream_t stream){
    auto& p=buf.p;double seed=2.5/n;
    if(phase!=1){
        void* one[]={&n,&p[0],&p[1],&p[2],&p[3],&p[6]};
        void* two[]={&n,&p[0],&p[1],&p[2],&p[6],&p[7]};
        dk(cuLaunchKernel(m.fwd,(n+255)/256,1,1,256,1,1,0,stream,mean?one:two,nullptr));
    }
    if(phase!=0){
        void* one[]={&n,&p[0],&p[1],&p[2],&p[3],&seed,&p[8],&p[9],&p[10],&p[11]};
        void* two[]={&n,&p[0],&p[1],&p[2],&p[4],&p[5],&p[8],&p[9],&p[10]};
        dk(cuLaunchKernel(m.bwd,(n+255)/256,1,1,256,1,1,0,stream,mean?one:two,nullptr));
    }
}
struct Batch{
    cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;int count=64;
    Batch(Module& m,bool mean,int phase,int n,Buffers& buffers,cudaStream_t stream){
        ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
        for(int i=0;i<count;++i)launch(m,mean,phase,n,buffers,stream);
        ck(cudaStreamEndCapture(stream,&graph));ck(cudaGraphInstantiate(&exec,graph,0));
    }
    ~Batch(){if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);}
    void run(cudaStream_t stream){ck(cudaGraphLaunch(exec,stream));}
};
struct Timer{
    cudaEvent_t start,stop;
    Timer(){ck(cudaEventCreate(&start));ck(cudaEventCreate(&stop));}
    ~Timer(){cudaEventDestroy(start);cudaEventDestroy(stop);}
    double measure(Batch& batch,int repeats,cudaStream_t stream){
        ck(cudaEventRecord(start,stream));for(int i=0;i<repeats;++i)batch.run(stream);
        ck(cudaEventRecord(stop,stream));ck(cudaEventSynchronize(stop));float ms;ck(cudaEventElapsedTime(&ms,start,stop));
        return ms*1000.0/(repeats*batch.count);
    }
};
void validate(Module& m,bool mean,int n,Buffers& buffers,cudaStream_t stream,std::ofstream& log,const std::string& name){
    launch(m,mean,2,n,buffers,stream);ck(cudaStreamSynchronize(stream));
    std::vector<int> indices=mean?std::vector<int>{6,8,9,10,11}:std::vector<int>{6,7,8,9,10};
    double max_error=0;
    for(int k:indices){
        std::vector<double> h(n);ck(cudaMemcpy(h.data(),buffers.p[k],size_t(n)*8,cudaMemcpyDeviceToHost));
        for(int i=0;i<n;++i){
            double a=.1+.01*(i%127),b=-.3+.02*(i%71),c=.7-.001*(i%97),d=-.2+.005*(i%113),s=.5+.1*(i%3),t=-.25,seed=2.5/n,expect=0;
            if(mean){if(k==6)expect=a+b*c-std::exp(d);if(k==8)expect=seed;if(k==9)expect=seed*c;if(k==10)expect=seed*b;if(k==11)expect=-seed*std::exp(d);}
            else{if(k==6)expect=a*b;if(k==7)expect=a*c;if(k==8)expect=s*b+t*c;if(k==9)expect=s*a;if(k==10)expect=t*a;}
            double e=std::abs(h[i]-expect);if(!std::isfinite(h[i]) || e>1e-12+std::abs(expect)*1e-12)throw std::runtime_error("Validation failed: "+name);
            max_error=std::max(max_error,e);
        }
    }
    log<<name<<','<<n<<','<<std::setprecision(17)<<max_error<<"\n";
}
int main(int argc,char**argv){
    if(argc<2 || argc>3)return 2;const std::string dir=argv[1];const bool focus=argc==3 && std::string(argv[2])=="focus";
    try{
        ck(cudaSetDevice(0));ck(cudaFree(nullptr));cudaDeviceProp prop{};ck(cudaGetDeviceProperties(&prop,0));
        std::string arch="--gpu-architecture=sm_"+std::to_string(prop.major)+std::to_string(prop.minor);
        cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        const std::vector<int> sizes=focus?std::vector<int>{1048576,4194304}:std::vector<int>{3,1024,16384,65536,262144,1048576,4194304};Buffers buffers(sizes.back());
        auto&p=buffers.p;initialize<<<(sizes.back()+255)/256,256,0,stream>>>(p[0],p[1],p[2],p[3],p[4],p[5],sizes.back());ck(cudaStreamSynchronize(stream));
        std::ofstream raw(dir+"/timings.csv"),compiles(dir+"/compile.csv"),checks(dir+"/validation.csv"),attrs(dir+"/attributes.csv"),host(dir+"/host_submit.csv");
        raw<<"case,n,phase,variant,round,repeats,us\n";
        compiles<<"case,variant,repetition,compile_ms,total_ms\n";
        checks<<"variant,n,max_abs_error\n";
        attrs<<"case,variant,function,registers,local_bytes,shared_bytes\n";
        host<<"case,n,variant,round,pair_us\n";
        const std::array<std::string,3> names={"graph_nvrtc","manual_nvrtc","manual_nvcc"};
        const std::array<std::string,3> phases={"forward","backward","pair"};std::mt19937 rng(12345);Timer timer;
        for(const std::string name:{"mean_exp","two_dims"}){
            bool mean=name=="mean_exp";std::vector<std::unique_ptr<Module>> modules;
            for(int variant=0;variant<2;++variant){
                const std::string source=read_file(dir+"/"+name+(variant==0?".graph.cu":".manual.cu"));
                for(int repeat=0;repeat<4;++repeat){
                    auto compiled=compile(source,arch);compiles<<name<<','<<names[variant]<<','<<repeat<<','<<compiled.compile_ms<<','<<compiled.total_ms<<'\n';
                    if(repeat==0){std::ofstream out(dir+"/"+name+"."+names[variant]+".cubin",std::ios::binary);out.write(compiled.cubin.data(),compiled.cubin.size());modules.emplace_back(std::make_unique<Module>(compiled.cubin.data()));}
                }
            }
            auto offline=read_file(dir+"/"+name+".manual_nvcc.cubin");modules.emplace_back(std::make_unique<Module>(offline.data()));
            for(int v=0;v<3;++v)for(int f=0;f<2;++f){int regs,local,shared;CUfunction fn=f?modules[v]->bwd:modules[v]->fwd;dk(cuFuncGetAttribute(&regs,CU_FUNC_ATTRIBUTE_NUM_REGS,fn));dk(cuFuncGetAttribute(&local,CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,fn));dk(cuFuncGetAttribute(&shared,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,fn));attrs<<name<<','<<names[v]<<','<<(f?"backward":"forward")<<','<<regs<<','<<local<<','<<shared<<'\n';}
            for(int n:sizes){
                for(int v=0;v<3;++v)validate(*modules[v],mean,n,buffers,stream,checks,name+"/"+names[v]);
                for(int phase=(focus?2:0);phase<3;++phase){
                    std::vector<std::unique_ptr<Batch>> batches;std::array<int,3> repetitions{};
                    for(int v=0;v<3;++v){
                        batches.emplace_back(std::make_unique<Batch>(*modules[v],mean,phase,n,buffers,stream));
                        for(int warm=0;warm<5;++warm)batches.back()->run(stream);ck(cudaStreamSynchronize(stream));
                        const double us=timer.measure(*batches[v],2,stream);
                        repetitions[v]=std::max(2,std::min(200,int(std::ceil(5000.0/(us*batches[v]->count)))));
                    }
                    for(int round=0;round<(focus?21:9);++round){
                        std::array<int,3> order={0,1,2};std::shuffle(order.begin(),order.end(),rng);
                        for(int v:order){double us=timer.measure(*batches[v],repetitions[v],stream);raw<<name<<','<<n<<','<<phases[phase]<<','<<names[v]<<','<<round<<','<<repetitions[v]<<','<<std::setprecision(10)<<us<<'\n';}
                    }
                }
                // 另外记录普通逐次提交的前后向整对耗时，包含 CPU 提交及最终同步。
                for(int round=0;round<5;++round){
                    std::array<int,3> order={0,1,2};std::shuffle(order.begin(),order.end(),rng);
                    for(int v:order){
                        for(int i=0;i<5;++i)launch(*modules[v],mean,2,n,buffers,stream);ck(cudaStreamSynchronize(stream));
                        auto start=Clock::now();for(int i=0;i<200;++i)launch(*modules[v],mean,2,n,buffers,stream);ck(cudaStreamSynchronize(stream));
                        host<<name<<','<<n<<','<<names[v]<<','<<round<<','<<millis(start)*1000/200<<'\n';
                    }
                }
                raw.flush();host.flush();std::cout<<"DONE "<<name<<" N="<<n<<std::endl;
            }
        }
        ck(cudaStreamDestroy(stream));std::cout<<"BENCHMARK_PASS "<<prop.name<<std::endl;
    }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}
}
