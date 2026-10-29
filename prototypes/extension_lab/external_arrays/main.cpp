// 外部数组接入的独立验证程序。
// GPUMD/Kokkos 持有坐标与力 → producer 流写入 → event → NVRTC kernel 读取并累加力
// → consumer 完成 → Host 对比解析值与有限差分。
// 此处使用每原子二次势 V_i=0.5*spring*|r_i|^2，将布局和同步问题单独验证。
#include "borrowed_array.hpp"
#include "kokkos_type.h"
#include "utilities/gpu_vector.cuh"
#include <cuda.h>
#include <nvrtc.h>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

// 64 位 ABI 下：指针 8 字节、两个 stride 共 16 字节、两个计数共 8 字节。
// kernel_source 中重复声明这一参数结构，字段顺序和宽度与头文件逐项对应。
static_assert(sizeof(BorrowedArray)==32,"Unexpected host/NVRTC view ABI");

// Runtime、Driver 和 NVRTC 各自返回不同的错误码；统一转成异常供 main 报错退出。
void cuda_check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
void driver_check(CUresult result) {
    if (result != CUDA_SUCCESS) { const char* message; cuGetErrorString(result, &message); throw std::runtime_error(message); }
}
void rtc_check(nvrtcResult result) {
    if (result != NVRTC_SUCCESS) throw std::runtime_error(nvrtcGetErrorString(result));
}
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// 运行时编译的源码只保存在此字符串中，NVRTC 直接接收它。
// 每个线程处理一个 owned 原子的三分量，将 F_bias=-spring*r 加到外部 forces。
// 坐标与力可以采用不同布局，分别用各自的 stride 寻址。
// 该例各线程独占自己的三个力位置，因此 += 足以累加已有的引擎力；
// 多中心向同一邻居散射导数的 CV 需要另行处理并发写入与 ghost 规约。
const char* kernel_source = R"CUDA(
struct BorrowedArray {
    double* data;
    unsigned long long atom_stride, component_stride;
    int capacity, owned;
};
extern "C" __global__ void evaluate(BorrowedArray positions, BorrowedArray forces, double spring) {
    const int atom = blockIdx.x * blockDim.x + threadIdx.x;
    // 尾部线程以及 ghost 都在本次计算负责的范围之外。
    if (atom >= positions.owned) return;
    for (int component = 0; component < 3; ++component) {
        const double coordinate = positions.data[atom * positions.atom_stride + component * positions.component_stride];
        forces.data[atom * forces.atom_stride + component * forces.component_stride] += -spring * coordinate;
    }
}
)CUDA";

// 持有一次 NVRTC 编译得到的 Driver 模块和函数句柄，供多个数组大小重复使用。
// 构造前由 CUDA Runtime/Kokkos 建立当前上下文；销毁前由调用方完成已提交的计算。
class RuntimeKernel {
    CUmodule module{};       // 模块持有 Driver 加载后的设备代码，析构时卸载。
    CUfunction function{};   // module 中 evaluate 的句柄，生命周期依附于 module。
public:
    RuntimeKernel() {
        rtc_check(nvrtcVersion(&major_version, &minor_version));
        cudaDeviceProp device; cuda_check(cudaGetDeviceProperties(&device, 0));
        // 本单 GPU 原型使用设备 0 的计算能力，生成对应 compute_XX 的 PTX。
        // PTX 随后交给 Driver 加载，在当前上下文中完成设备代码准备。
        const std::string architecture = "--gpu-architecture=compute_" + std::to_string(device.major) + std::to_string(device.minor);
        const char* options[]{"--std=c++17", architecture.c_str()};
        nvrtcProgram program;
        rtc_check(nvrtcCreateProgram(&program, kernel_source, "array_bridge.cu", 0, nullptr, nullptr));
        const nvrtcResult status = nvrtcCompileProgram(program, 2, options);
        // 编译失败时保留 NVRTC 的完整诊断信息，便于定位生成源码中的错误。
        if (status != NVRTC_SUCCESS) {
            std::size_t size; nvrtcGetProgramLogSize(program, &size);
            std::string log(size, '\0'); nvrtcGetProgramLog(program, log.data());
            nvrtcDestroyProgram(&program); throw std::runtime_error(log);
        }
        // PTX 长度包含结束符；先按报告长度分配，再取回文本并释放编译程序对象。
        std::size_t size; rtc_check(nvrtcGetPTXSize(program, &size));
        std::vector<char> ptx(size); rtc_check(nvrtcGetPTX(program, ptx.data()));
        rtc_check(nvrtcDestroyProgram(&program));
        driver_check(cuModuleLoadData(&module, ptx.data()));
        driver_check(cuModuleGetFunction(&function, module, "evaluate"));
    }
    int major_version{}, minor_version{};
    ~RuntimeKernel() { if(module) cuModuleUnload(module); }
    // 异步提交一次计算。positions 为读取视图，forces 为原位累加视图。
    // producer→consumer 的等待在调用方建立；返回时 kernel 可能仍在 stream 上执行。
    // 零 owned 原子直接返回，避免提交 grid 大小为 0 的 CUDA launch。
    void launch(BorrowedArray positions, BorrowedArray forces, double spring, cudaStream_t stream) {
        validate(positions); validate(forces);
        require(positions.owned == forces.owned, "Owned extents must agree");
        if (!positions.owned) return;
        // Driver 接收“各参数值在 Host 上的地址”。两个 view 作为结构体按值传入，
        // 其中 data 字段仍然是设备指针；arguments 自身只需活到 launch 调用结束。
        void* arguments[]{&positions, &forces, &spring};
        driver_check(cuLaunchKernel(function, (positions.owned+127)/128, 1, 1, 128, 1, 1,
                                   0, stream, arguments, nullptr));
    }
};

// 模拟 GPUMD 的上游计算：在 producer 流生成容易核对的 SoA 坐标与已有力 2.0。
// 坐标同时随 atom 和 component 变化，可以暴露 stride 颠倒或分量错位。
__global__ void prepare_soa(double* positions, double* forces, int capacity) {
    const int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if(atom < capacity) for(int component=0; component<3; ++component) {
        positions[atom+component*capacity] = 0.1*(atom+1)+0.3*component;
        forces[atom+component*capacity] = 2.0;
    }
}

// largest_error 汇总的是回写值相对解析值的最大绝对误差。
// check_value 另外用能量差分独立核对偏置力的负号与 spring 系数。
double largest_error = 0.0;
void check_value(double force, double coordinate, bool owned) {
    // owned 原子在原有力 2.0 上加偏置力；ghost 应保留初始化的 2.0。
    const double expected = 2.0 - (owned ? 1.7 * coordinate : 0.0);
    largest_error = std::max(largest_error, std::abs(force-expected));
    require(std::abs(force-expected) < 1e-12, "Force writeback or ghost extent mismatch");
    // 中心差分验证外部坐标到偏置力的符号和系数。
    if(owned) {
        const double step=1e-5;
        auto energy=[](double x){return 0.5*1.7*x*x;};
        const double numerical=-(energy(coordinate+step)-energy(coordinate-step))/(2*step);
        require(std::abs((force-2.0)-numerical)<2e-8, "Finite difference force mismatch");
    }
}

int main(int argc,char** argv) {
    // 先建立 MPI/Kokkos 运行环境，再创建其容器；结束时按相反顺序释放。
    MPI_Init(&argc,&argv);
    Kokkos::initialize(argc,argv);
    int exit_code=0;
    try {
        // 确保 Runtime 的当前设备上下文已建立，使 Driver 加载与容器使用同一上下文。
        cuda_check(cudaFree(nullptr));
        RuntimeKernel kernel;
        // 使用非默认流，并显式建立事件依赖，让正确性依赖于清楚的数据就绪关系。
        cudaStream_t producer,consumer;
        cudaEvent_t ready;
        cuda_check(cudaStreamCreateWithFlags(&producer,cudaStreamNonBlocking));
        cuda_check(cudaStreamCreateWithFlags(&consumer,cudaStreamNonBlocking));
        cuda_check(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming));
        {
            GPU_Vector<double> positions,forces;
            // 两个容量分别覆盖单 block 尾部和跨 block 尾部，并触发重新分配。
            // 每次 resize 后重新借用指针；最后三个原子用作 ghost 边界检查。
            for(const int capacity:{11,257}) {
                const int owned=capacity-3;
                positions.resize(3*capacity); forces.resize(3*capacity);
                auto x=borrow_soa(positions.data(),capacity,owned);
                auto f=borrow_soa(forces.data(),capacity,owned);
                require(x.data==positions.data() && f.data==forces.data(),"Borrowing copied data");
                prepare_soa<<<(capacity+127)/128,128,0,producer>>>(positions.data(),forces.data(),capacity);
                cuda_check(cudaGetLastError());
                // ready 记录 producer 中此前任务的完成点。consumer 等待它之后才能读取。
                // 这里的 stream wait 只建立 GPU 顺序，Host 仍可继续提交工作。
                cuda_check(cudaEventRecord(ready,producer));
                cuda_check(cudaStreamWaitEvent(consumer,ready));
                kernel.launch(x,f,1.7,consumer);
                // Host 校验前等待力回写结束，也使下一轮 resize 可以安全释放旧分配。
                cuda_check(cudaStreamSynchronize(consumer));
                std::vector<double> host(3*capacity); forces.copy_to_host(host.data());
                for(int atom=0;atom<capacity;++atom) for(int component=0;component<3;++component)
                    check_value(host[atom+component*capacity],0.1*(atom+1)+0.3*component,atom<owned);
            }
        }
        {
            // 与 AtomKokkos::k_x / k_f 完全相同的类型；实际 stride 由容器提供。
            DAT::tdual_x_array positions("coordinates",17);
            DAT::tdual_f_array forces("forces",17);
            auto x=borrow_kokkos(positions.d_view,14);
            auto f=borrow_kokkos(forces.d_view,14);
            require(x.data==positions.d_view.data(),"Kokkos pointer identity mismatch");
            auto device_x=positions.d_view; auto device_f=forces.d_view;
            // 将 Kokkos 的执行实例绑定到 producer，事件即可覆盖它提交的初始化任务。
            Kokkos::Cuda execution(producer);
            Kokkos::parallel_for("engine_producer",Kokkos::RangePolicy<Kokkos::Cuda>(execution,0,17),
                KOKKOS_LAMBDA(const int atom) {
                    for(int component=0;component<3;++component) {
                        device_x(atom,component)=0.1*(atom+1)+0.3*component;
                        device_f(atom,component)=2.0;
                    }
                });
            // modify_device 标记设备副本较新；标记本身与 CUDA 流的完成等待是两件事。
            positions.modify_device(); forces.modify_device();
            cuda_check(cudaEventRecord(ready,producer));
            cuda_check(cudaStreamWaitEvent(consumer,ready));
            kernel.launch(x,f,1.7,consumer);
            cuda_check(cudaStreamSynchronize(consumer));
            // 外部 NVRTC kernel 直接写了 d_view，完成后告知 DualView 设备端已更新，
            // 再通过 sync_host 得到最新的 Host 镜像用于校验。
            forces.modify_device(); forces.sync_host(); positions.sync_host();
            for(int atom=0;atom<17;++atom) for(int component=0;component<3;++component)
                check_value(forces.h_view(atom,component),positions.h_view(atom,component),atom<14);
            std::cout<<"Kokkos coordinate strides="<<x.atom_stride<<","<<x.component_stride
                     <<" force strides="<<f.atom_stride<<","<<f.component_stride<<"\n";
        }
        // 验证空任务可跳过，以及非空容量配空指针会被描述符检查拒绝。
        kernel.launch(borrow_soa(nullptr,0,0),borrow_soa(nullptr,0,0),1.7,consumer);
        bool rejected=false; try { borrow_soa(nullptr,4,4); } catch(const std::invalid_argument&) {rejected=true;}
        require(rejected,"Invalid view accepted");
        cuda_check(cudaEventDestroy(ready)); cuda_check(cudaStreamDestroy(producer)); cuda_check(cudaStreamDestroy(consumer));
        std::cout<<"{\"status\":\"pass\",\"nvrtc\":\""<<kernel.major_version<<"."<<kernel.minor_version
                 <<"\",\"max_force_error\":"<<largest_error
                 <<",\"containers\":[\"GPUMD GPU_Vector\",\"LAMMPS DAT::tdual_x_array/tdual_f_array\"]}\n";
    } catch(const std::exception& error) { std::cerr<<error.what()<<"\n"; exit_code=1; }
    Kokkos::finalize(); MPI_Finalize(); return exit_code;
}
