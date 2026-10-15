#include "zqc_graph.cuh"

#include <cuda.h>
#include <nvrtc.h>

// Experimental smoke test: emit, compile and load the fused kernels.
__global__ void graph_sample_inputs(double* a, double* b, double* c, double* d, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = i + 1.0;
    b[i] = i + 2.0;
    c[i] = i + 4.0;
    d[i] = (i + 1.0) / 10.0;
}

int main(){
    nvrtcProgram prog = nullptr;
    CUmodule module = nullptr;
    auto check_nvrtc = [](nvrtcResult status) {
        if (status != NVRTC_SUCCESS)
            throw std::runtime_error(nvrtcGetErrorString(status));
    };
    auto check_driver = [](CUresult status) {
        if (status != CUDA_SUCCESS) {
            const char* message = nullptr;
            cuGetErrorString(status, &message);
            throw std::runtime_error(message ? message : "CUDA Driver error");
        }
    };

    try {
        const int device = 0;
        // 先选择设备，再创建 Leaf 的显存；Runtime/Driver 共用 primary context。
        graph_check_cuda(cudaSetDevice(device));
        graph_check_cuda(cudaFree(nullptr));

        constexpr int count = 3;
        GpuBuffer<double> cv1_values(count), cv2_values(count);
        GpuBuffer<double> cv3_values(count), cv4_values(count);
        graph_sample_inputs<<<1, 32>>>(cv1_values.ptr, cv2_values.ptr,
                                     cv3_values.ptr, cv4_values.ptr, count);
        sync_kernel();
        Leaf cv1(cv1_values, count, {Domain::PerAtom, 1});
        Leaf cv2(cv2_values, count, {Domain::PerAtom, 1});
        Leaf cv3(cv3_values, count, {Domain::PerAtom, 1});
        Leaf cv4(cv4_values, count, {Domain::PerAtom, 1});
        const Graph::Symbols symbols{
            {"v1", &cv1}, {"v2", &cv2}, {"v3", &cv3}, {"v4", &cv4}
        };
        const Graph::Switches switches{
            {"gate", {MetaD_zqc::FERMI, 2.0, 0.0, 4.0, 0, 0}}
        };
        Graph graph(symbols, switches);
        graph.parse_expr("(v1 + v2 * v3 - exp(v4)).MEAN");
        graph.parse_expr("(v1 * sw_func(v1, gate)).MEAN");
        // build 内部：嵌入的设备函数文本 + 前向 kernel + 反向 kernel。
        graph.build();
        graph.print_code();

        cudaDeviceProp prop{};
        graph_check_cuda(cudaGetDeviceProperties(&prop, device));
        const std::string arch = "--gpu-architecture=sm_" +
            std::to_string(prop.major) + std::to_string(prop.minor);
        const char* opts[] = {arch.c_str(), "--std=c++17"};
        check_nvrtc(nvrtcCreateProgram(
            &prog, graph.kernel_source.c_str(), "cv.cu", 0, nullptr, nullptr));

        const auto compile_status = nvrtcCompileProgram(prog, 2, opts);
        size_t log_size = 0;
        check_nvrtc(nvrtcGetProgramLogSize(prog, &log_size));
        if (log_size > 1) {
            std::string log(log_size, '\0');
            check_nvrtc(nvrtcGetProgramLog(prog, log.data()));
            std::cerr << log.c_str();
        }
        check_nvrtc(compile_status);

        size_t cubin_size = 0;
        check_nvrtc(nvrtcGetCUBINSize(prog, &cubin_size));
        std::vector<char> cubin(cubin_size);
        check_nvrtc(nvrtcGetCUBIN(prog, cubin.data()));
        check_nvrtc(nvrtcDestroyProgram(&prog));

        CUfunction fwd, bwd;
        check_driver(cuModuleLoadData(&module, cubin.data()));
        check_driver(cuModuleGetFunction(&fwd, module, "cv_forward"));
        check_driver(cuModuleGetFunction(&bwd, module, "cv_backward"));
        check_driver(cuModuleUnload(module));
        module = nullptr;
        std::cerr << "NVRTC PASS: embedded " << graph_device_source().size()
                  << " bytes, cubin " << cubin_size
                  << " bytes; cv_forward/cv_backward loaded\n";
        return 0;
    }
    catch (const std::exception& error) {
        if (prog) nvrtcDestroyProgram(&prog);
        if (module) cuModuleUnload(module);
        std::cerr << "Graph smoke test failed: " << error.what() << '\n';
        return 1;
    }
}
