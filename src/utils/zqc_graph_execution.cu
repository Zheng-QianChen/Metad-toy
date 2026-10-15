#include "zqc_graph.cuh"
#include <cuda.h>
#include <nvrtc.h>
#include <stdexcept>
#include <chrono>

namespace {
void driver_check(CUresult status) {
    if (status == CUDA_SUCCESS) return;
    const char* message = nullptr;
    cuGetErrorString(status, &message);
    throw std::runtime_error(message ? message : "CUDA Driver error");
}
void rtc_check(nvrtcResult status) {
    if (status != NVRTC_SUCCESS) throw std::runtime_error(nvrtcGetErrorString(status));
}
}

struct GraphExecution::Impl {
    Graph& graph;
    std::vector<const Node*> topo;
    std::vector<Node*> outputs;
    std::string source;
    std::vector<Node*> inputs;
    std::vector<Node*> reductions;
    std::vector<std::unique_ptr<MetaD_zqc::GpuBuffer<double>>> device_outputs;
    std::vector<std::vector<double>> host_outputs;
    CUmodule module = nullptr;
    CUfunction fwd = nullptr, bwd = nullptr;
    CUcontext context = nullptr;
    int n = 0;
    bool forwarded = false;
    FILE* screen;
    FILE* logfile;

    void log(const std::string& message) const {
        FILE* files[] = {screen, logfile == screen ? nullptr : logfile};
        for (FILE* file : files) {
            if (!file) continue;
            std::fprintf(file, "[MetaD Graph] %s\n", message.c_str());
            std::fflush(file); // 编译前也刷新，便于辨认耗时发生在哪一阶段。
        }
    }

    explicit Impl(Graph& graph_, FILE* screen_, FILE* logfile_)
        : graph(graph_), screen(screen_), logfile(logfile_) {
        if (graph.topo.empty()) graph.build();
        topo = graph.topo;
        outputs = graph.outputs;
        source = graph.kernel_source;
        for (const Node* output : outputs)
            if (output->value_desc.domain != Domain::Global)
                throw std::runtime_error("GraphExecution: outputs must be Global scalars");
        for (const Node* node : topo)
            if (node->value_desc.domain == Domain::Global && (!node->val || !node->adj))
                throw std::runtime_error("GraphExecution: missing Host scalar storage");
        if (graph.kernel_outputs.empty()) return;

        Codegen cg;
        for (const Node* node : topo)
            if (node->value_desc.domain == Domain::PerAtom) cg.value(node);
        for (const auto& input : cg.inputs) inputs.push_back(const_cast<Node*>(input.first));
        if (inputs.empty()) throw std::runtime_error("GraphExecution: no PerAtom inputs");
        n = inputs.front()->n;
        for (Node* input : inputs)
            if (input->n != n) throw std::runtime_error("GraphExecution: input lengths differ");
        for (const Node* node : graph.kernel_seeds) {
            if (!node->is_reduction()) throw std::runtime_error("GraphExecution: expected a reduction boundary");
            reductions.push_back(const_cast<Node*>(node));
            device_outputs.push_back(std::make_unique<MetaD_zqc::GpuBuffer<double>>(n));
            host_outputs.emplace_back(n);
        }
        compile();
    }
    ~Impl() { if (module) cuModuleUnload(module); }

    void compile() {
        graph_check_cuda(cudaFree(nullptr));
        driver_check(cuCtxGetCurrent(&context));
        int device = 0;
        graph_check_cuda(cudaGetDevice(&device));
        cudaDeviceProp props{};
        graph_check_cuda(cudaGetDeviceProperties(&props, device));
        const std::string arch = "--gpu-architecture=sm_" + std::to_string(props.major) + std::to_string(props.minor);
        const char* options[] = {"--std=c++17", arch.c_str()};
        int major = 0, minor = 0;
        rtc_check(nvrtcVersion(&major, &minor));
        log("NVRTC begin: version=" + std::to_string(major) + "." + std::to_string(minor) +
            " " + arch + " source_bytes=" + std::to_string(source.size()) +
            " inputs=" + std::to_string(inputs.size()) + " reductions=" + std::to_string(reductions.size()));
        using Clock = std::chrono::steady_clock;
        const auto start = Clock::now();
        nvrtcProgram program = nullptr;
        rtc_check(nvrtcCreateProgram(&program, source.c_str(), "zqc_graph.cu", 0, nullptr, nullptr));
        try {
            const nvrtcResult status = nvrtcCompileProgram(program, 2, options);
            const auto compiled = Clock::now();
            size_t log_size = 0;
            rtc_check(nvrtcGetProgramLogSize(program, &log_size));
            std::string compiler_log(log_size, '\0');
            if (log_size) rtc_check(nvrtcGetProgramLog(program, compiler_log.data()));
            if (log_size > 1) log("NVRTC compiler log:\n" + compiler_log);
            if (status != NVRTC_SUCCESS)
                throw std::runtime_error("Graph NVRTC compilation failed: " + compiler_log);
            size_t size = 0;
            rtc_check(nvrtcGetCUBINSize(program, &size));
            std::vector<char> cubin(size);
            rtc_check(nvrtcGetCUBIN(program, cubin.data()));
            driver_check(cuModuleLoadData(&module, cubin.data()));
            driver_check(cuModuleGetFunction(&fwd, module, "cv_forward"));
            driver_check(cuModuleGetFunction(&bwd, module, "cv_backward"));
            rtc_check(nvrtcDestroyProgram(&program));
            const double compile_ms = std::chrono::duration<double, std::milli>(compiled-start).count();
            const double load_ms = std::chrono::duration<double, std::milli>(Clock::now()-compiled).count();
            log("NVRTC ready: compile_ms=" + std::to_string(compile_ms) +
                " load_ms=" + std::to_string(load_ms) + " cubin_bytes=" + std::to_string(size) +
                " kernels=cv_forward,cv_backward (reused across timesteps)");
        } catch (const std::exception& e) {
            log(std::string("NVRTC/module failed: ") + e.what());
            if (program) nvrtcDestroyProgram(&program);
            if (module) { cuModuleUnload(module); module = nullptr; }
            throw;
        }
    }
    void resize_inputs() {
        forwarded = false;
        if (graph.topo != topo || graph.outputs != outputs || graph.kernel_source != source)
            throw std::logic_error("GraphExecution: graph changed; rebuild the executor");
        const int count = inputs.empty() ? 0 : inputs.front()->n;
        if (count < 0) throw std::logic_error("GraphExecution: negative local size");
        for (Node* input : inputs) {
            if (input->n != count) throw std::logic_error("GraphExecution: input lengths differ");
            input->forward(); // 校验借用容量，并刷新迁移/扩容后的地址。
        }
        for (size_t k = 0; k < device_outputs.size(); ++k) {
            if (device_outputs[k]->capacity < static_cast<size_t>(count))
                device_outputs[k] = std::make_unique<MetaD_zqc::GpuBuffer<double>>(count);
            host_outputs[k].resize(count);
        }
        n = count; // 编译时不展开原子数；kernel 的 n 是运行时参数。
    }
    void check_graph() const {
        if (graph.topo != topo || graph.outputs != outputs || graph.kernel_source != source)
            throw std::logic_error("GraphExecution: graph changed; rebuild the executor");
        for (const Node* input : inputs)
            if (input->n != n) throw std::logic_error("GraphExecution: local size changed; call resize_inputs()");
        if (module) {
            CUcontext current = nullptr;
            driver_check(cuCtxGetCurrent(&current));
            if (current != context) throw std::logic_error("GraphExecution: CUDA context changed");
        }
    }
    void launch(CUfunction function, std::vector<void*>& args) {
        if (!n) return; // 空 rank 仍执行后面的 Host/MPI 部分。
        driver_check(cuLaunchKernel(function, (n+255)/256, 1, 1, 256, 1, 1,
                                   0, nullptr, args.data(), nullptr));
        driver_check(cuCtxSynchronize());
    }
    std::vector<double> forward() {
        forwarded = false;
        check_graph();
        std::vector<void*> args{&n};
        for (Node* input : inputs) {
            input->forward(); // 外部 GpuBuffer 可能更换了地址。
            args.push_back(&input->val);
        }
        for (auto& output : device_outputs) args.push_back(&output->ptr);
        if (module) launch(fwd, args);
        for (size_t k = 0; k < reductions.size(); ++k)
            if (n) graph_check_cuda(cudaMemcpy(host_outputs[k].data(), device_outputs[k]->ptr,
                                               n*sizeof(double), cudaMemcpyDeviceToHost));

        // 拓扑顺序保证 MPI 规约完成后才使用全局标量；同一规约节点只执行一次。
        size_t reduction = 0;
        for (const Node* item : topo) {
            Node* node = const_cast<Node*>(item);
            if (node->value_desc.domain != Domain::Global) continue;
            if (node->is_reduction()) {
                if (reduction >= reductions.size() || reductions[reduction] != node)
                    throw std::logic_error("GraphExecution: reduction order mismatch");
                node->reduce_host(host_outputs[reduction].data(), n);
                ++reduction;
            } else {
                node->forward();
            }
        }
        std::vector<double> values;
        for (const Node* node : outputs) values.push_back(*node->val);
        forwarded = true;
        return values;
    }
    void backward(const std::vector<double>& seeds) {
        check_graph();
        if (!forwarded) throw std::logic_error("GraphExecution: forward is required before backward");
        if (seeds.size() != outputs.size()) throw std::runtime_error("GraphExecution: output seed count mismatch");
        for (const Node* node : topo)
            if (node->value_desc.domain == Domain::Global) *node->adj = 0.0;
        for (size_t k = 0; k < outputs.size(); ++k) *outputs[k]->adj += seeds[k];
        for (auto it = topo.rbegin(); it != topo.rend(); ++it)
            if ((*it)->value_desc.domain == Domain::Global && !(*it)->is_reduction())
                const_cast<Node*>(*it)->backward();

        std::vector<double> expanded;
        for (Node* node : reductions) expanded.push_back(node->reduction_seed());
        std::vector<void*> args{&n};
        for (Node* input : inputs) args.push_back(&input->val);
        for (double& seed : expanded) args.push_back(&seed);
        for (Node* input : inputs) args.push_back(&input->adj);
        if (module) launch(bwd, args);
        for (Node* input : inputs) input->backward();
    }
};

GraphExecution::GraphExecution(Graph& graph, FILE* screen, FILE* logfile)
    : impl_(std::make_unique<Impl>(graph, screen, logfile)) {}
void GraphExecution::resize_inputs() { impl_->resize_inputs(); }
GraphExecution::~GraphExecution() = default;
std::vector<double> GraphExecution::forward() { return impl_->forward(); }
void GraphExecution::backward(const std::vector<double>& seeds) { impl_->backward(seeds); }
