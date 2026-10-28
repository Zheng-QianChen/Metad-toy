#include "zqc_graph.cuh"
#include "zqc_cuda_execution.cuh"
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
    MetaD_zqc::CudaExecutionStream execution; // 融合前后向、Leaf 暂存和规约下载按流排序。
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
    MetaD_zqc::Logger logger;

    void log(const std::string& message, MetaD_zqc::LogLevel level = MetaD_zqc::LogLevel::Info) const {
        logger.printf(level, "Graph", __FILE__, __LINE__, "%s", message.c_str());
    }

    explicit Impl(Graph& graph_, MetaD_zqc::Logger logger_)
        : graph(graph_), logger(std::move(logger_)) {
        if (graph.topo.empty()) graph.build();
        topo = graph.topo;
        outputs = graph.outputs;
        source = graph.kernel_source;
        for (const Node* node : topo)
            if (node->value_desc.domain == ValueDomain::Global && (!node->val || !node->adj))
                throw std::runtime_error("GraphExecution: missing Host scalar storage");
        if (graph.kernel_outputs.empty()) return;

        Codegen cg;
        for (const Node* node : topo)
            if (node->value_desc.domain == ValueDomain::PerAtom) cg.value(node);
        for (const auto& input : cg.inputs) inputs.push_back(const_cast<Node*>(input.first));
        if (inputs.empty()) throw std::runtime_error("GraphExecution: no PerAtom inputs");
        n = inputs.front()->n;
        for (Node* input : inputs)
            if (input->n != n) throw std::runtime_error("GraphExecution: input lengths differ");
        for (const Node* node : graph.kernel_seeds) {
            reductions.push_back(const_cast<Node*>(node));
            device_outputs.push_back(std::make_unique<MetaD_zqc::GpuBuffer<double>>(n));
            host_outputs.emplace_back(node->is_reduction() ? n : 0);
        }
        compile();
    }
    ~Impl() { cudaStreamSynchronize(execution.get()); if (module) cuModuleUnload(module); }

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
            log(std::string("NVRTC/module failed: ") + e.what(), MetaD_zqc::LogLevel::Error);
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
            static_cast<Leaf*>(input)->refresh_buffers(); // 校验并刷新借用地址；数据在 forward 时读取。
        }
        for (size_t k = 0; k < device_outputs.size(); ++k) {
            if (device_outputs[k]->capacity < static_cast<size_t>(count))
                device_outputs[k] = std::make_unique<MetaD_zqc::GpuBuffer<double>>(count);
            host_outputs[k].resize(reductions[k]->is_reduction() ? count : 0);
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
                                   0, execution.get(), args.data(), nullptr));
    }
    std::vector<double> forward() {
        forwarded = false;
        check_graph();
        std::vector<void*> args{&n};
        for (Node* input : inputs) {
            static_cast<Leaf*>(input)->forward_on(execution.get()); // 等待 CV/外部生产者。
            args.push_back(&input->val);
        }
        for (auto& output : device_outputs) args.push_back(&output->ptr);
        if (module) launch(fwd, args);
        for (auto& output : device_outputs) output->record_write(execution.get());
        for (size_t k = 0; k < reductions.size(); ++k)
            if (n && reductions[k]->is_reduction()) graph_check_cuda(cudaMemcpyAsync(host_outputs[k].data(), device_outputs[k]->ptr,
                                               n*sizeof(double), cudaMemcpyDeviceToHost, execution.get()));
        // Host 规约和 MPI 必须在所有本地图输出下载完成后开始。
        execution.synchronize();

        // 拓扑顺序保证 MPI 规约完成后才使用全局标量；同一规约节点只执行一次。
        size_t reduction = 0;
        for (const Node* item : topo) {
            Node* node = const_cast<Node*>(item);
            if (node->value_desc.domain != ValueDomain::Global) continue;
            if (node->is_reduction()) {
                while (reduction < reductions.size() && !reductions[reduction]->is_reduction()) ++reduction;
                if (reduction >= reductions.size() || reductions[reduction] != node)
                    throw std::logic_error("GraphExecution: reduction order mismatch");
                node->reduce_host(host_outputs[reduction].data(), n);
                ++reduction;
            } else {
                node->forward();
            }
        }
        std::vector<double> values;
        for (const Node* node : outputs)
            if (node->value_desc.domain == ValueDomain::Global) values.push_back(*node->val);
        forwarded = true;
        return values;
    }
    MetaD_zqc::ArrayView<const double> per_atom_output(size_t index) const {
        check_graph();
        if (!forwarded || index >= outputs.size() ||
            outputs[index]->value_desc.domain != ValueDomain::PerAtom)
            throw std::logic_error("GraphExecution: requested PerAtom output is unavailable");
        size_t ordinal = 0;
        for (size_t k = 0; k < index; ++k)
            if (outputs[k]->value_desc.domain == ValueDomain::PerAtom) ++ordinal;
        for (size_t boundary = 0; boundary < reductions.size(); ++boundary)
            if (!reductions[boundary]->is_reduction() && ordinal-- == 0)
                return device_outputs[boundary]->view(n);
        throw std::logic_error("GraphExecution: missing PerAtom boundary");
    }
    void backward(const std::vector<double>& seeds,
                  const std::vector<MetaD_zqc::ArrayView<const double>>& atom_seeds) {
        check_graph();
        if (!forwarded) throw std::logic_error("GraphExecution: forward is required before backward");
        size_t global_count = 0;
        for (const Node* output : outputs)
            if (output->value_desc.domain == ValueDomain::Global) ++global_count;
        if (seeds.size() != global_count || atom_seeds.size() != outputs.size()-global_count)
            throw std::runtime_error("GraphExecution: output seed count mismatch");
        for (const auto& seed : atom_seeds) {
            seed.validate();
            if (seed.item_count != static_cast<size_t>(n) || seed.component_count != 1 ||
                seed.memory != MetaD_zqc::MemorySpace::Cuda || seed.item_stride != 1)
                throw std::invalid_argument("GraphExecution: atom seeds require contiguous CUDA scalars");
            seed.wait_on(execution.get());
        }
        for (const Node* node : topo)
            if (node->value_desc.domain == ValueDomain::Global) *node->adj = 0.0;
        size_t global_index = 0;
        for (Node* output : outputs)
            if (output->value_desc.domain == ValueDomain::Global) *output->adj += seeds[global_index++];
        for (auto it = topo.rbegin(); it != topo.rend(); ++it)
            if ((*it)->value_desc.domain == ValueDomain::Global && !(*it)->is_reduction())
                const_cast<Node*>(*it)->backward();

        // Kernel 边界按生成顺序混合标量种子与数组种子；存储稳定到 launch 返回。
        std::vector<double> expanded(reductions.size());
        std::vector<const double*> pointers(reductions.size());
        std::vector<void*> args{&n};
        for (Node* input : inputs) args.push_back(&input->val);
        size_t atom_index = 0;
        for (size_t k = 0; k < reductions.size(); ++k) {
            if (reductions[k]->is_reduction()) {
                expanded[k] = reductions[k]->reduction_seed();
                args.push_back(&expanded[k]);
            } else {
                pointers[k] = atom_seeds[atom_index++].data;
                args.push_back(&pointers[k]);
            }
        }
        for (Node* input : inputs) {
            static_cast<Leaf*>(input)->wait_gradient_on(execution.get());
            args.push_back(&input->adj);
        }
        if (module) launch(bwd, args);
        for (Node* input : inputs) static_cast<Leaf*>(input)->backward_on(execution.get());
        // 保持公开 backward 返回时外部梯度可读的约定，只等待融合流。
        execution.synchronize();
    }
};

GraphExecution::GraphExecution(Graph& graph, MetaD_zqc::Logger logger)
    : impl_(std::make_unique<Impl>(graph, std::move(logger))) {}
void GraphExecution::resize_inputs() { impl_->resize_inputs(); }
GraphExecution::~GraphExecution() = default;
std::vector<double> GraphExecution::forward() { return impl_->forward(); }
void GraphExecution::backward(const std::vector<double>& seeds,
        const std::vector<MetaD_zqc::ArrayView<const double>>& atom_seeds) {
    impl_->backward(seeds, atom_seeds);
}
MetaD_zqc::ArrayView<const double> GraphExecution::per_atom_output(size_t index) const {
    return impl_->per_atom_output(index);
}
