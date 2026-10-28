// 用非阻塞流模拟尚未完成写入的 MD，验证 View 依赖、Leaf 暂存和流隔离。
#include "zqc_cuda_execution.cuh"
#include "zqc_graph.cuh"
#include "zqc_cv_assembly.cuh"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <thread>
#include <iostream>

using namespace MetaD_zqc;
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
__global__ void write_value(double* destination, int count, int stride, double value) {
    for (int index=threadIdx.x; index<count; index+=blockDim.x)
        destination[index*stride]=value+index;
}
__global__ void copy_values(ArrayView<const double> source, double* destination) {
    for (int index=threadIdx.x; index<source.item_count; index+=blockDim.x)
        destination[index]=source(index);
}

// CPU gate 只阻塞 MD 流；watchdog 保证错误的全设备等待也会退出并报失败。
// callback 内仅使用 Host 同步原语，不调用 CUDA。正常路径由主线程主动释放。
struct ProducerGate {
    cudaStream_t producer;
    std::mutex mutex;
    std::condition_variable condition;
    bool released=false;
    std::atomic<bool> timed_out{false};
    std::thread watchdog;
    explicit ProducerGate(cudaStream_t stream) : producer(stream), watchdog([this] {
        std::unique_lock<std::mutex> lock(mutex);
        if (!condition.wait_for(lock,std::chrono::seconds(5),[this]{return released;})) {
            timed_out=true;released=true;condition.notify_all();
        }
    }) {}
    void release() {
        std::lock_guard<std::mutex> lock(mutex);released=true;condition.notify_all();
    }
    ~ProducerGate() { release();cudaStreamSynchronize(producer);watchdog.join(); }
    static void CUDART_CB wait(void* context) {
        auto& gate=*static_cast<ProducerGate*>(context);
        std::unique_lock<std::mutex> lock(gate.mutex);
        gate.condition.wait(lock,[&]{return gate.released;});
    }
};

void verify_stream_isolation() {
    CudaExecutionStream md,consumer,independent;
    GpuBuffer<double> input(4),output(4),scratch(4);
    std::vector<double> host{1,2,3,4},downloaded(4);
    // 模块/内核的首次加载发生在 gate 之前，测量只覆盖稳定路径。
    write_value<<<1,32,0,md.get()>>>(input.ptr,4,1,-1);
    md.synchronize();
    copy_values<<<1,32,0,consumer.get()>>>(input.view(4),output.ptr);
    consumer.synchronize();
    ProducerGate gate(md.get());
    execution_check(cudaLaunchHostFunc(md.get(),ProducerGate::wait,&gate));
    write_value<<<1,32,0,md.get()>>>(input.ptr,4,1,7);
    auto writable=input.view(4);
    writable.readiness=md.ready();
    ArrayView<const double> source=writable; // const 转换必须保留 ready hook。
    source.wait_on(consumer.get());
    copy_values<<<1,32,0,consumer.get()>>>(source,output.ptr);
    const auto completion=consumer.ready();
    scratch.clear_async(independent.get());
    scratch.upload_from(host.data(),host.size(),independent.get());
    scratch.download_to(downloaded.data(),downloaded.size(),independent.get());
    const bool isolated=!gate.timed_out;
    const auto pending=cudaStreamQuery(consumer.get());
    gate.release();
    completion.wait_host();
    output.download_to(downloaded.data(),downloaded.size(),consumer.get());
    require(isolated,"GpuBuffer helper waited for unrelated MD stream");
    require(pending==cudaErrorNotReady,"View failed to hold consumer behind MD producer");
    for(int index=0;index<4;++index) require(downloaded[index]==7+index,"View consumer read stale MD values");
}

void verify_graph_dependencies() {
    CudaExecutionStream producer,assembly_stream;
    GpuBuffer<double> values(8),gradients(8),assembled(8);
    std::vector<int> ids{0,1,2,3};
    CVAtomLayout layout;
    layout.rebuild(4,0,{{all_atoms,4,{ids.data(),4,1,1,1,4,MemorySpace::Host}}});
    ArrayView<double> input{values.ptr,4,1,2,1,8,MemorySpace::Cuda};
    ArrayView<double> staged{assembled.ptr,4,1,2,1,8,MemorySpace::Cuda};
    ArrayView<double> gradient{gradients.ptr,4,1,2,1,8,MemorySpace::Cuda};
    Leaf leaf(staged,gradient);
    Graph graph({{"a",&leaf}});
    graph.parse_expr("AVE(a*a)");graph.build();
    GraphExecution execution(graph);
    // 预热装配、Leaf gather/scatter 和融合 kernel，排除驱动首次加载的同步。
    layout.group(all_atoms)->assemble_values(input,staged,assembly_stream.get());
    assembly_stream.synchronize();
    execution.forward();execution.backward({1.0});
    // 重复两轮，覆盖事件复用，以及没有扩容/首次 NVRTC 隐式同步的路径。
    for(int iteration=0;iteration<2;++iteration) {
        ProducerGate gate(producer.get());
        execution_check(cudaLaunchHostFunc(producer.get(),ProducerGate::wait,&gate));
        write_value<<<1,32,0,producer.get()>>>(values.ptr,4,2,2+iteration);
        write_value<<<1,32,0,producer.get()>>>(gradients.ptr,4,2,99);
        input.readiness=gradient.readiness=producer.ready();
        layout.group(all_atoms)->assemble_values(input,staged,assembly_stream.get());
        staged.readiness=assembly_stream.ready();
        leaf.bind(staged,gradient);
        const bool asynchronous=!gate.timed_out;
        gate.release();
        require(asynchronous,"assembly/Leaf binding blocked on the MD producer");
        const auto output=execution.forward();
        double expected=0;
        for(int index=0;index<4;++index) expected+=(2+iteration+index)*(2+iteration+index)/4.0;
        require(std::abs(output[0]-expected)<1e-12,"Graph forward lost assembly/MD dependency");
        execution.backward({1.0});
        std::vector<double> host(8);gradients.download_to(host.data(),host.size());
        for(int index=0;index<4;++index)
            require(std::abs(host[2*index]-(2+iteration+index)/2.0)<1e-12,"Graph gradient/strided scatter race");
    }
}

// 同一分配反复装配后直接执行 Graph，覆盖没有 Leaf::bind 的常用路径。
// gate 将装配推迟到 forward 已开始以后，使漏掉的事件等待稳定暴露为旧值。
void verify_buffer_reassembly() {
    CudaExecutionStream assembly_stream;
    GpuBuffer<double> raw(3),values(3),gradients(3),compact_gradients(3);
    std::vector<int> ids{0,1,2};
    CVAtomLayout layout;
    layout.rebuild(3,0,{{all_atoms,3,{ids.data(),3,1,1,1,3,MemorySpace::Host}}});
    auto assembly=layout.group(all_atoms);
    Leaf leaf(values,gradients,3);
    Graph graph({{"x",&leaf}});
    graph.parse_expr("AVE(x*x)");
    GraphExecution execution(graph);
    // 预热 CUDA 模块、装配/梯度 kernel 和 NVRTC，排除首次加载的隐式等待。
    assembly->assemble_values(raw,values,assembly_stream.get());
    assembly_stream.synchronize();
    execution.forward();execution.backward({1.0});
    assembly->gather_adjoint(gradients,compact_gradients,assembly_stream.get());
    assembly_stream.synchronize();
    for(int iteration=0;iteration<4;++iteration) {
        const std::vector<double> input{double(iteration+2),double(iteration+3),double(iteration+4)};
        raw.upload_from(input.data(),input.size());
        // 交替使用默认流和非阻塞流，重复记录同一缓冲区的完成事件。
        const cudaStream_t producer=iteration%2 ? assembly_stream.get() : nullptr;
        ProducerGate gate(producer);
        execution_check(cudaLaunchHostFunc(producer,ProducerGate::wait,&gate));
        assembly->assemble_values(raw,values,producer);
        if(iteration==2) leaf.bind(values.view(3),gradients.view(3));
        const bool asynchronous=!gate.timed_out;
        auto release=std::async(std::launch::async,[&gate] {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));gate.release();
        });
        const auto output=execution.forward();
        release.get();
        require(asynchronous,"GpuBuffer assembly/rebind blocked on the producer");
        double expected=0;
        for(double value:input) expected+=value*value/3.0;
        require(std::abs(output[0]-expected)<1e-12,"GpuBuffer reassembly lost readiness before Graph forward");
        execution.backward({1.0});

        // 反向回收同样是异步写入；在另一条流下载时须等待回收完成。
        ProducerGate gradient_gate(assembly_stream.get());
        execution_check(cudaLaunchHostFunc(assembly_stream.get(),ProducerGate::wait,&gradient_gate));
        assembly->gather_adjoint(gradients,compact_gradients,assembly_stream.get());
        const bool gradient_asynchronous=!gradient_gate.timed_out;
        auto release_gradient=std::async(std::launch::async,[&gradient_gate] {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));gradient_gate.release();
        });
        std::vector<double> downloaded(3);
        compact_gradients.download_to(downloaded.data(),downloaded.size());
        release_gradient.get();
        require(gradient_asynchronous,"GpuBuffer adjoint gather blocked on the producer");
        for(int index=0;index<3;++index)
            require(std::abs(downloaded[index]-2*input[index]/3.0)<1e-12,"GpuBuffer adjoint download lost readiness");
    }
}

// 清零或自定义 kernel 写入 -> CUB 扫描 -> Host 下载，分别使用三条流。
// 覆盖缓冲区既作为异步输出、又作为下一个算子的输入时的依赖传递。
void verify_buffer_write_chain() {
    CudaExecutionStream producer,scanner;
    GpuBuffer<double> input(3),offsets(4);
    std::vector<double> downloaded(4);
    write_value<<<1,32,0,producer.get()>>>(input.ptr,3,1,1);
    input.record_write(producer.get());
    input.scan_to(offsets,3,scanner.get());
    offsets.download_to(downloaded.data(),downloaded.size());
    for(int iteration=0;iteration<2;++iteration) {
        ProducerGate gate(producer.get());
        execution_check(cudaLaunchHostFunc(producer.get(),ProducerGate::wait,&gate));
        if(iteration==0) {
            input.clear_async(producer.get());
        } else {
            input.wait_on(producer.get());
            write_value<<<1,32,0,producer.get()>>>(input.ptr,3,1,4);
            execution_check(cudaGetLastError());
            input.record_write(producer.get());
        }
        input.scan_to(offsets,3,scanner.get());
        const bool asynchronous=!gate.timed_out;
        auto release=std::async(std::launch::async,[&gate] {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));gate.release();
        });
        offsets.download_to(downloaded.data(),downloaded.size());
        release.get();
        require(asynchronous,"buffer write/scan blocked on the producer");
        const std::vector<double> expected=iteration==0
            ? std::vector<double>{0,0,0,0} : std::vector<double>{0,4,9,15};
        require(downloaded==expected,"buffer write/scan/download lost readiness across streams");
    }
}
}
int main() {
    try {
        verify_stream_isolation();verify_buffer_reassembly();verify_buffer_write_chain();verify_graph_dependencies();
        std::cout<<"PASS: producer wait, buffer reassembly/rebind, adjoint gather, independent streams and strided Graph\n";
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; }
}
