// 测量 Leaf 非连续适配的实际代价：同一数学图，改变输入/梯度的物理 stride。
#include "zqc_graph.cuh"
#include "zqc_cuda_execution.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>

using namespace MetaD_zqc;
using Clock = std::chrono::steady_clock;

struct Layout {
    const char* name;
    int input_stride, gradient_stride;
};

void measure(int count, int iterations, int repeat, Layout layout) {
    GpuBuffer<double> values(static_cast<size_t>(count)*layout.input_stride);
    GpuBuffer<double> gradients(static_cast<size_t>(count)*layout.gradient_stride);
    std::vector<double> host_values(values.capacity, -77.0);
    double expected = 0;
    for(int index=0;index<count;++index) {
        const double value=.1+double(index%37)/100;
        host_values[static_cast<size_t>(index)*layout.input_stride]=value;
        expected+=(value*value+.3*value)/count;
    }
    values.upload_from(host_values.data(),host_values.size());
    ArrayView<const double> input{values.ptr,static_cast<size_t>(count),1,
        static_cast<size_t>(layout.input_stride),1,values.capacity,MemorySpace::Cuda};
    ArrayView<double> gradient{gradients.ptr,static_cast<size_t>(count),1,
        static_cast<size_t>(layout.gradient_stride),1,gradients.capacity,MemorySpace::Cuda};
    Leaf leaf(input,gradient);
    Graph graph({{"x",&leaf}});
    graph.parse_expr("AVE(x*x+linear(x,0.3,0))");
    const auto setup_start=Clock::now();
    GraphExecution execution(graph);
    const double actual=execution.forward()[0];
    execution.backward({1.0});
    const double setup_ms=std::chrono::duration<double,std::milli>(Clock::now()-setup_start).count();
    std::vector<double> host_gradients(gradients.capacity);
    gradients.download_to(host_gradients.data(),host_gradients.size());
    if(std::abs(actual-expected)>1e-10) throw std::runtime_error("layout forward differs");
    for(int index=0;index<count;++index) {
        const double value=host_values[static_cast<size_t>(index)*layout.input_stride];
        const double expected_gradient=(2*value+.3)/count;
        if(std::abs(host_gradients[static_cast<size_t>(index)*layout.gradient_stride]-expected_gradient)>1e-12)
            throw std::runtime_error("layout gradient differs");
        for(int padding=1;padding<layout.gradient_stride;++padding)
            if(host_gradients[static_cast<size_t>(index)*layout.gradient_stride+padding]!=0)
                throw std::runtime_error("layout scatter overwrote padding");
    }
    for(int warmup=0;warmup<50;++warmup) { execution.forward();execution.backward({1.0}); }
    const auto start=Clock::now();
    for(int iteration=0;iteration<iterations;++iteration) { execution.forward();execution.backward({1.0}); }
    const double graph_us=std::chrono::duration<double,std::micro>(Clock::now()-start).count()/iterations;

    // 单独排入适配操作，记录其设备时间区间；连续布局只有 Host 元数据处理。
    // 该区间含提交间隙，故它是适配批次的摊销设备时间，不声称是 kernel 独占时间。
    CudaExecutionStream stream;
    cudaEvent_t begin,end;
    execution_check(cudaEventCreate(&begin));execution_check(cudaEventCreate(&end));
    leaf.forward_on(stream.get());leaf.backward_on(stream.get());stream.synchronize();
    execution_check(cudaEventRecord(begin,stream.get()));
    for(int iteration=0;iteration<iterations;++iteration) {
        leaf.forward_on(stream.get());leaf.backward_on(stream.get());
    }
    execution_check(cudaEventRecord(end,stream.get()));execution_check(cudaEventSynchronize(end));
    float elapsed_ms=0;execution_check(cudaEventElapsedTime(&elapsed_ms,begin,end));
    execution_check(cudaEventDestroy(begin));execution_check(cudaEventDestroy(end));
    std::cout<<count<<','<<repeat<<','<<layout.name<<','<<iterations<<','<<std::setprecision(12)
             <<graph_us<<','<<1000.0*elapsed_ms/iterations<<','<<setup_ms<<'\n';
}

int main(int argc,char** argv) {
    try {
        const int iterations=argc>1 ? std::stoi(argv[1]) : 2000;
        if(iterations<=0) throw std::invalid_argument("iterations must be positive");
        const std::vector<Layout> layouts{{"contiguous",1,1},{"input_stride3",3,1},
                                         {"both_stride3",3,3},{"both_stride8",8,8}};
        std::mt19937 randomizer(20261022);
        std::cout<<"items,repeat,layout,iterations,graph_forward_backward_us,staging_device_interval_us,setup_ms\n";
        for(int count:{256,16384,262144}) {
            for(int repeat=0;repeat<3;++repeat) {
                auto order=layouts;std::shuffle(order.begin(),order.end(),randomizer);
                for(const auto& layout:order) measure(count,iterations,repeat,layout);
            }
        }
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; }
}
