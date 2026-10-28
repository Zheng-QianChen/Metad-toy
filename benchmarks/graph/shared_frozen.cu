#include "zqc_graph.cuh"
#include "zqc_shared_bias.cuh"
#include "zqc_cuda_execution.cuh"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace MetaD_zqc;
void close(double actual,double expected,const char* label) {
    if (std::abs(actual-expected)>2e-12) throw std::runtime_error(label);
}
template<class T> void put(GpuBuffer<T>& buffer,const std::vector<T>& values,cudaStream_t stream=0) {
    buffer.grow_to(values.size(),__FILE__,__LINE__);
    if (!values.empty()) {
        if (stream) graph_check_cuda(cudaMemcpyAsync(buffer.ptr,values.data(),values.size()*sizeof(T),
                                                    cudaMemcpyHostToDevice,stream));
        else graph_check_cuda(cudaMemcpy(buffer.ptr,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice));
    }
    buffer.record_write(stream);
}
std::vector<double> get(ArrayView<const double> values) {
    values.wait_host(); std::vector<double> host(values.item_count);
    if (!host.empty()) graph_check_cuda(cudaMemcpy(host.data(),values.data,host.size()*sizeof(double),cudaMemcpyDeviceToHost));
    return host;
}
double potential(double q) { return .3*q*q-.4*q+.2; }
double derivative(double q) { return .6*q-.4; }
int main() try {
    const char* path="shared-frozen-unit.table";
    {
        std::ofstream file(path);
        file<<"# METAD_TOY_SHARED_FROZEN 1\n";
        for (int q=0;q<=3;++q) file<<q<<" "<<potential(q)<<" "<<derivative(q)<<"\n";
    }
    FrozenSharedBias bias(path);
    GpuBuffer<double> input,gradient,extra;
    GpuBuffer<int> slots;
    CudaExecutionStream producer;
    std::vector<double> x={-.7,.2,1.3}, extra_values(3,.2);
    put(input,x,producer.get()); put(extra,extra_values,producer.get());
    put(slots,std::vector<int>{0,2});
    gradient.grow_to(3,__FILE__,__LINE__);
    Leaf leaf(input,gradient,3);
    Mul square(&leaf,&leaf); AVE mean(&square);
    Graph graph(std::vector<Node*>{&square,&mean,&square});
    graph.build(); GraphExecution execution(graph);
    for (int repeat=0;repeat<8;++repeat) {
        x[0]=-.7+.01*repeat;
        put(input,x,producer.get());
        const auto scalar=execution.forward();
        close(scalar.at(0),(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])/3,"global output");
        const auto result=bias.evaluate(execution.per_atom_output(0),slots.view(2));
        close(result.energy,(potential(x[0]*x[0])+potential(x[2]*x[2]))/2,"selected mean energy");
        if (result.count!=2) throw std::runtime_error("count");
        const auto seed=get(result.gradient);
        close(seed[1],0,"padding seed");
        execution.backward({.7},{result.gradient,extra.view(3)});
        const auto actual=get(gradient.view(3));
        for (int i=0;i<3;++i) close(actual[i],2*x[i]*(seed[i]+.7/3+.2),"mixed reverse seeds");
    }
    // 已编译 kernel 在迁移后调整本地长度，混合输出保持同一拓扑。
    leaf.n=1;
    execution.resize_inputs();
    put(input,std::vector<double>{.5});
    put(extra,std::vector<double>{.2});
    execution.forward();
    execution.backward({.7},{extra.view(1),extra.view(1)});
    close(get(gradient.view(1))[0],1.1,"resized mixed seeds");
    // 非连续输入由求值器读取真实 stride；端点使用线性延拓，导数来自同一表达式。
    std::vector<double> strided={-1,99,0,99,.37,99,3,99,4,99};
    put(input,strided); put(slots,std::vector<int>{0,1,2,3,4});
    ArrayView<const double> view{input.ptr,5,1,2,1,10,MemorySpace::Cuda,input.readiness()};
    auto result=bias.evaluate(view,slots.view(5));
    const auto seeds=get(result.gradient);
    close(seeds[0],derivative(0)/5,"left extension");
    close(seeds[2],derivative(.37)/5,"interior derivative");
    close(seeds[4],derivative(3)/5,"right extension");
    const double h=1e-5;
    strided[4]+=(h); put(input,strided); double plus=bias.evaluate(view,slots.view(5)).energy;
    strided[4]-=2*h; put(input,strided); double minus=bias.evaluate(view,slots.view(5)).energy;
    const double fd_error=std::abs((plus-minus)/(2*h)-seeds[2]);
    // 能量差再除以 h 会放大舍入；解析插值导数仍按上面的 2e-12 检查。
    if (fd_error>1e-9) throw std::runtime_error("table finite difference");
    std::cout<<"table_fd_error="<<fd_error<<"\n";
    // 本地空 rank 仍可接收其它 rank 的总和；零长输出无需启动数组 kernel。
    FrozenSharedBias empty_rank(path,[](double* values,size_t n) {
        if(n!=4) throw std::runtime_error("collective packet");
        values[0]+=potential(.5); values[1]+=.5; values[2]+=1;
    });
    GpuBuffer<double> empty; GpuBuffer<int> no_slots;
    const auto remote=empty_rank.evaluate(empty.view(0),no_slots.view(0));
    close(remote.energy,potential(.5),"empty rank");
    bool rejected=false;
    try { bias.evaluate(empty.view(0),no_slots.view(0)); } catch(const std::runtime_error&) { rejected=true; }
    if (!rejected) throw std::runtime_error("empty family accepted");
    put(slots,std::vector<int>{0,0});
    rejected=false;
    try { bias.evaluate(view,slots.view(2)); } catch(const std::runtime_error&) { rejected=true; }
    if (!rejected) throw std::runtime_error("duplicate slots accepted");
    // 错误之后下一轮重新清空暂存，且重载相同表完全复现能量。
    put(slots,std::vector<int>{0,2}); put(input,strided);
    const auto before=bias.evaluate(view,slots.view(2));
    FrozenSharedBias restored(path);
    const auto after=restored.evaluate(view,slots.view(2));
    if(before.energy!=after.energy || bias.fingerprint()!=restored.fingerprint())
        throw std::runtime_error("frozen replay");
    std::remove(path);
    std::cout<<"PASS shared frozen: mixed graph seeds, masks, streams, Hermite derivative, empty rank, replay\n";
    return 0;
} catch(const std::exception& error) { std::cerr<<error.what()<<"\n"; return 1; }
