#include "zqc_cv_assembly.cuh"
#include "zqc_graph.cuh"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using MetaD_zqc::CVAtomLayout;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
void close(double actual, double expected, const char* message) {
    if (!std::isfinite(actual) || std::abs(actual-expected) > 2e-11*(1+std::abs(expected)))
        throw std::runtime_error(message);
}
void put(GpuBuffer<double>& buffer, const std::vector<double>& values) {
    if (!values.empty()) graph_check_cuda(cudaMemcpy(buffer.ptr, values.data(), values.size()*sizeof(double),
                                                    cudaMemcpyHostToDevice));
}
std::vector<double> read(const GpuBuffer<double>& buffer, int count) {
    std::vector<double> values(count);
    if (count) graph_check_cuda(cudaMemcpy(values.data(), buffer.ptr, count*sizeof(double), cudaMemcpyDeviceToHost));
    return values;
}
template<class F> void rejects(F call, const char* message) {
    bool caught = false;
    try { call(); } catch (const std::exception&) { caught = true; }
    require(caught, message);
}

void assembly_and_graph() {
    // 10 个 owned 原子，后两项为 ghost；保留编号，但不进入 Leaf 规约。
    std::vector<int> mask(12, 0);
    mask[2]=1; mask[5]=3; mask[9]=2; mask[10]=7; mask[11]=3; // group 4 仅有 ghost
    CVAtomLayout layout;
    layout.rebuild(10, 2, mask.data(), {1, 2, 1, 4});
    require(layout.local_ids() == std::vector<int>({2,5,9,10,11}), "union local tags");
    require(layout.leaf_count()==3 && layout.total_count()==5, "ghost entered owned count");
    auto a=layout.group(1), b=layout.group(2), empty=layout.group(4);
    require(a == layout.group(1), "same group did not reuse assembly");
    require(a->local_ids() == std::vector<int>({2,5,10,11}), "A owned/ghost tags");
    require(b->local_ids() == std::vector<int>({5,9,10,11}), "B owned/ghost tags");
    require(a->leaf_slots() == std::vector<int>({0,1,-1,-1}), "A slots");
    require(b->leaf_slots() == std::vector<int>({1,2,-1,-1}), "B slots");
    require(a->group_count()==2 && a->total_count()==4, "owned group count includes ghost");
    require(a->nlocal()==10 && a->nghost()==2, "atom extents");
    require(empty->group_count()==0 && empty->total_count()==1, "ghost-only group count");
    require(empty->local_ids()[0]==a->nlocal(), "first ghost must be exactly nlocal");
    std::vector<int> device_ids(a->total_count());
    graph_check_cuda(cudaMemcpy(device_ids.data(), a->device_local_ids(),
                                device_ids.size()*sizeof(int), cudaMemcpyDeviceToHost));
    require(device_ids==a->local_ids(), "CV device localtag table omitted ghosts");
    std::vector<int> ghost_ids(empty->total_count());
    graph_check_cuda(cudaMemcpy(ghost_ids.data(), empty->device_local_ids(),
                                ghost_ids.size()*sizeof(int), cudaMemcpyDeviceToHost));
    require(ghost_ids==empty->local_ids(), "ghost-only group was not uploaded");

    GpuBuffer<double> raw_a(12), raw_b(12), va(5), vb(5), ga(5), gb(5);
    GpuBuffer<double> compact_ga(4), compact_gb(4), empty_values(5), no_values;
    std::vector<double> host_a(12, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> host_b(12, std::numeric_limits<double>::quiet_NaN());
    host_a[2]=.2; host_a[5]=.5; host_b[5]=.7; host_b[9]=0; // 有效零值仍是组成员。
    put(raw_a,host_a); put(raw_b,host_b);
    put(va,{99,99,99,123,456}); put(vb,{99,99,99,123,456});
    put(ga,{99,99,99,123,456}); put(gb,{99,99,99,123,456});
    put(compact_ga,{99,99,123,456}); put(compact_gb,{99,99,123,456});
    put(empty_values,{99,99,99,123,456});
    // A 的 source 甚至可以没有 ghost 存储，读取边界应仅由 owned 前缀决定。
    GpuBuffer<double> owned_source(6);
    put(owned_source,{0,0,.2,0,0,.5});
    a->assemble_values(owned_source,va); b->assemble_values(raw_b,vb);
    empty->assemble_values(no_values,empty_values);
    require(read(va,5) == std::vector<double>({.2,.5,0,123,456}), "A assembly/padding/guard");
    require(read(vb,5) == std::vector<double>({0,.7,0,123,456}), "B assembly/padding/guard");
    require(read(empty_values,5) == std::vector<double>({0,0,0,123,456}), "empty group not zeroed");

    Leaf x(va,ga,layout.leaf_count()), y(vb,gb,layout.leaf_count());
    Graph graph(Graph::Symbols{{"a",&x},{"b",&y}});
    // 此处明确对公共并集求平均；并非 A 自己的组平均。
    graph.parse_expr("AVE(exp(a)+b)");
    GraphExecution run(graph);
    close(run.forward()[0], (std::exp(.2)+std::exp(.5)+1+.7)/3, "assembled graph output");
    run.backward({2.0});
    a->gather_adjoint(ga,compact_ga); b->gather_adjoint(gb,compact_gb);
    const auto da=read(compact_ga,4), db=read(compact_gb,4);
    close(da[0],2*std::exp(.2)/3,"A gradient at atom 2");
    close(da[1],2*std::exp(.5)/3,"A gradient at atom 5");
    close(db[0],2.0/3,"B gradient at atom 5");
    close(db[1],2.0/3,"B gradient at atom 9");
    require(da[2]==123 && da[3]==456 && db[2]==123 && db[3]==456, "adjoint guard modified");
    close(read(ga,5)[2],2.0/3,"padding gradient should exist in graph, but not reach CV");
    require(read(ga,5)[3]==123 && read(gb,5)[4]==456,"graph wrote outside Leaf length");

    // 重复装配必须覆盖旧值；换值不重编译图。
    host_a[2]=-.3; host_b[9]=1.1; put(raw_a,host_a); put(raw_b,host_b);
    a->assemble_values(raw_a,va); b->assemble_values(raw_b,vb);
    close(run.forward()[0],(std::exp(-.3)+std::exp(.5)+1+.7+1.1)/3,"refreshed graph output");
    run.backward({-1});
    a->gather_adjoint(ga,compact_ga);
    close(read(compact_ga,4)[0],-std::exp(-.3)/3,"refreshed adjoint");

    GpuBuffer<double> tiny(1);
    rejects([&]{a->assemble_values(tiny,va);},"short source accepted");
    rejects([&]{a->assemble_values(raw_a,tiny);},"short target accepted");
    rejects([&]{a->assemble_values(raw_a,raw_a);},"in-place assembly accepted");
    rejects([&]{a->gather_adjoint(ga,ga);},"in-place adjoint accepted");
    rejects([&]{a->gather_adjoint(ga,tiny);},"short adjoint target accepted");
    rejects([&]{layout.group(8);},"unrequested group accepted");
    rejects([&]{layout.rebuild(10,2,nullptr,{1});},"missing mask accepted");
    rejects([&]{layout.rebuild(-1,0,nullptr,{1});},"negative nlocal accepted");
    rejects([&]{layout.rebuild(10,2,mask.data(),{0});},"zero groupbit accepted");
    rejects([&]{layout.rebuild(10,-1,mask.data(),{1});},"negative nghost accepted");
    rejects([&]{layout.rebuild(0,1,nullptr,{1});},"missing ghost mask accepted");
    rejects([&]{layout.rebuild(std::numeric_limits<int>::max(),1,mask.data(),{1});},
            "owned plus ghost count overflow accepted");
    require(layout.group(1)==a && layout.leaf_count()==3,"failed rebuild changed existing bindings");

    // 本地编号/成员变动后重建，即使 count 不变也必须拿新映射。
    mask[2]=0; mask[3]=1;
    layout.rebuild(10,2,mask.data(),{1,2});
    require(layout.local_ids()==std::vector<int>({3,5,9,10,11}),"changed membership was not refreshed");
    require(layout.group(1)->local_ids()==std::vector<int>({3,5,10,11}),"changed A mapping");
    require(a->local_ids()==std::vector<int>({2,5,10,11}),"old mapping lifetime was not retained");
    // owned 数量不变，ghost 数量/组别改变也要更新表尾。
    mask[10]=2;
    layout.rebuild(10,1,mask.data(),{1,2});
    require(layout.leaf_count()==3 && layout.total_count()==4, "ghost shrink changed owned extent");
    require(layout.group(1)->local_ids()==std::vector<int>({3,5}), "stale ghost in A");
    require(layout.group(2)->local_ids()==std::vector<int>({5,9,10}), "B ghost refresh");
    // 没有 owned 但有 ghost：仍保留 localtag 表，Leaf 长度与 group_count 都是零。
    const int ghost_mask[] = {1,3};
    layout.rebuild(0,2,ghost_mask,{1,2});
    require(layout.leaf_count()==0 && layout.total_count()==2, "ghost-only rank layout");
    require(layout.group(1)->group_count()==0 && layout.group(1)->total_count()==2,
            "ghost-only rank group count");
    layout.group(1)->assemble_values(no_values,no_values);
    layout.group(1)->gather_adjoint(no_values,no_values);
    // 全空 rank 不触碰空指针，也不启动零 block kernel。
    layout.rebuild(0,0,nullptr,{1,2});
    require(layout.leaf_count()==0 && layout.group(1)->group_count()==0,"empty rank counts");
    layout.group(1)->assemble_values(no_values,no_values);
    layout.group(1)->gather_adjoint(no_values,no_values);
    layout.rebuild(10,2,mask.data(),{});
    require(layout.leaf_count()==0,"no requested groups should produce an empty union");
}

int main() {
    try {
        assembly_and_graph();
        graph_check_cuda(cudaDeviceSynchronize());
        std::cout << "CV assembly, owned/ghost boundaries, NVRTC forward/backward and empty selections passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
