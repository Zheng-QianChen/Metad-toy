#include "zqc_cv_assembly.cuh"
#include "zqc_graph.cuh"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using MetaD_zqc::ArrayView;
using MetaD_zqc::CVAtomLayout;
using MetaD_zqc::MemorySpace;
using MetaD_zqc::SelectionId;
using MetaD_zqc::SelectionView;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
void close(double actual, double expected, const char* message) {
    if (!std::isfinite(actual) || std::abs(actual-expected) > 2e-11*(1+std::abs(expected)))
        throw std::runtime_error(message);
}
void put(GpuBuffer<double>& buffer, const std::vector<double>& values) {
    // 上传返回后数据已就绪，Leaf 重绑可以立即将其交给 Graph 的独立流。
    buffer.upload_from(values.data(), values.size());
}
std::vector<double> read(const GpuBuffer<double>& buffer, int count) {
    std::vector<double> values(count);
    buffer.download_to(values.data(), values.size());
    return values;
}
template<class F> void rejects(F call, const char* message) {
    bool caught = false;
    try { call(); } catch (const std::exception&) { caught = true; }
    require(caught, message);
}

SelectionView selection(SelectionId id, int owned_count, const std::vector<int>& ids) {
    return {id, owned_count,
            {ids.data(), ids.size(), 1, 1, 1, ids.size(), MemorySpace::Host}};
}

void assembly_and_graph() {
    // 10 个 owned 原子，后两项为 ghost。选择集编号是身份，不是 bitmask。
    const std::vector<int> a_ids = {2, 5, 10, 11};
    const std::vector<int> b_ids = {5, 9, 10, 11};
    const std::vector<int> empty_ids = {10}; // 仅 ghost
    const std::vector<int> huge_ids = {2, 5}; // 超大身份编号，验证非 bitmask
    CVAtomLayout layout;
    layout.rebuild(10, 2, {
        selection(1, 2, a_ids),
        selection(2, 2, b_ids),
        selection(1, 2, a_ids), // 重复合并
        selection(4, 0, empty_ids),
        selection(1000003, 2, huge_ids),
    });
    require(layout.local_ids() == std::vector<int>({2,5,9,10,11}), "union local tags");
    require(layout.leaf_count()==3 && layout.total_count()==5, "ghost entered owned count");
    auto a=layout.group(1), b=layout.group(2), empty=layout.group(4), huge=layout.group(1000003);
    require(a == layout.group(1), "same group did not reuse assembly");
    require(a->selection_id()==1 && huge->selection_id()==1000003, "selection identities");
    require(a->local_ids() == std::vector<int>({2,5,10,11}), "A owned/ghost tags");
    require(b->local_ids() == std::vector<int>({5,9,10,11}), "B owned/ghost tags");
    require(a->leaf_slots() == std::vector<int>({0,1,-1,-1}), "A slots");
    require(b->leaf_slots() == std::vector<int>({1,2,-1,-1}), "B slots");
    require(huge->local_ids() == std::vector<int>({2,5}), "large id membership");
    require(a->group_count()==2 && a->total_count()==4, "owned group count includes ghost");
    require(a->nlocal()==10 && a->nghost()==2, "atom extents");
    require(empty->group_count()==0 && empty->total_count()==1, "ghost-only group count");
    require(empty->local_ids()[0]==a->nlocal(), "first ghost must be exactly nlocal");
    std::vector<int> device_ids(a->total_count());
    graph_check_cuda(cudaMemcpy(device_ids.data(), a->device_local_ids(),
                                device_ids.size()*sizeof(int), cudaMemcpyDeviceToHost));
    require(device_ids==a->local_ids(), "CV device localtag table omitted ghosts");

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
    GpuBuffer<double> owned_source(6);
    put(owned_source,{0,0,.2,0,0,.5});
    a->assemble_values(owned_source,va); b->assemble_values(raw_b,vb);
    empty->assemble_values(no_values,empty_values);
    require(read(va,5) == std::vector<double>({.2,.5,0,123,456}), "A assembly/padding/guard");
    require(read(vb,5) == std::vector<double>({0,.7,0,123,456}), "B assembly/padding/guard");
    require(read(empty_values,5) == std::vector<double>({0,0,0,123,456}), "empty group not zeroed");

    // 跨度为 2 的填充 SoA：只写偶数槽，验证 stride 装配与 adjoint 回收。
    GpuBuffer<double> strided_source(24), strided_target(10), strided_leaf(10), strided_cv(4);
    std::vector<double> host_strided(24, -1.0);
    host_strided[2*2]=.2; host_strided[2*5]=.5;
    put(strided_source, host_strided);
    put(strided_target, std::vector<double>(10, 77));
    ArrayView<const double> source_view{strided_source.ptr, 12, 1, 2, 1, 24, MemorySpace::Cuda};
    ArrayView<double> target_view{strided_target.ptr, 3, 1, 2, 1, 10, MemorySpace::Cuda};
    a->assemble_values(source_view, target_view);
    const auto assembled = read(strided_target, 10);
    require(assembled[0]==.2 && assembled[2]==.5 && assembled[4]==0, "strided assembly values");
    require(assembled[1]==77 && assembled[3]==77 && assembled[5]==77, "strided assembly padding intact");
    put(strided_leaf, {1, 88, 3, 88, 5, 88, 7, 88, 9, 88});
    ArrayView<const double> leaf_view{strided_leaf.ptr, 3, 1, 2, 1, 10, MemorySpace::Cuda};
    ArrayView<double> cv_view{strided_cv.ptr, 2, 1, 2, 1, 4, MemorySpace::Cuda};
    put(strided_cv, {11, 22, 33, 44});
    a->gather_adjoint(leaf_view, cv_view);
    const auto adjoints = read(strided_cv, 4);
    require(adjoints[0]==1 && adjoints[2]==3 && adjoints[1]==22 && adjoints[3]==44,
            "strided adjoint gather");

    Leaf x(va,ga,layout.leaf_count()), y(vb,gb,layout.leaf_count());
    Graph graph(Graph::Symbols{{"a",&x},{"b",&y}});
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

    host_a[2]=-.3; host_b[9]=1.1; put(raw_a,host_a); put(raw_b,host_b);
    a->assemble_values(raw_a,va); b->assemble_values(raw_b,vb);
    close(run.forward()[0],(std::exp(-.3)+std::exp(.5)+1+.7+1.1)/3,"refreshed graph output");
    run.backward({-1});
    a->gather_adjoint(ga,compact_ga);
    close(read(compact_ga,4)[0],-std::exp(-.3)/3,"refreshed adjoint");

    // 别名与形状非法
    rejects([&]{a->assemble_values(raw_a.view(1), va.view(3));},"short source accepted");
    rejects([&]{a->assemble_values(raw_a.view(12), va.view(1));},"short target accepted");
    rejects([&]{a->assemble_values(raw_a.view(12), raw_a.view(3));},"in-place assembly accepted");
    rejects([&]{a->gather_adjoint(ga.view(3), ga.view(2));},"in-place adjoint accepted");
    rejects([&]{a->gather_adjoint(ga.view(3), compact_ga.view(1));},"short adjoint target accepted");
    rejects([&]{layout.group(8);},"unrequested group accepted");
    rejects([&]{layout.rebuild(-1,0,{});},"negative owned accepted");
    rejects([&]{layout.rebuild(10,-1,{selection(1,2,a_ids)});},"negative ghosts accepted");
    const std::vector<int> unsorted = {5, 2};
    rejects([&]{layout.rebuild(10,2,{selection(1,2,unsorted)});},"unsorted selection accepted");
    require(layout.group(1)==a && layout.leaf_count()==3,"failed rebuild changed existing bindings");

    // 成员变动后重建；旧 shared_ptr 保持原映射。
    const std::vector<int> a2 = {3, 5, 10, 11};
    const std::vector<int> b2 = {5, 9, 10, 11};
    layout.rebuild(10, 2, {selection(1, 2, a2), selection(2, 2, b2)});
    require(layout.local_ids()==std::vector<int>({3,5,9,10,11}),"changed membership was not refreshed");
    require(layout.group(1)->local_ids()==std::vector<int>({3,5,10,11}),"changed A mapping");
    require(a->local_ids()==std::vector<int>({2,5,10,11}),"old mapping lifetime was not retained");
    const std::vector<int> a3 = {3, 5};
    const std::vector<int> b3 = {5, 9, 10};
    layout.rebuild(10, 1, {selection(1, 2, a3), selection(2, 2, b3)});
    require(layout.leaf_count()==3 && layout.total_count()==4, "ghost shrink changed owned extent");
    require(layout.group(1)->local_ids()==std::vector<int>({3,5}), "stale ghost in A");
    require(layout.group(2)->local_ids()==std::vector<int>({5,9,10}), "B ghost refresh");

    const std::vector<int> ghost_a = {0, 1};
    const std::vector<int> ghost_b = {1};
    layout.rebuild(0, 2, {selection(1, 0, ghost_a), selection(2, 0, ghost_b)});
    require(layout.leaf_count()==0 && layout.total_count()==2, "ghost-only rank layout");
    require(layout.group(1)->group_count()==0 && layout.group(1)->total_count()==2,
            "ghost-only rank group count");
    layout.group(1)->assemble_values(no_values,no_values);
    layout.group(1)->gather_adjoint(no_values,no_values);
    layout.rebuild(0,0, {selection(1,0,{}), selection(2,0,{})});
    require(layout.leaf_count()==0 && layout.group(1)->group_count()==0,"empty rank counts");
    layout.group(1)->assemble_values(no_values,no_values);
    layout.group(1)->gather_adjoint(no_values,no_values);
    layout.rebuild(10,2,{});
    require(layout.leaf_count()==0,"no requested groups should produce an empty union");
}

void leaf_view_rebind() {
    GpuBuffer<double> values(4), gradients(4), grown_values(8), grown_gradients(8);
    put(values, {1,2,3,4});
    Leaf leaf(values, gradients, 2);
    Graph graph(Graph::Symbols{{"x",&leaf}});
    graph.parse_expr("AVE(x)");
    GraphExecution run(graph);
    close(run.forward()[0], 1.5, "initial contiguous average");
    run.backward({2});
    close(read(gradients,2)[0], 1.0, "initial contiguous adjoint");

    // 扩容后重新 bind：有效长度变为 3，必须 resize_inputs。
    put(grown_values, {2,4,6,0,0,0,0,0});
    leaf.bind(ArrayView<const double>(grown_values.view(3)), grown_gradients.view(3));
    run.resize_inputs();
    close(run.forward()[0], 4.0, "rebound average");
    run.backward({3});
    const auto grads = read(grown_gradients, 3);
    close(grads[0], 1.0, "rebound adjoint 0");
    close(grads[1], 1.0, "rebound adjoint 1");
    close(grads[2], 1.0, "rebound adjoint 2");

    // 非法形状
    rejects([&]{
        leaf.bind(grown_values.view(2), grown_gradients.view(3));
    }, "mismatched extents accepted");
    double host_value = 1.0, host_grad = 0.0;
    rejects([&]{
        leaf.bind(ArrayView<const double>{&host_value, 1, 1, 1, 1, 1, MemorySpace::Host},
                  ArrayView<double>{&host_grad, 1, 1, 1, 1, 1, MemorySpace::Host});
    }, "Host PerAtom view accepted");
}

int main() {
    try {
        assembly_and_graph();
        leaf_view_rebind();
        graph_check_cuda(cudaDeviceSynchronize());
        std::cout << "CV assembly, selection identities, strided views, Leaf rebind and NVRTC passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
