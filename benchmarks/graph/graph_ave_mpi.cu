#include "zqc_graph.cuh"
#include <mpi.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
void close(double actual, double expected, const char* message) {
    if (!std::isfinite(actual) || std::abs(actual-expected) > 2e-11*(1+std::abs(expected)))
        throw std::runtime_error(std::string(message)+": "+std::to_string(actual)+" != "+std::to_string(expected));
}
std::vector<double> read(const double* ptr, int count) {
    std::vector<double> values(count);
    if (count) graph_check_cuda(cudaMemcpy(values.data(), ptr, count*sizeof(double), cudaMemcpyDeviceToHost));
    return values;
}
void put(double* ptr, const std::vector<double>& values) {
    if (!values.empty()) graph_check_cuda(cudaMemcpy(ptr, values.data(), values.size()*sizeof(double), cudaMemcpyHostToDevice));
}

void host_only() {
    Leaf a({0.4}, {Domain::Global, 1});
    Graph graph(Graph::Symbols{{"a", &a}});
    graph.parse_expr("linear(a*a, 2, 1) + exp(-a)");
    graph.parse_expr("fermi(a, 0.5, 2)");
    GraphExecution run(graph);
    require(graph.kernel_source.empty(), "Host-only graph generated a kernel");
    for (double x : {0.4, -0.2}) {
        *a.val = x;
        const auto y = run.forward();
        const double sigmoid = 1/(1+std::exp(-2*(x-.5)));
        close(y[0], 2*x*x+1+std::exp(-x), "Host scalar output");
        close(y[1], sigmoid, "Host sigmoid output");
        run.backward({.7, -.3});
        close(*a.adj, .7*(4*x-std::exp(-x))-.3*2*sigmoid*(1-sigmoid), "Host scalar gradient");
    }
}

void borrowed_global_scalar() {
    GpuBuffer<double> values(1), gradients(1);
    put(values.ptr, {.6});
    Leaf x(values, gradients, 1, {Domain::Global, 1});
    Graph graph(Graph::Symbols{{"x", &x}});
    graph.parse_expr("x*x");
    GraphExecution run(graph);
    close(run.forward()[0], .36, "borrowed Global value");
    run.backward({2});
    close(read(gradients.ptr,1)[0], 2.4, "borrowed Global gradient");
}

// 同一套全局样本按不均匀数量划分；np>1 时 rank 0 故意没有样本。
void distributed_graph(MPI_Comm comm) {
    int rank = 0, size = 0;
    MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &size);
    const std::vector<double> x_all = {-.3, .2, .6, 1.1, 1.4};
    const std::vector<double> y_all = {.4, -.5, .8, .3, -.2};
    std::vector<int> ids;
    for (int i = 0; i < static_cast<int>(x_all.size()); ++i) {
        const int owner = size == 1 ? 0 : 1+(i*i)%(size-1);
        if (owner == rank) ids.push_back(i);
    }
    const int n = static_cast<int>(ids.size());
    GpuBuffer<double> vx(n+3), vy(n+3), gx(n+3), gy(n+3);
    Leaf x(vx, gx, n), y(vy, gy, n);
    Leaf coefficient({.75}, {Domain::Global, 1});
    Graph graph(Graph::Symbols{{"x", &x}, {"y", &y}}, {}, comm);
    Node* product = graph.make_node<Mul>(&x, &y);
    auto* u = graph.make_node<AVE>(product, comm);
    auto* v = graph.make_node<AVE>(&x, comm);
    Node* first = graph.make_node<Add>(u, graph.make_node<Linear>(v, 2.0, .1));
    Node* second = graph.make_node<Mul>(graph.make_node<Mul>(u, graph.make_node<Exp>(v)), &coefficient);
    graph.outputs = {first, second};
    GraphExecution run(graph);
    require(graph.kernel_seeds.size() == 2, "shared AVE executed more than once");
    bool before_forward = false;
    try { run.backward({1,1}); } catch (const std::logic_error&) { before_forward = true; }
    require(before_forward, "backward before forward accepted");
    const double guard = 98765;
    for (double offset : {0.0, .1}) {
        std::vector<double> local_x(n+3, guard), local_y(n+3, guard);
        for (int i = 0; i < n; ++i) { local_x[i]=x_all[ids[i]]+offset; local_y[i]=y_all[ids[i]]; }
        put(vx.ptr, local_x); put(vy.ptr, local_y);
        put(gx.ptr, std::vector<double>(n+3, guard)); put(gy.ptr, std::vector<double>(n+3, guard));
        *coefficient.val = .75+offset;
        double expected_u=0, expected_v=0;
        for (size_t i=0; i<x_all.size(); ++i) {
            expected_u += (x_all[i]+offset)*y_all[i]/x_all.size();
            expected_v += (x_all[i]+offset)/x_all.size();
        }
        const auto output = run.forward();
        close(output[0], expected_u+2*expected_v+.1, "distributed first output");
        close(output[1], expected_u*std::exp(expected_v)*(*coefficient.val), "distributed second output");
        require(u->global_count == 5 && v->global_count == 5, "global count is wrong");
        for (double seed : {-.2, .3}) {
            run.backward({.7, seed});
            const auto dx=read(gx.ptr,n+3), dy=read(gy.ptr,n+3);
            const double du=.7+seed*(*coefficient.val)*std::exp(expected_v);
            const double dv=1.4+seed*(*coefficient.val)*expected_u*std::exp(expected_v);
            for (int i=0;i<n;++i) {
                close(dx[i], (du*local_y[i]+dv)/5, "distributed input gradient x");
                close(dy[i], du*local_x[i]/5, "distributed input gradient y");
            }
            for (int i=n;i<n+3;++i) require(dx[i]==guard && dy[i]==guard, "gradient guard overwritten");
            close(*coefficient.adj, seed*expected_u*std::exp(expected_v), "Global parameter gradient");
        }
    }
    // 验证注册入口及规约后的非线性运算，不依赖手工构图。
    Graph parsed(Graph::Symbols{{"x",&x},{"y",&y}}, {}, comm);
    parsed.parse_expr("linear(AVE(x*y), 2, 1) + exp(AVE(x))");
    GraphExecution parsed_run(parsed);
    const auto output=parsed_run.forward();
    close(output[0], 2*(*u->val)+1+std::exp(*v->val), "parsed AVE output");
    parsed_run.backward({1});
    const auto dx=read(gx.ptr,n);
    for (int i=0;i<n;++i) close(dx[i], (2*y_all[ids[i]]+std::exp(*v->val))/5, "parsed AVE gradient");
    // 全局为空必须在所有 rank 完成 collective 后一致报错。
    Leaf empty(std::vector<double>{});
    AVE empty_mean(&empty, comm);
    bool rejected=false;
    try { empty_mean.forward(); } catch (const std::runtime_error&) { rejected=true; }
    require(rejected, "global empty average accepted");
}

// 模拟迁移：同一执行器从非空变空，再增长；核源码/module 保持不变。
void resize_and_logging(MPI_Comm comm) {
    int rank = 0; MPI_Comm_rank(comm, &rank);
    GpuBuffer<double> values(8), gradients(8);
    Leaf x(values, gradients, 1);
    Graph graph(Graph::Symbols{{"x", &x}}, {}, comm);
    graph.parse_expr("AVE(x*x)");
    FILE* logfile = std::tmpfile();
    require(logfile != nullptr, "tmpfile failed");
    {
        GraphExecution run(graph, logfile, logfile); // 相同句柄只记录一次。
        for (int phase : {1, 0, 7}) {
            // 空 rank 仍参与均值；至少 rank 0 持有一个元素。
            const int n = phase ? phase : (rank == 0 ? 1 : 0);
            x.n = n;
            put(values.ptr, std::vector<double>(n, 2.0));
            run.resize_inputs();
            bool rejected = false;
            try { run.backward({1}); } catch (const std::logic_error&) { rejected = true; }
            require(rejected, "resize must invalidate old forward");
            close(run.forward()[0], 4.0, "resized average");
            run.backward({1});
            int total = 0; MPI_Allreduce(&n, &total, 1, MPI_INT, MPI_SUM, comm);
            for (double value : read(gradients.ptr, n)) close(value, 4.0/total, "resized gradient");
        }
    }
    // 编译失败也必须留下诊断，不能只有成功日志。
    graph.kernel_source += "\n#error deliberate_NVRTC_log_test\n";
    bool failed = false;
    try { GraphExecution invalid(graph, nullptr, logfile); }
    catch (const std::runtime_error& e) {
        failed = std::string(e.what()).find("deliberate_NVRTC_log_test") != std::string::npos;
    }
    require(failed, "invalid source did not report its NVRTC diagnostic");
    std::rewind(logfile);
    std::string log; char line[1024];
    while (std::fgets(line, sizeof(line), logfile)) log += line;
    std::fclose(logfile);
    require(log.find("NVRTC/module failed:") != std::string::npos, "NVRTC failure was not logged");
    const auto first = log.find("NVRTC ready:");
    require(first != std::string::npos && log.find("NVRTC ready:", first+1) == std::string::npos,
            "resize recompiled or duplicate log handles wrote twice");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank=0; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    try {
        host_only();
        graph_check_cuda(cudaSetDevice(0));
        borrowed_global_scalar();
        distributed_graph(MPI_COMM_WORLD);
        resize_and_logging(MPI_COMM_WORLD);
        // 使用分割后的 communicator，防止实现偷偷使用 MPI_COMM_WORLD。
        MPI_Comm subcomm = MPI_COMM_NULL;
        MPI_Comm_split(MPI_COMM_WORLD, rank%2, rank, &subcomm);
        distributed_graph(subcomm);
        MPI_Comm_free(&subcomm);
        if (!rank) std::cout << "AVE_MPI_PASS Host scalars, unequal partitions, empty ranks, shared DAG, backward, subcommunicators\n";
    } catch (const std::exception& error) {
        std::cerr << "rank " << rank << ": " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
}
