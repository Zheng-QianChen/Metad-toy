#include "zqc_graph.cuh"
#include "mpi_sum_reducer.h"
#include <mpi.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
void close(double actual, double expected, const char* message) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > 2e-11 * (1.0 + std::abs(expected)))
        throw std::runtime_error(std::string(message) + ": " + std::to_string(actual) +
                                 " != " + std::to_string(expected));
}
std::vector<double> read(const GpuBuffer<double>& buffer, int count) {
    std::vector<double> values(count);
    if (count) buffer.download_to(values.data(), values.size());
    return values;
}
void put(GpuBuffer<double>& buffer, const std::vector<double>& values) {
    // Pageable cudaMemcpy(H2D) may return before DMA finishes. The blocking
    // buffer upload completes its stream before the nonblocking Graph consumes it.
    if (!values.empty()) buffer.upload_from(values.data(), values.size());
}

// 模拟 CV.SUM：只累加指定 leaf 槽位，组外 padding 不参与。
struct FamilySum : SUM {
    std::vector<int> slots;
    FamilySum(Node* src, MetaD_zqc::SumReducer sum, std::vector<int> slots_)
        : SUM(src, std::move(sum)), slots(std::move(slots_)) {}
    void reduce_host(const double* values, int count) override {
        require(count >= 0, "family sum count");
        double local_sum = 0.0, correction = 0.0;
        for (int slot : slots) {
            require(slot >= 0 && slot < count, "family slot");
            const double adjusted = values[slot] - correction;
            const double next = local_sum + adjusted;
            correction = (next - local_sum) - adjusted;
            local_sum = next;
        }
        reduce_sum(local_sum);
    }
};

void host_global_safe_div() {
    Leaf num({1.5}, {ValueDomain::Global, 1});
    Leaf den({0.5}, {ValueDomain::Global, 1});
    Graph graph(Graph::Symbols{{"a", &num}, {"b", &den}});
    graph.parse_expr("safe_div(a, b, 1e-12)");
    GraphExecution run(graph);
    close(run.forward()[0], 3.0, "global safe_div");
    run.backward({2.0});
    close(*num.adj, 4.0, "global safe_div d/da");
    close(*den.adj, -12.0, "global safe_div d/db");
}

void mean_solid_nested_sum() {
    // MEAN_SOLID = sum(q f)/sum(f) via nested SUM + safe_div；保持 AVE 语义不变。
    GpuBuffer<double> values_buffer(3),gradient_buffer(3);
    Leaf q(values_buffer,gradient_buffer,3);
    const std::vector<double> baseline{0.2,0.8,0.5};
    put(values_buffer,baseline);
    Graph solid(Graph::Symbols{{"q", &q}});
    solid.parse_expr("safe_div(SUM(q*fermi(q,0.45,12)), SUM(fermi(q,0.45,12)), 1e-12)");
    GraphExecution solid_run(solid);
    const auto values = solid_run.forward();
    double sum_qf = 0.0, sum_f = 0.0;
    for (double qi : {0.2, 0.8, 0.5}) {
        const double f = 1.0 / (1.0 + std::exp(-12.0 * (qi - 0.45)));
        sum_qf += qi * f;
        sum_f += f;
    }
    close(values[0], sum_qf / sum_f, "MEAN_SOLID via SUM/safe_div");
    solid_run.backward({1.0});
    const auto analytic=read(gradient_buffer,3);
    constexpr double step=1e-5;
    for(int i=0;i<3;++i) {
        auto plus=baseline,minus=baseline;
        plus[i]+=step; minus[i]-=step;
        put(values_buffer,plus); const double vp=solid_run.forward()[0];
        put(values_buffer,minus); const double vm=solid_run.forward()[0];
        require(std::abs((vp-vm)/(2*step)-analytic[i])<2e-8,"MEAN_SOLID gradient finite difference");
    }
}

void zero_and_near_zero_denominators() {
    double da=0,db=0;
    zqc_graph_safe_div_bwd(1.0,5e-201,1e-200,0.0,&da,&db);
    close(da,1e200,"tiny denominator d/da");
    close(db,-5e199,"tiny denominator d/db without squaring underflow");
    constexpr double eps = 0.1;
    Leaf a({2.0}, {ValueDomain::Global, 1});
    for (double b_host : {0.0, eps, -eps, 0.05, -0.05}) {
        Leaf b({b_host}, {ValueDomain::Global, 1});
        Graph graph(Graph::Symbols{{"a", &a}, {"b", &b}});
        graph.parse_expr("safe_div(a, b, 0.1)");
        GraphExecution run(graph);
        close(run.forward()[0], 0.0, "safe_div zero-band value");
        run.backward({1.0});
        close(*a.adj, 0.0, "safe_div zero-band d/da");
        close(*b.adj, 0.0, "safe_div zero-band d/db");
    }
    // 边界两侧离开 |b|<=epsilon 后恢复普通除法；两侧符号都测。
    for (double b_host : {eps + 1e-3, -(eps + 1e-3), 0.5, -0.5}) {
        Leaf b({b_host}, {ValueDomain::Global, 1});
        Graph graph(Graph::Symbols{{"a", &a}, {"b", &b}});
        graph.parse_expr("safe_div(a, b, 0.1)");
        GraphExecution run(graph);
        close(run.forward()[0], 2.0 / b_host, "safe_div away from boundary");
        run.backward({1.0});
        close(*a.adj, 1.0 / b_host, "safe_div away d/da");
        close(*b.adj, -2.0 / (b_host * b_host), "safe_div away d/db");
    }
    bool rejected = false;
    try {
        Leaf b({1.0}, {ValueDomain::Global, 1});
        Graph bad(Graph::Symbols{{"a", &a}, {"b", &b}});
        bad.parse_expr("safe_div(a, b, -1)");
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "negative epsilon accepted");
}

void per_atom_and_broadcast(MPI_Comm comm) {
    int rank = 0, size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    const auto sum = mpi_sum_reducer(comm);
    const std::vector<double> x_all = {0.5, -0.25, 1.0, 2.0, 0.0};
    std::vector<int> ids;
    for (int i = 0; i < static_cast<int>(x_all.size()); ++i) {
        const int owner = size == 1 ? 0 : 1 + (i * i) % (size - 1);
        if (owner == rank) ids.push_back(i);
    }
    const int n = static_cast<int>(ids.size());
    GpuBuffer<double> vx(n + 2), gx(n + 2), vy(n + 2), gy(n + 2);
    Leaf x(vx, gx, n), y(vy, gy, n);
    Leaf scale({0.5}, {ValueDomain::Global, 1});
    std::vector<double> local_x(n + 2, 7.0), local_y(n + 2, 7.0);
    for (int i = 0; i < n; ++i) {
        local_x[i] = x_all[ids[i]];
        local_y[i] = 0.25 + 0.1 * ids[i];
    }
    put(vx, local_x);
    put(vy, local_y);

    Graph same(Graph::Symbols{{"x", &x}, {"y", &y}}, {}, sum);
    same.parse_expr("SUM(safe_div(x, y, 1e-12))");
    GraphExecution same_run(same);
    double expect_sum = 0.0;
    for (int i = 0; i < static_cast<int>(x_all.size()); ++i)
        expect_sum += x_all[i] / (0.25 + 0.1 * i);
    close(same_run.forward()[0], expect_sum, "PerAtom safe_div inside SUM");
    same_run.backward({1.0});
    const auto dx = read(gx, n);
    for (int i = 0; i < n; ++i) close(dx[i], 1.0 / local_y[i], "PerAtom safe_div d/dx");

    put(gx, std::vector<double>(n + 2, 0.0));
    Graph broadcast(Graph::Symbols{{"x", &x}, {"c", &scale}}, {}, sum);
    broadcast.parse_expr("SUM(safe_div(x, c, 1e-12))");
    GraphExecution broadcast_run(broadcast);
    double broadcast_sum = 0.0;
    for (double xi : x_all) broadcast_sum += xi / 0.5;
    close(broadcast_run.forward()[0], broadcast_sum, "broadcast safe_div SUM");
    broadcast_run.backward({1.0});
    const auto dxb = read(gx, n);
    for (int i = 0; i < n; ++i) close(dxb[i], 1.0 / 0.5, "broadcast d/dx");
    // Every rank must receive the same Global adjoint, including an empty rank.
    const double db = *scale.adj;
    double db_expect = 0.0;
    for (double xi : x_all) db_expect += -xi / (0.5 * 0.5);
    close(db, db_expect, "broadcast d/dc");
}

void empty_ranks_and_global_empty(MPI_Comm comm) {
    int rank = 0, size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    const auto sum = mpi_sum_reducer(comm);

    // 多 rank 时 rank 0 本地为空；其余 rank 贡献样本。全局空则 SUM=0。
    const std::vector<double> samples = {1.0, 2.0, 3.0, 4.0};
    std::vector<int> ids;
    for (int i = 0; i < static_cast<int>(samples.size()); ++i) {
        const int owner = size == 1 ? 0 : 1 + i % (size - 1);
        if (owner == rank) ids.push_back(i);
    }
    const int n = static_cast<int>(ids.size());
    GpuBuffer<double> values(std::max(n, 1)), gradients(std::max(n, 1));
    Leaf x(values, gradients, n);
    std::vector<double> local(n);
    for (int i = 0; i < n; ++i) local[i] = samples[ids[i]];
    put(values, local);

    Graph graph(Graph::Symbols{{"x", &x}}, {}, sum);
    graph.parse_expr("SUM(x)");
    graph.parse_expr("AVE(x)");
    GraphExecution run(graph);
    const auto out = run.forward();
    close(out[0], 10.0, "SUM with possible empty local rank");
    close(out[1], 2.5, "AVE preserved beside SUM");
    run.backward({1.0, 0.0});
    for (double g : read(gradients, n)) close(g, 1.0, "SUM seed is identity");

    Leaf empty(std::vector<double>{});
    Graph empty_graph(Graph::Symbols{{"x", &empty}}, {}, sum);
    empty_graph.parse_expr("SUM(x)");
    GraphExecution empty_run(empty_graph);
    close(empty_run.forward()[0], 0.0, "globally empty SUM is 0");
    empty_run.backward({3.0});

    bool ave_rejected = false;
    try {
        Graph ave_empty(Graph::Symbols{{"x", &empty}}, {}, sum);
        ave_empty.parse_expr("AVE(x)");
        GraphExecution ave_run(ave_empty);
        ave_run.forward();
    } catch (const std::runtime_error&) {
        ave_rejected = true;
    }
    require(ave_rejected, "AVE empty must still fail");
}

void family_exclusion_and_finite_difference(MPI_Comm comm) {
    int rank = 0, size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    const auto sum = mpi_sum_reducer(comm);

    // leaf 布局 [in, pad, in, pad]；仅槽位 0/2 属于本 CV 中心族。
    // 多 rank 时仅 rank 0 持有中心，其余为空；所有 rank 必须走同一套 collective 序列。
    const int leaf_n = (rank == 0 || size == 1) ? 4 : 0;
    GpuBuffer<double> values(std::max(leaf_n, 1)), gradients(std::max(leaf_n, 1));
    Leaf x(values, gradients, leaf_n);
    auto fill = [&](const std::vector<double>& host) {
        if (leaf_n) {
            put(values, host);
            x.refresh_buffers();
        }
    };
    fill({1.0, 100.0, 3.0, 100.0});
    Graph graph(Graph::Symbols{}, {}, sum);
    auto* family = graph.make_node<FamilySum>(&x, sum, leaf_n ? std::vector<int>{0, 2} : std::vector<int>{});
    graph.outputs = {family};
    GraphExecution run(graph);
    double local = leaf_n ? 4.0 : 0.0;
    double global = local;
    MPI_Allreduce(MPI_IN_PLACE, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
    const double base = run.forward()[0];
    close(base, global, "family SUM excludes padding");

    if (leaf_n) put(gradients, std::vector<double>(leaf_n, 0.0));
    run.backward({1.0});
    if (leaf_n) {
        const auto analytic = read(gradients, leaf_n);
        close(analytic[0], 1.0, "family SUM d slot0");
        close(analytic[2], 1.0, "family SUM d slot2");
    }

    constexpr double step = 1e-6;
    for (int slot : {0, 2}) {
        auto plus = std::vector<double>{1.0, 100.0, 3.0, 100.0};
        auto minus = plus;
        plus[slot] += step;
        minus[slot] -= step;
        fill(plus);
        const double yp = run.forward()[0];
        close(yp, base + step, "family SUM forward plus");
        fill(minus);
        const double ym = run.forward()[0];
        close(ym, base - step, "family SUM forward minus");
        const double fd = (yp - ym) / (2 * step);
        if (!std::isfinite(fd) || std::abs(fd - 1.0) > 1e-8)
            throw std::runtime_error("family SUM FD: " + std::to_string(fd));
        fill({1.0, 100.0, 3.0, 100.0});
        require(std::abs(run.forward()[0] - base) < 1e-12, "restore after FD");
    }
    fill({1.0, 101.0, 3.0, 100.0});
    close(run.forward()[0], base, "padding excluded from family SUM");
    fill({1.0, 100.0, 3.0, 100.0});
    require(std::abs(run.forward()[0] - base) < 1e-12, "restore after padding probe");
}

void multi_output_nsolid_pattern(MPI_Comm comm) {
    int rank = 0, size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    const auto sum = mpi_sum_reducer(comm);
    const std::vector<double> q_all = {0.2, 0.7, 0.9, 0.1, 0.55};
    std::vector<int> ids;
    for (int i = 0; i < static_cast<int>(q_all.size()); ++i) {
        const int owner = size == 1 ? 0 : 1 + (i * 3) % (size - 1);
        if (owner == rank) ids.push_back(i);
    }
    const int n = static_cast<int>(ids.size());
    GpuBuffer<double> vq(std::max(n, 1)), gq(std::max(n, 1));
    Leaf q(vq, gq, n);
    std::vector<double> local(n);
    for (int i = 0; i < n; ++i) local[i] = q_all[ids[i]];
    put(vq, local);

    Graph graph(Graph::Symbols{{"q", &q}}, {}, sum);
    // NSOLID + MEAN_SOLID 双输出；AVE 并存确认未改语义。
    graph.parse_expr("SUM(fermi(q,0.5,8))");
    graph.parse_expr("safe_div(SUM(q*fermi(q,0.5,8)), SUM(fermi(q,0.5,8)), 1e-15)");
    graph.parse_expr("AVE(q)");
    GraphExecution run(graph);
    double nsolid = 0.0, num = 0.0, den = 0.0, qsum = 0.0;
    for (double qi : q_all) {
        const double f = 1.0 / (1.0 + std::exp(-8.0 * (qi - 0.5)));
        nsolid += f;
        num += qi * f;
        den += f;
        qsum += qi;
    }
    const auto out = run.forward();
    close(out[0], nsolid, "NSOLID");
    close(out[1], num / den, "MEAN_SOLID");
    close(out[2], qsum / q_all.size(), "AVE unchanged");
    run.backward({1.0, 0.0, 0.0});
}
} // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    try {
        host_global_safe_div();
        zero_and_near_zero_denominators();
        graph_check_cuda(cudaSetDevice(0));
        mean_solid_nested_sum();
        per_atom_and_broadcast(MPI_COMM_WORLD);
        empty_ranks_and_global_empty(MPI_COMM_WORLD);
        family_exclusion_and_finite_difference(MPI_COMM_WORLD);
        multi_output_nsolid_pattern(MPI_COMM_WORLD);
        if (!rank)
            std::cout << "SUM_SAFE_DIV_PASS forward, FD, empty ranks, global empty SUM=0, "
                         "zero/near-zero denom, broadcast, family exclusion, multi-output, AVE preserved\n";
    } catch (const std::exception& error) {
        std::cerr << "rank " << rank << ": " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
}
