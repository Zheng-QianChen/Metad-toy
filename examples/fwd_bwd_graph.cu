// C = (mean_i q4_i * q6_i)^2
// 计算图与下面这份 Python 相同：
//   Leaf -> Mul -> Mean -> Square 前向
//   c.adj = 1 之后按相反顺序反向
//
// 反向写成赋值，不累加。每个节点只能有一个消费者。
// Mean 的求和按 Python 的 sum 用单线程做；Mul 按元素并行。
//
// nvcc -std=c++17 -O2 -arch=sm_89 -o fwd_bwd_graph examples/fwd_bwd_graph.cu && ./fwd_bwd_graph
// q4 = [1, 3], q6 = [2, 4] 时：C = 49，dC/dq4 = [14, 28]，dC/dq6 = [7, 21]

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

void ck(cudaError_t err) {
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

void sync_kernel() {
    ck(cudaGetLastError());
    ck(cudaDeviceSynchronize());
}

__global__ void k_mul(const double* a, const double* b, double* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] * b[i];
}

__global__ void k_mul_bwd(const double* adj, const double* a, const double* b,
                          double* a_adj, double* b_adj, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a_adj[i] = adj[i] * b[i];
        b_adj[i] = adj[i] * a[i];
    }
}

__global__ void k_mean(const double* x, int n, double* y) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += x[i];
    y[0] = s / static_cast<double>(n);
}

__global__ void k_mean_bwd(const double* adj, int n, double* src_adj) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) src_adj[i] = adj[0] / static_cast<double>(n);
}

__global__ void k_square(const double* x, double* y) { y[0] = x[0] * x[0]; }

__global__ void k_square_bwd(const double* adj, const double* x, double* src_adj) {
    src_adj[0] = adj[0] * 2.0 * x[0];
}

int grid_size(int n) { return (n + 255) / 256; }

struct DeviceBuf {
    double* p = nullptr;
    int n = 0;

    DeviceBuf() = default;
    explicit DeviceBuf(int n_) : n(n_) {
        if (n <= 0) throw std::runtime_error("empty buffer");
        ck(cudaMalloc(&p, sizeof(double) * static_cast<size_t>(n)));
        ck(cudaMemset(p, 0, sizeof(double) * static_cast<size_t>(n)));
    }
    DeviceBuf(const DeviceBuf&) = delete;
    DeviceBuf& operator=(const DeviceBuf&) = delete;
    ~DeviceBuf() { cudaFree(p); }

    void copy_from(const std::vector<double>& host) const {
        ck(cudaMemcpy(p, host.data(), sizeof(double) * static_cast<size_t>(n),
                      cudaMemcpyHostToDevice));
    }
};

struct Node {
    double* val = nullptr;
    double* adj = nullptr;
    int n = 0;

    Node() = default;
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;
    virtual void forward() = 0;
    virtual void backward() = 0;
    virtual ~Node() = default;
};

struct Leaf : Node {
    DeviceBuf xbuf;
    DeviceBuf abuf;
    double* g = nullptr;

    explicit Leaf(const std::vector<double>& host)
        : xbuf(static_cast<int>(host.size())), abuf(static_cast<int>(host.size())) {
        n = xbuf.n;
        xbuf.copy_from(host);
        val = xbuf.p;
        adj = abuf.p;
    }
    void forward() override { val = xbuf.p; }
    void backward() override { g = adj; }
};

struct Mul : Node {
    Node* a;
    Node* b;
    DeviceBuf vbuf;
    DeviceBuf abuf;

    Mul(Node* a_, Node* b_) : a(a_), b(b_), vbuf(a_->n), abuf(a_->n) {
        if (a->n != b->n) throw std::runtime_error("Mul: length mismatch");
        n = a->n;
        val = vbuf.p;
        adj = abuf.p;
    }
    void forward() override {
        k_mul<<<grid_size(n), 256>>>(a->val, b->val, val, n);
        sync_kernel();
    }
    void backward() override {
        k_mul_bwd<<<grid_size(n), 256>>>(adj, a->val, b->val, a->adj, b->adj, n);
        sync_kernel();
    }
};

struct Mean : Node {
    Node* src;
    int src_n = 0;
    DeviceBuf vbuf;
    DeviceBuf abuf;

    explicit Mean(Node* src_) : src(src_), vbuf(1), abuf(1) {
        n = 1;
        val = vbuf.p;
        adj = abuf.p;
    }
    void forward() override {
        src_n = src->n;
        k_mean<<<1, 1>>>(src->val, src_n, val);
        sync_kernel();
    }
    void backward() override {
        k_mean_bwd<<<grid_size(src_n), 256>>>(adj, src_n, src->adj);
        sync_kernel();
    }
};

struct Square : Node {
    Node* src;
    DeviceBuf vbuf;
    DeviceBuf abuf;

    explicit Square(Node* src_) : src(src_), vbuf(1), abuf(1) {
        n = 1;
        val = vbuf.p;
        adj = abuf.p;
    }
    void forward() override {
        k_square<<<1, 1>>>(src->val, val);
        sync_kernel();
    }
    void backward() override {
        k_square_bwd<<<1, 1>>>(adj, src->val, src->adj);
        sync_kernel();
    }
};

struct Built {
    std::vector<std::unique_ptr<Node>> own;
    std::vector<Node*> order;
    Leaf* q4 = nullptr;
    Leaf* q6 = nullptr;
    Node* c = nullptr;
};

Built build(const std::vector<double>& q4v, const std::vector<double>& q6v) {
    Built g;
    auto q4 = std::make_unique<Leaf>(q4v);
    auto q6 = std::make_unique<Leaf>(q6v);
    auto prod = std::make_unique<Mul>(q4.get(), q6.get());
    auto mean = std::make_unique<Mean>(prod.get());
    auto c = std::make_unique<Square>(mean.get());
    g.q4 = q4.get();
    g.q6 = q6.get();
    g.c = c.get();
    g.order = {g.q4, g.q6, prod.get(), mean.get(), c.get()};
    g.own.push_back(std::move(q4));
    g.own.push_back(std::move(q6));
    g.own.push_back(std::move(prod));
    g.own.push_back(std::move(mean));
    g.own.push_back(std::move(c));
    return g;
}

double load_scalar(const double* p) {
    double host = 0.0;
    ck(cudaMemcpy(&host, p, sizeof(double), cudaMemcpyDeviceToHost));
    return host;
}

std::vector<double> load_vector(const double* p, int n) {
    std::vector<double> host(static_cast<size_t>(n));
    ck(cudaMemcpy(host.data(), p, sizeof(double) * static_cast<size_t>(n),
                  cudaMemcpyDeviceToHost));
    return host;
}

double C_of(const std::vector<double>& q4v, const std::vector<double>& q6v) {
    Built g = build(q4v, q6v);
    for (Node* node : g.order) node->forward();
    return load_scalar(g.c->val);
}

std::tuple<double, std::vector<double>, std::vector<double>> backprop(
    const std::vector<double>& q4v, const std::vector<double>& q6v) {
    Built g = build(q4v, q6v);
    for (Node* node : g.order) node->forward();
    const double one = 1.0;
    ck(cudaMemcpy(g.c->adj, &one, sizeof(double), cudaMemcpyHostToDevice));
    for (auto it = g.order.rbegin(); it != g.order.rend(); ++it) (*it)->backward();
    return {load_scalar(g.c->val), load_vector(g.q4->g, g.q4->n), load_vector(g.q6->g, g.q6->n)};
}

void print_vec(const char* name, const std::vector<double>& v) {
    std::printf("%s [", name);
    for (size_t i = 0; i < v.size(); ++i) std::printf("%s%.16g", i ? ", " : "", v[i]);
    std::printf("]\n");
}

bool close(const std::vector<double>& a, const std::vector<double>& b, double tol) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > tol) return false;
    }
    return true;
}

int main() {
    try {
        const std::vector<double> q4 = {1.0, 3.0};
        const std::vector<double> q6 = {2.0, 4.0};
        const auto [value, g4, g6] = backprop(q4, q6);

        const double eps = 1e-6;
        std::vector<double> fd4(q4.size()), fd6(q6.size());
        for (size_t a = 0; a < q4.size(); ++a) {
            auto plus = q4;
            auto minus = q4;
            plus[a] += eps;
            minus[a] -= eps;
            fd4[a] = (C_of(plus, q6) - C_of(minus, q6)) / (2.0 * eps);

            plus = q6;
            minus = q6;
            plus[a] += eps;
            minus[a] -= eps;
            fd6[a] = (C_of(q4, plus) - C_of(q4, minus)) / (2.0 * eps);
        }

        std::printf("C = %.16g\n", value);
        print_vec("dC/dq4 反向 =", g4);
        print_vec("dC/dq4 差分 =", fd4);
        print_vec("dC/dq6 反向 =", g6);
        print_vec("dC/dq6 差分 =", fd6);

        if (!close(g4, fd4, 1e-6) || !close(g6, fd6, 1e-6)) return 2;
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
