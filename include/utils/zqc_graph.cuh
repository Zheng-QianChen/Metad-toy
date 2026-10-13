#pragma once

// Experimental graph prototype: node evaluation, source generation and parsing.
// Standalone programs and checks live in benchmarks/graph/.
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <limits>
#include <cmath>
#include <cuda_runtime.h>
#include <memory>
#include <cctype>
#include <stdexcept>
#include <functional>
#include <iostream>
#include <fmt/printf.h>

#include "zqc_graph_device.cuh"
#include "zqc_switch_function_device.cuh"
#include "zqc_gpu_buffer.cuh"

using MetaD_zqc::GpuBuffer;

// xxd 自动生成的数组由一个独立的 .cpp 定义，不在这里重复保存源码。
extern unsigned char zqc_graph_device_source[];
extern unsigned int zqc_graph_device_source_len;

inline std::string graph_device_source()
{
    return std::string(
        reinterpret_cast<const char*>(zqc_graph_device_source),
        zqc_graph_device_source_len);
}

inline void graph_check_cuda(cudaError_t status)
{
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

inline void sync_kernel()
{
    graph_check_cuda(cudaGetLastError());
    graph_check_cuda(cudaDeviceSynchronize());
}

int block_size=256;
int grid_size(int n) { return (n + (block_size-1)) / block_size; }

__global__ void k_mul(const double* a, const double* b, double* y, int n);
__global__ void k_mul_bwd(const double* adj, const double* a, const double* b,
                          double* a_adj, double* b_adj, int n);
__global__ void k_add(const double* a, const double* b, double* y, int n);
__global__ void k_add_bwd(const double* adj, double* a_adj, double* b_adj, int n);
__global__ void k_neg(const double* a, double* y, int n);
__global__ void k_neg_bwd(const double* adj, double* a_adj, int n);
__global__ void k_linear(const double* a, const double k0, const double b0, double* y, int n);
__global__ void k_linear_bwd(const double* adj, const double* a,
                            const double k0, const double b0,
                          double* a_adj, int n);
__global__ void k_exp(const double* a, double* y, int n);
__global__ void k_exp_bwd(const double* adj, const double* a, double* a_adj, int n);
__global__ void k_constant_one(const double* a, double* y, int n);
__global__ void k_constant_one_bwd(const double* adj, const double* a, double* a_adj, int n);
__global__ void k_step_switch(const double* a, double* y, double r0, int n);
__global__ void k_step_switch_bwd(const double* adj, const double* a, double* a_adj, double r0, int n);
__global__ void k_fermi(const double* a, double* y, double r0, double alpha, int n);
__global__ void k_fermi_bwd(const double* adj, const double* a, double* a_adj, double r0, double alpha, int n);
__global__ void k_tanh_switch(const double* a, double* y, double r0, double alpha, int n);
__global__ void k_tanh_switch_bwd(const double* adj, const double* a, double* a_adj, double r0, double alpha, int n);
__global__ void k_rational_switch(const double* a, double* y, double r0, int power_n, int power_m, double d0, int n);
__global__ void k_rational_switch_bwd(const double* adj, const double* a, double* a_adj, double r0, int power_n, int power_m, double d0, int n);
__global__ void k_mean(const double* x, int n, double* y);
__global__ void k_mean_bwd(const double* adj, int n, double* src_adj);

struct Graph;
struct Codegen;
struct Node;
class GraphExprParser;

enum class Domain {
    PerAtom,  // 每个原子一份
    Global    // 整个系统一份
};

struct ValueDesc {
    Domain domain;
    int components;  // 1：标量；3：三分量向量
};

struct Node {
    ValueDesc value_desc;
    double* val = nullptr; // 节点的值
    double* adj = nullptr; // 节点的导数
    int n = 0;

    Node(ValueDesc desc);
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;
    virtual void forward() = 0;
    virtual void backward() = 0;
    virtual ~Node() = default;

    // 节点连接：列出输入节点
    virtual std::vector<const Node*> inputs() const = 0;

    // 节点生成方法
    virtual std::string emit_expr(Codegen& cg) const = 0;
    virtual std::string emit_expr_bwd(Codegen& cg) const = 0;
};
inline Node::Node(ValueDesc desc)
    : value_desc(desc)
{
    if (value_desc.components <= 0)
        throw std::runtime_error("Invalid component count");
}

// 负责kernel整合
struct Codegen {
    // 每项保存：叶子指针、参数名，例如 (&A, "v0")
    std::vector<std::pair<const Node*, std::string>> inputs;
    // 保存参数
    std::unordered_map<const Node*, std::string> value_names;
    std::unordered_map<const Node*, std::string> adjoint_names;

    // kernel 中反向的声明
    std::string adjoint_decls;
    // kernel 中的函数体
    std::string body;
    int next_val_id = 0;
    // 梯度变量单独编号
    int next_adj_id = 0;

    std::string value(const Node* node);
    std::string adjoint(const Node* node);
};
std::string Codegen::value(const Node* node){
    // 已经生成过，直接返回它的变量名
    auto it = value_names.find(node);
    if (it != value_names.end())
        return it->second;

    // 让节点生成表达式。
    // 如果节点还有输入，这里会递归调用 cg.value(输入节点)
    std::string expr = node->emit_expr(*this);
    // Leaf 已经登记自己的名字，直接使用。
    // 它的声明由输入加载阶段负责。
    it = value_names.find(node);
    if (it != value_names.end())
        return it->second;

    // 为当前节点分配变量名
    std::string name = fmt::sprintf("v%d", next_val_id++);

    // 把计算语句加入源码
    body += fmt::sprintf(
        "double %s = %s;\n",
        name, expr);

    // 在这里记录：节点 → 变量名
    value_names[node] = name;

    return name;
}
std::string Codegen::adjoint(const Node* node){
    auto it = adjoint_names.find(node);
    if (it != adjoint_names.end())
        return it->second;
    std::string name = fmt::sprintf("g%d", next_adj_id++);
    adjoint_decls += fmt::sprintf(
        "double %s = 0.0;\n", name
    );
    adjoint_names.emplace(node, name);
    return name;
}

inline ValueDesc infer_broadcast(ValueDesc a, ValueDesc b){
    if (a.components <= 0 || b.components <= 0)
        throw std::runtime_error("Invalid component count");

    int components;

    if (a.components == b.components) {
        components = a.components;
    } else if (a.components == 1) {
        components = b.components;
    } else if (b.components == 1) {
        components = a.components;
    } else {
        throw std::runtime_error(
            "Incompatible component counts for broadcasting");
    }

    // 任意一个输入是逐原子值，输出就是逐原子值。
    Domain domain =
        (a.domain == Domain::PerAtom ||
         b.domain == Domain::PerAtom)
        ? Domain::PerAtom
        : Domain::Global;

    return {domain, components};
}

// 叶子节点(最终节点)
struct Leaf : Node {
    GpuBuffer<double> xbuf; // host 便利构造使用；借用外部输入时为空
    GpuBuffer<double> abuf; // 未传入外部梯度时，由 Leaf 持有
    double* g = nullptr;

    // 借用缓冲区，不复制、不释放外部显存。count 是有效元素数，不是 capacity。
    // 调用方保证外部 GpuBuffer 对象在 Leaf 使用期间始终存活。
    explicit Leaf(GpuBuffer<double>& values, int count,
                  ValueDesc desc = {Domain::PerAtom, 1})
        : Node(desc), abuf(checked_count(count, desc)),
          input_buffer_(&values) {
        n = count;
        refresh_buffers();
    }

    // 可将梯度直接写入调用方提供的 GpuBuffer。
    Leaf(GpuBuffer<double>& values, GpuBuffer<double>& gradients, int count,
         ValueDesc desc = {Domain::PerAtom, 1})
        : Node(desc), input_buffer_(&values), gradient_buffer_(&gradients) {
        n = checked_count(count, desc);
        refresh_buffers();
    }

    // 小样例仍可从 host 初始化，内部也使用同一个 GpuBuffer 类型。
    explicit Leaf(const std::vector<double>& host,
                  ValueDesc desc = {Domain::PerAtom, 1})
        : Node(desc), xbuf(checked_count(host.size(), desc)), abuf(host.size()) {
        n = static_cast<int>(host.size());
        graph_check_cuda(cudaMemcpy(xbuf.ptr, host.data(), host.size() * sizeof(double),
                                   cudaMemcpyHostToDevice));
        refresh_buffers();
    }

    // 外部 grow_to 可能更换 ptr；启动 kernel、收集参数前刷新一次。
    // 有效元素数变化时需要重建/更新图；这里只处理相同 n 下的指针变化。
    void refresh_buffers() {
        auto& values = input_buffer_ ? *input_buffer_ : xbuf;
        auto& gradients = gradient_buffer_ ? *gradient_buffer_ : abuf;
        if (!values.ptr || values.capacity < static_cast<size_t>(n))
            throw std::runtime_error("Leaf: input buffer is too small");
        if (!gradients.ptr || gradients.capacity < static_cast<size_t>(n))
            throw std::runtime_error("Leaf: gradient buffer is too small");
        val = values.ptr;
        adj = gradients.ptr;
        if (g) g = adj;
    }

    void forward() override { refresh_buffers(); }
    void backward() override { g = adj; }

    std::vector<const Node*> inputs() const override;
    std::string emit_expr(Codegen& cg) const override;
    std::string emit_expr_bwd(Codegen& cg) const override;

private:
    GpuBuffer<double>* input_buffer_ = nullptr;
    GpuBuffer<double>* gradient_buffer_ = nullptr;

    static int checked_count(size_t count, ValueDesc desc) {
        if (count == 0 || count > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("Leaf: invalid element count");
        if (desc.components <= 0)
            throw std::runtime_error("Leaf: invalid components");
        if (desc.domain == Domain::Global) {
            if (count != static_cast<size_t>(desc.components))
                throw std::runtime_error("Leaf: global size mismatch");
        } else if (count % static_cast<size_t>(desc.components) != 0) {
            throw std::runtime_error("Leaf: incomplete atom components");
        }
        return static_cast<int>(count);
    }
};
inline std::string Leaf::emit_expr(Codegen& cg) const{
    auto it = cg.value_names.find(this);
    if (it != cg.value_names.end())
        return it->second;

    std::string name = fmt::sprintf(
        "v%d", cg.next_val_id++);

    cg.value_names.emplace(this, name);
    cg.inputs.emplace_back(this, name);

    return name;
}
inline std::string Leaf::emit_expr_bwd(Codegen&) const{
    return {};
}
std::vector<const Node*> Leaf::inputs() const {
    return {};
}

// 逐原子乘的计算节点
struct Mul : Node {
    Node* a; // 输入节点1
    Node* b; // 输入节点2
    GpuBuffer<double> vbuf;
    GpuBuffer<double> abuf;

    Mul(Node* a_, Node* b_) : Node(infer_broadcast(a_->value_desc, b_->value_desc)),
        a(a_), b(b_), vbuf(a_->n), abuf(a_->n) {
        if (a->n != b->n) throw std::runtime_error("Mul: length mismatch");
        n = a->n;
        val = vbuf.ptr;
        adj = abuf.ptr;
    }
    void forward() override {
        k_mul<<<grid_size(n), block_size>>>(a->val, b->val, val, n);
        sync_kernel();
    }
    void backward() override {
        k_mul_bwd<<<grid_size(n), block_size>>>(adj, a->val, b->val, a->adj, b->adj, n);
        sync_kernel();
    }

    std::vector<const Node*> inputs() const override;
    std::string emit_expr(Codegen& cg) const override;
    std::string emit_expr_bwd(Codegen& cg) const override;
};
__global__ void k_mul(const double* a, const double* b, double* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = zqc_graph_mul(a[i], b[i]);
}
__global__ void k_mul_bwd(const double* adj, const double* a, const double* b,
                          double* a_adj, double* b_adj, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        zqc_graph_mul_bwd(adj[i], a[i], b[i], &a_adj[i], &b_adj[i]);
    }
}
std::string Mul::emit_expr(Codegen& cg) const{
    // 获取对应的变量名字符串
    // 其中，a和b是输入节点，可能是叶子节点，也可能是计算节点
    auto av = cg.value(a);
    auto bv = cg.value(b);
    return fmt::sprintf("zqc_graph_mul( %s , %s )", av, bv);
}
std::string Mul::emit_expr_bwd(Codegen& cg) const{
    auto av = cg.value(a);
    auto bv = cg.value(b);

    auto gv  = cg.adjoint(this);
    auto gav = cg.adjoint(a);
    auto gbv = cg.adjoint(b);
    return fmt::sprintf(
        "zqc_graph_mul_bwd(%s, %s, %s, &(%s), &(%s));\n",
        gv, av, bv, gav, gbv);
}
std::vector<const Node*> Mul::inputs() const {
    return {a, b};
}

// 逐原子加的计算节点
struct Add : Node {
    Node* a; // 输入节点1
    Node* b; // 输入节点2
    GpuBuffer<double> vbuf;
    GpuBuffer<double> abuf;

    Add(Node* a_, Node* b_) : Node(infer_broadcast(a_->value_desc, b_->value_desc)),
        a(a_), b(b_), vbuf(a_->n), abuf(a_->n) {
        if (a->n != b->n) throw std::runtime_error("Mul: length mismatch");
        n = a->n;
        val = vbuf.ptr;
        adj = abuf.ptr;
    }
    void forward() override {
        k_add<<<grid_size(n), block_size>>>(a->val, b->val, val, n);
        sync_kernel();
    }
    void backward() override {
        k_add_bwd<<<grid_size(n), block_size>>>(adj, a->adj, b->adj, n);
        sync_kernel();
    }

    std::vector<const Node*> inputs() const override;
    std::string emit_expr(Codegen& cg) const override;
    std::string emit_expr_bwd(Codegen& cg) const override;
};
__global__ void k_add(const double* a, const double* b, double* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = zqc_graph_add(a[i] , b[i]);
}
__global__ void k_add_bwd(const double* adj, double* a_adj, double* b_adj, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        zqc_graph_add_bwd(adj[i], &a_adj[i], &b_adj[i]);
    }
}
std::string Add::emit_expr(Codegen& cg) const{
    // 获取对应的变量名字符串
    // 其中，a和b是输入节点，可能是叶子节点，也可能是计算节点
    auto av = cg.value(a);
    auto bv = cg.value(b);
    return fmt::sprintf("zqc_graph_add( %s , %s )", av, bv);
}
std::string Add::emit_expr_bwd(Codegen& cg) const{
    auto gv  = cg.adjoint(this);
    auto gav = cg.adjoint(a);
    auto gbv = cg.adjoint(b);
    return fmt::sprintf(
        "zqc_graph_add_bwd(%s, &(%s), &(%s));\n",
        gv, gav, gbv);
}
std::vector<const Node*> Add::inputs() const {
    return {a, b};
}

// 逐元素取负：y = -a，保持输入的 domain、components 和长度。
struct Neg : Node {
    Node* a;
    GpuBuffer<double> vbuf;
    GpuBuffer<double> abuf;

    explicit Neg(Node* a_) : Node(a_->value_desc),
                            a(a_), vbuf(a_->n), abuf(a_->n) {
        n = a->n;
        val = vbuf.ptr;
        adj = abuf.ptr;
    }
    void forward() override {
        k_neg<<<grid_size(n), block_size>>>(a->val, val, n);
        sync_kernel();
    }
    void backward() override {
        k_neg_bwd<<<grid_size(n), block_size>>>(adj, a->adj, n);
        sync_kernel();
    }

    std::vector<const Node*> inputs() const override;
    std::string emit_expr(Codegen& cg) const override;
    std::string emit_expr_bwd(Codegen& cg) const override;
};
__global__ void k_neg(const double* a, double* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = zqc_graph_neg(a[i]);
}
__global__ void k_neg_bwd(const double* adj, double* a_adj, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) zqc_graph_neg_bwd(adj[i], &a_adj[i]);
}
std::string Neg::emit_expr(Codegen& cg) const {
    auto av = cg.value(a);
    return fmt::sprintf("zqc_graph_neg( %s )", av);
}
std::string Neg::emit_expr_bwd(Codegen& cg) const {
    auto gv = cg.adjoint(this);
    auto gav = cg.adjoint(a);
    return fmt::sprintf("zqc_graph_neg_bwd(%s, &(%s));\n", gv, gav);
}
std::vector<const Node*> Neg::inputs() const {
    return {a};
}

// 逐原子线性计算的计算节点
struct Linear : Node {
    Node* a; // 输入节点1
    double k0, b0; // 输入节点2
    GpuBuffer<double> vbuf;
    GpuBuffer<double> abuf;

    Linear(Node* a_, double k0_, double b0_) : Node(a_->value_desc),
                                            a(a_), vbuf(a_->n), abuf(a_->n),
                                            k0(k0_), b0(b0_) {
        n = a->n;
        val = vbuf.ptr;
        adj = abuf.ptr;
    }
    void forward() override {
        k_linear<<<grid_size(n), block_size>>>(a->val, k0, b0, val, n);
        sync_kernel();
    }
    void backward() override {
        k_linear_bwd<<<grid_size(n), block_size>>>(adj, a->val, k0, b0, a->adj, n);
        sync_kernel();
    }

    std::vector<const Node*> inputs() const override;
    std::string emit_expr(Codegen& cg) const override;
    std::string emit_expr_bwd(Codegen& cg) const override;
};
__global__ void k_linear(const double* a, const double k0, const double b0, double* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = zqc_graph_linear(a[i], k0, b0);
}
__global__ void k_linear_bwd(const double* adj, const double* a,
                            const double k0, const double b0,
                          double* a_adj, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        zqc_graph_linear_bwd(adj[i], k0, &a_adj[i]);
    }
}
std::string Linear::emit_expr(Codegen& cg) const{
    // 获取对应的变量名字符串
    // 其中，a和b是输入节点，可能是叶子节点，也可能是计算节点
    auto av = cg.value(a);
    return fmt::sprintf("zqc_graph_linear( %s , %.18g, %.18g )", av, k0, b0);
}
std::string Linear::emit_expr_bwd(Codegen& cg) const{
    auto av = cg.value(a);

    auto gv  = cg.adjoint(this);
    auto gav = cg.adjoint(a);
    return fmt::sprintf(
        "zqc_graph_linear_bwd(%s, %.18g, &(%s));\n",
        gv, k0, gav);
}
std::vector<const Node*> Linear::inputs() const {
    return {a};
}

// 逐元素指数函数
struct Exp : Node {
    Node* a; // 后续节点
    GpuBuffer<double> vbuf;
    GpuBuffer<double> abuf;

    Exp(Node* a_) :  Node(a_->value_desc),
                    a(a_), vbuf(a_->n), abuf(a_->n){
        n = a->n;
        val = vbuf.ptr;
        adj = abuf.ptr;
    }
    void forward() override {
        k_exp<<<grid_size(n), block_size>>>(a->val, val, n);
        sync_kernel();
    }
    void backward() override {
        k_exp_bwd<<<grid_size(n), block_size>>>(adj, a->val, a->adj, n);
        sync_kernel();
    }

    std::vector<const Node*> inputs() const override;
    std::string emit_expr(Codegen& cg) const override;
    std::string emit_expr_bwd(Codegen& cg) const override;
};
__global__ void k_exp(const double* a, double* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = zqc_graph_exp(a[i]);
}
__global__ void k_exp_bwd(const double* adj, const double* a,
                          double* a_adj, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        zqc_graph_exp_bwd(adj[i], a[i], &a_adj[i]);
    }
}
std::string Exp::emit_expr(Codegen& cg) const{
    // 获取对应的变量名字符串
    // 其中，a和b是输入节点，可能是叶子节点，也可能是计算节点
    auto av = cg.value(a);
    return fmt::sprintf("zqc_graph_exp( %s )", av);
}
std::string Exp::emit_expr_bwd(Codegen& cg) const{
    auto av = cg.value(a);

    auto gv  = cg.adjoint(this);
    auto gav = cg.adjoint(a);
    return fmt::sprintf(
        "zqc_graph_exp_bwd(%s, %s, &(%s));\n",
        gv, av, gav);
}
std::vector<const Node*> Exp::inputs() const {
    return {a};
}

// Concrete elementwise nodes. Each captures only its own parameters.
struct ConstantOne : Node {
    Node* a;
    GpuBuffer<double> vbuf;
    GpuBuffer<double> abuf;

    explicit ConstantOne(Node* a_)
        : Node(a_->value_desc), a(a_), vbuf(a_->n), abuf(a_->n) {
        n = a->n;
        val = vbuf.ptr;
        adj = abuf.ptr;
    }
    void forward() override {
        k_constant_one<<<grid_size(n), block_size>>>(a->val, val, n);
        sync_kernel();
    }
    void backward() override {
        k_constant_one_bwd<<<grid_size(n), block_size>>>(adj, a->val, a->adj, n);
        sync_kernel();
    }
    std::vector<const Node*> inputs() const override { return {a}; }
    std::string emit_expr(Codegen& cg) const override {
        return fmt::sprintf("zqc_graph_constant_one(%s)", cg.value(a));
    }
    std::string emit_expr_bwd(Codegen& cg) const override {
        const auto av = cg.value(a);
        const auto gv = cg.adjoint(this);
        const auto gav = cg.adjoint(a);
        return fmt::sprintf("zqc_graph_constant_one_bwd(%s, %s, &(%s));\n",
                            gv, av, gav);
    }
};
__global__ void k_constant_one(const double* a, double* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = zqc_graph_constant_one(a[i]);
}
__global__ void k_constant_one_bwd(const double* adj, const double* a, double* a_adj, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) zqc_graph_constant_one_bwd(adj[i], a[i], &a_adj[i]);
}

struct StepSwitch : Node {
    Node* a;
    const double r0;
    GpuBuffer<double> vbuf;
    GpuBuffer<double> abuf;

    explicit StepSwitch(Node* a_, double r0_)
        : Node(a_->value_desc), a(a_), r0(r0_), vbuf(a_->n), abuf(a_->n) {
        if (!std::isfinite(r0)) throw std::runtime_error("StepSwitch: r0 must be finite");
        n = a->n;
        val = vbuf.ptr;
        adj = abuf.ptr;
    }
    void forward() override {
        k_step_switch<<<grid_size(n), block_size>>>(a->val, val, r0, n);
        sync_kernel();
    }
    void backward() override {
        k_step_switch_bwd<<<grid_size(n), block_size>>>(adj, a->val, a->adj, r0, n);
        sync_kernel();
    }
    std::vector<const Node*> inputs() const override { return {a}; }
    std::string emit_expr(Codegen& cg) const override {
        return fmt::sprintf("zqc_graph_step_switch(%s, %.18g)", cg.value(a), r0);
    }
    std::string emit_expr_bwd(Codegen& cg) const override {
        const auto av = cg.value(a);
        const auto gv = cg.adjoint(this);
        const auto gav = cg.adjoint(a);
        return fmt::sprintf("zqc_graph_step_switch_bwd(%s, %s, %.18g, &(%s));\n",
                            gv, av, r0, gav);
    }
};
__global__ void k_step_switch(const double* a, double* y, double r0, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = zqc_graph_step_switch(a[i], r0);
}
__global__ void k_step_switch_bwd(const double* adj, const double* a, double* a_adj, double r0, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) zqc_graph_step_switch_bwd(adj[i], a[i], r0, &a_adj[i]);
}

struct Fermi : Node {
    Node* a;
    const double r0;
    const double alpha;
    GpuBuffer<double> vbuf;
    GpuBuffer<double> abuf;

    explicit Fermi(Node* a_, double r0_, double alpha_)
        : Node(a_->value_desc), a(a_), r0(r0_), alpha(alpha_), vbuf(a_->n), abuf(a_->n) {
        if (!std::isfinite(r0) || !std::isfinite(alpha)) throw std::runtime_error("Fermi: parameters must be finite");
        n = a->n;
        val = vbuf.ptr;
        adj = abuf.ptr;
    }
    void forward() override {
        k_fermi<<<grid_size(n), block_size>>>(a->val, val, r0, alpha, n);
        sync_kernel();
    }
    void backward() override {
        k_fermi_bwd<<<grid_size(n), block_size>>>(adj, a->val, a->adj, r0, alpha, n);
        sync_kernel();
    }
    std::vector<const Node*> inputs() const override { return {a}; }
    std::string emit_expr(Codegen& cg) const override {
        return fmt::sprintf("zqc_graph_fermi(%s, %.18g, %.18g)", cg.value(a), r0, alpha);
    }
    std::string emit_expr_bwd(Codegen& cg) const override {
        const auto av = cg.value(a);
        const auto gv = cg.adjoint(this);
        const auto gav = cg.adjoint(a);
        return fmt::sprintf("zqc_graph_fermi_bwd(%s, %s, %.18g, %.18g, &(%s));\n",
                            gv, av, r0, alpha, gav);
    }
};
__global__ void k_fermi(const double* a, double* y, double r0, double alpha, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = zqc_graph_fermi(a[i], r0, alpha);
}
__global__ void k_fermi_bwd(const double* adj, const double* a, double* a_adj, double r0, double alpha, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) zqc_graph_fermi_bwd(adj[i], a[i], r0, alpha, &a_adj[i]);
}

struct TanhSwitch : Node {
    Node* a;
    const double r0;
    const double alpha;
    GpuBuffer<double> vbuf;
    GpuBuffer<double> abuf;

    explicit TanhSwitch(Node* a_, double r0_, double alpha_)
        : Node(a_->value_desc), a(a_), r0(r0_), alpha(alpha_), vbuf(a_->n), abuf(a_->n) {
        if (!std::isfinite(r0) || !std::isfinite(alpha)) throw std::runtime_error("TanhSwitch: parameters must be finite");
        n = a->n;
        val = vbuf.ptr;
        adj = abuf.ptr;
    }
    void forward() override {
        k_tanh_switch<<<grid_size(n), block_size>>>(a->val, val, r0, alpha, n);
        sync_kernel();
    }
    void backward() override {
        k_tanh_switch_bwd<<<grid_size(n), block_size>>>(adj, a->val, a->adj, r0, alpha, n);
        sync_kernel();
    }
    std::vector<const Node*> inputs() const override { return {a}; }
    std::string emit_expr(Codegen& cg) const override {
        return fmt::sprintf("zqc_graph_tanh_switch(%s, %.18g, %.18g)", cg.value(a), r0, alpha);
    }
    std::string emit_expr_bwd(Codegen& cg) const override {
        const auto av = cg.value(a);
        const auto gv = cg.adjoint(this);
        const auto gav = cg.adjoint(a);
        return fmt::sprintf("zqc_graph_tanh_switch_bwd(%s, %s, %.18g, %.18g, &(%s));\n",
                            gv, av, r0, alpha, gav);
    }
};
__global__ void k_tanh_switch(const double* a, double* y, double r0, double alpha, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = zqc_graph_tanh_switch(a[i], r0, alpha);
}
__global__ void k_tanh_switch_bwd(const double* adj, const double* a, double* a_adj, double r0, double alpha, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) zqc_graph_tanh_switch_bwd(adj[i], a[i], r0, alpha, &a_adj[i]);
}

struct RationalSwitch : Node {
    Node* a;
    const double r0;
    const int power_n;
    const int power_m;
    const double d0;
    GpuBuffer<double> vbuf;
    GpuBuffer<double> abuf;

    explicit RationalSwitch(Node* a_, double r0_, int power_n_, int power_m_, double d0_ = 0.0)
        : Node(a_->value_desc), a(a_), r0(r0_), power_n(power_n_), power_m(power_m_), d0(d0_), vbuf(a_->n), abuf(a_->n) {
        if (!std::isfinite(r0) || !std::isfinite(d0) || r0 <= 0.0 || power_n <= 0 || power_m <= 0)
            throw std::runtime_error("RationalSwitch: finite r0 > 0, finite d0, and positive powers required");
        n = a->n;
        val = vbuf.ptr;
        adj = abuf.ptr;
    }
    void forward() override {
        k_rational_switch<<<grid_size(n), block_size>>>(a->val, val, r0, power_n, power_m, d0, n);
        sync_kernel();
    }
    void backward() override {
        k_rational_switch_bwd<<<grid_size(n), block_size>>>(adj, a->val, a->adj, r0, power_n, power_m, d0, n);
        sync_kernel();
    }
    std::vector<const Node*> inputs() const override { return {a}; }
    std::string emit_expr(Codegen& cg) const override {
        return fmt::sprintf("zqc_graph_rational_switch(%s, %.18g, %d, %d, %.18g)", cg.value(a), r0, power_n, power_m, d0);
    }
    std::string emit_expr_bwd(Codegen& cg) const override {
        const auto av = cg.value(a);
        const auto gv = cg.adjoint(this);
        const auto gav = cg.adjoint(a);
        return fmt::sprintf("zqc_graph_rational_switch_bwd(%s, %s, %.18g, %d, %d, %.18g, &(%s));\n",
                            gv, av, r0, power_n, power_m, d0, gav);
    }
};
__global__ void k_rational_switch(const double* a, double* y, double r0, int power_n, int power_m, double d0, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = zqc_graph_rational_switch(a[i], r0, power_n, power_m, d0);
}
__global__ void k_rational_switch_bwd(const double* adj, const double* a, double* a_adj, double r0, int power_n, int power_m, double d0, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) zqc_graph_rational_switch_bwd(adj[i], a[i], r0, power_n, power_m, d0, &a_adj[i]);
}

// 均值节点：融合代码只输出规约前的值，规约由调用方完成。
struct MEAN : Node {
    Node* src;
    int src_n = 0;
    GpuBuffer<double> vbuf;
    GpuBuffer<double> abuf;

    explicit MEAN(Node* src_) : Node({Domain::Global, 1}), src(src_), vbuf(1), abuf(1) {
        n = 1;
        val = vbuf.ptr;
        adj = abuf.ptr;
    }
    void forward() override {
        src_n = src->n;
        k_mean<<<1, 1>>>(src->val, src_n, val);
        sync_kernel();
    }
    void backward() override {
        k_mean_bwd<<<grid_size(src_n), block_size>>>(adj, src_n, src->adj);
        sync_kernel();
    }

    std::vector<const Node*> inputs() const override;
    std::string emit_expr(Codegen& cg) const override;
    std::string emit_expr_bwd(Codegen& cg) const override;
};
__global__ void k_mean(const double* x, int n, double* y) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += x[i];
    y[0] = s / static_cast<double>(n);
}
__global__ void k_mean_bwd(const double* adj, int n, double* src_adj) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) src_adj[i] = adj[0] / static_cast<double>(n);
}
std::string MEAN::emit_expr(Codegen&) const {
    throw std::logic_error(
        "MEAN must be handled by the reduction stage");
}
std::string MEAN::emit_expr_bwd(Codegen&) const {
    throw std::logic_error(
        "MEAN backward must be handled by the reduction stage");
}
std::vector<const Node*> MEAN::inputs() const {
    return {src};
}

struct Graph {
    using Symbols = std::unordered_map<std::string, Node*>;
    using Switches = std::unordered_map<std::string, MetaD_zqc::SwitchFunctionRequest>;

    Symbols symbols;
    Switches switches; // Named configurations resolved to concrete nodes during parsing.

    // 外部传入的 Leaf 不归 Graph 所有。
    // 这里只保存解析过程中创建的节点。
    std::vector<std::unique_ptr<Node>> owned_nodes;

    std::vector<Node*> outputs;
    std::vector<const Node*> kernel_outputs; // 与上面一一对应，普通输出保存自身，Mean保存src
    std::vector<const Node*> topo;

    std::string kernel_source; // 交给 nvrtc 的源码
    std::string emit_kernel(bool backward);

    explicit Graph(Symbols symbols_, Switches switches_ = {})
        : symbols(std::move(symbols_)), switches(std::move(switches_)) {}

    explicit Graph(std::vector<Node*> outputs_)
        : outputs(std::move(outputs_)) {}

    // 创建节点：
    // 例如已有叶子节点a,b, 然后 cv = a*b;
    // Node* result = graph.make_node<Mul>(a,b);
    template<class T, class... Args>
    T* make_node(Args&&... args)
    {
        auto node = std::make_unique<T>(
            std::forward<Args>(args)...);

        T* result = node.get();
        owned_nodes.push_back(std::move(node));
        return result;
    }

    // Construction-time dispatch only; generated kernels call the concrete helpers.
    Node* make_switch_node(Node* input, const MetaD_zqc::SwitchFunctionRequest& p) {
        if (!std::isfinite(p.r_0) || !std::isfinite(p.d_0) || !std::isfinite(p.alpha))
            throw std::runtime_error("Switch configuration: parameters must be finite");
        switch (p.type) {
            case MetaD_zqc::LINE: return make_node<ConstantOne>(input);
            case MetaD_zqc::STEP: return make_node<StepSwitch>(input, p.r_0);
            case MetaD_zqc::FERMI: return make_node<Fermi>(input, p.r_0, p.alpha);
            case MetaD_zqc::TANH_TYPE: return make_node<TanhSwitch>(input, p.r_0, p.alpha);
            case MetaD_zqc::RATIONAL: return make_node<RationalSwitch>(input, p.r_0, p.n, p.m, p.d_0);
            default: throw std::runtime_error("Switch configuration: invalid type");
        }
    }

    void parse_expr(const std::string& expression);
    void build();

    std::string emit_forward();
    std::string emit_backward();
    void print_code();
};

class GraphExprParser {
    public:
        GraphExprParser(Graph& graph, const std::string& text)
            : graph_(graph), text_(text) {}
        Node* parse();

    private:
        Graph& graph_;
        const std::string& text_;
        std::size_t pos_ = 0;

        [[noreturn]] void fail(const std::string& message) const{
            throw std::runtime_error(fmt::sprintf(
                "Parse error at position %zu: %s",
                pos_, message));
        }

        void skip_spaces();
        bool take(char expected);
        void expect(char expected);
        std::string read_identifier();

        // 最低优先级：加法、减法。
        Node* parse_sum();
        // 乘法优先于加减法。
        Node* parse_product();
        // 一元正号、负号。
        Node* parse_unary();
        // 括号、函数调用、变量引用。
        Node* parse_primary();
};

inline void Graph::parse_expr(const std::string& expression){
    const std::size_t old_node_count = owned_nodes.size();

    try {
        GraphExprParser parser(*this, expression);
        Node* root = parser.parse();

        // 完整解析成功后，才登记为图的输出。
        outputs.push_back(root);

        // 图已经改变，需要重新 build。
        topo.clear();
        kernel_outputs.clear();
        kernel_source.clear();
    }
    catch (...) {
        // 删除本次失败解析创建的节点，保留已有图。
        owned_nodes.resize(old_node_count);
        throw;
    }
}

inline void Graph::build(){
    topo.clear();
    kernel_outputs.clear();
    kernel_source.clear();

    if (outputs.empty())
        throw std::runtime_error("Graph: no outputs");

    // 0：未访问；1：正在访问；2：访问完成。
    std::unordered_map<const Node*, int> state;
    int atom_count = -1;

    std::function<void(const Node*)> visit =
        [&](const Node* node)
    {
        if (!node)
            throw std::runtime_error("Graph: null node");

        const int current = state[node];

        if (current == 1)
            throw std::runtime_error("Graph: cycle detected");

        if (current == 2)
            return;

        state[node] = 1;

        const auto* mean = dynamic_cast<const MEAN*>(node);

        if (node->value_desc.components != 1)
            throw std::runtime_error(
                "Graph: only scalar components are supported");

        if (!mean) {
            if (node->value_desc.domain != Domain::PerAtom)
                throw std::runtime_error(
                    "Graph: only PerAtom values are supported here");

            if (node->n <= 0)
                throw std::runtime_error("Graph: invalid input length");

            if (atom_count < 0)
                atom_count = node->n;
            else if (atom_count != node->n)
                throw std::runtime_error(
                    "Graph: inconsistent atom counts");
        }

        for (const Node* input : node->inputs()) {
            // 当前只允许 Mean 出现在整张图的输出端。
            if (dynamic_cast<const MEAN*>(input))
                throw std::runtime_error(
                    "Graph: MEAN is only supported at graph outputs");

            visit(input);
        }

        state[node] = 2;
        topo.push_back(node);
    };

    for (const Node* output : outputs) {
        visit(output);

        if (const auto* mean = dynamic_cast<const MEAN*>(output))
            kernel_outputs.push_back(mean->src);
        else
            kernel_outputs.push_back(output);
    }

    std::string source = graph_device_source();
    source += emit_forward();
    source += emit_backward();

    kernel_source = std::move(source);
}

inline std::string Graph::emit_kernel(bool backward){
    if (topo.empty() || kernel_outputs.size() != outputs.size())
        throw std::logic_error("Graph: build is required");

    Codegen cg;

    // 为当前 kernel 生成独立的前向局部值。
    for (const Node* node : topo) {
        if (!dynamic_cast<const MEAN*>(node))
            cg.value(node);
    }

    std::string params = "int n";
    std::string body;

    // Leaf 当前只登记变量名，加载语句在这里生成。
    for (const auto& input : cg.inputs) {
        const std::string& name = input.second;

        params += fmt::sprintf(
            ", const double* %s_ptr", name);

        body += fmt::sprintf(
            "double %s = %s_ptr[i];\n", name, name);
    }

    body += cg.body;

    if (!backward) {
        for (std::size_t k = 0; k < kernel_outputs.size(); ++k) {
            params += fmt::sprintf(", double* out%zu", k);

            body += fmt::sprintf(
                "out%zu[i] = %s;\n",
                k, cg.value(kernel_outputs[k]));
        }
    }
    else {
        // 先登记全部梯度，再统一输出零初始化声明。
        for (const Node* node : topo) {
            if (!dynamic_cast<const MEAN*>(node))
                cg.adjoint(node);
        }

        body += cg.adjoint_decls;

        // 所有输出的种子必须在反向遍历之前注入。
        for (std::size_t k = 0; k < outputs.size(); ++k) {
            const std::string grad =
                cg.adjoint(kernel_outputs[k]);

            if (dynamic_cast<const MEAN*>(outputs[k])) {
                // 外部已经完成：seed = dV/dMean / N_global。
                params += fmt::sprintf(", double seed%zu", k);

                body += fmt::sprintf(
                    "// seed%zu = dV/dMean%zu / N_global\n"
                    "%s += seed%zu;\n",
                    k, k, grad, k);
            }
            else {
                // 普通逐原子输出允许每个原子有不同的上游梯度。
                params += fmt::sprintf(
                    ", const double* seed%zu", k);

                body += fmt::sprintf(
                    "%s += seed%zu[i];\n", grad, k);
            }
        }

        // 叶子梯度输出参数。
        for (const auto& input : cg.inputs) {
            params += fmt::sprintf(
                ", double* %s_adj", input.second);
        }

        // 共享节点只处理一次，梯度由节点 helper 累加。
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            const Node* node = *it;

            if (!dynamic_cast<const MEAN*>(node))
                body += node->emit_expr_bwd(cg);
        }

        // 全部贡献累加完成后，写回叶子梯度。
        for (const auto& input : cg.inputs) {
            body += fmt::sprintf(
                "%s_adj[i] = %s;\n",
                input.second, cg.adjoint(input.first));
        }
    }

    return fmt::sprintf(R"CUDA(
extern "C" __global__
void %s(%s)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

%s
}
)CUDA",
        backward ? "cv_backward" : "cv_forward",
        params,
        body);
}

inline std::string Graph::emit_forward(){
    return emit_kernel(false);
}

inline std::string Graph::emit_backward(){
    return emit_kernel(true);
}

inline void Graph::print_code(){
    if (kernel_source.empty())
        throw std::logic_error("Graph: call build before print_code");

    std::cout << kernel_source;
}

// 跳过空格
void GraphExprParser::skip_spaces(){
    while (pos_ < text_.size() &&
        std::isspace(
            static_cast<unsigned char>(text_[pos_]))) {
        ++pos_;
    }
}

Node* GraphExprParser::parse(){
    Node* root = parse_sum();

    // 当前只允许最外层出现一次 .MEAN。
    if (take('.')) {
        const std::string name = read_identifier();
        if (name != "MEAN")
            fail("expected MEAN after '.'");
        root = graph_.make_node<MEAN>(root);
    }

    skip_spaces();
    if (pos_ != text_.size())
        fail("unexpected trailing input");
    return root;
}

bool GraphExprParser::take(char expected){
    skip_spaces();

    if (pos_ < text_.size() && text_[pos_] == expected) {
        ++pos_;
        return true;
    }
    return false;
}

void GraphExprParser::expect(char expected){
    if (!take(expected)) {
        fail(fmt::sprintf("expected '%c'", expected));
    }
}

std::string GraphExprParser::read_identifier(){
    skip_spaces();

    if (pos_ == text_.size())
        fail("expected identifier");

    unsigned char ch =
        static_cast<unsigned char>(text_[pos_]);

    if (!std::isalpha(ch) && ch != '_')
        fail("expected identifier");

    const std::size_t begin = pos_++;

    while (pos_ < text_.size()) {
        ch = static_cast<unsigned char>(text_[pos_]);

        if (!std::isalnum(ch) && ch != '_')
            break;

        ++pos_;
    }

    return text_.substr(begin, pos_ - begin);
}

// 最低优先级：加法、减法。
Node* GraphExprParser::parse_sum(){
    Node* lhs = parse_product();

    for (;;) {
        if (take('+')) {
            Node* rhs = parse_product();
            lhs = graph_.make_node<Add>(lhs, rhs);
        }
        else if (take('-')) {
            Node* rhs = parse_product();
            Node* negative =
                graph_.make_node<Neg>(rhs);

            lhs = graph_.make_node<Add>(lhs, negative);
        }
        else {
            return lhs;
        }
    }
}

// 乘法优先于加减法。
Node* GraphExprParser::parse_product(){
    Node* lhs = parse_unary();

    while (take('*')) {
        Node* rhs = parse_unary();
        lhs = graph_.make_node<Mul>(lhs, rhs);
    }

    return lhs;
}

// 一元正号、负号。
Node* GraphExprParser::parse_unary(){
    if (take('+'))
        return parse_unary();

    if (take('-')) {
        Node* input = parse_unary();
        return graph_.make_node<Neg>(input);
    }

    return parse_primary();
}

// 括号、函数调用、变量引用。
Node* GraphExprParser::parse_primary(){
    if (take('(')) {
        Node* result = parse_sum();
        expect(')');
        return result;
    }

    const std::string name = read_identifier();
    if (take('(')) {
        if (name == "sw_func") {
            Node* input = parse_sum();
            expect(',');
            const std::string switch_name = read_identifier();
            expect(')');
            auto it = graph_.switches.find(switch_name);
            if (it == graph_.switches.end())
                fail(fmt::sprintf("unknown switching function: %s", switch_name));
            return graph_.make_switch_node(input, it->second);
        }
        if (name != "exp")
            fail(fmt::sprintf("unknown function: %s", name));
        Node* input = parse_sum();
        expect(')');
        return graph_.make_node<Exp>(input);
    }

    auto it = graph_.symbols.find(name);
    if (it == graph_.symbols.end())
        fail(fmt::sprintf("unknown input: %s", name));
    if (it->second == nullptr)
        fail(fmt::sprintf("null input node: %s", name));

    return it->second;
}
