#pragma once

// 先引入缓冲区及其 LAMMPS 依赖，再声明图使用的 Domain 类型。
#include "zqc_gpu_buffer.cuh"
#include "zqc_switch_function_device.cuh"

#include <cuda_runtime.h>
#include <mpi.h>
#include <cstdio>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// 节点基类与值描述。
struct Graph;
struct GraphExpr;
struct Codegen;

enum class Domain {
    PerAtom,  // 每个原子一份
    Global    // 整个系统一份
};

struct ValueDesc {
    Domain domain;
    int components;  // 1：标量；3：三分量向量
};

ValueDesc infer_broadcast(ValueDesc a, ValueDesc b);

struct Node {
    ValueDesc value_desc;
    double* val = nullptr; // 节点的值
    double* adj = nullptr; // 节点的导数
    int n = 0;
    // Global 标量保存在 Host；PerAtom 的 val/adj 仍指向设备缓冲区。
    double host_value = 0.0;
    double host_adjoint = 0.0;
    void bind_storage(double* device_value, double* device_adjoint);

    Node(ValueDesc desc);
    Node(const Node&) = delete;
    Node& operator=(const Node&) = delete;
    virtual void forward() = 0;
    virtual void backward() = 0;
    virtual ~Node() = default;

    // 外部规约只支持各元素共享同一反向标量种子，例如 SUM、MEAN。
    // kernel_output() 返回规约前的逐原子节点；执行器在 Host 上调用规约和种子换算。
    virtual bool is_reduction() const;
    virtual const Node* kernel_output() const;
    // GPU 段完成后，由 Host 执行规约；seed 已包括规约的导数。
    virtual void reduce_host(const double* values, int count);
    virtual double reduction_seed() const;

    // 节点连接：列出输入节点
    virtual std::vector<const Node*> inputs() const = 0;

    // 节点生成方法
    virtual std::string emit_expr(Codegen& cg) const = 0;
    virtual std::string emit_expr_bwd(Codegen& cg) const = 0;
};

// 表达式解析、图构建与代码生成。
// Syntax only: symbols, numeric parameters and calls. No GPU allocation during parsing.
struct GraphExpr {
    enum class Kind { Symbol, Number, Call };
    Kind kind = Kind::Symbol;
    std::string name;
    double number = 0.0;
    std::vector<GraphExpr> args;
    std::size_t position = 0;

    void require_args(std::size_t count) const;
    double number_arg(std::size_t index) const;
    int integer_arg(std::size_t index) const;
    const std::string& symbol_arg(std::size_t index) const;
    [[noreturn]] void fail(const std::string& message) const;
};

class GraphExprParser {
public:
    explicit GraphExprParser(std::string text);
    GraphExpr parse();

private:
    std::string text_;
    std::size_t pos_ = 0;

    [[noreturn]] void fail(const std::string& message) const;
    void skip_spaces();
    bool take(char expected);
    void expect(char expected);
    std::string read_identifier();
    GraphExpr read_number();
    GraphExpr parse_sum();
    GraphExpr parse_product();
    GraphExpr parse_unary();
    GraphExpr parse_primary();
};

struct Codegen {
    // Each input records its node and the local name loaded by the kernel.
    std::vector<std::pair<const Node*, std::string>> inputs;
    std::unordered_map<const Node*, std::string> value_names;
    std::unordered_map<const Node*, std::string> adjoint_names;
    std::string adjoint_decls;
    std::string body;
    int next_val_id = 0;
    int next_adj_id = 0;

    std::string value(const Node* node);
    std::string adjoint(const Node* node);
};

struct Graph {
    using Symbols = std::unordered_map<std::string, Node*>;
    using Switches = std::unordered_map<std::string, MetaD_zqc::SwitchFunctionRequest>;
    using NodeCreator = Node* (*)(Graph&, const GraphExpr&);

    Symbols symbols;
    Switches switches;
    // Leaves passed by the caller remain externally owned.
    std::vector<std::unique_ptr<Node>> owned_nodes;
    std::vector<Node*> outputs;
    std::vector<const Node*> kernel_outputs;
    // 每个 GPU 输出所对应的规约节点，或直接的 PerAtom 输出。
    std::vector<const Node*> kernel_seeds;
    MPI_Comm comm = MPI_COMM_NULL;
    std::vector<const Node*> topo;
    std::string kernel_source;

    explicit Graph(Symbols symbols_, Switches switches_ = {}, MPI_Comm comm_ = MPI_COMM_NULL);
    explicit Graph(std::vector<Node*> outputs_, MPI_Comm comm_ = MPI_COMM_NULL);

    template<class T, class... Args>
    T* make_node(Args&&... args) {
        auto node = std::make_unique<T>(std::forward<Args>(args)...);
        T* result = node.get();
        owned_nodes.push_back(std::move(node));
        return result;
    }

    // Node creators consume parsed arguments, and allocate through make_node().
    void register_node(const std::string& name, NodeCreator creator);
    Node* create_node(const GraphExpr& expression);
    // Compatibility entry point; the switch adapter is implemented with the nodes.
    Node* make_switch_node(Node* input, const MetaD_zqc::SwitchFunctionRequest& params);

    void parse_expr(const std::string& expression);
    void build();
    std::string emit_kernel(bool backward);
    std::string emit_forward();
    std::string emit_backward();
    void print_code();

private:
    std::unordered_map<std::string, NodeCreator> creators_;
};

void register_builtin_graph_nodes(Graph& graph);
std::string graph_device_source();
void graph_check_cuda(cudaError_t status);
void sync_kernel();

// 内置节点声明；实现及参数读取位于 src/utils/zqc_graph_nodes.cu。
using MetaD_zqc::GpuBuffer;

struct Leaf : Node {
    GpuBuffer<double> xbuf; // host 便利构造使用；借用外部输入时为空
    GpuBuffer<double> abuf; // 未传入外部梯度时，由 Leaf 持有
    double* g = nullptr;

    // 借用缓冲区，不复制、不释放外部显存。count 是有效元素数，不是 capacity。
    // 调用方保证外部 GpuBuffer 对象在 Leaf 使用期间始终存活。
    explicit Leaf(GpuBuffer<double>& values, int count,
                  ValueDesc desc = {Domain::PerAtom, 1});

    // 可将梯度直接写入调用方提供的 GpuBuffer；Global 梯度由 backward() 上传。
    Leaf(GpuBuffer<double>& values, GpuBuffer<double>& gradients, int count,
         ValueDesc desc = {Domain::PerAtom, 1});

    // PerAtom 值上传至 GpuBuffer；Global 标量保存在 Host，不分配显存。
    explicit Leaf(const std::vector<double>& host,
                  ValueDesc desc = {Domain::PerAtom, 1});

    // 外部 grow_to 可能更换 ptr；启动 kernel、收集参数前刷新一次。
    // 有效元素数变化时需要重建/更新图；这里只处理相同 n 下的指针变化。
    void refresh_buffers();

    void forward() override;
    void backward() override;

    std::vector<const Node*> inputs() const override;
    std::string emit_expr(Codegen& cg) const override;
    std::string emit_expr_bwd(Codegen& cg) const override;

private:
    GpuBuffer<double>* input_buffer_ = nullptr;
    GpuBuffer<double>* gradient_buffer_ = nullptr;

    static int checked_count(size_t count, ValueDesc desc);
};

struct Mul : Node {
    Node* a; // 输入节点1
    Node* b; // 输入节点2
    GpuBuffer<double> vbuf;
    GpuBuffer<double> abuf;

    Mul(Node* a_, Node* b_);
    void forward() override;
    void backward() override;

    std::vector<const Node*> inputs() const override;
    std::string emit_expr(Codegen& cg) const override;
    std::string emit_expr_bwd(Codegen& cg) const override;

    static Node* create(Graph& graph, const GraphExpr& call);
};

struct Add : Node {
    Node* a; // 输入节点1
    Node* b; // 输入节点2
    GpuBuffer<double> vbuf;
    GpuBuffer<double> abuf;

    Add(Node* a_, Node* b_);
    void forward() override;
    void backward() override;

    std::vector<const Node*> inputs() const override;
    std::string emit_expr(Codegen& cg) const override;
    std::string emit_expr_bwd(Codegen& cg) const override;

    static Node* create(Graph& graph, const GraphExpr& call);
};

struct Neg : Node {
    Node* a;
    GpuBuffer<double> vbuf;
    GpuBuffer<double> abuf;

    explicit Neg(Node* a_);
    void forward() override;
    void backward() override;

    std::vector<const Node*> inputs() const override;
    std::string emit_expr(Codegen& cg) const override;
    std::string emit_expr_bwd(Codegen& cg) const override;

    static Node* create(Graph& graph, const GraphExpr& call);
};

struct Linear : Node {
    Node* a; // 输入节点1
    double k0, b0; // 输入节点2
    GpuBuffer<double> vbuf;
    GpuBuffer<double> abuf;

    Linear(Node* a_, double k0_, double b0_);
    void forward() override;
    void backward() override;

    std::vector<const Node*> inputs() const override;
    std::string emit_expr(Codegen& cg) const override;
    std::string emit_expr_bwd(Codegen& cg) const override;

    static Node* create(Graph& graph, const GraphExpr& call);
};

struct Exp : Node {
    Node* a; // 后续节点
    GpuBuffer<double> vbuf;
    GpuBuffer<double> abuf;

    Exp(Node* a_);
    void forward() override;
    void backward() override;

    std::vector<const Node*> inputs() const override;
    std::string emit_expr(Codegen& cg) const override;
    std::string emit_expr_bwd(Codegen& cg) const override;

    static Node* create(Graph& graph, const GraphExpr& call);
};

struct ConstantOne : Node {
    Node* a;
    GpuBuffer<double> vbuf;
    GpuBuffer<double> abuf;

    explicit ConstantOne(Node* a_);
    void forward() override;
    void backward() override;
    std::vector<const Node*> inputs() const override;
    std::string emit_expr(Codegen& cg) const override;
    std::string emit_expr_bwd(Codegen& cg) const override;

    static Node* create(Graph& graph, const GraphExpr& call);
};

struct StepSwitch : Node {
    Node* a;
    const double r0;
    GpuBuffer<double> vbuf;
    GpuBuffer<double> abuf;

    explicit StepSwitch(Node* a_, double r0_);
    void forward() override;
    void backward() override;
    std::vector<const Node*> inputs() const override;
    std::string emit_expr(Codegen& cg) const override;
    std::string emit_expr_bwd(Codegen& cg) const override;

    static Node* create(Graph& graph, const GraphExpr& call);
};

struct Fermi : Node {
    Node* a;
    const double r0;
    const double alpha;
    GpuBuffer<double> vbuf;
    GpuBuffer<double> abuf;

    explicit Fermi(Node* a_, double r0_, double alpha_);
    void forward() override;
    void backward() override;
    std::vector<const Node*> inputs() const override;
    std::string emit_expr(Codegen& cg) const override;
    std::string emit_expr_bwd(Codegen& cg) const override;

    static Node* create(Graph& graph, const GraphExpr& call);
};

struct TanhSwitch : Node {
    Node* a;
    const double r0;
    const double alpha;
    GpuBuffer<double> vbuf;
    GpuBuffer<double> abuf;

    explicit TanhSwitch(Node* a_, double r0_, double alpha_);
    void forward() override;
    void backward() override;
    std::vector<const Node*> inputs() const override;
    std::string emit_expr(Codegen& cg) const override;
    std::string emit_expr_bwd(Codegen& cg) const override;

    static Node* create(Graph& graph, const GraphExpr& call);
};

struct RationalSwitch : Node {
    Node* a;
    const double r0;
    const int power_n;
    const int power_m;
    const double d0;
    GpuBuffer<double> vbuf;
    GpuBuffer<double> abuf;

    explicit RationalSwitch(Node* a_, double r0_, int power_n_, int power_m_, double d0_ = 0.0);
    void forward() override;
    void backward() override;
    std::vector<const Node*> inputs() const override;
    std::string emit_expr(Codegen& cg) const override;
    std::string emit_expr_bwd(Codegen& cg) const override;

    static Node* create(Graph& graph, const GraphExpr& call);
};

// 输入是本 rank 拥有的有效元素，不包含 ghost。comm 由调用方持有。
struct AVE : Node {
    Node* src;
    MPI_Comm comm;
    long long global_count = 0;

    explicit AVE(Node* src_, MPI_Comm comm_ = MPI_COMM_NULL);
    void forward() override;
    void backward() override;
    void reduce_host(const double* values, int count) override;
    double reduction_seed() const override;
    std::vector<const Node*> inputs() const override;
    std::string emit_expr(Codegen& cg) const override;
    std::string emit_expr_bwd(Codegen& cg) const override;
    static Node* create(Graph& graph, const GraphExpr& call);
    bool is_reduction() const override;
    const Node* kernel_output() const override;
protected:
    // 已完成局部求和后，把 {sum, count} 一次规约；允许子类选择自己的有效元素集合。
    void reduce_sum_count(double local_sum, int local_count);
};

// 保留实验阶段的名称；两者采用同一个 Host/MPI 实现。
using MEAN = AVE;

// 图执行器：编译并调度 GPU 融合段、Host 计算及 MPI 规约。
// Graph 和输入节点必须比执行器活得更久；图结构固定，本地元素数可显式更新。
class GraphExecution {
    public:
        // 日志句柄由调用方持有；LAMMPS 仅 rank 0 传入 screen/logfile。
        explicit GraphExecution(Graph& graph, FILE* screen = nullptr, FILE* logfile = nullptr);
        // 外部更新 Leaf.n 和显存容量后调用；仅调整执行缓冲区，不重新 NVRTC 编译。
        void resize_inputs();
        ~GraphExecution();
        GraphExecution(const GraphExecution&) = delete;
        GraphExecution& operator=(const GraphExecution&) = delete;

        // 返回所有 Global 输出；PerAtom -> Host 规约 -> Global 标量计算。
        std::vector<double> forward();
        // 每个 rank 传入同一组全局输出种子，不再对这些种子做 MPI 求和。
        void backward(const std::vector<double>& seeds);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
};
