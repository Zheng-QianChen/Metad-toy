#pragma once

// 图使用通用设备缓冲区与求和回调；诊断通过调用方提供的 Logger 回传。
#include "zqc_array_view.h"
#include "zqc_collective.h"
#include "zqc_gpu_buffer.cuh"
#include "zqc_switch_function_device.cuh"

#include <cuda_runtime.h>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// 节点基类与值描述。
struct Graph;
struct GraphExpr;
struct Codegen;

enum class ValueDomain {
    PerAtom,  // 每个原子一份
    Global    // 整个系统一份
};

struct ValueDesc {
    ValueDomain domain;
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
    // SafeDiv 等节点允许 Global Leaf 与 PerAtom 混用；规约结果仍不能回灌融合段。
    virtual bool allows_domain_broadcast() const;

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
    // Global Leaf 标量，作为融合 kernel 的按值参数（broadcast 到每个原子）。
    std::vector<std::pair<const Node*, std::string>> scalar_inputs;
    std::unordered_map<const Node*, std::string> value_names;
    std::unordered_map<const Node*, std::string> adjoint_names;
    std::string adjoint_decls;
    std::string body;
    int next_val_id = 0;
    int next_adj_id = 0;

    std::string value(const Node* node);
    // 仅用于 PerAtom 表达式中的 Global Leaf；不得传入规约或其他 Host Global 节点。
    std::string scalar_value(const Node* node);
    std::string adjoint(const Node* node);
};

struct Graph {
    using Symbols = std::unordered_map<std::string, Node*>;
    using Switches = std::unordered_map<std::string, MetaD_zqc::SwitchFunctionRequest>;
    using NodeCreator = std::function<Node*(Graph&, const GraphExpr&)>;

    Symbols symbols;
    Switches switches;
    // Leaves passed by the caller remain externally owned.
    std::vector<std::unique_ptr<Node>> owned_nodes;
    std::vector<Node*> outputs;
    std::vector<const Node*> kernel_outputs;
    // 每个 GPU 输出所对应的规约节点，或直接的 PerAtom 输出。
    std::vector<const Node*> kernel_seeds;
    // 空回调表示单进程原地求和；跨进程由宿主注入 SumReducer（例如 engine.sum_reducer）。
    MetaD_zqc::SumReducer sum_reducer;
    std::vector<const Node*> topo;
    std::string kernel_source;

    explicit Graph(Symbols symbols_, Switches switches_ = {}, MetaD_zqc::SumReducer sum = {});
    explicit Graph(std::vector<Node*> outputs_, MetaD_zqc::SumReducer sum = {});

    template<class T, class... Args>
    T* make_node(Args&&... args) {
        auto node = std::make_unique<T>(std::forward<Args>(args)...);
        T* result = node.get();
        owned_nodes.push_back(std::move(node));
        return result;
    }

    // Node creators consume parsed arguments, and allocate through make_node().
    void register_node(const std::string& name, NodeCreator creator, bool replace = false);
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
    GpuBuffer<double> xbuf; // host 便利构造 / 非连续视图暂存；借用外部输入时可能为空
    GpuBuffer<double> abuf; // 未传入外部梯度，或非连续视图暂存时由 Leaf 持有
    double* g = nullptr;

    // 借用任意 stride 的 CUDA/Host 视图。有效项数取自 values.item_count。
    // 融合 kernel 需要连续设备指针：连续 CUDA 标量零拷贝；其余路径显式暂存。
    Leaf(MetaD_zqc::ArrayView<const double> values, MetaD_zqc::ArrayView<double> gradients,
         ValueDesc desc = {ValueDomain::PerAtom, 1});

    // 借用缓冲区，不复制、不释放外部显存。count 是有效元素数，不是 capacity。
    // 调用方保证外部 GpuBuffer 对象在 Leaf 使用期间始终存活。
    explicit Leaf(GpuBuffer<double>& values, int count,
                  ValueDesc desc = {ValueDomain::PerAtom, 1});

    // 可将梯度直接写入调用方提供的 GpuBuffer；Global 梯度由 backward() 上传。
    Leaf(GpuBuffer<double>& values, GpuBuffer<double>& gradients, int count,
         ValueDesc desc = {ValueDomain::PerAtom, 1});

    // PerAtom 值上传至 GpuBuffer；Global 标量保存在 Host，不分配显存。
    explicit Leaf(const std::vector<double>& host,
                  ValueDesc desc = {ValueDomain::PerAtom, 1});

    // 重新绑定输入/梯度视图；扩容、重排或更换底层数组后调用。
    void bind(MetaD_zqc::ArrayView<const double> values, MetaD_zqc::ArrayView<double> gradients);
    // 外部 GpuBuffer::grow_to 可能更换 ptr；启动 kernel 前刷新一次。
    // 有效元素数变化时需要重建/更新图；这里只处理相同 n 下的指针变化。
    void refresh_buffers();
    // 清零当前梯度视图（或 Host 伴随），并重新校验绑定供 forward/backward 使用。
    void clear();

    void forward() override;
    void backward() override;
    // 融合执行器在自己的流上消费输入和写回梯度，等待所绑定 View 的生产者。
    void forward_on(cudaStream_t stream);
    void backward_on(cudaStream_t stream);
    void wait_gradient_on(cudaStream_t stream) const { gradient_view_.wait_on(stream); }

    std::vector<const Node*> inputs() const override;
    std::string emit_expr(Codegen& cg) const override;
    std::string emit_expr_bwd(Codegen& cg) const override;

private:
    GpuBuffer<double>* input_buffer_ = nullptr;
    GpuBuffer<double>* gradient_buffer_ = nullptr;
    MetaD_zqc::ArrayView<const double> input_view_{};
    MetaD_zqc::ArrayView<double> gradient_view_{};
    bool stages_values_ = false;
    bool stages_gradients_ = false;

    static int checked_count(size_t count, ValueDesc desc);
    void apply_bound_views();
    void stage_values_to_device(cudaStream_t stream = 0);
    void scatter_gradients_from_device(cudaStream_t stream = 0);
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

// 输入是本 rank 拥有的有效元素，不包含 ghost。
// sum_reducer 空表示仅本地求和；跨进程由宿主提供单次打包 {sum,count} 规约。
struct AVE : Node {
    Node* src;
    MetaD_zqc::SumReducer sum_reducer;
    long long global_count = 0;

    explicit AVE(Node* src_, MetaD_zqc::SumReducer sum = {});
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

// 保留实验阶段的名称；两者采用同一个 Host 规约实现。
using MEAN = AVE;

// 全局求和：空集合（含全局空）约定为 0；不改变 AVE 的空输入报错语义。
struct SUM : Node {
    Node* src;
    MetaD_zqc::SumReducer sum_reducer;
    bool ready = false;

    explicit SUM(Node* src_, MetaD_zqc::SumReducer sum = {});
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
    void reduce_sum(double local_sum);
};

// safe_div(a,b,epsilon)：|b|<=epsilon 时值为 0 且两路导数为 0；epsilon 为有限非负数。
// 同域 Global/PerAtom，以及 Global Leaf 与 PerAtom 的 broadcast。
struct SafeDiv : Node {
    Node* a;
    Node* b;
    double epsilon;
    GpuBuffer<double> vbuf;
    GpuBuffer<double> abuf;
    // 独立 broadcast 反向时暂存 Global 伴随；融合路径由执行器提供标量伴随缓冲。
    GpuBuffer<double> broadcast_scalar_adj;

    SafeDiv(Node* a_, Node* b_, double epsilon_);
    void forward() override;
    void backward() override;
    std::vector<const Node*> inputs() const override;
    std::string emit_expr(Codegen& cg) const override;
    std::string emit_expr_bwd(Codegen& cg) const override;
    static Node* create(Graph& graph, const GraphExpr& call);
    bool allows_domain_broadcast() const override;
};

// 图执行器：编译并调度 GPU 融合段、Host 计算及跨进程规约回调。
// Graph 和输入节点必须比执行器活得更久；图结构固定，本地元素数可显式更新。
class GraphExecution {
    public:
        // Logger 持有回调副本；捕获资源的生命周期由调用方管理。
        explicit GraphExecution(Graph& graph, MetaD_zqc::Logger logger = {});
        // 外部更新 Leaf.n 和显存容量后调用；仅调整执行缓冲区，不重新 NVRTC 编译。
        void resize_inputs();
        ~GraphExecution();
        GraphExecution(const GraphExecution&) = delete;
        GraphExecution& operator=(const GraphExecution&) = delete;

        // 返回所有 Global 输出（按输出中的 Global 顺序）；PerAtom 用 per_atom_output 借用。
        std::vector<double> forward();
        // 每个 rank 传入同一组全局输出种子，不再对这些种子做跨进程求和。
        void backward(const std::vector<double>& seeds,
                      const std::vector<MetaD_zqc::ArrayView<const double>>& atom_seeds = {});
        // output_index 按 Graph.outputs 编号；借用到下一次 forward/resize/析构。
        MetaD_zqc::ArrayView<const double> per_atom_output(size_t output_index) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
};
