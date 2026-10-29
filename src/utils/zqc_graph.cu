#include "zqc_graph.cuh"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <fmt/printf.h>

Graph::Graph(Symbols symbols_, Switches switches_, MetaD_zqc::SumReducer sum)
    : symbols(std::move(symbols_)), switches(std::move(switches_)), sum_reducer(std::move(sum)) {
    register_builtin_graph_nodes(*this);
}

Graph::Graph(std::vector<Node*> outputs_, MetaD_zqc::SumReducer sum)
    : outputs(std::move(outputs_)), sum_reducer(std::move(sum)) {
    register_builtin_graph_nodes(*this);
}

void Graph::register_node(const std::string& name, NodeCreator creator, bool replace) {
    if (name.empty() || !creator)
        throw std::runtime_error("Graph: a node registration needs a name and creator");
    if (replace) {
        const auto found=creators_.find(name);
        if(found==creators_.end()) throw std::runtime_error("Graph: cannot replace unregistered node: "+name);
        found->second=std::move(creator);
        return;
    }
    if (!creators_.emplace(name, creator).second)
        throw std::runtime_error("Graph: node already registered: " + name);
}

Node* Graph::create_node(const GraphExpr& expression) {
    if (expression.kind == GraphExpr::Kind::Symbol) {
        auto it = symbols.find(expression.name);
        if (it == symbols.end() || !it->second)
            expression.fail("unknown or null input: " + expression.name);
        return it->second;
    }
    if (expression.kind == GraphExpr::Kind::Number)
        expression.fail("numeric literals are operator parameters; use a Leaf for graph inputs");

    auto it = creators_.find(expression.name);
    if (it == creators_.end())
        expression.fail("unknown function or operator: " + expression.name);
    Node* node = it->second(*this, expression);
    if (!node)
        expression.fail("node creator returned null: " + expression.name);
    return node;
}

void GraphExpr::require_args(std::size_t count) const {
    if (args.size() != count)
        fail(fmt::sprintf("%s expects %zu arguments, got %zu", name, count, args.size()));
}

double GraphExpr::number_arg(std::size_t index) const {
    const GraphExpr& arg = args.at(index);
    if (arg.kind != Kind::Number || !std::isfinite(arg.number))
        arg.fail("expected a finite numeric parameter");
    return arg.number;
}

int GraphExpr::integer_arg(std::size_t index) const {
    double value = number_arg(index);
    if (std::trunc(value) != value || value < std::numeric_limits<int>::min()
            || value > std::numeric_limits<int>::max())
        args.at(index).fail("expected an integer parameter in int range");
    return static_cast<int>(value);
}

const std::string& GraphExpr::symbol_arg(std::size_t index) const {
    const GraphExpr& arg = args.at(index);
    if (arg.kind != Kind::Symbol)
        arg.fail("expected a configuration name");
    return arg.name;
}

void GraphExpr::fail(const std::string& message) const {
    throw std::runtime_error(fmt::sprintf("Expression error at position %zu: %s", position, message));
}

// xxd 自动生成的数组由一个独立的 .cpp 定义，不在这里重复保存源码。
extern unsigned char zqc_graph_device_source[];
extern unsigned int zqc_graph_device_source_len;

std::string graph_device_source()
{
    return std::string(
        reinterpret_cast<const char*>(zqc_graph_device_source),
        zqc_graph_device_source_len);
}

void graph_check_cuda(cudaError_t status)
{
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

void sync_kernel()
{
    graph_check_cuda(cudaGetLastError());
    graph_check_cuda(cudaStreamSynchronize(0));
}

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
std::string Codegen::scalar_value(const Node* node){
    if (!node || node->value_desc.domain != ValueDomain::Global || node->value_desc.components != 1)
        throw std::runtime_error("Codegen: scalar_value expects a Global scalar");
    if (!node->inputs().empty())
        throw std::runtime_error("Codegen: broadcast Global operand must be a Leaf");
    auto it = value_names.find(node);
    if (it != value_names.end()) return it->second;
    std::string name = fmt::sprintf("s%d", next_val_id++);
    value_names.emplace(node, name);
    scalar_inputs.emplace_back(node, name);
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

void Graph::parse_expr(const std::string& expression){
    const std::size_t old_node_count = owned_nodes.size();

    try {
        GraphExprParser parser(expression);
        GraphExpr syntax = parser.parse();
        Node* root = create_node(syntax);

        // 完整解析成功后，才登记为图的输出。
        outputs.push_back(root);

        // 图已经改变，需要重新 build。
        topo.clear();
        kernel_outputs.clear();
        kernel_seeds.clear();
        kernel_source.clear();
    }
    catch (...) {
        // 删除本次失败解析创建的节点，保留已有图。
        owned_nodes.resize(old_node_count);
        throw;
    }
}

void Graph::build(){
    topo.clear();
    kernel_outputs.clear();
    kernel_seeds.clear();
    kernel_source.clear();
    if (outputs.empty()) throw std::runtime_error("Graph: no outputs");

    std::unordered_map<const Node*, int> state;
    int atom_count = -1;
    std::function<void(const Node*)> visit = [&](const Node* node) {
        if (!node) throw std::runtime_error("Graph: null node");
        if (state[node] == 1) throw std::runtime_error("Graph: cycle detected");
        if (state[node] == 2) return;
        state[node] = 1;
        if (node->value_desc.components != 1)
            throw std::runtime_error("Graph: only scalar components are supported");
        const bool per_atom = node->value_desc.domain == ValueDomain::PerAtom;
        if (per_atom) {
            if (node->n < 0) throw std::runtime_error("Graph: invalid input length");
            if (atom_count < 0) atom_count = node->n;
            else if (atom_count != node->n) throw std::runtime_error("Graph: inconsistent atom counts");
        } else if (node->n != 1) {
            throw std::runtime_error("Graph: Global nodes must contain one scalar");
        }
        for (const Node* input : node->inputs()) {
            visit(input);
            if (node->is_reduction()) {
                if (per_atom || input->value_desc.domain != ValueDomain::PerAtom)
                    throw std::runtime_error("Graph: reductions require PerAtom input and Global output");
            } else if (input->value_desc.domain != node->value_desc.domain) {
                if (!node->allows_domain_broadcast())
                    throw std::runtime_error("Graph: use AVE or SUM to convert PerAtom values to Global; broadcasting is not implemented");
                if (input->value_desc.domain == ValueDomain::Global && !input->inputs().empty())
                    throw std::runtime_error("Graph: broadcast Global operand must be a Leaf; reductions stay on the Host side");
            }
        }
        state[node] = 2;
        topo.push_back(node);
        if (node->is_reduction()) {
            const Node* source = node->kernel_output();
            if (!source || source->value_desc.domain != ValueDomain::PerAtom || state[source] != 2)
                throw std::runtime_error("Graph: invalid reduction source");
            kernel_outputs.push_back(source);
            kernel_seeds.push_back(node);
        }
    };
    try {
        for (const Node* output : outputs) {
            visit(output);
            if (output->value_desc.domain == ValueDomain::PerAtom) {
                kernel_outputs.push_back(output);
                kernel_seeds.push_back(output);
            }
        }
        if (!kernel_outputs.empty())
            kernel_source = graph_device_source() + emit_forward() + emit_backward();
    } catch (...) {
        topo.clear();
        kernel_outputs.clear();
        kernel_seeds.clear();
        throw;
    }
}

std::string Graph::emit_kernel(bool backward){
    if (topo.empty() || kernel_outputs.size() != kernel_seeds.size())
        throw std::logic_error("Graph: build is required");

    if (kernel_outputs.empty()) return {};
    Codegen cg;

    // 为当前 kernel 生成独立的前向局部值。
    for (const Node* node : topo) {
        if (node->value_desc.domain == ValueDomain::PerAtom)
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
    for (const auto& input : cg.scalar_inputs) {
        // Global Leaf 以按值标量进入融合核，实现与 PerAtom 的 broadcast。
        params += fmt::sprintf(", double %s", input.second);
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
            if (node->value_desc.domain == ValueDomain::PerAtom)
                cg.adjoint(node);
        }

        body += cg.adjoint_decls;

        // 所有输出的种子必须在反向遍历之前注入。
        for (std::size_t k = 0; k < kernel_seeds.size(); ++k) {
            const std::string grad =
                cg.adjoint(kernel_outputs[k]);

            if (kernel_seeds[k]->is_reduction()) {
                // 标量种子已经由外部规约层转换为逐元素梯度。
                params += fmt::sprintf(", double seed%zu", k);

                body += fmt::sprintf(
                    "%s += seed%zu;\n", grad, k);
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
        for (const auto& input : cg.scalar_inputs) {
            params += fmt::sprintf(", double* %s_adj", input.second);
        }

        // 共享节点只处理一次，梯度由节点 helper 累加。
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            const Node* node = *it;

            if (node->value_desc.domain == ValueDomain::PerAtom)
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

std::string Graph::emit_forward(){
    return emit_kernel(false);
}

std::string Graph::emit_backward(){
    return emit_kernel(true);
}

void Graph::print_code(){
    if (topo.empty())
        throw std::logic_error("Graph: call build before print_code");

    std::cout << kernel_source;
}
