#include "zqc_graph.cuh"

#include <stdexcept>

namespace {
// A registered operation may construct a reusable composition of existing nodes.
Node* create_scaled_square(Graph& graph, const GraphExpr& call) {
    call.require_args(2);
    const double scale = call.number_arg(1);
    Node* input = graph.create_node(call.args[0]);
    Node* square = graph.make_node<Mul>(input, input);
    return graph.make_node<Linear>(square, scale, 0.0);
}

// Deliberately does not inherit MEAN: Graph must use the reduction interface.
struct ExternalAverage : Node {
    Node* input;
    explicit ExternalAverage(Node* input_) : Node({Domain::Global, 1}), input(input_) {
        n = 1;
    }
    void forward() override { throw std::logic_error("external reduction"); }
    void backward() override { throw std::logic_error("external reduction"); }
    std::vector<const Node*> inputs() const override { return {input}; }
    bool is_reduction() const override { return true; }
    const Node* kernel_output() const override { return input; }
    std::string emit_expr(Codegen&) const override { throw std::logic_error("must skip reduction"); }
    std::string emit_expr_bwd(Codegen&) const override { throw std::logic_error("must skip reduction"); }
    static Node* create(Graph& graph, const GraphExpr& call) {
        call.require_args(1);
        Node* input = graph.create_node(call.args[0]);
        return graph.make_node<ExternalAverage>(input);
    }
};
}

void register_test_graph_nodes(Graph& graph) {
    graph.register_node("scaled_square", create_scaled_square);
    graph.register_node(".AVERAGE", ExternalAverage::create);
    graph.register_node("external_average", ExternalAverage::create);
}
