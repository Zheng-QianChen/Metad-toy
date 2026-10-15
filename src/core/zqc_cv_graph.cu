#include "zqc_cv_graph.cuh"
#include "fix_crystallize.h"
#include "zqc_cv_assembly.cuh"
#include "zqc_graph.cuh"
#include "atom.h"
#include "comm.h"
#include <algorithm>
#include <stdexcept>

namespace MetaD_zqc {
namespace {
struct CVInput {
    CV* cv;
    GpuBuffer<double> values, gradients;
    Leaf leaf;
    std::shared_ptr<const CVAssembly> assembly;
    explicit CVInput(CV* cv_) : cv(cv_), leaf(values, gradients, 0) {}
};

// CV.AVE 只对该 CV 的 owned 组成员求均值，不能把公共 Leaf 的组外零值计入分母。
// 此节点只接原始 Leaf，因此反向给 padding 的种子最终会被 gather_adjoint 丢弃。
struct CVGroupAverage : AVE {
    CVInput& input;
    CVGroupAverage(CVInput& input_, MPI_Comm comm_) : AVE(&input_.leaf, comm_), input(input_) {}
    void reduce_host(const double* values, int count) override {
        if (!input.assembly || count != input.assembly->leaf_count())
            throw std::logic_error("CV.AVE: stale assembly");
        double local_sum = 0.0, correction = 0.0;
        const int local_count = input.assembly->group_count();
        for (int k = 0; k < local_count; ++k) {
            const double adjusted = values[input.assembly->leaf_slots()[k]] - correction;
            const double next = local_sum + adjusted;
            correction = (next - local_sum) - adjusted;
            local_sum = next;
        }
        // 与通用 AVE 共用 sum/count 二元规约；组外 padding 不计入 local_count。
        reduce_sum_count(local_sum, local_count);
    }
};
}

struct CVGraphExecution::Impl {
    LAMMPS_NS::LAMMPS* lmp;
    std::vector<CVGraphSymbol> symbols;
    std::vector<std::string> expressions;
    std::vector<std::unique_ptr<CVInput>> inputs;
    CVAtomLayout layout;
    std::vector<int> masks;
    int nlocal = -1, nghost = -1;
    // 声明顺序保证先卸载执行器，再销毁图，最后释放 Leaf 与缓冲区。
    std::unique_ptr<Graph> graph;
    std::unique_ptr<GraphExecution> execution;

    Impl(LAMMPS_NS::LAMMPS* lmp_, std::vector<CVGraphSymbol> symbols_, std::vector<std::string> expressions_)
        : lmp(lmp_), symbols(std::move(symbols_)), expressions(std::move(expressions_)) {
        for (const auto& symbol : symbols) {
            if (!symbol.cv || symbol.cv->graph_groupbit() <= 0)
                throw std::runtime_error("Graph: CV for symbol '" + symbol.name + "' has no raw graph interface");
            if (symbol.method != "AVE" && symbol.method != "MEAN" && symbol.method != "RAW" && symbol.method != "value")
                throw std::runtime_error("Graph: supported CV methods are AVE/MEAN/RAW/value; got " + symbol.method);
            auto found = std::find_if(inputs.begin(), inputs.end(), [&](const auto& x) { return x->cv == symbol.cv; });
            if (found == inputs.end()) {
                auto input = std::make_unique<CVInput>(symbol.cv);
                input->values.set_name("graph_values", nullptr, lmp->error, lmp);
                input->gradients.set_name("graph_adjoint", nullptr, lmp->error, lmp);
                inputs.push_back(std::move(input));
            }
        }
    }
    void prepare() {
        auto* atom = lmp->atom;
        const int total = atom->nlocal + atom->nghost;
        // 映射只依赖本地下标及 mask；同组内部排序不改变映射，无需检查全局 tag。
        const bool changed = nlocal != atom->nlocal || nghost != atom->nghost ||
            masks.size() != static_cast<size_t>(total) || !std::equal(masks.begin(), masks.end(), atom->mask);
        if (changed) {
            std::vector<int> groups;
            for (const auto& x : inputs) groups.push_back(x->cv->graph_groupbit());
            layout.rebuild(atom->nlocal, atom->nghost, atom->mask, groups);
            for (auto& x : inputs) {
                x->assembly = layout.group(x->cv->graph_groupbit());
                x->cv->bind_assembly(x->assembly);
                // GpuBuffer 对象地址保持稳定；Leaf 借用的是对象，不是旧 ptr。
                if (x->values.capacity < static_cast<size_t>(layout.leaf_count())) {
                    x->values.grow_to(layout.leaf_count(), __FILE__, __LINE__);
                    x->gradients.grow_to(layout.leaf_count(), __FILE__, __LINE__);
                }
                x->leaf.n = layout.leaf_count();
                x->leaf.refresh_buffers();
            }
            nlocal = atom->nlocal;
            nghost = atom->nghost;
            masks.clear();
            if (total) masks.assign(atom->mask, atom->mask + total);
            if (execution) execution->resize_inputs();
        }
        // 每轮执行，包括布局不变及同一步重复评估。必须在任何 CV::base_calc 前完成。
        for (auto& x : inputs) x->cv->prepare_graph_evaluation();
    }
    void build() {
        graph = std::make_unique<Graph>(Graph::Symbols{}, Graph::Switches{}, lmp->world);
        for (const auto& symbol : symbols) {
            auto found = std::find_if(inputs.begin(), inputs.end(), [&](const auto& x) { return x->cv == symbol.cv; });
            Node* node = &(*found)->leaf;
            if (symbol.method == "AVE" || symbol.method == "MEAN")
                node = graph->make_node<CVGroupAverage>(**found, lmp->world);
            if (!graph->symbols.emplace(symbol.name, node).second)
                throw std::runtime_error("Graph: duplicate symbol " + symbol.name);
        }
        for (const auto& expr : expressions) graph->parse_expr(expr);
        graph->build();
        // 每个 rank 建立自己的 CUDA module，只有 rank 0 写 LAMMPS screen/log。
        execution = std::make_unique<GraphExecution>(*graph,
            lmp->comm->me == 0 ? lmp->screen : nullptr, lmp->comm->me == 0 ? lmp->logfile : nullptr);
    }
    std::vector<double> forward() {
        if (!execution) build();
        for (auto& x : inputs) x->cv->assemble_values(x->values);
        // 当前 CV 与装配 kernel 均使用默认流，流内顺序保证全部依赖先于融合 kernel。
        return execution->forward();
    }
    void backward(const std::vector<double>& seeds) {
        if (!execution) throw std::logic_error("CV Graph: forward is required");
        execution->backward(seeds);
        for (auto& x : inputs) {
            if (std::find(graph->topo.begin(), graph->topo.end(), &x->leaf) != graph->topo.end())
                x->cv->apply_adjoint(x->gradients); // 内部负责 ghost 力的 reverse_comm。
        }
    }
};
CVGraphExecution::CVGraphExecution(LAMMPS_NS::LAMMPS* lmp, std::vector<CVGraphSymbol> symbols, std::vector<std::string> expressions)
    : impl_(std::make_unique<Impl>(lmp, std::move(symbols), std::move(expressions))) {}
CVGraphExecution::~CVGraphExecution() = default;
void CVGraphExecution::prepare() { impl_->prepare(); }
std::vector<double> CVGraphExecution::forward() { return impl_->forward(); }
void CVGraphExecution::backward(const std::vector<double>& seeds) { impl_->backward(seeds); }
}
