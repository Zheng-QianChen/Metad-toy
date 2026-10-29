#include "zqc_cv_graph.cuh"
#include "zqc_cv.h"
#include "zqc_cv_assembly.cuh"
#include "zqc_graph.cuh"
#include <algorithm>
#include <stdexcept>

namespace MetaD_zqc {
namespace {
struct CVInput {
    CV* cv;
    GpuBuffer<double> values, gradients;
    bool global;
    bool active = false; // 从输出拓扑可达的 Leaf。
    std::unique_ptr<Leaf> leaf;
    std::shared_ptr<const CVAssembly> assembly;
    explicit CVInput(CV* cv_) : cv(cv_), global(cv_->supports_global_graph()) {
        // Global 值驻留 Host；PerAtom Leaf 借用装配用的 GPU 缓冲区。
        if (global) leaf = std::make_unique<Leaf>(std::vector<double>{0.0}, ValueDesc{ValueDomain::Global, 1});
        else leaf = std::make_unique<Leaf>(values, gradients, 0);
    }
    void rebind_leaf(int leaf_count) {
        if (global) return;
        leaf->n = leaf_count;
        leaf->bind(ArrayView<const double>(values.view(leaf_count)), gradients.view(leaf_count));
    }
};

// The common Leaf layout is a union of families. A nonlinear function of a
// zero-padded input need not be zero, so SUM must mask both values and adjoints.
struct SumFamily {
    SelectionId id;
    GpuBuffer<double> values, gradients;
    std::unique_ptr<Leaf> leaf;
    explicit SumFamily(SelectionId id_) : id(id_), leaf(std::make_unique<Leaf>(values,gradients,0)) {}
    void bind(const CVAtomLayout& layout) {
        const auto assembly=layout.group(id);
        const int count=layout.leaf_count();
        std::vector<double> mask(count,0.0);
        for(int slot:assembly->leaf_slots()) if(slot>=0) mask[slot]=1.0;
        values.grow_to(count,__FILE__,__LINE__); gradients.grow_to(count,__FILE__,__LINE__);
        if(count) values.upload_from(mask.data(),count);
        leaf->n=count;
        leaf->bind(ArrayView<const double>(values.view(count)),gradients.view(count));
    }
};

// CV.AVE 只对该 CV 的 owned 组成员求均值，不能把公共 Leaf 的组外零值计入分母。
// 此节点只接原始 Leaf，因此反向给 padding 的种子最终会被 gather_adjoint 丢弃。
struct CVGroupAverage : AVE {
    CVInput& input;
    CVGroupAverage(CVInput& input_, SumReducer sum)
        : AVE(input_.leaf.get(), std::move(sum)), input(input_) {}
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

// CV.SUM 只对该 CV 的 owned 中心求和；组外 padding 不参与，即使非线性后非零也不计入。
// 反向仍可能给 padding 槽位写种子，gather_adjoint 会丢弃，与 CV.AVE 一致。
struct CVGroupSum : SUM {
    CVInput& input;
    CVGroupSum(CVInput& input_, SumReducer sum)
        : SUM(input_.leaf.get(), std::move(sum)), input(input_) {}
    void reduce_host(const double* values, int count) override {
        if (!input.assembly || count != input.assembly->leaf_count())
            throw std::logic_error("CV.SUM: stale assembly");
        double local_sum = 0.0, correction = 0.0;
        const int local_count = input.assembly->group_count();
        for (int k = 0; k < local_count; ++k) {
            const double adjusted = values[input.assembly->leaf_slots()[k]] - correction;
            const double next = local_sum + adjusted;
            correction = (next - local_sum) - adjusted;
            local_sum = next;
        }
        reduce_sum(local_sum);
    }
};

// 深拷贝成员列表，避免引擎下一轮求值后 SelectionView 悬空。
struct SelectionSnapshot {
    SelectionId id = all_atoms;
    int owned_count = 0;
    std::uint64_t generation = 0;
    std::vector<LocalAtomIndex> local_ids;
    explicit SelectionSnapshot(const SelectionView& view)
        : id(view.id), owned_count(view.owned_count), generation(view.generation), local_ids(view.local_ids.item_count) {
        for (size_t index = 0; index < local_ids.size(); ++index) local_ids[index] = view.local_ids(index);
    }
    bool same_as(const SelectionView& other) const {
        if (id != other.id || owned_count != other.owned_count || local_ids.size() != other.local_ids.item_count)
            return false;
        if (generation && other.generation) return generation == other.generation;
        for (size_t index = 0; index < local_ids.size(); ++index)
            if (local_ids[index] != other.local_ids(index)) return false;
        return true;
    }
};
} // namespace

struct CVGraphExecution::Impl {
    EngineAdapter& engine;
    std::vector<CVGraphSymbol> symbols;
    std::vector<std::string> expressions;
    std::vector<SelectionId> output_selections; // 外部逐原子偏势的中心集合，一并参与布局刷新。
    std::vector<std::unique_ptr<CVInput>> inputs;
    std::vector<std::unique_ptr<SumFamily>> sum_families;
    CVAtomLayout layout;
    std::vector<SelectionSnapshot> cached_selections;
    int owned = -1, total = -1;
    std::uint64_t last_evaluation_id = 0;
    // 声明顺序保证先卸载执行器，再销毁图，最后释放 Leaf 与缓冲区。
    std::unique_ptr<Graph> graph;
    std::unique_ptr<GraphExecution> execution;

    Impl(EngineAdapter& engine_, std::vector<CVGraphSymbol> symbols_,
         std::vector<std::string> expressions_, std::vector<SelectionId> selections_)
        : engine(engine_), symbols(std::move(symbols_)), expressions(std::move(expressions_)),
          output_selections(std::move(selections_)) {
        Logger logger = engine.logger();
        for (const auto& symbol : symbols) {
            if (!symbol.cv)
                throw std::runtime_error("Graph: null CV for symbol '" + symbol.name + "'");
            const SelectionId selection = symbol.cv->graph_selection();
            if (!symbol.cv->supports_global_graph() && selection == -2)
                throw std::runtime_error("Graph: CV for symbol '" + symbol.name + "' has no raw graph interface");
            if (symbol.method != "AVE" && symbol.method != "MEAN" && symbol.method != "SUM"
                && symbol.method != "RAW" && symbol.method != "value")
                throw std::runtime_error("Graph: supported CV methods are AVE/MEAN/SUM/RAW/value; got " + symbol.method);
            if (symbol.cv->supports_global_graph() && symbol.method != "RAW" && symbol.method != "value")
                throw std::runtime_error("Graph: Global CV methods are RAW/value; got " + symbol.method);
            auto found = std::find_if(inputs.begin(), inputs.end(),
                                      [&](const auto& x) { return x->cv == symbol.cv; });
            if (found == inputs.end()) {
                auto input = std::make_unique<CVInput>(symbol.cv);
                input->values.set_name("graph_values", logger);
                input->gradients.set_name("graph_adjoint", logger);
                inputs.push_back(std::move(input));
            }
        }
        // 构造期以 n=0 解析拓扑，确定实际输入；prepare 设置本地长度，首次 forward 执行 NVRTC。
        build();
        for (auto& input : inputs)
            input->active = std::find(graph->topo.begin(), graph->topo.end(), input->leaf.get()) != graph->topo.end();
    }

    bool uses(const CV* cv) const {
        return std::any_of(inputs.begin(), inputs.end(),
                           [cv](const auto& input) { return input->active && input->cv == cv; });
    }

    std::vector<SelectionView> collect_selections() {
        std::vector<SelectionId> ids = output_selections;
        for(const auto& family:sum_families)
            if(std::find(ids.begin(),ids.end(),family->id)==ids.end()) ids.push_back(family->id);
        for (const auto& x : inputs) {
            if (!x->active || x->global) continue;
            const SelectionId id = x->cv->graph_selection();
            if (std::find(ids.begin(), ids.end(), id) == ids.end()) ids.push_back(id);
        }
        std::vector<SelectionView> snapshots;
        snapshots.reserve(ids.size());
        for (SelectionId id : ids) {
            snapshots.push_back(engine.selection_members(id));
            snapshots.back().local_ids.wait_host();
        }
        return snapshots;
    }

    bool layout_changed(int next_owned, int next_total,
                        const std::vector<SelectionView>& next) const {
        if (owned != next_owned || total != next_total || cached_selections.size() != next.size())
            return true;
        for (size_t index = 0; index < next.size(); ++index)
            if (!cached_selections[index].same_as(next[index])) return true;
        return false;
    }

    void prepare() {
        const bool has_per_atom = std::any_of(inputs.begin(), inputs.end(),
                                              [](const auto& x) { return x->active && !x->global; });
        const int next_owned = engine.owned_atom_count();
        const int next_total = engine.total_atom_count();
        if (next_owned < 0 || next_total < next_owned)
            throw std::runtime_error("Graph: engine atom counts are invalid");
        std::vector<SelectionView> next_selections;
        if (has_per_atom) next_selections = collect_selections();
        const bool changed = has_per_atom && layout_changed(next_owned, next_total, next_selections);
        if (changed) {
            layout.rebuild(next_owned, next_total - next_owned, next_selections);
            for(auto& family:sum_families) family->bind(layout);
            for (auto& x : inputs) {
                if (!x->active || x->global) continue;
                x->assembly = layout.group(x->cv->graph_selection());
                x->cv->bind_assembly(x->assembly);
                // GpuBuffer 对象地址保持稳定；Leaf 在扩容后必须重新 bind 视图。
                if (x->values.capacity < static_cast<size_t>(layout.leaf_count())) {
                    x->values.grow_to(layout.leaf_count(), __FILE__, __LINE__);
                    x->gradients.grow_to(layout.leaf_count(), __FILE__, __LINE__);
                }
                x->rebind_leaf(layout.leaf_count());
            }
            owned = next_owned;
            total = next_total;
            cached_selections.clear();
            for (const auto& selection : next_selections) cached_selections.emplace_back(selection);
            if (execution) execution->resize_inputs();
        } else if (has_per_atom) {
            // 容量未变时仍刷新指针，覆盖同 capacity 下的底层再分配。
            for (auto& x : inputs) if (x->active && !x->global) x->rebind_leaf(layout.leaf_count());
        }
        for (const Node* node : graph->topo)
            if (node->value_desc.domain == ValueDomain::PerAtom)
                const_cast<Node*>(node)->n = layout.leaf_count();
        last_evaluation_id = engine.evaluation_id();
        // 每轮执行，包括布局不变及同一步重复评估。必须在任何 CV::base_calc 前完成。
        for (auto& x : inputs) if (x->active) x->cv->prepare_graph_evaluation();
    }

    Node* scoped_sum(Graph& target,const GraphExpr& call) {
        if(call.args.size()!=1 && call.args.size()!=2)
            call.fail("SUM expects (expression[, family_raw_symbol])");
        Node* source=target.create_node(call.args[0]);
        std::vector<SelectionId> families;
        auto add_family=[&](const Node* node) {
            for(const auto& input:inputs) if(!input->global && input->leaf.get()==node) {
                const auto id=input->cv->graph_selection();
                if(std::find(families.begin(),families.end(),id)==families.end()) families.push_back(id);
            }
        };
        if(call.args.size()==2) {
            if(call.args[1].kind!=GraphExpr::Kind::Symbol) call.fail("SUM family must be a RAW symbol");
            add_family(target.create_node(call.args[1]));
        } else {
            std::vector<const Node*> pending{source},seen;
            while(!pending.empty()) {
                const Node* node=pending.back(); pending.pop_back();
                if(std::find(seen.begin(),seen.end(),node)!=seen.end()) continue;
                seen.push_back(node); add_family(node);
                for(const auto* child:node->inputs()) pending.push_back(child);
            }
        }
        if(families.size()!=1)
            call.fail("SUM requires one center family; use SUM(expression, family_raw_symbol) for mixed groups");
        const auto id=families.front();
        auto found=std::find_if(sum_families.begin(),sum_families.end(),[&](const auto& x){return x->id==id;});
        if(found==sum_families.end()) {
            sum_families.push_back(std::make_unique<SumFamily>(id)); found=std::prev(sum_families.end());
        }
        auto* masked=target.make_node<Mul>(source,(*found)->leaf.get());
        return target.make_node<SUM>(masked,engine.sum_reducer());
    }

    void build() {
        graph = std::make_unique<Graph>(Graph::Symbols{}, Graph::Switches{}, engine.sum_reducer());
        auto creator=[this](Graph& target,const GraphExpr& call){return scoped_sum(target,call);};
        graph->register_node("SUM",creator,true);
        graph->register_node(".SUM",creator,true);
        for (const auto& symbol : symbols) {
            auto found = std::find_if(inputs.begin(), inputs.end(),
                                      [&](const auto& x) { return x->cv == symbol.cv; });
            Node* node = (*found)->leaf.get();
            if (symbol.method == "AVE" || symbol.method == "MEAN")
                node = graph->make_node<CVGroupAverage>(**found, engine.sum_reducer());
            else if (symbol.method == "SUM")
                node = graph->make_node<CVGroupSum>(**found, engine.sum_reducer());
            if (!graph->symbols.emplace(symbol.name, node).second)
                throw std::runtime_error("Graph: duplicate symbol " + symbol.name);
        }
        for (const auto& expr : expressions) graph->parse_expr(expr);
        graph->build();
    }

    std::vector<double> forward() {
        if (!execution) execution = std::make_unique<GraphExecution>(*graph, engine.logger());
        if (last_evaluation_id != engine.evaluation_id())
            throw std::logic_error("Graph: evaluation_id changed after prepare; call prepare again");
        for (auto& x : inputs) {
            if (!x->active) continue;
            if (x->global) {
                x->leaf->val[0] = x->cv->graph_global_value();
            } else {
                x->cv->assemble_values(x->values.view(layout.leaf_count()));
                auto values = x->values.view(layout.leaf_count());
                values.readiness = x->cv->device_readiness();
                x->leaf->bind(ArrayView<const double>(values), x->gradients.view(layout.leaf_count()));
            }
        }
        // Leaf 带有各环境装配完成的事件，融合流在读取前逐一等待。
        return execution->forward();
    }

    void backward(const std::vector<double>& seeds,
                  const std::vector<ArrayView<const double>>& atom_seeds) {
        if (!execution) throw std::logic_error("CV Graph: forward is required");
        if (last_evaluation_id != engine.evaluation_id())
            throw std::logic_error("Graph: evaluation_id changed after prepare; call prepare again");
        execution->backward(seeds, atom_seeds);
        // 每个 CV 先提交本轮坐标导数，完整反向结束后再统一回传 ghost 力/virial。
        // CV 异常时丢弃尚未写回的批次，避免下一次求值沿用部分结果。
        engine.begin_force_virial_batch();
        try {
            for (auto& x : inputs) {
                if (std::find(graph->topo.begin(), graph->topo.end(), x->leaf.get()) == graph->topo.end())
                    continue;
                if (x->global) x->cv->apply_global_adjoint(x->leaf->adj[0]);
                else x->cv->apply_adjoint(ArrayView<const double>(x->gradients.view(layout.leaf_count())));
            }
            engine.finish_force_virial_batch();
        } catch (...) {
            engine.cancel_force_virial_batch();
            throw;
        }
    }
};

CVGraphExecution::CVGraphExecution(EngineAdapter& engine, std::vector<CVGraphSymbol> symbols,
                                   std::vector<std::string> expressions, std::vector<SelectionId> selections)
    : impl_(std::make_unique<Impl>(engine, std::move(symbols), std::move(expressions), std::move(selections))) {}
CVGraphExecution::~CVGraphExecution() = default;
bool CVGraphExecution::uses(const CV* cv) const { return impl_->uses(cv); }
void CVGraphExecution::prepare() { impl_->prepare(); }
std::vector<double> CVGraphExecution::forward() { return impl_->forward(); }
void CVGraphExecution::backward(const std::vector<double>& seeds,
        const std::vector<ArrayView<const double>>& atom_seeds) { impl_->backward(seeds, atom_seeds); }
ArrayView<const double> CVGraphExecution::per_atom_output(size_t index) const {
    if (!impl_->execution) throw std::logic_error("CV Graph: forward is required");
    return impl_->execution->per_atom_output(index);
}
ArrayView<const int> CVGraphExecution::selected_slots(SelectionId selection) const {
    return impl_->layout.group(selection)->owned_leaf_slots();
}
} // namespace MetaD_zqc
