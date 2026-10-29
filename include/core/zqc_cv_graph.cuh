#pragma once
#include "zqc_engine.h"
#include <memory>
#include <string>
#include <vector>

namespace MetaD_zqc {
class CV;
class EngineAdapter;
struct CVGraphSymbol {
    std::string name;
    CV* cv;
    std::string method;
};

// 引擎与 Graph 的边界：PerAtom Leaf 借用 GPU 装配数组，Global Leaf 保存 Host 标量。
// 持有装配表、Leaf 和执行器，管理动态本地长度及两类 CV 的梯度回传。
// CV 与 EngineAdapter 由外部持有；本对象必须先于 CV 析构。
class CVGraphExecution {
public:
    CVGraphExecution(EngineAdapter&, std::vector<CVGraphSymbol>, std::vector<std::string>,
                     std::vector<SelectionId> output_selections = {});
    ~CVGraphExecution();
    bool uses(const CV*) const; // 表达式可达的 Leaf；多个符号引用同一 CV 时合并。
    void prepare(); // 必须在所有 CV::base_calc 之前绑定本步 owned/ghost 表。
    std::vector<double> forward(); // base_calc 提交后装配，融合流等待事件，再 Host/跨进程规约。
    void backward(const std::vector<double>& seeds,
                  const std::vector<ArrayView<const double>>& atom_seeds = {});
    // forward 返回 Global 输出；PerAtom 输出按表达式总编号单独借用。
    ArrayView<const double> per_atom_output(size_t output_index) const;
    ArrayView<const int> selected_slots(SelectionId selection) const; // 所有输出先汇总梯度，每个 CV 只反传一次。
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
