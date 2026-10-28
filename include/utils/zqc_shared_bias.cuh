#pragma once
#include "zqc_array_view.h"
#include "zqc_collective.h"
#include <memory>
#include <string>

namespace MetaD_zqc {
// 冻结的一维共享表：每行 q、u(q)、du/dq，q 等间距且严格递增。
// 区间内使用三次 Hermite 插值；两端按端点斜率线性延拓，能量和力保持 C1。
// 一份对象对应一个 family。表在构造时读入，此后所有求值都使用同一份不可变数据。
class FrozenSharedBias {
public:
    struct Result {
        double energy = 0.0;          // 所有 rank 的有效中心上 MEAN(u(q))
        double mean_coordinate = 0.0; // 同一中心集合上 MEAN(q)，用于 thermo 记录
        long long count = 0;          // 全局 owned 中心数
        ArrayView<const double> gradient; // 公共 Leaf 顺序的 dV/dq；组外为零
    };
    FrozenSharedBias(const std::string& table_path, SumReducer sum = {});
    ~FrozenSharedBias();
    FrozenSharedBias(const FrozenSharedBias&) = delete;
    FrozenSharedBias& operator=(const FrozenSharedBias&) = delete;
    // 两个输入均借用 CUDA View；owned_slots 是不重复的有效 Leaf 槽位。
    // 空 rank 参与同一次 {sum_u,sum_q,count,invalid} 规约，全局空组报错。
    // 返回梯度借用到下一次 evaluate/析构；消费者须遵守 View 的等待契约。
    Result evaluate(ArrayView<const double> coordinates, ArrayView<const int> owned_slots);
    const std::string& fingerprint() const; // 按表内容计算的 FNV-1a 标识，用于复查固定表
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace MetaD_zqc
