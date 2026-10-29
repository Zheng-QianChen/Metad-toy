#pragma once
#include <cstddef>
#include <functional>

namespace MetaD_zqc {
// 原地求和回调：输入本地值，返回各参与者之和；空回调表示单进程。
// 宿主负责通信器和同步。Graph 保存回调副本，捕获对象须覆盖图的生命周期。
using SumReducer = std::function<void(double*, size_t)>;
}
