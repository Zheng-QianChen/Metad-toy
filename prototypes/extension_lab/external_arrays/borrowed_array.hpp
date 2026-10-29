#pragma once
#include <cstddef>
#include <stdexcept>

// 三分量设备数组的非拥有视图，逻辑形状为 [capacity, 3]，元素类型为 double。
// 第 atom 个原子的第 component 个分量位于：
//     data[atom * atom_stride + component * component_stride]
// 两种 stride 均以 double 元素计数。例如 N 个原子的 SoA 为 (1,N)，AoS 为 (3,1)。
//
// 本原型假设 owned 原子排列在 [0,owned)，ghost 排列在 [owned,capacity)。
// 引擎负责底层存储与原子顺序；持有者必须活到消费流完成，且消费期间保持地址有效。
// 扩容、迁移或重排后由调用方重新取得 view，以便更新地址、数量和索引对应关系。
// 该 POD 按值作为 kernel 参数传入，因此 Host 和 NVRTC 中的字段类型、顺序须一致。
struct BorrowedArray {
    double* data;                  // 当前 CUDA 设备可访问的数组起点，由外部容器持有。
    std::size_t atom_stride;        // 原子索引增加 1 时跨过的元素数。
    std::size_t component_stride;   // x→y 或 y→z 时跨过的元素数。
    int capacity;   // 包括 ghost 的可寻址原子数。
    int owned;      // 此次计算负责的前 owned 个原子。
};
// 检查描述符的基本范围。capacity=0 时允许空指针，供零原子任务直接返回。
// 具体分配长度和 stride 所指地址范围由创建 view 的引擎保证。
inline void validate(const BorrowedArray& view) {
    if (view.owned < 0 || view.capacity < view.owned ||
        (view.capacity && (!view.data || !view.atom_stride || !view.component_stride)))
        throw std::invalid_argument("Invalid borrowed array extent or stride");
}
// 为 GPUMD 的连续三段布局建立视图：x[N]、y[N]、z[N]，N=capacity。
// data 指向 3*N 个 double 的设备分配；owned 只控制本次负责的原子范围。
// 构造视图只记录已有地址和布局，原数组可以同时保留已有的力贡献。
inline BorrowedArray borrow_soa(double* data, int capacity, int owned) {
    BorrowedArray view{data, 1, static_cast<std::size_t>(capacity), capacity, owned};
    validate(view);
    return view;
}
// 从 Kokkos 的设备 View 取得二维布局，第二轴要求是三个空间分量。
// stride_0/stride_1 同时覆盖 LayoutLeft、LayoutRight 以及底层分配的填充。
// 调用方先完成设备数据的同步/生产，在消费完成后按需标记 DualView 的设备修改。
// 返回值借用 view.data()；原始 Kokkos 容器继续负责其分配的引用与生命周期。
template<class View>
BorrowedArray borrow_kokkos(View view, int owned) {
    if (view.extent(1) != 3) throw std::invalid_argument("Expected three components");
    BorrowedArray result{view.data(), view.stride_0(), view.stride_1(),
                         static_cast<int>(view.extent(0)), owned};
    validate(result);
    return result;
}
