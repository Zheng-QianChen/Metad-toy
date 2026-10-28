#pragma once
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>

#ifdef __CUDACC__
#define ZQC_VIEW_INLINE __host__ __device__ __forceinline__
#else
#define ZQC_VIEW_INLINE inline
#endif

namespace MetaD_zqc {
enum class MemorySpace { Host, Cuda };

// 仅在 Host 调用的就绪钩子。context 借用生产者状态；空钩子表示数据已就绪。
// consumer_stream 为后端的 CUDA stream 句柄（nullptr 表示 CUDA 默认流）。
// 两个等待入口分开：GPU 等待排入消费流，CPU 等待保证返回后可读/可写。
// 生产者在发布视图前记录完成事件，并在消费结束前保持存储、事件和原子排序稳定。
struct ViewReadiness {
    const void* context = nullptr;
    void (*wait_on_callback)(const void*, void*) = nullptr;
    void (*wait_host_callback)(const void*) = nullptr;
    void wait_on(void* consumer_stream) const {
        if (wait_on_callback) wait_on_callback(context, consumer_stream);
        else if (wait_host_callback) wait_host_callback(context);
    }
    void wait_host() const {
        if (wait_host_callback) wait_host_callback(context);
        else if (wait_on_callback)
            throw std::logic_error("ViewReadiness: Host wait hook is required for CPU access");
    }
};

// 借用逻辑二维数组 [item, component]；stride 和 storage_size 均以 T 元素计。
// storage_size 是从 data 开始可访问的存储长度，item_count 是有效项数。
// AoS xyz: item_stride=3, component_stride=1；SoA xyz: item_stride=1,
// component_stride=引擎实际分配的原子槽位数（可能大于本轮有效原子数）。
// 所有者负责内存与 readiness；消费者在启动前调用 wait_on，CPU 访问前调用 wait_host。
// 本描述符按值传给 kernel，回调仅在 Host 执行；扩容、排序或下一次求值后重新取得视图。
template<class T> struct ArrayView {
    T* data = nullptr;
    size_t item_count = 0;
    size_t component_count = 1;
    size_t item_stride = 1;
    size_t component_stride = 1;
    size_t storage_size = 0;
    MemorySpace memory = MemorySpace::Host;
    ViewReadiness readiness{};

    void wait_on(void* consumer_stream) const { readiness.wait_on(consumer_stream); }
    void wait_host() const { readiness.wait_host(); }

    // 可写视图可作为只读输入借用，保持同一形状、跨度和生命周期。
    template<class U = T, typename std::enable_if<!std::is_const<U>::value, int>::type = 0>
    ZQC_VIEW_INLINE operator ArrayView<const U>() const {
        return {data,item_count,component_count,item_stride,component_stride,storage_size,memory,readiness};
    }

    // kernel 热循环使用已验证的尺寸与下标；此操作本身不访问宿主接口。
    ZQC_VIEW_INLINE T& operator()(size_t item, size_t component = 0) const {
        return data[item*item_stride + component*component_stride];
    }

    // 在 Host 绑定数组时验证可达范围；CUDA 地址仅检查元数据，不解引用。
    void validate() const {
        if (!component_count) throw std::invalid_argument("ArrayView: zero component count");
        if (memory != MemorySpace::Host && memory != MemorySpace::Cuda)
            throw std::invalid_argument("ArrayView: unknown memory space");
        if (!item_count) return;
        if (!data || !item_stride || !component_stride)
            throw std::invalid_argument("ArrayView: null storage or zero stride");
        const size_t maximum = std::numeric_limits<size_t>::max();
        if (item_count-1 > maximum/item_stride || component_count-1 > maximum/component_stride)
            throw std::length_error("ArrayView: stride overflow");
        const size_t item_offset = (item_count-1)*item_stride;
        const size_t component_offset = (component_count-1)*component_stride;
        if (component_offset > maximum-item_offset || item_offset+component_offset >= storage_size)
            throw std::length_error("ArrayView: storage does not cover logical shape");
    }
};
static_assert(std::is_trivially_copyable<ArrayView<double>>::value, "kernel views must be POD copies");

// 构造函数借用已有存储。storage_size 为从 data 开始的实际可访问元素数，
// item_count 为本轮有效项数；owned/ghost 的归属范围由引擎或选择集单独提供。
template<class T> ArrayView<T> make_aos_view(T* data, size_t item_count, size_t component_count,
                                           size_t storage_size, MemorySpace memory,
                                           ViewReadiness readiness = {}) {
    ArrayView<T> result{data,item_count,component_count,component_count,1,storage_size,memory,readiness};
    result.validate();
    return result;
}
// 每个分量占 allocated_items 个槽位，容量可大于本轮有效项数。
template<class T> ArrayView<T> make_soa_view(T* data, size_t item_count, size_t component_count,
                                           size_t allocated_items, size_t storage_size, MemorySpace memory,
                                           ViewReadiness readiness = {}) {
    if (item_count > allocated_items) throw std::invalid_argument("ArrayView: items exceed SoA slots");
    ArrayView<T> result{data,item_count,component_count,1,allocated_items,storage_size,memory,readiness};
    result.validate();
    return result;
}
// 读取二维 Kokkos View 的实际跨度和 span，覆盖 LayoutLeft/Right/Stride 及填充。
// 生产者负责同步 DualView，并用 readiness 声明写入完成点；本头文件不依赖 Kokkos。
template<class View> auto make_kokkos_view(const View& view, MemorySpace memory,
                                         ViewReadiness readiness = {}) {
    static_assert(View::rank == 2, "ArrayView requires a rank-two Kokkos View");
    using Element = typename std::remove_pointer<decltype(view.data())>::type;
    ArrayView<Element> result{view.data(),view.extent(0),view.extent(1),view.stride_0(),view.stride_1(),
                              view.span(),memory,readiness};
    result.validate();
    return result;
}
} // namespace MetaD_zqc
#undef ZQC_VIEW_INLINE
