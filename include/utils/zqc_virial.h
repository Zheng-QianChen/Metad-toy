#pragma once
#include "zqc_array_view.h"

#ifdef __CUDACC__
#define ZQC_VIRIAL_INLINE __host__ __device__ __forceinline__
#else
#define ZQC_VIRIAL_INLINE inline
#endif

namespace MetaD_zqc {
// 表示张量包含哪些分量；AoS/SoA、预留容量由 ArrayView 的 stride 描述。
// Symmetric6: xx yy zz xy xz yz，调用方保证每个原子的张量对称。
// Full9:      xx yy zz xy xz yz yx zx zy，三个转置分量各自独立。
enum class VirialRepresentation { Symmetric6, Full9 };

// 借用逐原子构型 virial W = sum(r tensor F)，单位为能量。
// 数值包含本 CV 的偏置种子；逐原子分配方式由 CV 定义，所有原子之和为总 W。
// 对旋转不变的能量，完整汇总后的总 W 对称；某个原子的 Full9 可有反对称部分。
// 六分量表示适用于计算公式保证逐原子对称的分配，不能仅凭总 W 对称来选择它。
// 压力的体积归一化、stress 的负号/单位转换由宿主输出接口负责。
// values 的行号是 localid，存储列数必须与 representation 一致。
// 例如 GPUMD Full9 SoA: item_stride=1，component_stride=实际原子槽位数。
struct VirialView {
    ArrayView<const double> values;
    VirialRepresentation representation = VirialRepresentation::Symmetric6;

    // Host 入口检查元数据后再传给 kernel，CUDA 内存只验证范围。
    void validate() const {
        size_t components;
        switch (representation) {
        case VirialRepresentation::Symmetric6: components = 6; break;
        case VirialRepresentation::Full9: components = 9; break;
        default: throw std::invalid_argument("VirialView: unknown representation");
        }
        if (values.component_count != components)
            throw std::invalid_argument("VirialView: component count does not match representation");
        values.validate();
    }

    // 按 Full9 顺序读取分量，component 范围为 [0,9)。对称输入补齐转置项，
    // Full9 直接读取独立项。每次访问仅做内联寻址，不分配九分量暂存。
    ZQC_VIRIAL_INLINE double full_component(size_t atom, size_t component) const {
        if (representation == VirialRepresentation::Symmetric6 && component >= 6)
            component -= 3;
        return values(atom, component);
    }
};
static_assert(std::is_trivially_copyable<VirialView>::value, "virial views must be kernel arguments");
} // namespace MetaD_zqc
#undef ZQC_VIRIAL_INLINE
