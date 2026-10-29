// ArrayView 构造入口：独立验证真实跨度、填充、const 类型及就绪依赖。
#include "zqc_array_view.h"
#include <iostream>
#include <type_traits>
#include <vector>
using namespace MetaD_zqc;
namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class Function> void rejects(Function action) {
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "invalid array view accepted");
}
// 复现二维 Kokkos View 的布局查询接口；Host 测试无需安装 Kokkos/CUDA。
struct PaddedView {
    static constexpr int rank = 2;
    const double* pointer;
    const double* data() const { return pointer; }
    size_t extent(int dimension) const { return dimension ? 3 : 2; }
    size_t stride_0() const { return 2; }
    size_t stride_1() const { return 9; }
    size_t span() const { return 24; }
};
void waited(const void* context, void*) { ++*static_cast<int*>(const_cast<void*>(context)); }
}
int main() {
    try {
        std::vector<double> storage(24, -1);
        int waits = 0;
        const ViewReadiness readiness{&waits, waited, nullptr};
        auto aos = make_aos_view(storage.data(), 2, 3, storage.size(), MemorySpace::Host, readiness);
        auto soa = make_soa_view(storage.data(), 2, 3, 8, storage.size(), MemorySpace::Host, readiness);
        for (size_t atom=0;atom<2;++atom) for(size_t component=0;component<3;++component) {
            aos(atom,component)=10*atom+component;
            require(storage[3*atom+component]==10*atom+component,"AoS indexing mismatch");
        }
        for (size_t atom=0;atom<2;++atom) for(size_t component=0;component<3;++component) {
            soa(atom,component)=20*atom+component;
            require(storage[atom+8*component]==20*atom+component,"SoA allocation stride mismatch");
        }
        auto kokkos = make_kokkos_view(PaddedView{storage.data()}, MemorySpace::Host, readiness);
        static_assert(std::is_same<decltype(kokkos),ArrayView<const double>>::value,"lost const element type");
        require(kokkos.item_stride==2 && kokkos.component_stride==9 && kokkos.storage_size==24,"Kokkos padding lost");
        storage[20]=123;
        require(kokkos(1,2)==123,"Kokkos strided indexing mismatch");
        aos.wait_on(nullptr); soa.wait_on(nullptr); kokkos.wait_on(nullptr);
        require(waits==3,"factory dropped readiness hook");
        make_aos_view<double>(nullptr,0,3,0,MemorySpace::Cuda);
        make_soa_view<double>(nullptr,0,3,0,0,MemorySpace::Cuda);
        rejects([&]{make_aos_view<double>(nullptr,1,3,3,MemorySpace::Cuda);});
        rejects([&]{make_aos_view(storage.data(),2,3,5,MemorySpace::Host);});
        rejects([&]{make_soa_view(storage.data(),3,3,2,24,MemorySpace::Host);});
        rejects([&]{make_soa_view(storage.data(),2,3,8,17,MemorySpace::Host);});
        std::cout << "PASS: ArrayView AoS/SoA/Kokkos construction, padding and readiness\n";
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n';return 1; }
}
