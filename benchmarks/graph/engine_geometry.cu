#include "CV_Stru_factor.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <vector>

using namespace MetaD_zqc;
namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class T> ArrayView<const T> view(const GpuBuffer<T>& buffer, size_t count) {
    return {buffer.ptr, count, 1, 1, 1, buffer.capacity, MemorySpace::Cuda};
}
// 模拟九分量宿主的逐原子累计。每线程独占原子，输出 stride 由宿主提供。
__global__ void accumulate_full_virial(VirialView source, ArrayView<double> destination) {
    const size_t atom = blockIdx.x*blockDim.x+threadIdx.x;
    if (atom >= source.values.item_count) return;
    for (int component = 0; component < 9; ++component)
        destination(atom,component) += source.full_component(atom,component);
}
void check_device_virials() {
    const double full[9] = {4,10,18,5,6,12,8,12,15};
    const double symmetric[9] = {4,10,18,5,6,12,5,6,12};
    const size_t atoms = 2, slots = 7;
    // 分别组合 6/9、输入 AoS/SoA 和输出 AoS/SoA，覆盖八条布局路径。
    for (bool nine : {false,true}) for (bool input_soa : {false,true}) for (bool output_soa : {false,true}) {
        const size_t components = nine ? 9 : 6;
        std::vector<double> input(slots*components, -999), output(slots*9, -777);
        for (size_t atom = 0; atom < atoms; ++atom) {
            for (size_t component = 0; component < components; ++component)
                input[input_soa?atom+slots*component:atom*components+component] = (atom+1)*full[component];
            for (size_t component = 0; component < 9; ++component)
                output[output_soa?atom+slots*component:atom*9+component] = 1;
        }
        auto expected = output;
        for (size_t atom = 0; atom < atoms; ++atom)
            for (size_t component = 0; component < 9; ++component)
                expected[output_soa?atom+slots*component:atom*9+component] +=
                    (atom+1)*(nine?full[component]:symmetric[component]);
        GpuBuffer<double> device_input(input.size()), device_output(output.size());
        device_input.upload_from(input.data(),input.size());
        device_output.upload_from(output.data(),output.size());
        VirialView source{{device_input.ptr,atoms,components,input_soa?1:components,input_soa?slots:1,
                           device_input.capacity,MemorySpace::Cuda},
                          nine?VirialRepresentation::Full9:VirialRepresentation::Symmetric6};
        ArrayView<double> destination{device_output.ptr,atoms,9,output_soa?1:size_t(9),output_soa?slots:1,
                                      device_output.capacity,MemorySpace::Cuda};
        source.validate(); destination.validate();
        accumulate_full_virial<<<1,32>>>(source,destination);
        const auto status = cudaGetLastError();
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        device_output.download_to(output.data(),output.size());
        require(output == expected, "CUDA virial transfer changed a component or padding");
        auto unchanged_input = input;
        device_input.download_to(unchanged_input.data(),unchanged_input.size());
        require(input == unchanged_input, "CUDA virial transfer changed source");
    }
}

struct Result { std::vector<double> values, force, virial; };
Result evaluate(bool soa) {
    // 中心 localid=1；0 是 owned 组外邻居，2 是 ghost 邻居，3 在 cutoff 外。
    // 0 的负坐标代表引擎已提供的周期镜像，几何层应直接作差。
    const std::array<double,12> coordinates{{-.2,1,1, .5,1,1, 1.1,1.8,1.4, 8,1,1}};
    const std::array<int,1> centers{{1}};
    const std::array<int,2> offsets{{0,4}};
    const std::array<int,4> candidates{{0,2,3,1}};
    GpuBuffer<int> device_centers(1), device_offsets(2), device_candidates(4);
    device_centers.upload_from(centers.data(),1); device_offsets.upload_from(offsets.data(),2);
    device_candidates.upload_from(candidates.data(),4);
    // SoA 使用预留 7 个原子槽位，验证 stride 独立于有效原子数 4。
    std::vector<double> storage(soa ? 21 : 12, -999);
    for (int atom = 0; atom < 4; ++atom)
        for (int axis = 0; axis < 3; ++axis) storage[soa ? atom+7*axis : 3*atom+axis] = coordinates[3*atom+axis];
    GpuBuffer<double> device_positions(storage.size()); device_positions.upload_from(storage.data(),storage.size());
    ArrayView<const double> positions{device_positions.ptr,4,3,soa?size_t(1):size_t(3),
                                      soa?size_t(7):size_t(1),device_positions.capacity,MemorySpace::Cuda};
    NeighborListView list{{centers.data(),1,1,1,1,1,MemorySpace::Host}, view(device_centers,1),
                          view(device_offsets,2), view(device_candidates,4),2,4,1,1,
                          NeighborCoordinates::PeriodicImages};
    StruFactorData data;
    strufactor_prepare_geometry(data,list,positions,2.5);
    int count = 0; data.d_neigh_in_cutoff_r.download_to(&count,1);
    require(count == 2, "geometry filtering or local indexing is incorrect");
    std::array<int,2> actual_neighbors{}; data.d_calculated_numneigh.download_to(actual_neighbors.data(),2);
    require(actual_neighbors[0] == 0 && actual_neighbors[1] == 2, "ghost or out-of-group neighbor dropped");
    std::array<double,8> geometry{}; data.d_group_dminneigh.download_to(geometry.data(),8);
    require(std::abs(geometry[0]+.7) < 1e-14 && std::abs(geometry[4]-.6) < 1e-14,
            "periodic-image displacement was changed");
    GpuBuffer<double> values(4), adjoint(1), force(12), virial(24);
    const double seed = 1.3; adjoint.upload_from(&seed,1);
    strufactor_forward(data,1.7,6.25,values.ptr);
    strufactor_backward(data,1.7,6.25,adjoint.ptr,force.ptr,virial.ptr);
    Result result{std::vector<double>(4),std::vector<double>(12),std::vector<double>(24)};
    values.download_to(result.values.data(),4); force.download_to(result.force.data(),12);
    virial.download_to(result.virial.data(),24);
    require(result.values[1] != 0 && result.values[0] == 0 && result.values[2] == 0, "value must use center localid");
    require(std::abs(result.force[0]) > 1e-6 && std::abs(result.force[6]) > 1e-6, "missing neighbor force");
    for (int axis = 0; axis < 3; ++axis) {
        double total = 0; for (int atom = 0; atom < 4; ++atom) total += result.force[3*atom+axis];
        require(std::abs(total)<1e-13, "force not conserved");
    }
    // 未声明镜像约定的 backend 必须显式处理周期性后才能进入此几何入口。
    list.coordinates = NeighborCoordinates::MinimumImageRequired;
    bool rejected = false;
    try { strufactor_prepare_geometry(data,list,positions,2.5); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "minimum-image requirement silently ignored");
    return result;
}
}
int main() {
    try {
        check_device_virials();
        const auto aos = evaluate(false), soa = evaluate(true);
        const auto compare = [](const auto& first, const auto& second) {
            for (size_t index = 0; index < first.size(); ++index)
                require(std::abs(first[index]-second[index])<1e-13, "AoS/SoA results differ");
        };
        compare(aos.values,soa.values); compare(aos.force,soa.force); compare(aos.virial,soa.virial);
        std::cout << "PASS: CUDA six/nine-component virials and shared AoS/SoA geometry with ghost contributions\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
