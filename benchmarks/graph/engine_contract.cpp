#include "zqc_distance.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <limits>

using namespace MetaD_zqc;
namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class Function> void rejects(Function action) {
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "invalid input was accepted");
}
bool close(double first, double second) { return std::abs(first-second) < 2e-8; }

// 独立 Host 宿主：不包含 LAMMPS/MPI/CUDA 头文件，使用稳定 ID 与可重排的本地数组。
struct TestEngine : EngineAdapter {
    std::uint64_t evaluation = 0;
    int registrations = 0, reductions = 0;
    std::array<AtomId, 2> ids{{(AtomId(1)<<40)+1, (AtomId(1)<<40)+2}};
    std::array<double, 6> coordinates{{0.2, 0.5, 0.7, 9.4, 1.1, 1.2}};
    std::array<double, 6> forces{};
    std::array<double, 9> virial{};
    double box = 10;
    void begin_evaluation() override { ++evaluation; forces.fill(0); virial.fill(0); }
    std::uint64_t evaluation_id() const override { return evaluation; }
    int owned_atom_count() const override { return 2; }
    int total_atom_count() const override { return 2; }
    ArrayView<const double> positions(MemorySpace memory) override {
        require(memory == MemorySpace::Host, "TestEngine only supports host arrays");
        return {coordinates.data(), 2, 3, 3, 1, 6, memory};
    }
    ArrayView<const int> atom_types(MemorySpace) override { return {}; }
    int register_neighbors(const NeighborRequest&) override { return registrations++; }
    NeighborListView prepare_neighbors(int) override { return {}; }
    void read_owned_atoms(const AtomId* requested, size_t count, double* values, int* indices) override {
        std::fill_n(values, 4*count, 0); std::fill_n(indices, count, -1);
        for (size_t endpoint = 0; endpoint < count; ++endpoint)
            for (size_t local = 0; local < ids.size(); ++local) if (ids[local] == requested[endpoint]) {
                indices[endpoint] = local;
                for (int axis = 0; axis < 3; ++axis) values[4*endpoint+axis] += coordinates[3*local+axis];
                values[4*endpoint+3] += 1;
            }
    }
    void sum(double*, size_t count) override { require(count == 8, "distance must use one packed reduction"); ++reductions; }
    void minimum_image(double* delta) const override {
        for (int axis = 0; axis < 3; ++axis) delta[axis] -= box*std::round(delta[axis]/box);
    }
    void commit_force_virial(ArrayView<const double> input_forces, VirialView input_virials) override {
        input_forces.validate(); input_virials.validate();
        require(input_forces.memory == MemorySpace::Host && input_virials.values.memory == MemorySpace::Host,
                "TestEngine requires host output");
        require(input_forces.item_count == 2 && input_forces.component_count == 3 &&
                input_virials.values.item_count == 2, "TestEngine output shape mismatch");
        for (size_t atom = 0; atom < 2; ++atom) {
            for (int axis = 0; axis < 3; ++axis) forces[3*atom+axis] += input_forces(atom,axis);
            for (int component = 0; component < 9; ++component)
                virial[component] += input_virials.full_component(atom,component);
        }
    }
    void add_owned_force_virial(int local, const double* force, VirialView contribution) override {
        contribution.validate();
        require(force && local >= 0 && local < 2 && contribution.values.item_count == 1 &&
                contribution.values.memory == MemorySpace::Host, "TestEngine requires one owned Host row");
        for (int axis = 0; axis < 3; ++axis) forces[3*local+axis] += force[axis];
        for (int component = 0; component < 9; ++component)
            virial[component] += contribution.full_component(0,component);
    }
};

void check_views() {
    double storage[24] = {};
    ArrayView<double> aos{storage, 2, 3, 3, 1, 6, MemorySpace::Host};
    ArrayView<double> soa{storage+6, 2, 3, 1, 6, 18, MemorySpace::Host};
    aos.validate(); soa.validate();
    for (int item = 0; item < 2; ++item)
        for (int component = 0; component < 3; ++component) {
            aos(item, component) = 10*item+component;
            soa(item, component) = aos(item, component);
            require(soa(item, component) == aos(item, component), "AoS/SoA indexing mismatch");
        }
    require(storage[6+1+12] == 12, "SoA must use allocation stride, not item count");
    auto invalid = soa; invalid.storage_size = 13; rejects([&] { invalid.validate(); });
    invalid = soa; invalid.item_count = 3; invalid.item_stride = std::numeric_limits<size_t>::max();
    rejects([&] { invalid.validate(); });
    invalid = soa; invalid.component_stride = 0; rejects([&] { invalid.validate(); });
    ArrayView<double>{}.validate();
}

void check_virials() {
    // r=(1,2,3), F=(4,5,6) 的 r tensor F，三个转置项与上三角各不相等。
    const std::array<double,9> full{{4,10,18,5,6,12,8,12,15}};
    const std::array<double,9> symmetric{{4,10,18,5,6,12,5,6,12}};
    const std::array<double,6> input_forces{{1,2,3,-1,-2,-3}};
    ArrayView<const double> force_view{input_forces.data(),2,3,3,1,6,MemorySpace::Host};
    TestEngine engine;
    for (bool nine : {false,true}) for (bool soa : {false,true}) {
        const size_t components = nine ? 9 : 6, slots = 5;
        std::vector<double> storage(components*slots, -999);
        for (size_t atom = 0; atom < 2; ++atom)
            for (size_t component = 0; component < components; ++component)
                storage[soa ? atom+slots*component : atom*components+component] = (atom+1)*full[component];
        const auto initial = storage;
        VirialView view{{storage.data(),2,components,soa?1:components,soa?slots:1,
                         storage.size(),MemorySpace::Host},
                        nine?VirialRepresentation::Full9:VirialRepresentation::Symmetric6};
        view.validate();
        engine.begin_evaluation();
        engine.commit_force_virial(force_view,view);
        const auto& expected = nine ? full : symmetric;
        for (size_t component = 0; component < 9; ++component)
            require(engine.virial[component] == 3*expected[component], "virial representation lost a component");
        require(engine.forces == input_forces && storage == initial, "output changed force or input storage");
        // owned 接口借用第二行；保留 SoA 分量跨度，确认单行九分量不退化为六分量。
        auto single = view;
        single.values.data += single.values.item_stride;
        single.values.storage_size -= single.values.item_stride;
        single.values.item_count = 1;
        engine.begin_evaluation();
        engine.add_owned_force_virial(1,input_forces.data()+3,single);
        for (size_t component = 0; component < 9; ++component)
            require(engine.virial[component] == 2*expected[component], "owned virial lost a component");
        auto invalid = view;
        invalid.representation = nine?VirialRepresentation::Symmetric6:VirialRepresentation::Full9;
        rejects([&] { invalid.validate(); });
        invalid = view; invalid.representation = static_cast<VirialRepresentation>(99);
        rejects([&] { invalid.validate(); });
        invalid = view; invalid.values.storage_size = 1;
        rejects([&] { invalid.validate(); });
        invalid = view; invalid.values.item_count = 1;
        rejects([&] { engine.commit_force_virial(force_view,invalid); });
        rejects([&] { engine.add_owned_force_virial(0,input_forces.data(),view); });
        invalid = single; invalid.values.item_count = 0;
        rejects([&] { engine.add_owned_force_virial(0,input_forces.data(),invalid); });
        invalid = single; invalid.values.memory = MemorySpace::Cuda;
        rejects([&] { engine.add_owned_force_virial(0,input_forces.data(),invalid); });
        // 空 rank 的批量视图仍需携带有效的表示和列数。
        VirialView empty{{nullptr,0,components,components,1,0,MemorySpace::Host},view.representation};
        empty.validate();
    }
}

// 三体能量 U=(r1-r0) dot (r2-r0) 具有旋转不变性。
// 按质心分配 rho_i tensor F_i，各原子可不对称，完整相互作用的总量应对称。
void check_manybody_virial() {
    const double positions[3][3]={{0,0,0},{1,0,0},{0,1,0}};
    const double forces[3][3]={{1,1,0},{0,-1,0},{-1,0,0}};
    const int rows[9]={0,1,2,0,0,1,1,2,2},columns[9]={0,1,2,1,2,2,0,0,1};
    double per_atom[27]={};
    std::array<double,9> total{};
    for(int atom=0;atom<3;++atom) for(int component=0;component<9;++component) {
        const int axis=rows[component];
        const double centroid=(positions[0][axis]+positions[1][axis]+positions[2][axis])/3;
        per_atom[9*atom+component]=(positions[atom][axis]-centroid)*forces[atom][columns[component]];
        total[component]+=per_atom[9*atom+component];
    }
    VirialView view{make_aos_view<const double>(per_atom,3,9,27,MemorySpace::Host),VirialRepresentation::Full9};
    view.validate();
    require(std::abs(view.full_component(1,3)-view.full_component(1,6))>.1,"fixture must expose asymmetric atomic virial");
    for(int component=3;component<6;++component)
        require(close(total[component],total[component+3]),"many-body total virial must be symmetric");
    require(close(total[3],-1),"many-body virial fixture sign");
}

void check_requests() {
    TestEngine engine;
    NeighborRequest first; first.cutoff = 3; first.centers = 2;
    const int first_handle = engine.neighbors().request(first);
    require(engine.neighbors().request(first) == first_handle && engine.registrations == 1, "request not shared");
    auto second = first; second.neighbors = 4;
    require(engine.neighbors().request(second) != first_handle, "neighbor selection lost");
    second = first; second.cutoff = 4; engine.neighbors().request(second);
    second = first; second.kind = NeighborKind::Half; engine.neighbors().request(second);
    second = first; second.ghost_centers = true; engine.neighbors().request(second);
    require(engine.registrations == 5, "request semantics omitted from identity");
    second.kind = NeighborKind::Half;
    rejects([&] { engine.neighbors().request(second); }); // ghost 环境必须请求 Full。
    second.cutoff = std::numeric_limits<double>::quiet_NaN();
    rejects([&] { engine.neighbors().request(second); });
    rejects([&] { engine.neighbors().prepare(-1); });
}

void check_distance() {
    TestEngine engine;
    DistanceCalculation distance(engine.ids[0], engine.ids[1]);
    rejects([&] { distance.forward(engine); });
    engine.begin_evaluation();
    const double length = distance.forward(engine);
    require(close(length, std::sqrt(.64+.36+.25)), "minimum-image value mismatch");
    require(engine.reductions == 1, "DISTANCE used more than one collective");
    const double seed = 1.7;
    distance.backward(engine, seed);
    // 两个端点各记一半 virial，合计等于 -seed/r * dr tensor dr。
    const double delta[3] = {-.8,.6,.5};
    const int rows[9] = {0,1,2,0,0,1,1,2,2};
    const int columns[9] = {0,1,2,1,2,2,0,0,1};
    for (int component = 0; component < 9; ++component)
        require(close(engine.virial[component], -seed/length*delta[rows[component]]*delta[columns[component]]),
                "distance virial sign, half-pair allocation or transpose expansion is incorrect");
    const auto analytic = engine.forces;
    const auto initial = engine.coordinates;
    const double step = 1e-6;
    for (size_t coordinate = 0; coordinate < initial.size(); ++coordinate) {
        engine.coordinates = initial; engine.coordinates[coordinate] += step;
        engine.begin_evaluation(); const double plus = distance.forward(engine);
        engine.coordinates = initial; engine.coordinates[coordinate] -= step;
        engine.begin_evaluation(); const double minus = distance.forward(engine);
        require(close(analytic[coordinate], -seed*(plus-minus)/(2*step)), "distance force finite difference failed");
    }
    engine.coordinates = initial;
    engine.begin_evaluation();
    rejects([&] { distance.backward(engine, seed); });
    distance.forward(engine); distance.backward(engine, seed);
    for (int axis = 0; axis < 3; ++axis) require(close(engine.forces[axis]+engine.forces[3+axis], 0), "net force");
    const auto initial_force = engine.forces;
    const auto initial_virial = engine.virial;
    std::swap(engine.ids[0], engine.ids[1]);
    for (int axis = 0; axis < 3; ++axis) std::swap(engine.coordinates[axis], engine.coordinates[3+axis]);
    engine.begin_evaluation();
    require(close(distance.forward(engine), length), "local sorting changed distance");
    distance.backward(engine, seed);
    for (int axis = 0; axis < 3; ++axis) require(close(engine.forces[axis], initial_force[3+axis]), "stale owner index");
    for (int component = 0; component < 9; ++component) require(close(engine.virial[component], initial_virial[component]), "virial changed after sorting");
    engine.ids[1] = 123;
    engine.begin_evaluation(); rejects([&] { distance.forward(engine); });
    rejects([&] { distance.backward(engine, seed); });
}
}
int main() {
    try {
        check_views(); check_virials(); check_manybody_virial(); check_requests(); check_distance();
        std::cout << "PASS: array/virial representations, request identity, distance force/virial and evaluation lifetime\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
