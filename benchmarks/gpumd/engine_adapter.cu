// GPUMD 数组/力适配器契约测试。
// 伪造 GPUMD 风格 CUDA SoA：分量跨度为 allocated_slots，有效原子数可小于槽位数。
// 同步意图：MD 非默认流生产坐标 → bind_frame 记录就绪事件 → 适配器私有流等待后
// 紧凑 gather / 累加 → finish 令 MD 流等待回写完成。禁止依赖 cudaDeviceSynchronize。
#include "zqc_gpumd_engine.cuh"
#include "zqc_distance.h"
#include "zqc_cuda_execution.cuh"
#include "zqc_gpu_buffer.cuh"
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

using namespace MetaD_zqc;

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
bool nearly_equal(double first, double second, double tolerance = 1e-10) {
    return std::abs(first - second) <= tolerance;
}
template<class Action>
void expect_throws(Action action, const char* message) {
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    require(rejected, message);
}

// 模拟 GPUMD 设备端存储：positions/forces 为 xyz SoA，virials 为 Full9 SoA，
// potential 为标量 SoA。allocated_atom_slots 故意大于有效原子数以检验 stride。
struct FakeGpumdDeviceArrays {
    static constexpr size_t allocated_atom_slots = 7;
    size_t active_atom_count = 0;
    GpuBuffer<double> device_positions;
    GpuBuffer<double> device_forces;
    GpuBuffer<double> device_virials_full9;
    GpuBuffer<double> device_potential;
    CudaExecutionStream molecular_dynamics_stream;

    explicit FakeGpumdDeviceArrays(size_t atom_count)
        : active_atom_count(atom_count),
          device_positions(3 * allocated_atom_slots),
          device_forces(3 * allocated_atom_slots),
          device_virials_full9(9 * allocated_atom_slots),
          device_potential(allocated_atom_slots) {
        require(atom_count <= allocated_atom_slots, "active atoms exceed allocated SoA slots");
        device_positions.set_name("fake_gpumd_positions");
        device_forces.set_name("fake_gpumd_forces");
        device_virials_full9.set_name("fake_gpumd_virials");
        device_potential.set_name("fake_gpumd_potential");
    }

    ArrayView<double> positions_view() {
        return make_soa_view(device_positions.ptr, active_atom_count, 3,
                             allocated_atom_slots, device_positions.capacity, MemorySpace::Cuda);
    }
    ArrayView<double> forces_view() {
        return make_soa_view(device_forces.ptr, active_atom_count, 3,
                             allocated_atom_slots, device_forces.capacity, MemorySpace::Cuda);
    }
    ArrayView<double> virials_view() {
        return make_soa_view(device_virials_full9.ptr, active_atom_count, 9,
                             allocated_atom_slots, device_virials_full9.capacity, MemorySpace::Cuda);
    }
    ArrayView<double> potential_view() {
        return make_soa_view(device_potential.ptr, active_atom_count, 1,
                             allocated_atom_slots, device_potential.capacity, MemorySpace::Cuda);
    }

    GpumdFrame make_frame(std::array<double, 3> box_lengths, std::array<int, 3> periodic,
                          std::int64_t step) {
        GpumdFrame frame;
        frame.positions = positions_view();
        frame.forces = forces_view();
        frame.virials = virials_view();
        frame.potential = potential_view();
        frame.box_lengths = box_lengths;
        frame.periodic = periodic;
        frame.step = step;
        frame.stream = molecular_dynamics_stream.get();
        return frame;
    }

    void upload_soa_positions(const std::vector<double>& host_aos_xyz) {
        require(host_aos_xyz.size() == 3 * active_atom_count, "position AoS size mismatch");
        std::vector<double> host_soa(3 * allocated_atom_slots, -999.0);
        for (size_t atom_index = 0; atom_index < active_atom_count; ++atom_index)
            for (int axis = 0; axis < 3; ++axis)
                host_soa[atom_index + allocated_atom_slots * axis] =
                    host_aos_xyz[3 * atom_index + axis];
        device_positions.upload_from(host_soa.data(), host_soa.size(),
                                     molecular_dynamics_stream.get());
    }

    void upload_baseline_forces_virials_potential(const std::vector<double>& forces_aos,
                                                  const std::vector<double>& virials_full9_aos,
                                                  const std::vector<double>& potential_aos) {
        require(forces_aos.size() == 3 * active_atom_count, "force AoS size mismatch");
        require(virials_full9_aos.size() == 9 * active_atom_count, "virial AoS size mismatch");
        require(potential_aos.size() == active_atom_count, "potential size mismatch");
        std::vector<double> force_soa(3 * allocated_atom_slots, 0.0);
        std::vector<double> virial_soa(9 * allocated_atom_slots, 0.0);
        std::vector<double> potential_soa(allocated_atom_slots, 0.0);
        for (size_t atom_index = 0; atom_index < active_atom_count; ++atom_index) {
            for (int axis = 0; axis < 3; ++axis)
                force_soa[atom_index + allocated_atom_slots * axis] = forces_aos[3 * atom_index + axis];
            for (int component = 0; component < 9; ++component)
                virial_soa[atom_index + allocated_atom_slots * component] =
                    virials_full9_aos[9 * atom_index + component];
            potential_soa[atom_index] = potential_aos[atom_index];
        }
        device_forces.upload_from(force_soa.data(), force_soa.size(), molecular_dynamics_stream.get());
        device_virials_full9.upload_from(virial_soa.data(), virial_soa.size(),
                                         molecular_dynamics_stream.get());
        device_potential.upload_from(potential_soa.data(), potential_soa.size(),
                                     molecular_dynamics_stream.get());
    }

    std::vector<double> download_forces_aos() {
        std::vector<double> force_soa(3 * allocated_atom_slots);
        device_forces.download_to(force_soa.data(), force_soa.size(), molecular_dynamics_stream.get());
        std::vector<double> forces_aos(3 * active_atom_count);
        for (size_t atom_index = 0; atom_index < active_atom_count; ++atom_index)
            for (int axis = 0; axis < 3; ++axis)
                forces_aos[3 * atom_index + axis] =
                    force_soa[atom_index + allocated_atom_slots * axis];
        return forces_aos;
    }
    std::vector<double> download_virials_full9_aos() {
        std::vector<double> virial_soa(9 * allocated_atom_slots);
        device_virials_full9.download_to(virial_soa.data(), virial_soa.size(),
                                         molecular_dynamics_stream.get());
        std::vector<double> virials_aos(9 * active_atom_count);
        for (size_t atom_index = 0; atom_index < active_atom_count; ++atom_index)
            for (int component = 0; component < 9; ++component)
                virials_aos[9 * atom_index + component] =
                    virial_soa[atom_index + allocated_atom_slots * component];
        return virials_aos;
    }
    std::vector<double> download_potential_aos() {
        std::vector<double> potential_soa(allocated_atom_slots);
        device_potential.download_to(potential_soa.data(), potential_soa.size(),
                                     molecular_dynamics_stream.get());
        return {potential_soa.begin(), potential_soa.begin() + static_cast<std::ptrdiff_t>(active_atom_count)};
    }
};

// 仅阻塞 MD 生产流的 Host gate；用于验证消费依赖未退化为全设备同步。
struct MolecularDynamicsProducerGate {
    cudaStream_t producer_stream;
    std::mutex mutex;
    std::condition_variable condition;
    bool released = false;
    std::atomic<bool> timed_out{false};
    std::thread watchdog;
    explicit MolecularDynamicsProducerGate(cudaStream_t stream)
        : producer_stream(stream), watchdog([this] {
              std::unique_lock<std::mutex> lock(mutex);
              if (!condition.wait_for(lock, std::chrono::seconds(5), [this] { return released; })) {
                  timed_out = true;
                  released = true;
                  condition.notify_all();
              }
          }) {}
    void release() {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        condition.notify_all();
    }
    ~MolecularDynamicsProducerGate() {
        release();
        cudaStreamSynchronize(producer_stream);
        watchdog.join();
    }
    static void CUDART_CB wait_callback(void* context) {
        auto& gate = *static_cast<MolecularDynamicsProducerGate*>(context);
        std::unique_lock<std::mutex> lock(gate.mutex);
        gate.condition.wait(lock, [&] { return gate.released; });
    }
};

__global__ void write_soa_position_component(double* soa_base, size_t atom_index,
                                             size_t component_stride, int axis, double value) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        soa_base[atom_index + component_stride * static_cast<size_t>(axis)] = value;
}

void verify_distance_additive_periodic_and_energy() {
    FakeGpumdDeviceArrays arrays(3);
    // 原子 1→(0.2,0,0)，原子 2→(9.4,0,0)，盒长 10 且 x 周期，最小镜像距离应为 0.8。
    arrays.upload_soa_positions({0.2, 0.0, 0.0, 9.4, 0.0, 0.0, 1.0, 2.0, 3.0});
    const std::vector<double> baseline_forces{10, 20, 30, 40, 50, 60, 70, 80, 90};
    const std::vector<double> baseline_virials(27, 1.5);
    const std::vector<double> baseline_potential{100, 200, 300};
    arrays.upload_baseline_forces_virials_potential(baseline_forces, baseline_virials,
                                                    baseline_potential);

    GpumdEngineAdapter engine;
    engine.bind_frame(arrays.make_frame({{10, 10, 10}}, {{1, 1, 1}}, 42));
    engine.begin_evaluation();
    require(engine.timestep() == 42, "timestep should mirror bound frame step");
    require(engine.atom_ids().item_count == 3 && engine.atom_ids()(0) == 1 &&
                engine.atom_ids()(2) == 3,
            "stable Host atom IDs must be 1..N");

    DistanceCalculation distance(1, 2);
    const double measured_separation = distance.forward(engine);
    require(nearly_equal(measured_separation, 0.8), "periodic minimum-image distance is wrong");
    // seed=1.6 → scale=seed/r=2；MIC 位移 delta≈(-0.8,0,0)。
    // 原子1力 = +scale*delta，原子2力 = -scale*delta。
    distance.backward(engine, 1.6);
    const double bias_energy = 6.0;
    engine.finish(bias_energy);

    const auto forces = arrays.download_forces_aos();
    const auto virials = arrays.download_virials_full9_aos();
    const auto potential = arrays.download_potential_aos();
    require(nearly_equal(forces[0], baseline_forces[0] - 1.6) &&
                nearly_equal(forces[3], baseline_forces[3] + 1.6),
            "DISTANCE forces must add onto MD baseline");
    require(nearly_equal(forces[1], baseline_forces[1]) &&
                nearly_equal(forces[6], baseline_forces[6]),
            "untouched force components must remain baseline");
    // Symmetric6 → Full9：xy/xz/yz 写入后，yx/zx/zy 槽位应镜像同一数值。
    const double delta_xx = virials[0] - baseline_virials[0];
    const double delta_xy = virials[3] - baseline_virials[3];
    const double delta_yx = virials[6] - baseline_virials[6];
    require(nearly_equal(delta_xx, -0.64), "owned Symmetric6 xx contribution mismatch");
    require(nearly_equal(delta_xy, delta_yx),
            "Symmetric6 owned path must mirror xy into Full9 yx");
    for (size_t atom_index = 0; atom_index < 3; ++atom_index)
        require(nearly_equal(potential[atom_index],
                             baseline_potential[atom_index] + bias_energy / 3.0),
                "bias energy must be split evenly onto per-atom potential");
}

void verify_full9_commit_preserves_asymmetry() {
    FakeGpumdDeviceArrays arrays(2);
    arrays.upload_soa_positions({0, 0, 0, 1, 0, 0});
    arrays.upload_baseline_forces_virials_potential(
        {0, 0, 0, 0, 0, 0}, std::vector<double>(18, 0.0), {0, 0});

    // 非对称 r⊗F 样例：xy≠yx，xz≠zx，yz≠zy。
    const std::array<double, 6> host_forces{{1, 2, 3, -1, -2, -3}};
    const std::array<double, 18> host_virials_full9{{
        4, 10, 18, 5, 6, 12, 8, 12, 15,
        4, 10, 18, 5, 6, 12, 8, 12, 15}};
    ArrayView<const double> force_view{host_forces.data(), 2, 3, 3, 1, 6, MemorySpace::Host};
    VirialView virial_view{{host_virials_full9.data(), 2, 9, 9, 1, 18, MemorySpace::Host},
                           VirialRepresentation::Full9};

    GpumdEngineAdapter engine;
    engine.bind_frame(arrays.make_frame({{5, 5, 5}}, {{0, 0, 0}}, 1));
    engine.begin_evaluation();
    engine.commit_force_virial(force_view, virial_view);
    engine.finish(0.0);

    const auto forces = arrays.download_forces_aos();
    const auto virials = arrays.download_virials_full9_aos();
    for (int index = 0; index < 6; ++index)
        require(nearly_equal(forces[index], host_forces[index]), "Host commit force lost");
    for (int component = 0; component < 9; ++component) {
        require(nearly_equal(virials[component], host_virials_full9[component]),
                "Full9 commit must not symmetrize atom 0");
        require(virials[6] == 8 && virials[3] == 5, "asymmetric yx must stay distinct from xy");
    }
}

void verify_cuda_commit_with_stride_and_second_cv_sum() {
    FakeGpumdDeviceArrays arrays(2);
    arrays.upload_soa_positions({0, 0, 0, 1, 1, 1});
    arrays.upload_baseline_forces_virials_potential(
        {1, 1, 1, 1, 1, 1}, std::vector<double>(18, 2.0), {0, 0});

    const size_t slots = 5;
    std::vector<double> force_soa(3 * slots, 0.0), virial_soa(9 * slots, 0.0);
    for (size_t atom_index = 0; atom_index < 2; ++atom_index) {
        for (int axis = 0; axis < 3; ++axis)
            force_soa[atom_index + slots * axis] = 0.5;
        for (int component = 0; component < 9; ++component)
            virial_soa[atom_index + slots * component] = 0.25 * (component + 1);
    }
    GpuBuffer<double> device_force_pack(force_soa.size()), device_virial_pack(virial_soa.size());
    device_force_pack.upload_from(force_soa.data(), force_soa.size());
    device_virial_pack.upload_from(virial_soa.data(), virial_soa.size());
    ArrayView<const double> force_view =
        make_soa_view(device_force_pack.ptr, 2, 3, slots, device_force_pack.capacity, MemorySpace::Cuda);
    VirialView virial_view{make_soa_view(device_virial_pack.ptr, 2, 9, slots,
                                         device_virial_pack.capacity, MemorySpace::Cuda),
                           VirialRepresentation::Full9};

    GpumdEngineAdapter engine;
    engine.bind_frame(arrays.make_frame({{8, 8, 8}}, {{1, 0, 1}}, 7));
    engine.begin_evaluation();
    engine.commit_force_virial(force_view, virial_view);
    engine.commit_force_virial(force_view, virial_view); // 同一原子多 CV：必须安全累加。
    engine.finish(0.0);

    const auto forces = arrays.download_forces_aos();
    const auto virials = arrays.download_virials_full9_aos();
    for (int index = 0; index < 6; ++index)
        require(nearly_equal(forces[index], 1.0 + 1.0), "repeated CUDA commit must sum");
    require(nearly_equal(virials[3], 2.0 + 2 * 0.25 * 4) &&
                nearly_equal(virials[6], 2.0 + 2 * 0.25 * 7),
            "strided Full9 CUDA commit lost a component");
}

__global__ void copy_one_soa_component(const double* soa_base, size_t atom_index,
                                       size_t component_stride, int axis,
                                       double* destination) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *destination = soa_base[atom_index + component_stride * static_cast<size_t>(axis)];
}

void verify_stream_input_output_ordering() {
    FakeGpumdDeviceArrays arrays(2);
    // 预热：先写入旧坐标，再在 gate 之后覆盖为新坐标。
    arrays.upload_soa_positions({0.0, 0.0, 0.0, 1.0, 0.0, 0.0});
    arrays.molecular_dynamics_stream.synchronize();

    // 门控开始前完成 CUDA 资源分配，避免分配器自身同步干扰流依赖测试。
    GpumdEngineAdapter engine;
    CudaExecutionStream consumer_stream;
    CudaExecutionStream independent_stream;
    GpuBuffer<double> gathered_component(1), independent_scratch(1);
    // 预先加载本测试 kernel 并创建缓冲事件；CUDA 首次模块加载可能等待其他流。
    write_soa_position_component<<<1, 1, 0, arrays.molecular_dynamics_stream.get()>>>(
        arrays.device_positions.ptr, 1, FakeGpumdDeviceArrays::allocated_atom_slots, 0, 1.0);
    copy_one_soa_component<<<1, 1, 0, consumer_stream.get()>>>(
        arrays.device_positions.ptr, 1, FakeGpumdDeviceArrays::allocated_atom_slots, 0, gathered_component.ptr);
    consumer_stream.ready().wait_host();
    independent_scratch.clear_async(independent_stream.get());
    independent_stream.synchronize();
    arrays.molecular_dynamics_stream.synchronize();
    MolecularDynamicsProducerGate gate(arrays.molecular_dynamics_stream.get());
    execution_check(cudaLaunchHostFunc(arrays.molecular_dynamics_stream.get(),
                                       MolecularDynamicsProducerGate::wait_callback, &gate));
    // 门控之后才把原子 2 移到 2.0，使错误的抢跑读到旧坐标 1.0。
    write_soa_position_component<<<1, 1, 0, arrays.molecular_dynamics_stream.get()>>>(
        arrays.device_positions.ptr, 1, FakeGpumdDeviceArrays::allocated_atom_slots, 0, 2.0);
    execution_check(cudaGetLastError());

    engine.bind_frame(arrays.make_frame({{20, 20, 20}}, {{0, 0, 0}}, 3));
    engine.begin_evaluation();

    // 输入依赖：positions(Cuda) 携带 MD 就绪事件；消费流只排队等待，Host 不 bulk sync。
    auto device_positions = engine.positions(MemorySpace::Cuda);
    device_positions.wait_on(consumer_stream.get());
    copy_one_soa_component<<<1, 1, 0, consumer_stream.get()>>>(
        device_positions.data, 1, device_positions.component_stride, 0, gathered_component.ptr);
    execution_check(cudaGetLastError());
    const auto consumer_completion = consumer_stream.ready();
    independent_scratch.clear_async(independent_stream.get());
    double independent_host_value = -1.0;
    independent_scratch.download_to(&independent_host_value, 1, independent_stream.get());
    require(independent_host_value == 0.0, "independent stream failed while MD producer was gated");
    const bool independent_finished_while_md_gated = !gate.timed_out;
    const auto consumer_still_pending = cudaStreamQuery(consumer_stream.get());
    gate.release();
    consumer_completion.wait_host();
    double host_gathered = 0;
    gathered_component.download_to(&host_gathered, 1, consumer_stream.get());
    require(independent_finished_while_md_gated,
            "adapter path required a device-wide wait that stalled unrelated streams");
    require(consumer_still_pending == cudaErrorNotReady,
            "Cuda positions readiness failed to hold consumer behind MD producer");
    require(nearly_equal(host_gathered, 2.0), "consumer read stale SoA positions across streams");

    // DISTANCE Host gather：后台释放 gate，避免与 read_owned_atoms 的流同步死锁。
    MolecularDynamicsProducerGate second_gate(arrays.molecular_dynamics_stream.get());
    execution_check(cudaLaunchHostFunc(arrays.molecular_dynamics_stream.get(),
                                       MolecularDynamicsProducerGate::wait_callback, &second_gate));
    write_soa_position_component<<<1, 1, 0, arrays.molecular_dynamics_stream.get()>>>(
        arrays.device_positions.ptr, 1, FakeGpumdDeviceArrays::allocated_atom_slots, 0, 4.0);
    execution_check(cudaGetLastError());
    engine.bind_frame(arrays.make_frame({{20, 20, 20}}, {{0, 0, 0}}, 4));
    engine.begin_evaluation();
    std::thread release_second_gate_after_delay([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        second_gate.release();
    });
    DistanceCalculation distance(1, 2);
    const double separation = distance.forward(engine);
    release_second_gate_after_delay.join();
    require(!second_gate.timed_out, "DISTANCE gather timed out waiting for MD stream");
    require(nearly_equal(separation, 4.0), "Host gather ignored MD stream completion event");

    distance.backward(engine, 4.0); // scale=1 → F1=+delta=(4,0,0), F2=-delta
    engine.finish(0.0);
    // 输出排序：finish 让 MD 流等待适配器回写；随后在 MD 流上下载应看到累加力。
    const auto forces = arrays.download_forces_aos();
    require(nearly_equal(forces[0], 4.0) && nearly_equal(forces[3], -4.0),
            "MD consumer stream observed forces before adapter writeback");
}

void verify_rebind_changed_frame_addresses() {
    FakeGpumdDeviceArrays first_arrays(2), second_arrays(2);
    first_arrays.upload_soa_positions({0, 0, 0, 1, 0, 0});
    second_arrays.upload_soa_positions({0, 0, 0, 3, 0, 0});
    first_arrays.upload_baseline_forces_virials_potential(
        {0, 0, 0, 0, 0, 0}, std::vector<double>(18, 0.0), {0, 0});
    second_arrays.upload_baseline_forces_virials_potential(
        {0, 0, 0, 0, 0, 0}, std::vector<double>(18, 0.0), {0, 0});

    GpumdEngineAdapter engine;
    engine.bind_frame(first_arrays.make_frame({{10, 10, 10}}, {{0, 0, 0}}, 1));
    engine.begin_evaluation();
    DistanceCalculation distance(1, 2);
    require(nearly_equal(distance.forward(engine), 1.0), "first frame distance");
    distance.backward(engine, 1.0);
    engine.finish(0.0);

    // 更换底层设备指针后，必须读取新帧坐标。
    engine.bind_frame(second_arrays.make_frame({{10, 10, 10}}, {{0, 0, 0}}, 2));
    engine.begin_evaluation();
    require(nearly_equal(distance.forward(engine), 3.0), "rebound frame address ignored");
    distance.backward(engine, 3.0); // scale=1 → F=(±3,0,0)
    engine.finish(0.0);
    const auto second_forces = second_arrays.download_forces_aos();
    require(nearly_equal(second_forces[0], 3.0) && nearly_equal(second_forces[3], -3.0),
            "forces written to the rebound frame buffers");
    const auto first_forces = first_arrays.download_forces_aos();
    require(nearly_equal(first_forces[0], 1.0) && nearly_equal(first_forces[3], -1.0),
            "original frame must keep its earlier writeback");
}

void verify_invalid_ids_and_stale_backward() {
    FakeGpumdDeviceArrays arrays(2);
    arrays.upload_soa_positions({0, 0, 0, 1, 0, 0});
    arrays.upload_baseline_forces_virials_potential(
        {0, 0, 0, 0, 0, 0}, std::vector<double>(18, 0.0), {0, 0});

    GpumdEngineAdapter engine;
    engine.bind_frame(arrays.make_frame({{4, 4, 4}}, {{1, 1, 1}}, 9));
    engine.begin_evaluation();

    const AtomId requested_ids[3] = {0, 99, 2};
    double coordinates_and_counts[12] = {};
    int owned_indices[3] = {};
    engine.read_owned_atoms(requested_ids, 3, coordinates_and_counts, owned_indices);
    require(owned_indices[0] < 0 && coordinates_and_counts[3] == 0, "ID 0 must be missing");
    require(owned_indices[1] < 0 && coordinates_and_counts[7] == 0, "out-of-range ID must be missing");
    require(owned_indices[2] == 1 && coordinates_and_counts[11] == 1.0,
            "valid ID must own exactly one row");

    DistanceCalculation distance(1, 2);
    distance.forward(engine);
    engine.begin_evaluation(); // 新一轮使上一轮 forward 过期
    expect_throws([&] { distance.backward(engine, 1.0); },
                  "stale backward must be rejected");

    expect_throws([&] {
        NeighborRequest request;
        request.cutoff = -1.0;
        engine.register_neighbors(request);
    }, "invalid neighbor registration must throw clearly");
    expect_throws([&] { engine.prepare_neighbors(0); },
                  "neighbor preparation must throw clearly");
}

} // namespace

int main() {
    try {
        verify_distance_additive_periodic_and_energy();
        verify_full9_commit_preserves_asymmetry();
        verify_cuda_commit_with_stride_and_second_cv_sum();
        verify_stream_input_output_ordering();
        verify_rebind_changed_frame_addresses();
        verify_invalid_ids_and_stale_backward();
        std::cout << "PASS: GPUMD array/force adapter contract "
                     "(DISTANCE, Full9, streams, rebind, invalid IDs)\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
