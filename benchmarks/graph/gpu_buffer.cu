#include "zqc_gpu_buffer.cuh"
#include <iostream>
#include <vector>

using namespace MetaD_zqc;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Function>
void rejects(Function operation, const char* expected) {
    try { operation(); }
    catch (const GpuBufferError& failure) {
        require(std::string(failure.what()).find(expected) != std::string::npos,
                "exception lost operation context");
        return;
    }
    throw std::runtime_error("invalid buffer operation was accepted");
}


void check_buffer() {
    std::vector<Diagnostic> diagnostics;
    GpuBuffer<int> values;
    values.set_name("test_values", Logger([&](const Diagnostic& record) {
        diagnostics.push_back(record);
    }, LogLevel::Debug));
    values.grow_to(4, "buffer_caller.cu", 123);
    require(diagnostics.size() == 1 && diagnostics[0].component == "test_values" &&
            diagnostics[0].line == 123, "growth did not reach the callback");
    int host[] = {1, 0, 1, 1};
    values.upload_from(host, 4);
    GpuBuffer<int> sum(1), offsets(5), indices(4);
    values.reduce_sum(sum.ptr, 4);
    int total = 0; sum.download_to(&total, 1);
    require(total == 3, "reduction failed");
    values.scan_to(offsets, 4);
    int scan[5]{}; offsets.download_to(scan, 5);
    require(std::vector<int>(scan, scan + 5) == std::vector<int>({0,1,1,2,3}), "scan failed");
    int selected = -1;
    values.flag_to(indices, 4, &selected);
    int chosen[3]{}; indices.download_to(chosen, 3);
    require(selected == 3 && std::vector<int>(chosen, chosen + 3) == std::vector<int>({0,2,3}),
            "selection failed");

    // 捕获错误后还能继续使用缓冲区，验证错误没有跳过主流程直接退出。
    rejects([&] { values.upload_from(host, values.capacity + 1, 0, "bad_upload.cu", 77); }, "bad_upload.cu:77");
    rejects([&] { values.download_to(nullptr, 1); }, "null host destination");
    rejects([&] { values.scan_to(values, 1); }, "distinct");
    rejects([&] { values.flag_to(indices, 4, nullptr); }, "null host count");
    rejects([&] { values.reduce_sum(sum.ptr, values.capacity + 1); }, "reduce_sum");
    const auto old_pointer = values.ptr;
    const auto old_capacity = values.capacity;
    rejects([&] { values.grow_to(std::numeric_limits<size_t>::max(), __FILE__, __LINE__); }, "overflow");
    require(values.ptr == old_pointer && values.capacity == old_capacity, "validation changed allocation");
    values.clear_async();
    values.download_to(host, 4);
    require(host[0] == 0 && host[2] == 0, "clear failed");
    values.grow_to(old_capacity + 1, __FILE__, __LINE__);
    values.download_to(host, 4);
    require(host[0] == 0 && host[3] == 0, "growth did not clear data");

    // 默认 Logger 为空，独立程序仍可按需分配、选择和写入 CSR 起点。
    GpuBuffer<int> empty, origin(1);
    empty.flag_to(indices, 0, &selected);
    require(selected == 0, "empty selection failed");
    empty.scan_to(origin, 0);
    origin.download_to(&total, 1);
    require(total == 0, "empty scan origin failed");
    empty.grow_to(1, __FILE__, __LINE__);
    rejects([] { GpuBuffer<double> oversized(std::numeric_limits<size_t>::max()); }, "overflow");
}

int main() {
    try {
        check_buffer();
        std::cout << "GPU_BUFFER_PASS callbacks, exceptions, growth, copy and CUB operations\n";
    } catch (const std::exception& failure) {
        std::cerr << failure.what() << '\n';
        return 1;
    }
}
