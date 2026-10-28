#pragma once
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cub/iterator/counting_input_iterator.cuh>
#include <climits>
#include <cstring>
#include <limits>
#include <stdexcept>
#include "zqc_diagnostics.h"
#include "zqc_array_view.h"
#include "zqc_cuda_execution.cuh"

namespace MetaD_zqc {

// 缓冲区失败时携带操作、名称和源码位置，交由宿主程序决定如何终止。
class GpuBufferError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// 拥有连续的设备数组，capacity 以 T 元素计数；CUB 临时区以字节计数。
// grow_to 会清空旧数据；借用 ptr 的 Leaf/视图须在扩容后刷新地址。
// 同一缓冲区的操作由调用方串行提交，CUB 操作复用一块临时区。
// 异步写入的完成事件随缓冲区持有，view() 将当前写依赖传给消费者。
// 覆写、扩容和析构前，调用方须结束旧消费者的访问；这里记录的是写入完成。
template <typename T>
struct GpuBuffer {
    char name[32] = "Unknown";
    T* ptr = nullptr;
    size_t capacity = 0;
    void* d_temp_storage = nullptr;
    size_t temp_storage_capacity = 0;
    int* d_select_count = nullptr;

    GpuBuffer() = default;
    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    // 以逻辑项数创建 AoS 视图，capacity 继续表示分配元素数；扩容后重新取得。
    ArrayView<T> view(size_t item_count, size_t component_count = 1) {
        ArrayView<T> result{ptr,item_count,component_count,component_count,1,capacity,MemorySpace::Cuda};
        result.readiness = readiness();
        result.validate();
        return result;
    }
    ArrayView<const T> view(size_t item_count, size_t component_count = 1) const {
        ArrayView<const T> result{ptr,item_count,component_count,component_count,1,capacity,MemorySpace::Cuda};
        result.readiness = readiness();
        result.validate();
        return result;
    }

    // 在访问流中等待已登记的写入；等待只排入 GPU 队列，Host 可继续提交。
    void wait_on(cudaStream_t stream) const {
        if (write_recorded_)
            check(cudaStreamWaitEvent(stream, write_completion_, 0), "wait for writer", __FILE__, __LINE__);
    }
    ViewReadiness readiness() const {
        return write_recorded_ ? cuda_event_readiness(write_completion_) : ViewReadiness{};
    }
    // 使用 ptr/ArrayView 自行提交 kernel 的生产者，在最后一次写入后调用。
    // 提交写入前先 wait_on(producer_stream)，再由新事件接续上一轮写依赖。
    // 事件按需创建并复用；每轮消费者先排入等待，再提交下一轮覆写。
    void record_write(cudaStream_t producer_stream = 0) {
        if (!write_completion_)
            check(cudaEventCreateWithFlags(&write_completion_, cudaEventDisableTiming),
                  "create write event", __FILE__, __LINE__);
        check(cudaEventRecord(write_completion_, producer_stream), "record write", __FILE__, __LINE__);
        write_recorded_ = true;
    }

    // 定长构造分配并清零 count 个元素；零长度保持空指针。
    explicit GpuBuffer(size_t count) {
        validate_allocation(count, __FILE__, __LINE__);
        if (!count) return;
        check(cudaMalloc(&ptr, count * sizeof(T)), "allocate", __FILE__, __LINE__);
        try {
            check(cudaMemsetAsync(ptr, 0, count * sizeof(T), 0), "initialize", __FILE__, __LINE__);
            check(cudaStreamSynchronize(0), "finish initialization", __FILE__, __LINE__);
        } catch (...) {
            cudaFree(ptr);
            ptr = nullptr;
            throw;
        }
        capacity = count;
    }

    // 名称写入错误消息；logger 负责将普通诊断传回调用方。
    void set_name(const char* buffer_name, Logger logger = {}) {
        if (buffer_name) {
            std::strncpy(name, buffer_name, sizeof(name) - 1);
            name[sizeof(name) - 1] = '\0';
        }
        logger_ = std::move(logger);
    }

    // 扩容后清零，预留约 10% + 100 个元素，分配路径完成默认流的初始化。
    // 失败后 capacity 反映实际可用空间，调用方捕获异常后可销毁或重新申请。
    void grow_to(size_t needed, const char* code_file, int code_line) {
        if (needed <= capacity) return;
        validate_allocation(needed, code_file, code_line);
        const size_t max_count = std::numeric_limits<size_t>::max() / sizeof(T);
        const size_t extra = needed / 10 + 100;
        const size_t new_capacity = extra <= max_count - needed ? needed + extra : needed;
        logger_.printf(LogLevel::Debug, name, code_file, code_line,
                       "grow from %zu to %zu elements", capacity, new_capacity);
        if (ptr) check(cudaFree(ptr), "free before grow", code_file, code_line);
        ptr = nullptr;
        capacity = 0;
        write_recorded_ = false;
        check(cudaMalloc(&ptr, new_capacity * sizeof(T)), "grow allocation", code_file, code_line);
        try {
            check(cudaMemsetAsync(ptr, 0, new_capacity * sizeof(T), 0), "clear grown buffer", code_file, code_line);
            check(cudaStreamSynchronize(0), "synchronize grown buffer", code_file, code_line);
        } catch (...) {
            release(ptr, "cleanup failed grow");
            ptr = nullptr;
            throw;
        }
        capacity = new_capacity;
    }

    // 在指定流上排入清零，并让后续 view()/download_to() 自动取得写依赖。
    void clear_async(cudaStream_t stream = 0) {
        if (!ptr || !capacity) return;
        wait_on(stream);
        check(cudaMemsetAsync(ptr, 0, capacity * sizeof(T), stream), "clear_async", __FILE__, __LINE__);
        record_write(stream);
    }
    void clear() {
        clear_async(0);
        check(cudaStreamSynchronize(0), "synchronize clear", __FILE__, __LINE__);
    }

    // 上传/下载以元素数计量，返回前只等待指定流；Host 源/目标可立即复用。
    // 可显式传入调用点，错误消息会同时包含缓冲区名称和调用位置。
    void upload_from(const T* host_source, size_t count, cudaStream_t stream = 0,
                     const char* code_file = __FILE__, int code_line = __LINE__) {
        validate_count(count, "upload", code_file, code_line);
        if (count && !host_source) fail("upload", "null host source", code_file, code_line);
        wait_on(stream);
        if (count) {
            const auto status = cudaMemcpyAsync(ptr, host_source, count * sizeof(T), cudaMemcpyHostToDevice, stream);
            check(status, "upload", code_file, code_line);
        }
        check(cudaStreamSynchronize(stream), "synchronize upload", code_file, code_line);
    }
    void download_to(T* host_destination, size_t count, cudaStream_t stream = 0,
                     const char* code_file = __FILE__, int code_line = __LINE__) const {
        validate_count(count, "download", code_file, code_line);
        if (count && !host_destination) fail("download", "null host destination", code_file, code_line);
        wait_on(stream);
        if (count) {
            const auto status = cudaMemcpyAsync(host_destination, ptr, count * sizeof(T), cudaMemcpyDeviceToHost, stream);
            check(status, "download", code_file, code_line);
        }
        check(cudaStreamSynchronize(stream), "synchronize download", code_file, code_line);
    }

    // 求和结果写入设备端一个 T；裸指针输出由调用方记录就绪事件或同步。
    void reduce_sum(T* device_output, size_t count, cudaStream_t stream = 0) {
        validate_count(count, "reduce_sum", __FILE__, __LINE__);
        if (!device_output) fail("reduce_sum", "null device output", __FILE__, __LINE__);
        wait_on(stream);
        size_t temp_bytes = 0;
        check(cub::DeviceReduce::Sum(nullptr, temp_bytes, ptr, device_output, count, stream),
              "query reduction storage", __FILE__, __LINE__);
        reserve_temp(temp_bytes);
        check(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_capacity, ptr, device_output, count, stream),
              "reduce_sum", __FILE__, __LINE__);
    }

    // 输出 CSR 偏移 [0, input[0], input[0]+input[1], ...]，共 count+1 项。
    // 输入与输出须为独立数组；输出缓冲区携带前缀和完成事件。
    void scan_to(GpuBuffer<T>& output, size_t count, cudaStream_t stream = 0) {
        validate_count(count, "scan input", __FILE__, __LINE__);
        if (count == std::numeric_limits<size_t>::max() || count >= output.capacity)
            fail("scan output", format_message("buffer [%s] needs count+1 slots, count=%zu capacity=%zu",
                 output.name, count, output.capacity), __FILE__, __LINE__);
        if (this == &output) fail("scan", "input and output must be distinct", __FILE__, __LINE__);
        wait_on(stream);
        output.wait_on(stream);
        check(cudaMemsetAsync(output.ptr, 0, sizeof(T), stream), "initialize scan origin", __FILE__, __LINE__);
        if (!count) { output.record_write(stream); return; }
        size_t temp_bytes = 0;
        check(cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, ptr, output.ptr + 1, count, stream),
              "query scan storage", __FILE__, __LINE__);
        reserve_temp(temp_bytes);
        check(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_capacity,
                                          ptr, output.ptr + 1, count, stream),
              "scan", __FILE__, __LINE__);
        output.record_write(stream);
    }

    // this 保存 0/1 标记，output 保存选中的原下标；host_selected 接收数量。
    // 返回前等待指定流，保证 CPU 可以立即使用选中数量。
    template <typename OutT>
    void flag_to(GpuBuffer<OutT>& output, size_t count, int* host_selected, cudaStream_t stream = 0) {
        if (!host_selected) fail("flag", "null host count", __FILE__, __LINE__);
        if (!count) { *host_selected = 0; return; }
        validate_count(count, "flag input", __FILE__, __LINE__);
        if (count > output.capacity)
            fail("flag output", format_message("buffer [%s] needs %zu slots, capacity=%zu",
                 output.name, count, output.capacity), __FILE__, __LINE__);
        if (count > static_cast<size_t>(INT_MAX)) fail("flag", "count exceeds INT_MAX", __FILE__, __LINE__);
        wait_on(stream);
        output.wait_on(stream);
        if (!d_select_count)
            check(cudaMalloc(&d_select_count, sizeof(int)), "allocate selected count", __FILE__, __LINE__);
        cub::CountingInputIterator<OutT> indices(OutT(0));
        const int num_items = static_cast<int>(count);
        size_t temp_bytes = 0;
        check(cub::DeviceSelect::Flagged(nullptr, temp_bytes, indices, ptr, output.ptr,
                                       d_select_count, num_items, stream),
              "query selection storage", __FILE__, __LINE__);
        reserve_temp(temp_bytes);
        check(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_capacity,
                                       indices, ptr, output.ptr, d_select_count, num_items, stream),
              "flag", __FILE__, __LINE__);
        check(cudaMemcpyAsync(host_selected, d_select_count, sizeof(int), cudaMemcpyDeviceToHost, stream),
              "download selected count", __FILE__, __LINE__);
        check(cudaStreamSynchronize(stream), "synchronize selection", __FILE__, __LINE__);
    }

    // 析构释放数据、CUB 临时区和选择计数器；清理失败通过 Warning 回传。
    ~GpuBuffer() noexcept {
        release(ptr, "free data");
        release(d_temp_storage, "free CUB storage");
        release(d_select_count, "free selected count");
        if (write_completion_) {
            const auto status = cudaEventDestroy(write_completion_);
            if (status != cudaSuccess)
                logger_.printf(LogLevel::Warning, name, __FILE__, __LINE__, "free write event: %s",
                               cudaGetErrorString(status));
        }
    }

private:
    Logger logger_;
    cudaEvent_t write_completion_ = nullptr;
    bool write_recorded_ = false;

    [[noreturn]] void fail(const char* operation, const std::string& message,
                          const char* file, int line) const {
        throw GpuBufferError(format_message("GpuBuffer [%s] %s: %s (%s:%d)",
                             name, operation, message.c_str(), file ? file : "unknown", line));
    }
    void check(cudaError_t status, const char* operation, const char* file, int line) const {
        if (status != cudaSuccess) fail(operation, cudaGetErrorString(status), file, line);
    }
    void validate_allocation(size_t count, const char* file, int line) const {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T))
            fail("allocate", "size overflow", file, line);
    }
    void validate_count(size_t count, const char* operation, const char* file, int line) const {
        if (count > capacity)
            fail(operation, format_message("need %zu elements, capacity=%zu", count, capacity), file, line);
    }
    void reserve_temp(size_t bytes) {
        if (bytes <= temp_storage_capacity) return;
        if (d_temp_storage) check(cudaFree(d_temp_storage), "free CUB storage", __FILE__, __LINE__);
        d_temp_storage = nullptr;
        temp_storage_capacity = 0;
        check(cudaMalloc(&d_temp_storage, bytes), "allocate CUB storage", __FILE__, __LINE__);
        temp_storage_capacity = bytes;
    }
    void release(void* pointer, const char* operation) const noexcept {
        if (!pointer) return;
        const auto status = cudaFree(pointer);
        if (status != cudaSuccess)
            logger_.printf(LogLevel::Warning, name, __FILE__, __LINE__, "%s: %s",
                           operation, cudaGetErrorString(status));
    }
};
} // namespace MetaD_zqc
