#pragma once
#include "zqc_array_view.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace MetaD_zqc {
inline void execution_check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

// 借用生产者已经 record 的 CUDA event。所有者保持 event、数组有效，直到消费完成。
// wait_on 只向消费流排入依赖；wait_host 用于 CPU 读数或 MPI 打包的完成边界。
inline ViewReadiness cuda_event_readiness(cudaEvent_t event) {
    if (!event) throw std::invalid_argument("View readiness: null CUDA event");
    return {event,
        [](const void* context, void* consumer) {
            execution_check(cudaStreamWaitEvent(static_cast<cudaStream_t>(consumer),
                static_cast<cudaEvent_t>(const_cast<void*>(context)), 0));
        },
        [](const void* context) {
            execution_check(cudaEventSynchronize(static_cast<cudaEvent_t>(const_cast<void*>(context))));
        }};
}

// 每个共享几何环境持有一条 nonblocking 流，Graph 执行器另持有一条。
// 同环境的几何、CUB scratch 和各 CV 操作顺序提交，独立环境可在 GPU 上重叠。
// ready() 记录当前队尾；取得的视图应在下次 ready()/数据覆盖前提交消费依赖。
// 这些 Host 接口由同一调度线程调用，流和事件的所有者必须覆盖借用者的生命周期。
class CudaExecutionStream {
    cudaStream_t stream_ = nullptr;
    cudaEvent_t completion_ = nullptr;
public:
    CudaExecutionStream() {
        execution_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        const auto status = cudaEventCreateWithFlags(&completion_, cudaEventDisableTiming);
        if (status != cudaSuccess) { cudaStreamDestroy(stream_); execution_check(status); }
    }
    CudaExecutionStream(const CudaExecutionStream&) = delete;
    CudaExecutionStream& operator=(const CudaExecutionStream&) = delete;
    ~CudaExecutionStream() {
        cudaStreamSynchronize(stream_);
        cudaEventDestroy(completion_);
        cudaStreamDestroy(stream_);
    }
    cudaStream_t get() const { return stream_; }
    void synchronize() const { execution_check(cudaStreamSynchronize(stream_)); }
    ViewReadiness ready() const {
        execution_check(cudaEventRecord(completion_, stream_));
        return cuda_event_readiness(completion_);
    }
};
} // namespace MetaD_zqc
