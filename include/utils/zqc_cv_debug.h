#pragma once
#include "zqc_diagnostics.h"
#include <cuda_runtime.h>
#include <stdexcept>

// CV 内部的错误由异常返回主流程；日志交给构造时绑定的回调。
#define ERR_COND(condition, ...) do { if (condition) throw std::runtime_error(MetaD_zqc::format_message(__VA_ARGS__)); } while (0)
#define LOG(...) logger.printf(MetaD_zqc::LogLevel::Info,"CV",__FILE__,__LINE__,__VA_ARGS__)
#define LOG_COND(condition, ...) do { if (condition) { LOG(__VA_ARGS__); } } while (0)
#ifdef DEBUG
#define DEBUG_LOG(...) logger.printf(MetaD_zqc::LogLevel::Debug,"CV",__FILE__,__LINE__,__VA_ARGS__)
#define DEBUG_LOG_COND(condition, ...) do { if (condition) { DEBUG_LOG(__VA_ARGS__); } } while (0)
#define DEBUG_RUN(code) do { code; } while (0)
#define DEBUG_RUN_COND(condition, code) do { if (condition) { code; } } while (0)
#else
#define DEBUG_LOG(...) do {} while (0)
#define DEBUG_LOG_COND(...) do {} while (0)
#define DEBUG_RUN(...) do {} while (0)
#define DEBUG_RUN_COND(...) do {} while (0)
#endif
namespace MetaD_zqc {
inline void cv_check_cuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
}
#define SAFE_CUDA_MEMCPY(destination, source, bytes, kind, ...) MetaD_zqc::cv_check_cuda(cudaMemcpy(destination,source,bytes,kind))
#define POW2(value) ((value)*(value))
#define POW3(value) ((value)*(value)*(value))
#define POW4(value) (POW2(value)*POW2(value))
#define POW5(value) (POW4(value)*(value))
#define POW6(value) (POW3(value)*POW3(value))
#define POW7(value) (POW6(value)*(value))
#define MAX_ATOMS_PER_PACK 150
#ifndef PI
#define PI 3.1415926535897932385
#endif
#ifndef REGISTER_CV_MACRO
#define REGISTER_CV_MACRO
#define CONCAT_IMPL(a,b) a##b
#define CONCAT(a,b) CONCAT_IMPL(a,b)
#define REGISTER_CV(name,creator) namespace { __attribute__((constructor)) void CONCAT(register_cv_,__LINE__)() { MetaD_zqc::CVFactory::register_cv(name,&creator); } }
#endif

// kernel 提交错误即时回传；执行错误由下一处流等待/Host 下载回传。
#define CUDA_LAUNCH_CHECK(...) MetaD_zqc::cv_check_cuda(cudaGetLastError())
