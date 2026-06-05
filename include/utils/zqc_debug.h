#pragma once  // 必须添加这一行

#define POW2(a) ((a) * (a))
#define POW3(a) ((a) * (a) * (a))
#define POW4(a) ((a) * (a) * (a) * (a))
#define POW5(a) ((a) * (a) * (a) * (a) * (a))
#define POW6(a) (POW3(a) * POW3(a))
#define POW7(a) (POW3(a) * POW4(a))

#define MAX_ATOMS_PER_PACK 150

#define PI 3.1415926535897932385

#ifndef REGISTER_CV_MACRO
    #define REGISTER_CV_MACRO
    #include <cstdio>
    // 定义宏：传入 CV 的字符串名称和对应的类名
    #define CONCAT_IMPL(a, b) a##b
    #define CONCAT(a, b) CONCAT_IMPL(a, b)

    #define REGISTER_CV(name_str, class_type)                                     \
    namespace {                                                                   \
        __attribute__((constructor, visibility("default")))                       \
        void CONCAT(reg_func_, __LINE__)() {                                      \
            MetaD_zqc::CVFactory::register_cv(name_str, &class_type);     \
        }                                                                         \
    }
#endif


// #define DEBUG
#ifdef DEBUG
    #define DEBUG_LOG(format, ...) do { \
        int _me = lmp->comm->me; \
        if (f_check != NULL) { \
            fprintf(f_check, "[Rank:%d][%s:%d] " format "\n", _me, __FILE__, __LINE__, ##__VA_ARGS__); \
            fflush(f_check); \
        } \
    } while(0)
#else
    #define DEBUG_LOG(...) do {} while(0)
#endif

#ifdef DEBUG
    #define DEBUG_LOG_COND(cond, format, ...) do { \
        int _me = lmp->comm->me; \
        if (cond) { \
            if (f_check != NULL) { \
                fprintf(f_check, "[Rank:%d][%s:%d] " format "\n", _me, __FILE__, __LINE__, ##__VA_ARGS__); \
                fflush(f_check); \
            } \
        } \
    } while(0)
#else
    #define DEBUG_LOG_COND(cond, format, ...) ((void)0)  // 替换为空操作
#endif

#ifdef DEBUG
    #define ERR_COND(cond, format, ...) do { \
        int _me = lmp->comm->me; \
        if (cond) { \
            if (f_check != NULL) { \
                fprintf(f_check, "[Rank:%d][%s:%d] " format "\n", _me, __FILE__, __LINE__, ##__VA_ARGS__); \
                fflush(f_check); \
            } \
            exit(1);\
        } \
    } while(0)
#else
    #define ERR_COND(cond, format, ...) do { \
        if (cond) { \
            char _err_buf[512]; \
            snprintf(_err_buf, sizeof(_err_buf), format, ##__VA_ARGS__); \
            error->all(FLERR, _err_buf); \
            exit(1); \
        } \
    } while(0)
#endif

#ifdef DEBUG
    #define DEBUG_RUN(code) do { \
        code; \
    } while(0)
#else
    #define DEBUG_RUN(code) ((void)0)
#endif

#ifdef DEBUG
    #define DEBUG_RUN_COND(cond, code) do { \
        if (cond) { \
            code; \
        } \
    } while(0)
#else
    #define DEBUG_RUN_COND(cond, code) ((void)0)
#endif

#define LOG_COND(cond, format, ...) do { \
    if (cond) { \
        int _me = lmp->comm->me; \
        if (lmp->screen) fprintf(lmp->screen, "[Rank:%d][%s:%d] " format "\n", _me, __FILE__, __LINE__, ##__VA_ARGS__); \
        if (lmp->logfile) fprintf(lmp->logfile, "[Rank:%d][%s:%d] " format "\n", _me, __FILE__, __LINE__, ##__VA_ARGS__); \
        if (f_check != NULL) { \
            fprintf(f_check, "[Rank:%d][%s:%d] " format "\n", _me, __FILE__, __LINE__, ##__VA_ARGS__); \
            fflush(f_check); \
        } \
    } \
} while(0)

#define LOG(format, ...) do { \
    int _me = lmp->comm->me; \
    if (lmp->screen) fprintf(lmp->screen, "[Rank:%d][%s:%d] " format "\n", _me, __FILE__, __LINE__, ##__VA_ARGS__); \
    if (lmp->logfile) fprintf(lmp->logfile, "[Rank:%d][%s:%d] " format "\n", _me, __FILE__, __LINE__, ##__VA_ARGS__); \
    if (f_check != NULL) { \
        fprintf(f_check, "[Rank:%d][%s:%d] " format "\n", _me, __FILE__, __LINE__, ##__VA_ARGS__); \
        fflush(f_check); \
    } \
} while(0)

#ifdef DEBUG
    #define SAFE_CUDA_FREE(ptr) do { \
        cudaGetLastError(); \
        cudaError_t err = cudaFree(ptr);\
        ptr = nullptr; \
        if (err != cudaSuccess) { \
            fprintf(f_check, "CUDA Error at %s:%d\n", __FILE__, __LINE__); \
            fprintf(f_check, "  Code: %d, Reason: %s\n", err,\
                cudaGetErrorString(err));\
            fflush(f_check); \
            exit(1);\
        } \
    } while(0)
    #define SAFE_CUDA_FREE_NOFILE(ptr,caller_file, caller_line) do { \
        cudaGetLastError(); \
        cudaError_t err = cudaFree(ptr);\
        ptr = nullptr; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d (Defined at %s:%d)\n", \
                caller_file, caller_line, __FILE__, __LINE__); \
            printf("  Code: %d, Reason: %s\n", err,\
                cudaGetErrorString(err));\
            fflush(stderr); \
        } \
    } while(0)
#else
    #define SAFE_CUDA_FREE(ptr) do{ \
        cudaGetLastError(); \
        cudaError_t err = cudaFree(ptr);\
        ptr = nullptr; \
        if (err != cudaSuccess) { \
            (error)->all(FLERR, "Device memory free failed more information \
                please check the crystallize fix outputfile\n"); \
            exit(1);\
        } \
    } while(0)
    #define SAFE_CUDA_FREE_NOFILE(ptr,caller_file, caller_line) do { \
        cudaGetLastError(); \
        cudaError_t err = cudaFree(ptr);\
        ptr = nullptr; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d (Defined at %s:%d)\n", \
                caller_file, caller_line, __FILE__, __LINE__); \
            printf("  Code: %d, Reason: %s\n", err,\
                cudaGetErrorString(err));\
            fflush(stderr); \
        } \
    } while(0)
#endif

#ifdef DEBUG
    #define SAFE_CUDA_MALLOC(ptr, size, f_check) do { \
            cudaGetLastError(); \
            size_t free_mem, total_mem; \
            cudaError_t mem_info_err = cudaMemGetInfo(&free_mem, &total_mem); \
            if (mem_info_err != cudaSuccess) { \
                fprintf(f_check, "cudaMemGetInfo failed: %s\n", cudaGetErrorString(mem_info_err)); \
                fflush(f_check); \
                (error)->all(FLERR, "GPU memory query failed"); \
            } \
            if ((size) > free_mem || mem_info_err != cudaSuccess) { \
                fprintf(f_check, "Memory Check Failed!\n"); \
                fprintf(f_check, "  Requested: %zu bytes (%.2f KB)\n", (size_t)(size), (size)/1024.0); \
                fprintf(f_check, "  Available: %zu bytes (%.2f MB)\n", free_mem, free_mem/(1024.0*1024.0)); \
                fflush(f_check); \
                (error)->all(FLERR, "GPU Memory check failed"); \
            } \
            cudaError_t MallocErr = cudaMalloc(ptr, (size)); \
            if (MallocErr != cudaSuccess) { \
                fprintf(f_check, "CUDA Malloc failed at %s:%d: %s (Size: %zu bytes)\n", \
                        __FILE__, __LINE__, cudaGetErrorString(MallocErr), (size_t)(size)); \
                fflush(f_check); \
                (error)->all(FLERR, "Device memory allocation failed\n"); \
            } \
        } while(0)
    #define SAFE_CUDA_MALLOC_NOLMP(ptr, size, f_check) do { \
            cudaGetLastError(); \
            size_t free_mem, total_mem; \
            cudaError_t mem_info_err = cudaMemGetInfo(&free_mem, &total_mem); \
            if (mem_info_err != cudaSuccess) { \
                fprintf(f_check, "cudaMemGetInfo failed: %s\n", cudaGetErrorString(mem_info_err)); \
                fflush(f_check); \
            } \
            if ((size) > free_mem || mem_info_err != cudaSuccess) { \
                fprintf(f_check, "Memory Check Failed!\n"); \
                fprintf(f_check, "  Requested: %zu bytes (%.2f KB)\n", (size_t)(size), (size)/1024.0); \
                fprintf(f_check, "  Available: %zu bytes (%.2f MB)\n", free_mem, free_mem/(1024.0*1024.0)); \
                fflush(f_check); \
            } \
            cudaError_t MallocErr = cudaMalloc(ptr, (size)); \
            if (MallocErr != cudaSuccess) { \
                fprintf(f_check, "CUDA Malloc failed at %s:%d: %s (Size: %zu bytes)\n", \
                        __FILE__, __LINE__, cudaGetErrorString(MallocErr), (size_t)(size)); \
                fflush(f_check); \
            } \
        } while(0)
#else
    #define SAFE_CUDA_MALLOC(ptr, size, f_check) do { \
            cudaGetLastError(); \
            size_t free_mem, total_mem; \
            cudaError_t mem_info_err = cudaMemGetInfo(&free_mem, &total_mem); \
            if (mem_info_err != cudaSuccess) { \
                (error)->all(FLERR, "GPU memory query failed"); \
                exit(1);\
            } \
            if ((size) > free_mem) { \
                (error)->all(FLERR, "Memory allocation aborted"); \
                exit(1);\
            } \
            cudaError_t MallocErr = cudaMalloc(ptr, (size)); \
            if (MallocErr != cudaSuccess) { \
                (error)->all(FLERR, "Device memory allocation failed\n"); \
                exit(1);\
            } \
        } while(0)
    #define SAFE_CUDA_MALLOC_NOLMP(ptr, size, f_check) do { \
            cudaGetLastError(); \
            size_t free_mem, total_mem; \
            cudaError_t mem_info_err = cudaMemGetInfo(&free_mem, &total_mem); \
            if (mem_info_err != cudaSuccess) { \
                fprintf(f_check, "cudaMemGetInfo failed: %s\n", cudaGetErrorString(mem_info_err)); \
            } \
            if ((size) > free_mem || mem_info_err != cudaSuccess) { \
                fprintf(f_check, "Memory Check Failed!\n"); \
                fprintf(f_check, "  Requested: %zu bytes (%.2f KB)\n", (size_t)(size), (size)/1024.0); \
                fprintf(f_check, "  Available: %zu bytes (%.2f MB)\n", free_mem, free_mem/(1024.0*1024.0)); \
                fflush(f_check); \
            } \
            cudaError_t MallocErr = cudaMalloc(ptr, (size)); \
            if (MallocErr != cudaSuccess) { \
                fprintf(f_check, "CUDA Malloc failed at %s:%d: %s (Size: %zu bytes)\n", \
                        __FILE__, __LINE__, cudaGetErrorString(MallocErr), (size_t)(size)); \
                fflush(f_check); \
            } \
        } while(0)
#endif


// #define SAFE_CUDA_MEMCPY(dst, src, size_bytes, kind, f_check) do { \
//         cudaError_t _err = cudaMemcpy((dst), (src), (size_bytes), (kind)); \
//         if (_err != cudaSuccess) { \
//             fprintf(f_check, "CUDA Memcpy failed at %s:%d\n", __FILE__, __LINE__); \
//             fprintf(f_check, "  Operation: %s -> %s\n", \
//                 ((kind) == cudaMemcpyHostToDevice) ? "HostToDevice" : \
//                 ((kind) == cudaMemcpyDeviceToHost) ? "DeviceToHost" : "UnknownDirection"); \
//             fprintf(f_check, "  Error: %s\n", cudaGetErrorString(_err)); \
//             fprintf(f_check, "  Size: %.2f MB\n", (double)(size_bytes) / (1024 * 1024)); \
//             (error)->all(FLERR, "CUDA memory copy operation failed"); \
//         } \
//         DEBUG_LOG_COND((dst) == NULL, "Destination pointer invalid after copy"); \
//     } while(0)


#ifdef DEBUG
    #define SAFE_CUDA_MEMCPY(dst, src, size_bytes, kind, f_check) do { \
            cudaGetLastError(); \
            cudaError_t _err = cudaMemcpy((dst), (src), (size_bytes), (kind)); \
            if (_err != cudaSuccess) { \
                fprintf(f_check, "CUDA Memcpy failed at %s:%d\n", __FILE__, __LINE__); \
                const char* _dir = ((kind) == cudaMemcpyHostToDevice) ? "Host to Device" : \
                                ((kind) == cudaMemcpyDeviceToHost) ? "Device to Host" : \
                                "Other Direction"; \
                fprintf(f_check, "  Direction: %s\n", _dir); \
                fprintf(f_check, "  Error: %s\n", cudaGetErrorString(_err)); \
                fprintf(f_check, "  Size: %.2f MB\n", (double)(size_bytes) / (1024 * 1024)); \
                fflush(f_check); \
                (error)->all(FLERR, "CUDA memory copy operation failed"); \
                exit(1); \
            } \
        } while(0)
#else
    #define SAFE_CUDA_MEMCPY(dst, src, size_bytes, kind, f_check) do { \
            cudaGetLastError(); \
            cudaError_t _err = cudaMemcpy((dst), (src), (size_bytes), (kind)); \
            if (_err != cudaSuccess) { \
                (error)->all(FLERR, "CUDA memory copy operation failed"); \
                exit(1); \
            } \
        } while(0)
#endif