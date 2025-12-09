#define POW2(a) ((a) * (a))
#define POW3(a) ((a) * (a) * (a))
#define POW4(a) ((a) * (a) * (a) * (a))
#define PI 3.1415926535897932385

#define DEBUG
#ifdef DEBUG
    #define DEBUG_LOG(format, ...) do { \
        if (f_check != NULL) { \
            fprintf(f_check, "[%s:%d] " format "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
            fflush(f_check); \
        } \
    } while(0)
#else
    #define DEBUG_LOG(...) do {} while(0)
#endif

#ifdef DEBUG
    #define DEBUG_LOG_COND(cond, format, ...) do { \
        if (cond) { \
                        if (f_check != NULL) { \
                fprintf(f_check, "[%s:%d] " format "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
                fflush(f_check); \
            } \
        } \
    } while(0)
#else
    #define DEBUG_LOG_COND(cond, format, ...) ((void)0)  // 替换为空操作
#endif

#ifdef DEBUG
    #define ERR_COND(cond, format, ...) do { \
        if (cond) { \
                        if (f_check != NULL) { \
                fprintf(f_check, "[%s:%d] " format "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
                fflush(f_check); \
            } \
        } \
    } while(0)
#else
    #define ERR_COND(cond, format, ...) do { \
        if (cond) { \
            char msg_buf[512]; \
            snprintf(msg_buf, sizeof(msg_buf), "[%s:%d] " format, __FILE__, __LINE__, ##__VA_ARGS__); \
            error->all(FLERR,msg_buf); \
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
        if (f_check != NULL) { \
            fprintf(f_check, "[%s:%d] " format "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
            fflush(f_check); \
        } \
    } \
} while(0)

#define LOG(format, ...) do { \
    if (f_check != NULL) { \
        fprintf(f_check, "[%s:%d] " format "\n", __FILE__, __LINE__, ##__VA_ARGS__); \
        fflush(f_check); \
    } \
} while(0)

#ifdef DEBUG
    #define SAFE_CUDA_FREE(ptr) do { \
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
#else
    #define SAFE_CUDA_FREE(ptr) do{ \
        cudaError_t err = cudaFree(ptr);\
        ptr = nullptr; \
        if (err != cudaSuccess) { \
            fprintf(f_check, "CUDA Error at %s:%d\n", __FILE__, __LINE__); \
            fprintf(f_check, "  Code: %d, Reason: %s\n", err,\
                cudaGetErrorString(err));\
            fflush(f_check); \
            exit(1);\
            (error)->all(FLERR, "Device memory free failed more information \
                please check the crystallize fix outputfile\n"); \
        } \
    } while(0)
#endif

#define SAFE_CUDA_MALLOC(ptr, size, f_check) do { \
        size_t free_mem, total_mem; \
        cudaError_t mem_info_err = cudaMemGetInfo(&free_mem, &total_mem); \
        if (mem_info_err != cudaSuccess) { \
            fprintf(f_check, "cudaMemGetInfo failed: %s\n", cudaGetErrorString(mem_info_err)); \
            (error)->all(FLERR, "GPU memory query failed"); \
        } \
        if ((size) > free_mem) { \
            fprintf(f_check, "Insufficient GPU memory at %s:%d\n", __FILE__, __LINE__); \
            fprintf(f_check, "  Requested: %zu MB\n", (size) / (1024 * 1024)); \
            fprintf(f_check, "  Available: %.2f MB\n", free_mem / (1024.0 * 1024.0)); \
            (error)->all(FLERR, "Memory allocation aborted"); \
        } \
        cudaError_t MallocErr = cudaMalloc(ptr, (size)); \
        if (MallocErr != cudaSuccess) { \
            fprintf(f_check, "CUDA Malloc failed at %s:%d: %s (Size: %zu bytes)\n", \
                    __FILE__, __LINE__, cudaGetErrorString(MallocErr), (size_t)(size)); \
            (error)->all(FLERR, "Device memory allocation failed\n"); \
        } \
        DEBUG_LOG_COND(*(ptr) == NULL, "Device pointer " #ptr " is NULL after allocation"); \
    } while(0)

#define SAFE_CUDA_MEMCPY(dst, src, size_bytes, kind, f_check) do { \
        cudaError_t _err = cudaMemcpy((dst), (src), (size_bytes), (kind)); \
        if (_err != cudaSuccess) { \
            fprintf(f_check, "CUDA Memcpy failed at %s:%d\n", __FILE__, __LINE__); \
            fprintf(f_check, "  Operation: %s -> %s\n", \
                ((kind) == cudaMemcpyHostToDevice) ? "HostToDevice" : \
                ((kind) == cudaMemcpyDeviceToHost) ? "DeviceToHost" : "UnknownDirection"); \
            fprintf(f_check, "  Error: %s\n", cudaGetErrorString(_err)); \
            fprintf(f_check, "  Size: %.2f MB\n", (double)(size_bytes) / (1024 * 1024)); \
            (error)->all(FLERR, "CUDA memory copy operation failed"); \
        } \
        DEBUG_LOG_COND((dst) == NULL, "Destination pointer invalid after copy"); \
    } while(0)