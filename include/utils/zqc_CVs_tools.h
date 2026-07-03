#pragma once  // 必须添加这一行
#include <cstring>
#include <source_location>
#include <cub/cub.cuh>

#include "lammps.h"
#include "pair.h"

#include "zqc_debug.h"
#include "zqc_switch_function.h"

namespace MetaD_zqc {
    class Averager {
    public:
        virtual void compute(int n, int glob_N, double* arr, int* mask, int groupbit, double &ave) = 0;
        virtual void compute_sw(int n, double* arr, int* mask, int groupbit, double &cvi_sum, double &sw_sum, SwitchFunction* sw_func) = 0;
    };
    class KahanAverager : public Averager {
        void compute(int n, int glob_N, double* arr, int* mask, int groupbit, double &ave) override;
        void compute_sw(int n, double* arr, int* mask, int groupbit, double &cvi_sum, double &sw_sum, SwitchFunction* sw_func) override;
    };
    class CUBAverager : public Averager {
    public:
        CUBAverager();  
        ~CUBAverager();
    private:
        double *d_sum;
        void compute(int n, int glob_N, double* arr, int* mask, int groupbit, double &ave) override;
        void compute_sw(int n, double* arr, int* mask, int groupbit, double &cvi_sum, double &sw_sum, SwitchFunction* sw_func) {cvi_sum=0;sw_sum=0;}
    };

    template <typename T>
    struct GpuBuffer {
        LAMMPS_NS::LAMMPS* lmp;
        FILE* f_check = nullptr;
        LAMMPS_NS::Error* error = nullptr;

        char name[32];
        T* ptr = nullptr;
        size_t capacity = 0;
        void* d_temp_storage = nullptr;
        size_t temp_storage_capacity = 0;

        // 构造函数初始化
        GpuBuffer() {
            std::strncpy(name, "Unknown", 31);
            name[31] = '\0';
        }

        /**
        * @brief 为缓冲区绑定元数据（Name, File Handle, Error Handle）
        */
        void set_name(const char* n, FILE* f, LAMMPS_NS::Error* err, LAMMPS_NS::LAMMPS* lmp) {
            if (n) {
                std::strncpy(name, n, 31);
                name[31] = '\0';
            }
            this->f_check = f;
            this->error = err;
            this->lmp = lmp;
        }

        /**
        * @brief 动态扩容显存：若当前容量不足，则重新分配并清理旧数据
        * @param needed 需要的最小容量
        * @note 包含 10% 的冗余策略，减少频繁分配引发的显存碎片
        */
        void grow_to(size_t needed, const char* code_file, int code_line) {
            // if (true) {
            if (needed > capacity) {
                LOG("DEBUG: GPU Buffer [%s] expanded from %zu to %zu\n", name, capacity, (size_t)(needed * 1.1) + 100);
                LOG("in %s : %d\n", code_file, code_line);
                LOG("the past location is %p\n", ptr);
                cudaError_t sync_err = cudaDeviceSynchronize();
                if (ptr) SAFE_CUDA_FREE_NOFILE(ptr, code_file, code_line);
                // 额外多给 10% 冗余，防止频繁 Malloc
                capacity = (size_t)(needed * 1.1) + 100;
                SAFE_CUDA_MALLOC_NOLMP(&ptr, capacity * sizeof(T), f_check);
                sync_err = cudaDeviceSynchronize();
                cudaMemset(ptr, 0, capacity * sizeof(T));
                LOG("now location is %p\n", ptr);
                cudaError_t syncErr = cudaDeviceSynchronize();
                ERR_COND((syncErr != cudaSuccess),"Kernel execution error: %s\n", cudaGetErrorString(syncErr));
            }
        }

        /**
        * @brief 异步内存清零：在指定的 CUDA 流上将缓冲区初始化为 0
        * @param stream 使用的异步队列
        */
        void clear_async(cudaStream_t stream = 0) {
            if (ptr && capacity > 0) {
                // 硬件级硬件并行擦除，速度极快
                cudaMemsetAsync(ptr, 0, capacity * sizeof(T), stream);
                cudaError_t syncErr = cudaDeviceSynchronize();
                ERR_COND((syncErr != cudaSuccess),"Kernel execution error: %s\n", cudaGetErrorString(syncErr));
            }
        }
        void clear() {
            // 强制同步
            if (ptr && capacity > 0) {cudaMemset(ptr, 0, capacity * sizeof(T));}
        }

        /**
        * @brief 将数据从 Host（CPU）传输到 Device（GPU）
        * @param h_src Host 端源地址
        * @param count 传输元素个数
        * @param stream CUDA 异步流
        */
        void upload_from(const T* h_src, size_t count, cudaStream_t stream = 0,
            const char* code_file = __FILE__, int code_line = __LINE__) {
            ERR_COND(count > capacity, 
                "==============================================================\n"
                "🚨 GPU BUFFER OVERFLOW ERROR 🚨\n"
                "  Buffer Name   : %s\n"
                "  Attempted to  : UPLOAD (Host to Device)\n"
                "  Required Size : %zu elements\n"
                "  Max Capacity  : %zu elements\n"
                "  Element Size  : %zu bytes\n"
                "  Total Weight  : %.2f MB\n"
                "  Caller Loc    : %s : %d\n"
                "==============================================================",
                name, count, capacity, sizeof(T), 
                (double)(count * sizeof(T)) / (1024 * 1024), 
                code_file, code_line
            );
            if (stream) {
                cudaMemcpyAsync(ptr, h_src, count * sizeof(T), cudaMemcpyHostToDevice, stream);
            } else {
                cudaMemcpy(ptr, h_src, count * sizeof(T), cudaMemcpyHostToDevice);
            }
            cudaError_t syncErr = cudaDeviceSynchronize();
            ERR_COND((syncErr != cudaSuccess),"Kernel execution error: %s\n", cudaGetErrorString(syncErr));
        }

        /**
        * @brief 将数据从 Device（GPU）拉回 Host（CPU）
        * @param h_dst Host 端目标地址
        * @param count 传输元素个数
        */
        void download_to(T* h_dst, size_t count, cudaStream_t stream = 0,
            const char* code_file = __FILE__, int code_line = __LINE__) const {
            ERR_COND(count > capacity, 
                "==============================================================\n"
                "🚨 GPU BUFFER OVERFLOW ERROR 🚨\n"
                "  Buffer Name   : %s\n"
                "  Attempted to  : UPLOAD (Host to Device)\n"
                "  Required Size : %zu elements\n"
                "  Max Capacity  : %zu elements\n"
                "  Element Size  : %zu bytes\n"
                "  Total Weight  : %.2f MB\n"
                "  Caller Loc    : %s : %d\n"
                "==============================================================",
                name, count, capacity, sizeof(T), 
                (double)(count * sizeof(T)) / (1024 * 1024), 
                code_file, code_line
            );
            if (stream) {
                cudaMemcpyAsync(h_dst, ptr, count * sizeof(T), cudaMemcpyDeviceToHost, stream);
            } else {
                cudaMemcpy(h_dst, ptr, count * sizeof(T), cudaMemcpyDeviceToHost);
            }
            cudaError_t syncErr = cudaDeviceSynchronize();
            ERR_COND((syncErr != cudaSuccess),"Kernel execution error: %s\n", cudaGetErrorString(syncErr));
        }

        /**
        * @brief 并行求和算子：使用 CUB 库对当前缓冲区进行归约求和
        * @param d_output GPU 端存储规约结果的指针
        * @param count 参与求和的元素数量
        * @param stream CUDA 异步流
        * @note 该函数自动管理临时显存（temp_storage），实现“即插即用”
        */
        void reduce_sum(T* d_output, size_t count, cudaStream_t stream = 0) {
            size_t temp_bytes = 0;
            // 1. 获取所需临时显存大小
            cub::DeviceReduce::Sum(nullptr, temp_bytes, ptr, d_output, count, stream);
            // 2. 如果当前临时显存不够，则扩容
            if (temp_bytes > temp_storage_capacity) {
                if (d_temp_storage) cudaFree(d_temp_storage);
                cudaMalloc(&d_temp_storage, temp_bytes);
                temp_storage_capacity = temp_bytes;
            }
            // 3. 执行正式规约
            cub::DeviceReduce::Sum(d_temp_storage, temp_storage_capacity, ptr, d_output, count, stream);
        }

        /**
        * @brief 执行并行前缀和 (Prefix Sum)
        * @tparam OutputT 输出缓冲区的数据类型
        * @param d_output_buf 目标缓冲区 (GpuBuffer 对象)
        * @param count 参与计算的元素数量 (注意：CSR格式通常需要 num_atoms + 1)
        * @param stream CUDA 异步流
        */
        void scan_to(GpuBuffer<T>& d_output_buf, size_t count, cudaStream_t stream = 0) {
            // 1. 运行时安全检查：防止 Buffer 容量不足
            ERR_COND(count > this->capacity, 
                    "Scan Input Buffer [%s] capacity too small: need %zu, have %zu", 
                    this->name, count, this->capacity);
            ERR_COND(count+1 > d_output_buf.capacity, 
                    "Scan Output Buffer [%s] capacity too small: need %zu, have %zu", 
                    d_output_buf.name, count, d_output_buf.capacity);
            ERR_COND((this == &d_output_buf), 
                    "In-place scan is not supported for GpuBuffer [%s]. Please use distinct buffers.", 
                    this->name);

            // 确保 CSR 的起点永远是 0
            T zero = 0;
            cudaMemcpyAsync(d_output_buf.ptr, &zero, sizeof(T), cudaMemcpyHostToDevice, stream);

            // 3. 执行前缀和算法
            size_t temp_bytes = 0;
            cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, 
                                        this->ptr, d_output_buf.ptr+1, 
                                        count, stream);
            
            // 动态扩容临时显存
            if (temp_bytes > this->temp_storage_capacity) {
                if (this->d_temp_storage) cudaFree(this->d_temp_storage);
                cudaMalloc(&this->d_temp_storage, temp_bytes);
                this->temp_storage_capacity = temp_bytes;
            }
            
            cub::DeviceScan::InclusiveSum(this->d_temp_storage, this->temp_storage_capacity, 
                                        this->ptr, d_output_buf.ptr+1, 
                                        count, stream);
        }

        ~GpuBuffer() { if (ptr) SAFE_CUDA_FREE_NOFILE(ptr, __FILE__, __LINE__); }
    };
    

}