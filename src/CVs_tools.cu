#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include "zqc_CVs_tools.h"
#include "zqc_debug.h"

void MetaD_zqc::KahanAverager::compute(int n, double* arr, double &ave) {
    double sum = 0.0;
    double c = 0.0; 
    for (int i = 0; i < n; i++) {
        double y = arr[i] - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    ave = sum / n;
}

MetaD_zqc::CUBAverager::CUBAverager() {
    this->d_sum = nullptr;
    cudaMalloc(&(this->d_sum), sizeof(double));
}

MetaD_zqc::CUBAverager::~CUBAverager() {
    cudaFree(this->d_sum);
}

void MetaD_zqc::CUBAverager::compute(int n, double* d_arr, double &ave) {// d_arr 是已经在 GPU 显存中的指针
    if (n <= 0) { ave = 0.0; return; }
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_arr, this->d_sum, n);
    // 2. 分配临时存储
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // 3. 执行归约求和
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_arr, this->d_sum, n);
    // 4. 将结果拷回 Host
    double h_sum;
    cudaMemcpy(&h_sum, this->d_sum, sizeof(double), cudaMemcpyDeviceToHost);
    ave = h_sum / n;
    // 5. 释放资源 (在生产环境中，建议将 temp_storage 缓存起来避免重复 malloc)
    if (d_temp_storage) {
        cudaFree(d_temp_storage);
    }
}