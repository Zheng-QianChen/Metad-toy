#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include "zqc_CVs_tools.h"
#include "zqc_debug.h"

void MetaD_zqc::KahanAverager::compute(int n, int glob_N, double* arr, 
                                        int* mask, int groupbit, double &ave) {
    double sum = 0.0;
    double c = 0.0; 
    for (int i = 0; i < n; i++) {
        if (!(mask[i] & (groupbit) )) continue; // 跳过不参与平均的元素
        double y = arr[i] - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    ave = sum / glob_N;
}

void MetaD_zqc::KahanAverager::compute_sw(int n, double* arr, int* mask, int groupbit, 
                                double &cvi_sum, double &sw_sum, SwitchFunction* sw_func) {
    // 累加器 1: 加权值累加 (sum(LQ_l * f_cut(LQ_l))) -> 这才是你的 cvi_sum
    double s_cvi = 0.0;
    double c_cvi = 0.0; 
    
    // 累加器 2: 权重累加 (sum(f_cut(LQ_l)))
    double s_sw = 0.0;
    double c_sw = 0.0;

    for (int i = 0; i < n; i++) {
        if (!(mask[i] & groupbit)) continue;

        // 1. 计算当前原子的权重
        double weight = sw_func->f(arr[i]); 
        
        // 2. 累加权重 (sum_sw)
        double y_sw = weight - c_sw;
        double t_sw = s_sw + y_sw;
        c_sw = (t_sw - s_sw) - y_sw;
        s_sw = t_sw;

        // 3. 累加加权后的值 (sum_cvi = arr[i] * weight)
        // 使用 Kahan 累加处理乘积结果
        double weighted_val = arr[i] * weight;
        double y_cvi = weighted_val - c_cvi;
        double t_cvi = s_cvi + y_cvi;
        c_cvi = (t_cvi - s_cvi) - y_cvi;
        s_cvi = t_cvi;
    }

    // 输出最终 Kahan 补偿后的总和
    cvi_sum = s_cvi;
    sw_sum = s_sw;
}


MetaD_zqc::CUBAverager::CUBAverager() {
    this->d_sum = nullptr;
    cudaMalloc(&(this->d_sum), sizeof(double));
}

MetaD_zqc::CUBAverager::~CUBAverager() {
    cudaFree(this->d_sum);
}

void MetaD_zqc::CUBAverager::compute(int n, int glob_N, double* d_arr, 
                                        int* mask, int groupbit, double &ave) {// d_arr 是已经在 GPU 显存中的指针
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
    ave = h_sum / glob_N;
    // 5. 释放资源 (在生产环境中，建议将 temp_storage 缓存起来避免重复 malloc)
    if (d_temp_storage) {
        cudaFree(d_temp_storage);
    }
}

// GPU_BUFFER!
