#pragma once  // 必须添加这一行
#include <cstring>
#include <cub/cub.cuh>
#include <cub/iterator/counting_input_iterator.cuh>

#include "lammps.h"
#include "pair.h"

#include "zqc_debug.h"
#include "zqc_gpu_buffer.cuh"
#include "zqc_switch_function.h"

namespace MetaD_zqc {
    class Averager {
    public:
        virtual ~Averager() = default;
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


    

}