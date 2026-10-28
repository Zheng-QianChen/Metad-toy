#pragma once
// 测试专用：把 MPI_Allreduce 包装成 Graph/AVE 使用的原地 SumReducer。
// 生产路径应使用 EngineAdapter::sum_reducer()，不要依赖本头文件。
#include "zqc_collective.h"
#include <mpi.h>
#include <stdexcept>
#include <vector>

inline MetaD_zqc::SumReducer mpi_sum_reducer(MPI_Comm communicator) {
    return [communicator](double* values, size_t count) {
        if (!values && count) throw std::invalid_argument("mpi_sum_reducer: null values");
        if (!count) return;
        std::vector<double> global(count);
        if (MPI_Allreduce(values, global.data(), static_cast<int>(count), MPI_DOUBLE, MPI_SUM,
                          communicator) != MPI_SUCCESS)
            throw std::runtime_error("mpi_sum_reducer: MPI_Allreduce failed");
        for (size_t index = 0; index < count; ++index) values[index] = global[index];
    };
}
