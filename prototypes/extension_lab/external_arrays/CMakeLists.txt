# 独立构建入口。使用 LAMMPS 自带的 Kokkos 和 GPUMD 的容器头文件。
# CXX 编译器由调用方指定为 Kokkos 的 nvcc_wrapper，使 .cpp 中也能编译 CUDA kernel。
cmake_minimum_required(VERSION 3.22)
project(extension_arrays LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT EXISTS "${LAMMPS_SOURCE}/lib/kokkos/CMakeLists.txt")
  message(FATAL_ERROR "Set LAMMPS_SOURCE to a LAMMPS source checkout")
endif()
if(NOT EXISTS "${GPUMD_SOURCE}/src/utilities/gpu_vector.cuh")
  message(FATAL_ERROR "Set GPUMD_SOURCE to a GPUMD source checkout")
endif()
# CUDA 负责设备 View，Serial 提供 Host 执行空间与镜像所需支持。
set(Kokkos_ENABLE_CUDA ON CACHE BOOL "CUDA execution")
set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "Host mirrors")
add_subdirectory("${LAMMPS_SOURCE}/lib/kokkos" kokkos)
find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)
# 同时覆盖 LAMMPS 的兼容布局和 KOKKOS_NO_LEGACY 布局。
foreach(target IN ITEMS array_bridge array_bridge_layoutleft)
  add_executable(${target} main.cpp)
  target_include_directories(${target} PRIVATE "${LAMMPS_SOURCE}/src" "${LAMMPS_SOURCE}/src/KOKKOS" "${GPUMD_SOURCE}/src")
  # nvrtc 编译字符串，cuda_driver 加载/启动模块，cudart 管理流和事件。
  # MPI 同时满足 LAMMPS 类型头文件和验证程序的运行环境。
  target_link_libraries(${target} PRIVATE Kokkos::kokkos MPI::MPI_CXX CUDA::nvrtc CUDA::cuda_driver CUDA::cudart)
endforeach()
# 此宏令 LAMMPS 坐标变为 LayoutLeft；力数组仍可保持 LayoutRight，验证混合布局。
target_compile_definitions(array_bridge_layoutleft PRIVATE LMP_KOKKOS_NO_LEGACY)
enable_testing()
add_test(NAME borrowed_gpu_arrays COMMAND array_bridge)
add_test(NAME borrowed_gpu_arrays_layoutleft COMMAND array_bridge_layoutleft)
# 本验证不传输 MPI 数据；避免 OpenMPI 为单进程容器测试预分配 CUDA IPC 池。
set_tests_properties(borrowed_gpu_arrays borrowed_gpu_arrays_layoutleft PROPERTIES ENVIRONMENT "OMPI_MCA_btl=self,tcp")
