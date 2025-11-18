cmake_minimum_required(VERSION 3.10)
project(fix_crystallize_plugin LANGUAGES CXX CUDA)  # 声明支持 C++ 和 CUDA [3,6](@ref)

# 设置 C++ 和 CUDA 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_C_COMPILER   /home/excelsior/apps/deepmd/3.0.0/bin/mpicc)
set(CMAKE_CXX_COMPILER /home/excelsior/apps/deepmd/3.0.0/bin/mpicxx)

# ABI=1
#set(_GLIBCXX_USE_CXX11_ABI 1)
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-D_GLIBCXX_USE_CXX11_ABI=1")
#target_compile_definitions(fix_crystallize_plugin PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

# 设置LAMMPS源码路径（需修改为实际路径）
set(MPI_CXX_INCLUDE_PATH /home/excelsior/apps/deepmd/3.0.0/include)
set(LAMMPS_SOURCE_DIR /home/excelsior/apps/_compile/lammps-29Aug2024/src/)

# 包含LAMMPS头文件
include_directories(${LAMMPS_SOURCE_DIR})
include_directories(${MPI_CXX_INCLUDE_PATH})
link_directories(/home/excelsior/apps/deepmd/3.0.0/lib/)
# link_directories(/home/excelsior/apps/lammps/29Aug2024/lib64/)

# 编译插件为动态库
add_library(fix_crystallize_plugin MODULE fix_crystallize.cu)
target_compile_definitions(fix_crystallize_plugin PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
target_link_libraries(fix_crystallize_plugin PRIVATE lammps)

# 设置输出文件名格式（避免前缀）
set_target_properties(fix_crystallize_plugin PROPERTIES 
CUDA_ARCHITECTURES "89"
PREFIX "" SUFFIX ".so")
