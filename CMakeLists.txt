cmake_minimum_required(VERSION 3.10)
project(fix_crystallize_plugin LANGUAGES CXX CUDA)

# -------------------------------------------------------------------
# 模式切换开关：默认关闭调试
# 使用时：cmake -DENABLE_DEBUG=ON ..
# -------------------------------------------------------------------
option(ENABLE_DEBUG "Enable debug mode with GDB support and line info" ON)
# option(ENABLE_DEBUG "Enable debug mode with GDB support and line info" OFF)
option(USE_ML_CVS   "Use Machine Learning models as Collective Variables" ON)
option(ENABLE_ASAN  "Enable AddressSanitizer for host-side memory checking" OFF)   # 新加这一行

# -------------------------------------------------------------------
# 1. 路径配置（通过环境变量获取，不再硬编码）
# -------------------------------------------------------------------

# 使用 "$ENV{变量名}" 获取本地环境，如果环境没设置则报错
if(NOT LAMMPS_SOURCE_DIR AND NOT DEFINED ENV{LAMMPS_SOURCE_DIR})
    message(FATAL_ERROR "please set the ENV: export LAMMPS_SOURCE_DIR=/path/to/lammps/src")
endif()

set(LAMMPS_SOURCE_DIR ${LAMMPS_SOURCE_DIR})
set(MPI_INCLUDE_PATH  ${MPI_HOME}/include) 

# -------------------------------------------------------------------
# 2. 编译器与标准设置
# -------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_C_COMPILER   ${MPI_HOME}/bin/mpicc)
set(CMAKE_CXX_COMPILER ${MPI_HOME}/bin/mpicxx)

set(CMAKE_CUDA_HOST_COMPILER ${CLEAN_HOST_COMPILER})
set(CUDA_NVCC_FLAGS -ccbin ${CLEAN_HOST_COMPILER})

set(ABI_DEFINITION -D_GLIBCXX_USE_CXX11_ABI=1)

# -------------------------------------------------------------------
# 3. 包含路径与库
# -------------------------------------------------------------------
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party)
include_directories(${LAMMPS_SOURCE_DIR})
include_directories(${MPI_INCLUDE_PATH})

# -------------------------------------------------------------------
# 4. 自动识别源文件
# -------------------------------------------------------------------
# 自动寻找 src 目录下的所有 cu 文件，方便未来扩展 CV
# file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cu")
# file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*/*.cu")
# file(GLOB_RECURSE CHILDREN LIST_DIRECTORIES true ${CMAKE_CURRENT_SOURCE_DIR}/include/*)
# foreach(child ${CHILDREN})
#     if(IS_DIRECTORY ${child})
#         include_directories(${child})
#         message(STATUS "Include path added: ${child}")
#     endif()
# endforeach()
# 创建一个全局列表，用于收集所有子包的源文件
set_property(GLOBAL PROPERTY PLUGIN_SOURCES "")
set_property(GLOBAL PROPERTY PLUGIN_INCLUDES "")
set_property(GLOBAL PROPERTY PLUGIN_LIBS "")

# 依次添加子包
add_subdirectory(src/cvs)
add_subdirectory(src/grids)
add_subdirectory(src/core)
add_subdirectory(src/utils)
if (USE_ML_CVS)
    add_subdirectory(src/ml_cvs)
endif()

# 包含收集到的所有头文件路径
# include_directories(${PLUGIN_INCLUDES})

get_property(FINAL_SOURCES GLOBAL PROPERTY PLUGIN_SOURCES)
get_property(FINAL_INCLUDES GLOBAL PROPERTY PLUGIN_INCLUDES)
get_property(FINAL_LIBS    GLOBAL PROPERTY PLUGIN_LIBS)

include_directories(${FINAL_INCLUDES})

# -------------------------------------------------------------------
# 5. 编译目标
# -------------------------------------------------------------------
# 找lammps库文件，路径由环境变量 LAMMPS_LIB_DIR 指定
find_library(LAMMPS_LIB lammps PATHS ${LAMMPS_LIB_DIR} NO_DEFAULT_PATH)
# add_library(fix_crystallize_plugin MODULE ${SOURCES})
add_library(fix_crystallize_plugin MODULE ${FINAL_SOURCES})
if(LAMMPS_LIB)
    message(STATUS "linking lammps lib path is: ${LAMMPS_LIB}")
    target_link_libraries(fix_crystallize_plugin PRIVATE ${LAMMPS_LIB} ${FINAL_LIBS})
else()
    message(FATAL_ERROR "cannot find liblammps.so, please check LAMMPS_LIB_DIR")
endif()

# AddressSanitizer 注入（仅作用于 host 端 C++ 代码，CUDA device 代码无效）
if(ENABLE_ASAN)
    message(STATUS "AddressSanitizer: ENABLED (host-side C++ only)")

    target_compile_options(fix_crystallize_plugin PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address -fno-omit-frame-pointer -g>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fsanitize=address,-fno-omit-frame-pointer>
    )

    target_link_options(fix_crystallize_plugin PRIVATE -fsanitize=address)
endif()

# 注入ABI定义，确保与LAMMPS库兼容
target_compile_options(fix_crystallize_plugin PRIVATE ${ABI_DEFINITION})

# 调试与优化标志注入
if(ENABLE_DEBUG)
    message(STATUS "Build Mode: DEBUG (with Nsight support)")
    
    # 1. 给 C++ 代码注入宏定义，相当于在头文件写了 #define PLUGIN_DEBUG
    target_compile_definitions(fix_crystallize_plugin PRIVATE DEBUG)
    
    # 2. 注入调试专用的编译器标志
    target_compile_options(fix_crystallize_plugin PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-g -G --generate-line-info> 
        $<$<COMPILE_LANGUAGE:CXX>:-g -O0>
    )
else()
    message(STATUS "Build Mode: RELEASE (Optimized)")
    
    # 优化标志
    target_compile_options(fix_crystallize_plugin PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --generate-line-info> 
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
    )
endif()

# 显卡架构配置（建议由用户从命令行传入，如 -DCMAKE_CUDA_ARCHITECTURES=89）
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set_target_properties(fix_crystallize_plugin PROPERTIES CUDA_ARCHITECTURES "89")
endif()

set_target_properties(fix_crystallize_plugin PROPERTIES PREFIX "" SUFFIX ".so")