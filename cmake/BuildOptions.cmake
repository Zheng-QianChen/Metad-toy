# 编译开关通过 target 传播。编译器由 toolchain/首次配置确定，项目不覆盖用户选择。
# Weave 设备 reverse 为显式可选；默认 Kokkos 构建不要求 Weave 头文件/库。
option(ZQC_LAMMPS_WEAVE "Enable WeaveMD device force/virial reverse for Kokkos batches" OFF)
set(WEAVEMD_BATCH_SOURCE "" CACHE PATH "Directory containing weave_metad.h")
set(WEAVEMD_PLUGIN_LIBRARY "" CACHE FILEPATH "Matching weavebatchplugin shared library")
add_library(zqc_build_options INTERFACE)
if(ENABLE_DEBUG)
    target_compile_definitions(zqc_build_options INTERFACE DEBUG)
    target_compile_options(zqc_build_options INTERFACE
        "$<$<COMPILE_LANGUAGE:CXX>:-g;-O0>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-g;-G>")
endif()
if(ENABLE_ASAN)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(FATAL_ERROR "ENABLE_ASAN currently supports GNU/Clang host compilers")
    endif()
    target_compile_options(zqc_build_options INTERFACE
        "$<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address;-fno-omit-frame-pointer>"
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fsanitize=address,-fno-omit-frame-pointer>")
    target_link_options(zqc_build_options INTERFACE
        "$<$<LINK_LANGUAGE:CXX>:-fsanitize=address>"
        "$<$<LINK_LANGUAGE:CUDA>:-Xcompiler=-fsanitize=address>")
endif()
