include("${PROJECT_SOURCE_DIR}/cmake/GraphDependencies.cmake")

# 设备函数源码只嵌入一次。修改 .cuh 后，普通 CUDA 与 NVRTC 文本自动同步。
set(graph_device_header "${PROJECT_SOURCE_DIR}/include/utils/zqc_graph_device.cuh")
set(graph_embed_source "${CMAKE_CURRENT_BINARY_DIR}/embed/zqc_graph_device.cpp")
add_custom_command(OUTPUT "${graph_embed_source}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/embed"
    COMMAND "${CMAKE_COMMAND}" "-DXXD=${XXD}" "-DINPUT=${graph_device_header}"
        "-DSYMBOL=zqc_graph_device_source" "-DOUTPUT=${graph_embed_source}"
        -P "${PROJECT_SOURCE_DIR}/cmake/xxd_embed.cmake"
    DEPENDS "${graph_device_header}" "${PROJECT_SOURCE_DIR}/cmake/xxd_embed.cmake"
    VERBATIM)
add_library(zqc_graph_embedded STATIC "${graph_embed_source}")
target_link_libraries(zqc_graph_embedded PRIVATE zqc_build_options)

add_library(zqc_graph_runtime STATIC
    zqc_graph.cu zqc_graph_parser.cu zqc_graph_nodes.cu zqc_graph_execution.cu)
add_library(MetadToy::graph ALIAS zqc_graph_runtime)
target_link_libraries(zqc_graph_runtime PUBLIC zqc_views zqc_build_options
    CUDA::cudart CUDA::cuda_driver CUDA::nvrtc
    PRIVATE zqc_graph_embedded zqc_fmt)

add_library(zqc_cv_assembly STATIC zqc_cv_assembly.cu)
target_link_libraries(zqc_cv_assembly PUBLIC zqc_views zqc_build_options CUDA::cudart)

# 桥接只使用抽象 CV 接口；Graph 验证可用 mock CV 单独构建此层。
add_library(zqc_cv_graph STATIC ../core/zqc_cv_graph.cu)
target_include_directories(zqc_cv_graph PUBLIC "${PROJECT_SOURCE_DIR}/include/core")
target_link_libraries(zqc_cv_graph PUBLIC zqc_graph_runtime zqc_cv_assembly)

# 冻结的原子共享偏势只依赖 View/CUDA/规约回调，可由各 MD 引擎复用。
add_library(zqc_shared_bias STATIC ../sampling/zqc_shared_bias.cu)
target_link_libraries(zqc_shared_bias PUBLIC zqc_views zqc_build_options CUDA::cudart)
