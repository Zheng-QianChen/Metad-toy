# 复用现有 GpuBuffer 的 LAMMPS/MPI 类型声明，不构建 LAMMPS 插件。
find_path(ZQC_GRAPH_FMT_INCLUDE_DIR fmt/printf.h
          HINTS "${LAMMPS_SOURCE_DIR}" "$ENV{LAMMPS_SOURCE_DIR}")
if(NOT ZQC_GRAPH_FMT_INCLUDE_DIR)
    message(FATAL_ERROR "Set ZQC_GRAPH_FMT_INCLUDE_DIR to the directory containing fmt/printf.h")
endif()
find_path(ZQC_GRAPH_LAMMPS_INCLUDE_DIR lammps.h
          HINTS "${LAMMPS_SOURCE_DIR}" "$ENV{LAMMPS_SOURCE_DIR}"
                "${ZQC_GRAPH_FMT_INCLUDE_DIR}")
if(NOT ZQC_GRAPH_LAMMPS_INCLUDE_DIR)
    message(FATAL_ERROR "Set ZQC_GRAPH_LAMMPS_INCLUDE_DIR to the LAMMPS src directory")
endif()
find_package(MPI REQUIRED COMPONENTS CXX)
add_executable(zqc_graph_smoke graph_smoke.cu)
add_executable(zqc_graph_sw_test graph_sw_func.cu)
foreach(graph_target zqc_graph_smoke zqc_graph_sw_test)
    target_include_directories(${graph_target} PRIVATE
        "${PROJECT_SOURCE_DIR}/include/utils"
        "${ZQC_GRAPH_FMT_INCLUDE_DIR}"
        "${ZQC_GRAPH_LAMMPS_INCLUDE_DIR}")
    target_compile_definitions(${graph_target} PRIVATE FMT_HEADER_ONLY)
    set_target_properties(${graph_target} PROPERTIES
        CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_link_libraries(${graph_target} PRIVATE
        zqc_graph_embedded CUDA::cudart CUDA::cuda_driver CUDA::nvrtc MPI::MPI_CXX)
endforeach()
add_test(NAME zqc_graph_nvrtc_smoke COMMAND zqc_graph_smoke)
add_test(NAME zqc_graph_sw_func COMMAND zqc_graph_sw_test)
