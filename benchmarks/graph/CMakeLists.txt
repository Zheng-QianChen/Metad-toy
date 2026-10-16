# The standalone runtime links graph, parser and node implementations once.
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

add_library(zqc_graph_runtime STATIC
    "${PROJECT_SOURCE_DIR}/src/utils/zqc_graph.cu"
    "${PROJECT_SOURCE_DIR}/src/utils/zqc_graph_parser.cu"
    "${PROJECT_SOURCE_DIR}/src/utils/zqc_graph_nodes.cu"
    "${PROJECT_SOURCE_DIR}/src/utils/zqc_graph_execution.cu"
    "${PROJECT_SOURCE_DIR}/src/utils/zqc_cv_assembly.cu")
target_include_directories(zqc_graph_runtime PUBLIC
    "${PROJECT_SOURCE_DIR}/include/utils"
    "${ZQC_GRAPH_FMT_INCLUDE_DIR}"
    "${ZQC_GRAPH_LAMMPS_INCLUDE_DIR}")
target_compile_definitions(zqc_graph_runtime PUBLIC FMT_HEADER_ONLY)
target_link_libraries(zqc_graph_runtime PUBLIC
    zqc_graph_embedded CUDA::cudart CUDA::cuda_driver CUDA::nvrtc MPI::MPI_CXX)

add_executable(zqc_graph_smoke graph_smoke.cu)
add_executable(zqc_graph_sw_test graph_sw_func.cu graph_extension.cu)
add_executable(zqc_graph_ave_test graph_ave_mpi.cu)
add_executable(zqc_cv_assembly_test cv_assembly.cu)
add_executable(zqc_strufactor_test strufactor_graph.cu
    "${PROJECT_SOURCE_DIR}/src/cvs/CVs_Strufactor.cu")
target_include_directories(zqc_strufactor_test PRIVATE "${PROJECT_SOURCE_DIR}/include/CVsets")
# 从合并后的生产源文件编译设备计算部分，供独立有限差分验证。
target_compile_definitions(zqc_strufactor_test PRIVATE ZQC_STRUFACTOR_KERNELS_ONLY)
add_executable(zqc_graph_emit ../graph_kernel_20261013/emit_graphs.cu)
foreach(graph_target zqc_graph_runtime zqc_graph_smoke zqc_graph_sw_test zqc_graph_emit zqc_graph_ave_test zqc_cv_assembly_test zqc_strufactor_test)
    set_target_properties(${graph_target} PROPERTIES
        CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
endforeach()
foreach(graph_target zqc_graph_smoke zqc_graph_sw_test zqc_graph_emit zqc_graph_ave_test zqc_cv_assembly_test zqc_strufactor_test)
    target_link_libraries(${graph_target} PRIVATE zqc_graph_runtime)
endforeach()
add_test(NAME zqc_graph_nvrtc_smoke COMMAND zqc_graph_smoke)
add_test(NAME zqc_graph_sw_func COMMAND zqc_graph_sw_test)
add_test(NAME zqc_cv_assembly COMMAND zqc_cv_assembly_test)

foreach(np 1 2 4)
    add_test(NAME zqc_graph_ave_mpi_${np}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:zqc_graph_ave_test> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(zqc_graph_ave_mpi_${np} PROPERTIES TIMEOUT 60 PROCESSORS ${np})
    add_test(NAME zqc_strufactor_mpi_${np}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:zqc_strufactor_test> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(zqc_strufactor_mpi_${np} PROPERTIES TIMEOUT 60 PROCESSORS ${np})
endforeach()

# CSS 生产 kernel 的配位数、力和 virial 有限差分验证。
add_executable(zqc_css_test css.cu
    "${PROJECT_SOURCE_DIR}/src/cvs/CVs_CSS.cu"
    "${PROJECT_SOURCE_DIR}/src/cvs/CVs_Strufactor.cu")
target_include_directories(zqc_css_test PRIVATE "${PROJECT_SOURCE_DIR}/include/CVsets")
target_compile_definitions(zqc_css_test PRIVATE ZQC_CSS_KERNELS_ONLY ZQC_STRUFACTOR_KERNELS_ONLY)
set_target_properties(zqc_css_test PROPERTIES CXX_STANDARD 17 CUDA_STANDARD 17)
target_link_libraries(zqc_css_test PRIVATE zqc_graph_runtime)
add_test(NAME zqc_css_kernel COMMAND zqc_css_test)

# 化学版复用普通版的独立参考公式与 Graph 测试，覆盖带符号/零权重及空 rank。
add_executable(zqc_strufactor_chem_test strufactor_graph.cu
    "${PROJECT_SOURCE_DIR}/src/cvs/CVs_Strufactor.cu")
target_include_directories(zqc_strufactor_chem_test PRIVATE "${PROJECT_SOURCE_DIR}/include/CVsets")
target_compile_definitions(zqc_strufactor_chem_test PRIVATE ZQC_STRUFACTOR_KERNELS_ONLY ZQC_TEST_CHEMICAL_WEIGHTS)
set_target_properties(zqc_strufactor_chem_test PROPERTIES CXX_STANDARD 17 CUDA_STANDARD 17)
target_link_libraries(zqc_strufactor_chem_test PRIVATE zqc_graph_runtime)
foreach(np 1 2 4)
    add_test(NAME zqc_strufactor_chem_mpi_${np}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:zqc_strufactor_chem_test> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(zqc_strufactor_chem_mpi_${np} PROPERTIES TIMEOUT 60 PROCESSORS ${np})
endforeach()

# WEIGHT_CHEM 使用同一 Graph/有限差分夹具，额外检查逐原子 virial 与平坦区。
add_executable(zqc_weighted_chem_test strufactor_graph.cu
    "${PROJECT_SOURCE_DIR}/src/cvs/CVs_Strufactor.cu")
target_include_directories(zqc_weighted_chem_test PRIVATE "${PROJECT_SOURCE_DIR}/include/CVsets")
target_compile_definitions(zqc_weighted_chem_test PRIVATE ZQC_STRUFACTOR_KERNELS_ONLY ZQC_TEST_WEIGHTED_CHEM)
set_target_properties(zqc_weighted_chem_test PROPERTIES CXX_STANDARD 17 CUDA_STANDARD 17)
target_link_libraries(zqc_weighted_chem_test PRIVATE zqc_graph_runtime)
foreach(np 1 2 4)
    add_test(NAME zqc_weighted_chem_mpi_${np}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:zqc_weighted_chem_test> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(zqc_weighted_chem_mpi_${np} PROPERTIES TIMEOUT 60 PROCESSORS ${np})
endforeach()

# Graph 的逐 Leaf Jacobian、多输出 VJP 和 Host/NVRTC/MPI 双边有限差分。
add_executable(zqc_graph_fd_test ../finite_difference/graph.cu)
set_target_properties(zqc_graph_fd_test PROPERTIES CXX_STANDARD 17 CUDA_STANDARD 17)
target_link_libraries(zqc_graph_fd_test PRIVATE zqc_graph_runtime)
foreach(np 1 2 4)
    add_test(NAME zqc_graph_fd_mpi_${np}
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:zqc_graph_fd_test> ${MPIEXEC_POSTFLAGS})
    set_tests_properties(zqc_graph_fd_mpi_${np} PROPERTIES TIMEOUT 120 PROCESSORS ${np})
endforeach()
