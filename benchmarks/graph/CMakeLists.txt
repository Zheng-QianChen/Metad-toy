# Host 契约测试独立于 CUDA/MPI/LAMMPS，可以先验证 view 和诊断接口。
foreach(test_pair "zqc_array_view_test;array_view.cpp"
                  "zqc_diagnostics_test;diagnostics.cpp" "zqc_engine_contract;engine_contract.cpp")
    list(GET test_pair 0 test_target)
    list(GET test_pair 1 test_source)
    add_executable(${test_target} ${test_source})
    target_link_libraries(${test_target} PRIVATE zqc_views zqc_build_options)
    add_test(NAME ${test_target} COMMAND ${test_target})
    set_tests_properties(${test_target} PROPERTIES LABELS "host")
endforeach()
add_executable(zqc_hill_bias_test hill_bias.cpp)
target_link_libraries(zqc_hill_bias_test PRIVATE zqc_bias zqc_build_options)
add_test(NAME zqc_hill_bias COMMAND zqc_hill_bias_test)
set_tests_properties(zqc_hill_bias PROPERTIES LABELS "host;sampling;finite-difference")
add_executable(zqc_bias_grid_test bias_grid.cpp)
target_link_libraries(zqc_bias_grid_test PRIVATE zqc_bias zqc_build_options)
add_test(NAME zqc_bias_grid COMMAND zqc_bias_grid_test)
set_tests_properties(zqc_bias_grid PROPERTIES LABELS "host;sampling;finite-difference")
add_executable(zqc_bias_session_test bias_session.cpp)
target_link_libraries(zqc_bias_session_test PRIVATE zqc_bias zqc_build_options)
add_test(NAME zqc_bias_session COMMAND zqc_bias_session_test)
set_tests_properties(zqc_bias_session PROPERTIES LABELS "host;sampling;finite-difference")
add_executable(zqc_bias_rebuild_benchmark ../highdim/rebuild_benchmark.cpp)
target_link_libraries(zqc_bias_rebuild_benchmark PRIVATE zqc_bias zqc_build_options)
add_library(zqc_bias_probe SHARED ../highdim/bias_probe.cpp)
target_link_libraries(zqc_bias_probe PRIVATE zqc_bias zqc_build_options)
# Shared-bias equilibrium validation is invoked explicitly with probability.py.
# Its C++ sampler reuses HillBias; the CUDA bridge checks the production table.
add_library(zqc_shared_probability_probe SHARED ../shared_frozen/probability_probe.cpp)
target_link_libraries(zqc_shared_probability_probe PRIVATE zqc_bias zqc_build_options)
# Probability-compression experiment, run explicitly via compression.py.
add_library(zqc_shared_compression_probe SHARED ../shared_frozen/compression_probe.cpp)
target_link_libraries(zqc_shared_compression_probe PRIVATE zqc_bias zqc_build_options)
if(NOT ZQC_ENABLE_CUDA)
    return()
endif()
add_library(zqc_shared_probability_cuda SHARED ../shared_frozen/probability_cuda.cu)
target_link_libraries(zqc_shared_probability_cuda PRIVATE zqc_shared_bias)


# 独立性能评估原型，按需运行；不进入生产偏势执行路径。
add_executable(zqc_bias_evaluate_benchmark ../highdim/evaluate_benchmark.cu)
target_link_libraries(zqc_bias_evaluate_benchmark PRIVATE CUDA::cudart zqc_build_options)

if(TARGET zqc_cvs)
    add_executable(zqc_production_cv_views production_cv_views.cu)
    target_link_libraries(zqc_production_cv_views PRIVATE zqc_cvs)
    add_test(NAME zqc_production_cv_views COMMAND zqc_production_cv_views)
    set_tests_properties(zqc_production_cv_views PROPERTIES TIMEOUT 120 LABELS "cuda;cv;finite-difference")
endif()

function(zqc_graph_test target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE zqc_graph_runtime zqc_cv_assembly)
endfunction()
zqc_graph_test(zqc_graph_smoke graph_smoke.cu)
zqc_graph_test(zqc_graph_sw_test graph_sw_func.cu graph_extension.cu)
target_link_libraries(zqc_graph_sw_test PRIVATE zqc_fmt)
zqc_graph_test(zqc_graph_emit ../graph_kernel_20261013/emit_graphs.cu)
zqc_graph_test(zqc_cv_assembly_test cv_assembly.cu)
# 性能入口按需运行；常规 CTest 只承担正确性验证。
zqc_graph_test(zqc_leaf_layout_benchmark ../runtime_performance/leaf_layout.cu)
find_package(Threads REQUIRED)
zqc_graph_test(zqc_stream_contract stream_contract.cu)
target_link_libraries(zqc_stream_contract PRIVATE Threads::Threads)
add_test(NAME zqc_stream_contract COMMAND zqc_stream_contract)
set_tests_properties(zqc_stream_contract PROPERTIES TIMEOUT 30 LABELS "cuda;streams")
add_test(NAME zqc_graph_nvrtc_smoke COMMAND zqc_graph_smoke)
add_test(NAME zqc_graph_sw_func COMMAND zqc_graph_sw_test)
add_test(NAME zqc_cv_assembly COMMAND zqc_cv_assembly_test)

# 生产设备算法的独立数值测试，与宿主配置/注册层分别编译。
add_library(zqc_validation_kernels STATIC
    "${PROJECT_SOURCE_DIR}/src/cvs/CVs_Strufactor.cu"
    "${PROJECT_SOURCE_DIR}/src/cvs/CVs_CSS.cu")
target_include_directories(zqc_validation_kernels PUBLIC "${PROJECT_SOURCE_DIR}/include/CVsets")
target_compile_definitions(zqc_validation_kernels PUBLIC ZQC_STRUFACTOR_KERNELS_ONLY ZQC_CSS_KERNELS_ONLY)
target_link_libraries(zqc_validation_kernels PUBLIC zqc_views zqc_build_options CUDA::cudart)
zqc_graph_test(zqc_css_test css.cu)
target_link_libraries(zqc_css_test PRIVATE zqc_validation_kernels)
add_test(NAME zqc_css_kernel COMMAND zqc_css_test)
add_executable(zqc_engine_geometry engine_geometry.cu)
target_link_libraries(zqc_engine_geometry PRIVATE zqc_validation_kernels)
add_test(NAME zqc_engine_geometry COMMAND zqc_engine_geometry)

foreach(buffer_variant release debug)
    add_executable(zqc_gpu_buffer_${buffer_variant} gpu_buffer.cu)
    target_link_libraries(zqc_gpu_buffer_${buffer_variant} PRIVATE zqc_views CUDA::cudart)
    if(buffer_variant STREQUAL "debug")
        target_compile_definitions(zqc_gpu_buffer_${buffer_variant} PRIVATE DEBUG)
    endif()
    add_test(NAME zqc_gpu_buffer_${buffer_variant} COMMAND zqc_gpu_buffer_${buffer_variant})
endforeach()

# MPI 依赖仅注入分布式测试。关闭后仍能编译和验证 Graph/NVRTC。
if(ZQC_TEST_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    zqc_graph_test(zqc_graph_ave_test graph_ave_mpi.cu)
    zqc_graph_test(zqc_graph_sum_safe_div_test graph_sum_safe_div.cu)
    zqc_graph_test(zqc_graph_fd_test ../finite_difference/graph.cu)
    zqc_graph_test(zqc_strufactor_test strufactor_graph.cu)
    zqc_graph_test(zqc_strufactor_chem_test strufactor_graph.cu)
    zqc_graph_test(zqc_weighted_chem_test strufactor_graph.cu)
    target_compile_definitions(zqc_strufactor_chem_test PRIVATE ZQC_TEST_CHEMICAL_WEIGHTS)
    target_compile_definitions(zqc_weighted_chem_test PRIVATE ZQC_TEST_WEIGHTED_CHEM)
    foreach(target zqc_strufactor_test zqc_strufactor_chem_test zqc_weighted_chem_test)
        target_link_libraries(${target} PRIVATE zqc_validation_kernels)
    endforeach()
    foreach(target zqc_graph_ave_test zqc_graph_sum_safe_div_test zqc_graph_fd_test
                   zqc_strufactor_test zqc_strufactor_chem_test zqc_weighted_chem_test)
        target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
        target_link_libraries(${target} PRIVATE MPI::MPI_CXX)
    endforeach()
    foreach(test_pair "zqc_graph_ave;zqc_graph_ave_test" "zqc_graph_sum_safe_div;zqc_graph_sum_safe_div_test"
                      "zqc_graph_fd;zqc_graph_fd_test"
                      "zqc_strufactor;zqc_strufactor_test" "zqc_strufactor_chem;zqc_strufactor_chem_test"
                      "zqc_weighted_chem;zqc_weighted_chem_test")
        list(GET test_pair 0 test_name)
        list(GET test_pair 1 test_target)
        foreach(process_count 1 2 4)
            add_test(NAME ${test_name}_mpi_${process_count}
                COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${process_count} ${MPIEXEC_PREFLAGS}
                    $<TARGET_FILE:${test_target}> ${MPIEXEC_POSTFLAGS})
            set_tests_properties(${test_name}_mpi_${process_count} PROPERTIES
                TIMEOUT 120 PROCESSORS ${process_count} LABELS "mpi;cuda")
        endforeach()
    endforeach()
endif()

# 混合 Global/PerAtom 输出与冻结共享表的独立正确性验证。
zqc_graph_test(zqc_shared_frozen_test shared_frozen.cu)
target_link_libraries(zqc_shared_frozen_test PRIVATE zqc_shared_bias)
add_test(NAME zqc_shared_frozen COMMAND zqc_shared_frozen_test)
set_tests_properties(zqc_shared_frozen PROPERTIES LABELS "cuda;sampling;finite-difference" TIMEOUT 90)
if(ZQC_TEST_MPI)
    zqc_graph_test(zqc_shared_online_test shared_online.cu)
    target_link_libraries(zqc_shared_online_test PRIVATE zqc_shared_bias MPI::MPI_CXX)
    foreach(process_count 1 2 4)
        add_test(NAME zqc_shared_online_mpi_${process_count}
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${process_count} ${MPIEXEC_PREFLAGS}
                $<TARGET_FILE:zqc_shared_online_test> ${MPIEXEC_POSTFLAGS})
        set_tests_properties(zqc_shared_online_mpi_${process_count} PROPERTIES
            TIMEOUT 120 PROCESSORS ${process_count} LABELS "mpi;cuda;sampling;finite-difference")
    endforeach()
endif()
