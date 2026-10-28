add_library(zqc_gpumd_engine STATIC zqc_gpumd_engine.cu zqc_gpumd_neighbors.cu)
target_include_directories(zqc_gpumd_engine PUBLIC "${PROJECT_SOURCE_DIR}/include/adapters")
target_link_libraries(zqc_gpumd_engine PUBLIC zqc_views CUDA::cudart Threads::Threads)
if(ZQC_BUILD_TESTS)
    find_package(Python3 3.8 REQUIRED COMPONENTS Interpreter)
    add_executable(zqc_gpumd_engine_test "${PROJECT_SOURCE_DIR}/benchmarks/gpumd/engine_adapter.cu")
    target_link_libraries(zqc_gpumd_engine_test PRIVATE zqc_gpumd_engine)
    add_executable(zqc_gpumd_neighbors_test "${PROJECT_SOURCE_DIR}/benchmarks/gpumd/neighbors.cu")
    target_link_libraries(zqc_gpumd_neighbors_test PRIVATE zqc_gpumd_engine)
    add_test(NAME zqc_gpumd_neighbors COMMAND zqc_gpumd_neighbors_test)
    set_tests_properties(zqc_gpumd_neighbors PROPERTIES TIMEOUT 90 LABELS "gpumd;cuda")
    add_test(NAME zqc_gpumd_engine COMMAND zqc_gpumd_engine_test)
    set_tests_properties(zqc_gpumd_engine PROPERTIES TIMEOUT 60 LABELS "gpumd;cuda")
endif()

# 算法库只依赖项目自有接口，单独构建时不需要 GPUMD 源码。
add_library(metadtoy SHARED zqc_gpumd_runtime.cu)
target_link_libraries(metadtoy PRIVATE zqc_gpumd_engine zqc_cv_graph zqc_cvs zqc_bias)
target_include_directories(metadtoy PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/adapters>
    $<INSTALL_INTERFACE:include/metadtoy>)
set_target_properties(metadtoy PROPERTIES VERSION 1.0.0 SOVERSION 1
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib"
    CUDA_RUNTIME_LIBRARY Shared)
# Linux 只导出版本化 C 入口，避免内部 C++/CUDA 符号成为部署接口。
if(UNIX AND NOT APPLE)
    target_link_options(metadtoy PRIVATE
        "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/metadtoy.exports")
    set_property(TARGET metadtoy APPEND PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/metadtoy.exports")
endif()
install(TARGETS metadtoy LIBRARY DESTINATION lib COMPONENT runtime)
install(FILES "${PROJECT_SOURCE_DIR}/include/adapters/zqc_metadtoy_api.h" DESTINATION include/metadtoy COMPONENT development)
if(ZQC_BUILD_TESTS)
    add_executable(zqc_gpumd_runtime_abi_test "${PROJECT_SOURCE_DIR}/benchmarks/gpumd/runtime_abi.cu")
    target_link_libraries(zqc_gpumd_runtime_abi_test PRIVATE metadtoy CUDA::cudart)
    add_test(NAME zqc_gpumd_runtime_abi COMMAND zqc_gpumd_runtime_abi_test "${CMAKE_CURRENT_BINARY_DIR}/runtime-abi")
    set_tests_properties(zqc_gpumd_runtime_abi PROPERTIES TIMEOUT 60 LABELS "gpumd;cuda")
endif()
if(NOT ZQC_BUILD_GPUMD)
    return()
endif()

# GPUMD 接入点写入构建目录中的副本，保留上游源码供其他构建复用。
if(NOT EXISTS "${ZQC_GPUMD_SOURCE}/src/measure/enhanced_sampling.cu")
    message(FATAL_ERROR "ZQC_GPUMD_SOURCE must point to GPUMD v5.9.1 sources")
endif()
find_package(Python3 3.8 REQUIRED COMPONENTS Interpreter)
set(gpumd_copy "${CMAKE_CURRENT_BINARY_DIR}/upstream")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/cmake/prepare_gpumd.py"
    --source "${ZQC_GPUMD_SOURCE}" --destination "${gpumd_copy}" RESULT_VARIABLE prepare_status)
if(NOT prepare_status EQUAL 0)
    message(FATAL_ERROR "Could not prepare the GPUMD EnhancedSampling bridge")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/cmake/prepare_gpumd.py")
# 与上游 src/makefile 中 SOURCES_GPUMD 的目录集合一致。
set(gpumd_sources "")
foreach(component main_gpumd minimize phonon integrate mc force measure model utilities)
    file(GLOB component_sources CONFIGURE_DEPENDS "${gpumd_copy}/src/${component}/*.cu")
    list(APPEND gpumd_sources ${component_sources})
endforeach()
add_executable(gpumd_metadtoy ${gpumd_sources} zqc_gpumd_bridge.cu)
target_include_directories(gpumd_metadtoy PRIVATE "${gpumd_copy}/src" "${PROJECT_SOURCE_DIR}/include/adapters")
target_compile_definitions(gpumd_metadtoy PRIVATE ZQC_USE_METADTOY)
target_link_libraries(gpumd_metadtoy PRIVATE metadtoy CUDA::cudart CUDA::cublas CUDA::cusolver CUDA::cufft)
set_target_properties(gpumd_metadtoy PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin"
    CUDA_RUNTIME_LIBRARY Shared BUILD_RPATH "$ORIGIN/../lib" BUILD_RPATH_USE_ORIGIN ON
    INSTALL_RPATH "$ORIGIN/../lib")
install(TARGETS gpumd_metadtoy RUNTIME DESTINATION bin COMPONENT gpumd)
if(ZQC_GPUMD_VALIDATION_MORSE)
    target_sources(gpumd_metadtoy PRIVATE "${PROJECT_SOURCE_DIR}/benchmarks/gpumd/morse_validation.cu")
    target_include_directories(gpumd_metadtoy PRIVATE "${PROJECT_SOURCE_DIR}/benchmarks/gpumd")
    target_compile_definitions(gpumd_metadtoy PRIVATE ZQC_GPUMD_VALIDATION_MORSE)
    if(ZQC_BUILD_TESTS)
        add_test(NAME zqc_gpumd_distance_smoke
            COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/benchmarks/gpumd/run.py"
                --gpumd $<TARGET_FILE:gpumd_metadtoy> --mode smoke)
        add_test(NAME zqc_gpumd_strufactor
            COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/benchmarks/gpumd/strufactor.py"
                --gpumd $<TARGET_FILE:gpumd_metadtoy>)
        set_tests_properties(zqc_gpumd_strufactor PROPERTIES TIMEOUT 240 LABELS "gpumd;cuda;finite-difference")
        set_tests_properties(zqc_gpumd_distance_smoke PROPERTIES TIMEOUT 180 LABELS "gpumd;cuda")
    endif()
endif()
