add_library(zqc_lammps_adapter OBJECT
    zqc_lammps_engine.cu zqc_lammps_diagnostics.cpp neigh_hub.cu)
target_link_libraries(zqc_lammps_adapter PRIVATE zqc_lammps_dependencies
    zqc_views zqc_build_options CUDA::cudart)

if(ZQC_LAMMPS_KOKKOS)
    target_sources(zqc_lammps_adapter PRIVATE zqc_lammps_kokkos.cu)
    target_compile_definitions(zqc_lammps_dependencies INTERFACE ZQC_LAMMPS_KOKKOS=1)
    target_include_directories(zqc_lammps_adapter PRIVATE "${LAMMPS_SOURCE_DIR}/KOKKOS")
    # Kokkos 运行时由 liblammps 唯一持有。只传播该构建的头文件/宏；
    # 将静态 kokkoscore 再链接进插件会重复注册全局对象并在退出时双重释放。
    foreach(kokkos_target kokkos kokkoscore kokkoscontainers kokkosalgorithms kokkossimd)
        if(TARGET Kokkos::${kokkos_target})
            target_include_directories(zqc_lammps_adapter PRIVATE
                $<TARGET_PROPERTY:Kokkos::${kokkos_target},INTERFACE_INCLUDE_DIRECTORIES>)
            target_compile_definitions(zqc_lammps_adapter PRIVATE
                $<TARGET_PROPERTY:Kokkos::${kokkos_target},INTERFACE_COMPILE_DEFINITIONS>)
        endif()
    endforeach()
    target_compile_options(zqc_lammps_adapter PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
endif()

add_library(fix_crystallize_plugin MODULE fix_crystallize.cu compute_MetaDToy.cu)
target_include_directories(fix_crystallize_plugin PRIVATE "${PROJECT_SOURCE_DIR}/include/ml_cvs")
target_link_libraries(fix_crystallize_plugin PRIVATE zqc_lammps_dependencies
    zqc_lammps_adapter zqc_sampling zqc_cvs zqc_cv_graph zqc_shared_bias zqc_build_options)
if(TARGET zqc_ml_cvs)
    target_link_libraries(fix_crystallize_plugin PRIVATE zqc_ml_cvs)
endif()
# 保持用户现有 plugin load 路径：产物位于构建目录根部。
set_target_properties(fix_crystallize_plugin PROPERTIES PREFIX "" SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

if(ZQC_LAMMPS_KOKKOS)
    # 在链接阶段确认宿主导出了桥接所需的 Kokkos 符号。
    target_link_options(fix_crystallize_plugin PRIVATE "-Wl,-z,defs")
endif()

# 直接读取 Host 力的邻居契约探针需要相同 Kokkos 头文件，默认构建不编译它。
if(ZQC_LAMMPS_KOKKOS)
    add_library(zqc_kokkos_neighbors_probe MODULE EXCLUDE_FROM_ALL
        "../../benchmarks/finite_difference/engine_neighbors_probe.cu")
    target_include_directories(zqc_kokkos_neighbors_probe PRIVATE
        $<TARGET_PROPERTY:zqc_lammps_adapter,INCLUDE_DIRECTORIES>)
    target_compile_definitions(zqc_kokkos_neighbors_probe PRIVATE
        $<TARGET_PROPERTY:zqc_lammps_adapter,COMPILE_DEFINITIONS>)
    target_compile_options(zqc_kokkos_neighbors_probe PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    target_link_libraries(zqc_kokkos_neighbors_probe PRIVATE zqc_lammps_dependencies zqc_views
        CUDA::cudart "-l:fix_crystallize_plugin.so")
    target_link_directories(zqc_kokkos_neighbors_probe PRIVATE "${PROJECT_BINARY_DIR}")
    add_dependencies(zqc_kokkos_neighbors_probe fix_crystallize_plugin)
    set_target_properties(zqc_kokkos_neighbors_probe PROPERTIES PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
endif()


# 按需构建真实适配器的批次探针；CUDA/Kokkos 头文件与插件使用同一份 ABI 配置。
add_library(zqc_force_batch_probe MODULE EXCLUDE_FROM_ALL
    "../../benchmarks/finite_difference/force_batch_probe.cu")
target_include_directories(zqc_force_batch_probe PRIVATE
    $<TARGET_PROPERTY:zqc_lammps_adapter,INCLUDE_DIRECTORIES>)
target_compile_definitions(zqc_force_batch_probe PRIVATE
    $<TARGET_PROPERTY:zqc_lammps_adapter,COMPILE_DEFINITIONS>)
target_compile_options(zqc_force_batch_probe PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_link_libraries(zqc_force_batch_probe PRIVATE zqc_lammps_dependencies zqc_views
    CUDA::cudart "-l:fix_crystallize_plugin.so")
target_link_directories(zqc_force_batch_probe PRIVATE "${PROJECT_BINARY_DIR}")
add_dependencies(zqc_force_batch_probe fix_crystallize_plugin)
set_target_properties(zqc_force_batch_probe PROPERTIES PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")
