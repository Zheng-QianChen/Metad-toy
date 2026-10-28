# 兼容表达式与 CV 工具桥接；偏势数值、历史和生命周期位于纯 C++ zqc_bias。
add_library(zqc_sampling OBJECT DimSet.cu
    ../utils/CVs_tools.cu)
target_include_directories(zqc_sampling PRIVATE "${PROJECT_SOURCE_DIR}/third_party")
target_link_libraries(zqc_sampling PRIVATE zqc_lammps_dependencies zqc_views
    zqc_build_options zqc_bias CUDA::cudart)
