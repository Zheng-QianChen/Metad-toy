# OBJECT 保留每种 CV 的注册构造函数；插件链接时全部纳入，避免静态库丢掉注册项。
add_library(zqc_cvs OBJECT
    CVs_DISTANCE.cu CVs_Strufactor.cu CVs_Strufactor_chem.cu CVs_CSS.cu
    CV_Weighted_Chemical_Pair_Order.cu CVs_Steinhardt.cu CVs_Steinhardt_Local.cu
    ../core/CVFactory.cu)
target_include_directories(zqc_cvs PUBLIC "${PROJECT_SOURCE_DIR}/include/CVsets")
target_link_libraries(zqc_cvs PUBLIC zqc_views zqc_cv_assembly zqc_build_options CUDA::cudart)
