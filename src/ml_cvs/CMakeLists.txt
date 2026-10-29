# ML 为显式可选层，只有启用时查找 ONNX Runtime 和 cuDNN。
find_path(ORT_INC onnxruntime_cxx_api.h HINTS "${ORT_HOME}/include" "$ENV{ORT_HOME}/include")
find_library(ORT_LIB onnxruntime HINTS "${ORT_HOME}/lib" "$ENV{ORT_HOME}/lib")
find_path(CUDNN_INC cudnn.h HINTS "$ENV{CONDA_PREFIX}/include")
find_library(CUDNN_LIB cudnn HINTS "$ENV{CONDA_PREFIX}/lib")
if(NOT ORT_INC OR NOT ORT_LIB OR NOT CUDNN_INC OR NOT CUDNN_LIB)
    message(FATAL_ERROR "USE_ML_CVS requires ONNX Runtime and cuDNN (ORT_HOME/CMAKE_PREFIX_PATH)")
endif()
add_library(zqc_ml_cvs OBJECT mlcvs.cu)
target_include_directories(zqc_ml_cvs PRIVATE "${ORT_INC}" "${CUDNN_INC}")
target_compile_definitions(zqc_ml_cvs PUBLIC USE_ML_CVS)
target_link_libraries(zqc_ml_cvs PUBLIC "${ORT_LIB}" "${CUDNN_LIB}"
    PRIVATE zqc_build_options CUDA::cudart)
