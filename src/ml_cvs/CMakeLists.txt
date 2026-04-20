# src/cvs/CMakeLists.txt
message(STATUS "Entering ML module: Checking dependencies...")

# --- 1. 环境检查 ---
if(NOT DEFINED ENV{CONDA_PREFIX})
    message(FATAL_ERROR "ML_CVs requires an active Conda environment (CONDA_PREFIX not set).")
endif()
set(CONDA_DIR $ENV{CONDA_PREFIX})

# --- 2. 寻找依赖 ---
find_path(CUDNN_INC cudnn.h PATHS ${CONDA_DIR}/include NO_DEFAULT_PATH)
find_library(CUDNN_LIB cudnn PATHS ${CONDA_DIR}/lib NO_DEFAULT_PATH)

find_path(ORT_INC onnxruntime_cxx_api.h PATHS ${ORT_HOME}/include NO_DEFAULT_PATH)
find_library(ORT_LIB onnxruntime PATHS ${ORT_HOME}/lib NO_DEFAULT_PATH)

# 这里我们还需要添加 ONNX Runtime 和 cuDNN 的头文件路径和库文件路径
set_property(GLOBAL APPEND PROPERTY PLUGIN_INCLUDES "${ORT_INC}")
set_property(GLOBAL APPEND PROPERTY PLUGIN_INCLUDES "${CUDNN_INC}")
set_property(GLOBAL APPEND PROPERTY PLUGIN_LIBS "${ORT_LIB}")
set_property(GLOBAL APPEND PROPERTY PLUGIN_LIBS "${CUDNN_LIB}")

# --- 3. 验证并提交成果 ---
if(CUDNN_INC AND CUDNN_LIB AND ORT_INC AND ORT_LIB)
    message(STATUS "  [ML Check] All dependencies found.")
    # 1. 收集当前目录下的所有源文件
    file(GLOB MLCVS_SOURCES "*.cu" "*.cpp")

    # 2. 将这些源文件追加到全局变量中
    # 注意：使用 才能让主目录的 CMakeLists.txt 看到更新后的变量
    set_property(GLOBAL APPEND PROPERTY PLUGIN_INCLUDES ${PLUGIN_INCLUDES} ${CMAKE_CURRENT_SOURCE_DIR}/../../include/ml)
    set_property(GLOBAL APPEND PROPERTY PLUGIN_SOURCES ${PLUGIN_SOURCES} ${MLCVS_SOURCES})
    set_property(GLOBAL APPEND PROPERTY PLUGIN_LIBS ${CUDNN_LIB} ${ORT_LIB})

    add_definitions(-DUSE_ML_CVS)
    set_property(GLOBAL APPEND PROPERTY PLUGIN_RPATHS "${ORT_HOME}/lib" "${CONDA_DIR}/lib")
    message(STATUS "Package mlcvs: Added ${MLCVS_SOURCES}")
else()
    message(FATAL_ERROR "  [ML Check] Failed! Check ORT_HOME or Conda environment.")
endif()

# 3. 如果有特定的宏定义，也可以在这里添加
message(STATUS "Package mlcvs: Added ${MLCVS_SOURCES}")