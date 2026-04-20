# src/cvs/CMakeLists.txt

# 1. 收集当前目录下的所有源文件
file(GLOB CORE_SOURCES "*.cu" "*.cpp")

# 2. 将这些源文件追加到全局变量中
# 注意：使用 PARENT_SCOPE 才能让主目录的 CMakeLists.txt 看到更新后的变量
set_property(GLOBAL APPEND PROPERTY PLUGIN_INCLUDES ${PLUGIN_INCLUDES} ${CMAKE_CURRENT_SOURCE_DIR}/../../include/core)
set_property(GLOBAL APPEND PROPERTY PLUGIN_INCLUDES ${PLUGIN_INCLUDES} ${CMAKE_CURRENT_SOURCE_DIR}/../../include/CVsets)
set_property(GLOBAL APPEND PROPERTY PLUGIN_INCLUDES ${PLUGIN_INCLUDES} ${CMAKE_CURRENT_SOURCE_DIR}/../../include/ml_cvs)
set_property(GLOBAL APPEND PROPERTY PLUGIN_SOURCES ${PLUGIN_SOURCES} ${CORE_SOURCES})

# 3. 如果有特定的宏定义，也可以在这里添加
message(STATUS "Package core: Added ${CORE_SOURCES}")