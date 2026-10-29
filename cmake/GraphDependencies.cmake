# fmt 仅在 Host 生成源码字符串时使用。固定使用随项目分发的头文件，
# 使独立 Graph 构建与 LAMMPS、系统 fmt 安装相互独立。
add_library(zqc_fmt INTERFACE)
target_include_directories(zqc_fmt INTERFACE "${PROJECT_SOURCE_DIR}/third_party/fmt/include")
target_compile_definitions(zqc_fmt INTERFACE FMT_HEADER_ONLY)
find_program(XXD NAMES xxd REQUIRED)
