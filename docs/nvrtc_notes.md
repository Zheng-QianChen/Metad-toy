# NVRTC 编译要点记录

## 1. 基本流程
- 用 NVRTC 运行时编译 .cu/.cuh 源码，生成 cubin，再由 Driver API 加载
- 编译选项包含 `--gpu-architecture=sm_XY`（从 cudaGetDeviceProperties 读取）和 `--std=c++17`
- 源码和自有头文件通过 xxd -i 转为 C 数组嵌进二进制，运行时从内存读取，不依赖磁盘上的源文件

## 2. 多卡 / MPI 并行
- 不要写死 GPU0，每个 MPI rank 用 cudaSetDevice(local_rank)，local_rank 是节点内编号不是全局 rank
- 两张不同架构的卡不能共用一个 cubin；同架构可复用但保险起见每卡查一次
- 每张卡的 NVRTC 编译读的是自己那张卡的 sm 编号

## 3. 头文件处理
- NVRTC 不自动找磁盘文件，自有 .cuh 必须手动读成字符串塞进 headers/includeNames 数组，#include 的名字必须和 includeNames 里的字符串一致
- CUDA 系统头（cuda_runtime.h 等）NVRTC 内置，用 -I$CUDA/include 即可，不走 xxd
- 自有头文件可以建一个总的 .cuh 做入口，嵌套 include 会递归解析，但每层都要 xxd
- xxd 生成的数组没有结尾 \0，要用长度构造 std::string

## 4. tagint / 精度宏（重要，AI 优化时容易忽略）
- LAMMPS 的 tagint 大小由编译宏决定：LAMMPS_BIGBIG 或 LAMMPS_SMALLBIG
- NVRTC 编译 zqc_graph 时，tagint 的类型必须根据 LAMMPS 本体的 tagint 定义来决定
- NVRTC 的编译选项要透传 LAMMPS 使用的精度宏（LAMMPS_BIGBIG 或 LAMMPS_SMALLBIG），从 LAMMPS 的 CMake/Makefile 配置读取，不要猜
- 确保运行时编译的 kernel 中 tagint 与 LAMMPS 侧一致；遗漏会导致类型不匹配、数据错乱

## 5. CMake 相关
- cmake/xxd_embed.cmake 是单独脚本，用 execute_process 调用 xxd -i
- 根 CMakeLists 里 find_program(XXD xxd REQUIRED)，循环对每个 NVRTC 源文件 add_custom_command 生成 .cpp 到 build/embed/，DEPENDS 原文件，target_sources 加进 fix_crystallize_plugin
- src/core/CMakeLists.txt 的 GLOB *.cu 扫不到 build 目录的生成文件，必须手动 target_sources
- 注意 add_subdirectory(src/utils) 和 add_library 的顺序：utils 的 CMakeLists 里用 target_sources 引用 fix_crystallize_plugin 时，目标必须已存在
- 最终决定把 xxd 相关逻辑放在根 CMakeLists.txt（用户选择，不放 utils 子目录）

## 6. 不做什么
- 不修改 test/ 目录
- 不把整个插件源码都 xxd，只嵌 NVRTC 要编的那几份
- kernel 源码里不要 #include "lammps.h" 这类主机头

## 7. 待 AI 优化的方向（代码完成后告诉 AI）
- tagint 宏与 LAMMPS 编译配置的自动对齐机制
- 多文件 xxd + headers 数组的自动化生成（减少手动维护）
- NVRTC 编译错误日志的友好输出
- 是否需要支持 PTX JIT 兜底（老架构卡）
- CUDA Graph 与 NVRTC 编译结果的结合方式
