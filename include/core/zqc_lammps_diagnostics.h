#pragma once
#include "../utils/zqc_diagnostics.h"
#include <cstdio>

namespace LAMMPS_NS { class LAMMPS; }

namespace MetaD_zqc {
// 宿主层负责 rank 策略和文件输出。lmp/debug_file 须比使用回调的对象活得久。
// Info 输出到 rank 0 的 screen/logfile；Warning/Error 保留发生问题的 rank。
// Debug 由 ENABLE_DEBUG 对应的 DEBUG 宏控制，并写入 debug_file。
Logger make_lammps_logger(LAMMPS_NS::LAMMPS* lmp, FILE* debug_file = nullptr);
} // namespace MetaD_zqc
