#include "zqc_lammps_diagnostics.h"
#include "lammps.h"
#include "comm.h"

namespace MetaD_zqc {
Logger make_lammps_logger(LAMMPS_NS::LAMMPS* lmp, FILE* debug_file) {
    if (!lmp) return {};
#ifdef DEBUG
    const auto minimum_level = LogLevel::Debug;
#else
    const auto minimum_level = LogLevel::Info;
#endif
    return Logger([lmp, debug_file](const Diagnostic& diagnostic) {
        const int rank = lmp->comm->me;
        if (diagnostic.level == LogLevel::Info && rank != 0) return;
        FILE* files[] = {
            diagnostic.level == LogLevel::Debug ? nullptr : lmp->screen,
            diagnostic.level == LogLevel::Debug ? nullptr : lmp->logfile,
            debug_file
        };
        for (int index = 0; index < 3; ++index) {
            FILE* file = files[index];
            if (!file) continue;
            bool duplicate = false;
            for (int previous = 0; previous < index; ++previous)
                if (files[previous] == file) duplicate = true;
            if (duplicate) continue;
            if (diagnostic.level == LogLevel::Info) {
                std::fprintf(file, "[MetaD %s] %s\n", diagnostic.component.c_str(), diagnostic.message.c_str());
            } else {
                const char* level = diagnostic.level == LogLevel::Debug ? "DEBUG" :
                                    diagnostic.level == LogLevel::Warning ? "WARNING" : "ERROR";
                std::fprintf(file, "[MetaD %s][rank %d][%s] %s (%s:%d)\n",
                             diagnostic.component.c_str(), rank, level, diagnostic.message.c_str(),
                             diagnostic.file.c_str(), diagnostic.line);
            }
            std::fflush(file); // NVRTC 开始日志立即可见，便于定位编译耗时。
        }
    }, minimum_level);
}
} // namespace MetaD_zqc
