// 通过生产适配器的计数确认 CUDA/Kokkos 坐标未经过 Host 上传暂存。
#include "../finite_difference/bias_probe.cpp"
#include "zqc_lammps_engine.h"
extern "C" unsigned long long kk_position_uploads(void* pointer) {
    const auto* adapter = dynamic_cast<MetaD_zqc::LammpsEngineAdapter*>(find_fix(pointer)->engine_adapter.get());
    return adapter->sync_stats().position_uploads;
}
