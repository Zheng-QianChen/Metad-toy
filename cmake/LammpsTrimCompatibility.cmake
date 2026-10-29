# 检测已复现的旧版 trim=1/custom-cutoff=0 路径。
# 源码 guard 只识别旧实现；最终库必须由此源码重新编译，运行回归验证实际二进制。
if(EXISTS "${LAMMPS_SOURCE_DIR}/npair.cpp")
    file(READ "${LAMMPS_SOURCE_DIR}/npair.cpp" npair_source)
    string(REPLACE "\r" "" npair_source "${npair_source}")
    string(FIND "${npair_source}"
        "cutoff_custom = 0.0;\n  if (nrq->cut) cutoff_custom = nrq->cutoff;"
        zero_default)
    if(NOT zero_default EQUAL -1)
        set(unsafe_trim_sources "")
        foreach(trim_source npair_halffull.cpp npair_trim.cpp npair_skip.cpp)
            if(EXISTS "${LAMMPS_SOURCE_DIR}/${trim_source}")
                file(READ "${LAMMPS_SOURCE_DIR}/${trim_source}" trim_code)
                string(FIND "${trim_code}" "if (rsq > cutsq_custom) continue;" zero_trim)
                if(NOT zero_trim EQUAL -1)
                    list(APPEND unsafe_trim_sources "${trim_source}")
                endif()
            endif()
        endforeach()
        if(unsafe_trim_sources)
            message(FATAL_ERROR
                "LAMMPS has the legacy zero-cutoff trim bug in ${unsafe_trim_sources}. "
                "A pair half list derived from a larger CV full list can become empty. "
                "Update LAMMPS, or apply benchmarks/kokkos/patches/lammps-29Aug2024-trim.patch "
                "to matching 29Aug2024 sources and rebuild liblammps before building this plugin. "
                "See benchmarks/kokkos/README.md.")
        endif()
    endif()
endif()
