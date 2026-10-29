#pragma once
#include "zqc_engine.h"
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <string>

namespace MetaD_zqc {
inline double cv_number(const char* text) {
    char* end = nullptr; errno = 0;
    const double value = std::strtod(text,&end);
    if (end == text || *end || errno == ERANGE || !std::isfinite(value))
        throw std::invalid_argument(std::string("Invalid finite number: ")+text);
    return value;
}
inline int cv_integer(const char* text) {
    char* end = nullptr; errno = 0;
    const long value = std::strtol(text,&end,10);
    if (end == text || *end || errno == ERANGE || value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
        throw std::invalid_argument(std::string("Invalid integer: ")+text);
    return static_cast<int>(value);
}
inline AtomId cv_atom_id(const char* text) {
    char* end = nullptr; errno = 0;
    const auto value = std::strtoull(text,&end,10);
    if (*text == '-' || end == text || *end || errno == ERANGE || value == 0)
        throw std::invalid_argument(std::string("Invalid positive atom ID: ")+text);
    return value;
}
} // namespace MetaD_zqc
