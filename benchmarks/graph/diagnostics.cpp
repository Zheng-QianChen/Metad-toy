#include "zqc_diagnostics.h"
#include <iostream>
#include <vector>

using namespace MetaD_zqc;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void check_logging() {
    std::vector<Diagnostic> diagnostics;
    Logger logger([&](const Diagnostic& record) { diagnostics.push_back(record); });
    // %n 仅用于检查过滤是否发生在格式化之前；输入为固定的测试字符串。
    int formatted = -1;
    logger.printf(LogLevel::Debug, "test", __FILE__, __LINE__, "disabled%n", &formatted);
    require(formatted == -1 && diagnostics.empty(), "disabled debug was formatted");
    logger.printf(LogLevel::Info, "test", "caller.cu", 42, "value=%d", 7);
    require(diagnostics.size() == 1 && diagnostics[0].message == "value=7" &&
            diagnostics[0].file == "caller.cu" && diagnostics[0].line == 42,
            "diagnostic lost message or source");
    Logger broken([](const Diagnostic&) { throw std::runtime_error("sink failed"); });
    broken.printf(LogLevel::Warning, "test", __FILE__, __LINE__, "%s", "cleanup warning");
}

int main() {
    try {
        check_logging();
        std::cout << "DIAGNOSTICS_PASS filtering, printf formatting, source context and throwing sink\n";
    } catch (const std::exception& failure) {
        std::cerr << failure.what() << '\n';
        return 1;
    }
}
