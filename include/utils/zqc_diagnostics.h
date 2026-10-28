#pragma once

#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>

namespace MetaD_zqc {

// 级别用于过滤普通诊断；计算失败通过异常交给调用方处理。
enum class LogLevel { Debug, Info, Warning, Error };

// 一次同步回调中的诊断记录。接收方若需长期保存，应复制记录。
// component 标识 Graph 或具体缓冲区；file/line 指出产生诊断的位置。
struct Diagnostic {
    LogLevel level;
    std::string component;
    std::string message;
    std::string file;
    int line = 0;
};

// 保留 printf 的 % 格式；仅在需要生成消息时分配字符串。
template <typename... Args>
std::string format_message(const char* format, Args... args) {
    const int length = std::snprintf(nullptr, 0, format, args...);
    if (length < 0) throw std::runtime_error("Diagnostic message formatting failed");
    std::string message(static_cast<size_t>(length) + 1, '\0');
    std::snprintf(message.data(), message.size(), format, args...);
    message.resize(static_cast<size_t>(length));
    return message;
}

// Logger 持有回调的副本。回调捕获的文件、引擎对象等资源须由调用方
// 保持有效，直到使用这个 Logger 的对象销毁。默认构造表示关闭日志。
class Logger {
public:
    using Sink = std::function<void(const Diagnostic&)>;

    Logger() = default;
    explicit Logger(Sink sink, LogLevel minimum_level = LogLevel::Info)
        : sink_(std::move(sink)), minimum_level_(minimum_level) {}

    bool enabled(LogLevel level) const noexcept {
        return static_cast<bool>(sink_) && level >= minimum_level_;
    }

    // 日志失败不能覆盖正在传播的计算异常，析构清理也可以安全调用。
    template <typename... Args>
    void printf(LogLevel level, const char* component, const char* file, int line,
                const char* format, Args... args) const noexcept {
        if (!enabled(level)) return;
        try {
            sink_({level, component ? component : "", format_message(format, args...),
                   file ? file : "", line});
        } catch (...) {
            // 回调属于观察者；计算错误仍由原操作的异常报告。
        }
    }

private:
    Sink sink_;
    LogLevel minimum_level_ = LogLevel::Info;
};

} // namespace MetaD_zqc
