// 示例：把多个 xxd 嵌入的 .cuh 整理成 NVRTC 需要的 headers / includeNames
// 仅作 docs 演示，不参与插件链接。C++17。

#include <nvrtc.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// 假的 xxd 符号（真实工程里由 build/embed/*.cpp 提供）
extern unsigned char demo_cuh[];
extern unsigned int demo_cuh_len;
extern unsigned char demo2_cuh[];
extern unsigned int demo2_cuh_len;

// 同样假设已有主源码的 xxd 符号
extern unsigned char demo_cu[];
extern unsigned int demo_cu_len;

struct NVRTCHeaders {
    std::vector<std::string> sources;    // 每个头文件的内容（保活，供 c_str 指向）
    std::vector<std::string> names;      // 每个头文件的名字（与 #include "..." 一致）
    std::vector<const char*> header_ptr; // 指向 sources 元素的 c_str
    std::vector<const char*> name_ptr;   // 指向 names 元素的 c_str
};

// file_symbols：xxd 符号名列表，不含 _len 后缀，例如 "demo_cuh"
NVRTCHeaders make_nvrtc_headers(const std::vector<std::string>& file_symbols) {
    // 符号名 → (内容指针, 长度)；查表后用长度构造 string，避免依赖结尾 '\0'
    static const std::unordered_map<std::string,
                                    std::pair<const unsigned char*, unsigned int>>
        kTable = {
            {"demo_cuh", {demo_cuh, demo_cuh_len}},
            {"demo2_cuh", {demo2_cuh, demo2_cuh_len}},
        };

    NVRTCHeaders out;
    out.sources.reserve(file_symbols.size());
    out.names.reserve(file_symbols.size());

    for (const auto& sym : file_symbols) {
        const auto it = kTable.find(sym);
        if (it == kTable.end()) {
            throw std::runtime_error("unknown xxd symbol: " + sym);
        }
        const unsigned char* bytes = it->second.first;
        const unsigned int len = it->second.second;

        // 内容：按长度拷进 string
        out.sources.emplace_back(reinterpret_cast<const char*>(bytes), len);

        // 名字：把符号末尾的 _cuh / _h 还原成带点的头文件名，供 #include 匹配
        // 这里演示固定映射；真实工程可维护另一张 name 表
        std::string include_name = sym;
        if (include_name.size() >= 4 &&
            include_name.compare(include_name.size() - 4, 4, "_cuh") == 0) {
            include_name.replace(include_name.size() - 4, 4, ".cuh");
        } else if (include_name.size() >= 2 &&
                   include_name.compare(include_name.size() - 2, 2, "_h") == 0) {
            include_name.replace(include_name.size() - 2, 2, ".h");
        }
        out.names.push_back(std::move(include_name));
    }

    // 指针数组必须在 sources/names 填完后再建，否则 resize 可能让旧指针失效
    out.header_ptr.reserve(out.sources.size());
    out.name_ptr.reserve(out.names.size());
    for (std::size_t i = 0; i < out.sources.size(); ++i) {
        out.header_ptr.push_back(out.sources[i].c_str());
        out.name_ptr.push_back(out.names[i].c_str());
    }
    return out;
}

int main() {
    // 1) 主翻译单元同样用 xxd 数组 + 长度构造
    std::string source(reinterpret_cast<const char*>(demo_cu), demo_cu_len);

    // 2) 按符号名拉齐多个头文件
    NVRTCHeaders headers = make_nvrtc_headers({"demo_cuh", "demo2_cuh"});

    // 3) 交给 NVRTC；numHeaders 与两个 data() 数组等长
    nvrtcProgram prog = nullptr;
    nvrtcResult rc = nvrtcCreateProgram(
        &prog,
        source.c_str(),
        "demo.cu",
        static_cast<int>(headers.header_ptr.size()),
        headers.header_ptr.data(),
        headers.name_ptr.data());
    if (rc != NVRTC_SUCCESS) {
        return 1;
    }

    nvrtcDestroyProgram(&prog);
    return 0;
}
