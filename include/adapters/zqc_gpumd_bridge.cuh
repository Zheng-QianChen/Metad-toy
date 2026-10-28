#pragma once
#include <memory>
#include <string>
class Atom;
class Box;
class Integrate;

// GPUMD Action 转发生命周期；Impl 只持有 C ABI 函数表和库内状态句柄。
// 原子数组归 GPUMD 所有，每次求值把地址、跨度和流交给 libmetadtoy。
class MetaDToyGPUMDBridge {
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit MetaDToyGPUMDBridge(const std::string& filename);
public:
    ~MetaDToyGPUMDBridge();
    static std::unique_ptr<MetaDToyGPUMDBridge> create_if_requested(const std::string&);
    void pre_run(Integrate&, Atom&, Box&);
    void evaluate(long long step, bool integrating, Atom&, Box&);
    void end_of_step(long long step);
    void post_run();
};
