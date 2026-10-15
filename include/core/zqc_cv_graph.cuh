#pragma once
#include <memory>
#include <string>
#include <vector>

namespace LAMMPS_NS { class LAMMPS; }
namespace MetaD_zqc {
class CV;
struct CVGraphSymbol {
    std::string name;
    CV* cv;
    std::string method;
};

// LAMMPS 与通用 Graph 的边界：持有装配表/Leaf，管理动态本地长度和 CV 梯度回传。
// CV 和 LAMMPS 由外部持有；本对象必须先于 CV 析构。
class CVGraphExecution {
public:
    CVGraphExecution(LAMMPS_NS::LAMMPS*, std::vector<CVGraphSymbol>, std::vector<std::string>);
    ~CVGraphExecution();
    void prepare(); // 必须在所有 CV::base_calc 之前绑定本步 owned/ghost 表。
    std::vector<double> forward(); // base_calc 完成后装配，GPU 融合，再 Host/MPI 规约。
    void backward(const std::vector<double>& seeds); // 所有输出先汇总梯度，每个 CV 只反传一次。
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
