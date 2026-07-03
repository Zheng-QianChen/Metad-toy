#ifndef LMP_COMPUTE_METAD_ATOM_H
#define LMP_COMPUTE_METAD_ATOM_H

#include "compute.h"
#include <string>
#include <vector>

namespace LAMMPS_NS {

  class ComputeMetaDToy : public Compute {

  private:
    char *id_fix; // 缓存绑定的主 Fix 的 ID 字符串（例如 "my_md"）
    FILE* f_check;

    // 定义一个内部结构体，用于存储用户请求的 CV 信息
    struct CVRequest {
      std::string cv_name; // 具体的 CV 名称（例如 "Q4", "Q6"）
      int iarg;             // 该参数在脚本中的位置，用于精准报错
    };
    
    std::vector<CVRequest> requests; // 存储用户在脚本中输入的所有目标 CV 列表

    tagint nmax;
  
  public:
    // 构造函数：解析 LAMMPS 脚本传入的参数
    ComputeMetaDToy(class LAMMPS *, int, char **, FILE* f_check);
    
    // 析构函数：释放动态分配的内存
    ~ComputeMetaDToy() override;
    
    // 初始化函数：在模拟开始前检查绑定的 Fix 实例是否存在
    void init() override;
    
    // 核心计算/拷贝函数：dump 触发时，通过总线拉取数据
    void compute_peratom() override;
    
    // 内存统计函数：向 LAMMPS 汇报当前 compute 占用的内存大小
    double memory_usage() override;

  };


} // namespace LAMMPS_NS

#endif // LMP_COMPUTE_METAD_ATOM_H