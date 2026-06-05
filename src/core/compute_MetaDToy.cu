#include <cstring>

#include "atom.h"
#include "update.h"
#include "modify.h"
#include "memory.h"
#include "error.h"
#include "utils.h"
#include "fix.h" // 🚨 只需要引入 LAMMPS 原生 Fix 基类，保持去中心化解耦

#include "compute_MetaDToy.h"
#include "zqc_debug.h"


using namespace LAMMPS_NS;

/* ----------------------------------------------------------------------
   构造函数：解析输入脚本参数
   语法：compute ID group-ID metad/atom <MetaD_id> <cv_name> <cv_prop>
------------------------------------------------------------------------- */
ComputeMetaDToy::ComputeMetaDToy(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg), id_fix(nullptr)
{
  // 基础参数校验：至少需要 5 个参数
  ERR_COND((narg < 5),"compute metad/atom need args more than 5.");

  // 1. 缓存被绑定的元动力学 Fix 的 ID (例如 "my_md")
  id_fix = utils::strdup(arg[3]);

  // 2. 动态解析后面所有的子变量请求 (例如 Q4, Q6 ...)
  requests.clear();
  for (int iarg = 4; iarg < narg; ++iarg) {
    CVRequest req;
    req.cv_name = std::string(arg[iarg]);
    req.iarg = iarg;
    requests.push_back(req);
  }

  // 3. 仿照官方设计：根据请求的数量，自动决定是输出“标量”还是“矩阵”
  peratom_flag = 1; // 声明输出的是每原子数据
  if (requests.size() == 1) {
    size_peratom_cols = 0; // 单个变量：输出为每原子标量 (vector_atom)
  } else {
    size_peratom_cols = requests.size(); // 多个变量：输出为每原子矩阵 (array_atom)
  }

  // 初始化内存相关指针与计数器
  nmax = 0;
  vector_atom = nullptr;
  array_atom = nullptr;
}

/* ----------------------------------------------------------------------
   析构函数：安全释放动态分配的内存
------------------------------------------------------------------------- */
ComputeMetaDToy::~ComputeMetaDToy()
{
  delete[] id_fix;
  memory->destroy(vector_atom);
  memory->destroy(array_atom);
}

/* ----------------------------------------------------------------------
   模拟开始前的初始化验证
------------------------------------------------------------------------- */
void ComputeMetaDToy::init()
{
  // 1. 运行时动态查找脚本中指定的那个 Fix 实例
  Fix *ifix = modify->get_fix_by_id(id_fix);
  ERR_COND((!ifix),"Fix ID %s specified in compute metad/atom does not exist",id_fix);
  // 2. 校验该 Fix 的注册样式是否是你的元动力学核心 (假设叫 "metad")
  ERR_COND((strcmp(ifix->style, "metad") != 0),"Fix  %s is not a valid 'metad' style fix",std::string(id_fix));
}

/* ----------------------------------------------------------------------
   核心数据总线：当 dump 写入触发时，此函数被调用
------------------------------------------------------------------------- */
void ComputeMetaDToy::compute_peratom()
{
  // 记录当前步数，防止在同一个时间步内被重复触发计算
  invoked_peratom = update->ntimestep;

  // 1. 动态内存自适应管理：如果当前本地原子数超过了最大缓存，进行扩容
  if (atom->nmax > nmax) {
    if (requests.size() == 1) {
      memory->destroy(vector_atom);
      nmax = atom->nmax;
      memory->create(vector_atom, nmax, "metad/atom:vector_atom");
    } else {
      memory->destroy(array_atom);
      nmax = atom->nmax;
      memory->create(array_atom, nmax, requests.size(), "metad/atom:array_atom");
    }
  }

  // 2. 再次获取 Fix 指针
  Fix *ifix = modify->get_fix_by_id(id_fix);
  ERR_COND((!ifix),"Internal error pulling metadata fix instance");

  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  int nstride;
  double *ptr;

  // 3. 核心总线路由循环：遍历所有请求的 CV
  int m = 0;
  for (auto &req : requests) {

    // 仿照官方的 Stride 机制控制指针移动
    if (requests.size() == 1) {
      ptr = vector_atom;
      nstride = 1;
    } else {
      ptr = &array_atom[0][m];
      nstride = requests.size();
    }

    // 🚨 跨模块动态路由握手点：
    // 拼接钥匙（例如 "colvar:Q4"），然后敲开 Fix 的 extract 接口
    std::string request_key = "colvar_peratom:" + req.cv_name;
    int dim = 0;
    
    // 拿着钥匙去 Fix 内部拉取指针
    double *fix_stein_q = (double *) ifix->extract(request_key.c_str(), dim);

    // 健壮性报错：如果用户在 compute 中错拼了 CV 名字，或者该 CV 没被编译进系统
    // if (!fix_stein_q) {
    //   std::string err_msg = "Fix '" + std::string(id_fix) + "' specified in compute metad/atom does not exist";
    //   error->all(FLERR, err_msg);
    // }
    ERR_COND((!fix_stein_q),"Fix %s specified in compute metad/atom does not exist",id_fix);
    // 4. 极速将数据拉取并拷贝进 Compute 的缓存区
    for (int i = 0; i < nlocal; i++, ptr += nstride) {
      *ptr = 0.0;
      // 严格检查 group 掩码，只有属于该 group 的原子才会被记录
      if (!(mask[i] & groupbit)) continue; 
      
      *ptr = fix_stein_q[i]; // 指针直连，完成总线数据拷贝
    }
    ++m;
  }
}

/* ----------------------------------------------------------------------
   内存使用统计
------------------------------------------------------------------------- */
double ComputeMetaDToy::memory_usage()
{
  double bytes = (double)nmax * requests.size() * sizeof(double);
  bytes += requests.size() * sizeof(CVRequest);
  return bytes;
}