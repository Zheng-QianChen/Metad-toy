#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include "zqc_CVs_tools.h"
#include "zqc_debug.h"
#include "zqc_DimSet.h"
#include "exprtk.hpp"

MetaD_zqc::MetaDimensionManager::MetaDimensionManager(): 
                                    Num_symbol(0) {
// create ptrs
    p_symbol_table = new exprtk::symbol_table<double>();
    p_parser = new exprtk::parser<double>();
    p_expressions = new std::map<int, exprtk::expression<double>>();
    // p_var_values = new std::vector<double>();
    p_links = new std::vector<SymbolLink>();
    // symbol_table.add_constants(); // 预加载 pi, e 等常量
}

MetaD_zqc::MetaDimensionManager::~MetaDimensionManager(){
// create ptrs
    delete static_cast<exprtk::symbol_table<double>*>(p_symbol_table);
    delete static_cast<exprtk::parser<double>*>(p_parser);
    delete static_cast<std::map<int, exprtk::expression<double>>*>(p_expressions);
    delete static_cast<std::vector<SymbolLink>*>(p_links);
}

// void MetaD_zqc::MetaDimensionManager::add_symbol(const std::string& name, CV* ptr, const std::string& func_name) {
void MetaD_zqc::MetaDimensionManager::add_symbol(const char* name, MetaD_zqc::CV* ptr, const char* func_name) {
    auto& links = (*static_cast<std::vector<SymbolLink>*>(p_links));
    auto& st = (*static_cast<exprtk::symbol_table<double>*>(p_symbol_table));
    
    SymbolLink link;
    strncpy(link.name, name, 63);
    link.name[63] = '\0'; // 确保强制结尾，防止溢出
    link.cv_ptr = ptr;

    
    // 使用辅助函数获取函数指针
    link.compute_func = ptr->set_CV_calculate(func_name);
    link.bias_func = ptr->set_CV_bias_force(func_name);

    link.var_idx = var_values.size();
    var_values.push_back(0.0); 
    links.push_back(link);

    // 将 exprtk 里的变量名直接链接到 CV 内部的 current_val
    // 这样每次 expression.value() 执行时，都会直接读取这个内存地址
    st.clear_variables(); // 清除之前的失效绑定
    for (const auto& link : links) {
        st.add_variable(link.name, var_values[link.var_idx]);
    }
}

void MetaD_zqc::MetaDimensionManager::reg_expression(int dim_idx, const std::string& expr_str) {
    auto& expressions = (*static_cast<std::map<int, exprtk::expression<double>>*>(p_expressions));
    auto& expr = expressions[dim_idx];
    auto& symbol_table = (*static_cast<exprtk::symbol_table<double>*>(p_symbol_table));
    auto& parser = *(static_cast<exprtk::parser<double>*>(p_parser));
    expr.register_symbol_table(symbol_table);

    if (!parser.compile(expr_str, expr)) {
        // 抛出错误或打印调试信息
        printf("ExprTk Error for Dim %d: %s\n", dim_idx, parser.error().c_str());
        return;
    }
}

void MetaD_zqc::MetaDimensionManager::compute_total_cv() {
    auto& links = (*static_cast<std::vector<SymbolLink>*>(p_links));
    // 更新所有被绑定的基础 CV 值
    for (auto& link : links) {
        var_values[link.var_idx] = (link.cv_ptr->*(link.compute_func))();
        // printf("[%d]=%g\n", link.var_idx, var_values[link.var_idx]);
    }
}

double MetaD_zqc::MetaDimensionManager::compute_dim_cv(int dim_idx) {
    auto& expressions = (*static_cast<std::map<int, exprtk::expression<double>>*>(p_expressions));
    // 注意：在计算之前，外部需要确保 links 里的 base_cv 已经更新了 current_val
    return expressions[dim_idx].value();
}

void MetaD_zqc::MetaDimensionManager::distribute_dim_bias_force(int dim_idx, double total_grad) {
    auto& expressions = (*static_cast<std::map<int, exprtk::expression<double>>*>(p_expressions));
    auto& expr = expressions[dim_idx];
    auto& links = (*static_cast<std::vector<SymbolLink>*>(p_links));
    
    for (auto& link : links) {
        // 自动微分：计算当前维度对该符号的偏导
        // 如果表达式是 v1 * v2，对 v1 求导就是 v2 的当前值
        double dS_dv = exprtk::derivative(expr, link.name);
        
        if (dS_dv == 0.0) continue; // 表达式中不含该符号则跳过

        // 链式法则：(dV/dS) * (dS/dv)
        (link.cv_ptr->*(link.bias_func))(total_grad * dS_dv);
    }
}