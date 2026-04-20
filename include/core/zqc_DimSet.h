#pragma once  // 必须添加这一行

#include "fix_crystallize.h"
#include "lammps.h"
#include "pair.h"
#include "neigh_request.h"
#include "neigh_list.h"
#include "zqc_CVs_tools.h"
// #include "exprtk.hpp"

namespace MetaD_zqc {

    struct SymbolLink {
        char name[64];       // 符号名，如 "v1"
        int var_idx=0;
        int padding;
        class CV* cv_ptr;             // 对应的底层 CV 实例
        // 成员函数指针，用于调用计算和受力
        double (CV::*compute_func)();
        void (CV::*bias_func)(double);
    };

    struct DimConfig {
        std::string name;
        std::string func;
    };

    class MetaDimensionManager {
    private:
        bool Flag_inited=false; // if init, this will be true
        int Num_symbol=0;

        // var_values
        std::vector<double> var_values;

        std::vector<DimConfig> dim_configs;
        // std::vector<MetaD_zqc::CV*> base_cv;
        // std::vector<MetaD_zqc::CV::CV_Calculation> cv_compute;
        // std::vector<MetaD_zqc::CV::CV_BiasForce> cv_biasforce;

        
        // exprtk::symbol_table<double>                 symbol_table;
        void*                                           p_symbol_table;
        // exprtk::parser<double>                       parser;
        void*                                           p_parser;
        // every dimensions
        // std::map<int, exprtk::expression<double>>    expressions;
        void*                                           p_expressions;

        // save links between CVs and symbol
        // std::vector<SymbolLink>                      links;
        void*                                           p_links;

    public:
        MetaDimensionManager();
        ~MetaDimensionManager();
        void add_symbol(const char* name, MetaD_zqc::CV* ptr, const char* func_name);
        void reg_expression(int dim_idx, const std::string& expr_str);
        void compute_total_cv();
        double compute_dim_cv(int dim_idx);
        void distribute_dim_bias_force(int dim_idx, double total_grad);
    };
}