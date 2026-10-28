#include <cmath>
#include <cstdio>
#include <algorithm>
#include <limits>

#include <cuda_runtime.h>

#include "lammpsplugin.h"
#include "version.h"
#include "atom.h"
#include "comm.h"
#include "update.h"
#include "domain.h"
#include "memory.h"
#include "error.h"
#include "exceptions.h"
#include "force.h"
#include "group.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "compute.h"

#include "fix_crystallize.h"
#include "zqc_lammps_engine.h"
#include "CV_Stru_factor.h"
#include "compute_MetaDToy.h"
#include "zqc_debug.h"
#include "zqc_DimSet.h"
#include "zqc_cv_graph.cuh"
#include "zqc_gpu_buffer.cuh"
#include "zqc_bias.h"
#include "zqc_shared_bias.cuh"
#include "zqc_mlcvs.h"
#include "zqc_switch_function.h"

using namespace LAMMPS_NS;



FixMetadynamics::FixMetadynamics(LAMMPS *lmp, int narg, char **arg)
                                : Fix(lmp, narg, arg),
                                  pace(100),
                                  cv_dim(1), nbin_num(100),
                                  bias_energy(0.0),
                                  f_before_bias(nullptr),
                                  max_f_before_bias(0),
                                  p_gaussian(nullptr),
                                  cv_values(nullptr),
                                  cv_history(nullptr),
                                  dVdcvs(nullptr),
                                  cv_configs(nullptr){
    neigh_hub.bind(lmp, this);
    if (comm->me==0){
        f_check = fopen("metad_debug_logging.txt","w");
        LOG("New JOB STARTING WITH DEBUG MOD!");
        fclose(f_check);
    }
    MPI_Barrier(world);
    f_check = fopen("metad_debug_logging.txt","a");
    engine_adapter = std::make_unique<MetaD_zqc::LammpsEngineAdapter>(
        lmp, this, MetaD_zqc::make_lammps_logger(lmp, f_check));
    LOG("MPI_COMM_WORLD size: %d", comm->nprocs);
    pthread_t tid = pthread_self();
    LOG("Process Rank: %d | Total Processes: %d | Thread ID: %lu",
            comm->me, comm->nprocs, (unsigned long)tid);
    // Fix 构造函数的基本参数检查： fix ID group style args...
    // arg[0]=fix, arg[1]=ID, arg[2]=group, arg[3]=style (metad)
    // if (narg < 4) error->all(FLERR, "Fix metadynamics requires at least 1 argument after style name.");
    LOG("There are %d args", narg);
    // --- 核心参数解析：循环读取关键词/数值对 ---
    int i = 3; // 从第 4 个参数开始，即 style 名之后
    // std::vector<MetaD_zqc::SteinhardtRequest> steinh_requests;
    cv_configs = new MetaD_zqc::MetaDimensionManager();
    double *cv_bound = nullptr;
    int *nbin = nullptr;
    int Gaussian_Hill_type = -1; // AUTO：维数由配置完成后决定。
    double sigma, height0, biasf, current_temp;
    sigma     = 0.05;
    height0   = 0.1;
    biasf     = 10.0;
    current_temp = 300.0;
    int continue_from_file=false;
    int WellT_bool=false;
    bool use_graph = false;
    std::vector<double> axis_sigmas, axis_periods;
    std::string hills_path = "HILLS";
    std::string shared_table_path, shared_group;
    auto shared_aggregation=MetaD_zqc::SharedAggregation::Mean;
    bool shared_aggregation_explicit=false;
    bool shared_online=false,shared_deposition_explicit=false;
    MetaD_zqc::SharedOnlineSettings shared_settings;
    std::string shared_identity;
    bool pace_explicit = false;
    unsigned history_workers=1;
    bool pb_sum_normalization=false;
    auto pb_kernel=MetaD_zqc::PBKernel::Plumed210;
    bool expression_only_dimensions = false;
    std::vector<MetaD_zqc::CVGraphSymbol> symbol_configs;
    std::vector<std::string> dim_expressions;

    virial_global_flag = 1;
    virial_peratom_flag = 1;
    while (i < narg) {
        const int keyword_begin=i;
        LOG("Im in arg loop");
        if (strcmp(arg[i], "GAUSSIAN") == 0) {
            ERR_COND(i + 3 >= narg, "Error: GAUSSIAN command requires 3 arguments: sigma, height, biasf.");
            sigma   = utils::numeric(FLERR, arg[i+1], false, lmp);
            height0 = utils::numeric(FLERR, arg[i+2], false, lmp);
            biasf   = utils::numeric(FLERR, arg[i+3], false, lmp);
            i += 4;

            // double R_SI = 8.314462618;
            // height0 = KB*height0/R_SI;

            LOG("Logging: set GAUSSIAN sigma, height, biasf: %g %g %g.", sigma, height0, biasf);
            LOG("Logging: attention, we use Joules/moles as the height's units, so if the lammps settings is not real, there will be a units transform.\n\
             If you set CV's bounds in physicals, this units will follows your lammps units settings.");
        } else if (strcmp(arg[i], "EXECUTOR") == 0) {
            ERR_COND(i + 1 >= narg, "EXECUTOR requires LEGACY or GRAPH");
            ERR_COND(strcmp(arg[i+1], "GRAPH") != 0 && strcmp(arg[i+1], "LEGACY") != 0,
                     "EXECUTOR requires LEGACY or GRAPH");
            use_graph = strcmp(arg[i+1], "GRAPH") == 0;
            i += 2;
        } else if (strcmp(arg[i], "SHARED_FROZEN") == 0) {
            ERR_COND(i+2>=narg || !shared_table_path.empty() || shared_online,
                     "SHARED_FROZEN requires one table file and one center group");
            shared_table_path=arg[i+1]; shared_group=arg[i+2]; i+=3;
        } else if (strcmp(arg[i], "SHARED_ONLINE") == 0) {
            ERR_COND(i+4>=narg || shared_online || !shared_table_path.empty(),
                     "SHARED_ONLINE requires group lower upper nodes and cannot coexist with SHARED_FROZEN");
            shared_group=arg[i+1];
            shared_settings.lower=utils::numeric(FLERR,arg[i+2],false,lmp);
            shared_settings.upper=utils::numeric(FLERR,arg[i+3],false,lmp);
            shared_settings.nodes=utils::inumeric(FLERR,arg[i+4],false,lmp);
            shared_online=true;i+=5;
        } else if (strcmp(arg[i], "SHARED_DEPOSIT") == 0) {
            ERR_COND(i+1>=narg || (strcmp(arg[i+1],"MEAN") && strcmp(arg[i+1],"SUM")),
                     "SHARED_DEPOSIT requires MEAN or SUM");
            shared_settings.mean_deposition=strcmp(arg[i+1],"MEAN")==0;
            shared_deposition_explicit=true;i+=2;
        } else if (strcmp(arg[i], "SHARED_AGGREGATION") == 0) {
            ERR_COND(i+1>=narg || (strcmp(arg[i+1],"MEAN") && strcmp(arg[i+1],"SUM")),
                     "SHARED_AGGREGATION requires MEAN or SUM");
            shared_aggregation=strcmp(arg[i+1],"SUM")==0 ? MetaD_zqc::SharedAggregation::Sum : MetaD_zqc::SharedAggregation::Mean;
            shared_aggregation_explicit=true;i+=2;
        } else if (strcmp(arg[i], "PACE") == 0) {
            pace_explicit = true;
            ERR_COND(i + 1 >= narg, "Error: PACE command requires 1 argument: integer timesteps.");
            pace = utils::inumeric(FLERR, arg[i+1], false, lmp);
            i += 2;
            LOG("Logging: pace = %d.",pace);
        } else if (strcmp(arg[i], "RECORD") == 0) {
            ERR_COND(i + 1 >= narg, "Error: RECORD command requires 1 argument: integer timesteps.");
            int iarg=1 + i;
            while (iarg < narg) {
                if (strcmp(arg[iarg], "FILE_NAME") == 0) {
                    ERR_COND((iarg + 1 >= narg) ,"Error: \'FILE_NAME\' keyword requires a value");
                    rec_file_name = arg[iarg+1];
                    iarg += 2;
                }
                else if (strcmp(arg[iarg], "REC_PACE") == 0) {
                    ERR_COND((iarg + 1 >= narg), "Error: \'REC_PACE\' keyword requires an integer");
                    rec_pace = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
                    iarg += 2;
                }
                else {
                  break;
                }
            }
            LOG("Logging: rec_file = %s, rec_pace = %d.",rec_file_name.c_str(), rec_pace);
            rec_file = fopen(rec_file_name.c_str(),"w+");
            i = iarg;
        } else if (strcmp(arg[i], "CV_dim") == 0) {
            DEBUG_LOG("In CV_dim settings");
            ERR_COND(i + 1 >= narg, "Error: PACE command requires 1 argument: integer timesteps.");
            cv_dim = utils::inumeric(FLERR, arg[i+1], false, lmp);
            ERR_COND(cv_dim<=0 || cv_dim>std::numeric_limits<int>::max()/2,
                     "CV_dim must be positive and fit the dimension buffers");
            ERR_COND(cv_bound!=nullptr, "CV_dim must be specified once");
            memory->create(nbin, cv_dim, "metad:nbin_size");
            std::fill_n(nbin,cv_dim,0);
            memory->create(cv_bound, cv_dim*2, "metad:GaussianHill:cv_bound");

            dim_expressions.resize(cv_dim);

            i += 2;
        } else if (strcmp(arg[i], "TEMP") == 0) {
            ERR_COND(i + 1 >= narg, "Error: TEMP command requires 1 argument: target temperature.");
            current_temp = utils::numeric(FLERR, arg[i+1], false, lmp);
            i += 2;
            LOG("Logging: current_temp = %g (用于Well-Tempered公式)", current_temp);
        } else if (strcmp(arg[i], "CAL") == 0){
          ERR_COND(strcmp(arg[i+1], "NAME") != 0, "Error: CAL requires NAME keyword.");
          std::string cal_name = arg[i+2];
          std::string type = arg[i+3];
          LOG("Dueling with %s",cal_name.c_str());
          i += 3;
          if (strcmp(type.c_str(), "SW_FUNC") == 0) {
            if (sw_registry.find(cal_name) != sw_registry.end()) {
              error->all(FLERR, "Error: CAL name '%s' is duplicated.", cal_name.c_str());
            } else {
               LOG("Creating switch function %s", cal_name.c_str());
            }
            // SW_FUNC 也将由 CAL 指定
            sw_registry[cal_name] = MetaD_zqc::SwitchFunction::create(narg, arg, i, engine_adapter->logger());
          } else {
            if (cal_registry.find(cal_name) != cal_registry.end()) {
              error->all(FLERR, "Error: CAL name '%s' is duplicated.", cal_name.c_str());
            } else {
               LOG("Creating switch function %s", cal_name.c_str());
            }
            // 通过工厂方法创建 CV 对象，并注册到 cal_registry 中
            MetaD_zqc::CVContext context{*engine_adapter, sw_registry, engine_adapter->logger(), f_check};
            auto new_cv = MetaD_zqc::CVFactory::create(type, context, narg, arg, i);
            if (new_cv == nullptr) {
              error->all(FLERR, "Metad-toy Error: Unknown or unregistered object type '%s' specified in CAL '%s'. "
                                "Please register it in CVFactory or check your input script.",
                        type.c_str(), cal_name.c_str());
            }
            cal_registry[cal_name] = new_cv;
          }
      } else if (strcmp(arg[i], "SYMBOL") == 0) {
            // SYMBOL v1 Q6.AVE -> SYMBOL <symbol_name> <cv_method>
            ERR_COND(i + 2 >= narg, "Error: DIM command requires 2 arguments: SYMBOL <symbol_name> <cv_method>.");
            // "Q4.mean"
            DEBUG_LOG("168L:%s", arg[i+2]);
            std::string target = arg[i+2];
            std::string target_name;
            std::string target_func;
            size_t dot_pos = target.find(".");
            if (dot_pos != std::string::npos) {
                target_name = target.substr(0, dot_pos);
                target_func = target.substr(dot_pos + 1);
            } else {
                target_name = target;
                target_func = "value";
            }
            DEBUG_LOG("VAL=%s, METHODS=%s",target_name.c_str(),target_func.c_str());
            // add_symbol(const std::string& name, CV* ptr, const std::string& func_name);
            auto found = cal_registry.find(target_name);
            ERR_COND(found == cal_registry.end(), "SYMBOL refers to unknown CV '%s'", target_name.c_str());
            symbol_configs.push_back({arg[i+1], found->second, target_func});
            DEBUG_LOG("1");
            i += 3;
      } else if (strcmp(arg[i], "DIM") == 0) {
            // DIM 1 0 40 1600 (v1+v2)/2 -> DIM index, lower_bound, upper_bound, bins, cv_expr
            ERR_COND(i + 5 >= narg, "Error: DIM command requires 5 arguments: index, lower, upper, bins, cv_expr.");
            int dim_idx = utils::inumeric(FLERR, arg[i+1], false, lmp) - 1; // 0-based
            ERR_COND(dim_idx < 0 || dim_idx >= cv_dim || !cv_bound || !nbin, "DIM requires CV_dim first and a valid 1-based index");
            cv_bound[dim_idx * 2] = utils::numeric(FLERR, arg[i+2], false, lmp); // lower
            cv_bound[dim_idx * 2 + 1] = utils::numeric(FLERR, arg[i+3], false, lmp); // upper
            int nbin_num = utils::inumeric(FLERR, arg[i+4], false, lmp); // 将 bins 数量赋给 nbin_num (仅用于单维度)
            nbin[dim_idx] = nbin_num;
            LOG("Logging: cv_compute=%d bound set at [%g,%g], total grid is %d.",dim_idx+1, cv_bound[dim_idx * 2], cv_bound[dim_idx * 2 +1], nbin_num);
            LOG("Logging: cv_compute=%d bound set expr as %s", dim_idx+1, arg[i+5]);
            // reg_expression(int dim_idx, const std::string& expr_str);
            dim_expressions[dim_idx] = arg[i+5];
            // 【标记该维度已配置，后续检查是否所有维度都配置了表达式
            // has_dim_configured[dim_idx] = true;
            i += 6;
        } else if (strcmp(arg[i], "HILLS_WORKERS") == 0) {
            ERR_COND(i+1>=narg,"HILLS_WORKERS requires 1-16");
            const int workers=utils::inumeric(FLERR,arg[i+1],false,lmp);
            ERR_COND(workers<1 || workers>16,"HILLS_WORKERS requires 1-16");
            history_workers=static_cast<unsigned>(workers);i+=2;
        } else if (strcmp(arg[i], "PB_KERNEL") == 0) {
            ERR_COND(i+1>=narg,"PB_KERNEL requires PLUMED_2_10 or SMOOTH_GAUSSIAN");
            const std::string kernel=arg[i+1];
            ERR_COND(kernel!="PLUMED_2_10" && kernel!="SMOOTH_GAUSSIAN","Unknown PB_KERNEL");
            pb_kernel=kernel=="PLUMED_2_10" ? MetaD_zqc::PBKernel::Plumed210 : MetaD_zqc::PBKernel::SmoothGaussian;
            i+=2;
        } else if (strcmp(arg[i], "PB_NORMALIZATION") == 0) {
            ERR_COND(i+1>=narg,"PB_NORMALIZATION requires SUM or MEAN");
            const std::string mode=arg[i+1];
            ERR_COND(mode!="SUM" && mode!="MEAN","PB_NORMALIZATION requires SUM or MEAN");
            pb_sum_normalization=(mode=="SUM");i+=2;
        } else if (strcmp(arg[i], "DIM_EXPR") == 0) {
            ERR_COND(i+2>=narg, "DIM_EXPR requires index and expression");
            const int dimension=utils::inumeric(FLERR,arg[i+1],false,lmp)-1;
            ERR_COND(!cv_bound || dimension<0 || dimension>=cv_dim,
                     "DIM_EXPR requires CV_dim first and a valid index");
            dim_expressions[dimension]=arg[i+2];
            expression_only_dimensions=true;
            i+=3;
        } else if (strcmp(arg[i], "BIAS") == 0) {
            ERR_COND(i+1>=narg, "BIAS requires AUTO, GRID, SPARSE_GRID, HILLS or PBMETAD");
            const std::string mode=arg[i+1];
            if (mode=="AUTO") Gaussian_Hill_type=-1;
            else if (mode=="GRID") Gaussian_Hill_type=0;
            else if (mode=="SPARSE_GRID") Gaussian_Hill_type=1;
            else if (mode=="HILLS") Gaussian_Hill_type=2;
            else if (mode=="PBMETAD") Gaussian_Hill_type=3;
            else error->all(FLERR,"Unknown BIAS backend: {}",mode);
            i+=2;
        } else if (strcmp(arg[i], "SIGMAS") == 0 || strcmp(arg[i], "PERIODS") == 0) {
            ERR_COND(!cv_bound || cv_dim>=narg-i, "SIGMAS/PERIODS require CV_dim followed by one value per dimension");
            auto& values=strcmp(arg[i],"SIGMAS")==0 ? axis_sigmas : axis_periods;
            values.resize(cv_dim);
            for (int dimension=0; dimension<cv_dim; ++dimension)
                values[dimension]=utils::numeric(FLERR,arg[i+1+dimension],false,lmp);
            i+=1+cv_dim;
        } else if (strcmp(arg[i], "HILLS_FILE") == 0) {
            ERR_COND(i+1>=narg, "HILLS_FILE requires a path");
            hills_path=arg[i+1];i+=2;
        } else if (strcmp(arg[i], "Gaussian_Hill_type") == 0) {
            ERR_COND(i + 1 >= narg, "Gaussian_Hill_type requires 0(GRID), 1(SPARSE_GRID), 2(HILLS), or 3(PBMETAD)");
            Gaussian_Hill_type = utils::inumeric(FLERR, arg[i+1], false, lmp);
            i += 2;
        } else if (strcmp(arg[i], "METAD_RESTART") == 0) {
            ERR_COND(i + 1 >= narg, "Error: METAD_RESTART command requires 1 argument: 0 or 1.");
            continue_from_file = (utils::inumeric(FLERR, arg[i+1], false, lmp) != 0);
            i += 2;
        } else if (strcmp(arg[i], "WT") == 0) {
            ERR_COND(i+1>=narg, "WT requires 0 or 1");
            WellT_bool = (utils::inumeric(FLERR, arg[i+1], false, lmp) != 0);
            i += 2;
        } else {
            ERR_COND(1, "Error: Unknown keyword in fix metadynamics command: %s", arg[i]);
            break;
        }
        const std::string keyword=arg[keyword_begin];
        if(keyword=="CAL" || keyword=="SYMBOL" || keyword=="CV_dim" || keyword=="DIM" || keyword=="DIM_EXPR")
            for(int token=keyword_begin;token<i;++token)
                shared_identity+=std::to_string(std::strlen(arg[token]))+":"+arg[token]+";";
    }
    const bool shared_mode=shared_online || !shared_table_path.empty();
    ERR_COND(shared_deposition_explicit && !shared_online,"SHARED_DEPOSIT requires SHARED_ONLINE");
    ERR_COND(shared_aggregation_explicit && !shared_mode,"SHARED_AGGREGATION requires a shared bias");
    if(shared_online) {
        ERR_COND(!use_graph || cv_dim!=1,"SHARED_ONLINE requires EXECUTOR GRAPH and CV_dim 1");
        ERR_COND(!axis_periods.empty() && axis_periods[0]!=0,"SHARED_ONLINE currently requires a nonperiodic coordinate");
        Gaussian_Hill_type=2;
    }
    if (!shared_table_path.empty()) {
        ERR_COND(!use_graph || cv_dim!=1, "SHARED_FROZEN requires EXECUTOR GRAPH and CV_dim 1");
        ERR_COND((pace_explicit && pace!=0) || WellT_bool || continue_from_file,
                 "SHARED_FROZEN uses an immutable table: PACE must be 0, WT and METAD_RESTART must be 0");
        pace=0;
        Gaussian_Hill_type=2; // 标量 Gaussian 配置在此模式下不建立，DIM_EXPR 无需网格边界。
    }
    const bool automatic_bias=Gaussian_Hill_type==-1;
    if (automatic_bias) Gaussian_Hill_type=cv_dim>3 ? 3 : 0;
    ERR_COND(Gaussian_Hill_type<0 || Gaussian_Hill_type>3, "Unknown bias backend");
    ERR_COND(pace<0, "PACE must be nonnegative; zero freezes deposition");
    ERR_COND(!std::isfinite(sigma) || sigma<=0 || !std::isfinite(height0) || height0<=0,
             "GAUSSIAN sigma and height must be finite and positive");
    ERR_COND(!std::isfinite(current_temp) || current_temp<=0, "TEMP must be finite and positive");
    ERR_COND(!std::isfinite(biasf) || (WellT_bool && biasf<=1), "WT bias factor must exceed one");
    if (Gaussian_Hill_type<2) {
        ERR_COND(expression_only_dimensions,
                 "Grid bias requires DIM with bounds and bins");
        ERR_COND(cv_dim>3 || !cv_bound || !nbin, "Grid bias requires 1-3 configured DIM axes");
        for (int dimension=0; dimension<cv_dim; ++dimension)
            ERR_COND(nbin[dimension]<4 || !std::isfinite(cv_bound[2*dimension]) ||
                     !std::isfinite(cv_bound[2*dimension+1]) ||
                     cv_bound[2*dimension+1]<=cv_bound[2*dimension], "Grid DIM requires finite increasing bounds and at least four bins");
    }
    // 完整读入后才构图，EXECUTOR 的位置不影响 SYMBOL/DIM 的含义。
    ERR_COND(dim_expressions.size() != static_cast<size_t>(cv_dim), "Missing CV_dim/DIM configuration");
    for (const auto& expr : dim_expressions) ERR_COND(expr.empty(), "Every DIM requires an expression");
    try {
        if (use_graph) {
            std::vector<MetaD_zqc::SelectionId> output_selections;
            if (shared_mode) {
                shared_frozen_selection=engine_adapter->selection_id(shared_group);
                output_selections.push_back(shared_frozen_selection);
                if(!shared_online) shared_frozen_bias=std::make_unique<MetaD_zqc::FrozenSharedBias>(
                    shared_table_path,engine_adapter->sum_reducer(),shared_aggregation);
            }
            graph_execution = std::make_unique<MetaD_zqc::CVGraphExecution>(
                *engine_adapter, symbol_configs, dim_expressions, output_selections);
        } else {
            for (const auto& symbol : symbol_configs)
                cv_configs->add_symbol(symbol.name.c_str(), symbol.cv, symbol.method.c_str());
            for (int k = 0; k < cv_dim; ++k) cv_configs->reg_expression(k, dim_expressions[k]);
        }
    } catch (const std::exception& e) {
        error->one(FLERR, "MetaD executor configuration: {}", e.what());
    }
    if (comm->me == 0) {
        for (FILE* file : {screen, logfile}) if (file) {
            fprintf(file, "[MetaD] executor=%s dimensions=%d\n", use_graph ? "GRAPH" : "LEGACY", cv_dim);
            fflush(file);
        }
    }
    LOG("init calc set end, start to define Gaussian.");

    // for (int j = 0; j < cv_dim; ++j) {
    //     if (!has_dim_configured[j]) {
    //         error->all(FLERR, "Error: CV_dim is set to %d, but DIM %d is missing in the input script!", cv_dim, j + 1);
    //     }
    // }

    MetaD_zqc::BiasSettings bias_settings;
    const MetaD_zqc::BiasBackend backends[]={MetaD_zqc::BiasBackend::Grid,MetaD_zqc::BiasBackend::SparseGrid,
                                           MetaD_zqc::BiasBackend::Hills,MetaD_zqc::BiasBackend::PBMetaD};
    bias_settings.backend=backends[Gaussian_Hill_type];
    bias_settings.hills.initial_height=height0;
    bias_settings.hills.thermal_energy=force->boltz*current_temp;
    bias_settings.hills.bias_factor=biasf;
    bias_settings.hills.well_tempered=WellT_bool;
    bias_settings.pace=pace;
    bias_settings.restart=continue_from_file;
    bias_settings.history_path=hills_path;
    bias_settings.history_workers=history_workers;
    bias_settings.pb_sum_normalization=pb_sum_normalization;
    bias_settings.pb_kernel=pb_kernel;
    for (int dimension=0;dimension<cv_dim;++dimension) {
        const double width=axis_sigmas.empty() ? sigma : axis_sigmas[dimension];
        const double period=axis_periods.empty() ? 0.0 : axis_periods[dimension];
        bias_settings.hills.axes.push_back({width,period});
        if (Gaussian_Hill_type<2)
            bias_settings.grid_axes.push_back({cv_bound[2*dimension],cv_bound[2*dimension+1],width,nbin[dimension],period});
    }
    memory->destroy(cv_bound);memory->destroy(nbin);
    MetaD_zqc::BiasHost bias_host;
    bias_host.root=comm->me==0;
    bias_host.broadcast=[this](double* values,std::size_t count) {
        MPI_Bcast(values,static_cast<int>(count),MPI_DOUBLE,0,world);
    };
    bias_host.log=[this](const std::string& message) {
        for (FILE* output:{screen,logfile}) if (output) {
            fprintf(output,"%s\n",message.c_str());fflush(output);
        }
    };
    if(shared_online) {
        shared_settings.sigma=axis_sigmas.empty() ? sigma : axis_sigmas[0];
        shared_settings.height=height0;shared_settings.thermal_energy=force->boltz*current_temp;
        shared_settings.bias_factor=biasf;shared_settings.well_tempered=WellT_bool;
        shared_settings.pace=pace;shared_settings.restart=continue_from_file;
        shared_settings.history_path=hills_path;shared_settings.identity=shared_group+";"+shared_identity;
        shared_settings.aggregation=shared_aggregation;
        try {
            shared_online_bias=std::make_unique<MetaD_zqc::OnlineSharedBias>(shared_settings,
                MetaD_zqc::SharedOnlineHost{engine_adapter->sum_reducer(),bias_host.broadcast,comm->me==0});
        } catch(const std::exception& failure) {error->one(FLERR,"MetaD shared configuration: {}",failure.what());}
        if(comm->me==0) for(FILE* output:{screen,logfile}) if(output) {
            fprintf(output,"[MetaD] bias=SHARED_ONLINE aggregation=%s deposition=%s group=%s nodes=%d WT=%d history=%s\n",
                shared_aggregation==MetaD_zqc::SharedAggregation::Mean ? "MEAN" : "SUM",
                shared_settings.mean_deposition ? "MEAN" : "SUM",shared_group.c_str(),shared_settings.nodes,WellT_bool,hills_path.c_str());
            fflush(output);
        }
    }
    if (!shared_mode)
        p_gaussian=new MetaD_zqc::BiasSession(std::move(bias_settings),std::move(bias_host));
    if (shared_frozen_bias && comm->me==0) for (FILE* output:{screen,logfile}) if (output) {
        fprintf(output,"[MetaD] bias=SHARED_FROZEN aggregation=%s group=%s table=%s fingerprint=%s\n",
                shared_aggregation==MetaD_zqc::SharedAggregation::Mean ? "MEAN" : "SUM",
                shared_group.c_str(),shared_table_path.c_str(),shared_frozen_bias->fingerprint().c_str());
        fflush(output);
    }
    if (p_gaussian && comm->me==0) for (FILE* output:{screen,logfile}) if (output) {
        fprintf(output,"[MetaD] bias=%s dimensions=%d selection=%s\n",
                MetaD_zqc::BiasSession::name(p_gaussian->backend()),cv_dim,automatic_bias ? "AUTO" : "explicit");
        fflush(output);
    }

    // 输出文件
    first_run = true;
    LOG("Fix init end.");

    // // 设置执行时机
    // force_integrate = 1;
    // vflag = 1;
    // extscalar = 0;
    // extvector = 0;

    // 设置compute_scalar和compute_vector的返回值
    scalar_flag = 1;                 // f_metad -> 偏置势 V_b（见 compute_scalar）
    vector_flag = 1;
    size_vector = cv_dim;            // f_metad[1]..f_metad[cv_dim] -> CV 值
    // Already globally reduced; this flag only tells thermo whether norm yes
    // should divide the extensive SUM energy by the system atom count.
    extscalar = shared_mode && shared_aggregation==MetaD_zqc::SharedAggregation::Sum ? 1 : 0;
    extvector = 0;

    // NPT/能量账本：偏置力必须进 virial；偏置势可进 PE（fix_modify energy）
    // 默认打开，与 fix plumed 一致；可用 fix_modify <id> energy no / virial no 关闭
    energy_global_flag = 1;
    virial_global_flag = 1;
    thermo_energy = 1;
    thermo_virial = 1;

    // virial
    memory->create(a_virial, (1)*6, "metad:GaussianHill:cv_bound");
}

FixMetadynamics::~FixMetadynamics() {
  graph_execution.reset(); // 先释放借用 CV 数据的图。
  memory->destroy(cv_values);
  memory->destroy(cv_history);
  memory->destroy(dVdcvs);
  memory->destroy(f_before_bias);
  memory->destroy(a_virial);
  cv_values = cv_history = dVdcvs = nullptr;
  f_before_bias = nullptr;
  for (auto const& pair : cal_registry) {
      delete pair.second; // 这里会调用 CV 及其派生类的析构函数
  }
  cal_registry.clear(); // 清空 map 容器
  MetaD_zqc::Stru_fact_env::clear_pool(engine_adapter.get());
  engine_adapter.reset();
  // 2. 释放 cv_configs
  if (cv_configs) {
      delete cv_configs;
      cv_configs = nullptr;
  }
  // 偏置 session 拥有历史、网格与文件流。
  if (p_gaussian) {
      delete p_gaussian;
      p_gaussian = nullptr;
  }
  for (auto const& pair : sw_registry) {
      delete pair.second;
  }
  sw_registry.clear();
  if (rec_file) {
      fclose(rec_file);
      rec_file = nullptr;
  }
  if (f_check) {
      fclose(f_check);
      f_check = nullptr;
  }
}

int FixMetadynamics::setmask() {
  return FixConst::POST_FORCE | FixConst::POST_NEIGHBOR | FixConst::MIN_POST_NEIGHBOR;
}

void FixMetadynamics::setup_post_neighbor() { post_neighbor(); }
void FixMetadynamics::min_post_neighbor() { post_neighbor(); }
void FixMetadynamics::post_neighbor() {
  static_cast<MetaD_zqc::LammpsEngineAdapter&>(*engine_adapter).invalidate_topology();
}

void FixMetadynamics::init() try {
  post_neighbor();
  output_cvs.swap(pending_output_cvs);
  pending_output_cvs.clear();
  // if (!atom->tag) error->all(FLERR, "Requires atom style with per-atom positions");
  // Every System init (run / write_restart): LAMMPS drops NeighRequest; re-add then
  // drop cached ensure state. Do NOT Neighbor::build() / ago here (corrupts MEAM).
  neigh_hub.rerequest_all();
  neigh_hub.invalidate_all();
  if(shared_online_bias) shared_online_bias->initialize(update->ntimestep);

  if (first_run) {
    if (p_gaussian) p_gaussian->initialize(update->ntimestep);
    memory->create(cv_values, cv_dim, "metad:cv_values");
    memory->create(cv_history, cv_dim, "metad:cv_history");
    memory->create(dVdcvs, cv_dim, "metad:dVdcvs"); // 修正：dVdcv 应该是其梯度
    for (int k = 0; k < cv_dim; ++k) {
        cv_values[k] = 0.0;
        cv_history[k] = 0.0;
        dVdcvs[k] = 0.0;
    }
    first_run = false;
  }
  // LAMMPS 随后在 Comm::init 中读取这些上限，准备通信缓冲。
  init_cv_communication();
} catch (const std::exception& failure) {
  error->one(FLERR, "MetaD initialization: {}", failure.what());
}

void FixMetadynamics::init_list(int id, NeighList *ptr)
{
  neigh_hub.on_init_list(id, ptr);
}

// 3. 真正偏置力（解析梯度，比数值差分快）
void FixMetadynamics::setup(int vflag) {
  // Apply the restored bias before the first integration step, without
  // advancing sampling history or depositing a hill during run setup.
  evaluate_bias(vflag, false);
}

void FixMetadynamics::post_force(int vflag) {
  evaluate_bias(vflag, true);
}

void FixMetadynamics::evaluate_bias(int vflag, bool advance_history) {
  DEBUG_LOG("evaluate_bias start");

  // 为 virial 记账准备：只累加“本 fix 新加上的力”
  v_init(vflag);
  memory->grow(a_virial, atom->nmax * 6, "metad:a_virial");
  std::fill(a_virial, a_virial + atom->nlocal * 6, 0.0);
  // double **f = atom->f;
  // double **x = atom->x;
  // imageint *image = atom->image;
  // int nlocal = atom->nlocal;
  // if (nlocal > max_f_before_bias) {
  //   memory->destroy(f_before_bias);
  //   max_f_before_bias = atom->nmax;
  //   memory->create(f_before_bias, max_f_before_bias, 3, "metad:f_before_bias");
  // }
  // for (int i = 0; i < nlocal; i++) {
  //   f_before_bias[i][0] = f[i][0];
  //   f_before_bias[i][1] = f[i][1];
  //   f_before_bias[i][2] = f[i][2];
  // }

  // -----calculate cv_compute and add cv_history-----
  try {
  engine_adapter->begin_evaluation();
  // Graph 合批在 backward 内 begin/finish；此处只声明本步 virial 策略，避免嵌套第二套批次。
  static_cast<MetaD_zqc::LammpsEngineAdapter&>(*engine_adapter)
      .configure_force_virial_batch(vflag_global, vflag_atom);
  if (graph_execution) graph_execution->prepare();
  for (auto* cv : output_cvs)
    if (graph_execution && !graph_execution->uses(cv)) cv->prepare_graph_evaluation();
  for (auto const& pair : cal_registry) {
    const std::string& name = pair.first;
    MetaD_zqc::CV* obj = pair.second;
    if (graph_execution && !graph_execution->uses(obj) && !output_cvs.count(obj)) continue;
    DEBUG_LOG("base_calc of %s is start", pair.first.c_str());
    obj->base_calc();
    DEBUG_LOG("base_calc of %s is end", pair.first.c_str());
  }
  if (graph_execution) {
    const auto values = graph_execution->forward();
    if (shared_frozen_bias || shared_online_bias) {
      if (!values.empty()) throw std::invalid_argument("Shared bias DIM_EXPR must return a PerAtom field");
    } else {
      if (values.size()!=static_cast<size_t>(cv_dim))
        throw std::invalid_argument("Global bias DIM expressions must return Global scalars");
      std::copy(values.begin(), values.end(), cv_values);
    }
  } else {
    cv_configs->compute_total_cv();
    for (int ii = 0; ii < cv_dim; ++ii) cv_values[ii] = cv_configs->compute_dim_cv(ii);
  }
  } catch (const std::exception& e) {
    // 单 rank 的 CUDA/NVRTC 失败也要终止整个 communicator，避免其它 rank 卡在规约。
    error->one(FLERR, "MetaD forward: {}", e.what());
  }
  MetaD_zqc::FrozenSharedBias::Result shared_result;
  // 冻结表的求值独立于 setup/积分节拍；输出能量与数组种子来自同一份固定表。
  try {
    if (shared_frozen_bias || shared_online_bias) {
      const auto field=graph_execution->per_atom_output(0);
      const auto slots=graph_execution->selected_slots(shared_frozen_selection);
      if(shared_online_bias) {
        if(advance_history) shared_online_bias->advance(update->ntimestep,field,slots);
        shared_result=shared_online_bias->evaluate(field,slots);
      } else shared_result=shared_frozen_bias->evaluate(field,slots);
      bias_energy=shared_result.energy;
      cv_values[0]=shared_result.mean_coordinate;
    } else {
    if (advance_history) p_gaussian->advance(update->ntimestep,cv_values);
    bias_energy=p_gaussian->evaluate(cv_values,dVdcvs);
    }
  } catch (const std::exception& failure) {
    error->one(FLERR,"MetaD bias: {}",failure.what());
  }

  // DEBUG_LOG("cv_compute = %g, dVdcv = %.g",cv_values[0], dVdcvs[0]);
  // printf("cv_compute = %g, dVdcv = %.g\n",cv_values[0], dVdcvs[0]);
  try {
    if (graph_execution) {
      if (shared_frozen_bias || shared_online_bias) graph_execution->backward({}, {shared_result.gradient});
      else graph_execution->backward(std::vector<double>(dVdcvs, dVdcvs + cv_dim));
    } else {
      for (int ii = 0; ii < cv_dim; ++ii) cv_configs->distribute_dim_bias_force(ii, dVdcvs[ii]);
    }
  } catch (const std::exception& e) {
    error->one(FLERR, "MetaD backward: {}", e.what());
  }

  // 设备路径若已归约并写入全局 virial，禁止再 v_tally，否则重复记账。
  const bool device_virial_complete =
      static_cast<MetaD_zqc::LammpsEngineAdapter&>(*engine_adapter)
          .force_virial_batch_device_complete();
  if (!device_virial_complete && (vflag_global || vflag_atom)) {
    for (int i = 0; i < atom->nlocal; i++) {
      v_tally(i, &a_virial[6*i]);
    }
  }

  // Compute-only CV 也可能仍在读取本步坐标；返回 MD 前结束这些借用。
  for (auto* cv : output_cvs)
    if (!graph_execution || !graph_execution->uses(cv)) cv->device_readiness().wait_host();
  DEBUG_LOG("evaluate_bias_end");
}

void FixMetadynamics::add_hill(double* cv_values){
    if (!p_gaussian) throw std::logic_error("SHARED_FROZEN has no deposition");
    p_gaussian->deposit_now(update->ntimestep,cv_values);
}

// 5. 网格梯度（中心差分）
void FixMetadynamics::get_dVdcv(double *cv_values,
                                    double *dVdcvs) {
  if (!p_gaussian) throw std::logic_error("SHARED_FROZEN requires a per-atom field");
  p_gaussian->evaluate(cv_values,dVdcvs);
}

double FixMetadynamics::compute_scalar() {
    // 与 fix plumed 一致：scalar = 偏置势，供 thermo PE / fix_modify energy
    // CV 请用 f_metad[1] ... f_metad[cv_dim]
    return bias_energy;
}

double FixMetadynamics::compute_vector(int n) {
    // n 是 0-based 索引 (LAMMPS 内部约定)
    if (n < 0 || n >= cv_dim) return 0.0;
    return cv_values[n];
}

void FixMetadynamics::init_cv_communication() {
    comm_forward = 0;
    comm_reverse = engine_adapter ? engine_adapter->reverse_communication_width() : 0;
    for (const auto& entry : cal_registry) {
        const auto& cv = *entry.second;
        const int forward = cv.get_comm_forward_width();
        const int reverse = cv.get_comm_reverse_width();
        if (forward < 0 || reverse < 0)
            error->all(FLERR, "MetaD CV communication width must be nonnegative");
        comm_forward = std::max(comm_forward, forward);
        comm_reverse = std::max(comm_reverse, reverse);
    }
}

void FixMetadynamics::communicate_forward(MetaD_zqc::AtomCommunication& payload) {
    communicate(payload, CommDirection::Forward);
}

void FixMetadynamics::communicate_reverse(MetaD_zqc::AtomCommunication& payload) {
    communicate(payload, CommDirection::Reverse);
}

void FixMetadynamics::communicate(MetaD_zqc::AtomCommunication& payload, CommDirection direction) {
    if (active_comm.payload)
        throw std::logic_error("MetaD: nested CV communication");
    const bool forward = direction == CommDirection::Forward;
    const int width = forward ? payload.get_comm_forward_width() : payload.get_comm_reverse_width();
    const int capacity = forward ? comm_forward : comm_reverse;
    if (width < 0 || width > capacity)
        throw std::logic_error("MetaD: CV communication width exceeds initialized capacity");
    if (width == 0) return;

    active_comm = {&payload, width, direction};
    try {
        // size 指定本轮包宽；Fix::comm_* 保持初始化时的最大容量。
        if (forward) comm->forward_comm(this, width);
        else comm->reverse_comm(this, width);
    } catch (...) {
        active_comm = {};
        throw;
    }
    active_comm = {};
}

int FixMetadynamics::comm_buffer_size(int n, CommDirection direction) const {
    if (!active_comm.payload || active_comm.direction != direction || active_comm.width <= 0)
        throw std::logic_error("MetaD: communication callback has no matching active CV");
    if (n < 0 || n > std::numeric_limits<int>::max()/active_comm.width)
        throw std::overflow_error("MetaD: CV communication count exceeds MPI int range");
    return n*active_comm.width;
}

int FixMetadynamics::pack_forward_comm(int n, int *list, double *buf, int, int*) {
    const int count = comm_buffer_size(n, CommDirection::Forward);
    const int width = active_comm.payload->pack_comm_forward_ubuf(n, list, buf, 0, active_comm.width);
    if (width != active_comm.width)
        throw std::logic_error("MetaD: CV forward pack width mismatch");
    return count;
}

void FixMetadynamics::unpack_forward_comm(int n, int first, double *buf) {
    comm_buffer_size(n, CommDirection::Forward);
    active_comm.payload->unpack_comm_forward_ubuf(n, first, buf, 0, active_comm.width);
}

int FixMetadynamics::pack_reverse_comm(int n, int first, double *buf) {
    const int count = comm_buffer_size(n, CommDirection::Reverse);
    const int width = active_comm.payload->pack_comm_reverse_ubuf(n, first, buf, 0, active_comm.width);
    if (width != active_comm.width)
        throw std::logic_error("MetaD: CV reverse pack width mismatch");
    return count;
}

void FixMetadynamics::unpack_reverse_comm(int n, int *list, double *buf) {
    comm_buffer_size(n, CommDirection::Reverse);
    active_comm.payload->unpack_comm_reverse_ubuf(n, list, buf, 0, active_comm.width);
}

void * FixMetadynamics::extract(const char *key, int &dim){
  std::string keystr(key);
  // compute 在初始化时声明数据依赖；返回 CV 身份作为注册成功标志。
  if (keystr.rfind("colvar_require:", 0) == 0) {
    const std::string query = keystr.substr(15);
    const auto found = cal_registry.find(query.substr(0, query.find('.')));
    if (found == cal_registry.end()) return nullptr;
    pending_output_cvs.insert(found->second);
    dim = 0;
    return found->second;
  }
  // 处理每原子数据请求
  if (keystr.rfind("colvar_peratom:", 0) == 0) {
    std::string full_query = keystr.substr(15); // 拿到 "Q4.stein_q"

    // 🔍 寻找点号位置进行切片
    size_t dot_pos = full_query.find('.');
    if (dot_pos == std::string::npos) return nullptr;

    std::string cv_instance_name = full_query.substr(0, dot_pos); // "Q4"
    std::string prop_name = full_query.substr(dot_pos + 1);       // "stein_q"

    // 从你 Fix 的自注册 Map 容器中路由找到 Q4 实例
    if (cal_registry.find(cv_instance_name) != cal_registry.end()) {
      dim = 1;
      // 调用该 CV 内部由属性名驱动的公共接口（或者让各个 CV 自己解析属性）
      try {
        const auto values=cal_registry[cv_instance_name]->peratom_values(prop_name);
        values.validate();
        values.wait_host();
        if(values.component_count!=1 || values.item_count<size_t(atom->nlocal))
            throw std::invalid_argument("invalid per-atom scalar view: "+full_query);
        // Fix 持有每个请求的连续 Host 暂存，供 compute 按 localid 读取。
        // 分量视图可有 stride；CUDA 数组在这里统一下载，CV 保留原存储布局。
        auto& output=peratom_output[full_query];
        output.resize(std::max(atom->nlocal,1));
        if(values.memory==MetaD_zqc::MemorySpace::Host) {
            for(int localid=0;localid<atom->nlocal;++localid) output[localid]=values(localid);
        } else if(atom->nlocal) {
            const auto status=cudaMemcpy2D(output.data(),sizeof(double),values.data,
                values.item_stride*sizeof(double),sizeof(double),atom->nlocal,cudaMemcpyDeviceToHost);
            if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        }
        return output.data();
      } catch(const std::exception& failure) {
        error->one(FLERR,"MetaD per-atom output: {}",failure.what());
      }
      return nullptr;
    }
  }
  return nullptr;
}

MetaD_zqc::SwitchFunction* FixMetadynamics::get_switching_function(const std::string& name) const {
  auto it = sw_registry.find(name);
  if (it != sw_registry.end()) {
    return it->second;
  }
  return nullptr;
}

// 构造期的缓冲区失败也由宿主处理，保持单 rank 失败时的终止语义。
static Fix *fix_metad(LAMMPS *lmp, int narg, char **arg) {
    try {
        return new FixMetadynamics(lmp, narg, arg);
    } catch (const LAMMPSException&) {
        throw;
    } catch (const std::exception& failure) {
        lmp->error->one(FLERR, "MetaD construction: {}", failure.what());
        throw;
    }
}
static Compute *compute_MetaD_toy(LAMMPS *lmp, int narg, char **arg) {
    return new ComputeMetaDToy(lmp, narg, arg, nullptr);
}

// 插件注册函数（必须命名为 lammpsplugin_init）
extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc) {
    lammpsplugin_t plugin{};
    lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc)regfunc;

    // 注册Fix类型插件
    plugin.version = LAMMPS_VERSION; // 避免加载日志读取未初始化的版本字符串。
    plugin.style = "fix";                 // 插件类型为fix
    plugin.name = "metad";            // 插件名称
    plugin.info = "Metad plugin v1.0";
    plugin.author = "ZQC";
    plugin.creator.v2 = (lammpsplugin_factory2 *)fix_metad; // v2工厂函数
    plugin.handle = handle;
    (*register_plugin)(&plugin, lmp);     // 注册插件
#ifdef ZQC_LAMMPS_KOKKOS
    // -sf kk 和显式 /kk/device 都进入同一适配器，运行时按宿主选择数组。
    plugin.name = "metad/kk";
    (*register_plugin)(&plugin, lmp);
    plugin.name = "metad/kk/device";
    (*register_plugin)(&plugin, lmp);
#endif

    // 注册compute类型的插件
    plugin.style = "compute";             // 🚨 切换插件类型为 compute
    plugin.name = "metad/atom";           // 🚨 设定样式名（用户在 in 脚本里调用的样式）
    plugin.info = "Metadynamics CV Diagnostic Bus Probe v1.0";
    plugin.author = "ZQC";
    plugin.creator.v2 = (lammpsplugin_factory2 *)compute_MetaD_toy; // 🚨 绑定 Compute 的工厂函数
    plugin.handle = handle;               // 共享相同的句柄
    (*register_plugin)(&plugin, lmp);     // 再次调用注册，将其挂载到 LAMMPS 核心总线上
}
