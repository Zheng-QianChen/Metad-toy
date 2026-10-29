#include "zqc_metadtoy_api.h"
#include "zqc_gpumd_engine.cuh"
#include "zqc_cv_graph.cuh"
#include "zqc_bias.h"
#include "CV_Distance.h"
#include "CV_Stru_factor.h"
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>

namespace {
using namespace MetaD_zqc;
constexpr double boltzmann_eV_per_K = 8.617333262145e-5;

std::vector<std::string> tokenize(std::string line) {
    line.erase(line.find('#') == std::string::npos ? line.size() : line.find('#'));
    std::istringstream input(line);
    std::vector<std::string> tokens;
    for (std::string word; input >> word;) tokens.push_back(word);
    return tokens;
}
double real(const std::string& token) {
    size_t consumed = 0;
    const double value = std::stod(token, &consumed);
    if (consumed != token.size() || !std::isfinite(value))
        throw std::invalid_argument("Expected a finite real number: " + token);
    return value;
}
long long integer(const std::string& token) {
    size_t consumed = 0;
    const long long value = std::stoll(token, &consumed);
    if (consumed != token.size()) throw std::invalid_argument("Expected an integer: " + token);
    return value;
}
void require(bool condition, const std::string& message) {
    if (!condition) throw std::invalid_argument(message);
}

template<class T> ArrayView<T> array_from(const ZqcArrayV1& value) {
    return {static_cast<T*>(value.data),value.items,value.components,value.item_stride,
            value.component_stride,value.capacity,MemorySpace::Cuda};
}
GpumdFrame frame_from(const ZqcFrameV1& input) {
    require(input.size == sizeof(ZqcFrameV1) && input.abi_version == ZQC_METADTOY_ABI_VERSION,
            "Incompatible MetaD-toy frame ABI");
    GpumdFrame frame;
    frame.positions=array_from<const double>(input.positions);
    frame.forces=array_from<double>(input.forces);
    frame.virials=array_from<double>(input.virials);
    frame.potential=array_from<double>(input.potential);
    frame.types=array_from<const int>(input.types);
    for (int axis=0;axis<3;++axis) {
        frame.box_lengths[axis]=input.box_lengths[axis];
        frame.periodic[axis]=input.periodic[axis];
    }
    frame.step=input.step;
    frame.stream=static_cast<cudaStream_t>(input.stream);
    return frame;
}
}

struct Runtime {
    struct CVInput {
        std::string name, kind, selection="all";
        AtomId first=0, second=0;
        double cutoff=0, wave_number=0;
        bool group_mean=false;
    };
    std::vector<CVInput> inputs;
    std::map<std::string,std::vector<int>> selections;
    double neighbor_skin=0.3;
    std::string expression, output_path = "COLVAR";
    long long output_interval = 10, warmup_steps = 0;
    int atom_count = 0;
    bool metad = false, prepared = false;
    double temperature = 300, harmonic_center = 0, harmonic_kappa = 0;
    BiasSettings settings;
    ZqcHostV1 host{};
    void log(const std::string& message) const {
        if (host.log) host.log(host.user,message.c_str());
    }
    Logger logger = Logger([this](const Diagnostic& entry) {
        log("[MetaD-toy/"+entry.component+"] "+entry.message);
    });
    GpumdEngineAdapter engine{logger};
    std::map<std::string,SwitchFunction*> switches;
    CVContext context{engine,switches,logger,stdout};
    // 声明顺序保证析构先释放 graph，再释放其借用的 CV 和 engine。
    std::vector<std::unique_ptr<CV>> cvs;
    std::unique_ptr<CVGraphExecution> graph;
    std::unique_ptr<BiasSession> bias;
    std::ofstream output;
    double last_value = 0, last_energy = 0, last_derivative = 0;
    long long last_step = -1;
    cudaStream_t last_stream = nullptr;

    explicit Runtime(const std::string& filename, const ZqcHostV1& callbacks) : host(callbacks) {
        std::ifstream input(filename);
        require(bool(input), "Cannot open " + filename);
        std::set<std::string> commands, cv_names;
        int line_number = 0;
        for (std::string line; std::getline(input,line);) {
            ++line_number;
            const auto tokens = tokenize(line);
            if (tokens.empty()) continue;
            try {
                const auto& command = tokens[0];
                if (command != "cv" && command != "group")
                    require(commands.insert(command).second, "Duplicate command: " + command);
                if (command == "backend") {
                    require(tokens.size()==2 && tokens[1]=="metadtoy", "backend metadtoy");
                } else if (command == "start") {
                    require(tokens.size()==2 && tokens[1]=="fresh",
                            "First GPUMD bridge supports start fresh; restart needs MD state restoration");
                } else if (command == "output") {
                    require(tokens.size()==3, "output <file> <interval>");
                    output_path=tokens[1]; output_interval=integer(tokens[2]);
                    require(output_interval>0, "Output interval must be positive");
                } else if (command == "warmup") {
                    require(tokens.size()==2, "warmup <steps>");
                    warmup_steps=integer(tokens[1]);
                    require(warmup_steps>=0, "Warmup must be nonnegative");
                } else if (command == "neighbor_skin") {
                    require(tokens.size()==2,"neighbor_skin <distance>");
                    neighbor_skin=real(tokens[1]);
                    require(neighbor_skin>=0,"neighbor_skin must be nonnegative");
                } else if (command == "group") {
                    require(tokens.size()>=4 && tokens[2]=="atoms","group <name> atoms <zero-based rows...>");
                    require(tokens[1]!="all" && !selections.count(tokens[1]),"Duplicate/reserved group name");
                    auto& rows=selections[tokens[1]];
                    for(size_t index=3;index<tokens.size();++index) {
                        const auto row=integer(tokens[index]);
                        require(row>=0 && row<std::numeric_limits<int>::max(),"Invalid group row");
                        rows.push_back(static_cast<int>(row));
                    }
                } else if (command == "cv") {
                    require(tokens.size()>=3,"cv <name> distance|strufactor ...");
                    require(cv_names.insert(tokens[1]).second,"Duplicate CV name");
                    CVInput cv; cv.name=tokens[1]; cv.kind=tokens[2];
                    if(cv.kind=="distance") {
                        require(tokens.size()==6 && tokens[3]=="atoms",
                                "cv <name> distance atoms <zero-based row> <zero-based row>");
                        const auto first=integer(tokens[4]), second=integer(tokens[5]);
                        require(first>=0 && second>=0 && first!=second &&
                                first<std::numeric_limits<int>::max() && second<std::numeric_limits<int>::max(),
                                "DISTANCE needs two distinct nonnegative row indices");
                        cv.first=static_cast<AtomId>(first+1); cv.second=static_cast<AtomId>(second+1);
                    } else {
                        require(cv.kind=="strufactor" && (tokens.size()==9 || tokens.size()==11) &&
                                tokens[3]=="group" && tokens[5]=="cutoff" && tokens[7]=="q",
                                "cv <name> strufactor group <group> cutoff <radius> q <wave number> [reduce raw|mean]");
                        cv.selection=tokens[4]; cv.cutoff=real(tokens[6]); cv.wave_number=real(tokens[8]);
                        require(cv.cutoff>0 && std::isfinite(cv.cutoff*cv.cutoff) && cv.wave_number>0,
                                "StruFactor cutoff and q must be positive");
                        if(tokens.size()==11) {
                            require(tokens[9]=="reduce" && (tokens[10]=="raw" || tokens[10]=="mean"),
                                    "reduce raw|mean");
                            cv.group_mean=tokens[10]=="mean";
                        }
                    }
                    inputs.push_back(std::move(cv));
                } else if (command == "expression") {
                    require(tokens.size()>1, "expression <Graph expression>");
                    for (size_t i=1;i<tokens.size();++i) expression += tokens[i]+" ";
                } else if (command == "history") {
                    require(tokens.size()==2, "history <HILLS file>");
                    settings.history_path=tokens[1];
                } else if (command == "grid") {
                    require(tokens.size()==4, "grid <lower> <upper> <bins>");
                    const auto bins=integer(tokens[3]);
                    require(bins>0 && bins<=std::numeric_limits<int>::max(), "Invalid grid bins");
                    settings.grid_axes={{real(tokens[1]),real(tokens[2]),0,static_cast<int>(bins),0}};
                } else if (command == "bias") {
                    require(tokens.size()>1, "bias metad ... or bias harmonic ...");
                    if (tokens[1]=="metad") {
                        require(tokens.size()==12 && tokens[2]=="temperature" && tokens[4]=="sigma" &&
                            tokens[6]=="height" && tokens[8]=="biasfactor" && tokens[10]=="pace",
                            "bias metad temperature <K> sigma <width> height <eV> biasfactor <gamma> pace <steps>");
                        metad=true;
                        temperature=real(tokens[3]);
                        settings.hills.axes={{real(tokens[5]),0}};
                        settings.hills.initial_height=real(tokens[7]);
                        settings.hills.bias_factor=real(tokens[9]);
                        settings.hills.well_tempered=true;
                        settings.hills.thermal_energy=temperature*boltzmann_eV_per_K;
                        settings.pace=integer(tokens[11]);
                    } else {
                        require(tokens.size()==6 && tokens[1]=="harmonic" &&
                            tokens[2]=="center" && tokens[4]=="kappa",
                            "bias harmonic center <value> kappa <eV/value^2>");
                        harmonic_center=real(tokens[3]); harmonic_kappa=real(tokens[5]);
                        require(harmonic_kappa>=0, "Harmonic kappa must be nonnegative");
                    }
                } else throw std::invalid_argument("Unknown command: "+command);
            } catch (const std::exception& error) {
                throw std::invalid_argument(filename+":"+std::to_string(line_number)+": "+error.what());
            }
        }
        require(commands.count("backend") && commands.count("start") && commands.count("bias"),
                "backend, start and bias are required");
        require(!inputs.empty() && !expression.empty(), "At least one cv and an expression are required");
        if (metad) {
            require(!settings.grid_axes.empty(), "The first GPUMD MetaD bridge requires a 1D grid");
            settings.grid_axes[0].sigma=settings.hills.axes[0].sigma;
            settings.backend=BiasBackend::Grid;
            require(output_path!=settings.history_path, "COLVAR and HILLS must use different files");
        } else {
            require(!commands.count("grid") && !commands.count("history") && warmup_steps==0,
                    "grid/history/warmup apply to MetaD");
        }
    }
    ~Runtime() {
        // 环境池借用 engine；先释放 Graph/CV，再释放属于本实例的共享几何。
        graph.reset();
        cvs.clear();
        Stru_fact_env::clear_pool(&engine);
    }
    void pre_run(const ZqcFrameV1&, const ZqcRunV1&);
    void evaluate(const ZqcFrameV1&, bool);
    void end_of_step(long long);
    void post_run();
    void write_row(long long step) {
        output << step << ' ' << last_value << ' ' << last_energy << ' ' << last_derivative << '\n';
        if (!output) throw std::runtime_error("Failed writing "+output_path);
    }
};


void Runtime::pre_run(const ZqcFrameV1& frame, const ZqcRunV1& run) {
        auto& state=*this;
        require(!state.prepared, "A GPUMD action owns one run");
        const auto borrowed_frame=frame_from(frame);
        require(run.size==sizeof(ZqcRunV1), "Incompatible MetaD-toy run ABI");
        require(run.has_thermostat==0 || run.has_thermostat==1, "Invalid thermostat flag");
        if (run.has_thermostat && state.metad)
            require(std::isfinite(run.temperature) && std::abs(run.temperature-state.temperature)<1e-8,
                    "MetaD temperature differs from GPUMD thermostat");
        require(frame.positions.items>0 && frame.positions.items<=std::numeric_limits<int>::max(), "Invalid atom count");
        state.atom_count=static_cast<int>(frame.positions.items);
        state.engine.bind_frame(borrowed_frame);
        state.last_stream=borrowed_frame.stream;
        state.engine.set_neighbor_skin(state.neighbor_skin);
        for(const auto& group:state.selections)
            state.engine.define_selection(group.first,group.second);
        std::vector<CVGraphSymbol> symbols;
        for (const auto& input:state.inputs) {
            if(input.kind=="distance") {
                require(input.first<=static_cast<AtomId>(state.atom_count) &&
                        input.second<=static_cast<AtomId>(state.atom_count),
                        "DISTANCE row exceeds model.xyz atom count");
                state.cvs.emplace_back(new Distance(state.context,input.first,input.second));
                symbols.push_back({input.name,state.cvs.back().get(),"value"});
            } else {
                StruFactorRequest request;
                request.selection=state.engine.selection_id(input.selection);
                request.selection_name=input.selection;
                request.cutoff_r=input.cutoff;
                request.q_factor=input.wave_number;
                auto* environment=Stru_fact_env::get_or_create(state.context,request);
                state.cvs.emplace_back(new Stru_factor(state.context,environment,input.wave_number,128));
                // mean 是 Graph 的组均值节点：分母仅统计该 CV 的 owned 组成员。
                symbols.push_back({input.name,state.cvs.back().get(),input.group_mean ? "AVE" : "RAW"});
            }
        }
        state.graph.reset(new CVGraphExecution(state.engine,std::move(symbols),{state.expression}));
        if (state.metad) {
            BiasHost host;
            host.log=[this](const std::string& message) { log("[MetaD-toy/bias] "+message); };
            state.bias.reset(new BiasSession(state.settings,std::move(host)));
            state.bias->initialize(0);
        }
        state.output.open(state.output_path,std::ios::trunc);
        require(bool(state.output), "Cannot open "+state.output_path);
        state.output << std::setprecision(17) << "# step value bias_eV dV_dvalue\n";
        state.prepared=true;
        state.log("Use MetaD-toy shared runtime ABI 1: "+std::to_string(state.inputs.size())+
                  " CV(s), Graph expression "+state.expression);
}
void Runtime::evaluate(const ZqcFrameV1& frame, bool integrating) {
        auto& state=*this;
        const auto borrowed_frame=frame_from(frame);
        const auto step=frame.step;
        require(state.prepared && frame.positions.items==static_cast<uint64_t>(state.atom_count),
                "GPUMD atom count changed during run");
        state.engine.bind_frame(borrowed_frame);
        state.last_stream=borrowed_frame.stream;
        state.engine.begin_evaluation();
        state.graph->prepare();
        for (const auto& cv:state.cvs) if (state.graph->uses(cv.get())) cv->base_calc();
        const auto values=state.graph->forward();
        require(values.size()==1 && std::isfinite(values[0]), "Expected one finite Graph output");
        state.last_value=values[0];
        if (state.metad) {
            const auto& axis=state.settings.grid_axes[0];
            require(values[0]>=axis.lower && values[0]<=axis.upper, "Graph value left the bias grid");
            // setup_force 求值初始力；真实积分步才按 PACE 沉积，warmup 期间保持空历史。
            if (integrating && step>state.warmup_steps) state.bias->advance(step,values.data());
            state.last_energy=state.bias->evaluate(values.data(),&state.last_derivative);
        } else {
            const double offset=values[0]-state.harmonic_center;
            state.last_energy=0.5*state.harmonic_kappa*offset*offset;
            state.last_derivative=state.harmonic_kappa*offset;
        }
        state.graph->backward({state.last_derivative});
        state.engine.finish(state.last_energy);
        state.last_step=step;
        if (!integrating) state.write_row(step);
}
void Runtime::end_of_step(long long step) {
        auto& state=*this;
        require(state.prepared && state.last_step>=0 && state.last_step==step, "GPUMD post_force/end_of_step mismatch");
        if (step%state.output_interval==0) state.write_row(step);
}
void Runtime::post_run() {
        require(prepared,"MetaD-toy runtime has not been prepared");
        const auto status=cudaStreamSynchronize(last_stream);
        if (status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        output.flush();
        if (!output) throw std::runtime_error("Failed flushing COLVAR");
        if (bias) log("[MetaD-toy/bias] "+std::to_string(bias->history().depositions())+" hills deposited");
        if(engine.neighbor_rebuild_count())
            log("[MetaD-toy/neighbors] "+std::to_string(engine.neighbor_rebuild_count())+" candidate rebuilds");
}

namespace {
// 所有异常在共享库一侧转换，调用方只接收状态码与自己的错误缓冲。
template<class Function> int boundary(char* error, uint64_t capacity, Function&& function) noexcept {
    if (error && capacity) error[0]='\0';
    try { function(); return 0; }
    catch (const std::exception& exception) {
        if (error && capacity) std::snprintf(error,static_cast<size_t>(capacity),"%s",exception.what());
    } catch (...) {
        if (error && capacity) std::snprintf(error,static_cast<size_t>(capacity),"Unknown MetaD-toy exception");
    }
    return 1;
}
Runtime& runtime(void* handle) {
    require(handle!=nullptr,"Null MetaD-toy handle");
    return *static_cast<Runtime*>(handle);
}
int create(const char* filename,const ZqcHostV1* host,void** handle,char* error,uint64_t capacity) noexcept {
    if (handle) *handle=nullptr;
    return boundary(error,capacity,[&] {
        require(filename && host && handle,"Null create argument");
        require(host->size==sizeof(ZqcHostV1),"Incompatible MetaD-toy host ABI");
        *handle=new Runtime(filename,*host);
    });
}
int pre_run(void* handle,const ZqcFrameV1* frame,const ZqcRunV1* run,char* error,uint64_t capacity) noexcept {
    return boundary(error,capacity,[&] {
        require(frame && run,"Null pre_run argument"); runtime(handle).pre_run(*frame,*run);
    });
}
int evaluate(void* handle,const ZqcFrameV1* frame,int32_t integrating,char* error,uint64_t capacity) noexcept {
    return boundary(error,capacity,[&] {
        require(frame && (integrating==0 || integrating==1),"Invalid evaluate argument");
        runtime(handle).evaluate(*frame,integrating!=0);
    });
}
int end_of_step(void* handle,int64_t step,char* error,uint64_t capacity) noexcept {
    return boundary(error,capacity,[&] { runtime(handle).end_of_step(step); });
}
int post_run(void* handle,char* error,uint64_t capacity) noexcept {
    return boundary(error,capacity,[&] { runtime(handle).post_run(); });
}
void destroy(void* handle) noexcept { delete static_cast<Runtime*>(handle); }
const ZqcApiV1 api={sizeof(ZqcApiV1),ZQC_METADTOY_ABI_VERSION,
                   create,pre_run,evaluate,end_of_step,post_run,destroy};
}
extern "C" const ZqcApiV1* zqc_metadtoy_get_api(uint32_t requested_version) {
    return requested_version==ZQC_METADTOY_ABI_VERSION ? &api : nullptr;
}
