// 不链接 LAMMPS 的生产 CV 验证：后端改变布局和类型基数，计算结果应保持一致。
#include "zqc_cv.h"
#include "zqc_cv_assembly.cuh"
#include "zqc_switch_function.h"
#include "CV_Stru_factor.h"
#include <array>
#include <cmath>
#include <iostream>
#include <sstream>

using namespace MetaD_zqc;
namespace {
constexpr size_t atoms=6;
// 在 MD 流中延迟发布新坐标，热路径不再依赖分配或默认流的隐式完成。
__global__ void delay_producer(unsigned long long cycles) {
    const auto start=clock64();
    while (clock64()-start<cycles) {}
}
void require(bool ok,const std::string& message) { if(!ok) throw std::runtime_error(message); }
template<class T> ArrayView<const T> host(const std::vector<T>& values,size_t count,size_t stride=1) {
    return {values.data(),count,1,stride,1,values.size(),MemorySpace::Host};
}
struct TestEngine : EngineAdapter {
    CudaExecutionStream producer;
    bool soa;
    int type_begin;
    std::uint64_t evaluation=0;
    std::vector<double> xyz{0.1,.2,.3, 1.2,.3,.2, .4,1.5,.2, .3,.5,1.6, 1.3,1.2,1.1, 2.1,.7,1.5};
    std::vector<double> position_storage,forces,virials;
    std::vector<int> types,flags,members;
    std::vector<AtomId> ids{11,17,23,29,31,37};
    GpuBuffer<double> device_positions, staged_positions;
    GpuBuffer<int> device_types,device_flags;
    struct Neighbors {
        std::vector<int> centers,host_centers,offsets,entries;
        GpuBuffer<int> device_centers,device_offsets,device_entries;
    };
    std::vector<std::unique_ptr<Neighbors>> lists;
    TestEngine(bool soa,int first):soa(soa),type_begin(first) {
        types.resize(atoms*2,-99);flags.resize(atoms*2,-99);members.resize(atoms*2,-99);
        for(size_t atom=0;atom<atoms;++atom) {
            types[2*atom]=int(atom%3)+first;flags[2*atom]=1;members[2*atom]=int(atom);
        }
        device_types.grow_to(types.size(),__FILE__,__LINE__);device_types.upload_from(types.data(),types.size());
        device_flags.grow_to(flags.size(),__FILE__,__LINE__);device_flags.upload_from(flags.data(),flags.size());
    }
    ~TestEngine() { Stru_fact_env::clear_pool(this); }
    void begin_evaluation() override {
        ++evaluation;forces.assign(atoms*3,0);virials.assign(atoms*9,0);
        position_storage.assign(soa?(atoms+3)*3:atoms*3,-999);
        for(size_t atom=0;atom<atoms;++atom) for(size_t axis=0;axis<3;++axis)
            position_storage[soa?atom+(atoms+3)*axis:3*atom+axis]=xyz[3*atom+axis];
        device_positions.grow_to(position_storage.size(),__FILE__,__LINE__);
        staged_positions.grow_to(position_storage.size(),__FILE__,__LINE__);
        staged_positions.upload_from(position_storage.data(),position_storage.size());
        delay_producer<<<1,1,0,producer.get()>>>(1000000);
        execution_check(cudaMemcpyAsync(device_positions.ptr,staged_positions.ptr,
            position_storage.size()*sizeof(double),cudaMemcpyDeviceToDevice,producer.get()));
    }
    std::uint64_t evaluation_id() const override { return evaluation; }
    int owned_atom_count() const override { return atoms; }
    int total_atom_count() const override { return atoms; }
    ArrayView<const double> positions(MemorySpace memory) override {
        return {memory==MemorySpace::Host?position_storage.data():device_positions.ptr,atoms,3,
                soa?1:size_t(3),soa?atoms+3:1,position_storage.size(),memory,
                memory==MemorySpace::Cuda?producer.ready():ViewReadiness{}};
    }
    ArrayView<const int> atom_types(MemorySpace memory) override {
        return {memory==MemorySpace::Host?types.data():device_types.ptr,atoms,1,2,1,types.size(),memory};
    }
    ArrayView<const AtomId> atom_ids() override { return host(ids,atoms); }
    int atom_type_count() const override { return 3; }
    int atom_type_begin() const override { return type_begin; }
    int atom_type_id(const std::string& label) const override {
        if(label=="A") return type_begin;if(label=="B") return type_begin+1;if(label=="C") return type_begin+2;
        throw std::invalid_argument("unknown test type");
    }
    SelectionId selection_id(const std::string& label) const override {
        require(label=="all","unknown selection");return 77; // 身份不受 bitmask 宽度限制。
    }
    SelectionView selection_members(SelectionId id) override { return {id,atoms,host(members,atoms,2)}; }
    ArrayView<const int> selection_flags(SelectionId,MemorySpace memory) override {
        return {memory==MemorySpace::Host?flags.data():device_flags.ptr,atoms,1,2,1,flags.size(),memory};
    }
    void communicate_forward(AtomCommunication&) override {} // 单 rank 无 ghost。
    void communicate_reverse(AtomCommunication&) override {}
    int register_neighbors(const NeighborRequest&) override {
        auto list=std::make_unique<Neighbors>();list->offsets.push_back(0);
        for(size_t atom=0;atom<atoms;++atom) {
            list->centers.push_back(atom);list->host_centers.push_back(atom);list->host_centers.push_back(-99);
            for(size_t neighbor=0;neighbor<atoms;++neighbor) if(neighbor!=atom) list->entries.push_back(neighbor);
            list->offsets.push_back(list->entries.size());
        }
        const auto upload=[](auto& device,const auto& values) {
            device.grow_to(values.size(),__FILE__,__LINE__);device.upload_from(values.data(),values.size());
        };
        upload(list->device_centers,list->centers);upload(list->device_offsets,list->offsets);upload(list->device_entries,list->entries);
        lists.push_back(std::move(list));return int(lists.size()-1);
    }
    NeighborListView prepare_neighbors(int handle) override {
        auto& list=*lists.at(handle);
        return {host(list.host_centers,atoms,2),list.device_centers.view(atoms),list.device_offsets.view(atoms+1),
            list.device_entries.view(list.entries.size()),atoms,atoms,evaluation,1,NeighborCoordinates::PeriodicImages,
            host(list.offsets,atoms+1),host(list.entries,list.entries.size())};
    }
    void sum(double*,size_t) override {}
    void minimum_image(double[3]) const override {}
    void read_owned_atoms(const AtomId* requested,size_t count,double* output,int* indices) override {
        std::fill_n(output,count*4,0.0);std::fill_n(indices,count,-1);
        for(size_t endpoint=0;endpoint<count;++endpoint) for(size_t atom=0;atom<atoms;++atom)
            if(requested[endpoint]==ids[atom]) {
                indices[endpoint]=atom;output[endpoint*4+3]=1;
                for(size_t axis=0;axis<3;++axis) output[endpoint*4+axis]=xyz[3*atom+axis];
            }
    }
    static double read(ArrayView<const double> view,size_t atom,size_t component) {
        if(view.memory==MemorySpace::Host) return view(atom,component);
        double value;
        auto status=cudaMemcpy(&value,view.data+atom*view.item_stride+component*view.component_stride,sizeof(double),cudaMemcpyDeviceToHost);
        require(status==cudaSuccess,cudaGetErrorString(status));return value;
    }
    void commit_force_virial(ArrayView<const double> force,VirialView virial) override {
        force.validate();virial.validate();
        force.wait_host();virial.values.wait_host();
        for(size_t atom=0;atom<atoms;++atom) {
            for(size_t axis=0;axis<3;++axis) forces[3*atom+axis]+=read(force,atom,axis);
            for(size_t component=0;component<9;++component)
                virials[9*atom+component]+=read(virial.values,atom,
                    virial.representation==VirialRepresentation::Symmetric6 && component>=6?component-3:component);
        }
    }
    void add_owned_force_virial(int atom,const double force[3],VirialView virial) override {
        for(size_t axis=0;axis<3;++axis) forces[3*atom+axis]+=force[axis];
        for(size_t component=0;component<9;++component) virials[9*atom+component]+=virial.full_component(0,component);
    }
};

struct Result { double energy;std::vector<double> forces,virials; };
Result run(const std::string& definition,bool soa,int type_begin,bool finite_difference) {
    TestEngine engine(soa,type_begin);
    std::map<std::string,SwitchFunction*> switches;
    std::vector<std::string> sw_words{"SW_FUNC","FERMI","r_0","1.8","alpha","-4"};
    std::vector<char*> arguments;for(auto& word:sw_words) arguments.push_back(word.data());
    int index=0;
    std::unique_ptr<SwitchFunction> radial(SwitchFunction::create(arguments.size(),arguments.data(),index));
    switches["radial"]=radial.get();
    CVContext context{engine,switches,{}};
    std::istringstream stream(definition);std::vector<std::string> words;std::string word;
    while(stream>>word) words.push_back(word);
    arguments.clear();for(auto& entry:words) arguments.push_back(entry.data());index=0;
    std::unique_ptr<CV> cv(CVFactory::create(words[0],context,arguments.size(),arguments.data(),index));
    require(index==int(words.size()),"CV parser left unused arguments");
    engine.begin_evaluation();
    const bool global=cv->supports_global_graph();
    CVAtomLayout layout;
    if(!global) {
        layout.rebuild(atoms,0,{engine.selection_members(77)});cv->bind_assembly(layout.group(77));
    }
    GpuBuffer<double> leaf(atoms*2),seed(atoms*2);
    std::vector<double> seeds(atoms*2,-123),values(atoms*2);
    for(size_t atom=0;atom<atoms;++atom) seeds[2*atom]=.3+.11*atom;
    seed.upload_from(seeds.data(),seeds.size());
    ArrayView<double> leaf_view{leaf.ptr,atoms,1,2,1,leaf.capacity,MemorySpace::Cuda};
    ArrayView<const double> seed_view{seed.ptr,atoms,1,2,1,seed.capacity,MemorySpace::Cuda};
    const auto evaluate=[&](bool backward) {
        engine.begin_evaluation();cv->prepare_graph_evaluation();cv->base_calc();
        if(global) { if(backward) cv->apply_global_adjoint(.7);return .7*cv->graph_global_value(); }
        cv->assemble_values(leaf_view);cv->device_readiness().wait_host();leaf.download_to(values.data(),values.size());
        double energy=0;for(size_t atom=0;atom<atoms;++atom) energy+=seeds[2*atom]*values[2*atom];
        if(backward) cv->apply_adjoint(seed_view);return energy;
    };
    Result result{evaluate(true),engine.forces,engine.virials};
    // 各 CV 的标量输出具有旋转不变性，任意固定原子种子加权后也应保持总内力矩为零。
    // 汇总九分量再比较转置项，允许多体 CV 的单个原子具有非对称 virial。
    std::array<double,9> total_virial{};
    double virial_scale = 1;
    for(size_t atom=0;atom<atoms;++atom) for(size_t component=0;component<9;++component) {
        total_virial[component] += result.virials[9*atom+component];
        virial_scale += std::abs(result.virials[9*atom+component]);
    }
    for(size_t component=3;component<6;++component)
        require(std::abs(total_virial[component]-total_virial[component+3]) < 1e-11*virial_scale,
                "total virial violates rotational invariance: "+definition);
    if(finite_difference) {
        const auto initial=engine.xyz;const double step=1e-5;
        for(size_t entry=0;entry<engine.xyz.size();++entry) {
            engine.xyz=initial;engine.xyz[entry]+=step;const double plus=evaluate(false);
            engine.xyz[entry]-=2*step;const double minus=evaluate(false);
            require(std::abs((plus-minus)/(2*step)+result.forces[entry])<2e-6,"coordinate finite difference: "+definition);
        }
        const int position_axis[9]={0,1,2,0,0,1,1,2,2},force_axis[9]={0,1,2,1,2,2,0,0,1};
        for(size_t component=0;component<9;++component) {
            double energy[2];
            for(int side=0;side<2;++side) {
                engine.xyz=initial;
                for(size_t atom=0;atom<atoms;++atom)
                    engine.xyz[3*atom+force_axis[component]]+=(side?-step:step)*initial[3*atom+position_axis[component]];
                energy[side]=evaluate(false);
            }
            double analytic=0;for(size_t atom=0;atom<atoms;++atom) analytic+=result.virials[9*atom+component];
            require(std::abs((energy[0]-energy[1])/(2*step)+analytic)<3e-6,"nine-component virial finite difference: "+definition);
        }
    }
    return result;
}
}
int main() {
    try {
        const std::vector<std::string> definitions{
            "DISTANCE 11 17", "STRU_FACTOR all cutoff_r 3.5 q_factor 1.7",
            "STRU_FACTOR all cutoff_r 3.5 q_factor 1.7 Chemical Chem_map (A,1) (B,-0.7) (C,0.35)",
            "WEIGHT_CHEM all cutoff_r 3.5 Chem_map (A,1) (B,-0.7) (C,0.35)",
            "CSS all CHANNEL A radial 2.0 1.0 CHANNEL B radial 1.0 1.5 CHANNEL C radial 1.5 1.1",
            "STEINH Q 3 all SW_FUNC_r radial", "STEINH Q 4 all SW_FUNC_r radial", "STEINH Q 6 all SW_FUNC_r radial",
            "STEINH L 3 all SW_FUNC_r radial", "STEINH L 4 all SW_FUNC_r radial", "STEINH L 6 all SW_FUNC_r radial"};
        for(const auto& definition:definitions) {
            const auto reference=run(definition,false,1,false),strided=run(definition,true,0,true);
            require(std::abs(reference.energy-strided.energy)<1e-12,"layout/type-base value mismatch");
            for(size_t entry=0;entry<reference.forces.size();++entry)
                require(std::abs(reference.forces[entry]-strided.forces[entry])<1e-11,"layout/type-base force mismatch");
            for(size_t entry=0;entry<reference.virials.size();++entry)
                require(std::abs(reference.virials[entry]-strided.virials[entry])<1e-11,"layout/type-base virial mismatch");
            std::cout<<"PASS: "<<definition<<std::endl;
        }
    } catch(const std::exception& failure) { std::cerr<<failure.what()<<std::endl;return 1; }
}
