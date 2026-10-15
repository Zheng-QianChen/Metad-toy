// 独立布局实验：直接编译生产实现，以共享完全相同的径向公式和中心线程基准。
#define ZQC_STRUFACTOR_KERNELS_ONLY
#include "../../src/cvs/CVs_Strufactor.cu"

#include <array>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <unordered_map>

using MetaD_zqc::GpuBuffer;
using MetaD_zqc::StruFactorData;
using MetaD_zqc::StruFactorKernelView;
using LAMMPS_NS::tagint;
using Position = std::array<double,3>;

void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

namespace MetaD_zqc {
namespace {
// pair 写两个端点；沿用生产核的力符号和每端点一半 virial 约定。
__device__ __forceinline__ void scatter_pair(int center_local_id, int neighbor_local_id,
    const double* displacement, double scale, double* forces, double* virials) {
    double pair_force[3];
    for (int component=0; component<3; ++component) {
        pair_force[component]=scale*displacement[component];
        atomicAdd(forces+3*center_local_id+component,pair_force[component]);
        atomicAdd(forces+3*neighbor_local_id+component,-pair_force[component]);
    }
    if (virials) {
        const double half_virial[6]={-.5*displacement[0]*pair_force[0],
            -.5*displacement[1]*pair_force[1],-.5*displacement[2]*pair_force[2],
            -.5*displacement[0]*pair_force[1],-.5*displacement[0]*pair_force[2],
            -.5*displacement[1]*pair_force[2]};
        for (int component=0; component<6; ++component) {
            atomicAdd(virials+6*center_local_id+component,half_virial[component]);
            atomicAdd(virials+6*neighbor_local_id+component,half_virial[component]);
        }
    }
}

// 槽位->中心表可随候选邻居布局建立；有效邻居数变化由 kernel 内判断。
__global__ void directed_pair_kernel(StruFactorKernelView geometry, int slot_count,
    const int* slot_centers, double wave_number, double cutoff_squared,
    const double* center_adjoint, double* forces, double* virials) {
    const int slot=blockIdx.x*blockDim.x+threadIdx.x;
    if (slot>=slot_count) return;
    const int center_index=slot_centers[slot];
    if (slot-geometry.d_group_numneigh[center_index]>=geometry.d_neigh_in_cutoff_r[center_index]) return;
    const double seed=center_adjoint[center_index];
    if (seed==0) return;
    const double* displacement=geometry.d_group_dminneigh+4*slot;
    const double scale=seed*radial(displacement[3],wave_number,cutoff_squared).derivative_over_r;
    scatter_pair(geometry.d_group_indices[center_index],geometry.d_calculated_numneigh[slot],
                 displacement,scale,forces,virials);
}

// 两个端点都有中心贡献时相加种子；单向边保留自己的种子。
// 此实验的映射在 Host 从当前有效邻居构建，构建成本单独记录。
__global__ void half_pair_kernel(StruFactorKernelView geometry, int pair_count,
    const int* first_centers, const int* second_centers, const int* pair_slots,
    double wave_number, double cutoff_squared, const double* center_adjoint,
    double* forces, double* virials) {
    const int pair_index=blockIdx.x*blockDim.x+threadIdx.x;
    if (pair_index>=pair_count) return;
    const int center_index=first_centers[pair_index];
    const int second_center=second_centers[pair_index];
    double seed=center_adjoint[center_index];
    if (second_center>=0) seed+=center_adjoint[second_center];
    if (seed==0) return;
    const int slot=pair_slots[pair_index];
    const double* displacement=geometry.d_group_dminneigh+4*slot;
    const double scale=seed*radial(displacement[3],wave_number,cutoff_squared).derivative_over_r;
    scatter_pair(geometry.d_group_indices[center_index],geometry.d_calculated_numneigh[slot],
                 displacement,scale,forces,virials);
}
} // namespace
} // namespace MetaD_zqc

struct Fixture {
    int atom_count=0;
    double wave_number=1.7, cutoff=2.5;
    std::vector<Position> positions;
    std::vector<tagint> centers,offsets,neighbors;
    std::vector<int> counts,slot_centers,first_centers,second_centers,pair_slots;
    std::vector<double> geometry,seeds;
    double mapping_ms=0;
    int edge_count=0;
};

// 在生产布局中保留段尾空槽，检查 pair 实现是否误读无效位置。
void append_center(Fixture& fixture,int center_local_id,
                   const std::vector<std::pair<int,Position>>& neighbors,int padding) {
    const int center_index=fixture.centers.size();
    fixture.centers.push_back(center_local_id);
    fixture.offsets.push_back(fixture.neighbors.size());
    fixture.counts.push_back(neighbors.size());
    fixture.edge_count+=neighbors.size();
    fixture.seeds.push_back(center_local_id%17==0 ? 0.0 : std::sin(.13*center_local_id)+.2);
    for (const auto& neighbor:neighbors) {
        const auto& displacement=neighbor.second;
        const double squared=std::inner_product(displacement.begin(),displacement.end(),displacement.begin(),0.0);
        fixture.neighbors.push_back(neighbor.first);
        fixture.geometry.insert(fixture.geometry.end(),{displacement[0],displacement[1],displacement[2],squared});
    }
    for (int padding_index=0; padding_index<padding; ++padding_index) {
        fixture.neighbors.push_back(-123);
        fixture.geometry.insert(fixture.geometry.end(),{NAN,NAN,NAN,NAN});
    }
    fixture.slot_centers.insert(fixture.slot_centers.end(),neighbors.size()+padding,center_index);
}

void finish_fixture(Fixture& fixture) {
    fixture.offsets.push_back(fixture.neighbors.size());
    const auto started=std::chrono::steady_clock::now();
    std::unordered_map<unsigned long long,int> directed_slots;
    auto key=[](int first,int second) {
        return (static_cast<unsigned long long>(static_cast<unsigned int>(first))<<32)
               |static_cast<unsigned int>(second);
    };
    directed_slots.reserve(fixture.edge_count);
    for (int center_index=0;center_index<static_cast<int>(fixture.centers.size());++center_index)
        for(int neighbor_offset=0;neighbor_offset<fixture.counts[center_index];++neighbor_offset) {
            const int slot=fixture.offsets[center_index]+neighbor_offset;
            directed_slots.emplace(key(fixture.centers[center_index],fixture.neighbors[slot]),slot);
        }
    for (int center_index=0;center_index<static_cast<int>(fixture.centers.size());++center_index)
        for(int neighbor_offset=0;neighbor_offset<fixture.counts[center_index];++neighbor_offset) {
            const int slot=fixture.offsets[center_index]+neighbor_offset;
            auto reverse=directed_slots.find(key(fixture.neighbors[slot],fixture.centers[center_index]));
            if(reverse!=directed_slots.end() && reverse->second<slot) continue;
            fixture.first_centers.push_back(center_index);
            fixture.second_centers.push_back(reverse==directed_slots.end()?-1:fixture.slot_centers[reverse->second]);
            fixture.pair_slots.push_back(slot);
        }
    fixture.mapping_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
}

Fixture make_small_fixture(const std::vector<Position>& positions,const std::vector<int>& centers) {
    Fixture fixture;
    fixture.positions=positions;fixture.atom_count=positions.size();
    for(int center:centers) {
        std::vector<std::pair<int,Position>> neighbors;
        for(int neighbor=0;neighbor<fixture.atom_count;++neighbor) {
            if(neighbor==center)continue;
            Position displacement{};
            double squared=0;
            for(int component=0;component<3;++component) {
                displacement[component]=positions[neighbor][component]-positions[center][component];
                squared+=displacement[component]*displacement[component];
            }
            if(squared<=fixture.cutoff*fixture.cutoff)neighbors.push_back({neighbor,displacement});
        }
        append_center(fixture,center,neighbors,3+center%3);
    }
    finish_fixture(fixture);return fixture;
}

Fixture make_fcc_fixture(int cells,double cutoff,int requested_centers) {
    Fixture fixture;fixture.atom_count=4*cells*cells*cells;fixture.cutoff=cutoff;fixture.wave_number=5.1;
    fixture.positions.resize(fixture.atom_count);
    const Position basis[4]={{0,0,0},{0,.5,.5},{.5,0,.5},{.5,.5,0}};
    auto atom_id=[cells](int cell_x,int cell_y,int cell_z,int basis_index) {
        return (((cell_z*cells+cell_y)*cells+cell_x)*4+basis_index);
    };
    auto wrap=[cells](int value){return (value%cells+cells)%cells;};
    for(int cell_z=0;cell_z<cells;++cell_z)
        for(int cell_y=0;cell_y<cells;++cell_y)
            for(int cell_x=0;cell_x<cells;++cell_x)
                for(int basis_index=0;basis_index<4;++basis_index) {
                    const int local_id=atom_id(cell_x,cell_y,cell_z,basis_index);
                    for(int component=0;component<3;++component)
                        fixture.positions[local_id][component]=(component==0?cell_x:component==1?cell_y:cell_z)
                            +basis[basis_index][component]+.006*std::sin(.754877666*(local_id+1)*(component+2));
                }
    const int center_count=requested_centers?std::min(requested_centers,fixture.atom_count):fixture.atom_count;
    const int span=static_cast<int>(std::ceil(cutoff));
    for(int center_index=0;center_index<center_count;++center_index) {
        const int center_local_id=static_cast<long long>(center_index)*fixture.atom_count/center_count;
        const int cell_x=(center_local_id/4)%cells;
        const int cell_y=(center_local_id/(4*cells))%cells;
        const int cell_z=center_local_id/(4*cells*cells);
        std::vector<std::pair<int,Position>> neighbors;
        for(int offset_z=-span;offset_z<=span;++offset_z)
            for(int offset_y=-span;offset_y<=span;++offset_y)
                for(int offset_x=-span;offset_x<=span;++offset_x)
                    for(int basis_index=0;basis_index<4;++basis_index) {
                        const int neighbor=atom_id(wrap(cell_x+offset_x),wrap(cell_y+offset_y),wrap(cell_z+offset_z),basis_index);
                        if(neighbor==center_local_id)continue;
                        Position displacement{};double squared=0;
                        for(int component=0;component<3;++component) {
                            double value=fixture.positions[neighbor][component]-fixture.positions[center_local_id][component];
                            value-=cells*std::nearbyint(value/cells);
                            displacement[component]=value;squared+=value*value;
                        }
                        if(squared<=cutoff*cutoff)neighbors.push_back({neighbor,displacement});
                    }
        std::sort(neighbors.begin(),neighbors.end(),[](const auto& first,const auto& second){return first.first<second.first;});
        append_center(fixture,center_local_id,neighbors,4+center_local_id%5);
    }
    finish_fixture(fixture);return fixture;
}

template<class Value> void upload(GpuBuffer<Value>& destination,const std::vector<Value>& source) {
    if(!source.empty())check_cuda(cudaMemcpy(destination.ptr,source.data(),source.size()*sizeof(Value),cudaMemcpyHostToDevice));
}
std::vector<double> download(const GpuBuffer<double>& source,int count) {
    std::vector<double> result(count);
    if(count)check_cuda(cudaMemcpy(result.data(),source.ptr,count*sizeof(double),cudaMemcpyDeviceToHost));
    return result;
}
struct DeviceFixture {
    const Fixture& host;
    StruFactorData geometry;
    GpuBuffer<int> slot_centers,first_centers,second_centers,pair_slots;
    GpuBuffer<double> seeds,values,forces,virials;
    explicit DeviceFixture(const Fixture& fixture):host(fixture),geometry(fixture.centers.size(),fixture.neighbors.size()),
        slot_centers(fixture.slot_centers.size()),first_centers(fixture.first_centers.size()),
        second_centers(fixture.second_centers.size()),pair_slots(fixture.pair_slots.size()),seeds(fixture.seeds.size()),
        values(fixture.atom_count+2),forces(3*fixture.atom_count+2),virials(6*fixture.atom_count+2) {
        upload(geometry.d_group_indices,fixture.centers);upload(geometry.d_group_numneigh,fixture.offsets);
        upload(geometry.d_neigh_in_cutoff_r,fixture.counts);upload(geometry.d_calculated_numneigh,fixture.neighbors);
        upload(geometry.d_group_dminneigh,fixture.geometry);upload(slot_centers,fixture.slot_centers);
        upload(first_centers,fixture.first_centers);upload(second_centers,fixture.second_centers);
        upload(pair_slots,fixture.pair_slots);upload(seeds,fixture.seeds);
        // 输出尾部哨兵不参加清零和 kernel 运算。
        const double sentinels[2]={123.25,-456.75};
        check_cuda(cudaMemcpy(values.ptr+fixture.atom_count,sentinels,sizeof(sentinels),cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(forces.ptr+3*fixture.atom_count,sentinels,sizeof(sentinels),cudaMemcpyHostToDevice));
        check_cuda(cudaMemcpy(virials.ptr+6*fixture.atom_count,sentinels,sizeof(sentinels),cudaMemcpyHostToDevice));
    }
    void forward(cudaStream_t stream) {
        MetaD_zqc::strufactor_forward(geometry,host.wave_number,host.cutoff*host.cutoff,values.ptr,stream,128);
    }
    void backward(int layout,cudaStream_t stream,bool with_virial=true) {
        check_cuda(cudaMemsetAsync(forces.ptr,0,3*host.atom_count*sizeof(double),stream));
        if(with_virial)check_cuda(cudaMemsetAsync(virials.ptr,0,6*host.atom_count*sizeof(double),stream));
        double* output_virial=with_virial?virials.ptr:nullptr;
        const auto view=MetaD_zqc::kernel_view(geometry);
        if(layout==0)MetaD_zqc::strufactor_backward(geometry,host.wave_number,host.cutoff*host.cutoff,
                                                  seeds.ptr,forces.ptr,output_virial,stream,128);
        if(layout==1 && !host.neighbors.empty())
            MetaD_zqc::directed_pair_kernel<<<(host.neighbors.size()+127)/128,128,0,stream>>>(view,
                host.neighbors.size(),slot_centers.ptr,host.wave_number,host.cutoff*host.cutoff,
                seeds.ptr,forces.ptr,output_virial);
        if(layout==2 && !host.pair_slots.empty())
            MetaD_zqc::half_pair_kernel<<<(host.pair_slots.size()+127)/128,128,0,stream>>>(view,
                host.pair_slots.size(),first_centers.ptr,second_centers.ptr,pair_slots.ptr,
                host.wave_number,host.cutoff*host.cutoff,seeds.ptr,forces.ptr,output_virial);
        check_cuda(cudaGetLastError());
    }
};

// CPU 独立前向公式用于有限差分，不调用生产径向 helper。
double reference_radial(double distance_squared,double wave_number,double cutoff) {
    if(distance_squared>cutoff*cutoff)return 0;
    const double argument=wave_number*std::sqrt(distance_squared);
    double value=argument==0?1:std::sin(argument)/argument;
    if(distance_squared>.8*cutoff*cutoff) {
        const double scaled=(distance_squared-.8*cutoff*cutoff)/(.2*cutoff*cutoff);
        value*=1-scaled*scaled*scaled;
    }
    return value;
}
double reference_energy(const Fixture& fixture,const std::vector<Position>& positions) {
    double energy=0;
    for(size_t center_index=0;center_index<fixture.centers.size();++center_index) {
        const int center=fixture.centers[center_index];double value=1;
        for(int neighbor=0;neighbor<fixture.atom_count;++neighbor) {
            if(neighbor==center)continue;
            double squared=0;
            for(int component=0;component<3;++component) {
                const double displacement=positions[neighbor][component]-positions[center][component];
                squared+=displacement*displacement;
            }
            value+=reference_radial(squared,fixture.wave_number,fixture.cutoff);
        }
        energy+=fixture.seeds[center_index]*value;
    }
    return energy;
}

double compare(const std::vector<double>& actual,const std::vector<double>& reference,double tolerance) {
    require(actual.size()==reference.size(),"output size mismatch");
    double maximum=0;
    for(size_t index=0;index<actual.size();++index) {
        require(std::isfinite(actual[index]),"nonfinite output");
        maximum=std::max(maximum,std::abs(actual[index]-reference[index]));
        if(std::abs(actual[index]-reference[index])>tolerance*(1+std::abs(reference[index])))
            throw std::runtime_error("output mismatch at "+std::to_string(index));
    }
    return maximum;
}
void check_guards(const std::vector<double>& values) {
    require(values[values.size()-2]==123.25 && values.back()==-456.75,"output sentinel modified");
}
struct Errors {double force=0,virial=0,finite_force=0,finite_virial=0,net_force=0;};
Errors validate(DeviceFixture& device,bool finite_difference) {
    Errors errors;const auto& fixture=device.host;
    device.forward(nullptr);device.backward(0,nullptr);
    const auto baseline_force=download(device.forces,3*fixture.atom_count+2);
    const auto baseline_virial=download(device.virials,6*fixture.atom_count+2);
    const auto baseline_value=download(device.values,fixture.atom_count+2);
    check_guards(baseline_value);check_guards(baseline_force);check_guards(baseline_virial);
    // 按生产几何逐项求独立参考值，同时核对组外元素保持为零。
    std::vector<double> reference_values(fixture.atom_count+2,0);
    reference_values[fixture.atom_count]=123.25;reference_values[fixture.atom_count+1]=-456.75;
    for(size_t center_index=0;center_index<fixture.centers.size();++center_index) {
        double value=1;
        for(int neighbor_offset=0;neighbor_offset<fixture.counts[center_index];++neighbor_offset) {
            const int slot=fixture.offsets[center_index]+neighbor_offset;
            value+=reference_radial(fixture.geometry[4*slot+3],fixture.wave_number,fixture.cutoff);
        }
        reference_values[fixture.centers[center_index]]=value;
    }
    compare(baseline_value,reference_values,3e-12);
    for(int layout=1;layout<=2;++layout) {
        device.backward(layout,nullptr);
        const auto force=download(device.forces,3*fixture.atom_count+2);
        const auto virial=download(device.virials,6*fixture.atom_count+2);
        check_guards(force);check_guards(virial);
        errors.force=std::max(errors.force,compare(force,baseline_force,2e-11));
        errors.virial=std::max(errors.virial,compare(virial,baseline_virial,2e-11));
        for(int component=0;component<3;++component) {
            double total=0;for(int atom=0;atom<fixture.atom_count;++atom)total+=force[3*atom+component];
            errors.net_force=std::max(errors.net_force,std::abs(total));
        }
        if(finite_difference) {
            constexpr double displacement=2e-6;
            for(int atom=0;atom<fixture.atom_count;++atom)
                for(int component=0;component<3;++component) {
                    auto plus=fixture.positions,minus=fixture.positions;
                    plus[atom][component]+=displacement;minus[atom][component]-=displacement;
                    const double reference=-(reference_energy(fixture,plus)-reference_energy(fixture,minus))/(2*displacement);
                    errors.finite_force=std::max(errors.finite_force,std::abs(reference-force[3*atom+component]));
                }
            const int rows[6]={0,1,2,1,2,2},columns[6]={0,1,2,0,0,1};
            for(int component=0;component<6;++component) {
                auto plus=fixture.positions,minus=fixture.positions;double total=0;
                for(int atom=0;atom<fixture.atom_count;++atom) {
                    plus[atom][rows[component]]+=displacement*fixture.positions[atom][columns[component]];
                    minus[atom][rows[component]]-=displacement*fixture.positions[atom][columns[component]];
                    total+=virial[6*atom+component];
                }
                const double reference=-(reference_energy(fixture,plus)-reference_energy(fixture,minus))/(2*displacement);
                errors.finite_virial=std::max(errors.finite_virial,std::abs(total-reference));
            }
        }
        device.backward(layout,nullptr,false);
        compare(download(device.forces,3*fixture.atom_count+2),baseline_force,2e-11);
    }
    require(errors.finite_force<5e-7 && errors.finite_virial<5e-7,"finite difference failed");
    return errors;
}

// 用多步 CUDA Graph 重放排除短 kernel 的 Host 提交间隙；每一步包含输出清零。
struct TimedGraph {
    cudaGraph_t graph=nullptr;
    cudaGraphExec_t executable=nullptr;
    int layout=0,phase=0;
    std::vector<double> samples;
    ~TimedGraph(){if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);}
};
constexpr int steps_per_graph=32;
double time_graph(TimedGraph& graph,cudaStream_t stream,int repetitions) {
    cudaEvent_t started,finished;
    check_cuda(cudaEventCreate(&started));check_cuda(cudaEventCreate(&finished));
    check_cuda(cudaEventRecord(started,stream));
    for(int repetition=0;repetition<repetitions;++repetition)check_cuda(cudaGraphLaunch(graph.executable,stream));
    check_cuda(cudaEventRecord(finished,stream));check_cuda(cudaEventSynchronize(finished));
    float milliseconds=0;check_cuda(cudaEventElapsedTime(&milliseconds,started,finished));
    check_cuda(cudaEventDestroy(started));check_cuda(cudaEventDestroy(finished));
    return milliseconds/(repetitions*steps_per_graph);
}
void benchmark(Fixture& fixture,int rounds,std::ofstream& output) {
    DeviceFixture device(fixture);const Errors errors=validate(device,false);
    cudaStream_t stream;check_cuda(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    std::array<TimedGraph,6> graphs;
    for(int phase=0;phase<2;++phase)for(int layout=0;layout<3;++layout) {
        auto& graph=graphs[phase*3+layout];graph.phase=phase;graph.layout=layout;
        check_cuda(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        for(int step=0;step<steps_per_graph;++step) {
            if(phase==1)device.forward(stream);
            device.backward(layout,stream);
        }
        check_cuda(cudaStreamEndCapture(stream,&graph.graph));
        check_cuda(cudaGraphInstantiate(&graph.executable,graph.graph,nullptr,nullptr,0));
        for(int warmup=0;warmup<8;++warmup)check_cuda(cudaGraphLaunch(graph.executable,stream));
    }
    check_cuda(cudaStreamSynchronize(stream));
    std::array<int,6> repetitions;
    for(int index=0;index<6;++index) {
        const double pilot=time_graph(graphs[index],stream,2);
        repetitions[index]=std::max(2,std::min(1000,static_cast<int>(30.0/(pilot*steps_per_graph))));
    }
    std::array<int,6> order={0,1,2,3,4,5};std::mt19937 random(614);
    for(int round=0;round<rounds;++round) {
        std::shuffle(order.begin(),order.end(),random);
        for(int index:order)graphs[index].samples.push_back(time_graph(graphs[index],stream,repetitions[index]));
    }
    const char* layouts[3]={"center","directed_pair","half_pair"};
    for(auto& graph:graphs) {
        std::sort(graph.samples.begin(),graph.samples.end());
        const double median=graph.samples[graph.samples.size()/2];
        output<<fixture.atom_count<<','<<fixture.centers.size()<<','<<fixture.cutoff<<','
            <<fixture.edge_count<<','<<fixture.neighbors.size()<<','<<fixture.pair_slots.size()<<','
            <<(graph.phase==0?"backward":"forward_backward")<<','<<layouts[graph.layout]<<','
            <<median<<','<<graph.samples.front()<<','<<graph.samples.back()<<','
            <<errors.force<<','<<errors.virial<<','<<errors.net_force<<','<<fixture.mapping_ms<<'\n';
        output.flush();
        std::cout<<"TIMING atoms="<<fixture.atom_count<<" centers="<<fixture.centers.size()
            <<" neighbors="<<static_cast<double>(fixture.edge_count)/fixture.centers.size()
            <<" phase="<<(graph.phase==0?"backward":"forward_backward")
            <<" layout="<<layouts[graph.layout]<<" ms="<<median<<std::endl;
    }
    check_cuda(cudaStreamDestroy(stream));
}

int main(int argument_count,char** arguments) {
    try {
        const std::string output_path=argument_count>1?arguments[1]:"results.csv";
        const int rounds=argument_count>2?std::stoi(arguments[2]):9;
        require(rounds>=3,"at least three rounds required");
        cudaDeviceProp properties{};check_cuda(cudaGetDeviceProperties(&properties,0));
        std::cout<<std::setprecision(12)<<"GPU "<<properties.name<<" rounds="<<rounds<<std::endl;
        const std::vector<Position> positions={{0,0,0},{.8,.1,.2},{1.5,.3,-.1},{2.35,-.2,.4},
            {3.6,.2,.1},{8,0,0},{.8000001,.1,.2},{-.35,.6,.2}};
        auto small=make_small_fixture(positions,{7,1,5,3});
        DeviceFixture small_device(small);auto errors=validate(small_device,true);
        std::cout<<"VALIDATION force="<<errors.force<<" virial="<<errors.virial
            <<" finite_force="<<errors.finite_force<<" finite_virial="<<errors.finite_virial<<std::endl;
        upload(small_device.seeds,std::vector<double>(small.seeds.size(),0));
        for(int layout=0;layout<3;++layout) {
            small_device.backward(layout,nullptr);
            const auto force=download(small_device.forces,3*small.atom_count);
            require(std::all_of(force.begin(),force.end(),[](double value){return value==0;}),"zero seed force");
        }
        auto empty=make_small_fixture(positions,{});DeviceFixture empty_device(empty);validate(empty_device,false);
        auto isolated=make_small_fixture(positions,{5});DeviceFixture isolated_device(isolated);validate(isolated_device,false);
        std::ofstream output(output_path);require(static_cast<bool>(output),"cannot write results");
        output<<std::setprecision(14)<<"atoms,centers,cutoff,edges,slots,half_pairs,phase,layout,median_ms,min_ms,max_ms,max_force_error,max_virial_error,net_force_error,host_half_mapping_ms\n";
        for(int cells:{11,16})for(double cutoff:{.82,1.30,2.05})for(int center_count:{0,64}) {
            std::cout<<"CASE_START cells="<<cells<<" cutoff="<<cutoff<<" requested_centers="<<center_count<<std::endl;
            auto fixture=make_fcc_fixture(cells,cutoff,center_count);benchmark(fixture,rounds,output);
        }
        check_cuda(cudaDeviceSynchronize());std::cout<<"STRUFACTOR_PAIR_PASS"<<std::endl;
    }catch(const std::exception& error){std::cerr<<"FAILED: "<<error.what()<<std::endl;return 1;}
}
