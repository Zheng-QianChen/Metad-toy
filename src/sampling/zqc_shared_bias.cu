#include "zqc_shared_bias.cuh"
#include "zqc_gpu_buffer.cuh"
#include "zqc_cuda_execution.cuh"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <algorithm>

namespace MetaD_zqc {
namespace {
constexpr int block_size = 256;
void check(cudaError_t code) {
    if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
// 表按 [节点][能量,斜率] 存储，两列具有各自物理单位。
__device__ void interpolate(double q, const double* table, int count, double lower, double spacing,
                            double& energy, double& derivative) {
    const double upper = lower + (count-1)*spacing;
    if (q <= lower || q >= upper) {
        const int node = q <= lower ? 0 : count-1;
        derivative = table[2*node+1];
        energy = table[2*node] + derivative*(q-(lower+node*spacing));
        return;
    }
    const int left = min(static_cast<int>((q-lower)/spacing),count-2);
    const double t = (q-(lower+left*spacing))/spacing, t2=t*t, t3=t2*t;
    const double u0=table[2*left], m0=table[2*left+1];
    const double u1=table[2*left+2], m1=table[2*left+3];
    energy=(2*t3-3*t2+1)*u0+(t3-2*t2+t)*spacing*m0+
           (-2*t3+3*t2)*u1+(t3-t2)*spacing*m1;
    derivative=((6*t2-6*t)*u0+(-6*t2+6*t)*u1)/spacing+
               (3*t2-4*t+1)*m0+(3*t2-2*t)*m1;
}
// 每个 block 规约局部能量、坐标和非法样本数；第二个 kernel 再合并成三个标量。
__global__ void evaluate_table(ArrayView<const double> coordinates, ArrayView<const int> slots,
        const double* table, int count, double lower, double spacing,
        double* gradient, int* seen, double* partials) {
    __shared__ double sums[3][block_size];
    const int lane=threadIdx.x;
    const size_t index=static_cast<size_t>(blockIdx.x)*blockDim.x+lane;
    double energy=0, coordinate=0, invalid=0;
    if (index < slots.item_count) {
        const int slot=slots(index);
        if (slot < 0 || static_cast<size_t>(slot) >= coordinates.item_count) invalid=1;
        else if (atomicCAS(seen+slot,0,1) != 0) invalid=1;
        else {
            coordinate=coordinates(slot);
            double derivative=0;
            if (!isfinite(coordinate)) { coordinate=0; invalid=1; }
            else {
                interpolate(coordinate,table,count,lower,spacing,energy,derivative);
                if (!isfinite(energy) || !isfinite(derivative)) { energy=0; invalid=1; }
                else gradient[slot]=derivative;
            }
        }
    }
    sums[0][lane]=energy; sums[1][lane]=coordinate; sums[2][lane]=invalid;
    __syncthreads();
    for (int offset=block_size/2;offset;offset/=2) {
        if (lane<offset) for (int field=0;field<3;++field) sums[field][lane]+=sums[field][lane+offset];
        __syncthreads();
    }
    if (!lane) for (int field=0;field<3;++field) partials[3*blockIdx.x+field]=sums[field][0];
}
__global__ void finish_sum(const double* partials, int blocks, double* packed) {
    __shared__ double sums[3][block_size];
    const int lane=threadIdx.x;
    for (int field=0;field<3;++field) {
        double sum=0;
        for (int block=lane;block<blocks;block+=block_size) sum+=partials[3*block+field];
        sums[field][lane]=sum;
    }
    __syncthreads();
    for (int offset=block_size/2;offset;offset/=2) {
        if (lane<offset) for (int field=0;field<3;++field) sums[field][lane]+=sums[field][lane+offset];
        __syncthreads();
    }
    if (!lane) for (int field=0;field<3;++field) packed[field]=sums[field][0];
}
__global__ void scale_gradient(double* gradient, size_t count, double inverse_count) {
    const size_t i=static_cast<size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i<count) gradient[i]*=inverse_count;
}
} // namespace

struct FrozenSharedBias::Impl {
    CudaExecutionStream execution;
    GpuBuffer<double> table, gradient, partials, packed;
    GpuBuffer<int> seen;
    SumReducer sum;
    SharedAggregation aggregation;
    int nodes=0;
    double lower=0, spacing=0;
    std::string fingerprint;
    Impl(int count,double low,double high,SumReducer reducer,SharedAggregation mode)
        : sum(std::move(reducer)),aggregation(mode),nodes(count),lower(low),spacing((high-low)/(count-1)) {
        table.grow_to(size_t(nodes)*2,__FILE__,__LINE__);
        table.clear_async(execution.get());table.record_write(execution.get());
        packed.grow_to(3,__FILE__,__LINE__);execution.synchronize();
    }
    Impl(const std::string& path, SumReducer reducer, SharedAggregation mode) : sum(std::move(reducer)),aggregation(mode) {
        std::ifstream input(path);
        if (!input) throw std::runtime_error("Shared frozen bias: cannot open "+path);
        std::string line;
        if (!std::getline(input,line) || line!="# METAD_TOY_SHARED_FROZEN 1")
            throw std::runtime_error("Shared frozen bias: expected version 1 table header");
        std::vector<double> coordinates, data;
        while (std::getline(input,line)) {
            const auto begin=line.find_first_not_of(" \t\r");
            if (begin==std::string::npos || line[begin]=='#') continue;
            std::istringstream row(line);
            double q,energy,derivative; std::string extra;
            if (!(row>>q>>energy>>derivative) || (row>>extra) ||
                !std::isfinite(q) || !std::isfinite(energy) || !std::isfinite(derivative))
                throw std::runtime_error("Shared frozen bias: expected finite q energy derivative");
            coordinates.push_back(q); data.push_back(energy); data.push_back(derivative);
        }
        if (input.bad() || coordinates.size()<2 || coordinates.size()>static_cast<size_t>(std::numeric_limits<int>::max()/2))
            throw std::runtime_error("Shared frozen bias: invalid table length/read failure");
        nodes=static_cast<int>(coordinates.size());
        lower=coordinates[0]; spacing=coordinates[1]-lower;
        if (!(spacing>0) || !std::isfinite(spacing)) throw std::runtime_error("Shared frozen bias: invalid spacing");
        for (int i=1;i<nodes;++i) {
            const double expected=lower+i*spacing;
            if (!(coordinates[i]>coordinates[i-1]) || !std::isfinite(expected) ||
                std::abs(coordinates[i]-expected)>1e-10*spacing)
                throw std::runtime_error("Shared frozen bias: q must be strictly increasing and equally spaced");
        }
        // 对解析后的 IEEE 双精度内容计算指纹；空格和注释不改变固定势的身份。
        std::uint64_t hash=14695981039346656037ULL;
        auto add=[&](double value) {
            std::uint64_t bits; std::memcpy(&bits,&value,sizeof(bits));
            for (int byte=0;byte<8;++byte) { hash^=(bits>>(8*byte))&255; hash*=1099511628211ULL; }
        };
        for (double q:coordinates) add(q);
        for (double value:data) add(value);
        std::ostringstream identity; identity<<std::hex<<std::setfill('0')<<std::setw(16)<<hash;
        fingerprint=identity.str();
        table.grow_to(data.size(),__FILE__,__LINE__);
        check(cudaMemcpyAsync(table.ptr,data.data(),data.size()*sizeof(double),cudaMemcpyHostToDevice,execution.get()));
        table.record_write(execution.get());
        packed.grow_to(3,__FILE__,__LINE__);
        execution.synchronize(); // 构造时完成表上传，Host 解析存储随后可释放。
    }
    ~Impl() { cudaStreamSynchronize(execution.get()); }

    Result evaluate(ArrayView<const double> coordinates, ArrayView<const int> slots) {
        coordinates.validate(); slots.validate();
        if (coordinates.memory!=MemorySpace::Cuda || slots.memory!=MemorySpace::Cuda ||
            coordinates.component_count!=1 || slots.component_count!=1 ||
            coordinates.item_count>static_cast<size_t>(std::numeric_limits<int>::max()) ||
            slots.item_count>coordinates.item_count)
            throw std::invalid_argument("Shared frozen bias: expected CUDA scalar fields and owned slots");
        const size_t size=coordinates.item_count;
        const int blocks=static_cast<int>((slots.item_count+block_size-1)/block_size);
        gradient.grow_to(size,__FILE__,__LINE__); seen.grow_to(size,__FILE__,__LINE__);
        partials.grow_to(3*static_cast<size_t>(blocks),__FILE__,__LINE__);
        coordinates.wait_on(execution.get()); slots.wait_on(execution.get());
        gradient.wait_on(execution.get());
        if (size) {
            check(cudaMemsetAsync(gradient.ptr,0,size*sizeof(double),execution.get()));
            check(cudaMemsetAsync(seen.ptr,0,size*sizeof(int),execution.get()));
        }
        if (blocks) evaluate_table<<<blocks,block_size,0,execution.get()>>>(
            coordinates,slots,table.ptr,nodes,lower,spacing,gradient.ptr,seen.ptr,partials.ptr);
        finish_sum<<<1,block_size,0,execution.get()>>>(partials.ptr,blocks,packed.ptr);
        check(cudaGetLastError());
        double local[3]={};
        check(cudaMemcpyAsync(local,packed.ptr,sizeof(local),cudaMemcpyDeviceToHost,execution.get()));
        execution.synchronize();
        double aggregate[4]={local[0],local[1],static_cast<double>(slots.item_count),local[2]};
        if (sum) sum(aggregate,4);
        if (aggregate[3]!=0 || !std::isfinite(aggregate[0]) || !std::isfinite(aggregate[1]))
            throw std::runtime_error("Shared frozen bias: invalid/duplicate slot or nonfinite coordinate/value");
        if (!(aggregate[2]>0) || aggregate[2]>9007199254740991.0 || !std::isfinite(aggregate[2]))
            throw std::runtime_error("Shared frozen bias: global family must be nonempty with an exact count");
        const double inverse_count=1.0/aggregate[2];
        const double scale=aggregation==SharedAggregation::Mean ? inverse_count : 1.0;
        if (size) scale_gradient<<<(size+block_size-1)/block_size,block_size,0,execution.get()>>>(
            gradient.ptr,size,scale);
        check(cudaGetLastError());
        gradient.record_write(execution.get());
        return {aggregate[0]*scale,aggregate[1]*inverse_count,
                static_cast<long long>(aggregate[2]),gradient.view(size)};
    }
};

FrozenSharedBias::FrozenSharedBias(const std::string& path, SumReducer sum, SharedAggregation aggregation)
    : impl_(std::make_unique<Impl>(path,std::move(sum),aggregation)) {}
FrozenSharedBias::~FrozenSharedBias() = default;
FrozenSharedBias::Result FrozenSharedBias::evaluate(ArrayView<const double> coordinates,
                                                   ArrayView<const int> slots) {
    return impl_->evaluate(coordinates,slots);
}
const std::string& FrozenSharedBias::fingerprint() const { return impl_->fingerprint; }

namespace {
// Each center updates the common grid. No center coordinates or hills cross the
// Host boundary; only the grid increment and an error scalar are reduced.
__global__ void shared_heights(ArrayView<const double> coordinates,ArrayView<const int> slots,
        const double* table,int nodes,double lower,double spacing,double sigma,double height,
        double tempering,double normalization,double* heights,double* invalid) {
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=slots.item_count) return;
    const double q=coordinates(slots(index));
    const double upper=lower+(nodes-1)*spacing;
    heights[index]=0;
    if(q<lower || q>upper) {atomicAdd(invalid,1.0);return;}
    double energy,derivative;interpolate(q,table,nodes,lower,spacing,energy,derivative);
    const double weight=height*normalization*(tempering>0 ? exp(-energy/tempering) : 1.0);
    if(!isfinite(weight)) {atomicAdd(invalid,1.0);return;}
    heights[index]=weight;
}
__global__ void deposit_shared(ArrayView<const double> coordinates,ArrayView<const int> slots,
        const double* heights,double lower,double spacing,double sigma,double* increment) {
    // A block owns one node. Reducing all centers here avoids O(N*nodes)
    // contended global atomics while retaining the full Gaussian, including tails.
    const double q=lower+blockIdx.x*spacing;
    double energy=0,derivative=0;
    for(size_t index=threadIdx.x;index<slots.item_count;index+=blockDim.x) {
        const double distance=(q-coordinates(slots(index)))/sigma;
        const double value=heights[index]*exp(-.5*distance*distance);
        energy+=value;derivative-=distance/sigma*value;
    }
    __shared__ double energies[256],derivatives[256];
    energies[threadIdx.x]=energy;derivatives[threadIdx.x]=derivative;__syncthreads();
    for(int offset=128;offset;offset/=2) {
        if(threadIdx.x<offset) {energies[threadIdx.x]+=energies[threadIdx.x+offset];derivatives[threadIdx.x]+=derivatives[threadIdx.x+offset];}
        __syncthreads();
    }
    if(!threadIdx.x) {increment[2*blockIdx.x]=energies[0];increment[2*blockIdx.x+1]=derivatives[0];}
}
std::string shared_hash(const std::string& text) {
    std::uint64_t hash=14695981039346656037ULL;
    for(unsigned char c:text) {hash^=c;hash*=1099511628211ULL;}
    std::ostringstream output;output<<std::hex<<std::setfill('0')<<std::setw(16)<<hash;return output.str();
}
constexpr std::int64_t exact_step_limit=9007199254740991LL;
}

struct OnlineSharedBias::Impl {
    SharedOnlineSettings settings;
    SharedOnlineHost host;
    std::unique_ptr<FrozenSharedBias::Impl> bias;
    GpuBuffer<double> increment, invalid, heights;
    std::vector<double> values;
    std::ofstream output;
    std::string configuration;
    std::int64_t last_step=-1,seen_step=-1;
    std::uint64_t records=0;
    bool initialized=false;

    Impl(SharedOnlineSettings s,SharedOnlineHost h):settings(std::move(s)),host(std::move(h)) {
        const auto finite=[](double x){return std::isfinite(x);};
        if(settings.nodes<4 || settings.nodes>1000000 || !finite(settings.lower) || !finite(settings.upper) ||
           !(settings.upper>settings.lower) || !finite(settings.sigma) || !(settings.sigma>0) ||
           !finite(settings.height) || !(settings.height>0) || settings.pace<0 ||
           !finite(settings.thermal_energy) || !(settings.thermal_energy>0) || !finite(settings.bias_factor) ||
           (settings.well_tempered && !(settings.bias_factor>1)) || settings.history_path.empty() ||
           (!host.root && !host.broadcast))
            throw std::invalid_argument("Shared online: invalid grid, deposition, WT or host configuration");
        const double spacing=(settings.upper-settings.lower)/(settings.nodes-1);
        if(!finite(spacing) || !(spacing>0) || spacing>settings.sigma/2)
            throw std::invalid_argument("Shared online: grid spacing must be positive and at most sigma/2");
        if(settings.well_tempered && !finite(settings.thermal_energy*(settings.bias_factor-1)))
            throw std::invalid_argument("Shared online: WT thermal scale overflow");
        std::ostringstream config;config<<std::setprecision(17)
            <<settings.nodes<<' '<<settings.lower<<' '<<settings.upper<<' '<<settings.sigma<<' '<<settings.height
            <<' '<<settings.thermal_energy<<' '<<settings.bias_factor<<' '<<settings.well_tempered
            <<' '<<int(settings.aggregation)<<' '<<settings.mean_deposition<<' '<<shared_hash(settings.identity);
        configuration=config.str();
        bias=std::make_unique<FrozenSharedBias::Impl>(settings.nodes,settings.lower,settings.upper,host.sum,settings.aggregation);
        values.assign(size_t(settings.nodes)*2,0.0);
        increment.grow_to(values.size(),__FILE__,__LINE__);invalid.grow_to(1,__FILE__,__LINE__);
    }
    void broadcast(double* data,size_t size) {if(host.broadcast) host.broadcast(data,size);}
    template<class Function> void root_operation(Function function) {
        double failed=0;std::string message;
        if(host.root) try {function();} catch(const std::exception& error) {failed=1;message=error.what();}
        broadcast(&failed,1);
        if(failed) throw std::runtime_error(host.root ? message : "Shared online: root history operation failed");
    }
    std::string record_text(std::int64_t step,std::uint64_t count,const std::vector<double>& table) const {
        std::ostringstream text;text<<std::setprecision(17)<<"STATE "<<step<<' '<<count;
        for(double value:table) text<<' '<<value;
        return text.str();
    }
    void write_record(std::int64_t step,std::uint64_t count,const std::vector<double>& table) {
        const auto text=record_text(step,count,table);
        output<<text<<' '<<shared_hash(configuration+'\n'+text)<<'\n';output.flush();
        if(!output) throw std::runtime_error("Shared online: failed writing complete history record");
    }
    void load_history(std::int64_t step) {
        std::ifstream input(settings.history_path);std::string line;
        if(!input || !std::getline(input,line) || line!="# METAD_TOY_SHARED_ONLINE 1" ||
           !std::getline(input,line) || line!="CONFIG "+configuration)
            throw std::invalid_argument("Shared online: history configuration/header mismatch");
        bool initial=false;
        while(std::getline(input,line)) {
            if(line.empty() || input.eof()) throw std::invalid_argument("Shared online: empty/truncated history record");
            const auto end=line.find_last_of(' ');
            if(end==std::string::npos || line.substr(end+1)!=shared_hash(configuration+'\n'+line.substr(0,end)))
                throw std::invalid_argument("Shared online: history checksum mismatch");
            std::istringstream row(line.substr(0,end));std::string marker,extra;
            std::int64_t next_step;std::uint64_t next_count;
            if(!(row>>marker>>next_step>>next_count) || marker!="STATE" || next_step>step || next_step>exact_step_limit ||
               (!initial && (next_step!=-1 || next_count!=0)) ||
               (initial && (next_step<=last_step || next_count!=records+1)))
                throw std::invalid_argument("Shared online: invalid/future history step or record sequence");
            for(auto& value:values)
                if(!(row>>value) || !std::isfinite(value)) throw std::invalid_argument("Shared online: invalid history table");
            if(row>>extra) throw std::invalid_argument("Shared online: extra history fields");
            last_step=next_step;records=next_count;initial=true;
        }
        if(input.bad() || !initial) throw std::runtime_error("Shared online: incomplete history");
    }
    void upload() {
        bias->table.upload_from(values.data(),values.size(),bias->execution.get());
        bias->table.record_write(bias->execution.get());bias->execution.synchronize();
    }
    void initialize(std::int64_t step) {
        if(step<0 || step>exact_step_limit) throw std::invalid_argument("Shared online: step outside exact range");
        if(initialized) {
            if(step<seen_step) throw std::invalid_argument("Shared online: simulation step moved backwards");
            return;
        }
        root_operation([&]{
            if(settings.restart) load_history(step);
            else {
                std::ifstream existing(settings.history_path);
                if(existing) throw std::invalid_argument("Shared online: history already exists; choose restart or a new path");
            }
            output.open(settings.history_path,settings.restart ? std::ios::app : std::ios::trunc);
            if(!output) throw std::runtime_error("Shared online: cannot write history");
            if(!settings.restart) {
                output<<"# METAD_TOY_SHARED_ONLINE 1\nCONFIG "<<configuration<<'\n';
                write_record(-1,0,values);
            }
        });
        double state[2]={double(last_step),double(records)};broadcast(state,2);
        last_step=static_cast<std::int64_t>(state[0]);records=static_cast<std::uint64_t>(state[1]);
        broadcast(values.data(),values.size());upload();seen_step=step;initialized=true;
    }
    bool advance(std::int64_t step,ArrayView<const double> coordinates,ArrayView<const int> slots) {
        if(!initialized) throw std::logic_error("Shared online: initialize before deposition");
        if(step<seen_step || step<0 || step>exact_step_limit) throw std::invalid_argument("Shared online: invalid/backward step");
        seen_step=step;
        if(!settings.pace || step%settings.pace || step==last_step) return false;
        // Reuse the exact same owned-slot validation and global count as force evaluation.
        const auto before=bias->evaluate(coordinates,slots);
        const auto stream=bias->execution.get();coordinates.wait_on(stream);slots.wait_on(stream);
        invalid.clear_async(stream);heights.grow_to(slots.item_count,__FILE__,__LINE__);
        const double tempering=settings.well_tempered ? settings.thermal_energy*(settings.bias_factor-1) : 0;
        if(slots.item_count) shared_heights<<<(slots.item_count+255)/256,256,0,stream>>>(
            coordinates,slots,bias->table.ptr,settings.nodes,settings.lower,bias->spacing,settings.sigma,
            settings.height,tempering,settings.mean_deposition ? 1.0/before.count : 1.0,heights.ptr,invalid.ptr);
        deposit_shared<<<settings.nodes,256,0,stream>>>(coordinates,slots,heights.ptr,
            settings.lower,bias->spacing,settings.sigma,increment.ptr);
        check(cudaGetLastError());
        std::vector<double> delta(values.size()+1,0);
        increment.download_to(delta.data(),values.size(),stream);invalid.download_to(&delta.back(),1,stream);
        if(host.sum) host.sum(delta.data(),delta.size());
        if(delta.back()!=0) throw std::invalid_argument("Shared online: center outside grid or nonfinite WT height");
        std::vector<double> next(values.size());
        for(size_t i=0;i<values.size();++i) {
            next[i]=values[i]+delta[i];
            if(!std::isfinite(next[i])) throw std::runtime_error("Shared online: table overflow");
        }
        root_operation([&]{write_record(step,records+1,next);});
        values.swap(next);upload();last_step=step;++records;return true;
    }
};
OnlineSharedBias::OnlineSharedBias(SharedOnlineSettings settings,SharedOnlineHost host)
    :impl_(std::make_unique<Impl>(std::move(settings),std::move(host))) {}
OnlineSharedBias::~OnlineSharedBias()=default;
void OnlineSharedBias::initialize(std::int64_t step) {impl_->initialize(step);}
bool OnlineSharedBias::advance(std::int64_t step,ArrayView<const double> coordinates,ArrayView<const int> slots) {
    return impl_->advance(step,coordinates,slots);
}
OnlineSharedBias::Result OnlineSharedBias::evaluate(ArrayView<const double> coordinates,ArrayView<const int> slots) {
    if(!impl_->initialized) throw std::logic_error("Shared online: initialize before evaluation");
    return impl_->bias->evaluate(coordinates,slots);
}
std::int64_t OnlineSharedBias::last_deposition_step() const {return impl_->last_step;}
std::uint64_t OnlineSharedBias::depositions() const {return impl_->records;}
} // namespace MetaD_zqc
