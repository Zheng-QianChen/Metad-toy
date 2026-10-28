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
    int nodes=0;
    double lower=0, spacing=0;
    std::string fingerprint;
    Impl(const std::string& path, SumReducer reducer) : sum(std::move(reducer)) {
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
        if (size) scale_gradient<<<(size+block_size-1)/block_size,block_size,0,execution.get()>>>(
            gradient.ptr,size,inverse_count);
        check(cudaGetLastError());
        gradient.record_write(execution.get());
        return {aggregate[0]*inverse_count,aggregate[1]*inverse_count,
                static_cast<long long>(aggregate[2]),gradient.view(size)};
    }
};

FrozenSharedBias::FrozenSharedBias(const std::string& path, SumReducer sum)
    : impl_(std::make_unique<Impl>(path,std::move(sum))) {}
FrozenSharedBias::~FrozenSharedBias() = default;
FrozenSharedBias::Result FrozenSharedBias::evaluate(ArrayView<const double> coordinates,
                                                   ArrayView<const int> slots) {
    return impl_->evaluate(coordinates,slots);
}
const std::string& FrozenSharedBias::fingerprint() const { return impl_->fingerprint; }
} // namespace MetaD_zqc
