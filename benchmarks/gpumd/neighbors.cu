// 独立枚举周期平移，对照生产 CSR；验证缓存跨边界、skin 重建、半表及像贡献回写。
#include "zqc_gpumd_engine.cuh"
#include "zqc_gpu_buffer.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>
using namespace MetaD_zqc;
void require(bool condition,const char* message) { if(!condition) throw std::runtime_error(message); }
void check(cudaError_t status) { require(status==cudaSuccess,cudaGetErrorString(status)); }
using Key=std::array<int,4>;

void verify_list(GpumdEngineAdapter& engine,int handle,double radius,bool half,int asymmetric=0) {
    const auto list=engine.prepare_neighbors(handle);
    const auto positions=engine.positions(MemorySpace::Host);
    const auto ids=engine.atom_ids();
    const std::array<double,3> box{{2.4,2.8,3.2}};
    require(list.coordinates==NeighborCoordinates::PeriodicImages,"periodic-image contract");
    require(list.kind==(half?NeighborKind::Half:NeighborKind::Full),"kind contract");
    require(list.pair_ownership==(half?NeighborPairOwnership::UniqueAcrossRanks:
        NeighborPairOwnership::DirectedByCenter),"ownership contract");
    for(size_t row=0;row<list.host_centers.item_count;++row) {
        const int center=list.host_centers(row);
        std::set<Key> expected,actual;
        for(int owner=0;owner<3;++owner)
            for(int x=-4;x<=4;++x) for(int y=-4;y<=4;++y) for(int z=-4;z<=4;++z) {
                const Key key{{owner,x,y,z}};
                if(owner==center && x==0 && y==0 && z==0) continue;
                if(asymmetric==1 && owner==0) continue;
                if(asymmetric==2 && owner!=0) continue;
                if(half && !asymmetric && (owner<center || (owner==center && Key{{0,x,y,z}}<Key{{0,0,0,0}}))) continue;
                double squared=0;const int shift[3]={x,y,z};
                for(int axis=0;axis<3;++axis) {
                    const double delta=positions(owner,axis)+shift[axis]*box[axis]-positions(center,axis);
                    squared+=delta*delta;
                }
                if(squared<=radius*radius) expected.insert(key);
            }
        for(int slot=list.host_offsets(row);slot<list.host_offsets(row+1);++slot) {
            const int image=list.host_neighbors(slot),owner=static_cast<int>(ids(image)-1);
            Key key{{owner,0,0,0}};
            for(int axis=0;axis<3;++axis) key[axis+1]=static_cast<int>(
                std::llround((positions(image,axis)-positions(owner,axis))/box[axis]));
            require(actual.insert(key).second,"duplicate image in CSR row");
        }
        require(actual==expected,"CSR differs from independent image enumeration");
    }
}
int main() {
    try {
        GpumdEngineAdapter engine;
        engine.set_neighbor_skin(.4);
        const int left=engine.define_selection("left",{0}),right=engine.define_selection("right",{1,2});
        NeighborRequest full;full.cutoff=3.7;
        const int full_handle=engine.register_neighbors(full);
        NeighborRequest half=full;half.kind=NeighborKind::Half;
        const int half_handle=engine.register_neighbors(half);
        NeighborRequest asymmetric=half;asymmetric.centers=left;asymmetric.neighbors=right;
        const int asymmetric_handle=engine.register_neighbors(asymmetric);
        NeighborRequest reverse=asymmetric;reverse.centers=right;reverse.neighbors=left;
        const int reverse_handle=engine.register_neighbors(reverse);
        GpuBuffer<double> coordinates(9),forces(9),virials(27),potential(3);
        GpuBuffer<int> types(3);
        const int host_types[3]={0,1,2};
        types.upload_from(host_types,3);
        cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        GpumdFrame frame;
        frame.positions=coordinates.view(3,3);frame.types=types.view(3);frame.forces=forces.view(3,3);
        frame.virials=virials.view(3,9);frame.potential=potential.view(3);
        frame.box_lengths={{2.4,2.8,3.2}};frame.periodic={{1,1,1}};frame.stream=stream;
        std::vector<double> xyz{.02,.2,.3, 1.1,.8,.4, 2.2,1.9,2.8};
        auto begin=[&] {
            coordinates.upload_from(xyz.data(),xyz.size(),stream);
            forces.clear_async(stream);virials.clear_async(stream);potential.clear_async(stream);
            engine.bind_frame(frame);engine.begin_evaluation();
        };
        begin();
        require(engine.neighbor_rebuild_count()==1,"first candidate build");
        verify_list(engine,full_handle,4.1,false);
        verify_list(engine,half_handle,4.1,true);
        verify_list(engine,asymmetric_handle,4.1,true,1);
        verify_list(engine,reverse_handle,4.1,true,2);
        const auto selection=engine.selection_members(left);
        const auto ids=engine.atom_ids();
        require(selection.owned_count==1 && selection.local_ids(0)==0,"owned selection prefix");
        for(size_t row=0;row<selection.local_ids.item_count;++row)
            require(ids(selection.local_ids(row))==1,"image selection inheritance");
        const auto count=static_cast<size_t>(engine.total_atom_count());
        std::vector<double> contributions(3*count),tensor(9*count),expected_force(9,0),expected_tensor(27,0);
        for(size_t row=0;row<count;++row) {
            const size_t owner=ids(row)-1;
            for(int axis=0;axis<3;++axis) expected_force[3*owner+axis]+=(contributions[3*row+axis]=.001*(row+1)*(axis+1));
            for(int component=0;component<9;++component)
                expected_tensor[9*owner+component]+=(tensor[9*row+component]=.0001*(row+1)*(component+1));
        }
        // Host 与 Cuda 两条提交路径都应把每个像累加回同一个 owned 槽位。
        engine.commit_force_virial(make_aos_view<const double>(contributions.data(),count,3,contributions.size(),MemorySpace::Host),
            {make_aos_view<const double>(tensor.data(),count,9,tensor.size(),MemorySpace::Host),VirialRepresentation::Full9});
        GpuBuffer<double> device_force(contributions.size()),device_tensor(tensor.size());
        device_force.upload_from(contributions.data(),contributions.size(),stream);
        device_tensor.upload_from(tensor.data(),tensor.size(),stream);
        engine.commit_force_virial(device_force.view(count,3),{device_tensor.view(count,9),VirialRepresentation::Full9});
        engine.finish(0);
        check(cudaStreamSynchronize(stream));
        std::vector<double> actual_force(9),actual_tensor(27);
        forces.download_to(actual_force.data(),9,stream);virials.download_to(actual_tensor.data(),27,stream);
        for(size_t i=0;i<9;++i) require(std::abs(actual_force[i]-2*expected_force[i])<1e-9,"image force fold");
        for(size_t i=0;i<27;++i) require(std::abs(actual_tensor[i]-2*expected_tensor[i])<1e-9,"image Full9 fold");

        // 跨周期边界的刚体平移小于 skin/2，缓存布局保持且几何连续。
        for(int i=0;i<3;++i) xyz[3*i]-=.05;
        begin();require(engine.neighbor_rebuild_count()==1,"wrapped movement should reuse cache");
        verify_list(engine,full_handle,4.1,false);
        engine.finish(0);
        xyz[0]+=.5;
        begin();require(engine.neighbor_rebuild_count()==2,"skin displacement rebuild");
        verify_list(engine,full_handle,4.1,false);
        engine.finish(0);
        frame.box_lengths[0]=2.5;
        begin();require(engine.neighbor_rebuild_count()==3,"box change rebuild");engine.finish(0);
        check(cudaStreamSynchronize(stream));check(cudaStreamDestroy(stream));
        std::cout<<"PASS: periodic multi-image/full/half/group/skin/wrapping/Full9 owner folding\n";
    } catch(const std::exception& exception) { std::cerr<<exception.what()<<'\n';return 1; }
}
