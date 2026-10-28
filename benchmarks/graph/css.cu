#include "CV_CSS.h"
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace MetaD_zqc;
using Point = std::array<double,3>;
using Positions = std::vector<Point>;
using tagint = MetaD_zqc::LocalAtomIndex;
const std::vector<int> centers{1,3,5,7};
const std::vector<int> atom_types{1,2,3,1,2,3,1,2};
const std::vector<double> center_adjoints{.7,-.4,1.1,.3};
void check(cudaError_t status) { if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status)); }
void near(double actual,double expected,double tolerance,const char* name) {
    if(!std::isfinite(actual) || !std::isfinite(expected) || std::abs(actual-expected)>tolerance*(1+std::abs(expected)))
        throw std::runtime_error(std::string(name)+": "+std::to_string(actual)+" vs "+std::to_string(expected));
}
template<class T> void upload(GpuBuffer<T>& buffer,const std::vector<T>& values) {
    if(!values.empty())check(cudaMemcpy(buffer.ptr,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice));
}
std::vector<double> download(GpuBuffer<double>& buffer,size_t element_count) {
    std::vector<double> values(element_count);if(element_count)check(cudaMemcpy(values.data(),buffer.ptr,
        element_count*sizeof(double),cudaMemcpyDeviceToHost));return values;
}
// 独立 CPU 公式：按原子坐标与类型遍历，不调用生产径向函数或 GPU 导数。
double radial(const CSSChannel& channel,double neighbor_distance) {
    const auto& radial_parameters=channel.radial;
    if(radial_parameters.type==FERMI)return 1/(1+std::exp(-radial_parameters.alpha*(neighbor_distance-radial_parameters.r_0)));
    if(radial_parameters.type==TANH_TYPE)return .5*(1+std::tanh(radial_parameters.alpha*(neighbor_distance-radial_parameters.r_0)));
    const double scaled_distance=neighbor_distance/radial_parameters.r_0;
    if(std::abs(scaled_distance-1)<1e-12)return double(radial_parameters.n)/radial_parameters.m;
    return (1-std::pow(scaled_distance,radial_parameters.n))/(1-std::pow(scaled_distance,radial_parameters.m));
}
std::vector<double> reference(const Positions& positions,const std::vector<CSSChannel>& channels,
                              std::vector<double>* coordination_numbers=nullptr) {
    std::vector<double> css_values(positions.size(),0);
    if(coordination_numbers)coordination_numbers->assign(positions.size()*channels.size(),0);
    for(int center_local_id:centers) {
        double mismatch=0;
        for(size_t channel_index=0;channel_index<channels.size();++channel_index) {
            double coordination_number=0;
            for(size_t neighbor_local_id=0;neighbor_local_id<positions.size();++neighbor_local_id) {
                if(int(neighbor_local_id)==center_local_id || atom_types[neighbor_local_id]!=channels[channel_index].neighbor_type)continue;
                double distance_squared=0;for(int component=0;component<3;++component)distance_squared+=std::pow(positions[neighbor_local_id][component]-positions[center_local_id][component],2);
                if(distance_squared<=channels[channel_index].cutoff*channels[channel_index].cutoff)coordination_number+=radial(channels[channel_index],std::sqrt(distance_squared));
            }
            if(coordination_numbers)(*coordination_numbers)[channel_index*positions.size()+center_local_id]=coordination_number;
            mismatch+=std::pow((coordination_number-channels[channel_index].target)/channels[channel_index].sigma,2);
        }
        css_values[center_local_id]=std::exp(-.5*mismatch);
    }
    return css_values;
}
double energy(const Positions& positions,const std::vector<CSSChannel>& channels) {
    const auto css_values=reference(positions,channels);double weighted_energy=0;
    for(size_t center_index=0;center_index<centers.size();++center_index)weighted_energy+=center_adjoints[center_index]*css_values[centers[center_index]];
    return weighted_energy;
}
void test(const Positions& positions,const std::vector<CSSChannel>& channels) {
    const int atom_count=positions.size(),channel_count=channels.size();
    std::vector<tagint> offsets,neighbors,center_local_ids(centers.begin(),centers.end());
    std::vector<int> neighbor_counts;std::vector<double> geometry;
    for(int atom_local_id:centers) {
        offsets.push_back(neighbors.size());neighbor_counts.push_back(0);
        for(int neighbor_local_id=0;neighbor_local_id<atom_count;++neighbor_local_id) {
            if(atom_local_id==neighbor_local_id)continue;
            Point displacement{};double distance_squared=0;for(int component=0;component<3;++component){displacement[component]=positions[neighbor_local_id][component]-positions[atom_local_id][component];distance_squared+=displacement[component]*displacement[component];}
            if(distance_squared>25)continue;
            neighbors.push_back(neighbor_local_id);geometry.insert(geometry.end(),{displacement[0],displacement[1],
                displacement[2],distance_squared});++neighbor_counts.back();
        }
    }
    offsets.push_back(neighbors.size());
    StruFactorData data(centers.size(),neighbors.size());
    upload(data.d_group_indices,center_local_ids);upload(data.d_group_numneigh,offsets);
    upload(data.d_calculated_numneigh,neighbors);upload(data.d_group_dminneigh,geometry);
    upload(data.d_neigh_in_cutoff_r,neighbor_counts);
    GpuBuffer<CSSChannel> d_channels(channel_count);upload(d_channels,channels);
    GpuBuffer<int> d_atom_types(atom_count);upload(d_atom_types,atom_types);
    GpuBuffer<double> d_css_values(atom_count+1),d_coordination_numbers(atom_count*channel_count+1),
        d_center_adjoints(centers.size()),d_atom_forces(3*atom_count+1),d_atom_virials(6*atom_count+1);
    upload(d_center_adjoints,center_adjoints);
    for(auto* buffer:{&d_css_values,&d_coordination_numbers,&d_atom_forces,&d_atom_virials}) {
        const double guard=12345;check(cudaMemcpy(buffer->ptr+buffer->capacity-1,&guard,sizeof(double),
            cudaMemcpyHostToDevice));
    }
    css_forward(kernel_view(data),d_atom_types.ptr,atom_count,d_channels.ptr,channel_count,d_css_values.ptr,
        d_coordination_numbers.ptr);
    css_backward(kernel_view(data),d_atom_types.ptr,atom_count,d_channels.ptr,channel_count,d_css_values.ptr,
        d_coordination_numbers.ptr,d_center_adjoints.ptr,d_atom_forces.ptr,d_atom_virials.ptr);
    auto css_values=download(d_css_values,atom_count+1),coordination_numbers=download(d_coordination_numbers,
        atom_count*channel_count+1),atom_forces=download(d_atom_forces,3*atom_count+1),
        atom_virials=download(d_atom_virials,6*atom_count+1);
    std::vector<double> reference_coordination_numbers;const auto reference_css_values=reference(positions,channels,
        &reference_coordination_numbers);
    for(int atom_local_id=0;atom_local_id<atom_count;++atom_local_id)near(css_values[atom_local_id],
        reference_css_values[atom_local_id],5e-14,"score");
    for(int atom_local_id=0;atom_local_id<atom_count*channel_count;++atom_local_id)near(coordination_numbers[atom_local_id],reference_coordination_numbers[atom_local_id],5e-14,"coordination");
    for(double guard_value:{css_values.back(),coordination_numbers.back(),atom_forces.back(),
        atom_virials.back()})near(guard_value,12345,0,"guard");
    const double difference_step=2e-6;double max_force_error=0,max_virial_error=0;
    for(int atom_local_id=0;atom_local_id<atom_count;++atom_local_id)for(int component=0;component<3;++component) {
        auto plus=positions,
            minus=positions;plus[atom_local_id][component]+=difference_step;minus[atom_local_id][component]-=difference_step;
        const double expected=-(energy(plus,channels)-energy(minus,channels))/(2*difference_step);
        max_force_error=std::max(max_force_error,std::abs(atom_forces[3*atom_local_id+component]-expected));
        near(atom_forces[3*atom_local_id+component],expected,3e-8,"force finite difference");
    }
    const int strain_rows[6]={0,1,2,1,2,2},strain_columns[6]={0,1,2,0,0,1};
    for(int component=0;component<6;++component) {
        auto plus=positions,minus=positions;
        for(int atom_local_id=0;atom_local_id<atom_count;++atom_local_id){plus[atom_local_id][strain_rows[component]]+=difference_step*positions[atom_local_id][strain_columns[component]];minus[atom_local_id][strain_rows[component]]-=difference_step*positions[atom_local_id][strain_columns[component]];}
        const double expected=-(energy(plus,channels)-energy(minus,channels))/(2*difference_step);
        double total_virial=0;for(int atom_local_id=0;atom_local_id<atom_count;++atom_local_id)total_virial+=atom_virials[6*atom_local_id+component];
        max_virial_error=std::max(max_virial_error,std::abs(total_virial-expected));
        near(total_virial,expected,5e-8,"strain finite difference");
    }
    for(int component=0;component<3;++component){double total_force=0;for(int atom_local_id=0;atom_local_id<atom_count;++atom_local_id)total_force+=atom_forces[3*atom_local_id+component];near(total_force,0,1e-13,"net force");}
    std::cout<<"channels="<<channel_count<<" max_force_error="<<max_force_error<<" max_virial_error="<<max_virial_error<<'\n';
    // 无中心 rank 的启动入口与可选 virial 指针。
    auto empty_geometry=kernel_view(data);empty_geometry.group_count=0;
    css_forward(empty_geometry,nullptr,0,d_channels.ptr,channel_count,nullptr,nullptr);
    css_backward(empty_geometry,nullptr,0,d_channels.ptr,channel_count,nullptr,nullptr,nullptr,nullptr,nullptr);
    check(cudaMemset(d_atom_forces.ptr,0,3*atom_count*sizeof(double)));
    css_backward(kernel_view(data),d_atom_types.ptr,atom_count,d_channels.ptr,channel_count,d_css_values.ptr,
        d_coordination_numbers.ptr,d_center_adjoints.ptr,d_atom_forces.ptr,nullptr);
    auto forces_without_virial=download(d_atom_forces,
        3*atom_count);for(int atom_local_id=0;atom_local_id<3*atom_count;++atom_local_id)near(forces_without_virial[atom_local_id],atom_forces[atom_local_id],1e-14,"optional virial");
}
int main() {
    try {
        const Positions positions={{0,0,0},{.8,.1,.2},{1.5,.3,-.1},{2.35,-.2,.4},
                           {3.6,.2,.1},{4.1,.8,.2},{2.0,1.1,.4},{12,0,0}};
        std::vector<CSSChannel> channels={
            {2,{FERMI,1.8,0,-3,6,12},0,.8,5},
            {1,{TANH_TYPE,2.2,0,-1.7,6,12},1.2,1.1,5},
            {3,{RATIONAL,1.6,0,0,6,12},.7,.9,5},
            {2,{FERMI,3.7,0,-2,6,12},1.5,1.3,5}};
        test(positions,{channels[0]});test(positions,{channels[0],channels[1]});test(positions,channels);
        channels[0].target=1e6;test(positions,{channels[0]});
        std::cout<<"CSS_KERNEL_PASS\n";
    } catch(const std::exception& exception){std::cerr<<exception.what()<<'\n';return 1;}
}
