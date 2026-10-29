#include "zqc_shared_bias.cuh"
#include "zqc_gpu_buffer.cuh"
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>
using namespace MetaD_zqc;
void require(bool ok,const char* text) {if(!ok) throw std::runtime_error(text);}
void close(double a,double b,const char* text,double tolerance=2e-12) {require(std::abs(a-b)<tolerance,text);}
template<class T> void put(GpuBuffer<T>& buffer,const std::vector<T>& values) {
    buffer.grow_to(values.size(),__FILE__,__LINE__);buffer.upload_from(values.data(),values.size());
}
std::vector<double> get(ArrayView<const double> view) {
    view.wait_host();std::vector<double> data(view.item_count);
    if(!data.empty()) require(cudaMemcpy(data.data(),view.data,data.size()*sizeof(double),cudaMemcpyDeviceToHost)==cudaSuccess,"download");
    return data;
}
int main(int argc,char** argv) {
    MPI_Init(&argc,&argv);int rank,size;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    try {
        int devices=0;require(cudaGetDeviceCount(&devices)==cudaSuccess && devices>0,"CUDA required");
        MPI_Comm local;MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,0,MPI_INFO_NULL,&local);
        int local_rank;MPI_Comm_rank(local,&local_rank);MPI_Comm_free(&local);require(cudaSetDevice(local_rank%devices)==cudaSuccess,"select GPU");
        SharedOnlineHost host;
        host.root=rank==0;
        host.sum=[](double* data,size_t n){MPI_Allreduce(MPI_IN_PLACE,data,int(n),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);};
        host.broadcast=[](double* data,size_t n){MPI_Bcast(data,int(n),MPI_DOUBLE,0,MPI_COMM_WORLD);};
        const std::vector<double> centers{.2,.6,.8};
        std::vector<double> local_values{999};std::vector<int> local_slots,global_ids;
        for(int i=0;i<3;++i) if(i%size==rank) {
            local_slots.push_back(int(local_values.size()));local_values.push_back(centers[i]);
            global_ids.push_back(i);local_values.push_back(999);
        }
        GpuBuffer<double> input;GpuBuffer<int> slots;put(input,local_values);put(slots,local_slots);
        for(bool mean_energy:{false,true}) for(bool mean_deposit:{false,true}) for(bool wt:{false,true}) {
            SharedOnlineSettings config;config.nodes=101;config.sigma=.12;config.height=.035;
            config.pace=2;config.thermal_energy=.2;config.bias_factor=4;config.well_tempered=wt;
            config.aggregation=mean_energy ? SharedAggregation::Mean : SharedAggregation::Sum;
            config.mean_deposition=mean_deposit;config.identity="independent-three-center-fixture";
            config.history_path="shared-online-test-"+std::to_string(size)+"-"+std::to_string(mean_energy)+std::to_string(mean_deposit)+std::to_string(wt)+".history";
            if(!rank) std::remove(config.history_path.c_str());MPI_Barrier(MPI_COMM_WORLD);
            const double height=config.height/(mean_deposit?3:1),energy_scale=mean_energy?1./3:1;
            const auto gaussian=[&](double q,double c){double d=(q-c)/config.sigma;return std::exp(-.5*d*d);};
            const auto initial=[&](double q){double sum=0;for(double c:centers)sum+=height*gaussian(q,c);return sum;};
            {
                OnlineSharedBias bias(config,host);bias.initialize(0);
                close(bias.evaluate(input.view(local_values.size()),slots.view(local_slots.size())).energy,0,"initial zero");
                require(!bias.advance(1,input.view(local_values.size()),slots.view(local_slots.size())),"pace");
                require(bias.advance(2,input.view(local_values.size()),slots.view(local_slots.size())),"first deposit");
                require(!bias.advance(2,input.view(local_values.size()),slots.view(local_slots.size())),"duplicate deposit");
                const auto result=bias.evaluate(input.view(local_values.size()),slots.view(local_slots.size()));
                double expected=0;for(double q:centers)expected+=initial(q)*energy_scale;
                close(result.energy,expected,"first Gaussian energy");require(result.count==3,"global count including empty rank");
                close(result.mean_coordinate,(.2+.6+.8)/3,"monitor mean");
            }
            config.restart=true;
            {
                OnlineSharedBias bias(config,host);bias.initialize(2);
                require(bias.depositions()==1 && bias.last_deposition_step()==2,"history state");
                require(!bias.advance(2,input.view(local_values.size()),slots.view(local_slots.size())),"restart duplicate");
                require(bias.advance(4,input.view(local_values.size()),slots.view(local_slots.size())),"restart continuation");
                const auto result=bias.evaluate(input.view(local_values.size()),slots.view(local_slots.size()));
                const auto gradient=get(result.gradient);double expected=0;
                const auto potential=[&](double q){double sum=initial(q);for(double c:centers)
                    sum+=height*(wt?std::exp(-initial(c)/(config.thermal_energy*(config.bias_factor-1))):1)*gaussian(q,c);return sum;};
                for(double q:centers)expected+=potential(q)*energy_scale;
                close(result.energy,expected,"simultaneous WT history energy");
                for(size_t i=0;i<local_slots.size();++i) {
                    const double q=centers[global_ids[i]];double derivative=0;
                    for(double c:centers) derivative+=height*(1+(wt?std::exp(-initial(c)/(config.thermal_energy*(config.bias_factor-1))):1))
                        *gaussian(q,c)*(-(q-c)/(config.sigma*config.sigma))*energy_scale;
                    close(gradient[local_slots[i]],derivative,"WT derivative");
                }
                for(size_t i=0;i<gradient.size();i+=2)close(gradient[i],0,"unselected slots");
                const double h=1e-6;double energies[2];
                for(int sign=0;sign<2;++sign) {
                    auto displaced=local_values;
                    for(size_t i=0;i<global_ids.size();++i)if(global_ids[i]==0)displaced[local_slots[i]]+=(sign?1:-1)*h;
                    put(input,displaced);energies[sign]=bias.evaluate(input.view(local_values.size()),slots.view(local_slots.size())).energy;
                }
                put(input,local_values);double seed=0;
                for(size_t i=0;i<global_ids.size();++i)if(global_ids[i]==0)seed=gradient[local_slots[i]];
                host.sum(&seed,1);close((energies[1]-energies[0])/(2*h),seed,"interpolated coordinate finite difference",2e-8);
                // One bad rank must reject collectively without committing a record.
                auto bad_values=local_values;
                if(!rank)bad_values[local_slots[0]]=2;
                put(input,bad_values);bool rejected=false;
                try {bias.advance(6,input.view(local_values.size()),slots.view(local_slots.size()));}
                catch(const std::exception&) {rejected=true;}
                require(rejected && bias.depositions()==2,"out-of-grid transaction");put(input,local_values);
                auto bad_slots=local_slots;if(!rank)bad_slots.push_back(local_slots[0]);
                GpuBuffer<int> duplicated;put(duplicated,bad_slots);rejected=false;
                try {bias.advance(6,input.view(local_values.size()),duplicated.view(bad_slots.size()));}
                catch(const std::exception&) {rejected=true;}
                require(rejected && bias.depositions()==2,"duplicate-slot transaction");
            }
            auto expect_rejection=[&](SharedOnlineSettings wrong,long long step) {
                bool rejected=false;try {OnlineSharedBias bias(wrong,host);bias.initialize(step);} catch(const std::exception&) {rejected=true;}
                require(rejected,"invalid history accepted");
            };
            expect_rejection(config,2); // history contains step 4
            auto wrong=config;wrong.identity+="-changed";expect_rejection(wrong,4);
            wrong=config;wrong.height*=2;expect_rejection(wrong,4);
            wrong=config;wrong.restart=false;expect_rejection(wrong,4);
            config.pace=0;
            {OnlineSharedBias frozen(config,host);frozen.initialize(4);require(!frozen.advance(6,input.view(local_values.size()),slots.view(local_slots.size())),"freeze deposition");}
            if(!rank) {std::ofstream damage(config.history_path,std::ios::app);damage<<"STATE truncated";}
            MPI_Barrier(MPI_COMM_WORLD);expect_rejection(config,6);
            if(!rank) std::remove(config.history_path.c_str());MPI_Barrier(MPI_COMM_WORLD);
        }
        if(!rank)std::cout<<"PASS shared online: 8 aggregation/deposition/WT cases, exact Gaussian energy/gradient, empty ranks, FD, restart, freeze and corrupt-history rejection\n";
        MPI_Finalize();return 0;
    } catch(const std::exception& error) {std::cerr<<"rank "<<rank<<": "<<error.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}
