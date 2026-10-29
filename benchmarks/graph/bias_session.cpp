#include "zqc_bias.h"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <chrono>
using namespace MetaD_zqc;
static void check(bool value,const char* message) {if(!value) throw std::runtime_error(message);}
int main() {
    try {
        const auto stamp=std::chrono::steady_clock::now().time_since_epoch().count();
        auto directory=std::filesystem::temp_directory_path()/("metad_bias_session_"+std::to_string(stamp));
        std::filesystem::create_directory(directory);
        for (auto backend:{BiasBackend::Grid,BiasBackend::SparseGrid,BiasBackend::Hills,BiasBackend::PBMetaD}) {
            BiasSettings settings;
            settings.backend=backend;settings.hills.axes={{0.45,0},{0.6,0}};
            settings.hills.initial_height=0.2;settings.hills.well_tempered=true;
            settings.grid_axes={{-3,3,0.45,72},{-3,3,0.6,80}};
            settings.history_path=(directory/BiasSession::name(backend)).string();
            settings.pace=2;settings.history_workers=4;
            std::vector<double> wire;
            std::size_t broadcasts=0;
            BiasHost root;root.broadcast=[&](double* values,std::size_t count){wire.assign(values,values+count);++broadcasts;};
            BiasHost follower;follower.root=false;
            follower.broadcast=[&](double* values,std::size_t count){check(count==wire.size(),"packet size");std::copy(wire.begin(),wire.end(),values);};
            double point[]={0.15,-0.31},gradient[2],other[2];
            double expected_energy;
            {
                BiasSession bias(settings,root),receiver(settings,follower);
                bias.initialize(0);receiver.initialize(0);
                check(!bias.advance(1,point),"pace ignored");
                for(int step=2;step<=20;step+=2) {
                    point[0]+=0.01;
                    check(bias.advance(step,point),"deposit missing");
                    check(!bias.advance(step,point),"duplicate deposition");
                }
                expected_energy=bias.evaluate(point,gradient);
                check(receiver.evaluate(point,other)==expected_energy,"host broadcast energy");
                check(std::memcmp(gradient,other,sizeof gradient)==0,"host broadcast gradient");
                check(broadcasts==1,"bias uses more than one broadcast");
                bias.initialize(20);
                check(bias.evaluate(point)==expected_energy,"reinitialize changed history");
            }
            settings.restart=true;
            {
                BiasSession resumed(settings);resumed.initialize(20);
                check(resumed.evaluate(point,other)==expected_energy,"restart energy bits changed");
                check(std::memcmp(gradient,other,sizeof gradient)==0,"restart gradient bits changed");
                check(!resumed.advance(20,point),"restart repeated deposition");
            }
            std::filesystem::remove(settings.history_path);
        }
        // 8192 行保证并行解析真正启动多个 worker；每一项 FP64 与分块结果逐位比较。
        HillBiasSettings settings;settings.axes.assign(8,{0.7,0});
        HillBias original(settings);std::ostringstream serialized;original.write_header(serialized);
        double point[8]={0},gradient[8];
        for(int step=0;step<8192;++step) {
            for(int axis=0;axis<8;++axis) point[axis]=std::sin(step*0.13+axis);
            original.deposit(step,point);original.write_last_deposition(serialized);
        }
        original.set_block_size(8192);
        const double expected=original.evaluate(point,gradient);
        for (unsigned workers:{1,4}) {
            HillBias loaded(settings);std::istringstream input(serialized.str());loaded.load(input,workers);
            check(loaded.centers()==original.centers() && loaded.heights()==original.heights() && loaded.steps()==original.steps(),
                  "parallel parse changed stored FP64 values");
            for(std::size_t block:{1,7,256,8192}) {
                loaded.set_block_size(block);double actual_gradient[8];
                const double actual=loaded.evaluate(point,actual_gradient);
                check(std::memcmp(&actual,&expected,sizeof(double))==0,"block energy bits changed");
                check(std::memcmp(actual_gradient,gradient,sizeof gradient)==0,"block gradient bits changed");
            }
        }
        // PLUMED 2.10 PB 固定宽度核：截断、最小镜像、导数和历史类型校验。
        {
            HillBiasSettings configuration;
            configuration.mode=HillBiasMode::Parallel;
            configuration.pb_kernel=PBKernel::Plumed210;
            configuration.axes={{0.4,8.0}};
            HillBias bias(configuration);
            double center=3.8,height=0.7,query=-3.9,derivative;
            bias.append(2,&center,&height);
            const double expected_value=height*std::exp(-0.5*0.75*0.75);
            check(std::abs(bias.evaluate(&query,&derivative)-expected_value)<1e-14,"PLUMED minimum image kernel");
            check(std::abs(derivative+expected_value*0.3/(0.4*0.4))<1e-14,"PLUMED derivative");
            query=center+4*0.4;
            check(bias.evaluate(&query,&derivative)==0 && derivative==0,"PLUMED cutoff");
            std::ostringstream saved;bias.write_header(saved);bias.write_last_deposition(saved);
            HillBias restored(configuration);std::istringstream input(saved.str());restored.load(input,4);
            query=-3.9;
            check(restored.evaluate(&query)==bias.evaluate(&query),"PLUMED restart");
            configuration.pb_kernel=PBKernel::SmoothGaussian;
            HillBias wrong_kernel(configuration);bool rejected=false;
            try {std::istringstream wrong(saved.str());wrong_kernel.load(wrong);}
            catch(const std::invalid_argument&) {rejected=true;}
            check(rejected,"different restart kernel accepted");
        }
        BiasSettings automatic;automatic.hills.axes.assign(8,{0.3,0});
        automatic.history_path=(directory/"automatic").string();
        BiasSession selected(automatic);
        check(selected.backend()==BiasBackend::PBMetaD,"AUTO did not select PB");
        std::filesystem::remove(directory);
        std::cout<<"BIAS_SESSION_PASS restart_bitwise=1 block_bitwise=1 parallel_parse=1 host_callbacks=1\n";
    } catch(const std::exception& failure) {std::cerr<<failure.what()<<"\n";return 1;}
}

