#include "zqc_hill_bias.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace MetaD_zqc;
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class Operation> void rejects(Operation operation) {
    bool rejected=false;
    try { operation(); } catch (const std::exception&) { rejected=true; }
    check(rejected,"invalid input was accepted");
}
int main() {
    try {
        double maximum_error=0;
        for (auto mode : {HillBiasMode::Hills,HillBiasMode::Parallel})
          for (bool well_tempered : {false,true})
            for (int dimension : {1,2,8,32,128}) {
                HillBiasSettings settings;
                settings.mode=mode;settings.axes.assign(dimension,{0.7,0});
                settings.well_tempered=well_tempered;settings.initial_height=0.2;
                HillBias bias(settings);
                std::ostringstream history;bias.write_header(history);
                std::vector<double> point(dimension),gradient(dimension);
                for (int step=0;step<24;++step) {
                    for (int axis=0;axis<dimension;++axis)
                        point[axis]=0.3*std::sin(0.31*step+axis)/std::sqrt(dimension);
                    check(bias.deposit(step,point.data()),"missing deposition");
                    bias.write_last_deposition(history);
                }
                check(!bias.deposit(23,point.data()),"duplicate deposition");
                rejects([&]{bias.deposit(22,point.data());});
                for (int axis=0;axis<dimension;++axis)
                    point[axis]=0.19*std::cos(axis)/std::sqrt(dimension);
                bias.evaluate(point.data(),gradient.data());
                check(*std::max_element(gradient.begin(),gradient.end())!=0,"zero derivative fixture");
                for (int axis=0;axis<dimension;++axis) {
                    const double original=point[axis], increment=1e-5;
                    point[axis]=original+increment;const double plus=bias.evaluate(point.data());
                    point[axis]=original-increment;const double minus=bias.evaluate(point.data());
                    point[axis]=original;
                    const double error=std::abs(gradient[axis]-(plus-minus)/(2*increment));
                    maximum_error=std::max(maximum_error,error);
                    check(error<2e-8,"bias finite difference failed");
                }
                HillBias restored(settings);std::istringstream input(history.str());restored.load(input);
                check(!restored.deposit(23,point.data()),"restart repeated last deposition");
                std::vector<double> restored_gradient(dimension);
                check(restored.evaluate(point.data(),restored_gradient.data())==bias.evaluate(point.data()),
                      "history energy changed");
                check(restored_gradient==gradient,"history derivative changed");
                for (int step=24;step<30;++step) { restored.deposit(step,point.data());bias.deposit(step,point.data()); }
                check(restored.evaluate(point.data())==bias.evaluate(point.data()),"restart continuation differs");
                std::istringstream incomplete(history.str().substr(0,history.str().size()-1));
                rejects([&]{restored.load(incomplete);});
                check(restored.evaluate(point.data())==bias.evaluate(point.data()),"failed load modified history");
                auto changed=settings;changed.axes[0].sigma*=2;HillBias incompatible(changed);
                std::istringstream wrong(history.str());rejects([&]{incompatible.load(wrong);});
            }
        // 周期接缝同时检查值与导数；极不均匀的一维偏势检查稳定 soft-min。
        for (auto mode : {HillBiasMode::Hills,HillBiasMode::Parallel}) {
            HillBiasSettings settings;settings.mode=mode;
            settings.axes={{0.4,6.283185307179586},{0.7,0}};
            HillBias bias(settings);double center[]={2.9,0.2};bias.deposit(0,center);
            double left[]={-3.141592653589793,0.4},right[]={3.141592653589793,0.4},lg[2],rg[2];
            check(std::abs(bias.evaluate(left,lg)-bias.evaluate(right,rg))<1e-14,"periodic energy seam");
            check(std::abs(lg[0]-rg[0])<1e-14,"periodic gradient seam");
            left[0]+=1e-5;const double plus=bias.evaluate(left);
            left[0]-=2e-5;const double minus=bias.evaluate(left);
            check(std::abs((plus-minus)/2e-5-lg[0])<1e-8,"periodic derivative");
        }
        HillBiasSettings settings;settings.mode=HillBiasMode::Parallel;settings.axes.assign(2,{0.5,0});
        settings.well_tempered=true;settings.initial_height=0.2;
        HillBias pb(settings);double center[]={0,0};pb.deposit(0,center);
        std::ostringstream first;pb.write_last_deposition(first);std::istringstream fields(first.str());
        long long step;double x,y,a,b;fields>>step>>x>>y>>a>>b;
        check(a==0.1 && b==0.1,"PB initial conditional height");
        double point[]={0.6,-0.2};pb.deposit(1,point);
        std::ostringstream second;pb.write_last_deposition(second);std::istringstream updated(second.str());
        updated>>step>>x>>y>>a>>b;
        const double v0=0.1*std::exp(-0.5*0.6*0.6/0.25);
        const double v1=0.1*std::exp(-0.5*0.2*0.2/0.25);
        const double denominator=std::exp(-v0)+std::exp(-v1);
        check(std::abs(a-0.2*std::exp(-v0)/denominator*std::exp(-v0/9))<1e-15,"PB old-state height 0");
        check(std::abs(b-0.2*std::exp(-v1)/denominator*std::exp(-v1/9))<1e-15,"PB old-state height 1");
        settings.initial_height=1e6;settings.well_tempered=false;HillBias stiff(settings);
        stiff.deposit(0,center);double far[]={0,100},gradient[2];
        check(std::isfinite(stiff.evaluate(far,gradient)) && std::isfinite(gradient[0]),"PB exponential underflow");
        // 空历史的最后一行也必须完整，保证后续记录从新行开始。
        std::ostringstream empty_header;stiff.write_header(empty_header);
        std::istringstream missing_newline(empty_header.str().substr(0,empty_header.str().size()-1));
        rejects([&]{stiff.load(missing_newline);});
        settings.axes[0].sigma=0;rejects([&]{HillBias invalid(settings);});
        std::cout<<"HILL_BIAS_PASS gradient_max_error="<<maximum_error<<"\n";
    } catch (const std::exception& error) {
        std::cerr<<error.what()<<"\n";return 1;
    }
}
