#include "zqc_bias.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>
using namespace MetaD_zqc;
using Clock=std::chrono::steady_clock;
template<class Operation> double median_time(Operation operation) {
    operation();
    std::vector<double> samples;
    for(int repeat=0;repeat<5;++repeat) {
        auto begin=Clock::now();operation();
        samples.push_back(std::chrono::duration<double,std::milli>(Clock::now()-begin).count());
    }
    std::sort(samples.begin(),samples.end());return samples[2];
}
int main() {
    try {
        // 同一组 hills、同一宽度和查询点，两条生产后端交叉检查。
        for(int dimension:{1,2,3}) {
            HillBiasSettings settings;settings.axes.assign(dimension,{0.45,0});
            HillBias list(settings);std::vector<GridAxis> axes;
            for(int axis=0;axis<dimension;++axis) axes.push_back({-3,3,0.45,(dimension==3 ? 128 : 192)+8*axis,0});
            BiasGrid grid(axes),sparse(axes,true);
            std::vector<double> point(dimension),list_gradient(dimension),grid_gradient(dimension);
            for(int hill=0;hill<20;++hill) {
                for(int axis=0;axis<dimension;++axis) point[axis]=0.8*std::sin(0.31*hill+axis);
                const double height=0.2+0.01*hill;
                list.append(hill,point.data(),&height);
                grid.add(point.data(),height);sparse.add(point.data(),height);
            }
            double squared=0,gradient_error=0,sparse_error=0;
            for(int sample=0;sample<1000;++sample) {
                for(int axis=0;axis<dimension;++axis) point[axis]=1.4*std::sin(0.71*sample+0.4*axis);
                const double exact=list.evaluate(point.data(),list_gradient.data());
                const double interpolated=grid.evaluate(point.data(),grid_gradient.data());
                squared+=(exact-interpolated)*(exact-interpolated);
                sparse_error=std::max(sparse_error,std::abs(interpolated-sparse.evaluate(point.data())));
                for(int axis=0;axis<dimension;++axis)
                    gradient_error=std::max(gradient_error,std::abs(grid_gradient[axis]-list_gradient[axis]));
            }
            const double rmse=std::sqrt(squared/1000);
            const bool passed=rmse<3e-4 && gradient_error<0.015 && sparse_error==0;
            std::cout<<"{\"kind\":\"grid_list_cross\",\"D\":"<<dimension<<",\"energy_rmse\":"<<rmse
                     <<",\"wt_fes_rmse\":"<<rmse*(10.0/9)<<",\"gradient_max_error\":"<<gradient_error
                     <<",\"dense_sparse_error\":"<<sparse_error<<",\"passed\":"<<(passed?"true":"false")<<"}\n";
            if(!passed) return 1;
        }
        for(int dimension:{2,3}) {
            std::vector<GridAxis> axes(dimension,{-2,2,0.08,dimension==2 ? 192 : 48,0});
            const std::size_t hills=dimension==2 ? 3000 : 1000;
            std::vector<double> centers(hills*dimension),heights(hills,0.1);
            for(std::size_t hill=0;hill<hills;++hill)
                for(int axis=0;axis<dimension;++axis) centers[hill*dimension+axis]=1.6*std::sin(hill*0.173+axis);
            for(bool sparse:{false,true}) {
                BiasGrid grid(axes,sparse);
                double reference=0;
                for(unsigned workers:{1u,4u}) {
                    double ms=median_time([&]{grid.rebuild(centers.data(),heights.data(),hills,workers);});
                    double query[3]={0.13,-0.1,0.2};const double result=grid.evaluate(query);
                    if(workers==1) reference=result;
                    const bool exact=std::memcmp(&result,&reference,sizeof(double))==0;
                    std::cout<<"{\"kind\":\"grid_rebuild\",\"D\":"<<dimension<<",\"H\":"<<hills
                             <<",\"sparse\":"<<(sparse?"true":"false")<<",\"workers\":"<<workers
                             <<",\"median_ms\":"<<ms<<",\"passed\":"<<(exact?"true":"false")<<"}\n";
                    if(!exact) return 1;
                }
            }
        }
        HillBiasSettings settings;settings.axes.assign(8,{0.5,0});
        HillBias source(settings);std::ostringstream output;source.write_header(output);
        std::vector<double> point(8,0.2);
        for(int hill=0;hill<50000;++hill) {
            source.deposit(hill,point.data());source.write_last_deposition(output);
        }
        const std::string text=output.str();
        for(unsigned workers:{1u,4u}) {
            HillBias restored(settings);
            double ms=median_time([&]{std::istringstream input(text);restored.load(input,workers);});
            const bool exact=restored.centers()==source.centers() && restored.heights()==source.heights();
            std::cout<<"{\"kind\":\"history_parse\",\"D\":8,\"H\":50000,\"bytes\":"<<text.size()
                     <<",\"workers\":"<<workers<<",\"median_ms\":"<<ms<<",\"passed\":"<<(exact?"true":"false")<<"}\n";
            if(!exact) return 1;
        }
    } catch(const std::exception& error) {std::cerr<<error.what()<<"\n";return 1;}
}

