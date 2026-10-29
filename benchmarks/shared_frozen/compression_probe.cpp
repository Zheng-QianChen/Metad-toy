// Controlled compression experiment: CDF(x) -> q/(81-80q).
// Tables hold [coordinate, value, derivative] on a uniform grid. The CDF table
// represents the unbiased marginal; bias tables represent one shared u(s).
// Metropolis sweeps sample equilibrium and have no physical time meaning.
#include "zqc_hill_bias.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
thread_local std::string last_error;
using MetaD_zqc::HillBias;
using MetaD_zqc::HillBiasSettings;
struct Table {
    double lower=0, spacing=1.0/4096;
    std::vector<double> values, derivatives;
    Table(): values(4097,0), derivatives(4097,0) {}
    explicit Table(const char* path) {
        std::ifstream file(path); std::string line;
        if(!std::getline(file,line) || line!="# METAD_TOY_SHARED_FROZEN 1")
            throw std::runtime_error("invalid table header");
        std::vector<double> coordinates;
        while(std::getline(file,line)) {
            if(line.empty() || line[0]=='#') continue;
            std::istringstream row(line); double coordinate,value,derivative;
            if(!(row>>coordinate>>value>>derivative))
                throw std::runtime_error("invalid table row");
            coordinates.push_back(coordinate); values.push_back(value); derivatives.push_back(derivative);
        }
        if(values.size()<2) throw std::runtime_error("incomplete table");
        lower=coordinates[0]; spacing=coordinates[1]-lower;
        if(!(spacing>0)) throw std::runtime_error("nonpositive table spacing");
        for(size_t i=0;i<values.size();++i)
            if(std::abs(coordinates[i]-(lower+i*spacing))>1e-12)
                throw std::runtime_error("nonuniform table");
    }
    std::array<double,2> evaluate(double coordinate) const {
        const double upper=lower+spacing*(values.size()-1);
        if(coordinate<=lower || coordinate>=upper) {
            size_t node=coordinate<=lower?0:values.size()-1;
            return {values[node]+derivatives[node]*(coordinate-(lower+node*spacing)),derivatives[node]};
        }
        size_t node=std::min(size_t((coordinate-lower)/spacing),values.size()-2);
        double t=(coordinate-lower-node*spacing)/spacing;
        double linear=spacing*derivatives[node];
        double quadratic=3*(values[node+1]-values[node])-spacing*(2*derivatives[node]+derivatives[node+1]);
        double cubic=2*(values[node]-values[node+1])+spacing*(derivatives[node]+derivatives[node+1]);
        return {values[node]+t*(linear+t*(quadratic+t*cubic)),
                (linear+t*(2*quadratic+3*t*cubic))/spacing};
    }
    void write(const std::string& path) const {
        std::ofstream file(path);
        file<<"# METAD_TOY_SHARED_FROZEN 1\n"<<std::setprecision(17);
        for(size_t i=0;i<values.size();++i)
            file<<lower+i*spacing<<" "<<values[i]<<" "<<derivatives[i]<<"\n";
        if(!file) throw std::runtime_error("table write failed");
    }
    void add_hill(double center,double height,double sigma) {
        for(size_t i=0;i<values.size();++i) {
            double delta=lower+i*spacing-center;
            double contribution=height*std::exp(-0.5*delta*delta/(sigma*sigma));
            values[i]+=contribution;
            derivatives[i]-=contribution*delta/(sigma*sigma);
        }
    }
};
std::array<double,2> transform(const Table& cdf,double coordinate,bool compressed) {
    auto value=cdf.evaluate(coordinate);
    value[0]=std::max(0.,std::min(1.,value[0]));
    if(compressed) {
        double denominator=81-80*value[0];
        value[1]*=81/(denominator*denominator);
        value[0]/=denominator;
    }
    return value;
}
double potential(double x,double y) {
    return 4*((x*x-1)*(x*x-1)+(y*y-1)*(y*y-1))+(x-y)*(x-y);
}
struct Model {
    const Table& cdf; const Table& bias; bool compressed; double scale;
    double energy(double x,double y) const {
        if(std::abs(x)>=3 || std::abs(y)>=3) return INFINITY;
        return potential(x,y)+scale*(bias.evaluate(transform(cdf,x,compressed)[0])[0]+
                                    bias.evaluate(transform(cdf,y,compressed)[0])[0]);
    }
};
bool sweep(double& x,double& y,const Model& model,std::mt19937_64& random,
           std::normal_distribution<double>& normal) {
    double proposed_x=x+0.65*normal(random),proposed_y=y+0.65*normal(random);
    double delta=model.energy(proposed_x,proposed_y)-model.energy(x,y);
    double uniform=std::generate_canonical<double,53>(random);
    if(delta<=0 || uniform<std::exp(-delta)) {x=proposed_x;y=proposed_y;return true;}
    return false;
}
template<class Action> int guard(Action action) {
    try {action();last_error.clear();return 0;}
    catch(const std::exception& error) {last_error=error.what();return -1;}
}
}
extern "C" {
const char* compression_error() {return last_error.c_str();}
// Return [s(x), ds/dx, u(s(x)), du/dx] for chain-rule and CUDA checks.
int compression_evaluate(const char* cdf_path,const char* bias_path,int compressed,
                         int count,const double* coordinates,double* output) {
    return guard([&] {
        Table cdf(cdf_path),bias(bias_path);
        for(int i=0;i<count;++i) {
            auto mapped=transform(cdf,coordinates[i],compressed);
            auto value=bias.evaluate(mapped[0]);
            output[4*i]=mapped[0]; output[4*i+1]=mapped[1];
            output[4*i+2]=value[0]; output[4*i+3]=value[1]*mapped[1];
        }
    });
}
// Two centers deposit into one u(s), which enters total energy as (u1+u2)/2.
// The uniform sampling table is updated at every hill. The final table and
// intermediate coordinates are checked against production HillBias evaluation.
int compression_train(const char* cdf_path,const char* prefix,int compressed,
                      unsigned long long seed,int sweeps,double* diagnostics) {
    return guard([&] {
        Table cdf(cdf_path),bias;
        HillBiasSettings settings; settings.axes={{0.005,0}};settings.initial_height=0.02;
        HillBias history(settings);
        std::ofstream hills(std::string(prefix)+".hills"); history.write_header(hills);
        std::ofstream progress(std::string(prefix)+".progress.csv");
        progress<<"sweep,range_u,mean_u,low_mass\n"<<std::setprecision(17);
        std::mt19937_64 random(seed); std::normal_distribution<double> normal;
        Model model{cdf,bias,bool(compressed),0.5};
        double x=-1,y=-1; long long low_count=0;
        for(int step=1;step<=sweeps;++step) {
            sweep(x,y,model,random,normal);
            low_count+=(transform(cdf,x,compressed)[0]<=0.1);
            low_count+=(transform(cdf,y,compressed)[0]<=0.1);
            if(step%200==0) {
                for(double coordinate:{x,y}) {
                    double center=transform(cdf,coordinate,compressed)[0],height=0.02;
                    history.append(step,&center,&height,true);
                    history.write_last_deposition(hills);
                    bias.add_hill(center,height,0.005);
                }
            }
            if(step%50000==0) {
                auto range=std::minmax_element(bias.values.begin(),bias.values.end());
                double sum=0;for(double value:bias.values) sum+=value;
                progress<<step<<","<<*range.second-*range.first<<","<<sum/bias.values.size()
                        <<","<<double(low_count)/100000<<"\n";
                low_count=0;
            }
        }
        hills.close();progress.close();
        HillBias replay(settings); std::ifstream saved(std::string(prefix)+".hills"); replay.load(saved,2);
        double energy_error=0,derivative_error=0;
        for(int i=0;i<1003;++i) {
            double point=double(i)/1002,derivative=0,replay_derivative=0;
            double value=history.evaluate(&point,&derivative);
            double restored=replay.evaluate(&point,&replay_derivative);
            if(std::memcmp(&value,&restored,sizeof(double)) ||
               std::memcmp(&derivative,&replay_derivative,sizeof(double)))
                throw std::runtime_error("HILLS replay not bitwise identical");
            auto interpolated=bias.evaluate(point);
            energy_error=std::max(energy_error,std::abs(value-interpolated[0]));
            derivative_error=std::max(derivative_error,std::abs(derivative-interpolated[1]));
        }
        bias.write(std::string(prefix)+".table");
        diagnostics[0]=history.depositions();diagnostics[1]=energy_error;diagnostics[2]=derivative_error;
    });
}
// Output rows [x,y,B,u1,u2,s1,s2]. Rejected moves remain in the chain.
int compression_sample(const char* cdf_path,const char* bias_path,int compressed,double scale,
                       unsigned long long seed,int basin,int count,double* output,double* acceptance) {
    return guard([&] {
        Table cdf(cdf_path),bias(bias_path);Model model{cdf,bias,bool(compressed),scale};
        std::mt19937_64 random(seed);std::normal_distribution<double> normal;
        double x=(basin&1)?1:-1,y=(basin&2)?1:-1;
        const int burn=20000,stride=8; long long accepted=0,total=burn+static_cast<long long>(count)*stride;
        int row=0;
        for(long long step=0;step<total;++step) {
            accepted+=sweep(x,y,model,random,normal);
            if(step>=burn && (step-burn+1)%stride==0) {
                double sx=transform(cdf,x,compressed)[0],sy=transform(cdf,y,compressed)[0];
                double ux=bias.evaluate(sx)[0],uy=bias.evaluate(sy)[0];
                double* target=output+7*row++;
                target[0]=x;target[1]=y;target[2]=scale*(ux+uy);
                target[3]=ux;target[4]=uy;target[5]=sx;target[6]=sy;
            }
        }
        *acceptance=double(accepted)/total;
    });
}
}
