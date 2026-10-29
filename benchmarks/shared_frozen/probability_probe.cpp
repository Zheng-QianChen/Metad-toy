// Statistical test bridge. The production HillBias owns Gaussian evaluation and
// HILLS persistence. The small CPU table reader below is a sampling reference:
// it is checked separately against the production CUDA FrozenSharedBias.
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
constexpr double lower_bound=-3.0, spacing=0.0078125; // exactly representable
constexpr int table_nodes=769;
struct Table {
    std::vector<double> energy, slope;
    explicit Table(const char* path) {
        std::ifstream file(path); std::string line;
        if (!std::getline(file,line) || line!="# METAD_TOY_SHARED_FROZEN 1")
            throw std::runtime_error("invalid frozen table header");
        while (std::getline(file,line)) {
            if (line.empty() || line[0]=='#') continue;
            std::istringstream row(line); double coordinate, value, derivative;
            if (!(row>>coordinate>>value>>derivative) ||
                coordinate!=lower_bound+spacing*energy.size())
                throw std::runtime_error("unexpected validation table grid");
            energy.push_back(value); slope.push_back(derivative);
        }
        if (energy.size()!=table_nodes) throw std::runtime_error("incomplete validation table");
    }
    std::array<double,2> evaluate(double coordinate) const {
        const double upper=lower_bound+spacing*(table_nodes-1);
        if (coordinate<=lower_bound || coordinate>=upper) {
            const int node=coordinate<=lower_bound ? 0 : table_nodes-1;
            return {energy[node]+slope[node]*(coordinate-(lower_bound+node*spacing)),slope[node]};
        }
        int node=std::min(int((coordinate-lower_bound)/spacing),table_nodes-2);
        double t=(coordinate-lower_bound-node*spacing)/spacing;
        // Hermite coefficients written as a cubic polynomial, independently of
        // the basis-function expression in the production CUDA implementation.
        double constant=energy[node], linear=spacing*slope[node];
        double quadratic=3*(energy[node+1]-energy[node])-spacing*(2*slope[node]+slope[node+1]);
        double cubic=2*(energy[node]-energy[node+1])+spacing*(slope[node]+slope[node+1]);
        return {constant+t*(linear+t*(quadratic+t*cubic)),
                (linear+t*(2*quadratic+3*t*cubic))/spacing};
    }
};
HillBiasSettings settings() {
    HillBiasSettings result;
    result.axes={{0.45,0.0}};
    result.initial_height=0.025;
    result.thermal_energy=1.0;
    return result;
}
double potential(double x,double y,double coupling) {
    return 4*(std::pow(x*x-1,2)+std::pow(y*y-1,2))+0.5*coupling*(x-y)*(x-y);
}
// One symmetric random-walk Metropolis sweep updates both coordinates.
// This samples the Boltzmann distribution without a Langevin timestep error;
// sweep counts are sampling iterations and have no physical time interpretation.
template<class Energy> bool move(double& x,double& y,Energy energy,
        std::mt19937_64& random,std::normal_distribution<double>& normal) {
    double proposed_x=x+0.65*normal(random), proposed_y=y+0.65*normal(random);
    double delta=energy(proposed_x,proposed_y)-energy(x,y);
    double uniform=std::generate_canonical<double,53>(random);
    if (delta<=0 || uniform<std::exp(-delta)) {x=proposed_x;y=proposed_y;return true;}
    return false;
}
void export_table(const HillBias& bias,const std::string& path) {
    std::ofstream file(path);
    file<<std::setprecision(17)<<"# METAD_TOY_SHARED_FROZEN 1\n";
    for(int i=0;i<table_nodes;++i) {
        double q=lower_bound+i*spacing, derivative=0;
        double value=bias.evaluate(&q,&derivative);
        file<<q<<" "<<value<<" "<<derivative<<"\n";
    }
    if(!file) throw std::runtime_error("table write failed");
}
template<class Action> int guard(Action action) {
    try { action(); last_error.clear(); return 0; }
    catch(const std::exception& error) { last_error=error.what(); return -1; }
}
}
extern "C" {
const char* shared_probability_error() {return last_error.c_str();}

// kind=0: zero table; 1: asymmetric fixed hills; 2: both correlated CVs
// deposit simultaneously into one history, with 0.025 energy per center.
// Only the frozen final table is used by subsequent independent production runs.
int shared_probability_prepare(const char* prefix,int kind,unsigned long long seed,
                               double height_scale,double* diagnostics) {
    return guard([&] {
        HillBias bias(settings());
        std::ofstream history(std::string(prefix)+".hills");
        bias.write_header(history);
        if(kind==1) {
            const double centers[]={-1.0,0.1}, heights[]={2.0*height_scale,0.4*height_scale};
            for(int i=0;i<2;++i) {
                bias.append(i,&centers[i],&heights[i]); bias.write_last_deposition(history);
            }
        } else if(kind==2) {
            std::mt19937_64 random(seed); std::normal_distribution<double> normal;
            double x=seed%2 ? -1:1, y=-x;
            auto energy=[&](double px,double py) {
                return potential(px,py,2.0)+0.5*(bias.evaluate(&px)+bias.evaluate(&py));
            };
            for(int step=1;step<=24000;++step) {
                move(x,y,energy,random,normal);
                if(step%200==0) {
                    double height=0.025;
                    bias.append(step,&x,&height); bias.write_last_deposition(history);
                    bias.append(step,&y,&height,true); bias.write_last_deposition(history);
                }
            }
        } else if(kind!=0) throw std::runtime_error("invalid table kind");
        history.close();
        HillBias restored(settings());
        std::ifstream input(std::string(prefix)+".hills");
        restored.load(input,2);
        if(bias.centers()!=restored.centers() || bias.heights()!=restored.heights() ||
           bias.steps()!=restored.steps()) throw std::runtime_error("history content changed");
        export_table(bias,std::string(prefix)+".table");
        export_table(restored,std::string(prefix)+".reloaded.table");
        double history_error=0, interpolation_error=0, slope_error=0, fd_error=0;
        Table table((std::string(prefix)+".table").c_str());
        for(int i=0;i<601;++i) {
            double q=-2.4+i*0.008, derivative=0, restored_derivative=0;
            double value=bias.evaluate(&q,&derivative);
            double replay=restored.evaluate(&q,&restored_derivative);
            if(std::memcmp(&value,&replay,sizeof(double)) ||
               std::memcmp(&derivative,&restored_derivative,sizeof(double)))
                throw std::runtime_error("history evaluation not bitwise reproducible");
            auto sample=table.evaluate(q);
            interpolation_error=std::max(interpolation_error,std::abs(value-sample[0]));
            slope_error=std::max(slope_error,std::abs(derivative-sample[1]));
            double h=1e-5;
            fd_error=std::max(fd_error,std::abs((table.evaluate(q+h)[0]-table.evaluate(q-h)[0])/(2*h)-sample[1]));
        }
        diagnostics[0]=double(bias.depositions()); diagnostics[1]=history_error;
        diagnostics[2]=interpolation_error; diagnostics[3]=slope_error; diagnostics[4]=fd_error;
    });
}
int shared_probability_evaluate(const char* path,int count,const double* points,double* result) {
    return guard([&] {
        Table table(path);
        for(int i=0;i<count;++i) {
            auto value=table.evaluate(points[i]);
            result[2*i]=value[0]; result[2*i+1]=value[1];
        }
    });
}
// Output rows are [x,y,B,u(x),u(y)]. Every retained row represents the same
// number of Metropolis sweeps, including rejected proposals.
int shared_probability_sample(const char* path,double coupling,double scale,
        unsigned long long seed,int initial_basin,int burn,int count,int stride,
        double* samples,double* acceptance) {
    return guard([&] {
        if(count<=0 || stride<=0 || burn<0) throw std::runtime_error("invalid sample size");
        Table table(path); std::mt19937_64 random(seed); std::normal_distribution<double> normal;
        double x=(initial_basin&1)?1:-1, y=(initial_basin&2)?1:-1;
        auto energy=[&](double px,double py) {
            return potential(px,py,coupling)+scale*(table.evaluate(px)[0]+table.evaluate(py)[0]);
        };
        long long accepted=0, sweeps=burn+static_cast<long long>(count)*stride;
        int row=0;
        for(long long step=0;step<sweeps;++step) {
            accepted+=move(x,y,energy,random,normal);
            if(step>=burn && (step-burn+1)%stride==0) {
                double ux=table.evaluate(x)[0], uy=table.evaluate(y)[0];
                double* output=samples+5*row++;
                output[0]=x; output[1]=y; output[2]=scale*(ux+uy);
                output[3]=ux; output[4]=uy;
            }
        }
        *acceptance=double(accepted)/sweeps;
    });
}
}
