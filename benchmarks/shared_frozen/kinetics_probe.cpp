// Langevin kinetics benchmark in reduced units: mass=kBT=friction=1.
// BAOAB advances positions AND velocities. A force evaluation, rather than a
// Metropolis proposal, determines the displacement at every MD step.
#include "zqc_hill_bias.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
thread_local std::string last_error;
struct Table {
    double lower=0,spacing=1./4096;
    std::vector<double> value,slope;
    Table(double begin,double end):lower(begin),spacing((end-begin)/4096),
        value(4097,0),slope(4097,0) {}
    explicit Table(const char* path) {
        std::ifstream input(path);std::string line;
        if(!std::getline(input,line)||line!="# METAD_TOY_SHARED_FROZEN 1")
            throw std::runtime_error("invalid table header");
        std::vector<double> coordinate;
        while(std::getline(input,line)) {
            if(line.empty()||line[0]=='#')continue;
            std::istringstream row(line);double x,v,d;
            if(!(row>>x>>v>>d))throw std::runtime_error("invalid table row");
            coordinate.push_back(x);value.push_back(v);slope.push_back(d);
        }
        if(value.size()<2)throw std::runtime_error("empty table");
        lower=coordinate[0];spacing=coordinate[1]-lower;
        if(!(spacing>0))throw std::runtime_error("invalid spacing");
        for(size_t i=0;i<value.size();++i)
            if(std::abs(coordinate[i]-lower-i*spacing)>1e-12)
                throw std::runtime_error("nonuniform table");
    }
    std::array<double,2> evaluate(double x)const {
        const double upper=lower+spacing*(value.size()-1);
        if(x<=lower||x>=upper) {
            size_t node=x<=lower?0:value.size()-1;
            return {value[node]+slope[node]*(x-lower-node*spacing),slope[node]};
        }
        size_t node=std::min(size_t((x-lower)/spacing),value.size()-2);
        double t=(x-lower-node*spacing)/spacing,linear=spacing*slope[node];
        double quadratic=3*(value[node+1]-value[node])-spacing*(2*slope[node]+slope[node+1]);
        double cubic=2*(value[node]-value[node+1])+spacing*(slope[node]+slope[node+1]);
        return {value[node]+t*(linear+t*(quadratic+t*cubic)),
                (linear+t*(2*quadratic+3*t*cubic))/spacing};
    }
    void deposit(double center,double height,double sigma) {
        // Eight-sigma truncation bounds each omitted value by height*exp(-32).
        // Direct HillBias comparison checks the accumulated interpolation error.
        int first=std::max(0,int(std::floor((center-8*sigma-lower)/spacing)));
        int last=std::min(int(value.size())-1,int(std::ceil((center+8*sigma-lower)/spacing)));
        for(int i=first;i<=last;++i) {
            double delta=lower+i*spacing-center;
            double contribution=height*std::exp(-0.5*delta*delta/(sigma*sigma));
            value[i]+=contribution;slope[i]-=contribution*delta/(sigma*sigma);
        }
    }
    void write(const std::string& path) const {
        std::ofstream file(path);file<<"# METAD_TOY_SHARED_FROZEN 1\n"<<std::setprecision(17);
        for(size_t i=0;i<value.size();++i)
            file<<lower+i*spacing<<" "<<value[i]<<" "<<slope[i]<<"\n";
    }
};
double physical_energy(double x,double y) {
    return 4*((x*x-1)*(x*x-1)+(y*y-1)*(y*y-1))+(x-y)*(x-y);
}
struct Model {
    const Table& cdf;Table table;int mode;
    // 0=unbiased; 1=frozen CDF; 2=frozen compressed CDF;
    // 3=frozen table on physical x; 4=online x; 5=online compressed CDF.
    Model(const Table& cdf_,const char* path,int mode_,bool start_empty=true):
        cdf(cdf_),table(path),mode(mode_) {
        if(mode>=4&&start_empty)table=Table(mode==4?-3:0,mode==4?3:1);
    }
    std::array<double,2> coordinate(double x)const {
        if(mode==0||mode==3||mode==4)return {x,1};
        auto mapped=cdf.evaluate(x);
        mapped[0]=std::max(0.,std::min(1.,mapped[0]));
        if(mode==2||mode==5) {
            double denominator=81-80*mapped[0];
            mapped[1]*=81/(denominator*denominator);mapped[0]/=denominator;
        }
        return mapped;
    }
    std::array<double,2> local_bias(double x)const {
        if(mode<=0)return {0,0};
        auto mapped=coordinate(x);auto bias=table.evaluate(mapped[0]);
        return {0.5*bias[0],0.5*bias[1]*mapped[1]};
    }
    std::array<double,3> evaluate(double x,double y)const {
        if(mode<0)return {0,-x,-y}; // harmonic dynamical calibration
        auto left=local_bias(x),right=local_bias(y);
        return {left[0]+right[0],
                -16*x*(x*x-1)-2*(x-y)-left[1],
                -16*y*(y*y-1)+2*(x-y)-right[1]};
    }
};
struct State {double x,y,vx,vy,fx,fy;};
void update_force(State& state,const Model& model) {
    auto values=model.evaluate(state.x,state.y);state.fx=values[1];state.fy=values[2];
    if(!std::isfinite(state.fx)||!std::isfinite(state.fy)||(model.mode>=0&&(std::abs(state.x)>4||std::abs(state.y)>4)))
        throw std::runtime_error("unstable MD trajectory");
}
void baoab(State& state,const Model& model,double step,double damping,double noise,
           std::mt19937_64& random,std::normal_distribution<double>& gaussian) {
    state.vx+=0.5*step*state.fx;state.vy+=0.5*step*state.fy;
    state.x+=0.5*step*state.vx;state.y+=0.5*step*state.vy;
    state.vx=damping*state.vx+noise*gaussian(random);
    state.vy=damping*state.vy+noise*gaussian(random);
    state.x+=0.5*step*state.vx;state.y+=0.5*step*state.vy;
    update_force(state,model);
    state.vx+=0.5*step*state.fx;state.vy+=0.5*step*state.fy;
}
double uniform(std::mt19937_64& random){return std::generate_canonical<double,53>(random);}
State initial_state(std::mt19937_64& random,std::normal_distribution<double>& gaussian,
                    const Model& model,int direction,bool equilibrium) {
    // First passages share the same ORIGINAL conditional basin ensemble across
    // all biases. Stationary runs instead start from their own biased ensemble.
    State state{};
    bool accepted=false;
    for(int attempt=0;attempt<10000000;++attempt) {
        if(equilibrium) {state.x=-3+6*uniform(random);state.y=-3+6*uniform(random);}
        else {
            state.x=-3+2.4*uniform(random);state.y=-3+2.4*uniform(random);
            if(direction>0){state.x=-state.x;state.y=-state.y;}
        }
        double energy=physical_energy(state.x,state.y);
        if(equilibrium)energy+=model.evaluate(state.x,state.y)[0];
        if(energy<-1e-8)throw std::runtime_error("rejection envelope invalid");
        if(uniform(random)<std::exp(-energy)){accepted=true;break;}
    }
    if(!accepted)throw std::runtime_error("initialization failed");
    state.vx=gaussian(random);state.vy=gaussian(random);
    update_force(state,model);return state;
}
int basin(const State& state) {
    if(state.x<-.6&&state.y<-.6)return 0;
    if(state.x>.6&&state.y>.6)return 1;
    return -1;
}
template<class Function> int guard(Function function) {
    try {function();last_error.clear();return 0;}
    catch(const std::exception& error){last_error=error.what();return -1;}
}
}
extern "C" {
const char* kinetics_error(){return last_error.c_str();}
int kinetics_evaluate(const char* cdf_path,const char* table_path,int mode,int count,
                      const double* coordinates,double* output) {
    return guard([&]{
        Table cdf(cdf_path);Model model(cdf,table_path,mode);
        for(int i=0;i<count;++i) {
            auto value=model.evaluate(coordinates[2*i],coordinates[2*i+1]);
            output[3*i]=value[0]+physical_energy(coordinates[2*i],coordinates[2*i+1]);
            output[3*i+1]=value[1];output[3*i+2]=value[2];
        }
    });
}
// One independently seeded path returns [min(T,cap), event, hills, final x,y,
// force evaluations, maximum |bias force|, Gaussian/table error in energy/gradient].
int kinetics_first_passage(const char* cdf_path,const char* table_path,int mode,double step,
        double cap,unsigned long long seed,int direction,const char* history_prefix,double* output) {
    return guard([&]{
        Table cdf(cdf_path);Model model(cdf,table_path,mode);
        std::mt19937_64 random(seed);std::normal_distribution<double> gaussian;
        State state=initial_state(random,gaussian,model,direction,false);
        double damping=std::exp(-step),noise=std::sqrt(-std::expm1(-2*step));
        long long limit=std::llround(cap/step),deposition_stride=std::llround(.5/step);
        if(std::abs(limit*step-cap)>1e-9||std::abs(deposition_stride*step-.5)>1e-9)
            throw std::runtime_error("time grid does not resolve cap/deposition pace");
        MetaD_zqc::HillBiasSettings settings;settings.axes={{mode==4?.12:.02,0}};
        settings.initial_height=.02;
        MetaD_zqc::HillBias history(settings);
        bool record=history_prefix&&*history_prefix&&mode>=4;
        std::ofstream history_file;
        if(record){history_file.open(std::string(history_prefix)+".hills");history.write_header(history_file);}
        double max_bias_force=0;long long deposits=0,completed=0;bool event=false;
        for(long long tick=1;tick<=limit;++tick) {
            baoab(state,model,step,damping,noise,random,gaussian);completed=tick;
            max_bias_force=std::max(max_bias_force,std::max(std::abs(model.local_bias(state.x)[1]),
                                                           std::abs(model.local_bias(state.y)[1])));
            if(basin(state)==(direction>0?0:1)){event=true;break;}
            if(mode>=4&&tick%deposition_stride==0) {
                for(double position:{state.x,state.y}) {
                    double center=model.coordinate(position)[0],height=.02;
                    model.table.deposit(center,height,settings.axes[0].sigma);++deposits;
                    if(record){history.append(tick,&center,&height,true);history.write_last_deposition(history_file);}
                }
                update_force(state,model); // refreshed force after changing the Hamiltonian
            }
        }
        double energy_error=0,gradient_error=0;
        if(record) {
            history_file.close();model.table.write(std::string(history_prefix)+".table");
            for(int i=0;i<1001;++i) {
                double coordinate=model.table.lower+(model.table.value.size()-1)*model.table.spacing*i/1000;
                double gradient=0,value=history.evaluate(&coordinate,&gradient);
                auto tabulated=model.table.evaluate(coordinate);
                energy_error=std::max(energy_error,std::abs(value-tabulated[0]));
                gradient_error=std::max(gradient_error,std::abs(gradient-tabulated[1]));
            }
        }
        output[0]=completed*step;output[1]=event;output[2]=deposits;output[3]=state.x;output[4]=state.y;
        output[5]=completed+1+(mode>=4?deposits/2:0);output[6]=max_bias_force;
        output[7]=energy_error;output[8]=gradient_error;
    });
}
// Equal physical recording interval .1 at both step sizes. Output rows contain
// [x,y,B,vx^2,vy^2,last committed basin]; hysteresis removes rapid dividing-surface recrossings.
int kinetics_stationary(const char* cdf_path,const char* table_path,int mode,double step,
        double duration,unsigned long long seed,double* samples,double* diagnostics) {
    return guard([&]{
        if(mode>=4)throw std::runtime_error("stationary test requires frozen bias");
        Table cdf(cdf_path);Model model(cdf,table_path,mode);
        std::mt19937_64 random(seed);std::normal_distribution<double> gaussian;
        State state=initial_state(random,gaussian,model,0,true);
        double damping=std::exp(-step),noise=std::sqrt(-std::expm1(-2*step));
        long long stride=std::llround(.1/step),limit=std::llround(duration/step);
        int previous=basin(state),transitions=0,left_to_right=0,right_to_left=0,row=0;
        double time_left=0,time_right=0;
        for(long long tick=1;tick<=limit;++tick) {
            baoab(state,model,step,damping,noise,random,gaussian);
            int current=basin(state);
            if(current>=0) {
                if(previous>=0&&current!=previous) {
                    ++transitions;if(current==1)++left_to_right;else++right_to_left;
                }
                previous=current;
            }
            if(previous==0)time_left+=step;if(previous==1)time_right+=step;
            if(tick%stride==0) {
                auto values=model.evaluate(state.x,state.y);
                double* target=samples+6*row++;
                target[0]=state.x;target[1]=state.y;target[2]=values[0];
                target[3]=state.vx*state.vx;target[4]=state.vy*state.vy;target[5]=previous;
            }
        }
        diagnostics[0]=transitions;diagnostics[1]=left_to_right;diagnostics[2]=right_to_left;
        diagnostics[3]=time_left;diagnostics[4]=time_right;diagnostics[5]=limit+1;
    });
}
// An analytic harmonic-oscillator check tests the MD clock and velocity memory,
// beyond invariant-density tests. Two independent components supply 2*replicas
// samples of <x(t)x(0)> at each physical lag.
int kinetics_harmonic(const char* cdf_path,const char* table_path,double step,
                      int replicas,double* output) {
    return guard([&]{
        Table cdf(cdf_path);Model model(cdf,table_path,-1);
        const double lags[]={.1,.5,1,2,5};
        double sums[5]={},squares[5]={},temperature[5]={};
        std::mt19937_64 random(827341);std::normal_distribution<double> gaussian;
        double damping=std::exp(-step),noise=std::sqrt(-std::expm1(-2*step));
        for(int replica=0;replica<replicas;++replica) {
            State state{gaussian(random),gaussian(random),gaussian(random),gaussian(random),0,0};
            double initial_x=state.x,initial_y=state.y;update_force(state,model);int index=0;
            for(long long tick=1;tick<=std::llround(5/step);++tick) {
                baoab(state,model,step,damping,noise,random,gaussian);
                if(tick==std::llround(lags[index]/step)) {
                    double first=state.x*initial_x,second=state.y*initial_y;
                    sums[index]+=first+second;squares[index]+=first*first+second*second;
                    temperature[index]+=state.vx*state.vx+state.vy*state.vy;
                    ++index;if(index==5)break;
                }
            }
        }
        for(int index=0;index<5;++index) {
            double count=2.*replicas,mean=sums[index]/count;
            double variance=(squares[index]-count*mean*mean)/(count-1);
            double frequency=std::sqrt(.75),lag=lags[index];
            output[6*index]=lag;output[6*index+1]=mean;
            output[6*index+2]=std::sqrt(variance/count);
            output[6*index+3]=std::exp(-.5*lag)*(std::cos(frequency*lag)+.5/frequency*std::sin(frequency*lag));
            output[6*index+4]=temperature[index]/count;output[6*index+5]=count;
        }
    });
}

// Fixed-duration online MD training. Each CV family produces its own table.
// This uses exactly the height, physical deposition pace and mapping of the
// earlier online experiments, with an explicit end time independent of escape.
int kinetics_train_shared(const char* cdf_path,const char* empty_table_path,int mode,
        double step,double duration,unsigned long long seed,int direction,
        const char* prefix,double* output) {
    return guard([&]{
        if(mode!=4&&mode!=5)throw std::runtime_error("training mode must be physical or compressed");
        Table cdf(cdf_path);Model model(cdf,empty_table_path,mode,true);
        std::mt19937_64 random(seed);std::normal_distribution<double> gaussian;
        State state=initial_state(random,gaussian,model,direction,false);
        const long long limit=std::llround(duration/step),pace=std::llround(.5/step);
        if(std::abs(limit*step-duration)>1e-9||std::abs(pace*step-.5)>1e-9||limit%pace)
            throw std::runtime_error("training horizon must align with the physical deposition pace");
        const double damping=std::exp(-step),noise=std::sqrt(-std::expm1(-2*step));
        MetaD_zqc::HillBiasSettings settings;settings.axes={{mode==4?.12:.02,0}};
        settings.initial_height=.02;MetaD_zqc::HillBias history(settings);
        std::ofstream history_file(std::string(prefix)+".hills");history.write_header(history_file);
        double kinetic_sum=0;long long changes=0;int previous=basin(state);
        for(long long tick=1;tick<=limit;++tick) {
            baoab(state,model,step,damping,noise,random,gaussian);
            kinetic_sum+=.5*(state.vx*state.vx+state.vy*state.vy);
            int current=basin(state);
            if(current>=0){if(previous>=0&&current!=previous)++changes;previous=current;}
            if(tick%pace==0) {
                for(double position:{state.x,state.y}) {
                    double center=model.coordinate(position)[0],height=.02;
                    history.append(tick,&center,&height,true);history.write_last_deposition(history_file);
                    model.table.deposit(center,height,settings.axes[0].sigma);
                }
                update_force(state,model);
            }
        }
        history_file.close();model.table.write(std::string(prefix)+".table");
        double energy_error=0,gradient_error=0;
        for(int point=0;point<1001;++point) {
            double coordinate=model.table.lower+(model.table.value.size()-1)*model.table.spacing*point/1000;
            double gradient=0,value=history.evaluate(&coordinate,&gradient);
            auto interpolation=model.table.evaluate(coordinate);
            energy_error=std::max(energy_error,std::abs(value-interpolation[0]));
            gradient_error=std::max(gradient_error,std::abs(gradient-interpolation[1]));
        }
        output[0]=history.depositions();output[1]=changes;output[2]=kinetic_sum/limit;
        output[3]=state.x;output[4]=state.y;output[5]=energy_error;output[6]=gradient_error;
    });
}
// Load the same MD-trained table for both arms. Independent validation paths
// share physical starting coordinates, velocities and thermal-noise seeds.
// continue_deposition changes only subsequent hill updates; the input table
// remains unchanged on disk. The initial state and force are returned for an
// exact paired-arm check in Python.
int kinetics_matched_passage(const char* cdf_path,const char* table_path,int mode,
        int continue_deposition,double step,double cap,unsigned long long seed,
        int direction,double* output) {
    return guard([&]{
        if(mode!=0&&mode!=4&&mode!=5)throw std::runtime_error("invalid paired mode");
        if(mode==0&&continue_deposition)throw std::runtime_error("unbiased mode cannot deposit");
        Table cdf(cdf_path);Model model(cdf,table_path,mode,false);
        std::mt19937_64 random(seed);std::normal_distribution<double> gaussian;
        State state=initial_state(random,gaussian,model,direction,false);
        auto initial=model.evaluate(state.x,state.y);
        output[5]=state.x;output[6]=state.y;output[7]=state.vx;output[8]=state.vy;
        output[9]=initial[0];output[10]=initial[1];output[11]=initial[2];
        const long long limit=std::llround(cap/step),pace=std::llround(.5/step);
        if(std::abs(limit*step-cap)>1e-9||std::abs(pace*step-.5)>1e-9)
            throw std::runtime_error("invalid MD clock");
        const double damping=std::exp(-step),noise=std::sqrt(-std::expm1(-2*step));
        const double sigma=mode==4?.12:.02;
        long long completed=0,hills=0;bool event=false;double maximum_force=0;
        for(long long tick=1;tick<=limit;++tick) {
            baoab(state,model,step,damping,noise,random,gaussian);completed=tick;
            maximum_force=std::max(maximum_force,std::max(std::abs(model.local_bias(state.x)[1]),
                                                         std::abs(model.local_bias(state.y)[1])));
            if(basin(state)==(direction>0?0:1)){event=true;break;}
            if(continue_deposition&&tick%pace==0) {
                for(double position:{state.x,state.y}) {
                    model.table.deposit(model.coordinate(position)[0],.02,sigma);++hills;
                }
                update_force(state,model);
            }
        }
        output[0]=completed*step;output[1]=event;output[2]=hills;
        output[3]=completed+1+hills/2;output[4]=maximum_force;
        output[12]=state.x;output[13]=state.y;
    });
}


// Frozen-bias FES sampling starts in one of four coordinate quadrants.
// Burn-in advances the same Langevin trajectory; only the subsequent fixed
// duration is recorded. The physical CV is x/y, so no reference CDF is loaded.
// Rows: [x, y, full B=(u(x)+u(y))/2, vx^2, vy^2, last committed core].
// Diagnostics: transitions, A->B, B->A, time in last A/B, force evaluations,
// all-step production kinetic temperature, and recorded row count.
int kinetics_fes_sample(const char* table_path,int biased,double step,
        double burn_time,double duration,unsigned long long seed,int initial_quadrant,
        int sample_capacity,double* samples,double* diagnostics) {
    return guard([&]{
        if(!(step>0)||!std::isfinite(step)||!(burn_time>=0)||!std::isfinite(burn_time)||
           !(duration>0)||!std::isfinite(duration)||initial_quadrant<0||initial_quadrant>3)
            throw std::runtime_error("invalid frozen FES sampling parameters");
        const long long stride=std::llround(.1/step);
        const long long burn_steps=std::llround(burn_time/step),steps=std::llround(duration/step);
        if(stride<1||std::abs(stride*step-.1)>1e-9||
           std::abs(burn_steps*step-burn_time)>1e-9||
           std::abs(steps*step-duration)>1e-9||steps%stride||
           steps/stride!=sample_capacity)
            throw std::runtime_error("FES recording clock or buffer size mismatch");
        Table unused_mapping(-3,3);
        Model model(unused_mapping,table_path,biased?3:0);
        std::mt19937_64 random(seed);std::normal_distribution<double> gaussian;
        State state{initial_quadrant&2?1.:-1.,initial_quadrant&1?1.:-1.,
                    gaussian(random),gaussian(random),0,0};
        update_force(state,model);
        const double damping=std::exp(-step),noise=std::sqrt(-std::expm1(-2*step));
        for(long long tick=0;tick<burn_steps;++tick)
            baoab(state,model,step,damping,noise,random,gaussian);
        int previous=basin(state),transitions=0,left_to_right=0,right_to_left=0,row=0;
        double time_left=0,time_right=0,kinetic_sum=0;
        for(long long tick=1;tick<=steps;++tick) {
            baoab(state,model,step,damping,noise,random,gaussian);
            kinetic_sum+=.5*(state.vx*state.vx+state.vy*state.vy);
            int current=basin(state);
            if(current>=0) {
                if(previous>=0&&current!=previous) {
                    ++transitions;if(current==1)++left_to_right;else++right_to_left;
                }
                previous=current;
            }
            if(previous==0)time_left+=step;if(previous==1)time_right+=step;
            if(tick%stride==0) {
                double* target=samples+6*row++;
                target[0]=state.x;target[1]=state.y;
                target[2]=model.evaluate(state.x,state.y)[0];
                target[3]=state.vx*state.vx;target[4]=state.vy*state.vy;target[5]=previous;
            }
        }
        diagnostics[0]=transitions;diagnostics[1]=left_to_right;diagnostics[2]=right_to_left;
        diagnostics[3]=time_left;diagnostics[4]=time_right;
        diagnostics[5]=burn_steps+steps+1;diagnostics[6]=kinetic_sum/steps;diagnostics[7]=row;
    });
}

}
