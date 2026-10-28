#include "zqc_graph.cuh"
#include "zqc_switch_function.h"
#include <cuda.h>
#include <nvrtc.h>
#include <fmt/printf.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>

using Params = MetaD_zqc::SwitchFunctionRequest;
using namespace MetaD_zqc;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void close(double actual, long double expected, const char* message, double tol=2e-8) {
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(static_cast<long double>(actual)-expected)>tol*(1+std::abs(expected)))
        throw std::runtime_error(fmt::sprintf("%s: got %.17g expected %.17g",message,actual,static_cast<double>(expected)));
}
void driver(CUresult status) {
    if(status!=CUDA_SUCCESS) { const char* msg=nullptr; cuGetErrorString(status,&msg); throw std::runtime_error(msg?msg:"driver error"); }
}
void rtc(nvrtcResult status) { if(status!=NVRTC_SUCCESS) throw std::runtime_error(nvrtcGetErrorString(status)); }
struct Module {
    CUmodule module=nullptr; CUfunction forward, backward;
    explicit Module(const std::string& source) {
        nvrtcProgram prog=nullptr; rtc(nvrtcCreateProgram(&prog,source.c_str(),"switch_test.cu",0,nullptr,nullptr));
        try {
            cudaDeviceProp prop{}; graph_check_cuda(cudaGetDeviceProperties(&prop,0));
            const std::string arch="--gpu-architecture=sm_"+std::to_string(prop.major)+std::to_string(prop.minor);
            const char* opts[]={arch.c_str(),"--std=c++17"};
            auto status=nvrtcCompileProgram(prog,2,opts);
            if(status!=NVRTC_SUCCESS) {
                size_t size=0; nvrtcGetProgramLogSize(prog,&size); std::string log(size,'\0'); nvrtcGetProgramLog(prog,log.data());
                throw std::runtime_error(log);
            }
            size_t size=0;rtc(nvrtcGetCUBINSize(prog,&size));std::vector<char> cubin(size);rtc(nvrtcGetCUBIN(prog,cubin.data()));
            driver(cuModuleLoadData(&module,cubin.data()));
            driver(cuModuleGetFunction(&forward,module,"cv_forward"));driver(cuModuleGetFunction(&backward,module,"cv_backward"));
            rtc(nvrtcDestroyProgram(&prog));
        } catch(...) { if(prog)nvrtcDestroyProgram(&prog);if(module)cuModuleUnload(module);throw; }
    }
    ~Module(){ if(module)cuModuleUnload(module); }
};
void launch(CUfunction fn,int n,void** args) {
    driver(cuLaunchKernel(fn,(n+255)/256,1,1,256,1,1,0,nullptr,args,nullptr));driver(cuCtxSynchronize());
}
std::vector<double> read(const double* p,int n) {
    std::vector<double> values(n);graph_check_cuda(cudaMemcpy(values.data(),p,n*sizeof(double),cudaMemcpyDeviceToHost));return values;
}
void put(double* p,const std::vector<double>& x) {
    graph_check_cuda(cudaMemcpy(p,x.data(),x.size()*sizeof(double),cudaMemcpyHostToDevice));
}
// Independent long-double value oracle; derivatives use central differences.
long double reference(const Params& p,long double x) {
    switch(p.type) {
        case LINE:return 1;
        case STEP:return x<p.r_0?1:0;
        case FERMI:return 1/(1+std::exp(-static_cast<long double>(p.alpha)*(x-p.r_0)));
        case TANH_TYPE:return (1+std::tanh(static_cast<long double>(p.alpha)*(x-p.r_0)))/2;
        case RATIONAL: {
            const long double z=(x-p.d_0)/p.r_0;
            // 偶数幂先约去 1-z^2，使 z=-1 附近的有限差分也使用正项求和。
            // 这对 long double 与 double 等精度的平台尤其重要。
            const bool even_powers = p.n%2 == 0 && p.m%2 == 0;
            const long double ratio = even_powers ? z*z : z;
            const int numerator_terms = even_powers ? p.n/2 : p.n;
            const int denominator_terms = even_powers ? p.m/2 : p.m;
            long double numerator = 0, denominator = 0, term = 1;
            for (int power = 0; power < std::max(numerator_terms, denominator_terms); ++power) {
                if (power < numerator_terms) numerator += term;
                if (power < denominator_terms) denominator += term;
                term *= ratio;
            }
            return numerator/denominator;
        }
    }
    throw std::runtime_error("invalid oracle type");
}
long double derivative(const Params& p,long double x) {
    if(p.type==STEP || p.type==LINE)return 0;
    const long double h=1e-5L*std::max(1.0L,std::abs(x));
    // Fourth-order central difference to keep the check independent of df().
    return (-reference(p,x+2*h)+8*reference(p,x+h)-8*reference(p,x-h)+reference(p,x-2*h))/(12*h);
}
__global__ void legacy_device_eval(const double* x, double* f, double* df, Params p, int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) { f[i]=SwitchFunction::f(p,x[i]); df[i]=SwitchFunction::df(p,x[i]); }
}
bool matches_kind(const Node* node, SwitchType type) {
    switch(type) {
        case LINE:return dynamic_cast<const ConstantOne*>(node)!=nullptr;
        case STEP:return dynamic_cast<const StepSwitch*>(node)!=nullptr;
        case FERMI:return dynamic_cast<const Fermi*>(node)!=nullptr;
        case TANH_TYPE:return dynamic_cast<const TanhSwitch*>(node)!=nullptr;
        case RATIONAL:return dynamic_cast<const RationalSwitch*>(node)!=nullptr;
    }
    return false;
}

void register_test_graph_nodes(Graph& graph);

std::string direct_switch_call(const Params& p) {
    switch (p.type) {
        case LINE: return "constant_one(cv)";
        case STEP: return fmt::sprintf("step(cv, %.18g)", p.r_0);
        case FERMI: return fmt::sprintf("fermi(cv, %.18g, %.18g)", p.r_0, p.alpha);
        case TANH_TYPE: return fmt::sprintf("tanh(cv, %.18g, %.18g)", p.r_0, p.alpha);
        case RATIONAL: return fmt::sprintf("rational(cv, %.18g, %d, %d, %.18g)", p.r_0, p.n, p.m, p.d_0);
    }
    throw std::runtime_error("unknown direct switching function");
}

void check_syntax_tree() {
    const auto call = GraphExprParser("rational(cv, 1.25, 6, 12)").parse();
    require(call.kind == GraphExpr::Kind::Call && call.name == "rational" && call.args.size() == 4,
            "syntax parser needs no graph or GPU allocation");
    require(call.args[0].kind == GraphExpr::Kind::Symbol && call.args[0].name == "cv",
            "symbol argument was not retained");
    close(call.number_arg(1), 1.25, "numeric syntax argument");
    const auto precedence = GraphExprParser("a + b * -c").parse();
    require(precedence.name == "+" && precedence.args[1].name == "*" &&
            precedence.args[1].args[1].name == "neg", "operator precedence changed");
}

void check_case(const char* name,const Params& p) {
    const std::array<double,16> basis={-2,-1,-1+1e-8,-.5,0,.25,.9,1-1e-8,1,1+1e-8,1.0002,1.25,1.6,2,2.5,3};
    int n=257;
    std::vector<double> x(n),s(n),t(n);
    for(int i=0;i<n;++i) {
        x[i]=p.type==RATIONAL?p.d_0+p.r_0*basis[i%basis.size()]:basis[i%basis.size()];
        s[i]=.4+.01*(i%7);t[i]=-.2-.005*(i%5);
    }
    if(p.type==FERMI) {x[0]=-1000;x[1]=1000;}
    Leaf a(x);
    Graph eager_graph(Graph::Symbols{{"cv",&a}});
    Node* eager=eager_graph.make_switch_node(&a,p);
    require(matches_kind(eager,p.type),"factory produced wrong concrete node");
    eager->forward();
    auto eager_value=read(eager->val,n);
    GpuBuffer<double> legacy_f(n),legacy_df(n);
    legacy_device_eval<<<(n+255)/256,256>>>(a.val,legacy_f.ptr,legacy_df.ptr,p,n);sync_kernel();
    auto legacy_values=read(legacy_f.ptr,n),legacy_derivatives=read(legacy_df.ptr,n);
    put(eager->adj,s);put(a.adj,std::vector<double>(n,.75));eager->backward();
    auto eager_gradient=read(a.adj,n);
    Graph graph(Graph::Symbols{{"cv",&a}},Graph::Switches{{"gate",p}});
    graph.parse_expr("sw_func(cv, gate)");graph.parse_expr("cv * " + direct_switch_call(p));
    require(matches_kind(graph.outputs[0],p.type),"parser produced wrong concrete node");
    // Nodes own parameter snapshots; a later registry edit must not alter them.
    graph.switches["gate"]={LINE,0,0,0,0,0};graph.build();
    const std::string kernels=graph.emit_forward()+graph.emit_backward();
    require(kernels.find("zqc_graph_sw_func")==std::string::npos &&
            kernels.find("SwitchFunctionRequest")==std::string::npos &&
            kernels.find("SwitchFunctionMath")==std::string::npos,"kernel still uses generic dispatch");
    Module compiled(graph.kernel_source);
    GpuBuffer<double> y0(n+3),y1(n+3),gradient(n+3),ds(n),dt(n);
    const double guard=98765.0;
    put(y0.ptr,std::vector<double>(n+3,guard));put(y1.ptr,std::vector<double>(n+3,guard));put(gradient.ptr,std::vector<double>(n+3,guard));
    put(ds.ptr,s);put(dt.ptr,t);
    void* fargs[]={&n,&a.val,&y0.ptr,&y1.ptr};launch(compiled.forward,n,fargs);
    void* bargs[]={&n,&a.val,&ds.ptr,&dt.ptr,&gradient.ptr};launch(compiled.backward,n,bargs);
    const auto ys=read(y0.ptr,n+3),weighted=read(y1.ptr,n+3),grad=read(gradient.ptr,n+3);
    SwitchFunction legacy(p.type,p.r_0,p.d_0,p.alpha,p.n,p.m);
    for(int i=0;i<n;++i) {
        const long double f=reference(p,x[i]),df=derivative(p,x[i]);
        close(eager_value[i],f,"eager value");close(ys[i],f,"NVRTC value");
        close(weighted[i],x[i]*f,"weighted value");
        close(eager_gradient[i],.75+s[i]*df,"eager gradient accumulation");
        close(grad[i],s[i]*df+t[i]*(f+x[i]*df),"graph shared-input gradient");
        close(legacy_values[i],f,"legacy device f");close(legacy_derivatives[i],df,"legacy device df");
        close(legacy.f(x[i]),f,"legacy host f");close(SwitchFunction::df(p,x[i]),df,"legacy static df");
    }
    for(int i=n;i<n+3;++i)require(ys[i]==guard && weighted[i]==guard && grad[i]==guard,"out-of-range write");
    // Exact removable-singularity derivative has a known closed form.
    if(p.type==RATIONAL) {
        close(SwitchFunction::f(p,p.d_0+p.r_0),static_cast<long double>(p.n)/p.m,"rational limit",1e-13);
        close(SwitchFunction::df(p,p.d_0+p.r_0),static_cast<long double>(p.n)*(p.n-p.m)/(2*p.m*p.r_0),"rational derivative limit",1e-13);
    }
    std::cout<<"PASS "<<name<<" n="<<n<<" host/eager/NVRTC values, gradients, guards\n";
}
void check_parser() {
    Leaf a({.2,.4,.6});Params p{FERMI,.5,0,4,0,0};
    Graph graph(Graph::Symbols{{"cv",&a}},Graph::Switches{{"gate",p}});
    graph.parse_expr("(cv * sw_func(-cv + cv, gate)).MEAN");graph.build();Module compiled(graph.kernel_source);
    int n=3;double seed=2.5/n;
    GpuBuffer<double> output(n),gradient(n);
    void* fargs[]={&n,&a.val,&output.ptr};launch(compiled.forward,n,fargs);
    void* bargs[]={&n,&a.val,&seed,&gradient.ptr};launch(compiled.backward,n,bargs);
    const auto y=read(output.ptr,n),g=read(gradient.ptr,n);
    const long double f0=reference(p,0);
    for(int i=0;i<n;++i) {
        close(y[i],(.2L+.2L*i)*f0,"Mean input with switching");
        close(g[i],seed*f0,"Mean switching chain gradient");
    }
    const auto nodes=graph.owned_nodes.size(),outputs=graph.outputs.size();const auto source=graph.kernel_source;
    for(const char* bad:{"sw_func(cv, missing)","sw_func(cv)","sw_func(cv, gate, cv)","sw_func(cv, gate"}) {
        bool threw=false;try{graph.parse_expr(bad);}catch(const std::runtime_error&){threw=true;}
        require(threw && graph.owned_nodes.size()==nodes && graph.outputs.size()==outputs && graph.kernel_source==source,"parse rollback");
    }
    int rejected=0;
    for(const Params& bad:std::vector<Params>{{static_cast<SwitchType>(99),1,0,0,6,12},{RATIONAL,0,0,0,6,12},
         {RATIONAL,1,0,0,0,12},{RATIONAL,1,0,0,6,0},{FERMI,1,0,std::numeric_limits<double>::infinity(),0,0}}) {
        try{graph.make_switch_node(&a,bad);}catch(const std::runtime_error&){++rejected;}
    }
    require(rejected==5,"invalid params accepted");
    // The old specialized public functions remain available after extraction.
    close(SwitchFunction::f_fermi(p,.2),reference(p,.2),"legacy specialized API");
    std::cout<<"PASS parser, nested expression/Mean values and gradients, snapshots, invalid parameters\n";
}
void check_direct_nodes_and_radial() {
    Leaf a({.2,.4,.6});
    Fermi gate(&a,.5,4.0);
    Mul weighted(&a,&gate);
    Graph graph(std::vector<Node*>{&gate,&weighted});graph.build();
    Module compiled(graph.kernel_source);
    int n=3;
    GpuBuffer<double> y0(n),y1(n),g(n),s(n),t(n);
    put(s.ptr,{.4,.5,.6});put(t.ptr,{-.1,-.2,-.3});
    void* fargs[]={&n,&a.val,&y0.ptr,&y1.ptr};launch(compiled.forward,n,fargs);
    void* bargs[]={&n,&a.val,&s.ptr,&t.ptr,&g.ptr};launch(compiled.backward,n,bargs);
    const auto values=read(y0.ptr,n),products=read(y1.ptr,n),grads=read(g.ptr,n);
    const Params fp{FERMI,.5,0,4,0,0};
    for(int i=0;i<n;++i) {
        const long double x=.2L+.2L*i,f=reference(fp,x),df=derivative(fp,x);
        close(values[i],f,"direct Fermi output");close(products[i],x*f,"direct shared output");
        close(grads[i],(.4L+.1L*i)*df+(-.1L-.1L*i)*(f+x*df),"direct shared-node gradient");
    }
    // Radial switching remains a parameterized f(r), df/dr interface.
    const double eps=1e-5;
    for(const Params& p:std::vector<Params>{{FERMI,1.25,0,-8,0,0},{TANH_TYPE,1.25,0,-4,0,0},
            {RATIONAL,1.25,0,0,6,12},{RATIONAL,1.25,0,0,3,7}}) {
        auto support=SwitchFunction::query_radial_support(p,eps);
        require(support.has_radius(),"radial support unavailable");
        close(SwitchFunction::f(p,support.radius),eps,"radial cutoff weight",1e-12);
        for(double r:{.1,.5,1.25,2.0,4.0}) {
            close(SwitchFunction::f(p,r),reference(p,r),"radial value");
            close(SwitchFunction::df(p,r),derivative(p,r),"radial derivative");
        }
    }
    std::cout<<"PASS direct Fermi/shared node, radial weights/derivatives/cutoffs\n";
}

void check_registered_nodes() {
    const std::vector<double> x = {-.2, .4, 1.25};
    Leaf a(x);
    Graph graph(Graph::Symbols{{"cv", &a}});
    register_test_graph_nodes(graph);
    graph.parse_expr("scaled_square(cv, -2e0) + linear(cv, .5, +.25)");
    graph.parse_expr("exp(-cv) + rational(cv, 1.25, 6, 12)");
    graph.parse_expr("scaled_square(cv, 1).AVERAGE");
    graph.build();
    require(graph.outputs.back()->is_reduction(), "custom reduction capability missing");
    Module module(graph.kernel_source);
    int n = static_cast<int>(x.size());
    double reduced_seed = 2.5 / n;
    GpuBuffer<double> y0(n), y1(n), y2(n), grad(n), s0(n), s1(n);
    put(s0.ptr, std::vector<double>(n, .4));
    put(s1.ptr, std::vector<double>(n, .7));
    void* fargs[] = {&n, &a.val, &y0.ptr, &y1.ptr, &y2.ptr};
    launch(module.forward, n, fargs);
    void* bargs[] = {&n, &a.val, &s0.ptr, &s1.ptr, &reduced_seed, &grad.ptr};
    launch(module.backward, n, bargs);
    const auto first = read(y0.ptr, n);
    const auto second = read(y1.ptr, n);
    const auto third = read(y2.ptr, n);
    const auto gradients = read(grad.ptr, n);
    const Params p{RATIONAL, 1.25, 0, 0, 6, 12};
    for (int i = 0; i < n; ++i) {
        close(first[i], -2*x[i]*x[i] + .5*x[i] + .25, "registered composition value");
        close(second[i], std::exp(-x[i]) + reference(p, x[i]), "builtin nested value");
        close(third[i], x[i]*x[i], "custom reduction input");
        close(gradients[i], .4*(-4*x[i] + .5) + .7*(-std::exp(-x[i]) + derivative(p, x[i]))
                           + reduced_seed*2*x[i], "registered mixed-output gradients");
    }
    const auto count = graph.owned_nodes.size();
    const auto outputs = graph.outputs.size();
    const auto source = graph.kernel_source;
    for (const char* bad : {"unknown(cv)", "exp(cv,cv)", "scaled_square(cv)",
            "rational(cv,1,6.5,12)", "rational(cv,1,2147483648,12)",
            "rational(cv,1,-1,12)", "rational(cv,0,6,12)", "fermi(cv,1,1e999)",
            "fermi(cv,1,cv)", "exp(cv) + unknown(cv)", "exp(cv) + missing",
            "exp(cv,)", "sw_func(cv,1)", "cv.UNKNOWN", "cv + 1", "exp(cv"}) {
        bool threw = false;
        try { graph.parse_expr(bad); } catch (const std::runtime_error&) { threw = true; }
        require(threw && graph.owned_nodes.size() == count && graph.outputs.size() == outputs
                && graph.kernel_source == source, "registered parse rollback changed the graph");
    }
    bool duplicate = false;
    try { register_test_graph_nodes(graph); } catch (const std::runtime_error&) { duplicate = true; }
    require(duplicate, "duplicate registration accepted");
    Graph isolated(Graph::Symbols{{"cv", &a}});
    bool unknown = false;
    try { isolated.parse_expr("scaled_square(cv,2)"); } catch (const std::runtime_error&) { unknown = true; }
    require(unknown, "registration leaked to another graph");
    Leaf single(std::vector<double>{0.4});
    Graph invalid(Graph::Symbols{{"cv", &single}});
    register_test_graph_nodes(invalid);
    bool rejected = false;
    try { invalid.parse_expr("external_average(cv) * cv"); invalid.build(); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "implicit Global-to-PerAtom broadcasting accepted");
    std::cout << "PASS separate-TU registration, direct calls, custom reduction, parser rollback\n";
}

int main() {
    try {
        check_syntax_tree();
        graph_check_cuda(cudaSetDevice(0));graph_check_cuda(cudaFree(nullptr));
        const std::vector<std::pair<const char*,Params>> cases={
            {"LINE",{LINE,1,0,0,0,0}}, {"STEP",{STEP,1,0,0,0,0}},
            {"FERMI up",{FERMI,.5,0,8,0,0}}, {"FERMI down",{FERMI,.5,0,-8,0,0}},
            {"FERMI flat",{FERMI,.5,0,0,0,0}}, {"TANH up",{TANH_TYPE,.5,0,4,0,0}},
            {"TANH down",{TANH_TYPE,.5,0,-4,0,0}},
            {"RATIONAL 6/12",{RATIONAL,1.25,.25,0,6,12}},
            {"RATIONAL 4/8",{RATIONAL,1.25,.25,0,4,8}},
            {"RATIONAL 3/7",{RATIONAL,1.25,.25,0,3,7}},
            {"RATIONAL 4/10",{RATIONAL,1.25,.25,0,4,10}},
            {"RATIONAL 12/6",{RATIONAL,1.25,.25,0,12,6}},
            {"RATIONAL 5/5",{RATIONAL,1.25,.25,0,5,5}}
        };
        for(const auto& item:cases)check_case(item.first,item.second);
        check_parser();check_direct_nodes_and_radial();check_registered_nodes();std::cout<<"SW_FUNC_PASS "<<cases.size()<<" configurations\n";
    } catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
