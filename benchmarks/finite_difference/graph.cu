// 对实际 GraphExecution 前向做双边差分，逐项核对 Leaf Jacobian 和多输出 VJP。
#include "zqc_graph.cuh"
#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void upload(double* target, const std::vector<double>& values) {
    if (!values.empty()) graph_check_cuda(cudaMemcpy(target, values.data(), values.size()*sizeof(double), cudaMemcpyHostToDevice));
}
std::vector<double> download(const double* source, int count) {
    std::vector<double> values(count);
    if (count) graph_check_cuda(cudaMemcpy(values.data(), source, count*sizeof(double), cudaMemcpyDeviceToHost));
    return values;
}

// setter/getter 仅负责输入布局；所有导数均来自生产 GraphExecution::backward。
void finite_difference(const char* name, GraphExecution& execution,
                       const std::vector<double>& initial,
                       const std::function<void(const std::vector<double>&)>& set_inputs,
                       const std::function<std::vector<double>()>& gradients,
                       MPI_Comm communicator) {
    int rank=0, ranks=0;
    MPI_Comm_rank(communicator, &rank); MPI_Comm_size(communicator, &ranks);
    set_inputs(initial);
    const auto baseline=execution.forward();
    const int outputs=baseline.size(), variables=initial.size();
    std::vector<std::vector<double>> analytic;
    for (int output=0; output<outputs; ++output) {
        std::vector<double> seeds(outputs, 0.0); seeds[output]=1.0;
        execution.backward(seeds); analytic.push_back(gradients());
    }
    std::vector<double> mixed_seed(outputs);
    for (int output=0; output<outputs; ++output)
        mixed_seed[output]=(output%2 ? -1.0 : 1.0)*(.2+.03*output);
    execution.backward(mixed_seed); analytic.push_back(gradients());
    execution.backward(std::vector<double>(outputs,0.0));
    for (double gradient:gradients()) require(gradient==0.0, "zero seed retained previous adjoints");
    execution.backward(mixed_seed);
    const auto repeated=gradients();
    for (int variable=0; variable<variables; ++variable)
        require(std::abs(repeated[variable]-analytic.back()[variable])<1e-13,
                "repeated backward accumulated old gradients");

    for (double step:{1e-3,1e-4,1e-5}) {
        std::vector<double> errors(outputs+1,0.0), scores(outputs+1,0.0);
        int worst_variable=0, worst_output=0;
        double worst_score=0.0;
        for (int variable=0; variable<variables; ++variable) {
            auto plus=initial, minus=initial;
            plus[variable]+=step; minus[variable]-=step;
            set_inputs(plus); const auto positive=execution.forward();
            set_inputs(minus); const auto negative=execution.forward();
            std::vector<double> numerical(outputs+1,0.0);
            for (int output=0; output<outputs; ++output) {
                numerical[output]=(positive[output]-negative[output])/(2*step);
                numerical.back()+=mixed_seed[output]*numerical[output];
            }
            for (int output=0; output<=outputs; ++output) {
                require(std::isfinite(numerical[output]) && std::isfinite(analytic[output][variable]), "nonfinite derivative");
                const double error=std::abs(analytic[output][variable]-numerical[output]);
                const double scale=3e-8+2e-6*std::max(std::abs(analytic[output][variable]),std::abs(numerical[output]));
                errors[output]=std::max(errors[output],error);
                scores[output]=std::max(scores[output],error/scale);
                if (error/scale>worst_score) { worst_score=error/scale;worst_variable=variable;worst_output=output; }
            }
        }
        if (rank==0) {
            std::cout<<std::setprecision(17)<<"GRAPH_FD_RESULT {\"case\":\""<<name
                <<"\",\"ranks\":"<<ranks<<",\"variables\":"<<variables<<",\"outputs\":"<<outputs
                <<",\"step\":"<<step<<",\"max_abs_error\":"<<*std::max_element(errors.begin(),errors.end())
                <<",\"max_scaled_error\":"<<worst_score<<",\"worst_output\":"<<worst_output
                <<",\"worst_variable\":"<<worst_variable<<",\"output_errors\":[";
            for (int output=0; output<=outputs; ++output) std::cout<<(output ? ",":"")<<errors[output];
            std::cout<<"]}"<<std::endl;
        }
        if (step<=1e-4) require(worst_score<=1.0, "Graph finite difference exceeded tolerance");
    }
    set_inputs(initial);
    const auto restored=execution.forward();
    for (int output=0; output<outputs; ++output)
        require(std::abs(restored[output]-baseline[output])<1e-13,"forward did not restore after perturbation");
}

void distributed(MPI_Comm communicator) {
    int rank=0,ranks=0;MPI_Comm_rank(communicator,&rank);MPI_Comm_size(communicator,&ranks);
    constexpr int sample_count=19;
    std::vector<int> local_ids;
    // 多 rank 时 rank 0 为空，其余 rank 使用不均匀分配。
    for (int id=0;id<sample_count;++id)
        if ((ranks==1 ? 0 : 1+(id*id)%(ranks-1))==rank) local_ids.push_back(id);
    const int local_count=local_ids.size();
    Leaf first(std::vector<double>(local_count,0.0)),second(std::vector<double>(local_count,0.0));
    Leaf coefficient({.43},{Domain::Global,1});
    Graph graph(Graph::Symbols{{"a",&first},{"b",&second},{"c",&coefficient}},
                Graph::Switches{{"gate",{MetaD_zqc::FERMI,.2,0,2.1,0,0}}},communicator);
    // 两个别名实际指向相同节点，检查共享乘积和共享规约的梯度累加。
    auto* product=graph.make_node<Mul>(&first,&second);
    graph.symbols["product"]=product;
    graph.symbols["mean_product"]=graph.make_node<AVE>(product,communicator);
    const std::vector<std::string> expressions={
        "AVE(a)","AVE(a+b)","AVE(a-b)","AVE(-a)","AVE(a*b)","AVE(a*a)",
        "AVE(linear(a,-1.7,0.3))","AVE(exp(linear(a,0.3,0)))","AVE(constant_one(a))",
        "AVE(step(a,0.37))","AVE(fermi(a,0.2,2.1))","AVE(tanh(a,0.1,-1.2))",
        "AVE(rational(a,1,6,12))","AVE(rational(a,1,3,7))","AVE(sw_func(a,gate))",
        "AVE(fermi(product,0.2,1.3))","mean_product*exp(AVE(a))*c",
        "linear(mean_product,0.6,0.2)+fermi(AVE(b),0.8,3)"};
    for (const auto& expression:expressions) graph.parse_expr(expression);
    GraphExecution execution(graph);
    std::vector<double> initial(2*sample_count+1);
    for (int id=0;id<sample_count;++id) {
        initial[id]=-.65+.13*id;
        initial[sample_count+id]=.7+.05*std::cos(.41*id);
    }
    initial[0]=-1;initial[5]=0;initial[7]=1;initial.back()=.43;
    const auto setter=[&](const std::vector<double>& values) {
        std::vector<double> a(local_count),b(local_count);
        for (int local=0;local<local_count;++local) {
            a[local]=values[local_ids[local]];b[local]=values[sample_count+local_ids[local]];
        }
        upload(first.val,a);upload(second.val,b);*coefficient.val=values.back();
    };
    const auto getter=[&]() {
        const auto a=download(first.adj,local_count),b=download(second.adj,local_count);
        std::vector<double> local(2*sample_count,0),global(2*sample_count+1,0);
        for (int index=0;index<local_count;++index) {
            local[local_ids[index]]=a[index];local[sample_count+local_ids[index]]=b[index];
        }
        MPI_Allreduce(local.data(),global.data(),2*sample_count,MPI_DOUBLE,MPI_SUM,communicator);
        // Global 输入在每个 rank 上代表同一个量，取一次梯度。
        global.back()=*coefficient.adj;
        return global;
    };
    finite_difference("nvrtc_mpi",execution,initial,setter,getter,communicator);
}

void host_only(MPI_Comm communicator) {
    Leaf a({.43},{Domain::Global,1}),b({.77},{Domain::Global,1});
    Graph graph(Graph::Symbols{{"a",&a},{"b",&b}}, {}, communicator);
    for (const char* expression:{"a+b","a-b","-a","a*b","a*a","linear(a,-1.7,0.3)",
            "exp(a)","constant_one(a)","step(a,0.2)","fermi(a,0.2,2.1)",
            "tanh(a,0.1,-1.2)","rational(a,1,6,12)","rational(a,1,3,7)","a*b+exp(-a)*b"})
        graph.parse_expr(expression);
    GraphExecution execution(graph);
    require(graph.kernel_source.empty(),"Host-only graph unexpectedly generated GPU code");
    finite_difference("host",execution,{.43,.77},[&](const std::vector<double>& values) {
        *a.val=values[0];*b.val=values[1];
    },[&]() {return std::vector<double>{*a.adj,*b.adj};},communicator);
}
}
int main(int argc,char** argv) {
    MPI_Init(&argc,&argv);
    int rank=0;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    try {
        graph_check_cuda(cudaSetDevice(0));
        host_only(MPI_COMM_WORLD);distributed(MPI_COMM_WORLD);
        if (rank==0) std::cout<<"GRAPH_FINITE_DIFFERENCE_PASS"<<std::endl;
    } catch (const std::exception& error) {
        std::cerr<<"rank "<<rank<<": "<<error.what()<<std::endl;
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    MPI_Finalize();return 0;
}
