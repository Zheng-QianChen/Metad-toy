#include "zqc_graph.cuh"
#include <fstream>
int main(int argc, char** argv) {
    if(argc!=2) return 2;
    const std::string dir=argv[1];
    GpuBuffer<double> av(1),bv(1),cv(1),dv(1);
    Leaf a(av,1),b(bv,1),c(cv,1),d(dv,1);
    Graph first(Graph::Symbols{{"a",&a},{"b",&b},{"c",&c},{"d",&d}});
    first.parse_expr("(a+b*c-exp(d)).MEAN"); first.build();
    std::ofstream(dir+"/mean_exp.graph.cu")<<first.kernel_source;
    Graph second(Graph::Symbols{{"a",&a},{"b",&b},{"c",&c}});
    second.parse_expr("a*b");second.parse_expr("a*c");second.build();
    std::ofstream(dir+"/two_dims.graph.cu")<<second.kernel_source;
}
