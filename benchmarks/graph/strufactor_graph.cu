#include "CV_Stru_factor.h"
#include "CV_Weighted_Chemical_Pair_Order.h"
#include "zqc_cv_assembly.cuh"
#include "zqc_graph.cuh"
#include "mpi_sum_reducer.h"
#include <algorithm>
#include <array>
#include <utility>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>

using namespace MetaD_zqc;
using Position = std::array<double,3>;
using Positions = std::vector<Position>;
using tagint = MetaD_zqc::LocalAtomIndex;
constexpr double q = 1.7, cutoff = 2.5;
const std::vector<int> selected = {1,3,5};
// 三个目标共用独立 CPU/有限差分测试，覆盖结构因子及化学加权配位数。
#if defined(ZQC_TEST_CHEMICAL_WEIGHTS) || defined(ZQC_TEST_WEIGHTED_CHEM)
constexpr bool use_chemical_weights = true;
#else
constexpr bool use_chemical_weights = false;
#endif
#ifdef ZQC_TEST_WEIGHTED_CHEM
constexpr bool use_structure_factor = false;
#else
constexpr bool use_structure_factor = true;
#endif
const std::vector<int> atom_types = {1,2,3,1,2,3};
const std::vector<double> type_weights = {0.0,0.8,-1.3,0.0};
double pair_weight(int center, int neighbor) {
    return use_chemical_weights ? type_weights[atom_types[center]]*type_weights[atom_types[neighbor]] : 1.0;
}

void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void near(double actual, double expected, double tol, const char* message) {
    if (!std::isfinite(actual) || std::abs(actual-expected)>tol*(1+std::abs(expected)))
        throw std::runtime_error(std::string(message)+": "+std::to_string(actual)+" != "+std::to_string(expected));
}
template<class T> void upload(GpuBuffer<T>& buffer, const std::vector<T>& values) {
    if (!values.empty()) graph_check_cuda(cudaMemcpy(buffer.ptr,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice));
}
std::vector<double> download(const GpuBuffer<double>& buffer, int n) {
    std::vector<double> values(n);
    if (n) graph_check_cuda(cudaMemcpy(values.data(),buffer.ptr,n*sizeof(double),cudaMemcpyDeviceToHost));
    return values;
}

// 将全局类型按测试使用的 localid 顺序上传，验证类型与几何重排后仍对应同一原子。
struct TestKernels {
    const StruFactorData& data;
    GpuBuffer<int> d_atom_types;
    GpuBuffer<double> d_type_weights;
    TestKernels(const StruFactorData& data, const std::vector<int>& ids)
        : data(data), d_atom_types(ids.size()), d_type_weights(type_weights.size()) {
        std::vector<int> local_types;
        for (int global_id : ids) local_types.push_back(atom_types[global_id]);
        upload(d_atom_types,local_types);
        upload(d_type_weights,type_weights);
    }
    void forward(double wave_number, double cutoff_rsq, double* values) {
        if constexpr (!use_structure_factor)
            weighted_chem_forward(data,cutoff_rsq,d_atom_types.ptr,d_type_weights.ptr,values);
        else if constexpr (use_chemical_weights)
            strufactor_chem_forward(data,wave_number,cutoff_rsq,d_atom_types.ptr,d_type_weights.ptr,values);
        else strufactor_forward(data,wave_number,cutoff_rsq,values);
    }
    void backward(double wave_number, double cutoff_rsq, const double* seeds, double* forces, double* virials) {
        if constexpr (!use_structure_factor)
            weighted_chem_backward(data,cutoff_rsq,d_atom_types.ptr,d_type_weights.ptr,seeds,forces,virials);
        else if constexpr (use_chemical_weights)
            strufactor_chem_backward(data,wave_number,cutoff_rsq,d_atom_types.ptr,d_type_weights.ptr,seeds,forces,virials);
        else strufactor_backward(data,wave_number,cutoff_rsq,seeds,forces,virials);
    }
};

// 独立 CPU 能量公式；有限差分不使用生产 kernel 的导数或径向实现。
std::vector<double> reference_values(const Positions& x) {
    std::vector<double> values(x.size(),0.0);
    for (int i:selected) {
        values[i]=1;
        for (int j=0;j<static_cast<int>(x.size());++j) {
            if (j==i) continue;
            double r2=0;
            for (int c=0;c<3;++c) r2+=(x[j][c]-x[i][c])*(x[j][c]-x[i][c]);
            if (r2>cutoff*cutoff) continue;
            const double theta=q*std::sqrt(r2);
            double s=1;
            if (r2>.8*cutoff*cutoff) {
                const double u=(r2-.8*cutoff*cutoff)/(.2*cutoff*cutoff);
                s=1-u*u*u;
            }
            values[i]+=pair_weight(i,j)*(use_structure_factor ? std::sin(theta)/theta : 1.0)*s;
        }
    }
    return values;
}
double weighted_energy(const Positions& x) {
    const auto v=reference_values(x);
    return .7*v[1]-.45*v[3]+1.2*v[5];
}
double graph_energy(const Positions& x) {
    const auto v=reference_values(x);
    double avg=0, avg2=0;
    for (int i:selected) { avg+=v[i]/3; avg2+=v[i]*v[i]/3; }
    return .4*avg2-.2*std::exp(avg);
}

void verify_derivatives(const Positions& x, const std::vector<double>& force,
                        const std::vector<double>& virial,
                        const std::function<double(const Positions&)>& energy) {
    const double h=2e-6;
    double max_error=0;
    for (int i=0;i<static_cast<int>(x.size());++i) {
        for (int c=0;c<3;++c) {
            auto plus=x, minus=x;
            plus[i][c]+=h; minus[i][c]-=h;
            const double ref=-(energy(plus)-energy(minus))/(2*h);
            max_error=std::max(max_error,std::abs(force[3*i+c]-ref));
            near(force[3*i+c],ref,3e-7,"coordinate finite difference");
        }
    }
    const int rows[6]={0,1,2,1,2,2}, cols[6]={0,1,2,0,0,1};
    for (int c=0;c<6;++c) {
        auto plus=x, minus=x;
        for (int i=0;i<static_cast<int>(x.size());++i) {
            plus[i][rows[c]]+=h*x[i][cols[c]];
            minus[i][rows[c]]-=h*x[i][cols[c]];
        }
        const double ref=-(energy(plus)-energy(minus))/(2*h);
        near(virial[c],ref,5e-7,"strain finite difference / virial");
    }
    for (int c=0;c<3;++c) {
        double total=0;
        for (int i=0;i<static_cast<int>(x.size());++i) total+=force[3*i+c];
        near(total,0,3e-12,"net force");
    }
    require(std::abs(force[0])+std::abs(force[1])+std::abs(force[2])>1e-8,
            "neighbor outside center group received no force");
    int rank;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if (!rank) std::cout<<"max force finite-difference error: "<<max_error<<'\n';
}

void distributed_test() {
    int rank,size;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);
    // WEIGHT_CHEM 在径向平坦区的力为零，因此安排中心-中心及中心-组外邻居落在过渡区。
    const Positions x = use_structure_factor
        ? Positions{{0,0,0},{.8,.1,.2},{1.5,.3,-.1},{2.35,-.2,.4},{3.6,.2,.1},{8,0,0}}
        : Positions{{-1.5,0,0},{.8,.1,.2},{1.5,.3,-.1},{3.1,-.2,.4},{5.4,.2,.1},{8,0,0}};
    const int natoms=x.size();
    auto owner=[&](int id){return size==1?0:1+id%(size-1);}; // 多 rank 时 rank0 故意没有 owned。
    std::vector<int> ids;
    for (int id=natoms-1;id>=0;--id) if (owner(id)==rank) ids.push_back(id);
    const int nlocal=ids.size();
    for (int id=0;id<natoms;++id) if (owner(id)!=rank) ids.push_back(id);
    std::vector<int> local_of(natoms);
    std::vector<int> center_ids;
    for (int i=0;i<natoms;++i) {
        local_of[ids[i]]=i;
        if (std::find(selected.begin(),selected.end(),ids[i])!=selected.end())
            center_ids.push_back(i);
    }
    std::sort(center_ids.begin(), center_ids.end());
    int owned_centers=0;
    for (int local:center_ids) if (local<nlocal) ++owned_centers;
    SelectionView centers_view{1, owned_centers,
        {center_ids.data(), center_ids.size(), 1, 1, 1, center_ids.size(), MemorySpace::Host}};
    CVAtomLayout atoms;
    atoms.rebuild(nlocal,natoms-nlocal,{centers_view});
    auto assembly=atoms.group(1);
    require(assembly->group_count()==atoms.leaf_count(),"single-group owned extent");
    require(assembly->total_count()==3,"group ghosts were lost");
    const int nc=assembly->group_count();
    std::vector<tagint> centers, offsets, neighbors;
    std::vector<int> counts;
    std::vector<double> geometry;
    for (int k=0;k<nc;++k) {
        const int li=assembly->local_ids()[k], gi=ids[li];
        centers.push_back(li);offsets.push_back(neighbors.size());counts.push_back(0);
        for (int gj=0;gj<natoms;++gj) {
            if (gi==gj) continue;
            double dr[3],r2=0;
            for (int c=0;c<3;++c) {dr[c]=x[gj][c]-x[gi][c];r2+=dr[c]*dr[c];}
            if (r2>cutoff*cutoff) continue;
            neighbors.push_back(local_of[gj]);
            geometry.insert(geometry.end(),{dr[0],dr[1],dr[2],r2});
            ++counts.back();
        }
        // 候选段尾保留空槽；有效计数必须阻止 kernel 读取其中的 NaN 几何。
        neighbors.push_back(0);
        geometry.insert(geometry.end(),4,std::numeric_limits<double>::quiet_NaN());
    }
    StruFactorData data(nc,neighbors.size());
    GpuBuffer<double> raw(natoms+2), values(nc+2), grad(nc+2), seeds(nc);
    offsets.push_back(neighbors.size());
    upload(data.d_group_indices,centers);upload(data.d_group_numneigh,offsets);
    upload(data.d_calculated_numneigh,neighbors);upload(data.d_neigh_in_cutoff_r,counts);
    upload(data.d_group_dminneigh,geometry);
    std::vector<double> initial(natoms+2,0);initial[natoms]=123;initial[natoms+1]=456;upload(raw,initial);
    TestKernels kernels(data,ids);
    kernels.forward(q,cutoff*cutoff,raw.ptr);
    const auto result=download(raw,natoms+2), reference=reference_values(x);
    for (int i=0;i<natoms;++i) {
        const bool selected_center = std::binary_search(center_ids.begin(), center_ids.end(), i);
        near(result[i],i<nlocal && selected_center?reference[ids[i]]:0,3e-13,"raw CV forward");
    }
    require(result[natoms]==123 && result[natoms+1]==456,"raw value guard modified");
    assembly->assemble_values(raw,values);

    auto evaluate_force=[&](const std::vector<double>& center_seed) {
        upload(seeds,center_seed);
        GpuBuffer<double> df(3*natoms+2),dv(6*natoms+2);
        std::vector<double> f0(3*natoms+2,0),v0(6*natoms+2,0);
        f0[3*natoms]=123;f0[3*natoms+1]=456;v0[6*natoms]=123;v0[6*natoms+1]=456;
        upload(df,f0);upload(dv,v0);
        kernels.backward(q,cutoff*cutoff,seeds.ptr,df.ptr,dv.ptr);
        const auto local_f=download(df,3*natoms+2),local_v=download(dv,6*natoms+2);
        require(local_f[3*natoms]==123 && local_f[3*natoms+1]==456,"force guard modified");
        require(local_v[6*natoms]==123 && local_v[6*natoms+1]==456,"virial guard modified");
        std::vector<double> contribution(3*natoms,0), force(3*natoms,0),virial(6,0),global_virial(6,0);
        for (int li=0;li<natoms;++li) {
            for (int c=0;c<3;++c) contribution[3*ids[li]+c]+=local_f[3*li+c];
            for (int c=0;c<6;++c) virial[c]+=local_v[6*li+c];
        }
        // 基准按全局编号汇总 ghost 贡献，验证分区数学；不是 LAMMPS reverse_comm 集成测试。
        MPI_Allreduce(contribution.data(),force.data(),3*natoms,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        MPI_Allreduce(virial.data(),global_virial.data(),6,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        return std::make_pair(force,global_virial);
    };
    std::vector<double> weights(nc);
    for (int k=0;k<nc;++k) {
        const int id=ids[centers[k]];
        weights[k]=id==1?.7:id==3?-.45:1.2;
    }
    auto weighted=evaluate_force(weights);
    verify_derivatives(x,weighted.first,weighted.second,weighted_energy);

    Leaf leaf(values,grad,nc);
    Graph graph(Graph::Symbols{{"s",&leaf}}, {}, mpi_sum_reducer(MPI_COMM_WORLD));
    graph.parse_expr("AVE(s*s)");graph.parse_expr("exp(AVE(s))");
    GraphExecution run(graph);
    const auto outputs=run.forward();
    double mean=0,mean2=0;
    for (int i:selected){mean+=reference[i]/3;mean2+=reference[i]*reference[i]/3;}
    near(outputs[0],mean2,3e-13,"graph AVE(s*s)");near(outputs[1],std::exp(mean),3e-13,"graph exp(AVE(s))");
    run.backward({.4,-.2});assembly->gather_adjoint(grad,seeds);
    const auto graph_seeds=download(seeds,nc);
    for(int k=0;k<nc;++k)
        near(graph_seeds[k],(.8*reference[ids[centers[k]]]-.2*std::exp(mean))/3,3e-13,"graph center seed");
    auto composite=evaluate_force(graph_seeds);
    verify_derivatives(x,composite.first,composite.second,graph_energy);
    auto zero=evaluate_force(std::vector<double>(nc,0));
    for(double f:zero.first) near(f,0,0,"zero seed force");
    for(double v:zero.second) near(v,0,0,"zero seed virial");
    if constexpr (use_chemical_weights && use_structure_factor) {
        // 全部权重为 1 时应退化为普通 Stru_factor，包括力和逐原子 virial。
        upload(kernels.d_type_weights,std::vector<double>(type_weights.size(),1.0));
        GpuBuffer<double> plain_values(natoms),chemical_values(natoms),plain_force(3*natoms),chemical_force(3*natoms);
        GpuBuffer<double> plain_virial(6*natoms),chemical_virial(6*natoms);
        upload(seeds,weights);
        strufactor_forward(data,q,cutoff*cutoff,plain_values.ptr);
        kernels.forward(q,cutoff*cutoff,chemical_values.ptr);
        strufactor_backward(data,q,cutoff*cutoff,seeds.ptr,plain_force.ptr,plain_virial.ptr);
        kernels.backward(q,cutoff*cutoff,seeds.ptr,chemical_force.ptr,chemical_virial.ptr);
        const auto expected_values=download(plain_values,natoms),actual_values=download(chemical_values,natoms);
        const auto expected_force=download(plain_force,3*natoms),actual_force=download(chemical_force,3*natoms);
        const auto expected_virial=download(plain_virial,6*natoms),actual_virial=download(chemical_virial,6*natoms);
        for (int i=0;i<natoms;++i) near(actual_values[i],expected_values[i],1e-14,"unit weight values");
        for (int i=0;i<3*natoms;++i) near(actual_force[i],expected_force[i],1e-14,"unit weight forces");
        for (int i=0;i<6*natoms;++i) near(actual_virial[i],expected_virial[i],1e-14,"unit weight virials");
    }
}

void small_radius_test() {
    // 不依赖有限差分精度的小 qr 极限；中心编号为 1，邻居为 0。
    StruFactorData data(1,1);
    GpuBuffer<double> values(2),seed(1),force(6);
    upload(data.d_group_indices,std::vector<tagint>{1});
    upload(data.d_group_numneigh,std::vector<tagint>{0,1});
    upload(data.d_calculated_numneigh,std::vector<tagint>{0});
    upload(data.d_neigh_in_cutoff_r,std::vector<int>{1});
    upload(data.d_group_dminneigh,std::vector<double>{-1e-7,0,0,1e-14});
    upload(seed,std::vector<double>{.7});
    TestKernels kernels(data,{0,1});
    kernels.forward(2,9,values.ptr);
    kernels.backward(2,9,seed.ptr,force.ptr,nullptr);
    const double expected_value = 1+pair_weight(1,0)*(use_structure_factor ? 1-4e-14/6 : 1);
    near(download(values,2)[1],expected_value,1e-15,"small qr / flat window value");
    const auto f=download(force,6);
    near(f[3],use_structure_factor ? pair_weight(1,0)*.7*4e-7/3 : 0,1e-20,"small qr / flat window force");
    near(f[0],-f[3],1e-20,"small qr action/reaction");
}

// 单个有向项验证逐原子 virial 的分摊和 += 语义，随后检查平坦区及 cutoff 外的零导数。
void weighted_virial_layout_test() {
    if constexpr (use_structure_factor) return;
    StruFactorData data(1,1);
    upload(data.d_group_indices,std::vector<tagint>{1});
    upload(data.d_group_numneigh,std::vector<tagint>{0,1});
    upload(data.d_calculated_numneigh,std::vector<tagint>{0});
    upload(data.d_neigh_in_cutoff_r,std::vector<int>{1});
    upload(data.d_group_dminneigh,std::vector<double>{1.2,1.6,1.2,5.44});
    TestKernels kernels(data,{0,1});
    GpuBuffer<double> values(2),seeds(1),forces(6),virials(12);
    upload(seeds,std::vector<double>{.7});
    upload(forces,std::vector<double>(6,2.0));
    upload(virials,std::vector<double>(12,3.0));
    kernels.backward(0,6.25,seeds.ptr,forces.ptr,virials.ptr);
    const auto actual_forces=download(forces,6),actual_virials=download(virials,12);
    const double displacement[3]={1.2,1.6,1.2};
    const int rows[6]={0,1,2,0,0,1},cols[6]={0,1,2,1,2,2};
    require(std::abs(actual_forces[3]-2.0)>1e-6,"virial fixture has zero force");
    for (int component=0;component<3;++component)
        near(actual_forces[component]+actual_forces[3+component],4.0,1e-14,"force accumulation/action-reaction");
    for (int component=0;component<6;++component) {
        const double half_virial=-0.5*displacement[rows[component]]*(actual_forces[3+cols[component]]-2.0);
        near(actual_virials[component],3.0+half_virial,1e-14,"neighbor half virial");
        near(actual_virials[6+component],3.0+half_virial,1e-14,"center half virial");
    }
    for (double radius : {1.0,3.0}) {
        upload(data.d_group_dminneigh,std::vector<double>{radius,0,0,radius*radius});
        upload(forces,std::vector<double>(6,0.0));
        upload(virials,std::vector<double>(12,0.0));
        kernels.forward(0,6.25,values.ptr);
        kernels.backward(0,6.25,seeds.ptr,forces.ptr,virials.ptr);
        near(download(values,2)[1],radius<2.5 ? 1+pair_weight(1,0) : 1,1e-15,"flat/outside window value");
        for (double value : download(forces,6)) near(value,0.0,0.0,"flat/outside window force");
        for (double value : download(virials,12)) near(value,0.0,0.0,"flat/outside window virial");
    }
    StruFactorData empty(0,0);
    weighted_chem_forward(empty,6.25,nullptr,nullptr,nullptr);
    weighted_chem_backward(empty,6.25,nullptr,nullptr,nullptr,nullptr,nullptr);
}

int main(int argc,char** argv) {
    MPI_Init(&argc,&argv);
    int rank;MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    try {
        distributed_test();small_radius_test();weighted_virial_layout_test();
        graph_check_cuda(cudaDeviceSynchronize());
        if (!rank) std::cout<<(!use_structure_factor ? "WEIGHT_CHEM" : use_chemical_weights ? "Stru_factor_chem" : "Stru_factor")<<" raw values, graph gradients, forces and virial passed\n";
    } catch(const std::exception& e) {
        std::cerr<<"rank "<<rank<<": "<<e.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);
    }
    MPI_Finalize();return 0;
}
