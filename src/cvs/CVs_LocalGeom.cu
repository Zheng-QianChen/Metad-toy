#include "CV_LocalGeom.h"
#include "zqc_cv_assembly.cuh"
#include "zqc_cv_debug.h"
#include "zqc_cv_parse.h"
#include "zqc_switch_function.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace MetaD_zqc {
namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
// Consume periodic-image coordinates as supplied by the engine; never remap or
// deduplicate ghost images. One thread owns a center's scalar sums.
__global__ void forward_kernel(ArrayView<const int> centers, ArrayView<const int> offsets,
    ArrayView<const int> neighbors, ArrayView<const double> x, ArrayView<const int> types,
    int neighbor_type, double cutoff, double epsilon, SwitchFunctionRequest radial,
    bool tet, double target, double sigma, double* values, double* coordination,
    double* rnn, double* valid) {
    const int k = blockIdx.x*blockDim.x+threadIdx.x;
    if (k >= centers.item_count) return;
    const int i = centers(k);
    double n = 0.0, weighted_distance = 0.0;
    for (int p = offsets(k); p < offsets(k+1); ++p) {
        const int j = neighbors(p);
        if (neighbor_type >= 0 && types(j) != neighbor_type) continue;
        double r2 = 0.0;
        for (int c=0;c<3;++c) { const double d=x(j,c)-x(i,c); r2+=d*d; }
        if (r2 < 1e-24 || r2 > cutoff*cutoff) continue;
        const double r=sqrt(r2), weight=SwitchFunctionMath::f(radial,r);
        if (weight < epsilon) continue;
        n+=weight; weighted_distance+=r*weight;
    }
    coordination[i]=n;
    const double z=(n-target)/sigma;
    values[i]=tet ? exp(-0.5*z*z) : n;
    valid[i]=n>epsilon ? 1.0 : 0.0;
    rnn[i]=n>epsilon ? weighted_distance/n : 0.0;
}
__global__ void backward_kernel(ArrayView<const int> centers, ArrayView<const int> offsets,
    ArrayView<const int> neighbors, ArrayView<const double> x, ArrayView<const int> types,
    int neighbor_type, double cutoff, double epsilon, SwitchFunctionRequest radial,
    bool tet, double target, double sigma, const double* values, const double* coordination,
    const double* adjoints, double* force, double* virial) {
    const int k=blockIdx.x*blockDim.x+threadIdx.x;
    if (k>=centers.item_count) return;
    const int i=centers(k);
    double seed=adjoints[k];
    if (tet) seed *= -values[i]*(coordination[i]-target)/sigma/sigma;
    if (seed==0.0) return;
    for (int p=offsets(k);p<offsets(k+1);++p) {
        const int j=neighbors(p);
        if (neighbor_type>=0 && types(j)!=neighbor_type) continue;
        double d[3],r2=0;
        for(int c=0;c<3;++c) { d[c]=x(j,c)-x(i,c); r2+=d[c]*d[c]; }
        if(r2<1e-24 || r2>cutoff*cutoff) continue;
        const double r=sqrt(r2);
        if(SwitchFunctionMath::f(radial,r)<epsilon) continue;
        const double scale=seed*SwitchFunctionMath::df(radial,r)/r;
        double f[3];
        for(int c=0;c<3;++c) {
            f[c]=scale*d[c]; atomicAdd(force+3*i+c,f[c]); atomicAdd(force+3*j+c,-f[c]);
        }
        const double v[6]={-.5*d[0]*f[0],-.5*d[1]*f[1],-.5*d[2]*f[2],
                            -.5*d[0]*f[1],-.5*d[0]*f[2],-.5*d[1]*f[2]};
        for(int c=0;c<6;++c) { atomicAdd(virial+6*i+c,v[c]); atomicAdd(virial+6*j+c,v[c]); }
    }
}
}

CV* LocalGeom::create(CVContext& context,int argc,char** argv,int& i) {
    if(i+1>=argc) throw std::invalid_argument("LOCAL_GEOM requires a center group");
    const auto selection=context.engine.selection_id(argv[i+1]);
    i+=2;
    double cutoff=4.0,epsilon=1e-12,target=4.0,sigma=.5;
    int neighbor_type=-1;
    bool explicit_cutoff=false,tet=false;
    auto radial=SwitchFunction::get_default_step()->params;
    while(i<argc) {
        const std::string key=argv[i];
        if(key!="cutoff_r" && key!="cutoff_eps" && key!="TET_CENTER" && key!="TET_SIGMA" &&
           key!="NEIGH_TYPE" && key!="SW_FUNC_r" && key!="OUTPUT") break;
        if(i+1>=argc) throw std::invalid_argument("LOCAL_GEOM: missing value for "+key);
        const char* value=argv[i+1];
        if(key=="cutoff_r") { cutoff=cv_number(value); explicit_cutoff=true; }
        else if(key=="cutoff_eps") epsilon=cv_number(value);
        else if(key=="TET_CENTER") target=cv_number(value);
        else if(key=="TET_SIGMA") sigma=cv_number(value);
        else if(key=="SW_FUNC_r") radial=context.switching_function(value)->params;
        else if(key=="NEIGH_TYPE") neighbor_type=std::strcmp(value,"0")==0 ? -1 : context.engine.atom_type_id(value);
        else {
            if(std::strcmp(value,"COORD") && std::strcmp(value,"TET"))
                throw std::invalid_argument("LOCAL_GEOM OUTPUT must be COORD or TET; rnn is diagnostic only");
            tet=std::strcmp(value,"TET")==0;
        }
        i+=2;
    }
    if(!std::isfinite(epsilon) || epsilon<=0 || epsilon>=1 ||
       !std::isfinite(sigma) || sigma<=0 || !std::isfinite(target))
        throw std::invalid_argument("LOCAL_GEOM: require 0<cutoff_eps<1, finite TET_CENTER, TET_SIGMA>0");
    if(!explicit_cutoff && radial.type!=LINE) cutoff=SwitchFunction::invert_for_eps(radial,epsilon);
    if(!std::isfinite(cutoff*cutoff) || cutoff<=0)
        throw std::invalid_argument("LOCAL_GEOM: cutoff_r must be positive and finite");
    return new LocalGeom(context,selection,cutoff,epsilon,neighbor_type,radial,tet,target,sigma);
}
LocalGeom::LocalGeom(CVContext& context,SelectionId selection,double cutoff,double epsilon,
    int neighbor_type,SwitchFunctionRequest radial,bool tet,double target,double sigma)
    : CV(context),selection_(selection),neighbor_type_(neighbor_type),cutoff_(cutoff),epsilon_(epsilon),
      center_(target),sigma_(sigma),tet_(tet),radial_(radial) {
    NeighborRequest request;
    request.cutoff=cutoff_; request.centers=selection_; request.host_csr=false;
    neighbor_handle_=engine.neighbors().request(request);
}
CV::CV_Calculation LocalGeom::set_CV_calculate(std::string) {
    throw std::runtime_error("LOCAL_GEOM requires EXECUTOR GRAPH; use RAW with SUM or AVE");
}
CV::CV_BiasForce LocalGeom::set_CV_bias_force(std::string) {
    throw std::runtime_error("LOCAL_GEOM requires Graph adjoints");
}
void LocalGeom::summary(FILE* f) {
    if(f) fprintf(f,"LOCAL_GEOM cutoff=%g epsilon=%g neighbor_type=%d OUTPUT=%s center=%g sigma=%g\n",
        cutoff_,epsilon_,neighbor_type_,tet_?"TET":"COORD",center_,sigma_);
}
void LocalGeom::bind_assembly(std::shared_ptr<const CVAssembly> assembly) {
    if(assembly && assembly->selection_id()!=selection_)
        throw std::invalid_argument("LOCAL_GEOM: incompatible selection assembly");
    ready_=false; checked_assembly_=nullptr;
    CV::bind_assembly(std::move(assembly));
}
void LocalGeom::base_calc() {
    ready_=false;
    neighbors_=engine.neighbors().prepare(neighbor_handle_);
    if(neighbors_.coordinates!=NeighborCoordinates::PeriodicImages || neighbors_.kind!=NeighborKind::Full)
        throw std::invalid_argument("LOCAL_GEOM requires full periodic-image neighbor rows");
    const auto stream=execution_.get();
    neighbors_.host_centers.wait_host();
    const size_t count=neighbors_.host_centers.item_count;
    total_=engine.total_atom_count();
    if(assembly_) {
        if(assembly_->nlocal()!=engine.owned_atom_count() || assembly_->nghost()!=total_-engine.owned_atom_count() ||
           assembly_->group_count()!=count) throw std::logic_error("LOCAL_GEOM: stale assembly");
        if(!neighbors_.topology_generation || checked_generation_!=neighbors_.topology_generation || checked_assembly_!=assembly_.get()) {
            for(size_t k=0;k<count;++k)
                if(neighbors_.host_centers(k)!=assembly_->local_ids()[k])
                    throw std::logic_error("LOCAL_GEOM: neighbor and assembly center order differs");
            checked_generation_=neighbors_.topology_generation; checked_assembly_=assembly_.get();
        }
    }
    positions_=engine.positions(MemorySpace::Cuda);
    types_=engine.atom_types(MemorySpace::Cuda);
    positions_.wait_on(stream); types_.wait_on(stream);
    neighbors_.centers.wait_on(stream); neighbors_.offsets.wait_on(stream); neighbors_.neighbors.wait_on(stream);
    for(auto* buffer:{&values_,&coordination_,&rnn_,&valid_}) {
        buffer->grow_to(total_,__FILE__,__LINE__);
        if(total_) check(cudaMemsetAsync(buffer->ptr,0,total_*sizeof(double),stream));
    }
    if(count) forward_kernel<<<(count+127)/128,128,0,stream>>>(neighbors_.centers,neighbors_.offsets,
        neighbors_.neighbors,positions_,types_,neighbor_type_,cutoff_,epsilon_,radial_,tet_,center_,sigma_,
        values_.ptr,coordination_.ptr,rnn_.ptr,valid_.ptr);
    check(cudaGetLastError()); evaluation_=engine.evaluation_id(); ready_=true;
}
void LocalGeom::check_values() const {
    if(!ready_ || evaluation_!=engine.evaluation_id() || total_!=engine.total_atom_count())
        throw std::logic_error("LOCAL_GEOM: current base_calc required");
}
void LocalGeom::assemble_values(ArrayView<double> values) {
    check_values();
    if(!assembly_) throw std::logic_error("LOCAL_GEOM: assembly required");
    assembly_->assemble_values(values_.view(total_),values,execution_.get());
}
void LocalGeom::apply_adjoint(ArrayView<const double> leaf) {
    check_values();
    if(!assembly_) throw std::logic_error("LOCAL_GEOM: assembly required");
    const auto stream=execution_.get();
    const int count=assembly_->group_count();
    adjoints_.grow_to(count,__FILE__,__LINE__);
    assembly_->gather_adjoint(leaf,adjoints_.view(count),stream);
    forces_.grow_to(size_t(total_)*3,__FILE__,__LINE__); virials_.grow_to(size_t(total_)*6,__FILE__,__LINE__);
    if(total_) { check(cudaMemsetAsync(forces_.ptr,0,size_t(total_)*3*sizeof(double),stream));
                 check(cudaMemsetAsync(virials_.ptr,0,size_t(total_)*6*sizeof(double),stream)); }
    if(count) backward_kernel<<<(count+127)/128,128,0,stream>>>(neighbors_.centers,neighbors_.offsets,
        neighbors_.neighbors,positions_,types_,neighbor_type_,cutoff_,epsilon_,radial_,tet_,center_,sigma_,
        values_.ptr,coordination_.ptr,adjoints_.ptr,forces_.ptr,virials_.ptr);
    check(cudaGetLastError());
    auto force=forces_.view(total_,3),virial=virials_.view(total_,6);
    force.readiness=virial.readiness=execution_.ready();
    engine.commit_force_virial(force,{virial,VirialRepresentation::Symmetric6});
}
ArrayView<const double> LocalGeom::peratom_values(const std::string& name) {
    check_values();
    GpuBuffer<double>* buffer=nullptr;
    if(name=="value") buffer=&values_;
    else if(name=="n" || name=="coordination") buffer=&coordination_;
    else if(name=="rnn") buffer=&rnn_;
    else if(name=="rnn_valid") buffer=&valid_;
    else throw std::invalid_argument("LOCAL_GEOM: unknown diagnostic "+name);
    diagnostic_.resize(std::max(1,total_));
    if(total_) buffer->download_to(diagnostic_.data(),total_,execution_.get());
    return {diagnostic_.data(),static_cast<size_t>(total_),1,1,1,diagnostic_.size(),MemorySpace::Host};
}
}
REGISTER_CV("LOCAL_GEOM", MetaD_zqc::LocalGeom::create);
