#!/usr/bin/env python3
"""Probability compression and shared bias: reproducible equilibrium experiment.

All energies are in kBT. The physical double well is unchanged across cases.
The marginal CDF is deliberately constructed from its known reference, so that
q is uniform and s=q/(81-80q) compresses exactly 90% probability into [0,.1].
This diagnostic CV construction is not an online production training method.
"""
import argparse
import ctypes
import hashlib
import json
import time
import shutil
from pathlib import Path
import numpy as np
from probability import pointer, quadrature, estimates, histogram, compact, CRITERIA, cli_check

def write_table(path, coordinates, values, slopes):
    np.savetxt(path, np.column_stack((coordinates, values, slopes)), fmt="%.17g",
               header="METAD_TOY_SHARED_FROZEN 1")
def compress(q):
    return q/(81-80*q)
def inverse(s):
    return 81*s/(1+80*s)

class Bridge:
    def __init__(self, library, cuda, cdf):
        self.host=ctypes.CDLL(str(library.resolve()))
        self.cuda=ctypes.CDLL(str(cuda.resolve()))
        self.cdf=str(cdf).encode()
        p=ctypes.POINTER(ctypes.c_double)
        self.host.compression_error.restype=ctypes.c_char_p
        self.host.compression_train.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int,ctypes.c_ulonglong,ctypes.c_int,p]
        self.host.compression_evaluate.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int,ctypes.c_int,p,p]
        self.host.compression_sample.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int,ctypes.c_double,
                                               ctypes.c_ulonglong,ctypes.c_int,ctypes.c_int,p,p]
        self.cuda.shared_probability_cuda_error.restype=ctypes.c_char_p
        self.cuda.shared_probability_cuda_evaluate.argtypes=[ctypes.c_char_p,ctypes.c_int,p,p,p]
    def call(self,name,*args):
        if getattr(self.host,name)(*args):
            raise RuntimeError(self.host.compression_error().decode())
    def evaluate(self,table,compressed,coordinates):
        coordinates=np.ascontiguousarray(coordinates,dtype=float).reshape(-1)
        output=np.empty((len(coordinates),4))
        self.call("compression_evaluate",self.cdf,str(table).encode(),compressed,len(coordinates),
                  pointer(coordinates),pointer(output))
        return output
    def sample(self,table,compressed,scale,seed,basin,count):
        output=np.empty((count,7));acceptance=np.zeros(1)
        self.call("compression_sample",self.cdf,str(table).encode(),compressed,scale,seed,basin,count,
                  pointer(output),pointer(acceptance))
        return output,float(acceptance[0])
    def check_cuda(self,table,compressed):
        random=np.random.default_rng(3628)
        coordinates=random.uniform(-1.6,1.6,(256,2))
        coordinates[:3]=[[-3,3],[-1,1],[0,0]]
        def evaluate(points):
            host=self.evaluate(table,compressed,points).reshape(-1,2,4)
            cv=np.ascontiguousarray(host[:,:,0]);energy=np.empty(len(cv));gradient=np.empty_like(cv)
            status=self.cuda.shared_probability_cuda_evaluate(str(table).encode(),len(cv),pointer(cv),
                                                              pointer(energy),pointer(gradient))
            if status: raise RuntimeError(self.cuda.shared_probability_cuda_error().decode())
            return energy,gradient*host[:,:,1],host
        energy,gradient,host=evaluate(coordinates)
        energy_error=float(np.max(abs(energy-host[:,:,2].mean(axis=1))))
        gradient_error=float(np.max(abs(gradient-host[:,:,3]/2)))
        step=2e-7
        plus,minus=coordinates.copy(),coordinates.copy()
        plus[:,0]+=step;minus[:,0]-=step
        fd=(evaluate(plus)[0]-evaluate(minus)[0])/(2*step)
        fd_error=float(np.max(abs(fd-gradient[:,0])))
        assert energy_error<1e-11 and gradient_error<1e-9 and fd_error<0.003
        return dict(energy_max=energy_error,physical_gradient_max=gradient_error,
                    physical_finite_difference_max=fd_error,step=step,pairs=len(coordinates))

def create_cdf(path):
    # Independent Gauss-Legendre integration over y supplies p0(x). Integrating
    # this density within each x interval avoids trapezoidal CDF distortion.
    nodes,weights=np.polynomial.legendre.leggauss(16)
    centers=(np.arange(32)+.5)*6/32-3
    y=(centers[:,None]+nodes*3/32).ravel()
    measure=np.tile(weights*3/32,32)
    def marginal(x):
        x=np.asarray(x).reshape(-1)
        output=np.empty_like(x)
        for first in range(0,len(x),1024):
            xx=x[first:first+1024,None]
            potential=4*((xx*xx-1)**2+(y*y-1)**2)+(xx-y)**2
            output[first:first+1024]=np.exp(-potential)@measure
        return output
    x=np.linspace(-3,3,8193)
    inner,inner_weights=np.polynomial.legendre.leggauss(8)
    mid=(x[:-1]+x[1:])/2; half=(x[1]-x[0])/2
    increments=(marginal((mid[:,None]+half*inner).ravel()).reshape(-1,8)@inner_weights)*half
    partition=increments.sum()
    cdf=np.r_[0,np.cumsum(increments)]/partition
    cdf[-1]=1
    density=marginal(x)/partition
    write_table(path,x,cdf,density)
    return x,cdf,dict(partition=float(partition),cdf_endpoint=float(cdf[-1]),
                      probability_threshold_x=float(np.interp(.9,cdf,x)))

def flat_control(directory):
    # Draw exactly from the frozen, flattened-CV target: uniform s. The raw
    # physical q then has density g'(q); exp(B) restores the uniform q density.
    random=np.random.default_rng(21937)
    samples=random.random((8,100000))
    physical=inverse(samples)
    bias=2*np.log(81/(1+80*samples))
    estimates=[]
    edges=np.linspace(0,1,65)
    raw=np.histogram(physical,edges)[0].astype(float);raw/=raw.sum()
    weights=np.exp(bias-bias.max())
    recovered=np.histogram(physical,edges,weights=weights)[0];recovered/=recovered.sum()
    for q,w in zip(physical,weights):
        estimates.append(np.average(q<=.9,weights=w))
    probability=float(np.average(physical<=.9,weights=weights))
    uncertainty=float(np.std(estimates,ddof=1)/np.sqrt(len(estimates)))
    assert abs(probability-.9)<4*uncertainty+.002
    np.savez_compressed(directory/"flat_control.npz",raw=raw,recovered=recovered,edges=edges)
    return dict(raw_probability_q_le_09=float(np.mean(physical<=.9)),
                reweighted_probability_q_le_09=probability,independent_sem=uncertainty,
                expected_probability=.9,relative_bias_range_kbt=float(2*np.log(81)),
                bias_q09_minus_q1_kbt=float(2*np.log(9)),
                total_variation=float(abs(recovered-1/64).sum()/2),
                weight_ess=float(weights.sum()**2/np.sum(weights**2)))

def biased_reference(bridge,table,compressed,scale,order):
    """Block quadrature limits memory while resolving narrow mapped hills."""
    nodes,weights=np.polynomial.legendre.leggauss(order)
    centers=(np.arange(32)+.5)*6/32-3
    axis=(centers[:,None]+nodes*3/32).ravel()
    measure=np.tile(weights*3/32,32)
    per_center=bridge.evaluate(table,compressed,axis)[:,2]*scale
    y=axis[None,:]
    probability=np.zeros((32,32));moments=np.zeros(6)
    for cell in range(32):
        indices=slice(cell*order,(cell+1)*order)
        x=axis[indices,None]
        potential=4*((x*x-1)**2+(y*y-1)**2)+(x-y)**2
        mass=np.exp(-potential-per_center[indices,None]-per_center[None,:])
        mass*=measure[indices,None]*measure[None,:]
        probability[cell]=mass.reshape(order,32,order).sum(axis=(0,2))
        row_mass=mass.sum(axis=1);column_mass=mass.sum(axis=0)
        moments[0]+=row_mass@(x[:,0]>0)
        moments[1]+=column_mass@(axis>0)
        moments[2]+=mass[(x*y)>0].sum()
        moments[3]+=np.einsum("ij,i,j->",mass,x[:,0],axis)
        moments[4]+=row_mass@(x[:,0]**2)
        moments[5]+=column_mass@(axis**2)
    total=probability.sum()
    return dict(probability=probability/total,moments=moments/total)

def plot_results(directory,records,reference,flat):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    q=np.linspace(0,1,2001);s=np.linspace(0,1,2001)
    figure,axes=plt.subplots(2,2,figsize=(11,7.5),constrained_layout=True)
    axes[0,0].plot(q,q,label="Ordinary: s = q",color="#4488aa")
    axes[0,0].plot(q,compress(q),label="Compressed: s = q / (81 - 80q)",color="#c45332")
    axes[0,0].plot([0,.9,.9],[.1,.1,0],"--",color="gray")
    axes[0,0].set(xlabel="Physical coordinate q",ylabel="CV s",title="90% physical probability -> 10% CV range")
    axes[0,0].legend(fontsize=9)
    axes[0,1].plot(s,2*np.log(1+80*s),label="CV free energy F(s)",color="#386f98")
    axes[0,1].plot(s,2*np.log(81/(1+80*s)),label="Ideal flattening bias B(s)",color="#c45332")
    axes[0,1].set(xlabel="Compressed CV s",ylabel="Energy / kBT",title="Flat U(q)=0; non-flat CV free energy")
    axes[0,1].legend()
    saved=np.load(directory/"flat_control.npz");mid=(saved["edges"][:-1]+saved["edges"][1:])/2
    for key,label,color in [("raw","Raw biased trajectory","#c45332"),("recovered","Full-bias reweighted","#278573")]:
        fes=-np.log(saved[key]*64);fes-=fes.min()
        axes[1,0].plot(mid,fes,label=label,color=color)
    axes[1,0].axhline(0,color="black",ls="--",label="Reference F(q)=0")
    axes[1,0].set(xlabel="Original coordinate q",ylabel="F(q) / kBT",title="Return to the same physical coordinate")
    axes[1,0].legend(fontsize=9)
    axes[1,1].bar(["Reference","Raw biased","Reweighted"],
                  [.9,flat["raw_probability_q_le_09"],flat["reweighted_probability_q_le_09"]],
                  color=["#777777","#c45332","#278573"])
    axes[1,1].set(ylim=(0,1),ylabel="P(q <= 0.9)",title="Probability in the compressed physical region")
    figure.suptitle("Analytic control: coordinate compression alone raises the relative bias",fontsize=14)
    figure.savefig(directory/"compression_flat.png",dpi=180);plt.close(figure)

    chosen=[r for r in records if r["name"] in ("learned_plain","learned_compressed","ideal_compressed_sum")]
    figure,axes=plt.subplots(len(chosen),3,figsize=(10,9),constrained_layout=True)
    colors=plt.get_cmap("viridis").copy();colors.set_bad("#dedede")
    for row,record in enumerate(chosen):
        for col,(title,probability,mask) in enumerate([
                ("Reference",reference["probability"],reference["probability"]>0),
                ("Raw biased",record["raw_array"],record["raw_mask"]),
                ("Full-bias reweighted",record["correct_array"],record["correct_mask"])]):
            fes=np.full_like(probability,np.nan)
            fes[mask]=-np.log(probability[mask]);fes-=np.nanmin(fes)
            artist=axes[row,col].imshow(fes.T,origin="lower",extent=(-3,3,-3,3),
                      vmin=0,vmax=8,cmap=colors)
            axes[row,col].set(xlim=(-1.8,1.8),ylim=(-1.8,1.8),xlabel="x",ylabel="y")
            axes[row,col].set_title(title)
        axes[row,0].set_ylabel(record["name"].replace("_","\n")+"\ny")
    figure.colorbar(artist,ax=axes,label="F(x,y) / kBT",shrink=.8)
    figure.suptitle("Coupled double well: same physical coordinates in every panel\nGray = fewer than 60 effective bin samples",fontsize=13)
    figure.savefig(directory/"compression_joint_fes.png",dpi=180);plt.close(figure)

    figure,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    cv_edges=np.linspace(0,1,65);cv_mid=(cv_edges[:-1]+cv_edges[1:])/2
    for name,color,label in [("learned_plain","#4488aa","Ordinary CDF CV"),("learned_compressed","#c45332","Compressed CDF CV")]:
        record=next(r for r in records if r["name"]==name)
        table=np.loadtxt(directory/(name+".table"))
        axes[0,0].plot(table[:,0],.5*(table[:,1]-table[:,1].min()),label=label,color=color)
        probabilities=record["cv_probability"]
        reference_cv=np.diff(inverse(cv_edges)) if record["compressed"] else np.diff(cv_edges)
        measured=-np.log(probabilities/np.diff(cv_edges));measured-=measured.min()
        expected=-np.log(reference_cv/np.diff(cv_edges));expected-=expected.min()
        axes[0,1].plot(cv_mid,measured,label=label+" reweighted",color=color)
        axes[0,1].plot(cv_mid,expected,ls="--",color=color)
        progress=np.genfromtxt(directory/(name+".progress.csv"),delimiter=",",names=True)
        axes[1,0].plot(progress["sweep"],.5*progress["range_u"],label=label,color=color)
        hist=record["force_histogram"]
        axes[1,1].stairs(hist["probability"],hist["edges"],label=label,color=color)
    axes[0,0].set(xlabel="CV s",ylabel="One-center contribution / kBT",title="Frozen learned 0.5 [u(s) - min u]")
    axes[0,1].set(xlabel="CV s",ylabel="CV free energy / kBT",title="Reweighted CV FES; dashed = reference bin masses")
    axes[1,0].set(xlabel="Training Metropolis sweeps",ylabel="0.5 [max u - min u] / kBT",title="Relative bias growth (common offset removed)")
    axes[1,1].set(xlabel="|dB/dx| in reference configurations",ylabel="Probability per logarithmic bin",
                  xscale="log",title="Physical bias gradient after chain rule")
    for axis in axes.ravel(): axis.legend(fontsize=8)
    figure.suptitle("Common deposition: same height, width and training length",fontsize=14)
    figure.savefig(directory/"compression_bias.png",dpi=180);plt.close(figure)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library",type=Path,required=True)
    parser.add_argument("--cuda-library",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--samples",type=int,default=120000,help="retained frames per independent chain")
    parser.add_argument("--training-sweeps",type=int,default=1000000)
    parser.add_argument("--extend-from",type=Path,help="reuse fixed tables and extend only the two MEAN compression cases")
    arguments=parser.parse_args();directory=arguments.output.resolve();directory.mkdir(parents=True,exist_ok=False)
    previous=None
    if arguments.extend_from:
        previous=json.loads((arguments.extend_from/"summary.json").read_text())
    started=time.time()
    axis,cdf,cdf_info=create_cdf(directory/"cdf.table")
    bridge=Bridge(arguments.library,arguments.cuda_library,directory/"cdf.table")
    reference=quadrature(2,16)
    reference_fine=quadrature(2,24)
    cdf_info["reference_bin_convergence"]=float(abs(reference["probability"]-reference_fine["probability"]).max())
    cdf_info["partition_relative_error"]=abs(cdf_info["partition"]/reference["partition"]-1)
    assert cdf_info["partition_relative_error"]<1e-10 and cdf_info["reference_bin_convergence"]<1e-10
    cv_axis=np.linspace(0,1,4097)
    write_table(directory/"zero.table",cv_axis,cv_axis*0,cv_axis*0)
    write_table(directory/"ideal.table",cv_axis,2*np.log(81/(1+80*cv_axis)),-160/(1+80*cv_axis))
    flat=flat_control(directory);print("analytic control:",flat,flush=True)
    training={}
    for compressed in [False,True]:
        name="learned_compressed" if compressed else "learned_plain"
        diagnostics=np.zeros(3)
        if previous:
            for suffix in [".hills",".table",".progress.csv"]:
                shutil.copyfile(arguments.extend_from/(name+suffix),directory/(name+suffix))
            training[name]=previous["training"][name]
        else:
            bridge.call("compression_train",bridge.cdf,str(directory/name).encode(),compressed,71837,
                        arguments.training_sweeps,pointer(diagnostics))
            assert diagnostics[1]<1e-6 and diagnostics[2]<.01,diagnostics
            training[name]=dict(hills=int(diagnostics[0]),interpolation_energy_max=float(diagnostics[1]),
                                interpolation_gradient_max=float(diagnostics[2]),history_bitwise=True)
        print(name,"training",training[name],flush=True)
    definitions=[("unbiased","zero.table",False,0.5),
                 ("learned_plain","learned_plain.table",False,.5),
                 ("learned_compressed","learned_compressed.table",True,.5),
                 ("ideal_compressed_mean","ideal.table",True,.5),
                 ("ideal_compressed_sum","ideal.table",True,1.)]
    records=[];reference_samples=None
    for case_index,(name,file,compressed,scale) in enumerate(definitions):
        print("sampling",name,flush=True)
        chains=[];acceptances=[];chain_moments=[];chain_region=[]
        table=directory/file
        extend_case=name in ("learned_compressed","ideal_compressed_mean")
        old_chains=np.load(arguments.extend_from/(name+".npz"))["samples"] if previous else None
        for chain_index in range(8):
            if previous and not extend_case:
                samples=old_chains[chain_index]
                old_record=next(record for record in previous["records"] if record["name"]==name)
                acceptance=old_record["acceptance_range"][chain_index%2]
            else:
                samples,acceptance=bridge.sample(table,compressed,scale,991+case_index*100+chain_index,
                                                 chain_index%4,arguments.samples)
                if previous:
                    assert np.array_equal(samples[:len(old_chains[chain_index])],old_chains[chain_index]), "chain prefix changed"
            chains.append(samples);acceptances.append(acceptance)
            estimate=estimates(samples,samples[:,2],reference)
            chain_moments.append(estimate["moments"])
            weight=np.exp(samples[:,2]-samples[:,2].max())
            chain_region.append(np.average(samples[:,0]<=cdf_info["probability_threshold_x"],weights=weight))
        chain_array=np.stack(chains);samples=np.concatenate(chains)
        np.savez_compressed(directory/(name+".npz"),samples=chain_array)
        if reference_samples is None: reference_samples=samples[::16,:2].copy()
        raw=estimates(samples,np.zeros(len(samples)),reference)
        correct=estimates(samples,samples[:,2],reference)
        biased=biased_reference(bridge,table,compressed,scale,32)
        for reference_order in (64,128,256,512):
            biased_fine=biased_reference(bridge,table,compressed,scale,reference_order)
            biased_convergence=float(abs(biased["probability"]-biased_fine["probability"]).max())
            if biased_convergence<2e-7:
                break
            biased=biased_fine
        assert biased_convergence<2e-7,biased_convergence
        sampled_biased=estimates(samples,np.zeros(len(samples)),biased_fine)
        sem=np.std(chain_moments,axis=0,ddof=1)/np.sqrt(8)
        difference=abs(correct["moments"]-reference["moments"])
        passed=bool(difference.max()<CRITERIA["moment_max_abs"] and sem.max()<CRITERIA["moment_max_sem"]
                    and np.all(difference<4*sem+.003) and correct["total_variation"]<.035
                    and correct["fes_rmse_kbt"]<.12 and correct["covered_reference_mass"]>=.98)
        weight=np.exp(samples[:,2]-samples[:,2].max())
        cv_mass=np.histogram(samples[:,5],np.linspace(0,1,65),weights=weight)[0];cv_mass/=cv_mass.sum()
        evaluated=bridge.evaluate(table,compressed,reference_samples).reshape(-1,2,4)
        forces=abs(scale*evaluated[:,:,3]).reshape(-1)
        force_edges=np.logspace(-4,3,50);force_counts=np.histogram(forces,force_edges)[0]
        cuda=bridge.check_cuda(table,compressed)
        _,correct_ess,_=histogram(samples,samples[:,2]);_,raw_ess,_=histogram(samples,np.zeros(len(samples)))
        record=dict(name=name,compressed=compressed,scale=scale,frames=len(samples),
            acceptance_range=[min(acceptances),max(acceptances)],passed=passed,
            raw=compact(raw),correct=compact(correct),moment_sem=sem.tolist(),moment_error=difference.tolist(),
            biased_reference_error=biased_convergence,biased_reference_order=reference_order,
            raw_vs_biased_reference=compact(sampled_biased),
            chain_prefix_bitwise=True if previous and extend_case else None,
            actual_samples_per_chain=len(chain_array[0]),
            region_raw=float(np.mean(samples[:,0]<=cdf_info["probability_threshold_x"])),
            region_reweighted=float(np.average(samples[:,0]<=cdf_info["probability_threshold_x"],weights=weight)),
            region_sem=float(np.std(chain_region,ddof=1)/np.sqrt(8)),
            half_length_correct=compact(estimates(chain_array[:,:len(chain_array[0])//2].reshape(-1,7),
                chain_array[:,:len(chain_array[0])//2,2].reshape(-1),reference)),
            force_reference_percentiles=np.quantile(forces,[.5,.9,.99,1]).tolist(),cuda=cuda,
            table_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),
            cv_probability=cv_mass,correct_array=correct["probability"],raw_array=raw["probability"],
            correct_mask=correct_ess>=60,raw_mask=raw_ess>=60,
            force_histogram=dict(edges=force_edges,probability=force_counts/len(forces)))
        records.append(record)
        print(name,"PASS" if passed else "FAIL",compact(correct),"region",record["region_reweighted"],flush=True)
        if name=="learned_compressed":
            cli_directory=directory/"cli";cli_directory.mkdir()
            record["cli"]=cli_check(chains[0],cli_directory,reference)
    plot_results(directory,records,reference,flat)
    result=dict(model="U=4[(x^2-1)^2+(y^2-1)^2]+(x-y)^2; kBT=1; x,y in [-3,3]",
        mapping="q=unbiased marginal CDF(x); compressed s=q/(81-80q)",
        physical_reference_moments=reference["moments"].tolist(),criteria=CRITERIA,
        samples_per_chain=arguments.samples,independent_chains=8,burn_sweeps=20000,stride=8,
        training_sweeps=arguments.training_sweeps,height=.02,sigma=.005,pace=200,
        cdf=cdf_info,flat_control=flat,training=training,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cpp_source_sha256=hashlib.sha256(Path(__file__).with_name("compression_probe.cpp").read_bytes()).hexdigest(),
        extended_from=str(arguments.extend_from) if previous else None,
        host_library_sha256=hashlib.sha256(arguments.library.read_bytes()).hexdigest(),
        cuda_library_sha256=hashlib.sha256(arguments.cuda_library.read_bytes()).hexdigest(),
        records=[{key:value for key,value in record.items()
                  if key not in ["cv_probability","correct_array","raw_array","correct_mask","raw_mask","force_histogram"]}
                 for record in records],seconds=time.time()-started)
    (directory/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print("complete",directory,"seconds",result["seconds"],flush=True)
if __name__=="__main__":
    main()
