#!/usr/bin/env python3
"""BAOAB Langevin transition experiment, including right-censored first passages.

Frozen runs compare the same original potential under four known biases.
Online shared deposition starts afresh on every path and has a physical pace.
All clocks use reduced MD time; reweighted equilibrium frames are never used
to rescale a transition time. Full state is (x,y,vx,vy), plus hills when online.
"""
import argparse
import concurrent.futures
import ctypes
import hashlib
import json
import resource
import shutil
import time
from pathlib import Path
import numpy as np
from probability import pointer, quadrature, estimates, compact
from compression import write_table

CASES=[
    ("unbiased",0,"zero.table"),
    ("frozen_plain",1,"learned_plain.table"),
    ("frozen_compressed",2,"learned_compressed.table"),
    ("well_fill_control",3,"well_fill.table"),
    ("online_physical",4,"zero.table"),
    ("online_compressed",5,"zero.table"),
]
LABELS=["Unbiased","Frozen ordinary CV","Frozen compressed CV","Known-well control",
        "Online physical CV","Online compressed CV"]

class Bridge:
    def __init__(self,path,cdf):
        self.library=ctypes.CDLL(str(path.resolve()))
        self.cdf=str(cdf).encode()
        p=ctypes.POINTER(ctypes.c_double)
        self.library.kinetics_error.restype=ctypes.c_char_p
        self.library.kinetics_harmonic.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_double,ctypes.c_int,p]
        self.library.kinetics_evaluate.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int,
                                               ctypes.c_int,p,p]
        self.library.kinetics_first_passage.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int,
            ctypes.c_double,ctypes.c_double,ctypes.c_ulonglong,ctypes.c_int,ctypes.c_char_p,p]
        self.library.kinetics_stationary.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int,
            ctypes.c_double,ctypes.c_double,ctypes.c_ulonglong,p,p]
    def call(self,name,*args):
        if getattr(self.library,name)(*args):raise RuntimeError(self.library.kinetics_error().decode())
    def first(self,table,mode,step,cap,seed,direction,prefix):
        output=np.empty(9)
        self.call("kinetics_first_passage",self.cdf,str(table).encode(),mode,step,cap,seed,direction,
                  str(prefix).encode() if prefix else b"",pointer(output))
        return output
    def stationary(self,table,mode,step,duration,seed):
        output=np.empty((round(duration/.1),6));diagnostics=np.empty(6)
        self.call("kinetics_stationary",self.cdf,str(table).encode(),mode,step,duration,seed,
                  pointer(output),pointer(diagnostics))
        return output,diagnostics
    def evaluate(self,table,mode,coordinates):
        coordinates=np.ascontiguousarray(coordinates,dtype=float)
        output=np.empty((len(coordinates),3))
        self.call("kinetics_evaluate",self.cdf,str(table).encode(),mode,len(coordinates),
                  pointer(coordinates),pointer(output))
        return output
    def force_check(self,table,mode):
        random=np.random.default_rng(5145)
        coordinates=random.uniform(-1.7,1.7,(512,2))
        measured=self.evaluate(table,mode,coordinates)
        finite=np.empty((len(coordinates),2));step=2e-7
        for axis in (0,1):
            plus,minus=coordinates.copy(),coordinates.copy()
            plus[:,axis]+=step;minus[:,axis]-=step
            finite[:,axis]=-(self.evaluate(table,mode,plus)[:,0]-self.evaluate(table,mode,minus)[:,0])/(2*step)
        error=float(np.max(abs(finite-measured[:,1:])))
        assert error<2e-4,error
        return error

def interval(values,random):
    means=values[random.integers(0,len(values),(2000,len(values)))].mean(axis=1)
    return [float(v) for v in np.quantile(means,[.025,.975])]

def passage_summary(paths,cap,random):
    time=paths[:,0];observed=paths[:,1].astype(bool)
    return dict(replicas=len(paths),events=int(observed.sum()),censored=int((~observed).sum()),
                restricted_mean_time=float(time.mean()),restricted_mean_ci=interval(time,random),
                observed_fraction=float(observed.mean()),cap=cap,
                maximum_bias_force=float(paths[:,6].max()),
                maximum_hills=int(paths[:,2].max()),
                integrator_force_evaluations=int(paths[:,5].sum()),
                table_energy_error=float(paths[:,7].max()),table_gradient_error=float(paths[:,8].max()))

def projected_transitions(chains):
    # The four sign regions are an observed projection, not an assumed exact
    # Markov state. Counts and Chapman-Kolmogorov discrepancies are both saved.
    states=(chains[:,:,0]>0).astype(int)*2+(chains[:,:,1]>0).astype(int)
    result={}
    for lag_time in (1.,5.,10.):
        lag=round(lag_time/.1)
        counts=np.zeros((4,4),dtype=np.int64);double_counts=counts.copy()
        for row in states:
            counts+=np.bincount(4*row[:-lag]+row[lag:],minlength=16).reshape(4,4)
            double_counts+=np.bincount(4*row[:-2*lag]+row[2*lag:],minlength=16).reshape(4,4)
        mass=counts.sum(axis=1,keepdims=True)
        doubled_mass=double_counts.sum(axis=1,keepdims=True)
        matrix=np.divide(counts,mass,out=np.zeros((4,4)),where=mass>0)
        doubled=np.divide(double_counts,doubled_mass,out=np.zeros((4,4)),where=doubled_mass>0)
        result[str(lag_time)]=dict(counts=counts.tolist(),matrix=matrix.tolist(),
            ck_row_total_variation=(abs(matrix@matrix-doubled).sum(axis=1)/2).tolist(),
            row_origins=mass[:,0].tolist())
    return result

def stationary_summary(chains,diagnostics,duration,random):
    estimates_by_chain=[];temperatures=[];weights_ess=[]
    for samples in chains:
        weight=np.exp(samples[:,2]-samples[:,2].max())
        estimates_by_chain.append(np.average(samples[:,0]>0,weights=weight))
        temperatures.append(samples[:,3:5].mean())
        weights_ess.append(weight.sum()**2/np.sum(weight**2))
    probabilities=np.asarray(estimates_by_chain);temperatures=np.asarray(temperatures)
    samples=chains.reshape(-1,6)
    reference=quadrature(2,16)
    reweighted=estimates(samples,samples[:,2],reference)
    rates=diagnostics[:,0]/duration
    return dict(replicas=len(chains),duration_per_replica=duration,retained_per_replica=len(chains[0]),
        complete_core_transitions=int(diagnostics[:,0].sum()),left_to_right=int(diagnostics[:,1].sum()),
        right_to_left=int(diagnostics[:,2].sum()),round_trips=int(np.floor(diagnostics[:,0]/2).sum()),
        transition_rate=float(rates.mean()),transition_rate_ci=interval(rates,random),
        kinetic_temperature=float(temperatures.mean()),kinetic_temperature_ci=interval(temperatures,random),
        reweighted_p_x_positive=float(reweighted["moments"][0]),
        independent_chain_p_mean=float(probabilities.mean()),
        independent_chain_p_sem=float(probabilities.std(ddof=1)/np.sqrt(len(chains))),
        independent_chain_mse=float(np.mean((probabilities-.5)**2)),
        independent_chain_probabilities=probabilities.tolist(),reweighted=compact(reweighted),
        transitions_per_replica=diagnostics[:,0].astype(int).tolist(),
        mean_weight_ess=float(np.mean(weights_ess)),projected_transitions=projected_transitions(chains))

def figures(directory,records,first_paths,stationary_chains,step,cap):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected=[record for record in records if record["step"]==step]
    colors=["#444444","#5485ad","#c84e3c","#28a080","#a18526","#8753a5"]
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    axis_time=np.linspace(0,cap,401)
    for case_index,record in enumerate(selected):
        for direction in (0,1):
            paths=first_paths[(record["name"],step,direction)]
            survival=np.mean(paths[:,0,None]>axis_time[None,:],axis=0)
            # At the administrative censor time, paths without an event remain
            # alive; do not make the survival curve falsely drop to zero.
            survival[-1]=np.mean(paths[:,1]==0)
            axes[0,direction].step(axis_time,survival,where="post",label=LABELS[case_index],color=colors[case_index])
    for direction,title in enumerate(["A -> B first passage","B -> A first passage"]):
        axes[0,direction].set(xlabel="Reduced MD time",ylabel="Fraction not yet arrived",title=title,ylim=(-.02,1.02))
    axes[0,0].legend(fontsize=8)
    positions=np.arange(len(selected));width=.36
    for direction,color in [(0,"#3d7e9d"),(1,"#d18c3f")]:
        values=np.array([r["passage"][direction]["restricted_mean_time"] for r in selected])
        confidence=np.array([r["passage"][direction]["restricted_mean_ci"] for r in selected])
        axes[1,0].bar(positions+(direction-.5)*width,values,width,
                     yerr=np.array([values-confidence[:,0],confidence[:,1]-values]),
                     label=["A -> B","B -> A"][direction],color=color,capsize=2)
    axes[1,0].set_xticks(positions,LABELS,rotation=25,ha="right",fontsize=8)
    axes[1,0].set(ylabel="Restricted mean time",title="Includes all paths censored at the common cap")
    axes[1,0].legend()
    frozen=[r for r in selected if "stationary" in r]
    rates=np.array([r["stationary"]["transition_rate"] for r in frozen])
    confidence=np.array([r["stationary"]["transition_rate_ci"] for r in frozen])
    axes[1,1].bar(np.arange(len(frozen)),rates,color=colors[:len(frozen)],
                 yerr=np.array([rates-confidence[:,0],confidence[:,1]-rates]),capsize=3)
    axes[1,1].set_xticks(np.arange(len(frozen)),LABELS[:len(frozen)],rotation=25,ha="right",fontsize=8)
    axes[1,1].set(ylabel="Completed basin changes / MD time",title="Frozen-bias bidirectional exploration")
    fig.suptitle("Langevin MD: directional escape and repeated exploration are different",fontsize=14)
    fig.savefig(directory/"kinetics_acceleration.png",dpi=170);plt.close(fig)

    fig,axes=plt.subplots(2,len(frozen),figsize=(13,6.5),constrained_layout=True,squeeze=False)
    for case_index,record in enumerate(frozen):
        transition=np.asarray(record["stationary"]["projected_transitions"]["5.0"]["matrix"])
        artist=axes[0,case_index].imshow(transition,vmin=0,vmax=1,cmap="Blues")
        for row in range(4):
            for column in range(4):
                axes[0,case_index].text(column,row,f"{transition[row,column]:.2f}",ha="center",va="center",
                                       color="white" if transition[row,column]>.5 else "black",fontsize=8)
        axes[0,case_index].set_xticks(range(4),["--","-+","+-","++"])
        axes[0,case_index].set_yticks(range(4),["--","-+","+-","++"])
        axes[0,case_index].set_title(LABELS[case_index],fontsize=10)
        chains=stationary_chains[(record["name"],step)]
        shown=chains[0,:min(len(chains[0]),10000)]
        time_axis=np.arange(1,len(shown)+1)*.1
        axes[1,case_index].plot(time_axis,(shown[:,0]+shown[:,1])/2,lw=.6)
        axes[1,case_index].axhline(.6,color="gray",ls="--",lw=.6)
        axes[1,case_index].axhline(-.6,color="gray",ls="--",lw=.6)
        axes[1,case_index].set(xlabel="MD time",ylim=(-2,2),title="One equilibrium path")
    axes[0,0].set_ylabel("Origin sign region")
    axes[1,0].set_ylabel("(x+y)/2")
    fig.colorbar(artist,ax=axes[0,:],label="P(region at t+5 | region at t)",shrink=.85)
    fig.suptitle("Observed transition matrices; CV projections need not be Markovian",fontsize=13)
    fig.savefig(directory/"kinetics_transitions.png",dpi=170);plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library",type=Path,required=True)
    parser.add_argument("--tables",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--replicas",type=int,default=128)
    parser.add_argument("--stationary-replicas",type=int,default=16)
    parser.add_argument("--cap",type=float,default=1000.)
    parser.add_argument("--duration",type=float,default=2000.)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--steps",type=float,nargs="+",default=[.002,.001])
    parser.add_argument("--cases",nargs="+",choices=[case[0] for case in CASES])
    arguments=parser.parse_args()
    selected_cases=[case for case in CASES if arguments.cases is None or case[0] in arguments.cases]
    output=arguments.output.resolve();output.mkdir(parents=True,exist_ok=False)
    for name in ("cdf.table","zero.table","learned_plain.table","learned_compressed.table"):
        shutil.copyfile(arguments.tables/name,output/name)
    x=np.linspace(-3,3,4097)
    left=np.exp(-.5*((x+1)/.35)**2);right=np.exp(-.5*((x-1)/.35)**2)
    write_table(output/"well_fill.table",x,4*(left+right),-4*((x+1)*left+(x-1)*right)/.35**2)
    bridge=Bridge(arguments.library,output/"cdf.table")
    harmonic=[]
    for step in arguments.steps:
        measured=np.empty((5,6))
        bridge.call("kinetics_harmonic",bridge.cdf,str(output/"zero.table").encode(),step,20000,pointer(measured))
        zscore=np.abs(measured[:,1]-measured[:,3])/measured[:,2]
        assert zscore.max()<5,measured
        harmonic.append(dict(step=step,rows=measured.tolist(),max_zscore=float(zscore.max())))
        print("HARMONIC",step,"max z-score",float(zscore.max()),flush=True)
    random=np.random.default_rng(271016)
    results=[];all_paths={};all_chains={}
    wall_start=time.perf_counter();cpu_start=resource.getrusage(resource.RUSAGE_SELF).ru_utime
    with concurrent.futures.ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        for step in arguments.steps:
            for name,mode,filename in selected_cases:
                started=time.perf_counter();print("RUN",name,"dt",step,flush=True)
                table=output/filename
                record=dict(name=name,mode=mode,step=step,force_fd_max=bridge.force_check(table,mode),passage=[])
                for direction in (0,1):
                    passage_wall_start=time.perf_counter()
                    passage_cpu_start=resource.getrusage(resource.RUSAGE_SELF).ru_utime
                    futures=[pool.submit(bridge.first,table,mode,step,arguments.cap,
                        731019+index,direction,
                        output/f"{name}_{step}_{direction}_path0" if index==0 else None)
                        for index in range(arguments.replicas)]
                    paths=np.stack([future.result() for future in futures])
                    passage_wall=time.perf_counter()-passage_wall_start
                    passage_cpu=resource.getrusage(resource.RUSAGE_SELF).ru_utime-passage_cpu_start
                    all_paths[(name,step,direction)]=paths
                    np.save(output/f"{name}_{step}_{direction}_first.npy",paths)
                    summary=passage_summary(paths,arguments.cap,random)
                    summary["batch_wall_seconds"]=passage_wall
                    summary["batch_cpu_seconds"]=passage_cpu
                    assert summary["table_energy_error"]<1e-6 and summary["table_gradient_error"]<.01
                    record["passage"].append(summary)
                    print("FPT",name,step,direction,"events",summary["events"],
                          "RMST",round(summary["restricted_mean_time"],3),flush=True)
                if mode<4:
                    stationary_wall_start=time.perf_counter()
                    stationary_cpu_start=resource.getrusage(resource.RUSAGE_SELF).ru_utime
                    futures=[pool.submit(bridge.stationary,table,mode,step,arguments.duration,
                                         912018+index) for index in range(arguments.stationary_replicas)]
                    runs=[future.result() for future in futures]
                    stationary_wall=time.perf_counter()-stationary_wall_start
                    stationary_cpu=resource.getrusage(resource.RUSAGE_SELF).ru_utime-stationary_cpu_start
                    chains=np.stack([run[0] for run in runs]);diagnostics=np.stack([run[1] for run in runs])
                    np.savez_compressed(output/f"{name}_{step}_stationary.npz",samples=chains,diagnostics=diagnostics)
                    all_chains[(name,step)]=chains
                    record["stationary"]=stationary_summary(chains,diagnostics,arguments.duration,random)
                    record["stationary"]["batch_wall_seconds"]=stationary_wall
                    record["stationary"]["batch_cpu_seconds"]=stationary_cpu
                    record["stationary"]["cpu_microseconds_per_step"]=stationary_cpu/(diagnostics[:,5].sum()-len(diagnostics))*1e6
                    print("MIX",name,step,"events",record["stationary"]["complete_core_transitions"],
                          "T",record["stationary"]["kinetic_temperature"],flush=True)
                record["wall_seconds"]=time.perf_counter()-started
                results.append(record)
                (output/"progress.json").write_text(json.dumps(results,indent=2)+"\n")
    comparisons=[]
    for step in arguments.steps:
        baseline=next((record for record in results if record["name"]=="unbiased" and record["step"]==step),None)
        if baseline is None:
            continue
        for name,mode,_ in [case for case in selected_cases if case[0]!="unbiased"]:
            record=next(record for record in results if record["name"]==name and record["step"]==step)
            entry=dict(name=name,step=step,restricted_acceleration=[])
            for direction in (0,1):
                base=all_paths[("unbiased",step,direction)][:,0]
                other=all_paths[(name,step,direction)][:,0]
                indices=random.integers(0,len(base),(4000,len(base)))
                ratios=base[indices].mean(axis=1)/other[indices].mean(axis=1)
                entry["restricted_acceleration"].append(dict(ratio=float(base.mean()/other.mean()),
                    ci=[float(v) for v in np.quantile(ratios,[.025,.975])],
                    cpu_batch_gain=baseline["passage"][direction]["batch_cpu_seconds"]/record["passage"][direction]["batch_cpu_seconds"],
                    wall_batch_gain=baseline["passage"][direction]["batch_wall_seconds"]/record["passage"][direction]["batch_wall_seconds"]))
            if "stationary" in record:
                entry["round_trip_rate_ratio"]=record["stationary"]["transition_rate"]/baseline["stationary"]["transition_rate"]
                entry["same_time_probability_mse_gain"]=baseline["stationary"]["independent_chain_mse"]/record["stationary"]["independent_chain_mse"]
            comparisons.append(entry)
    step_comparisons=[]
    if len(arguments.steps)==2:
        for name,mode,_ in selected_cases:
            first=next(r for r in results if r["name"]==name and r["step"]==arguments.steps[0])
            second=next(r for r in results if r["name"]==name and r["step"]==arguments.steps[1])
            entry=dict(name=name,restricted_mean_ratios=[
                second["passage"][i]["restricted_mean_time"]/first["passage"][i]["restricted_mean_time"] for i in (0,1)])
            if mode<4:
                entry["transition_rate_ratio"]=second["stationary"]["transition_rate"]/first["stationary"]["transition_rate"]
            step_comparisons.append(entry)
    if len(selected_cases)==len(CASES):
        figures(output,results,all_paths,all_chains,arguments.steps[0],arguments.cap)
    summary=dict(model="U=4[(x²-1)²+(y²-1)²]+(x-y)²",
        units=dict(mass=1,thermal_energy=1,friction=1,time="reduced MD time"),
        integrator="BAOAB",harmonic_calibration=harmonic,basins=dict(A="x<-.6 and y<-.6",B="x>.6 and y>.6"),
        initial_ensemble="original Boltzmann conditional on source core, Maxwell velocities; all bias modes share seeds",
        stationary_initial_ensemble="exact rejection under frozen biased potential within [-3,3]^2",
        online=dict(height=.02,physical_sigma=.12,compressed_sigma=.02,pace=.5,aggregation="MEAN",well_tempered=False),
        configuration=vars(arguments)|{"library":str(arguments.library),"tables":str(arguments.tables),"output":str(output)},
        records=results,comparisons=comparisons,timestep_comparison=step_comparisons,
        wall_seconds=time.perf_counter()-wall_start,cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime-cpu_start,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cpp_sha256=hashlib.sha256(Path(__file__).with_name("kinetics_probe.cpp").read_bytes()).hexdigest(),
        library_sha256=hashlib.sha256(arguments.library.read_bytes()).hexdigest())
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print("COMPLETE",output,"seconds",summary["wall_seconds"],flush=True)
if __name__=="__main__":
    main()
