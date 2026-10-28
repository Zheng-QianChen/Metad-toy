#!/usr/bin/env python3
"""Matched comparison: the SAME CV and MD-trained table, frozen or continued.

Each CV family trains eight independent tables with Langevin dynamics. Held-out
first-passage replicas start from the same original basin ensemble in all arms.
Freeze/continue branches share table bytes, physical state and thermal-noise
seed; only future deposition differs. Bootstrap resamples both training tables
and validation replicas, keeping matched arms paired.
"""
import argparse
import concurrent.futures
import ctypes
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np
from probability import pointer
from kinetics import Bridge

FAMILIES=[("physical",4),("compressed",5)]
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
class MatchedBridge(Bridge):
    def __init__(self,library,cdf):
        super().__init__(library,cdf)
        p=ctypes.POINTER(ctypes.c_double)
        self.library.kinetics_train_shared.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int,
            ctypes.c_double,ctypes.c_double,ctypes.c_ulonglong,ctypes.c_int,ctypes.c_char_p,p]
        self.library.kinetics_matched_passage.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int,
            ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_ulonglong,ctypes.c_int,p]
    def train(self,empty,mode,step,duration,seed,direction,prefix):
        result=np.empty(7)
        self.call("kinetics_train_shared",self.cdf,str(empty).encode(),mode,step,duration,seed,direction,
                  str(prefix).encode(),pointer(result))
        return result
    def passage(self,table,mode,update,step,cap,seed,direction):
        result=np.empty(14)
        self.call("kinetics_matched_passage",self.cdf,str(table).encode(),mode,update,step,cap,seed,
                  direction,pointer(result))
        return result

def bootstrap_means(data,table_indices,replica_indices):
    return data[table_indices[:,:,None],replica_indices].mean(axis=(1,2))
def summarize(paths,table_indices,replica_indices,cap):
    means=bootstrap_means(paths[:,:,0],table_indices,replica_indices)
    return dict(replicas=int(np.prod(paths.shape[:2])),events=int(paths[:,:,1].sum()),
        censored=int(np.sum(paths[:,:,1]==0)),restricted_mean_time=float(paths[:,:,0].mean()),
        restricted_mean_ci=np.quantile(means,[.025,.975]).tolist(),
        per_table_time=paths[:,:,0].mean(axis=1).tolist(),
        mean_added_hills=float(paths[:,:,2].mean()),maximum_bias_force=float(paths[:,:,4].max()),
        censor_time=cap)
def plot(directory,records,all_paths,step,cap):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(2,2,figsize=(12,7.5),constrained_layout=True)
    colors=["#555555","#c45a3e","#347ca0"]
    times=np.linspace(0,cap,401)
    for row,(family,_) in enumerate(FAMILIES):
        for direction in (0,1):
            for arm,color in zip(["unbiased","frozen","continued"],colors):
                paths=all_paths[(family,step,direction,arm)].reshape(-1,14)
                survival=np.mean(paths[:,0,None]>times[None,:],axis=0)
                survival[-1]=np.mean(paths[:,1]==0)
                axes[row,direction].step(times,survival,where="post",label=arm,color=color)
            axes[row,direction].set(xlabel="Reduced MD time",ylabel="Fraction not yet arrived",
                title=f"{family.capitalize()} CV: "+["A -> B","B -> A"][direction],ylim=(-.02,1.02))
    axes[0,0].legend(title="Same validation initial states")
    figure.suptitle("Freeze vs continue: same CV, same MD-trained initial bias\n8 training tables; 16 validation replicas per table and direction",fontsize=14)
    figure.savefig(directory/"matched_md_freeze.png",dpi=170)
    plt.close(figure)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library",type=Path,required=True)
    parser.add_argument("--inputs",type=Path,required=True,help="directory supplying cdf.table and zero.table")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--training-tables",type=int,default=8)
    parser.add_argument("--replicas-per-table",type=int,default=16)
    parser.add_argument("--training-duration",type=float,default=200.)
    parser.add_argument("--training-step",type=float,default=.0005)
    parser.add_argument("--steps",type=float,nargs="+",default=[.0005,.00025])
    parser.add_argument("--cap",type=float,default=1000.)
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    for filename in ["cdf.table","zero.table"]:
        shutil.copyfile(args.inputs/filename,output/filename)
    bridge=MatchedBridge(args.library,output/"cdf.table")
    table_count=args.training_tables;replica_count=args.replicas_per_table
    random=np.random.default_rng(921018)
    table_indices=random.integers(0,table_count,(4000,table_count))
    replica_indices=random.integers(0,replica_count,(4000,table_count,replica_count))
    records=[];all_paths={};training=[];checks=[]
    start=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for family,mode in FAMILIES:
            print("TRAIN",family,flush=True)
            futures=[pool.submit(bridge.train,output/"zero.table",mode,args.training_step,args.training_duration,
                187210+table_index,table_index%2,output/f"{family}_{table_index}")
                for table_index in range(table_count)]
            for table_index,future in enumerate(futures):
                result=future.result();table=output/f"{family}_{table_index}.table"
                assert result[0]==2*round(args.training_duration/.5)
                assert result[5]<1e-6 and result[6]<.01
                frozen_mode=3 if mode==4 else 2
                force_error=bridge.force_check(table,frozen_mode)
                early=[]
                for update in (0,1):
                    early.append(bridge.passage(table,mode,update,.0005,.25,81631,0))
                assert np.array_equal(early[0],early[1]),"arms differ before first new hill"
                training.append(dict(family=family,index=table_index,seed=187210+table_index,
                    initial_basin=table_index%2,hills=int(result[0]),transitions=int(result[1]),
                    mean_kinetic_temperature=float(result[2]),table_energy_error=float(result[5]),
                    table_gradient_error=float(result[6]),force_fd_max=force_error,
                    sha256=digest(table),early_trajectory_bitwise=True))
            print("TRAINED",family,table_count,"tables",flush=True)
        for step in args.steps:
            for direction in (0,1):
                futures=[pool.submit(bridge.passage,output/"zero.table",0,0,step,args.cap,
                    821901+table_index*1000+replica,direction)
                    for table_index in range(table_count) for replica in range(replica_count)]
                baseline=np.stack([f.result() for f in futures]).reshape(table_count,replica_count,14)
                np.save(output/f"unbiased_{step}_{direction}.npy",baseline)
                for family,mode in FAMILIES:
                    all_paths[(family,step,direction,"unbiased")]=baseline
                    groups={"unbiased":baseline}
                    for arm,update in [("frozen",0),("continued",1)]:
                        futures=[pool.submit(bridge.passage,output/f"{family}_{table_index}.table",
                            mode,update,step,args.cap,821901+table_index*1000+replica,direction)
                            for table_index in range(table_count) for replica in range(replica_count)]
                        paths=np.stack([f.result() for f in futures]).reshape(table_count,replica_count,14)
                        all_paths[(family,step,direction,arm)]=paths;groups[arm]=paths
                        np.save(output/f"{family}_{step}_{direction}_{arm}.npy",paths)
                    assert np.array_equal(groups["frozen"][:,:,5:12],groups["continued"][:,:,5:12])
                    assert np.array_equal(groups["frozen"][:,:,5:9],baseline[:,:,5:9])
                    assert np.all(groups["frozen"][:,:,2]==0)
                    summaries={arm:summarize(paths,table_indices,replica_indices,args.cap) for arm,paths in groups.items()}
                    bootstrap={arm:bootstrap_means(paths[:,:,0],table_indices,replica_indices) for arm,paths in groups.items()}
                    comparisons={}
                    for numerator,denominator,label in [
                            ("unbiased","frozen","baseline_over_frozen"),
                            ("unbiased","continued","baseline_over_continued"),
                            ("frozen","continued","frozen_over_continued")]:
                        ratio=bootstrap[numerator]/bootstrap[denominator]
                        comparisons[label]=dict(ratio=summaries[numerator]["restricted_mean_time"]/summaries[denominator]["restricted_mean_time"],
                            ci=np.quantile(ratio,[.025,.975]).tolist())
                    record=dict(family=family,mode=mode,step=step,direction=direction,arms=summaries,
                        comparisons=comparisons,initial_state_bias_force_bitwise=True)
                    records.append(record)
                    print("RESULT",family,step,direction,
                          {arm:round(value["restricted_mean_time"],3) for arm,value in summaries.items()},
                          "freeze/continue",comparisons["frozen_over_continued"],flush=True)
                    (output/"progress.json").write_text(json.dumps(records,indent=2)+"\n")
    for table in training:
        assert digest(output/f'{table["family"]}_{table["index"]}.table')==table["sha256"],"initial table mutated"
    if len(args.steps)==2:
        for family,mode in FAMILIES:
            for direction in (0,1):
                for arm in ("unbiased","frozen","continued"):
                    first=all_paths[(family,args.steps[0],direction,arm)][:,:,0]
                    second=all_paths[(family,args.steps[1],direction,arm)][:,:,0]
                    difference=bootstrap_means(second-first,table_indices,replica_indices)
                    interval=np.quantile(difference,[.025,.975])
                    checks.append(dict(family=family,direction=direction,arm=arm,
                        refined_minus_coarse=float((second-first).mean()),ci=interval.tolist(),
                        compatible_at_current_precision=bool(interval[0]<=0<=interval[1])))
    plot(output,records,all_paths,args.steps[0],args.cap)
    result=dict(configuration=vars(args)|{"library":str(args.library),"inputs":str(args.inputs),"output":str(output)},
        model="U=4[(x²-1)²+(y²-1)²]+(x-y)²; m=kBT=gamma=1",
        protocol="Independent online Langevin training, then held-out matched initial states; freeze vs continued updates",
        training=training,records=records,timestep_checks=checks,
        hierarchy="Paired bootstrap over independently trained tables and validation replicas",
        table_files_unchanged=True,seconds=time.perf_counter()-start,
        cdf_sha256=digest(output/"cdf.table"),source_sha256=digest(Path(__file__)),
        cpp_sha256=digest(Path(__file__).with_name("kinetics_probe.cpp")),library_sha256=digest(args.library))
    (output/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print("COMPLETE",output,"seconds",result["seconds"],flush=True)
if __name__=="__main__":
    main()
