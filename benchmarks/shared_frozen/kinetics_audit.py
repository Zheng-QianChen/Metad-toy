#!/usr/bin/env python3
"""Postprocess timestep refinement without regenerating trajectories."""
import argparse
import json
from pathlib import Path
import numpy as np
from kinetics import Bridge

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse",type=Path,required=True)
    parser.add_argument("--refined",type=Path,required=True)
    parser.add_argument("--finest",type=Path,required=True)
    parser.add_argument("--library",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    arguments=parser.parse_args()
    random=np.random.default_rng(830218)
    temperature_checks=[];summaries={};force_checks=[]
    for label,directory in [("coarse",arguments.coarse),("refined",arguments.refined),("finest",arguments.finest)]:
        summary=json.loads((directory/"summary.json").read_text());summaries[label]=summary
        bridge=Bridge(arguments.library,directory/"cdf.table")
        for record in summary["records"]:
            if "stationary" not in record:continue
            arrays=np.load(directory/f'{record["name"]}_{record["step"]}_stationary.npz')
            samples=arrays["samples"]
            means=samples[:,:,3:5].mean(axis=(1,2));sem=means.std(ddof=1)/np.sqrt(len(means))
            consistent=abs(means.mean()-1)<4*sem+.005
            temperature_checks.append(dict(run=label,name=record["name"],step=record["step"],
                mean=float(means.mean()),independent_sem=float(sem),
                rule="abs(T-1) < 4 independent-chain SEM + .005",consistent=bool(consistent)))
        for table in directory.glob("online_*_path0.table"):
            mode=2 if "compressed" in table.name else 3
            error=bridge.force_check(table,mode)
            force_checks.append(dict(run=label,path=table.name,physical_gradient_fd_max=error))
    convergence=[]
    for name in ("frozen_compressed","online_compressed"):
        for direction in (0,1):
            first=np.load(arguments.refined/f"{name}_0.0005_{direction}_first.npy")[:,0]
            second=np.load(arguments.finest/f"{name}_0.00025_{direction}_first.npy")[:,0]
            indices=random.integers(0,len(first),(4000,len(first)))
            differences=(second[indices]-first[indices]).mean(axis=1)
            interval=np.quantile(differences,[.025,.975])
            convergence.append(dict(name=name,direction=direction,
                change=float(second.mean()-first.mean()),paired_bootstrap_ci=interval.tolist(),
                compatible_at_current_precision=bool(interval[0]<=0<=interval[1])))
    result=dict(temperature_checks=temperature_checks,timestep_passage_comparison=convergence,
                online_frozen_table_force_checks=force_checks,
                harmonic_max_zscore=max(case["max_zscore"] for s in summaries.values() for case in s["harmonic_calibration"]),
                interpretation="Finite precision agreement at these tested steps; not a proof of exact continuous-time rates.")
    arguments.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
if __name__=="__main__":
    main()
