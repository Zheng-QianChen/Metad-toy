#!/usr/bin/env python3
"""Interleave the original and batched plugins on identical frozen Al3Sc inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"shared_frozen"))
import performance

parser=argparse.ArgumentParser(description=__doc__)
for name in ("lammps","mpiexec","before","after","assets","table","output"):
    parser.add_argument("--"+name,type=Path,required=True)
parser.add_argument("--repeats",type=int,default=3)
options=parser.parse_args()
for name in ("lammps","mpiexec","before","after","assets","table","output"):
    setattr(options,name,getattr(options,name).resolve())
options.output.mkdir(parents=True,exist_ok=False)
arguments=SimpleNamespace(lammps=options.lammps,mpiexec=options.mpiexec,assets=options.assets,
                          table=options.table,output=options.output,warmup=50,timeout=180)
metadata=dict(hashes={name:performance.sha256(getattr(options,name)) for name in ("before","after","lammps","table")},
              arguments={key:str(value) if isinstance(value,Path) else value for key,value in vars(options).items()},
              gpu_before=performance.gpu_snapshot(),method="Same performance.py NPT inputs, shuffled A/B; 50 warmup; 300 steps np1 / 200 steps np2")
(options.output/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
records=[]
randomizer=random.Random(20261028)
for repeat in range(options.repeats):
    order=[(ranks,mode,variant) for ranks in (1,2) for mode in ("global","shared") for variant in ("before","after")]
    randomizer.shuffle(order)
    for ranks,mode,variant in order:
        arguments.plugin=getattr(options,variant)
        record=performance.run_one(arguments,"kk",ranks,mode,variant,repeat,300 if ranks==1 else 200)
        records.append(record)
        (options.output/"samples.json").write_text(json.dumps(records,indent=2)+"\n")
        summary={}
        for group_ranks in (1,2):
            for group_mode in ("global","shared"):
                group={}
                for group_variant in ("before","after"):
                    rows=[row for row in records if row["ranks"]==group_ranks and
                          row["mode"]==group_mode and row["phase"]==group_variant]
                    if rows:
                        group[group_variant]=dict(median_ms=statistics.median(row["ms_per_step"] for row in rows),
                            samples_ms=[row["ms_per_step"] for row in rows],
                            modify_ms=statistics.median(row["stages_ms"]["Modify"] for row in rows),
                            neighbor_ms=statistics.median(row["stages_ms"]["Neigh"] for row in rows))
                if len(group)==2:
                    group["time_change_percent"]=100*(group["after"]["median_ms"]/group["before"]["median_ms"]-1)
                summary[f"kk_np{group_ranks}_{group_mode}"]=group
        (options.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
for mode in ("global","shared"):
    initial=[row["initial"]["f_bias"] for row in records if row["mode"]==mode]
    if max(initial)-min(initial)>1e-13: raise AssertionError("Different initial bias")
for name in ("before","after","lammps","table"):
    if metadata["hashes"][name]!=performance.sha256(getattr(options,name)): raise AssertionError("Artifact changed")
metadata["gpu_after"]=performance.gpu_snapshot()
(options.output/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
print("FORCE_BATCH_PERFORMANCE_COMPLETE",flush=True)
