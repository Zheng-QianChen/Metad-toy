#!/usr/bin/env python3
"""Compare the completed Al3Sc shared-frozen verification matrix."""
import argparse
import json
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root",type=Path,required=True)
parser.add_argument("--output",type=Path,required=True)
args=parser.parse_args()
def load(name):
    result=json.loads((args.root/name/"result.json").read_text())
    if not result["passed"]: raise AssertionError(name)
    return result

names=["al3sc-host-np1","al3sc-host-np2","al3sc2-kk-np1","al3sc2-kk-np2",
       "al3sc-single-host-np2","al3sc2-single-kk-np2","production4-host","production4-kk",
       "replay-host","replay-kk"]
results={name:load(name) for name in names}
if len({value["plugin_sha256"] for value in results.values()})!=1: raise AssertionError("plugin differs")
if len({value["table_sha256"] for value in results.values()})!=1: raise AssertionError("table differs")

def compare_state(first,second):
    a=np.array(first["rows"]);b=np.array(second["rows"])
    if a.shape!=b.shape or not np.array_equal(a[:,:2],b[:,:2]): raise AssertionError("atom identity")
    cell=np.array(second.get("cell",np.eye(3)*12.45))
    difference=b[:,2:5]-a[:,2:5]
    fractional=difference@np.linalg.inv(cell).T
    coordinate_error=float(np.max(np.abs((fractional-np.rint(fractional))@cell.T)))
    output=dict(coordinates=coordinate_error,forces=float(np.max(np.abs(a[:,5:8]-b[:,5:8]))),
                descriptors=float(np.max(np.abs(a[:,8:]-b[:,8:]))),
                energy=abs(first["energy"]-second["energy"]),mean=abs(first["mean"]-second["mean"]),
                virial=float(np.max(np.abs(np.array(first["virial"])-second["virial"]))))
    if any(value>2e-9 for value in output.values()): raise AssertionError(output)
    return output

comparisons={}
for reference,candidate in [("al3sc-host-np1","al3sc-host-np2"),("al3sc-host-np1","al3sc2-kk-np1"),
                            ("al3sc-host-np1","al3sc2-kk-np2"),
                            ("al3sc-single-host-np2","al3sc2-single-kk-np2"),
                            ("production4-host","production4-kk")]:
    a=results[reference];b=results[candidate]
    pairs=list(zip([a["baseline"],*a["frames"],a["refreshed"]],
                   [b["baseline"],*b["frames"],b["refreshed"]]))
    if len(a["frames"])!=len(b["frames"]): raise AssertionError("frame count")
    errors=[compare_state(x,y) for x,y in pairs]
    comparisons[candidate]=dict(reference=reference,states=len(errors),
                               maximum={key:max(row[key] for row in errors) for key in errors[0]})
replay={}
for mode in ("host","kk"):
    original=results["production4-"+mode]["frames"][-1]
    restored=results["replay-"+mode]["baseline"]
    replay[mode]=compare_state(original,restored)
fd=[row for result in results.values() for row in result["finite_difference"]]
summary=dict(passed=True,finite_difference_configurations=sum(bool(r["finite_difference"]) for r in results.values()),
             maximum_force_fd_error=max(row["max_force_error"] for row in fd),
             maximum_virial_fd_error=max(row["max_virial_error"] for row in fd),
             comparisons=comparisons,restart_replay=replay,
             table_sha256=next(iter(results.values()))["table_sha256"],
             plugin_sha256=next(iter(results.values()))["plugin_sha256"],
             cases={name:{key:value[key] for key in ("ranks","single","steps","force_peak","finite_difference")}
                    for name,value in results.items()})
args.output.write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
