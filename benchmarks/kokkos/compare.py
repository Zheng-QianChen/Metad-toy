#!/usr/bin/env python3
"""比较普通/Kokkos 有限差分状态和连续积分结果；失败返回非零退出码。"""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--candidate", type=Path, required=True)
parser.add_argument("--reference-update", type=Path)
parser.add_argument("--candidate-update", type=Path)
parser.add_argument("--lifecycle", type=Path, nargs="+", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

def load_runs(directory, update):
    runs = {}
    for parent in (directory, update):
        if parent is None:
            continue
        for row in json.loads((parent/"results.json").read_text()):
            assert row["exit_code"] == 0, row
            key = row["case"], row["ranks"]
            runs[key] = (row, parent/(row["case"]+"_np"+str(row["ranks"]))/"state.json")
    assert all(row["passed"] for row,_ in runs.values()), "finite difference checks failed"
    return runs

def difference(first, second):
    a, b = np.asarray(first), np.asarray(second)
    assert a.shape == b.shape, (a.shape,b.shape)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    return float(np.max(np.abs(a-b)))

reference = load_runs(args.reference, args.reference_update)
candidate = load_runs(args.candidate, args.candidate_update)
cross = []
for (case,ranks),(row,path) in candidate.items():
    expected = json.loads(reference[case,1][1].read_text())
    actual = json.loads(path.read_text())
    errors = {field:difference(actual[field],expected[field])
              for field in ("rows","values","energy","virial")}
    assert max(errors.values()) < 2e-8, (case,ranks,errors)
    cross.append(dict(case=case,ranks=ranks,errors=errors))
lifecycles = [json.loads((path/"result.json").read_text()) for path in args.lifecycle]
assert lifecycles[0]["ranks"] == 1 and not lifecycles[0]["args"], "put single-rank Host reference first"
trajectory = []
for path,actual in zip(args.lifecycle,lifecycles):
    assert actual["passed"]
    errors = {field:max(difference(a[field],b[field])
                       for a,b in zip(actual["states"],lifecycles[0]["states"]))
              for field in ("rows","values","energy","virial")}
    assert max(errors.values()) < 5e-8, (path,errors)
    trajectory.append(dict(directory=path.name,ranks=actual["ranks"],errors=errors,
                           position_uploads=actual["position_uploads"],
                           migrated=actual["states"][0]["owned_ids"] != actual["states"][1]["owned_ids"]))
report = dict(passed=True,finite_difference_cases=len(reference)+len(candidate),
              finite_difference_cross=cross,trajectory_cross=trajectory)
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(dict(passed=True,finite_difference_cases=report["finite_difference_cases"],
    maximum_state_error=max(max(row["errors"].values()) for row in cross),
    trajectory_cases=len(trajectory),
    maximum_trajectory_error=max(max(row["errors"].values()) for row in trajectory)),indent=2))