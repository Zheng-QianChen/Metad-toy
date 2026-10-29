#!/usr/bin/env python3
"""Reproduce whole-command FES timings against a frozen baseline checkout.

Usage: python perf_fes.py --baseline PATH --optimized PATH --o2 PATH --out PATH
Each timed run is a new process. Output generation, imports and source hashes
are included. A warmup per case primes plotting caches; repetitions alternate
version order. No MD is launched. O2 inputs are read-only.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import numpy as np


def synthetic(root, dimension, count, hill_interval):
    root.mkdir(parents=True, exist_ok=True)
    steps = np.arange(count)
    points = np.column_stack([.75*np.sin(steps*(.017+.009*axis)+axis) for axis in range(dimension)])
    periods = [2. if axis%2 == 0 else 0. for axis in range(dimension)]
    sigma = .2
    bias = np.zeros(count)
    rows = []
    for step in range(0, count, hill_interval):
        center = points[step]
        height = .003
        quadratic = np.zeros(count-step)
        for axis, period in enumerate(periods):
            delta = points[step:, axis]-center[axis]
            scaled = np.sin(np.pi*delta/period)/(np.pi*sigma/period) if period else delta/sigma
            quadratic += scaled*scaled
        bias[step:] += height*np.exp(-.5*quadratic)
        rows.append(" ".join(map(str, [step]+list(center)+[height])))
    header = ["# METAD_TOY_BIAS 3 HILLS %d" % dimension, "# PARAMETERS .003 .025 9 1 SMOOTH_GAUSSIAN",
              "# SIGMA "+" ".join([str(sigma)]*dimension), "# PERIOD "+" ".join(map(str, periods)),
              "# FIELDS step "+" ".join("s%d" % (axis+1) for axis in range(dimension))+" height1"]
    (root/"HILLS").write_text("\n".join(header+rows)+"\n")
    with (root/"COLVAR").open("w") as stream:
        stream.write("# step "+" ".join("s%d" % (axis+1) for axis in range(dimension))+" bias\n")
        for step in steps:
            stream.write(" ".join(map(str, [int(step)]+list(points[step])+[bias[step]]))+"\n")
    return root/"COLVAR", root/"HILLS"


def compare(left, right):
    errors = {}
    for suffix in (".csv", ".weights.csv"):
        with Path(str(left)+suffix).open() as stream:
            first = list(csv.DictReader(stream))
        with Path(str(right)+suffix).open() as stream:
            second = list(csv.DictReader(stream))
        assert len(first) == len(second)
        columns = list(first[0])
        maximum = 0.
        for column in columns:
            a = [row[column] for row in first]
            b = [row[column] for row in second]
            assert [value == "" for value in a] == [value == "" for value in b]
            if column in ("step", "row", "count", "in_histogram"):
                assert a == b, column
                continue
            av = np.array([float(value) for value in a if value])
            bv = np.array([float(value) for value in b if value])
            np.testing.assert_allclose(av, bv, rtol=2e-11, atol=2e-12, err_msg=column)
            maximum = max(maximum, float(np.max(np.abs(av-bv))) if len(av) else 0.)
        errors[suffix] = maximum
    a, b = (json.loads(Path(str(path)+".json").read_text()) for path in (left, right))
    errors["weight_ess_relative"] = abs(a["weight_ess"]-b["weight_ess"])/a["weight_ess"]
    assert errors["weight_ess_relative"] < 2e-11
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "optimized", "o2", "out"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = {}
    cases["o2_wt_1d"] = [str(args.o2/"COLVAR"), "--hills", str(args.o2/"HILLS"), "--from-step", "200000",
                         "--bins", "120", "--range", "1", "1.05", "1.60",
                         "--ct-range", "1", ".85", "3", "--ct-bins", "1024", "--energy-unit", "eV"]
    for dimension, count, resolution in ((2, 5000, 128), (3, 3000, 48)):
        colvar, hills = synthetic(output/("input_%dd" % dimension), dimension, count, 10)
        options = [str(colvar), "--hills", str(hills), "--bins", "24", "--ct-bins", str(resolution)]
        for axis in range(dimension):
            options += ["--ct-range", str(axis+1), "-1", "1"]
        for axis in range(2):
            options += ["--range", str(axis+1), "-1", "1"]
        cases["synthetic_wt_%dd" % dimension] = options
    results = {"environment": {"host": platform.node(), "python": sys.version, "numpy": np.__version__,
                              "cpu": Path("/proc/cpuinfo").read_text().split("model name")[1].split("\n")[0].strip(": \t"),
                              "threads": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
                              "timing": "fresh process wall time including imports, SHA256 and CSV/PNG; warmed caches"},
               "cases": {}}
    def run(version, name, options, progress=False):
        prefix = output/(name+"_"+version)
        command = [sys.executable, str(getattr(args, version)/"tools/metadtoy.py"), "fes", "reconstruct"] + options
        command += ["--out", str(prefix), "--force"]
        if version == "optimized":
            command += ["--progress", "always" if progress else "never"]
        start = time.perf_counter()
        completed = subprocess.run(command, text=True, capture_output=True)
        elapsed = time.perf_counter()-start
        if completed.returncode:
            raise RuntimeError(completed.stderr)
        if progress:
            (output/"progress.txt").write_text(completed.stderr)
        return elapsed
    # Warm O2 first, then use its audit CSV to measure the supplied-weight path.
    for version in ("baseline", "optimized"):
        run(version, "o2_wt_1d", cases["o2_wt_1d"])
    cases["o2_supplied_weights"] = [str(output/"o2_wt_1d_baseline.weights.csv"), "--kbt", ".025851999786435",
                                   "--bins", "120", "--range", "1", "1.05", "1.60"]
    for name, options in cases.items():
        if name != "o2_wt_1d":
            for version in ("baseline", "optimized"):
                run(version, name, options)
        durations = {"baseline": [], "optimized": []}
        for repeat in range(args.repeats):
            order = ("baseline", "optimized") if repeat%2 == 0 else ("optimized", "baseline")
            for version in order:
                elapsed = run(version, name, options)
                durations[version].append(elapsed)
                print("%s %s %.3fs" % (name, version, elapsed), flush=True)
        medians = {key: statistics.median(values) for key, values in durations.items()}
        results["cases"][name] = {"seconds": durations, "median_seconds": medians,
                                  "speedup": medians["baseline"]/medians["optimized"],
                                  "comparison": compare(output/(name+"_baseline"), output/(name+"_optimized"))}
    # Paired on/off runs measure progress rendering cost with the same workload.
    progress = {"off": [], "on": []}
    for repeat in range(args.repeats):
        for enabled in ((False, True) if repeat%2 == 0 else (True, False)):
            elapsed = run("optimized", "progress_o2", cases["o2_wt_1d"], progress=enabled)
            progress["on" if enabled else "off"].append(elapsed)
            print("progress %s %.3fs" % (enabled, elapsed), flush=True)
    results["progress"] = progress
    results["source_sha256"] = {
        version: {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in (getattr(args, version)/"tools/metadtoy_cli").glob("*.py")}
        for version in ("baseline", "optimized")}
    (output/"benchmark.json").write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
