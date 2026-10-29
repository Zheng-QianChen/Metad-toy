#!/usr/bin/env python3
"""Compare MEAM, global CV bias and frozen shared bias on one Al3Sc snapshot.

Every sample launches a fresh process and reloads the same restart. NVRTC, loading
and warmup are excluded from the final LAMMPS Loop time. All modes use NPT, so CV
virial work is included each step. The two bias potentials have different semantics;
this measures computational throughput, not sampling efficiency.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import time

MODES = ("meam", "global", "shared")
BACKENDS = ("host", "kk")

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def gpu_snapshot():
    commands = [
        ["nvidia-smi", "--query-gpu=name,uuid,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw", "--format=csv,noheader"],
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"],
    ]
    return [subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()
            for command in commands]

def input_text(args, mode, steps):
    common = f"""units metal
atom_style atomic
newton on
atom_modify map array sort 5 0.0
boundary p p p
read_restart {args.assets / 'restart.start'}
mass 1 26.9815385
mass 2 44.955908
pair_style meam
pair_coeff * * {args.assets / 'Al-Sc.library.meam'} Al Sc {args.assets / 'Al-Sc.meam'} Al Sc
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
comm_modify cutoff 12.0
group Sc type 2
"""
    if mode != "meam":
        if mode == "shared":
            bias_options = f"SHARED_FROZEN {args.table} Sc PACE 0"
            expression = 'CV_dim 1 DIM_EXPR 1 "q*c*fermi(q*c,0.35,15)"'
        else:
            bias_options = "GAUSSIAN 0.05 0.02 50 PACE 0 WT 0 BIAS GRID HILLS_FILE HILLS METAD_RESTART 1"
            expression = 'CV_dim 1 DIM 1 -0.2 1.2 4096 "AVE(q*c*fermi(q*c,0.35,15))"'
        common += f"""plugin load {args.plugin}
fix bias all metad EXECUTOR GRAPH {bias_options} &
 CAL NAME SW_r6 SW_FUNC FERMI r_0 3.50 alpha -20 &
 CAL NAME SW_css SW_FUNC FERMI r_0 3.30 alpha -20 &
 CAL NAME LQ6 STEINH L 6 Sc cutoff_eps_r 1e-12 SW_FUNC_r SW_r6 &
 CAL NAME JOINT CSS Sc CHANNEL 2 SW_css 0 0.45 cutoff_eps 1e-12 &
 SYMBOL q LQ6.RAW SYMBOL c JOINT.RAW &
 {expression}
fix_modify bias energy yes
"""
    return common + f"""timestep 0.001
fix integrate all npt temp 1450 1450 0.1 iso 1 1 1.0
thermo 100000000
thermo_style custom step temp pe etotal press vol{" f_bias f_bias[1]" if mode != "meam" else ""}
thermo_modify norm no format float %.17g
run {args.warmup}
run {steps} pre no
"""

def summarize(records):
    summary = {}
    for backend in BACKENDS:
        for ranks in sorted({row["ranks"] for row in records}):
            group = {}
            for mode in MODES:
                rows = [row for row in records if row["backend"] == backend and
                        row["ranks"] == ranks and row["mode"] == mode and row["phase"] == "measure"]
                if not rows:
                    continue
                times = [row["ms_per_step"] for row in rows]
                group[mode] = dict(median_ms=statistics.median(times), samples_ms=times,
                                   min_ms=min(times), max_ms=max(times),
                                   stages_ms={key: statistics.median([row["stages_ms"][key] for row in rows])
                                              for key in rows[0]["stages_ms"]},
                                   neighbor_builds=[row["neighbor_builds"] for row in rows])
            if "meam" in group:
                baseline = group["meam"]["median_ms"]
                pair_baseline = group["meam"]["stages_ms"]["Pair"]
                for mode, value in group.items():
                    value["total_relative_to_meam_run"] = value["median_ms"] / baseline
                    value["increment_relative_to_meam_run"] = (value["median_ms"] - baseline) / baseline
                    value["total_relative_to_meam_pair"] = value["median_ms"] / pair_baseline
                    value["increment_relative_to_meam_pair"] = (value["median_ms"] - baseline) / pair_baseline
            if group:
                summary[f"{backend}_np{ranks}"] = group
    return summary

def run_one(args, backend, ranks, mode, phase, repeat, steps):
    directory = args.output / f"{phase}_{repeat}_{backend}_np{ranks}_{mode}"
    directory.mkdir()
    (directory / "input.lmp").write_text(input_text(args, mode, steps))
    # A nonzero frozen global hill exercises both forward and reverse force work.
    # Its center is 0.05 away from the initial descriptor mean (~0.008).
    if mode == "global":
        (directory / "HILLS").write_text("0 0.058 0.02 0.05\n")
    command = [str(args.mpiexec), "-np", str(ranks), str(args.lammps)]
    if backend == "kk":
        command += ["-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos",
                    "gpu/aware", "off", "neigh", "half", "newton", "on"]
    command += ["-in", "input.lmp", "-log", "log.lammps"]
    snapshot_before = gpu_snapshot()
    if snapshot_before[1]:
        raise RuntimeError("Another GPU process is active before timing: " + snapshot_before[1])
    started = time.monotonic()
    with (directory / "stdout.txt").open("w") as output:
        result = subprocess.run(command, cwd=directory, stdout=output, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, timeout=args.timeout)
    wall_seconds = time.monotonic() - started
    text = (directory / "stdout.txt").read_text(errors="replace")
    if result.returncode:
        raise RuntimeError(f"{directory}: exit {result.returncode}\n{text[-5000:]}")
    matches = list(re.finditer(r"Loop time of ([0-9.eE+-]+) on .*? for (\d+) steps with (\d+) atoms", text))
    if len(matches) != 2:
        raise RuntimeError(f"Expected warmup + measured run in {directory}")
    seconds, count, atoms = matches[-1].groups()
    if int(count) != steps or int(atoms) != 6912:
        raise RuntimeError("Unexpected workload")
    tail = text[matches[-1].end():]
    stages = {}
    for name, minimum, average, maximum, variance, percent in re.findall(
        r"^(\w+)\s*\|\s*([\d.eE+-]+)\s*\|\s*([\d.eE+-]+)\s*\|\s*([\d.eE+-]+)\s*\|\s*([\d.eE+-]+)\s*\|\s*([\d.eE+-]+)",
        tail, re.M):
        stages[name] = 1000 * float(average) / steps
    if not all(key in stages for key in ("Pair", "Modify", "Comm", "Neigh")):
        raise RuntimeError("Missing timing stages")
    thermo = []
    columns = None
    for line in text.splitlines():
        words = line.split()
        if words and words[0] == "Step" and "Temp" in words:
            columns = words
        elif columns and len(words) == len(columns):
            try:
                values = list(map(float, words))
            except ValueError:
                continue
            if not all(map(math.isfinite, values)):
                raise RuntimeError("Nonfinite thermo")
            thermo.append(dict(zip(columns, values)))
    if len(thermo) != 4:
        raise RuntimeError(f"Expected 4 boundary thermo rows, got {len(thermo)}")
    compilations = re.findall(r"NVRTC ready: compile_ms=([0-9.]+) load_ms=([0-9.]+)", text)
    if mode != "meam" and len(compilations) != 1:
        raise RuntimeError("Expected one root NVRTC compile log")
    record = dict(backend=backend, ranks=ranks, mode=mode, phase=phase, repeat=repeat,
                  steps=steps, loop_seconds=float(seconds), ms_per_step=1000*float(seconds)/steps,
                  wall_seconds=wall_seconds, stages_ms=stages, initial=thermo[0], final=thermo[-1],
                  nvrtc_ms=float(compilations[0][0]) if compilations else None,
                  neighbor_builds=int(re.search(r"Neighbor list builds =\s*(\d+)", tail).group(1)),
                  gpu_before=snapshot_before, gpu_after=gpu_snapshot(), directory=str(directory),
                  command=command)
    (directory / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"{phase} {repeat} {backend} np{ranks} {mode}: {record['ms_per_step']:.5f} ms/step "
          f"(Pair {stages['Pair']:.5f}, Modify {stages['Modify']:.5f})", flush=True)
    return record

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("lammps", "mpiexec", "plugin", "assets", "table", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--target-seconds", type=float, default=6)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    for flag in ("lammps", "mpiexec", "plugin", "assets", "table", "output"):
        setattr(args, flag, getattr(args, flag).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    assets = [args.plugin, args.table, args.lammps] + [args.assets / name for name in
              ("restart.start", "Al-Sc.library.meam", "Al-Sc.meam")]
    metadata = dict(arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                    hashes={str(path): sha256(path) for path in assets}, gpu=gpu_snapshot(),
                    cpu=subprocess.run(["lscpu"], capture_output=True, text=True, check=True).stdout,
                    omp_threads=os.environ.get("OMP_NUM_THREADS"),
                    method="Final Loop time; NPT 1450 K / 1 bar, dt=1 fs; frozen biases, warmup excluded",
                    created=time.strftime("%Y-%m-%d %H:%M:%S %z"))
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    shutil.copy2(__file__, args.output / "runner.py")
    records = []
    counts = {}
    configurations = [(backend, ranks, mode) for backend in BACKENDS for ranks in args.ranks for mode in MODES]
    for backend, ranks, mode in configurations:
        row = run_one(args, backend, ranks, mode, "pilot", 0, 50)
        records.append(row)
        # Use the same measured step count within each backend/rank group. Pilot
        # selects a window long enough for the fastest member of that group.
        (args.output / "results.json").write_text(json.dumps(records, indent=2) + "\n")
    for backend in BACKENDS:
        for ranks in args.ranks:
            fastest = min(row["ms_per_step"] for row in records if row["backend"] == backend and row["ranks"] == ranks)
            counts[(backend, ranks)] = max(200, min(3000, math.ceil(args.target_seconds * 1000 / fastest / 100)*100))
    randomizer = random.Random(20261027)
    for repeat in range(args.repeats):
        order = list(configurations)
        randomizer.shuffle(order)
        for backend, ranks, mode in order:
            records.append(run_one(args, backend, ranks, mode, "measure", repeat, counts[(backend, ranks)]))
            (args.output / "results.json").write_text(json.dumps(records, indent=2) + "\n")
            (args.output / "summary.json").write_text(json.dumps(summarize(records), indent=2) + "\n")
    metadata["gpu_after"] = gpu_snapshot()
    if any(sha256(path) != metadata["hashes"][str(path)] for path in assets):
        raise RuntimeError("An input artifact changed during the benchmark")
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("AL3SC_SHARED_PERFORMANCE_COMPLETE", flush=True)

if __name__ == "__main__":
    main()
