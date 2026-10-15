#!/usr/bin/env python3
"""编译独立 Stru_factor 布局实验，将构建产物和原始结果写入 build。"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def source_hashes():
    paths = [path for directory in ("src", "include", "cmake")
             for path in (ROOT/directory).rglob("*") if path.is_file()]
    paths.append(ROOT/"CMakeLists.txt")
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT/"build/strufactor-pair")
    parser.add_argument("--rounds", type=int, default=9)
    arguments = parser.parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    before = source_hashes()
    compiler = os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc")
    lammps = Path(os.environ.get("LAMMPS_SOURCE_DIR", "/home/zqc/apps/lammps/lammps-stable_29Aug2024/src"))
    mpi = Path(os.environ.get("MPI_HOME", "/home/zqc/apps/deepmd"))
    command = [compiler, "-O3", "-std=c++17", "-arch=sm_89", "-lineinfo", "-ccbin", "/usr/bin/g++",
               "-I"+str(ROOT/"include/utils"), "-I"+str(ROOT/"include/CVsets"),
               "-I"+str(lammps), "-I"+str(mpi/"include"), str(HERE/"benchmark.cu"),
               "-L"+str(mpi/"lib"), "-Xlinker", "-rpath", "-Xlinker", str(mpi/"lib"),
               "-lmpi", "-o", str(output/"benchmark")]
    print("Building independent benchmark", flush=True)
    with (output/"build.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        print((output/"build.log").read_text())
        raise SystemExit(result.returncode)
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="1")
    print("Running benchmark; progress: "+str(output/"run.log"), flush=True)
    with (output/"run.log").open("w") as log:
        result = subprocess.run([str(output/"benchmark"), str(output/"results.csv"), str(arguments.rounds)],
                                env=environment, stdout=log, stderr=subprocess.STDOUT)
    print((output/"run.log").read_text(), flush=True)
    after = source_hashes()
    if before != after:
        raise RuntimeError("production files changed during experiment")
    if result.returncode:
        raise SystemExit(result.returncode)
    rows = list(csv.DictReader((output/"results.csv").open()))
    metadata = {"command": command, "rounds": arguments.rounds,
                "timing": "CUDA events, 32-step CUDA Graph replay; clears and virial included",
                "production_unchanged": True, "production_sha256": before,
                "benchmark_sha256": hashlib.sha256((HERE/"benchmark.cu").read_bytes()).hexdigest(),
                "gpu": subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=name,driver_version,temperature.gpu,utilization.gpu", "--format=csv,noheader"], text=True).strip(),
                "compiler": subprocess.check_output([compiler,"--version"],text=True), "results": rows}
    (output/"results.json").write_text(json.dumps(metadata, indent=2))
    print("Production source hashes unchanged", flush=True)


if __name__ == "__main__":
    main()
