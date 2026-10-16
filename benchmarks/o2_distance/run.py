#!/usr/bin/env python3
"""O-mass diatomic Morse validation: run LAMMPS; reconstruction is in reconstruct.py."""
import argparse
import sys
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_common import plugin_path, lammps_executable

ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=10000000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[492845, 782531, 196837])
    parser.add_argument('--plugin', type=Path, default=None)
    parser.add_argument('--lammps', type=Path, default=None)
    args = parser.parse_args()
    try:
        args.plugin = plugin_path(args.plugin)
        args.lammps = lammps_executable(args.plugin,args.lammps)
    except ValueError as error: parser.error(str(error))
    args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=False)
    # 示例参数仅用于验证；参考模型单独保存，重建程序无需读取。
    model = dict(depth_eV=5.2, alpha_inverse_A=2.65, equilibrium_A=1.21, mass_amu=15.999,
                 description='Illustrative classical O2 Morse model, not a fitted electronic PES')
    sampling = dict(temperature_K=300., bias_factor=20., sigma_A=.015, height_eV=.02,
                    pace=100, timestep_ps=.0002, steps=args.steps, grid_min=.85, grid_max=3., bins=4300,
                    radial_measure='3D unconstrained relative coordinate; p(r) proportional to r^2 exp(-beta U(r))')
    (args.output/'reference_model.json').write_text(json.dumps(model, indent=2)+'\n')
    (args.output/'sampling.json').write_text(json.dumps(sampling, indent=2)+'\n')
    results = []
    for seed in args.seeds:
        directory = args.output/f'seed_{seed}'; directory.mkdir()
        configuration = f"""# O-mass diatomic; units are Angstrom, ps, eV, K.
units metal
atom_style atomic
boundary p p p
region cell block -10 10 -10 10 -10 10
create_box 1 cell
create_atoms 1 single -0.605 0 0
create_atoms 1 single 0.605 0 0
mass 1 {model['mass_amu']}
pair_style morse 8.0
pair_coeff * * {model['depth_eV']} {model['alpha_inverse_A']} {model['equilibrium_A']}
neighbor 0.3 bin
neigh_modify every 1 delay 0 check yes
velocity all create {sampling['temperature_K']} {seed} mom yes rot no dist gaussian
fix integrate all nve
fix bath all langevin {sampling['temperature_K']} {sampling['temperature_K']} 0.02 {seed+17} zero yes
timestep {sampling['timestep_ps']}
thermo 10000
thermo_style custom step temp pe
run 20000
reset_timestep 0
plugin load "{args.plugin.resolve()}"
fix m all metad EXECUTOR GRAPH GAUSSIAN {sampling['sigma_A']} {sampling['height_eV']} {sampling['bias_factor']} &
  PACE {sampling['pace']} TEMP {sampling['temperature_K']} &
  CAL NAME OO DISTANCE 1 2 SYMBOL r OO.value &
  CV_dim 1 DIM 1 {sampling['grid_min']} {sampling['grid_max']} {sampling['bins']} r METAD_RESTART 0 WT 1
# 每十步记录距离与偏势；HILLS 保存每次沉积的位置、实际高度和宽度。
fix sample all ave/time 10 1 10 f_m[1] f_m file trajectory.dat
thermo_style custom step temp f_m[1] f_m pe
thermo_modify format float %.16g
run {args.steps}
"""
        (directory/'in.lammps').write_text(configuration)
        started = time.monotonic()
        with (directory/'screen.txt').open('w') as output:
            result = subprocess.run([str(args.lammps), '-in', 'in.lammps'], cwd=directory,
                                    stdout=output, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'OMP_NUM_THREADS':'1'}, timeout=1200)
        record = dict(seed=seed, exit_code=result.returncode, elapsed_seconds=time.monotonic()-started)
        results.append(record); print(json.dumps(record), flush=True)
        (args.output/'runs.json').write_text(json.dumps(results, indent=2)+'\n')
        if result.returncode: raise RuntimeError((directory/'screen.txt').read_text()[-3000:])
    metadata = dict(plugin_sha256=hashlib.sha256(args.plugin.read_bytes()).hexdigest(), sampling=sampling, runs=results)
    (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')

if __name__ == '__main__': main()
