#!/usr/bin/env python3
"""Compare plugin versions on identical LAMMPS trajectories; time only warmed run blocks."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shlex
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'benchmarks'))
from benchmark_common import lammps_executable, mpi_tools


def initial_data(path, cells):
    # 同一文件定义全局 ID/类型；rank 分解和版本变化均复用这些物理初态。
    atoms = []
    for x in range(cells):
        for y in range(cells):
            for z in range(cells):
                for basis, (dx, dy, dz) in enumerate([(0,0,0), (.5,.5,0), (.5,0,.5), (0,.5,.5)]):
                    atom_type = 2 if basis == 0 else 3 if basis == 3 else 1
                    atoms.append(f'{len(atoms)+1} {atom_type} {4.05*(x+dx):.17g} {4.05*(y+dy):.17g} {4.05*(z+dz):.17g}')
    length = 4.05*cells
    path.write_text(f'Runtime comparison\n\n{len(atoms)} atoms\n3 atom types\n\n'
                    + ''.join(f'0 {length} {axis}lo {axis}hi\n' for axis in 'xyz')
                    + '\nMasses\n\n1 30\n2 30\n3 30\n\nAtoms # atomic\n\n'+'\n'.join(atoms)+'\n')


def input_text(plugin, data, case, steps, warmup, validation=False, neighbor_rebuild='displacement', pair_cutoff=4.0):
    declarations = [
        'CAL NAME S1 STRU_FACTOR all cutoff_r 3.30 q_factor 1.7 d_block_size 128',
        'CAL NAME S2 STRU_FACTOR all cutoff_r 3.30 q_factor 2.3 d_block_size 128',
        'SYMBOL s1 S1.RAW SYMBOL s2 S2.RAW',
    ]
    expressions = ['AVE(s1)*AVE(s2)', 'AVE(s1)+AVE(s2)']
    if case == 'mixed':
        # Stru_factor 共用环境；CSS 和球谐环境提供可独立排队的工作。
        declarations += [
            'CAL NAME radial SW_FUNC FERMI r_0 2.8 alpha -8',
            'CAL NAME C CSS all CHANNEL 1 radial 4.0 3.0 CHANNEL 2 radial 4.5 3.0 CHANNEL 3 radial 4.0 3.0 cutoff_eps 1e-5',
            'CAL NAME Q STEINH Q 6 all cutoff_r 3.30 SW_FUNC_r radial cutoff_eps_r 1e-5',
            'CAL NAME L STEINH L 6 all cutoff_r 3.30 SW_FUNC_r radial cutoff_eps_r 1e-5',
            'SYMBOL c C.RAW SYMBOL q Q.RAW SYMBOL l L.RAW',
        ]
        expressions = ['AVE(s1)*AVE(s2)+linear(AVE(c),0.2,0)+linear(AVE(q),0.1,0)',
                       'AVE(s1)+AVE(s2)+linear(AVE(l),0.2,0)']
    elif case in ['css', 'q6', 'localq6']:
        # 独立 CV 复用 mixed 的径向参数，辅助区分各计算链的尺度相关成本。
        declarations = ['CAL NAME radial SW_FUNC FERMI r_0 2.8 alpha -8']
        if case == 'css':
            declarations.append('CAL NAME X CSS all CHANNEL 1 radial 4.0 3.0 CHANNEL 2 radial 4.5 3.0 CHANNEL 3 radial 4.0 3.0 cutoff_eps 1e-5')
        else:
            kind = 'Q' if case == 'q6' else 'L'
            declarations.append(f'CAL NAME X STEINH {kind} 6 all cutoff_r 3.30 SW_FUNC_r radial cutoff_eps_r 1e-5')
        declarations.append('SYMBOL x X.RAW')
        expressions = ['AVE(x)', 'linear(AVE(x),0.2,0)']
    declarations_text = ' &\n    '.join(declarations)
    # 每步重建用于观察候选表反复失效的成本；默认由位移触发重建。
    neighbor_check = 'no' if neighbor_rebuild == 'every-step' else 'yes'
    text = f'''units metal
atom_style atomic
boundary p p p
read_data "{data}"
variable dx0 atom 0.018*sin(0.731*id)
variable dy0 atom 0.016*cos(0.527*id)
variable dz0 atom 0.014*sin(0.319*id+0.2)
displace_atoms all move v_dx0 v_dy0 v_dz0 units box
pair_style zero {pair_cutoff:.17g}
pair_coeff * *
neighbor 0.35 bin
neigh_modify every 1 delay 0 check {neighbor_check}
plugin load "{plugin}"
reset_timestep 99999
fix m all metad EXECUTOR GRAPH GAUSSIAN 0.10 0.10 10.0 &
    PACE 100000 &
    {declarations_text} &
    CV_dim 2 DIM 1 -5 5 500 "{expressions[0]}" DIM 2 -5 5 500 "{expressions[1]}" &
    METAD_RESTART 0 WT 0
velocity all set 0 0 0
fix integrate all nve
fix hold all setforce 0 0 0
compute bias_pressure all pressure NULL virial
thermo 100000
thermo_style custom step atoms f_m[1] f_m[2] f_m c_bias_pressure[1] c_bias_pressure[2] c_bias_pressure[3] c_bias_pressure[4] c_bias_pressure[5] c_bias_pressure[6]
thermo_modify norm no format float %.17g
run 1
unfix hold
variable dx1 atom 0.07*sin(0.413*id+0.1)
variable dy1 atom 0.06*cos(0.293*id+0.4)
variable dz1 atom 0.05*sin(0.619*id+0.7)
displace_atoms all move v_dx1 v_dy1 v_dz1 units box
reset_timestep 100001
unfix integrate
fix integrate all move linear 0.10 0.05 0.025 units box
'''
    if validation:
        return text+'''dump state all custom 1 atom_state.dump id x y z fx fy fz
dump_modify state sort id format float %.17g
run 1
'''
    # 计时段只含平移轨迹：所有版本执行相同、非零的偏置力工作，避免 NVE 分叉。
    return text+f'run {warmup}\nrun {steps}\n'


def execute(args, variant, case, cells, ranks, tag, steps, validation=False):
    directory = args.output/f'{case}_{cells}_np{ranks}_{variant}_{tag}'
    directory.mkdir()
    plugin = args.variants[variant]
    configuration = input_text(plugin, args.output/f'initial_{cells}.data', case, steps, args.warmup,
                               validation, args.neighbor_rebuild, args.pair_cutoff)
    (directory/'in.lammps').write_text(configuration)
    command = [str(args.mpi), *shlex.split(args.mpi_args), '-np', str(ranks), str(args.lmp), '-in', 'in.lammps', '-log', 'log.lammps']
    environment = dict(os.environ, OMP_NUM_THREADS='1')
    started = time.perf_counter()
    with (directory/'screen.txt').open('w') as output:
        result = subprocess.run(command, cwd=directory, env=environment, stdout=output, stderr=subprocess.STDOUT, timeout=args.timeout)
    wall = time.perf_counter()-started
    screen = (directory/'screen.txt').read_text(errors='replace')
    if result.returncode:
        raise RuntimeError(f'{directory}: exit {result.returncode}\n{screen[-4000:]}')
    loops = re.findall(r'Loop time of ([0-9.eE+-]+) on .*? for (\d+) steps with (\d+) atoms', screen)
    if not loops: raise RuntimeError(f'{directory}: no Loop time')
    seconds, measured_steps, atoms = loops[-1]
    compile_times = re.findall(r'NVRTC ready: compile_ms=([0-9.]+) load_ms=([0-9.]+)', screen)
    if len(compile_times) != 1: raise RuntimeError(f'{directory}: expected one NVRTC compilation, found {len(compile_times)}')
    if hashlib.sha256(plugin.read_bytes()).hexdigest() != args.hashes[variant]:
        raise RuntimeError(f'{variant}: plugin changed during measurement')
    thermo = []
    in_thermo = False
    for line in screen.splitlines():
        line = line.strip()
        if line.startswith('Step '): in_thermo = True; continue
        if line.startswith('Loop time'): in_thermo = False
        if in_thermo:
            try: row = list(map(float, line.split()))
            except ValueError: continue
            if len(row) == 11: thermo.append(row)
    if not thermo or not all(math.isfinite(value) for value in thermo[-1]):
        raise RuntimeError(f'{directory}: missing/nonfinite thermo output')
    # LAMMPS 自身的阶段计时辅助识别 MPI/Modify 比例；它不是 GPU kernel profiler。
    timing = {}
    for name, minimum, average, maximum, variance, percent in re.findall(
            r'^(\w+)\s*\|\s*([\d.eE+-]+)\s*\|\s*([\d.eE+-]+)\s*\|\s*([\d.eE+-]+)\s*\|\s*([\d.eE+-]+)\s*\|\s*([\d.eE+-]+)', screen, re.M):
        timing[name] = dict(seconds=float(average), percent=float(percent))
    neighbor_builds = re.findall(r'Neighbor list builds =\s*(\d+)', screen)
    record = dict(variant=variant, case=case, cells=cells, ranks=ranks, tag=tag, atoms=int(atoms),
                  neighbor_rebuild=args.neighbor_rebuild,
                  pair_cutoff=args.pair_cutoff,
                  neighbor_builds=int(neighbor_builds[-1]) if neighbor_builds else None,
                  steps=int(measured_steps), loop_seconds=float(seconds), us_per_step=1e6*float(seconds)/int(measured_steps),
                  process_wall_seconds=wall, nvrtc_ms=float(compile_times[0][0]), load_ms=float(compile_times[0][1]),
                  observables=thermo[-1][2:], lammps_timing=timing, directory=str(directory))
    (directory/'result.json').write_text(json.dumps(record, indent=2))
    return record


def last_dump(directory):
    lines = (Path(directory)/'atom_state.dump').read_text().splitlines()
    offset = max(index for index, line in enumerate(lines) if line.startswith('ITEM: ATOMS'))
    return {int(row[0]): list(map(float, row[1:])) for row in (line.split() for line in lines[offset+1:])}


def compare(reference, result):
    first, second = last_dump(reference['directory']), last_dump(result['directory'])
    if first.keys() != second.keys(): raise RuntimeError('atom IDs differ')
    coordinate_error = max(abs(first[atom][column]-second[atom][column]) for atom in first for column in range(3))
    force_error = max(abs(first[atom][column]-second[atom][column]) for atom in first for column in range(3,6))
    maximum_force = max(abs(value) for row in first.values() for value in row[3:])
    observable_error = max(abs(a-b) for a,b in zip(reference['observables'][:3], result['observables'][:3]))
    pressure_error = max(abs(a-b) for a,b in zip(reference['observables'][3:], result['observables'][3:]))
    passed = (coordinate_error < 1e-11 and force_error < 1e-10 and observable_error < 1e-10
              and pressure_error < 1e-8 and maximum_force > 1e-12)
    return dict(variant=result['variant'], case=result['case'], cells=result['cells'], ranks=result['ranks'],
                passed=passed, coordinate_error=coordinate_error, force_error=force_error,
                cv_energy_error=observable_error, pressure_error=pressure_error, maximum_force=maximum_force)


def summarize(rows, variants):
    summaries = []
    for case, cells, ranks in sorted({(row['case'],row['cells'],row['ranks']) for row in rows}):
        samples = {variant:[row['us_per_step'] for row in rows if (row['case'],row['cells'],row['ranks'],row['variant']) == (case,cells,ranks,variant)] for variant in variants}
        medians = {variant:statistics.median(values) for variant,values in samples.items() if values}
        if len(medians) != len(variants): continue
        summaries.append(dict(case=case, atoms=4*cells**3, cells=cells, ranks=ranks, median_us=medians,
                              ranges_us={variant:[min(values),max(values)] for variant,values in samples.items()},
                              samples_us=samples,
                              speedup_vs_first={variant:medians[variants[0]]/value for variant,value in medians.items()},
                              last_vs_previous=medians[variants[-2]]/medians[variants[-1]]))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', action='append', required=True, help='NAME=/absolute/path/plugin.so; first is baseline, last is current')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lmp', type=Path)
    parser.add_argument('--mpi', type=Path)
    parser.add_argument('--mpi-args', default='', help='launcher options, e.g. --mpi-args="--bind-to none" for OpenMPI')
    parser.add_argument('--np', nargs='+', type=int, default=[1,2])
    parser.add_argument('--cells', nargs='+', type=int, default=[4,16])
    parser.add_argument('--cases', nargs='+', choices=['shared','mixed','css','q6','localq6'], default=['shared','mixed'])
    parser.add_argument('--neighbor-rebuild', choices=['displacement','every-step'], default='displacement',
                        help='normal displacement checks, or force a neighbor rebuild on every timestep')
    parser.add_argument('--pair-cutoff', type=float, default=4.0,
                        help='pair zero cutoff; CV cutoffs and physical coordinates remain fixed')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--steps', type=int, help='fixed common measured step count; otherwise calibrate with last variant')
    parser.add_argument('--target-seconds', type=float, default=3)
    parser.add_argument('--max-steps', type=int, default=20000)
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()
    args.variants = {}
    for specification in args.variant:
        name, path = specification.split('=',1)
        if not re.fullmatch(r'[a-zA-Z0-9_-]+',name) or name in args.variants: parser.error('variant names must be unique simple identifiers')
        args.variants[name] = Path(path).resolve(strict=True)
    variants = list(args.variants)
    if len(variants) < 2: parser.error('provide at least two variants')
    if args.steps is not None and args.steps <= 0: parser.error('--steps must be positive')
    if not math.isfinite(args.pair_cutoff) or args.pair_cutoff <= 0: parser.error('--pair-cutoff must be finite and positive')
    if min(args.np+args.cells+[args.repeats,args.warmup,args.max_steps]) <= 0 or args.target_seconds <= 0: parser.error('counts and target seconds must be positive')
    if args.warmup+max(args.steps or 0,args.max_steps) >= 99999: parser.error('keep all measurement runs before the next hill deposition')
    args.output = args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    args.lmp = lammps_executable(args.variants[variants[-1]],args.lmp)
    args.mpi,_ = mpi_tools(args.variants[variants[-1]],args.mpi,False)
    args.hashes = {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in args.variants.items()}
    metadata = dict(plugins={name:dict(path=str(path),sha256=args.hashes[name]) for name,path in args.variants.items()},
                    lmp=str(args.lmp),mpi=str(args.mpi),repeats=args.repeats,warmup=args.warmup,target_seconds=args.target_seconds,
                    cpu_threads=1,mpi_args=args.mpi_args,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    neighbor_rebuild=args.neighbor_rebuild,
                    pair_cutoff=args.pair_cutoff,
                    timing='LAMMPS final Loop time, warmed fixed translation, shared GPU',
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        metadata['gpu'] = subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.draw','--format=csv'],text=True)
    except (OSError,subprocess.CalledProcessError): metadata['gpu'] = 'unavailable'
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    # 将实际执行的入口一并保存，后续增加场景后仍能核对旧样本的源码 hash。
    (args.output/'benchmark_source.py').write_bytes(Path(__file__).read_bytes())
    for cells in args.cells: initial_data(args.output/f'initial_{cells}.data',cells)
    validations, rows, calibrations = [], [], []
    randomizer = random.Random(20261022)
    for case in args.cases:
        for cells in args.cells:
            for ranks in args.np:
                reference = None
                for variant in variants:
                    record = execute(args,variant,case,cells,ranks,'validate',1,True)
                    if reference is None: reference = record
                    validation = compare(reference,record);validations.append(validation)
                    (args.output/'validation.json').write_text(json.dumps(validations,indent=2))
                    if not validation['passed']: raise RuntimeError(f'validation failed: {validation}')
                print('VALIDATED',case,cells,ranks,flush=True)
                if args.steps:
                    steps = args.steps
                else:
                    pilot = execute(args,variants[-1],case,cells,ranks,'pilot',200)
                    steps = min(args.max_steps,max(300,math.ceil(args.target_seconds*1e6/pilot['us_per_step']/100)*100))
                    pilot['selected_steps'] = steps;calibrations.append(pilot)
                    (args.output/'calibration.json').write_text(json.dumps(calibrations,indent=2))
                for repeat in range(args.repeats):
                    order = variants.copy();randomizer.shuffle(order)
                    for variant in order:
                        record = execute(args,variant,case,cells,ranks,f'r{repeat}',steps)
                        record['repeat'] = repeat;rows.append(record)
                        (args.output/'results.json').write_text(json.dumps(rows,indent=2))
                        (args.output/'summary.json').write_text(json.dumps(summarize(rows,variants),indent=2))
                        print(case,cells,ranks,variant,repeat,round(record['us_per_step'],3),'us/step',flush=True)
    print('PERFORMANCE_COMPARISON_PASS',flush=True)


if __name__ == '__main__':
    main()
