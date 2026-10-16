#!/usr/bin/env python3
"""Installed-plugin checks: smoke, finite-difference regression, Graph CTest or O2 reconstruction."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from benchmark_common import ROOT, cmake_cache, executable, plugin_path, mpi_tools, lammps_executable, lammps_source

# 同一确定性小体系覆盖所有非 ML CV；输入在测试目录自包含保存。
SMOKE_CASES = {
    'distance': ({'d':'DISTANCE 1 2'}, 'd'),
    'q': ({f'q{order}':f'STEINH Q {order} all cutoff_r 3.3 SW_FUNC_r radial cutoff_eps_r 1e-5' for order in (3,4,6)}, 'AVE(q3*q6)+AVE(q4)'),
    'local_q': ({f'l{order}':f'STEINH L {order} all cutoff_r 3.3 SW_FUNC_r radial cutoff_eps_r 1e-5' for order in (3,4,6)}, 'AVE(l3*l6)+AVE(l4)'),
    'chemical': ({'s':'STRU_FACTOR all cutoff_r 3.05 q_factor 1.7',
                  'c':'STRU_FACTOR all cutoff_r 3.05 q_factor 1.7 Chemical Chem_ctarget 1.8 Chem_sigma 1.1 Chem_map (1,1) (2,-0.7) (3,0.35)',
                  'w':'WEIGHT_CHEM all cutoff_r 3.05 Chem_ctarget 1.8 Chem_sigma 1.1 Chem_map (1,1) (2,-0.7) (3,0.35)',
                  'css':'CSS all CHANNEL 1 radial 4 3 CHANNEL 2 radial 4.5 3 CHANNEL 3 radial 4 3 cutoff_eps 1e-5'},
                 'AVE(s*c)+linear(AVE(w),0.1,0)+AVE(css)')
}

def run_command(command, directory, log, timeout):
    started = time.monotonic()
    # 测试超时终止整个进程组，回收 launcher 及其 MPI 子进程。
    with Path(log).open('w') as stream:
        process = subprocess.Popen(list(map(str,command)), cwd=directory, stdout=stream, stderr=subprocess.STDOUT,
                                   env={**os.environ,'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS','1')}, start_new_session=True)
        try: code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL); process.wait(); code = 'timeout'
    return dict(command=list(map(str,command)),exit_code=code,seconds=time.monotonic()-started,log=str(log),passed=code==0)

def fixture(directory):
    positions=[]
    for first in range(2):
        for second in range(2):
            for third in range(2):
                for dx,dy,dz in [(0,0,0),(.5,.5,0),(.5,0,.5),(0,.5,.5)]:
                    tag=len(positions)+1
                    positions.append((4.05*(first+dx)+.23+.11*math.sin(.731*tag),
                                      4.05*(second+dy)+.27+.09*math.cos(.527*tag),
                                      4.05*(third+dz)+.21+.08*math.sin(.319*tag)))
    (directory/'initial.data').write_text('Installation fixture\n\n32 atoms\n3 atom types\n\n0 8.1 xlo xhi\n0 8.1 ylo yhi\n0 8.1 zlo zhi\n\nMasses\n\n1 1\n2 1\n3 1\n\nAtoms # atomic\n\n'+''.join(f'{tag} {tag%3+1} {x:.17g} {y:.17g} {z:.17g}\n' for tag,(x,y,z) in enumerate(positions,1)))

def smoke_input(case, plugin):
    calculations,expression=SMOKE_CASES[case]
    declarations=['CAL NAME radial SW_FUNC FERMI r_0 2.8 alpha -8'] if case!='distance' else []
    declarations += [f'CAL NAME {name} {definition}' for name,definition in calculations.items()]
    symbols=' '.join(f'SYMBOL {name} {name}.RAW' for name in calculations)
    observations=' '.join(['$(f_m[1]:%.17g)','$(f_m:%.17g)']+[f'$(c_pressure[{index}]*vol:%.17g)' for index in range(1,7)])
    return f"""units lj
atom_style atomic
atom_modify map array sort 0 0.0
boundary p p p
read_data initial.data
pair_style zero 2.5
pair_coeff * *
neighbor 0.35 bin
neigh_modify every 1 delay 0 check yes
plugin load "{plugin}"
fix integrate all nve
fix m all metad EXECUTOR GRAPH GAUSSIAN .1 .1 10 PACE 1 &
  {' '.join(declarations)} {symbols} &
  CV_dim 1 DIM 1 -100 100 20000 "{expression}" METAD_RESTART 0 WT 0
compute pressure all pressure NULL virial
thermo 1
thermo_style custom step f_m[1] f_m c_pressure[1] c_pressure[2] c_pressure[3] c_pressure[4] c_pressure[5] c_pressure[6]
thermo_modify norm no format float %.17g
# 沉积一次偏势，再移动一颗原子，产生非零反向梯度。
run 1
group moved id 1
displace_atoms moved move .03 .02 -.01 units box
run 0
print "SMOKE_STATE {observations}"
write_dump all custom state.dump id x y z fx fy fz modify sort id format float %.17g
"""

def smoke(args):
    records=[]
    for case in SMOKE_CASES:
        reference=None
        for ranks in args.np:
            directory=args.output/f'{case}_np{ranks}';directory.mkdir()
            fixture(directory);(directory/'in.lammps').write_text(smoke_input(case,args.plugin))
            command=[args.lammps,'-in','in.lammps']
            if ranks>1:command=[args.mpi,*shlex.split(args.mpi_args),'-np',str(ranks),*command]
            record=run_command(command,directory,directory/'screen.txt',args.timeout)
            record.update(case=case,ranks=ranks)
            if record['passed']:
                try:
                    lines=(directory/'screen.txt').read_text().splitlines()
                    values=[float(x) for x in next(line for line in lines if line.startswith('SMOKE_STATE ')).split()[1:]]
                    dump=(directory/'state.dump').read_text().splitlines()
                    start=next(index for index,line in enumerate(dump) if line.startswith('ITEM: ATOMS'))+1
                    rows=sorted([[float(x) for x in line.split()] for line in dump[start:]],key=lambda row:row[0])
                    forces=[value for row in rows for value in row[4:7]]
                    if len(values)!=8 or len(rows)!=32 or not all(math.isfinite(value) for value in values+forces):
                        raise ValueError('Unexpected shape or non-finite CV/energy/force/virial')
                    if max(map(abs,forces))<1e-10:raise ValueError('No nonzero bias force: backward path was not exercised')
                    if case=='distance':
                        delta=[rows[1][axis]-rows[0][axis] for axis in range(1,4)]
                        expected=math.sqrt(sum((x-round(x/8.1)*8.1)**2 for x in delta))
                        record['distance_error']=abs(values[0]-expected)
                        if record['distance_error']>1e-10:raise ValueError('DISTANCE differs from coordinates')
                    state=values+forces
                    if reference is None:reference=state
                    record['mpi_max_error']=max(abs(a-b) for a,b in zip(reference,state))
                    if record['mpi_max_error']>1e-9:raise ValueError('MPI state differs from first rank configuration')
                    record['observables']=values;record['force_peak']=max(map(abs,forces))
                except (OSError,ValueError,StopIteration) as error:record.update(passed=False,error=str(error))
            records.append(record)
            print(f"{case} np={ranks}: {'PASS' if record['passed'] else 'FAIL'}",flush=True)
    return records

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite',choices=['smoke','regression','graph','o2'])
    parser.add_argument('--build-dir',type=Path,default=ROOT/'build')
    parser.add_argument('--plugin',type=Path)
    parser.add_argument('--lammps')
    parser.add_argument('--mpi')
    parser.add_argument('--mpicxx')
    parser.add_argument('--mpi-args',default='',help='Extra launcher arguments, e.g. --mpi-args="--bind-to none"')
    parser.add_argument('--lammps-src',type=Path)
    parser.add_argument('--graph-build-dir',type=Path,default=ROOT/'build/graph')
    parser.add_argument('--ctest',default='ctest')
    parser.add_argument('--np',type=int,nargs='+',default=[1])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--timeout',type=int,default=300,help='Per smoke run timeout in seconds')
    args=parser.parse_args();args.np=sorted(set(args.np))
    if not args.np or min(args.np)<1:parser.error('--np requires positive ranks')
    args.output=(args.output or args.build_dir/'benchmarks'/f"{args.suite}-{datetime.now().strftime('%Y%m%d-%H%M%S')}").resolve()
    try:
        args.plugin=plugin_path(args.plugin,args.build_dir) if args.suite!='graph' else None
        if args.suite in ('smoke','o2'):args.lammps=lammps_executable(args.plugin,args.lammps)
        if args.suite=='regression' or (args.suite=='smoke' and max(args.np)>1):
            args.mpi,args.mpicxx=mpi_tools(args.plugin,args.mpi,args.mpicxx if args.suite=='regression' else False)
        if args.suite=='regression':
            args.lammps_src=lammps_source(args.plugin,args.lammps_src)
            for module in ['numpy','mpi4py','lammps']:__import__(module)
        if args.suite=='o2':
            for module in ['numpy','matplotlib']:__import__(module)
        if args.suite=='graph':
            args.ctest=executable(args.ctest,[],'CTest')
            if not (args.graph_build_dir/'CTestTestfile.cmake').is_file():raise ValueError('Configure and build standalone Graph first; see benchmarks/README.md')
        args.output.mkdir(parents=True,exist_ok=False)
    except (ValueError,OSError,ImportError) as error:parser.error(str(error))
    if args.suite=='smoke':records=smoke(args)
    elif args.suite=='graph':
        records=[run_command([args.ctest,'--test-dir',args.graph_build_dir.resolve(),'--output-on-failure'],ROOT,args.output/'ctest.log',1200)]
    elif args.suite=='regression':
        command=[sys.executable,ROOT/'benchmarks/finite_difference/run.py','--output',args.output/'finite-difference',
                 '--plugin',args.plugin,'--mpi',args.mpi,'--mpicxx',args.mpicxx,'--mpi-args='+args.mpi_args,
                 '--lammps-src',args.lammps_src,'--np',*map(str,args.np)]
        records=[run_command(command,ROOT,args.output/'regression.log',max(1200,600*len(args.np)))]
    else:
        records=[]
        commands=[['run.py','--output',args.output/'o2','--plugin',args.plugin,'--lammps',args.lammps],
                  ['reconstruct.py',args.output/'o2'],['validate.py',args.output/'o2']]
        for index,(script,*options) in enumerate(commands):
            record=run_command([sys.executable,ROOT/'benchmarks/o2_distance'/script,*options],ROOT,args.output/f'o2-{index}.log',1800)
            records.append(record)
            if not record['passed']:break
    report=dict(suite=args.suite,utc=datetime.now(timezone.utc).isoformat(),python=sys.executable,
                plugin=str(args.plugin) if args.plugin else None,
                plugin_sha256=hashlib.sha256(args.plugin.read_bytes()).hexdigest() if args.plugin else None,
                cmake={key:value for key,value in cmake_cache(args.plugin.parent).items() if key in ['CMAKE_BUILD_TYPE','CMAKE_CUDA_ARCHITECTURES','CMAKE_CUDA_COMPILER','LAMMPS_SOURCE_DIR','LAMMPS_LIB','MPI_HOME','ENABLE_DEBUG','USE_ML_CVS']} if args.plugin else {},passed=all(row['passed'] for row in records),runs=records)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f"{'PASS' if report['passed'] else 'FAIL'}: {args.output}/summary.json")
    return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
