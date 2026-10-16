#!/usr/bin/env python3
"""Expected-error tests run in subprocesses because LAMMPS aborts the MPI communicator."""
import argparse
import shlex
import sys
import json
from pathlib import Path
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_common import plugin_path, mpi_tools, lammps_executable

parser=argparse.ArgumentParser()
parser.add_argument('--plugin',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--mpi',default=None)
parser.add_argument('--lammps',default=None)
parser.add_argument('--mpi-args',default='')
args=parser.parse_args()
args.plugin=plugin_path(args.plugin)
args.mpi,_=mpi_tools(args.plugin,args.mpi,False)
args.lammps=lammps_executable(args.plugin,args.lammps)
args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
results=[]
for name,ids,second,expected in [
    ('same_id','1 1',1.,'distinct positive atom IDs'),
    ('missing_id','1 999',1.,'exactly one owner'),
    ('coincident','1 2',0.,'zero or non-finite separation')]:
    for ranks in [1,4]:
        directory=args.output/f'{name}_np{ranks}';directory.mkdir()
        (directory/'in.lammps').write_text(f"""units lj
atom_style atomic
region cell block -5 5 -5 5 -5 5
create_box 1 cell
create_atoms 1 single 0 0 0
create_atoms 1 single {second} 0 0
mass 1 1
pair_style zero 2
pair_coeff * *
plugin load "{args.plugin.resolve()}"
fix m all metad EXECUTOR GRAPH PACE 0 CAL NAME D DISTANCE {ids} SYMBOL r D.value CV_dim 1 DIM 1 0 10 1000 r METAD_RESTART 0 WT 0
run 0
""")
        with (directory/'screen.txt').open('w') as stream:
            result=subprocess.run([str(args.mpi),*shlex.split(args.mpi_args),'-np',str(ranks),args.lammps,'-in','in.lammps'],cwd=directory,stdout=stream,stderr=subprocess.STDOUT,timeout=30)
        passed=result.returncode!=0 and expected in (directory/'screen.txt').read_text()
        results.append(dict(case=name,ranks=ranks,passed=passed,exit_code=result.returncode))
        assert passed,results[-1]
(args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results))
