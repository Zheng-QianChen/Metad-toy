#!/usr/bin/env python3
"""真实 LAMMPS Half/Full 对照：独立生成周期夹具，覆盖 Newton、组筛选、镜像及 reverse。"""
import argparse
import ctypes
import hashlib
import json
import os
import shlex
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks'))
from benchmark_common import cmake_cache, executable, lammps_source, mpi_tools, plugin_path


def worker(args):
    from mpi4py import MPI
    from lammps import lammps
    comm=MPI.COMM_WORLD
    os.chdir(args.output)
    probe=ctypes.CDLL(str(args.probe))
    probe.engine_neighbors_register.argtypes=[ctypes.c_void_p]
    probe.engine_neighbors_check.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_double)]
    probe.engine_neighbors_error.restype=ctypes.c_char_p
    results=[]
    for geometry in ('orthogonal','triclinic','self-images'):
        length=2.8 if geometry=='self-images' else 5.2
        for newton in ('on','off'):
            simulation=lammps(cmdargs=['-screen','none','-log',f'log.{geometry}.{newton}',*shlex.split(args.lammps_args)],comm=comm)
            try:
                commands=['units lj','atom_style atomic','atom_modify map array sort 0 0.0',
                          'boundary p p p',f'newton {newton}',
                          f'region box block 0 {length} 0 {length} 0 {length}', 'create_box 2 box','mass * 1']
                # 12 个位置无重合；小盒长度低于 cutoff，明确包含同 ID 的周期自镜像。
                for atom in range(12):
                    x=length*(.13+.29*(atom%3));y=length*(.17+.43*((atom//3)%2));z=length*(.21+.41*(atom//6))
                    commands.append(f'create_atoms {1+atom%2} single {x} {y} {z} units box')
                if geometry=='triclinic':commands+=['change_box all triclinic','change_box all xy final .6 xz final .3 yz final .2 remap units box']
                commands+=['pair_style zero 2.5','pair_coeff * *','neighbor .35 bin','neigh_modify every 1 delay 0 check yes',
                           'group first id 1:6','group second id 7:12','group overlap id 4:9','group empty empty',
                           f'plugin load "{args.plugin}"',
                           'fix m all metad EXECUTOR GRAPH GAUSSIAN 0.1 0.1 10 PACE 0 '
                           'CAL NAME structure STRU_FACTOR all cutoff_r 3.05 q_factor 1.7 '
                           'SYMBOL structure structure.AVE CV_dim 1 DIM 1 -20 20 2000 "structure" METAD_RESTART 0 WT 0',
                           'compute pressure all pressure NULL virial','thermo 1',
                           'thermo_style custom step f_m[1] c_pressure[1] c_pressure[4]','thermo_modify norm no']
                simulation.commands_list(commands)
                if probe.engine_neighbors_register(simulation.lmp):raise RuntimeError(probe.engine_neighbors_error().decode())
                for stage in ('initial','moved','groups-changed'):
                    if stage=='moved':
                        simulation.commands_list(['displace_atoms all move .13 -.17 .09 units box','atom_modify sort 1 .8'])
                    elif stage=='groups-changed':
                        simulation.commands_list(['group first clear','group first id 2:7','group overlap clear','group overlap id 5:10'])
                    simulation.command('run 0')
                    error=ctypes.c_double()
                    if probe.engine_neighbors_check(simulation.lmp,ctypes.byref(error)):
                        raise RuntimeError(f'{geometry}/{newton}/{stage}: '+probe.engine_neighbors_error().decode())
                    results.append(dict(ranks=comm.size,geometry=geometry,newton=newton,stage=stage,
                                        selection_cases=9,max_error=error.value,passed=True))
            finally:simulation.close()
    simulation.lib.lammps_kokkos_finalize()
    if comm.rank==0:
        (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        print(f'PASS: {len(results)*9} selection/configuration checks on {comm.size} ranks',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plugin',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--np',type=int,nargs='+',default=[1,2,4])
    for name in ('mpi','mpicxx','lammps-src','nvcc','probe'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--lammps-args',default='')
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args();args.plugin=plugin_path(args.plugin);args.output=args.output.resolve()
    if args.worker:return worker(args)
    if min(args.np)<1:parser.error('--np must be positive')
    launcher,compiler=mpi_tools(args.plugin,args.mpi,args.mpicxx);headers=lammps_source(args.plugin,args.lammps_src)
    cache=cmake_cache(args.plugin.parent);nvcc=executable(args.nvcc,[cache.get('CMAKE_CUDA_COMPILER','nvcc')],'CUDA compiler')
    args.output.mkdir(parents=True,exist_ok=False)
    supplied_probe = args.probe is not None
    args.probe = args.probe.resolve() if supplied_probe else args.output/'engine_neighbors_probe.so'
    environment=os.environ.copy();environment.setdefault('OMPI_CXX',cache.get('CLEAN_HOST_COMPILER','g++'));environment.setdefault('OMP_NUM_THREADS','1')
    if not supplied_probe:
        subprocess.run([str(nvcc),'-ccbin',str(compiler),'-std=c++17','--shared','--cudart','shared','-Xcompiler=-fPIC',
                        str(Path(__file__).with_name('engine_neighbors_probe.cu')),'-I'+str(headers),
                        '-I'+str(ROOT/'include/core'),'-I'+str(ROOT/'include/utils'),str(args.plugin),'-o',str(args.probe)],
                       env=environment,check=True)
    results=[]
    for ranks in args.np:
        directory=args.output/f'np{ranks}';directory.mkdir()
        command=[str(launcher),'-np',str(ranks),sys.executable,str(Path(__file__).resolve()),'--worker',
                 '--plugin',str(args.plugin),'--output',str(directory),'--probe',str(args.probe),'--lammps-args='+args.lammps_args]
        with (directory/'screen.txt').open('w') as output:
            subprocess.run(command,env=environment,stdout=output,stderr=subprocess.STDOUT,timeout=240,check=True)
        rows=json.loads((directory/'results.json').read_text());results.extend(rows)
        print(f'{ranks} ranks: {len(rows)*9} selection/configuration checks PASS',flush=True)
    (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    (args.output/'metadata.json').write_text(json.dumps(dict(plugin_sha256=hashlib.sha256(args.plugin.read_bytes()).hexdigest(),
        selection_checks=sum(row['selection_cases'] for row in results),maximum_error=max(row['max_error'] for row in results)),indent=2)+'\n')


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        traceback.print_exc()
        if '--worker' in sys.argv:
            from mpi4py import MPI
            MPI.COMM_WORLD.Abort(1)
        raise
