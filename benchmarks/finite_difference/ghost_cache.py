#!/usr/bin/env python3
"""Exercise cached ghost rows with moving pairs, delayed demand, groups and NPT.

The native probe enumerates current-coordinate physical pairs independently of
the cached CSR. Cold run-0 checks compare CV, biased forces and virial.
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
import shlex
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'benchmarks'))
from benchmark_common import cmake_cache, executable, lammps_source, mpi_tools, plugin_path
RADIUS = 2.8 + math.log(1/1e-5-1)/8
CASES = ['crossing', 'dynamic', 'npt', 'triclinic']


def worker(args):
    import numpy as np
    from mpi4py import MPI
    from lammps import lammps
    comm = MPI.COMM_WORLD
    os.chdir(args.output)
    probe = ctypes.CDLL(str(args.probe))
    counter_type = ctypes.POINTER(ctypes.c_ulonglong)
    probe.ghost_cache_register.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_int]
    probe.ghost_cache_stats.argtypes = [ctypes.c_void_p, counter_type]
    probe.ghost_cache_verify.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_int, counter_type]
    probe.ghost_cache_error.restype = ctypes.c_char_p
    probe.fd_add_hill.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
    results = []

    def require(condition, message):
        if not comm.allreduce(bool(condition), op=MPI.LAND):
            raise RuntimeError(message)

    for case in args.cases:
        directory = args.output/case
        if comm.rank == 0: directory.mkdir()
        comm.Barrier()
        crossing = case in ('crossing', 'dynamic')
        configuration = f'''units metal
atom_style atomic
atom_modify map array sort 0 0.0
boundary p p p
processors {comm.size} 1 1
'''
        if crossing:
            # I=1 sees the periodic image of J=2; K=3 is a second-hop neighbor.
            positions = [(19.7,10,10),(1.7,10,10),(1.7+RADIUS+.35,10,10),
                         (18.2,11.1,10.2),(2,11.4,10.3),(5.2,11.5,9.8),
                         (8.5,9,10),(11,11,10),(14,9.5,10)]
            configuration += 'region box block 0 20 0 20 0 20 units box\ncreate_box 1 box\n'
            configuration += ''.join(f'create_atoms 1 single {x:.17g} {y:.17g} {z:.17g} units box\n' for x,y,z in positions)
            configuration += 'pair_style zero 2.5\npair_coeff * *\n'
        else:
            configuration += '''lattice fcc 3.0
region box block 0 4 0 4 0 4
create_box 1 box
create_atoms 1 box
pair_style lj/cut 2.5
pair_coeff * * 0.1 1.9
'''
            if case == 'triclinic':
                configuration += 'change_box all triclinic\nchange_box all xy final 0.8 xz final 0.4 remap\n'
        if not crossing:
            configuration += 'variable jitterx atom 0.023*sin(0.731*id)\nvariable jittery atom 0.019*cos(0.527*id)\nvariable jitterz atom 0.017*sin(0.319*id)\ndisplace_atoms all move v_jitterx v_jittery v_jitterz units box\n'
        configuration += f'''mass 1 30
timestep 0.001
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
variable member atom "(id%2)==(step%2)"
group moving dynamic all var member every 1
plugin load "{args.plugin}"
fix m all metad EXECUTOR GRAPH GAUSSIAN 0.1 0.1 10 PACE 0 &
 CAL NAME radial SW_FUNC FERMI r_0 2.8 alpha -8 &
 CAL NAME L STEINH L 6 {'moving' if case == 'dynamic' else 'all'} cutoff_r 3.30 SW_FUNC_r radial cutoff_eps_r 1e-5 &
 SYMBOL q L.RAW CV_dim 1 DIM 1 -2 2 2000 "AVE(q)" METAD_RESTART 0 WT 0
compute pressure all pressure NULL virial
thermo 1
thermo_style custom step f_m[1] f_m c_pressure[1] c_pressure[2] c_pressure[3] c_pressure[4] c_pressure[5] c_pressure[6] vol
thermo_modify norm no format float %.17g
'''
        if crossing:
            # Each displacement stays below skin/2, while J-K changes by 0.8 A.
            configuration += '''group J id 2
group K id 3
variable displacementJ equal -0.20*sin(2*PI*step/40.0)
variable displacementK equal 0.20*sin(2*PI*step/40.0)
variable velocityJ equal -0.20*2*PI/0.04*cos(2*PI*step/40.0)
variable velocityK equal 0.20*2*PI/0.04*cos(2*PI*step/40.0)
fix moveJ J move variable v_displacementJ NULL NULL v_velocityJ NULL NULL
fix moveK K move variable v_displacementK NULL NULL v_velocityK NULL NULL
'''
        else:
            configuration += 'velocity all create 300 98231 mom yes rot no dist gaussian\n'
            if case == 'npt':
                configuration += 'fix integrate all npt temp 300 300 0.1 iso 1 1 0.1\n'
            else:
                configuration += 'fix integrate all nve\nfix deformbox all deform 2 x erate -0.2 y erate 0.1 xy erate 0.1 remap x\n'
        if comm.rank == 0: (directory/'input.lmp').write_text(configuration+'run 0\n')
        simulation = lammps(cmdargs=['-screen','none','-log',str(directory/'log.lammps'),*shlex.split(args.lammps_args)],comm=comm)
        if args.weave_plugin:
            simulation.command(f'plugin load "{args.weave_plugin}"')
            simulation.command('weave/batch_comm')
        try:
            simulation.commands_string(configuration)
            selection = simulation.available_ids('group').index('moving')
            late_handle = probe.ghost_cache_register(simulation.lmp, RADIUS, -1) if crossing else -1
            selected_handle = probe.ghost_cache_register(simulation.lmp, RADIUS, selection) if crossing else -1
            require(not crossing or min(late_handle,selected_handle)>=0, 'extra request registration failed')
            simulation.command('run 0')
            hill = (ctypes.c_double*1)(simulation.extract_fix('m',0,1,nrow=0)+.06)
            require(probe.fd_add_hill(simulation.lmp,hill)==0, 'failed to seed fixed bias')
            simulation.command('run 0')

            def counts():
                values = (ctypes.c_ulonglong*5)()
                require(probe.ghost_cache_stats(simulation.lmp,values)==0, probe.ghost_cache_error().decode())
                return list(values)

            def verify(handle=0, group=-1):
                values = (ctypes.c_ulonglong*3)()
                status = probe.ghost_cache_verify(simulation.lmp,handle,RADIUS,group,values)
                require(status==0, probe.ghost_cache_error().decode())
                return comm.allreduce(np.array(values,dtype=np.int64),op=MPI.SUM).tolist()

            def state():
                local_count = simulation.extract_setting('nlocal')
                local = np.column_stack([simulation.numpy.extract_atom('id')[:local_count],
                                         simulation.numpy.extract_atom('f')[:local_count]])
                rows = np.concatenate(comm.allgather(local));rows=rows[np.argsort(rows[:,0])]
                return dict(cv=simulation.extract_fix('m',0,1,nrow=0), energy=simulation.extract_fix('m',0,0),
                            forces=rows[:,1:].copy(), virial=simulation.numpy.extract_compute('pressure',0,1).copy()*simulation.get_thermo('vol'))

            def cold_check(label):
                warm = state();simulation.command('run 0');cold=state()
                differences = {name:float(np.max(np.abs(np.array(warm[name])-np.array(cold[name])))) for name in warm}
                for name in warm:
                    require(np.allclose(warm[name],cold[name],atol=1e-9 if name!='virial' else 1e-5,rtol=1e-9),
                            f'{label}: stale {name}: {differences[name]}')
                return dict(label=label,maximum_absolute_errors=differences)

            initial = counts();initial_volume=simulation.get_thermo('vol')
            device_neighbors = comm.allreduce(initial[4],op=MPI.SUM)>0
            if device_neighbors:
                require(initial[1]==0 and initial[2]==0, 'device fixture used Host ghost repair or CSR upload')
            else:
                require(comm.allreduce(initial[1],op=MPI.SUM)>0, 'fixture missed ghost fallback')
            require(np.max(np.abs(state()['forces']))>1e-9, 'zero force fixture')
            checked_edges = verify();samples=[];late_edges_before=None;late_edges_after=None
            for step in range(1,41 if crossing else (82 if case == 'triclinic' else 81)):
                previous = counts()
                # fix deform needs a shared run interval to accumulate strain across chunks.
                if case == 'triclinic':
                    simulation.command('run 1 start 0 stop 81' + ('' if step == 1 else ' pre no post no'))
                else:
                    simulation.command('run 1 pre no post no')
                # On a deform step, owned coordinates have already been remapped
                # after the force evaluation. Verify when ghosts and owned atoms
                # again represent the same force-time geometry.
                if case != 'triclinic' or step % 2:
                    checked_edges = [max(a,b) for a,b in zip(checked_edges,verify())]
                if crossing:
                    verify(selected_handle,selection)
                    if step >= 10:
                        edges = verify(late_handle)
                        if step==10:late_edges_before=edges
                        if step==30:late_edges_after=edges
                current = counts()
                if device_neighbors:
                    require(current[1]==0 and current[2]==0, 'device trajectory fell back to Host neighbors')
                if crossing:
                    require(current[3]==initial[3], 'crossing fixture unexpectedly rebuilt host neighbors')
                    if step not in (1,10):
                        require(current[1]==previous[1], 'cached geometry performed another spatial search')
                samples.append(dict(step=step,counters=comm.allgather(current)))
            if crossing:
                require(late_edges_before[2]==0 and late_edges_after[2]>0, 'target ghost edge did not cross physical cutoff')
            else:
                require(abs(simulation.get_thermo('vol')-initial_volume)>1e-5, 'box did not deform')
            # Odd final step has no end-of-step deform remap, so both force
            # snapshots use the same coordinates and box.
            checks=[cold_check('trajectory')]
            if case == 'triclinic': simulation.command('unfix deformbox')
            simulation.command('atom_modify sort 1 0.0')
            simulation.command('displace_atoms all move 0.8 0 0 units box')
            before=counts();simulation.command('run 1')
            require(counts()[0]>before[0], 'migration failed to invalidate topology')
            verify();checks.append(cold_check('migration'))
            before=counts();simulation.command('neighbor 0.7 bin');simulation.command('run 0')
            require(counts()[0]>before[0], 'skin change failed to invalidate topology')
            verify();checks.append(cold_check('skin'))
            results.append(dict(case=case,ranks=comm.size,initial_counters=comm.allgather(initial),samples=samples,
                                verified_edges=checked_edges,late_edges_before=late_edges_before,late_edges_after=late_edges_after,
                                cold_checks=checks,passed=True))
            if comm.rank==0:
                (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
                print(f'PASS {case} on {comm.size} ranks',flush=True)
        finally: simulation.close()
    if comm.rank==0:(args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plugin',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--np',type=int,nargs='+',default=[1,2,4])
    parser.add_argument('--cases',nargs='+',choices=CASES,default=CASES)
    parser.add_argument('--mpi',type=Path);parser.add_argument('--mpicxx',type=Path)
    parser.add_argument('--lammps-src',type=Path);parser.add_argument('--nvcc',type=Path)
    parser.add_argument('--probe',type=Path);parser.add_argument('--worker',action='store_true')
    parser.add_argument('--lammps-args',default='')
    parser.add_argument('--weave-plugin',type=Path)
    args=parser.parse_args();args.output=args.output.resolve();args.plugin=plugin_path(args.plugin)
    if args.worker:return worker(args)
    launcher,compiler=mpi_tools(args.plugin,args.mpi,args.mpicxx)
    headers=lammps_source(args.plugin,args.lammps_src);cache=cmake_cache(args.plugin.parent)
    nvcc=executable(args.nvcc,[cache.get('CMAKE_CUDA_COMPILER','nvcc')],'CUDA compiler')
    args.output.mkdir(parents=True,exist_ok=False);args.probe=args.output/'ghost_cache_probe.so'
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMPI_MCA_btl='self,tcp',OMPI_MCA_pml='ob1')
    environment.setdefault('OMPI_CXX',cache.get('CLEAN_HOST_COMPILER','g++'))
    subprocess.run([str(nvcc),'-ccbin',str(compiler),'-std=c++17','-O2','--shared','--cudart','shared','-Xcompiler=-fPIC',
                    str(Path(__file__).with_name('ghost_cache_probe.cu')),'-I'+str(headers),'-I'+str(ROOT/'include/core'),
                    '-I'+str(ROOT/'include/utils'),str(args.plugin),'-o',str(args.probe)],env=environment,check=True)
    records=[]
    for ranks in args.np:
        directory=args.output/f'np{ranks}';directory.mkdir()
        command=[str(launcher),'-np',str(ranks),sys.executable,str(Path(__file__).resolve()),'--worker',
                 '--plugin',str(args.plugin),'--output',str(directory),'--probe',str(args.probe),'--cases',*args.cases]
        command.append('--lammps-args='+args.lammps_args)
        if args.weave_plugin: command.extend(['--weave-plugin',str(args.weave_plugin)])
        with (directory/'screen.txt').open('w') as output:
            subprocess.run(command,env=environment,stdout=output,stderr=subprocess.STDOUT,timeout=300,check=True)
        records.extend(json.loads((directory/'results.json').read_text()))
        print(f'{ranks} ranks PASS',flush=True)
    (args.output/'results.json').write_text(json.dumps(records,indent=2)+'\n')
    (args.output/'metadata.json').write_text(json.dumps(dict(plugin_sha256=hashlib.sha256(args.plugin.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),radius=RADIUS,cases=len(records)),indent=2)+'\n')

if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        traceback.print_exc()
        if '--worker' in sys.argv:
            from mpi4py import MPI
            MPI.COMM_WORLD.Abort(1)
        raise

