#!/usr/bin/env python3
"""Validate C++ high-dimensional deposition through LAMMPS, Graph and restart.

The Python prototype supplies independent kernel/deposition reference values.
All timestep computation is performed by the compiled plugin.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'benchmarks'))
from benchmark_common import mpi_tools, plugin_path


def worker(args):
    import numpy as np
    from mpi4py import MPI
    from lammps import lammps
    sys.path.insert(0, str(ROOT/'prototypes/extension_lab/highdim'))
    from bias import HillBias
    comm = MPI.COMM_WORLD
    os.chdir(args.output)

    def require(condition, message):
        if not comm.allreduce(bool(condition), op=MPI.LAND):
            raise RuntimeError(message)

    class PlumedPBComponent:
        """Independent scalar reference of PLUMED v2.10 PBMetaD evaluateGaussian."""
        def __init__(self, width, period):
            self.width, self.period = width, period
            self.hills = []
        def add(self, center, height):
            self.hills.append((float(center[0]), float(height)))
        def evaluate(self, point):
            energy, derivative = 0.0, 0.0
            for center, height in self.hills:
                displacement = float(point[0])-center
                if self.period:
                    displacement -= self.period*np.rint(displacement/self.period)
                scaled = displacement/self.width
                if .5*scaled*scaled < 6.25:
                    contribution = height*np.exp(-.5*scaled*scaled)
                    energy += contribution
                    derivative -= contribution*scaled/self.width
            return energy, np.array([derivative])

    def read_reference(path, mode, dimension, point):
        lines = path.read_text().splitlines()
        height, thermal, factor, wt = map(float, lines[1].split()[2:6])
        widths = np.array(list(map(float, lines[2].split()[2:])))
        periods = np.array(list(map(float, lines[3].split()[2:])))
        reference = HillBias(widths, periods) if mode == 'HILLS' else [
            PlumedPBComponent(width, period) for width, period in zip(widths, periods)]
        records = [list(map(float, line.split())) for line in lines[5:] if line]
        max_height_error = 0.0
        for record in records:
            center = np.array(record[1:1+dimension])
            actual = np.array(record[1+dimension:])
            if mode == 'HILLS':
                old, _ = reference.evaluate(center)
                expected = height*np.exp(-old/(thermal*(factor-1))) if wt else height
                max_height_error = max(max_height_error, float(np.max(np.abs(actual-expected))))
                reference.add(center, actual[0])
            else:
                old = np.array([component.evaluate([value])[0] for component, value in zip(reference, center)])
                weights = np.exp(-(old-old.min())/thermal)
                weights /= weights.sum()
                expected = height*weights*(np.exp(-old/(thermal*(factor-1))) if wt else 1)
                max_height_error = max(max_height_error, float(np.max(np.abs(actual-expected))))
                for component, value, addition in zip(reference, center, actual):
                    component.add([value], addition)
        if mode == 'HILLS':
            energy, gradient = reference.evaluate(point)
        else:
            pairs = [component.evaluate([value]) for component, value in zip(reference, point)]
            values = np.array([pair[0] for pair in pairs])
            derivatives = np.array([pair[1][0] for pair in pairs])
            raw = np.exp(-(values-values.min())/thermal)
            energy = values.min()-thermal*np.log(raw.mean())
            gradient = raw/raw.sum()*derivatives
        return float(energy), gradient, max_height_error, records

    results = []
    for mode, wt, mixed in [(mode, wt, False) for mode in ['HILLS', 'PBMETAD'] for wt in [0, 1]] + [
            ('HILLS', 1, True), ('PBMETAD', 1, True), ('AUTO', 1, False)]:
        name = f'{mode.lower()}_wt{wt}_'+('mixed8' if mixed else 'global4')
        directory = args.output/name
        if comm.rank == 0:
            directory.mkdir()
        comm.Barrier()
        os.chdir(directory)
        expressions = ['d1', 'd2', 'd1*d2', 'linear(d1,0.4,0)-d2']
        if mixed:
            expressions += ['AVE(s)', 'AVE(s*s)', 'AVE(s)*d1', 'AVE(s)*AVE(s)+linear(d2,0.2,0)']
        dimension = len(expressions)
        widths = [0.35, 0.45, 0.8, 0.5]+([0.4]*4 if mixed else [])
        periods = [0.0, 0.0, 0.0, 6.283185307179586]+([0.0]*4 if mixed else [])
        parameters = f'''pair_style zero 3.0
pair_coeff * *
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
plugin load "{args.plugin}"
fix m all metad EXECUTOR GRAPH BIAS {mode} GAUSSIAN 0.35 0.12 9 TEMP 300 PACE 2 &
 CAL NAME D1 DISTANCE 1 2 CAL NAME D2 DISTANCE 1 3 &
 SYMBOL d1 D1.RAW SYMBOL d2 D2.RAW &
'''
        if mixed:
            parameters += ' CAL NAME S STRU_FACTOR all cutoff_r 2.7 q_factor 1.7 SYMBOL s S.RAW &\n'
        parameters += f' CV_dim {dimension} SIGMAS '+' '.join(map(str,widths))+' &\n PERIODS '+' '.join(map(str,periods))+' &\n'
        parameters += ''.join(f' DIM_EXPR {index+1} "{expression}" &\n' for index,expression in enumerate(expressions))
        parameters += f' HILLS_FILE bias.hills HILLS_WORKERS 4 METAD_RESTART RESTART_FLAG WT {wt}\n'
        parameters += '''fix integrate all nve
compute pressure all pressure NULL virial
timestep 0.01
thermo 1
thermo_style custom step f_m f_m[1] vol c_pressure[1] c_pressure[2] c_pressure[3] c_pressure[4] c_pressure[5] c_pressure[6]
thermo_modify norm no format float %.17g
'''
        beginning = '''units metal
atom_style atomic
atom_modify map array sort 0 0.0
boundary p p p
region box block 0 12 0 12 0 12 units box
create_box 1 box
create_atoms 1 single 4 4 4 units box
create_atoms 1 single 6 4.2 4 units box
create_atoms 1 single 4.3 6 4.1 units box
create_atoms 1 single 10 10 10 units box
mass 1 30
group moving id 2
velocity moving set 0.8 0.1 -0.2 units box
'''
        def create(configuration, suffix):
            if comm.rank == 0:
                (directory/f'input_{suffix}.lmp').write_text(configuration)
            instance = lammps(cmdargs=['-screen','none','-log',str(directory/f'log_{suffix}.lammps')],comm=comm)
            instance.commands_string(configuration)
            return instance

        simulation = create(beginning+parameters.replace('RESTART_FLAG','0'),'initial')
        def state():
            forces = np.ctypeslib.as_array(simulation.gather_atoms('f',1,3),shape=(12,)).copy()
            coordinates = np.ctypeslib.as_array(simulation.gather_atoms('x',1,3),shape=(12,)).copy()
            pressure = simulation.numpy.extract_compute('pressure',0,1).copy()
            return dict(energy=simulation.extract_fix('m',0,0),
                        values=np.array([simulation.extract_fix('m',0,1,nrow=index) for index in range(dimension)]),
                        forces=forces, coordinates=coordinates,
                        virial=pressure*simulation.get_thermo('vol')/1.602176634e6)
        try:
            simulation.command('run 8')
            reference_state = state()
            history = (directory/'bias.hills').read_bytes()
            require(len(history.splitlines())==9, 'expected four deposition records')
            for _ in range(2):
                simulation.command('run 0')
                require((directory/'bias.hills').read_bytes()==history, 'setup changed deposition history')
                current = state()
                require(np.allclose(current['forces'],reference_state['forces'],atol=1e-12,rtol=1e-12),
                        'repeated run setup changed force')
            expected, derivative, height_error, records = read_reference(directory/'bias.hills',mode,dimension,reference_state['values'])
            require(abs(expected-reference_state['energy'])<1e-12 and height_error<1e-13,
                    'prototype energy/deposition mismatch')
            rank_energies = comm.allgather(reference_state['energy'])
            require(max(rank_energies)-min(rank_energies)<1e-15, 'rank energy mismatch')
            require(np.max(np.abs(reference_state['forces']))>1e-7, 'zero force test signal')

            # Frozen-history finite differences exercise Graph VJP and force/virial assembly.
            base = reference_state['coordinates'].copy()
            increment = 1e-5
            force_error = 0.0
            for coordinate in range(12):
                energies=[]
                for sign in [1,-1]:
                    perturbed=base.copy();perturbed[coordinate]+=sign*increment
                    simulation.scatter_atoms('x',1,3,perturbed.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
                    simulation.command('run 0');energies.append(simulation.extract_fix('m',0,0))
                force_error=max(force_error,abs(-(energies[0]-energies[1])/(2*increment)-reference_state['forces'][coordinate]))
            simulation.scatter_atoms('x',1,3,base.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
            simulation.command('run 0')
            require(force_error<2e-7, 'coordinate force finite difference failed')

            # Apply six box strains and transform positions together; fix energy gives total bias virial.
            virial_error=0.0
            for component,(first,second) in enumerate([(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]):
                energies=[]
                for sign in [1,-1]:
                    matrix=np.eye(3);matrix[first,second]+=sign*increment
                    transformed=(base.reshape(-1,3)@matrix.T).reshape(-1)
                    if first==second:
                        axis='xyz'[first]
                        simulation.command(f'change_box all {axis} final 0 {12*(1+sign*increment):.17g} units box')
                    else:
                        simulation.command('change_box all triclinic')
                        axis={3:'xy',4:'xz',5:'yz'}[component]
                        simulation.command(f'change_box all {axis} final {12*sign*increment:.17g} units box')
                    simulation.scatter_atoms('x',1,3,transformed.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
                    simulation.command('run 0');energies.append(simulation.extract_fix('m',0,0))
                    if first==second:
                        simulation.command(f'change_box all {axis} final 0 12 units box')
                    else:
                        simulation.command(f'change_box all {axis} final 0 units box')
                        simulation.command('change_box all ortho')
                virial_error=max(virial_error,abs(-(energies[0]-energies[1])/(2*increment)-reference_state['virial'][component]))
            simulation.scatter_atoms('x',1,3,base.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
            simulation.command('run 0')
            require(virial_error<2e-6, 'strain virial finite difference failed')
            require((directory/'bias.hills').read_bytes()==history, 'finite difference changed hills')

            # Compare uninterrupted steps 9-12 with a restart at step 8.
            simulation.command('write_restart checkpoint')
            simulation.command('run 4')
            uninterrupted = state()
            final_history = (directory/'bias.hills').read_bytes()
            simulation.close()
            if comm.rank==0:
                (directory/'bias.hills').write_bytes(history)
            comm.Barrier()
            resumed = 'units metal\natom_style atomic\natom_modify map array sort 0 0.0\nread_restart checkpoint\n'
            simulation = create(resumed+parameters.replace('RESTART_FLAG','1'),'resumed')
            simulation.command('run 0')
            require((directory/'bias.hills').read_bytes()==history, 'restart setup deposited twice')
            simulation.command('run 4')
            continued=state()
            for field in ['energy','values','forces','coordinates','virial']:
                require(np.allclose(uninterrupted[field],continued[field],atol=2e-10,rtol=2e-10),
                        f'restart changed {field}')
            require((directory/'bias.hills').read_bytes()==final_history, 'restart changed deposition records')
            require(len(final_history.splitlines())==11, 'wrong continued deposition count')
            results.append(dict(case=name,ranks=comm.size,dimensions=dimension,force_error=force_error,
                                virial_error=virial_error,deposition_height_error=height_error,
                                prototype_energy_error=abs(expected-reference_state['energy']),
                                restart_exact=True,passed=True))
            if comm.rank==0:
                (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
                print('PASS',name,comm.size,flush=True)
        finally:
            simulation.close()
        os.chdir(args.output)


    # 3D 网格反复 run 0 走相同 init 入口，历史和已沉积偏势应保持不变。
    for mode,grid_dimension in [(mode,dimension) for mode in ['GRID','SPARSE_GRID'] for dimension in [2,3]]:
        name=mode.lower()+f'_repeat{grid_dimension}'
        directory=args.output/name
        if comm.rank==0:
            directory.mkdir()
        comm.Barrier()
        os.chdir(directory)
        configuration=beginning+f'''pair_style zero 3.0
pair_coeff * *
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes
plugin load "{args.plugin}"
fix m all metad EXECUTOR GRAPH BIAS {mode} GAUSSIAN 0.35 0.12 9 TEMP 300 PACE 2 &
 CAL NAME D1 DISTANCE 1 2 CAL NAME D2 DISTANCE 1 3 &
 SYMBOL d1 D1.RAW SYMBOL d2 D2.RAW CV_dim {grid_dimension} &
 DIM 1 0 6 48 "d1" DIM 2 0 6 52 "d2" &
 {('DIM 3 -4 2 56 "linear(d1,0.4,0)-d2" ' if grid_dimension==3 else '')} WT 1 HILLS_WORKERS 4 METAD_RESTART 0
fix integrate all nve
timestep 0.01
thermo 1
thermo_style custom step f_m f_m[1] f_m[2]
thermo_modify norm no format float %.17g
'''
        simulation=create(configuration,'grid')
        try:
            simulation.command('run 7')
            energy=simulation.extract_fix('m',0,0)
            forces=np.ctypeslib.as_array(simulation.gather_atoms('f',1,3),shape=(12,)).copy()
            history=Path('HILLS').read_bytes()
            require(energy>0 and np.max(np.abs(forces))>1e-7,'grid bias has no signal')
            for _ in range(2):
                simulation.command('run 0')
                current_energy=simulation.extract_fix('m',0,0)
                current_forces=np.ctypeslib.as_array(simulation.gather_atoms('f',1,3),shape=(12,)).copy()
                require(abs(current_energy-energy)<1e-12 and np.allclose(current_forces,forces,atol=1e-12,rtol=1e-12),
                        'repeated grid setup changed energy or force')
                require(Path('HILLS').read_bytes()==history,'repeated grid setup rewrote history')
            # 不同轴 bins 暴露扁平索引错位，坐标差分同时验证能量/梯度来自同一插值。
            coordinates=np.ctypeslib.as_array(simulation.gather_atoms('x',1,3),shape=(12,)).copy()
            increment=1e-5
            force_error=0.0
            for coordinate in range(12):
                sampled=[]
                for sign in [1,-1]:
                    moved=coordinates.copy();moved[coordinate]+=sign*increment
                    simulation.scatter_atoms('x',1,3,moved.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
                    simulation.command('run 0');sampled.append(simulation.extract_fix('m',0,0))
                force_error=max(force_error,abs(-(sampled[0]-sampled[1])/(2*increment)-forces[coordinate]))
            simulation.scatter_atoms('x',1,3,coordinates.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
            simulation.command('run 0')
            require(force_error<2e-7,'three-dimensional grid force finite difference failed')
            energies=comm.allgather(energy)
            require(max(energies)-min(energies)<1e-15,'grid energy differs across ranks')
            simulation.command('write_restart grid.checkpoint')
            simulation.command('run 2')
            final_energy=simulation.extract_fix('m',0,0)
            final_forces=np.ctypeslib.as_array(simulation.gather_atoms('f',1,3),shape=(12,)).copy()
            final_history=Path('HILLS').read_bytes()
            simulation.close()
            if comm.rank==0: Path('HILLS').write_bytes(history)
            comm.Barrier()
            resumed='units metal\natom_style atomic\natom_modify map array sort 0 0.0\nread_restart grid.checkpoint\n'
            simulation=create(resumed+configuration[len(beginning):].replace('METAD_RESTART 0','METAD_RESTART 1'),'grid_restart')
            simulation.command('run 2')
            require(simulation.extract_fix('m',0,0)==final_energy,'grid restart energy differs')
            resumed_forces=np.ctypeslib.as_array(simulation.gather_atoms('f',1,3),shape=(12,)).copy()
            require(np.array_equal(resumed_forces,final_forces),'grid restart forces differ')
            require(Path('HILLS').read_bytes()==final_history,'grid restart history differs')
            results.append(dict(case=name,ranks=comm.size,dimensions=grid_dimension,repeated_setup=True,
                                force_error=force_error,restart_exact=True,passed=True))
            if comm.rank==0:
                (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
                print('PASS',name,comm.size,flush=True)
        finally:
            simulation.close()
        os.chdir(args.output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plugin',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--np',type=int,nargs='+',default=[1,2,4])
    parser.add_argument('--mpi',type=Path)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args();args.plugin=plugin_path(args.plugin);args.output=args.output.resolve()
    if args.worker:
        return worker(args)
    launcher,_=mpi_tools(args.plugin,args.mpi,False)
    args.output.mkdir(parents=True,exist_ok=False)
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMPI_MCA_btl='self,tcp',OMPI_MCA_pml='ob1')
    results=[]
    for ranks in args.np:
        directory=args.output/f'np{ranks}';directory.mkdir()
        command=[str(launcher),'-np',str(ranks),sys.executable,str(Path(__file__).resolve()),
                 '--worker','--plugin',str(args.plugin),'--output',str(directory)]
        with (directory/'screen.txt').open('w') as output:
            subprocess.run(command,env=environment,stdout=output,stderr=subprocess.STDOUT,timeout=300,check=True)
        results.extend(json.loads((directory/'results.json').read_text()))
        print(f'{ranks} ranks PASS',flush=True)
    (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    (args.output/'metadata.json').write_text(json.dumps(dict(cases=len(results),
        plugin_sha256=hashlib.sha256(args.plugin.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')


if __name__=='__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        if '--worker' in sys.argv:
            from mpi4py import MPI
            MPI.COMM_WORLD.Abort(1)
        raise

