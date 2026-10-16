#!/usr/bin/env python3
"""DISTANCE boundary/ownership regression, run with mpirun after compiling bias_probe.so."""
import argparse
import ctypes
import json
from pathlib import Path
import os
import numpy as np
from mpi4py import MPI
from lammps import lammps

parser=argparse.ArgumentParser()
parser.add_argument('--plugin',type=Path,required=True)
parser.add_argument('--probe',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();args.output=args.output.resolve();args.plugin=args.plugin.resolve();args.probe=args.probe.resolve()
comm=MPI.COMM_WORLD
if comm.rank==0:args.output.mkdir(parents=True,exist_ok=False)
comm.Barrier();os.chdir(args.output)
probe=ctypes.CDLL(str(args.probe))
for name in ['fd_bias_gradient','fd_add_hill']:
    fn=getattr(probe,name);fn.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_double)];fn.restype=ctypes.c_int

def call(simulation,name,value):
    payload=(ctypes.c_double*1)(value)
    assert getattr(probe,name)(simulation.lmp,payload)==0
    return payload[0]

results=[]
for periodic in [True,False]:
    label='triclinic_periodic' if periodic else 'nonperiodic'
    cell=np.array([[10.,1.4,.8],[0.,10.,.6],[0.,0.,10.]])
    ids=np.array([11,57,103,901])
    initial=np.array([[.03,.05,.06],[.2,.6,.3],[.7,.4,.8],[.96,.94,.92]])@cell.T
    data=args.output/f'{label}.data'
    if comm.rank==0:
        data.write_text('Distance edge fixture\n\n4 atoms\n1 atom types\n\n0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n1.4 .8 .6 xy xz yz\n\nMasses\n\n1 1\n\nAtoms # atomic\n\n'+''.join(f'{tag} 1 {x} {y} {z}\n' for tag,(x,y,z) in zip(ids,initial)))
    comm.Barrier()
    simulation=lammps(cmdargs=['-screen','none','-log',str(args.output/f'{label}.log')],comm=comm)
    simulation.commands_string(f"""units lj
atom_style atomic
atom_modify map array sort 1 1.0
boundary {'p p p' if periodic else 'f f f'}
read_data "{data}"
pair_style zero 2.0
pair_coeff * *
neighbor 0.2 bin
plugin load "{args.plugin}"
fix m all metad EXECUTOR GRAPH GAUSSIAN .1 .1 10 PACE 0 CAL NAME D DISTANCE 11 901 SYMBOL d D.value CV_dim 1 DIM 1 0 20 20000 d WT 0 METAD_RESTART 0
compute pressure all pressure NULL virial
thermo_style custom step f_m[1] f_m c_pressure[1] c_pressure[2] c_pressure[3] c_pressure[4] c_pressure[5] c_pressure[6]
thermo_modify norm no
run 0
""")
    call(simulation,'fd_add_hill',simulation.extract_fix('m',0,1,nrow=0)+.06)
    for phase in range(3 if periodic else 1):
        if phase:
            # 平移相同分数坐标后 wrap，跨 MPI 子域并触发排序/owner 迁移。
            shifted=(np.array([[.03,.05,.06],[.2,.6,.3],[.7,.4,.8],[.96,.94,.92]])+phase*.31)%1
            for tag,position in zip(ids,shifted@cell.T):
                simulation.command(f'set atom {tag} x {position[0]} y {position[1]} z {position[2]}')
        simulation.command('run 0')
        count=simulation.extract_setting('nlocal')
        local=np.column_stack([simulation.numpy.extract_atom('id')[:count],simulation.numpy.extract_atom('x')[:count],simulation.numpy.extract_atom('f')[:count]])
        rows=np.concatenate(comm.allgather(local));rows=rows[np.argsort(rows[:,0])]
        delta=rows[-1,1:4]-rows[0,1:4]
        if periodic:
            images=np.array([delta-cell@np.array([i,j,k]) for i in [-1,0,1] for j in [-1,0,1] for k in [-1,0,1]])
            delta=images[np.argmin(np.sum(images**2,axis=1))]
        distance=np.linalg.norm(delta);value=simulation.extract_fix('m',0,1,nrow=0)
        seed=call(simulation,'fd_bias_gradient',0)
        expected=np.zeros((4,3));expected[0]=seed*delta/distance;expected[-1]=-expected[0]
        tensor=-seed*np.outer(delta,delta)/distance
        virial=np.array([tensor[0,0],tensor[1,1],tensor[2,2],tensor[0,1],tensor[0,2],tensor[1,2]])
        actual=simulation.numpy.extract_compute('pressure',0,1).copy()*simulation.get_thermo('vol')
        record=dict(case=label,phase=phase,ranks=comm.size,owned_counts=comm.allgather(count),
                    distance_error=float(abs(value-distance)),force_error=float(np.max(np.abs(rows[:,4:]-expected))),
                    virial_error=float(np.max(np.abs(actual-virial))))
        assert max(record[key] for key in ['distance_error','force_error','virial_error'])<1e-10,record
        assert max(comm.allgather(value))-min(comm.allgather(value))<1e-12
        results.append(record)
    simulation.close()
if comm.rank==0:
    (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results))
