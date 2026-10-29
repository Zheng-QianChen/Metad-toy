#!/usr/bin/env python3
"""LocalGeom: independent periodic CPU reference, MPI, forces and six strains."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

CASES = ['coord', 'tet', 'typed', 'groups', 'isolated', 'empty', 'excluded', 'npt']


def worker(a):
    import numpy as np
    from mpi4py import MPI
    from lammps import lammps
    comm = MPI.COMM_WORLD
    os.chdir(a.output)
    length = 10.8
    pos = np.array([(3.6*(i+u)+.25, 3.6*(j+v)+.25, 3.6*(k+w)+.25)
                    for i in range(3) for j in range(3) for k in range(3)
                    for u, v, w in [(0,0,0),(.5,.5,0),(.5,0,.5),(0,.5,.5)]])
    if a.case == 'isolated': pos = pos[:1]
    ids = np.arange(1, len(pos)+1)
    pos += np.column_stack([.08*np.sin(ids*.71), .06*np.cos(ids*.51), .07*np.sin(ids*.33)])
    types = ids % 2 + 1
    centers = [] if a.case == 'empty' else ([1,5,9] if a.case == 'groups' else list(ids))
    data = a.output/'initial.data'
    if comm.rank == 0:
        data.write_text(f'LocalGeom fixture\n\n{len(pos)} atoms\n2 atom types\n\n'
            + ''.join(f'0 {length} {c}lo {c}hi\n' for c in 'xyz')
            + '\nMasses\n\n1 1\n2 1\n\nAtoms # atomic\n\n'
            + ''.join(f'{i} {t} {x:.17g} {y:.17g} {z:.17g}\n' for i,t,(x,y,z) in zip(ids,types,pos)))
    comm.Barrier()
    sim = lammps(cmdargs=['-screen','none','-log','log.lammps',*shlex.split(a.lammps_args)], comm=comm)
    if a.weave_plugin:
        sim.command(f'plugin load "{a.weave_plugin}"'); sim.command('weave/batch_comm')
    # AVE is supported before integrating the independent Graph SUM worktree.
    expression = 'AVE(n)' if a.reduction == 'ave' else 'SUM(n)'
    # Keep the biased energy scale comparable while checking the extensive CV.
    width = max(len(centers),1) if a.reduction=='sum' else 1
    output = 'TET' if a.case == 'tet' else 'COORD'
    center_command = 'group centers empty' if not centers else 'group centers id '+' '.join(map(str,centers))
    config = f'''units lj
atom_style atomic
atom_modify map array
boundary p p p
read_data "{data}"
change_box all triclinic
pair_style zero 3.4
pair_coeff * *
neighbor .35 bin
neigh_modify every 1 delay 0 check yes
{center_command}
{'neigh_modify exclude type 1 2' if a.case == 'excluded' else ''}
plugin load "{a.plugin}"
reset_timestep 99999
fix m all metad EXECUTOR GRAPH GAUSSIAN {0.4*width} 1 10 PACE 100000 &
 CAL NAME rad SW_FUNC FERMI r_0 2.5 alpha -5 &
 CAL NAME geom LOCAL_GEOM centers cutoff_r 3.0 cutoff_eps 1e-12 SW_FUNC_r rad OUTPUT {output} TET_CENTER 5 TET_SIGMA 1.2 {'NEIGH_TYPE 2' if a.case == 'typed' else ''} &
 SYMBOL n geom.RAW CV_dim 1 DIM 1 -100 2000 10500 "{expression}" METAD_RESTART 0 WT 0
compute raw all metad/atom m geom.value geom.n geom.rnn geom.rnn_valid
compute pressure all pressure NULL virial
compute stress all stress/atom NULL virial
compute stress_sum all reduce sum c_stress[*]
thermo 1
thermo_style custom step f_m f_m[1] c_pressure[1] c_pressure[2] c_pressure[3] c_pressure[4] c_pressure[5] c_pressure[6] c_stress_sum[1]
thermo_modify norm no format float %.17g
run 1
variable dx atom .007*sin(.43*id)
variable dy atom .006*cos(.29*id)
displace_atoms all move v_dx v_dy 0 units box
reset_timestep 100001
run 0
'''
    if comm.rank == 0: (a.output/'in.lammps').write_text(config)
    sim.commands_string(config)

    def state():
        count = sim.extract_setting('nlocal')
        local = np.column_stack([sim.numpy.extract_atom('id')[:count], sim.numpy.extract_atom('x')[:count,:3],
            sim.numpy.extract_atom('f')[:count,:3], sim.numpy.extract_compute('raw',1,2)[:count,:4],
            sim.numpy.extract_compute('stress',1,2)[:count,:6]])
        rows = np.concatenate(comm.allgather(local)); rows = rows[np.argsort(rows[:,0])]
        return dict(rows=rows, value=sim.extract_fix('m',0,1,nrow=0), energy=sim.extract_fix('m',0,0),
                    virial=sim.numpy.extract_compute('pressure',0,1).copy()*sim.get_thermo('vol'))

    def validate(s):
        xyz = s['rows'][:,1:4]
        lo,hi,xy,yz,xz,*_ = sim.extract_box()
        cell=np.array([[hi[0]-lo[0],xy,xz],[0,hi[1]-lo[1],yz],[0,0,hi[2]-lo[2]]])
        delta = xyz[None,:,:]-xyz[:,None,:]
        fraction=delta@np.linalg.inv(cell).T; fraction-=np.rint(fraction); delta=fraction@cell.T
        r = np.linalg.norm(delta,axis=2)
        keep = (r>=1e-12)&(r<=3.0)
        if a.case == 'typed': keep &= (types[None,:]==2)
        if a.case == 'excluded': keep &= (types[None,:]==types[:,None])
        w = np.where(keep,1/(1+np.exp(5*(r-2.5))),0)
        n = w.sum(axis=1)
        valid = n>1e-12
        rnn = np.divide((w*r).sum(axis=1),n,out=np.zeros_like(n),where=valid)
        value = np.exp(-.5*((n-5)/1.2)**2) if output=='TET' else n
        expected = np.column_stack([value,n,rnn,valid])
        mask = np.isin(ids,centers); expected[~mask]=0
        raw_error = float(np.max(np.abs(expected-s['rows'][:,7:11])))
        expected_cv = expected[:,0].sum() if a.reduction=='sum' else expected[mask,0].mean()
        cv_error = abs(expected_cv-s['value'])
        stress_error = float(np.max(np.abs(s['rows'][:,11:17].sum(axis=0)+s['virial'])))
        assert raw_error<2e-12,(a.case,'raw',raw_error)
        assert cv_error<2e-10,(a.case,'CV',cv_error,expected_cv,s['value'])
        assert stress_error<2e-9,('stress',stress_error)
        return dict(raw_error=raw_error,cv_error=cv_error,stress_error=stress_error,
                    mean_rnn=float(expected[:,2].sum()/max(expected[:,3].sum(),1)))

    s=state(); report=validate(s)
    initial={k:v.tolist() if hasattr(v,'tolist') else v for k,v in s.items()}
    if a.fd:
        base=s['rows'][:,1:4].copy(); h=2e-5
        def place(x):
            payload=(ctypes.c_double*x.size)(*x.ravel())
            sim.scatter_atoms('x',1,3,payload); sim.command('run 0')
            return sim.extract_fix('m',0,0)
        errors=[]
        for atom in [0,1,4,9]:
            for c in range(3):
                plus=base.copy(); minus=base.copy(); plus[atom,c]+=h; minus[atom,c]-=h
                errors.append(abs((place(plus)-place(minus))/(2*h)+s['rows'][atom,4+c]))
        place(base)
        report['force_fd_error']=max(errors)
        assert max(errors)<2e-7,report
        errors=[]
        for component,(r,c) in enumerate([(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]):
            energies=[]
            for sign in [1,-1]:
                m=np.eye(3); m[r,c]+=sign*h; box=length*m
                sim.command(f'change_box all x final 0 {box[0,0]} y final 0 {box[1,1]} z final 0 {box[2,2]} xy final {box[0,1]} xz final {box[0,2]} yz final {box[1,2]} units box')
                energies.append(place(base@m.T))
            errors.append(abs((energies[0]-energies[1])/(2*h)+s['virial'][component]))
        report['virial_fd_error']=max(errors)
        assert max(errors)<2e-5,report
        sim.command(f'change_box all x final 0 {length} y final 0 {length} z final 0 {length} xy final 0 xz final 0 yz final 0 units box')
        place(base)
        assert np.max(np.abs(state()['rows']-s['rows']))<1e-8
    if a.case=='groups':
        centers=[2,12,42,58]
        sim.commands_string('group centers clear\ngroup centers id 2 12 42 58\nrun 0')
        report['changed_group']=validate(state())
        sim.commands_string('timestep .1\nfix drift all move linear .8 .3 .5 units box\nrun 40\nunfix drift')
        report['migration']=validate(state())
    if a.case=='npt':
        sim.commands_string('velocity all create .05 91321\ntimestep .001\nfix integrate all npt temp .05 .05 .1 iso .01 .01 1\nrun 60\nunfix integrate')
        report['npt']=validate(state())
    sim.close()
    if comm.rank==0:
        (a.output/'state.json').write_text(json.dumps(initial,indent=2)+'\n')
        (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plugin',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--np',type=int,nargs='+',default=[1,2,4]); p.add_argument('--cases',nargs='+',default=CASES)
    p.add_argument('--lammps-args',default=''); p.add_argument('--weave-plugin',type=Path)
    p.add_argument('--reduction',choices=['sum','ave'],default='sum')
    p.add_argument('--worker',action='store_true'); p.add_argument('--case'); p.add_argument('--fd',action='store_true')
    a=p.parse_args(); a.output=a.output.resolve(); a.plugin=a.plugin.resolve()
    if a.worker: return worker(a)
    import numpy as np
    a.output.mkdir(parents=True,exist_ok=False); results=[]
    (a.output/'metadata.json').write_text(json.dumps(dict(command=sys.argv,plugin=str(a.plugin),
        plugin_sha256=hashlib.sha256(a.plugin.read_bytes()).hexdigest(),
        lammps_library_path=os.environ.get('LAMMPS_LIBRARY_PATH'),
        lammps_args=a.lammps_args,weave_plugin=str(a.weave_plugin) if a.weave_plugin else None),indent=2)+'\n')
    for case in a.cases:
        reference=None
        for ranks in a.np:
            directory=a.output/f'{case}-np{ranks}'; directory.mkdir()
            command=['mpiexec','--oversubscribe','-np',str(ranks),sys.executable,str(Path(__file__).resolve()),
                '--worker','--case',case,'--plugin',str(a.plugin),'--output',str(directory),
                '--reduction',a.reduction,'--lammps-args='+a.lammps_args]
            if a.weave_plugin: command+=['--weave-plugin',str(a.weave_plugin)]
            if ranks==1 and case not in ['isolated','empty']: command+=['--fd']
            with (directory/'screen.txt').open('w') as log:
                completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=240)
            if completed.returncode:
                raise RuntimeError(str(directory)+'\n'+(directory/'screen.txt').read_text()[-6000:])
            s=json.loads((directory/'state.json').read_text()); result=json.loads((directory/'result.json').read_text())
            if reference is None: reference=s
            error=max(float(np.max(np.abs(np.array(s[k])-np.array(reference[k])))) for k in s)
            assert error<2e-8,(case,ranks,error)
            results.append(dict(case=case,ranks=ranks,mpi_error=error,**result))
            (a.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps(results[-1]),flush=True)
    print('LOCALGEOM_PASS',flush=True)


if __name__=='__main__': main()
