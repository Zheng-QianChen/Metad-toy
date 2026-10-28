#!/usr/bin/env python3
"""实际 LAMMPS Q/Local Q Graph：独立 CPU 前向、MPI、坐标/应变有限差分。"""
import argparse
import shlex
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_common import plugin_path, mpi_tools

ROOT = Path(__file__).resolve().parents[2]
CASES = ["mean", "nonlinear", "groups", "radial", "isolated", "empty_rank", "excluded"]


def worker(args):
    import numpy as np
    from mpi4py import MPI
    from lammps import lammps
    comm = MPI.COMM_WORLD
    rank = comm.rank
    directory = args.output.resolve()
    os.chdir(directory)
    box_length = 12.15
    positions = np.array([(4.05*(i+dx)+0.3, 4.05*(j+dy)+0.3, 4.05*(k+dz)+0.3)
                         for i in range(3) for j in range(3) for k in range(3)
                         for dx,dy,dz in [(0,0,0),(.5,.5,0),(.5,0,.5),(0,.5,.5)]])
    if args.case == "isolated":
        positions = np.array([[1., 1., 1.]])
    elif args.case == "empty_rank":
        positions = positions[:8].copy()
        positions[:,2] *= 0.4
    atom_ids = np.arange(1, len(positions)+1)
    positions += np.column_stack([0.09*np.sin(.731*atom_ids), .08*np.cos(.527*atom_ids), .07*np.sin(.319*atom_ids)])
    types = atom_ids % 3 + 1
    tiny_ids = [int(i) for i in [1,5,9] if i <= len(positions)]
    data = directory/'initial.data'
    if rank == 0:
        data.write_text(f"Q graph fixture\n\n{len(positions)} atoms\n3 atom types\n\n"
            + ''.join(f"0 {box_length} {axis}lo {axis}hi\n" for axis in 'xyz')
            + "\nMasses\n\n1 1\n2 1\n3 1\n\nAtoms # atomic\n\n"
            + ''.join(f"{i} {t} {x:.17g} {y:.17g} {z:.17g}\n" for i,t,(x,y,z) in zip(atom_ids,types,positions)))
    comm.Barrier()
    simulation = lammps(cmdargs=['-screen','none','-log',str(directory/'log.lammps'),*shlex.split(args.lammps_args)], comm=comm)
    if args.weave_plugin:
        simulation.command(f'plugin load "{args.weave_plugin}"')
        simulation.command('weave/batch_comm')
    groups = ['tiny','type2','all'] if args.case == 'groups' else ['all']*3
    radial = args.case in ('radial','excluded')
    kinds = ['Q','L','L'] if args.kind == 'mixed' else [args.kind]*3
    declarations = ['CAL NAME radius SW_FUNC FERMI r_0 2.8 alpha -8'] if radial else []
    for index, order in enumerate([3,4,6]):
        declarations.append(f"CAL NAME Q{order} STEINH {kinds[index]} {order} {groups[index]} cutoff_r 3.3 "
                            + f"d_block_size {64 if order == 4 else 128} "
                            + ('SW_FUNC_r radius cutoff_eps_r 1e-5' if radial else ''))
    mode = 'AVE' if args.case == 'mean' else 'RAW'
    expressions = ['q3*q4', 'q4+q6'] if args.case == 'mean' else [
        'AVE(q3*q4)+AVE(q4*q6)', 'AVE(q3*q4)+AVE(fermi(q6,0.45,12))']
    if args.legacy_statistics:
        mode = 'RAW'
        expressions = ['SUM(fermi(q3,0.1,12))',
            'safe_div(SUM(q6*fermi(q6,0.45,12)),SUM(fermi(q6,0.45,12)),0)']
        if args.explicit_family: expressions[0]='SUM(fermi(q3+q4,0.1,12),q3)'
    configuration = f"""units lj
atom_style atomic
atom_modify map array
boundary p p p
read_data "{data}"
change_box all triclinic
pair_style zero {2.5 if radial else 4.0}
pair_coeff * *
neighbor 0.35 bin
neigh_modify every 1 delay 0 check yes
group tiny id {' '.join(map(str,tiny_ids))}
group type2 type 2
{'neigh_modify exclude type 1 2 exclude group tiny type2' if args.case=='excluded' else ''}
plugin load "{args.plugin}"
reset_timestep 99999
fix m all metad EXECUTOR {args.executor} GAUSSIAN 0.10 0.10 10 PACE 100000 &
    {' &'.join(declarations)} &
    SYMBOL q3 Q3.{mode} SYMBOL q4 Q4.{mode} SYMBOL q6 Q6.{mode} &
    CV_dim 2 DIM 1 -1 2 300 "{expressions[0]}" DIM 2 -1 2 300 "{expressions[1]}" &
    METAD_RESTART 0 WT 0
compute raw all metad/atom m Q3.stein_q Q4.stein_q Q6.stein_q
compute pressure all pressure NULL virial
compute stress all stress/atom NULL virial
compute stress_sum all reduce sum c_stress[*]
thermo 1
thermo_style custom step f_m[1] f_m[2] f_m c_pressure[1] c_pressure[2] c_pressure[3] c_pressure[4] c_pressure[5] c_pressure[6] c_stress_sum[1] c_stress_sum[2] c_stress_sum[3] c_stress_sum[4] c_stress_sum[5] c_stress_sum[6]
thermo_modify norm no format float %.17g
run 1
variable dx atom 0.07*sin(.413*id+.1)
variable dy atom 0.06*cos(.293*id+.4)
variable dz atom 0.05*sin(.619*id+.7)
displace_atoms all move v_dx v_dy v_dz units box
reset_timestep 100001
run 0
"""
    # 用独立 worker 验证旧接口报出迁移提示，避免 MPI 异常中断正常场景。
    if args.legacy_statistics:
        configuration=configuration.replace('DIM 1 -1 2 300 ', 'DIM 1 -1 150 15100 ')
        configuration=configuration.replace('CV_dim 2 DIM', f'CV_dim 2 SIGMAS {0.1*len(positions)} 0.1 DIM')
    if args.obsolete_interface:
        configuration = configuration.replace('EXECUTOR GRAPH', 'EXECUTOR LEGACY') if args.obsolete_interface.startswith('legacy') else configuration
        if args.obsolete_interface in ('legacy_switch', 'value_switch'):
            configuration = configuration.replace('CAL NAME Q3 ', 'CAL NAME value_switch SW_FUNC FERMI r_0 0.45 alpha 12 &CAL NAME Q3 ', 1)
        legacy_suffixes = {
            'legacy_ave': 'AVE',
            'legacy_mean_solid': 'MEAN_SOLID',
            'legacy_switch': 'SW_FUNC.value_switch',
            'legacy_nsolid': 'NSOLID',
            'legacy_frac': 'FRAC',
        }
        if args.obsolete_interface in legacy_suffixes:
            configuration = configuration.replace('Q3.AVE', 'Q3.' + legacy_suffixes[args.obsolete_interface])
        elif args.obsolete_interface == 'value_switch':
            configuration = configuration.replace(f'STEINH {kinds[0]} 3 all ', f'STEINH {kinds[0]} 3 all SW_FUNC_cv value_switch ', 1)
    # 每条 CAL 放在独立续行上，便于保存和手动复现。
    configuration = configuration.replace(' &CAL ', ' &\n    CAL ')
    if rank == 0: (directory/'in.lammps').write_text(configuration)
    simulation.commands_string(configuration)

    def state():
        count = simulation.extract_setting('nlocal')
        ids = simulation.numpy.extract_atom('id')[:count].copy()
        coords = simulation.numpy.extract_atom('x')[:count].copy()
        force = simulation.numpy.extract_atom('f')[:count].copy()
        raw = simulation.numpy.extract_compute('raw',1,2)[:count,:3].copy()
        stress = simulation.numpy.extract_compute('stress',1,2)[:count,:6].copy()
        local = np.column_stack([ids,coords,force,raw,stress])
        rows = np.concatenate(comm.allgather(local))
        rows = rows[np.argsort(rows[:,0])]
        values = [simulation.extract_fix('m',0,1,nrow=i) for i in range(2)]
        energy = simulation.extract_fix('m',0,0)
        pressure = simulation.numpy.extract_compute('pressure',0,1).copy()
        return dict(rows=rows, values=np.array(values), energy=energy, virial=pressure*simulation.get_thermo('vol'))

    def expected_raw(rows):
        coords = rows[:,1:4]
        displacement = coords[None,:,:]-coords[:,None,:]
        displacement -= np.rint(displacement/box_length)*box_length
        distance = np.linalg.norm(displacement,axis=2)
        cutoff = 2.8+np.log(1/1e-5-1)/8 if radial else 3.3
        selected = (distance>1e-12)&(distance<cutoff)
        if args.case == 'excluded':
            type_one = types == 1; type_two = types == 2
            tiny = np.isin(atom_ids,tiny_ids)
            selected &= ~((type_one[:,None]&type_two[None,:]) | (type_two[:,None]&type_one[None,:]) |
                          (tiny[:,None]&type_two[None,:]) | (type_two[:,None]&tiny[None,:]))
        weights = np.where(selected, 1/(1+np.exp(8*(distance-2.8))) if radial else 1.0, 0.0)
        weight_sum = weights.sum(axis=1)
        raw = np.zeros((len(coords),3))
        for index, order in enumerate([3,4,6]):
            centers = range(len(coords))
            if groups[index] == 'tiny': centers = [i-1 for i in tiny_ids]
            if groups[index] == 'type2': centers = np.flatnonzero(types == 2)
            if kinds[index] == 'L':
                # SciPy 球谐独立计算 q_lm，再对自身与邻居 q_lm 做径向加权平均。
                from scipy.special import sph_harm
                harmonics = np.zeros((len(coords),order+1),dtype=complex)
                for atom in range(len(coords)):
                    neighbors = selected[atom]
                    if weight_sum[atom] < 1e-12: continue
                    directions = displacement[atom,neighbors]/distance[atom,neighbors,None]
                    azimuth = np.arctan2(directions[:,1],directions[:,0])
                    polar = np.arccos(np.clip(directions[:,2],-1,1))
                    for component in range(order+1):
                        harmonics[atom,component] = np.dot(weights[atom,neighbors], sph_harm(component,order,azimuth,polar))/weight_sum[atom]
                averaged = (harmonics + weights@harmonics)/(1+weight_sum[:,None])
                for center in centers:
                    if weight_sum[center] < 1e-12: continue
                    norm = abs(averaged[center,0])**2 + 2*np.sum(abs(averaged[center,1:])**2)
                    raw[center,index] = np.sqrt(4*np.pi/(2*order+1)*norm)
            else:
                for center in centers:
                    neighbors = selected[center]
                    if weight_sum[center] < 1e-12: continue
                    directions = displacement[center,neighbors]/distance[center,neighbors,None]
                    coeff = np.zeros(order+1);coeff[-1] = 1
                    legendre = np.polynomial.legendre.legval(np.clip(directions@directions.T,-1,1),coeff)
                    center_weights = weights[center,neighbors]
                    raw[center,index] = np.sqrt(max(0,float(center_weights@legendre@center_weights)))/weight_sum[center]
        return raw

    def validate(current):
        raw = expected_raw(current['rows'])
        error = float(np.max(np.abs(raw-current['rows'][:,7:10])))
        assert error < 2e-12, ('raw',error,np.max(np.abs(raw-current['rows'][:,7:10]),axis=0),raw[:2],current['rows'][:2,7:10])
        if args.legacy_statistics:
            mask = np.ones(len(raw),dtype=bool)
            if groups[0]=='tiny': mask=np.isin(atom_ids,tiny_ids)
            elif groups[0]=='type2': mask=types==2
            local_score=raw[mask,0]+(raw[mask,1] if args.explicit_family else 0)
            weight3=1/(1+np.exp(-12*(local_score-.1)))
            weight6=1/(1+np.exp(-12*(raw[:,2]-.45)))
            denominator=weight6.sum()
            expected=np.array([weight3.sum(),np.dot(raw[:,2],weight6)/denominator if denominator else 0])
        elif args.case == 'mean':
            means=raw.mean(axis=0); expected=np.array([means[0]*means[1],means[1]+means[2]])
        else:
            expected=np.array([np.mean(raw[:,0]*raw[:,1]+raw[:,1]*raw[:,2]),
                               np.mean(raw[:,0]*raw[:,1]+1/(1+np.exp(-12*(raw[:,2]-.45))))])
        cv_error=float(np.max(np.abs(expected-current['values'])))
        assert cv_error < 2e-12, ('cv',cv_error,expected,current['values'])
        net=float(np.max(np.abs(current['rows'][:,4:7].sum(axis=0))))
        stress_error=float(np.max(np.abs(current['virial']+current['rows'][:,10:16].sum(axis=0))))
        assert net<1e-10 and stress_error<1e-10,(net,stress_error)
        peak=float(np.max(np.abs(current['rows'][:,4:7])))
        assert (peak == 0 if args.case=='isolated' else peak>1e-9),('force',peak)
        return dict(raw_error=error,cv_error=cv_error,net_force=net,stress_sum_error=stress_error,max_force=peak)

    current=state();report=validate(current)
    initial_state={k:v.tolist() if hasattr(v,'tolist') else v for k,v in current.items()}
    if args.finite_difference:
        base=current['rows'][:,1:4].copy(); step=2e-5
        def place(coords):
            payload=(ctypes.c_double*(coords.size))(*coords.ravel())
            simulation.scatter_atoms('x',1,3,payload)
            simulation.command('run 0')
            return simulation.extract_fix('m',0,0)
        errors=[]
        for atom in [0,1,4,9]:
            for axis in range(3):
                plus=base.copy();minus=base.copy();plus[atom,axis]+=step;minus[atom,axis]-=step
                derivative=(place(plus)-place(minus))/(2*step)
                errors.append(abs(derivative+current['rows'][atom,4+axis]))
        place(base)
        report['force_fd_error']=max(errors)
        assert max(errors)<2e-7,('coordinate FD',max(errors))
        strain_errors=[]
        for component,(row,col) in enumerate([(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]):
            energies=[]
            for sign in [1,-1]:
                matrix=np.eye(3);matrix[row,col]+=sign*step
                cell=box_length*matrix
                simulation.command(f"change_box all x final 0 {cell[0,0]} y final 0 {cell[1,1]} z final 0 {cell[2,2]} xy final {cell[0,1]} xz final {cell[0,2]} yz final {cell[1,2]} units box")
                energies.append(place(base@matrix.T))
            strain_errors.append(abs((energies[0]-energies[1])/(2*step)+current['virial'][component]))
        simulation.command(f"change_box all x final 0 {box_length} y final 0 {box_length} z final 0 {box_length} xy final 0 xz final 0 yz final 0 units box")
        place(base)
        report['virial_fd_error']=max(strain_errors)
        assert max(strain_errors)<2e-6,('strain FD',strain_errors)
        report['same_step_restore_error']=float(np.max(np.abs(state()['rows']-current['rows'])))
        assert report['same_step_restore_error']<1e-10
    if args.case=='groups':
        tiny_ids=[2,12,42,58]
        simulation.commands_string('group tiny clear\ngroup tiny id 2 12 42 58\nrun 0')
        report['changed_group']=validate(state())
    if args.case=='nonlinear':
        simulation.commands_string('timestep 0.1\nfix drift all move linear 0.8 0.3 0.5 units box\nrun 40\nunfix drift')
        report['migration']=validate(state())
    simulation.close()
    simulation.lib.lammps_kokkos_finalize()
    if rank==0:
        log=(directory/'log.lammps').read_text(encoding='utf-8')
        if args.executor=='GRAPH': assert log.count('NVRTC ready:')==1,'unexpected recompilation'
        (directory/'state.json').write_text(json.dumps(initial_state,indent=2)+'\n')
        (directory/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(case=args.case,ranks=comm.size,executor=args.executor,**report)),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plugin',type=Path,default=None)
    parser.add_argument('--mpi',type=Path,default=None)
    parser.add_argument('--mpi-args',default='')
    parser.add_argument('--lammps-args',default='')
    parser.add_argument('--weave-plugin',type=Path)
    parser.add_argument('--np',type=int,nargs='+',default=[1,2,4])
    parser.add_argument('--cases',nargs='+',default=CASES,choices=CASES)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--case',default='nonlinear')
    parser.add_argument('--executor',default='GRAPH')
    parser.add_argument('--kind',choices=['Q','L','mixed'],default='Q')
    parser.add_argument('--finite-difference',action='store_true')
    parser.add_argument('--legacy-statistics',action='store_true',help='Validate Graph NSOLID and MEAN_SOLID formulas')
    parser.add_argument('--explicit-family',action='store_true',help='Test SUM(expression, raw_family_symbol)')
    parser.add_argument('--obsolete-interface', choices=['legacy_ave','legacy_mean_solid','legacy_switch','legacy_nsolid','legacy_frac','value_switch'])
    args=parser.parse_args();args.output=args.output.resolve();args.plugin=plugin_path(args.plugin)
    if args.worker: return worker(args)
    args.mpi,_=mpi_tools(args.plugin,args.mpi,False)
    import numpy as np
    args.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for case in args.cases:
        reference=None
        for ranks in args.np:
            executors=['GRAPH']
            for executor in executors:
                directory=args.output/f'{case}_{executor.lower()}_np{ranks}';directory.mkdir()
                command=[str(args.mpi),*shlex.split(args.mpi_args),'-np',str(ranks),sys.executable,str(Path(__file__).resolve()),
                         '--worker','--kind',args.kind,'--case',case,'--executor',executor,'--output',str(directory),'--plugin',str(args.plugin)]
                if ranks==1 and case in ['nonlinear','groups','radial','excluded']: command.append('--finite-difference')
                if args.legacy_statistics: command.append('--legacy-statistics')
                if args.explicit_family: command.append('--explicit-family')
                command.append('--lammps-args='+args.lammps_args)
                if args.weave_plugin: command.extend(['--weave-plugin',str(args.weave_plugin)])
                with (directory/'screen.txt').open('w') as output:
                    result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,timeout=180)
                if result.returncode: raise RuntimeError(str(directory)+'\n'+(directory/'screen.txt').read_text(encoding='utf-8')[-5000:])
                state=json.loads((directory/'state.json').read_text(encoding='utf-8'));report=json.loads((directory/'result.json').read_text(encoding='utf-8'))
                error=0.0
                if reference is None: reference=state
                else:
                    error=max(float(np.max(np.abs(np.array(state[key])-np.array(reference[key])))) for key in state)
                    assert error<1e-9,(case,ranks,executor,error)
                row=dict(case=case,ranks=ranks,executor=executor,reference_error=error,**report)
                results.append(row);print(json.dumps(row),flush=True)
                (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    interface_results=[]
    interfaces = ['legacy_ave','legacy_mean_solid','legacy_switch','legacy_nsolid','legacy_frac','value_switch'] if args.kind=='L' else ['legacy_ave','legacy_switch','value_switch']
    for interface in interfaces:
        directory=args.output/interface;directory.mkdir()
        command=[str(args.mpi),*shlex.split(args.mpi_args),'-np','1',sys.executable,str(Path(__file__).resolve()),
                 '--worker','--kind',args.kind,'--case','mean','--obsolete-interface',interface,'--output',str(directory),'--plugin',str(args.plugin)]
        command.append('--lammps-args='+args.lammps_args)
        if args.weave_plugin: command.extend(['--weave-plugin',str(args.weave_plugin)])
        with (directory/'screen.txt').open('w') as output:
            result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,timeout=60)
        output=(directory/'screen.txt').read_text(encoding='utf-8')
        if interface=='value_switch':
            expected='SW_FUNC_cv has moved to Graph'
        else: expected=f'STEINH {"L" if args.kind=="L" else "Q"} requires EXECUTOR GRAPH'
        assert result.returncode!=0 and expected in output,(interface,result.returncode,output[-3000:])
        interface_results.append(dict(interface=interface,exit_code=result.returncode,message=expected))
    (args.output/'interface_results.json').write_text(json.dumps(interface_results,indent=2)+'\n')
    print('STEINHARDT_GRAPH_PASS',flush=True)

if __name__=='__main__': main()
