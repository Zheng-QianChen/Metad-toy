#!/usr/bin/env python3
"""固定 hill 的 LAMMPS 有限差分：单 CV、线性组合、非线性 Graph、MPI。"""
import argparse
import shlex
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_common import plugin_path, mpi_tools, lammps_source, cmake_cache

ROOT = Path(__file__).resolve().parents[2]
# 每项保留实际 CAL 参数；单项和线性组合共用相同定义与初态。
CALCULATIONS = {
    **{f'q{order}': f'STEINH Q {order} all cutoff_r 3.30 SW_FUNC_r radial cutoff_eps_r 1e-5' for order in (3, 4, 6)},
    **{f'l{order}': f'STEINH L {order} all cutoff_r 3.30 SW_FUNC_r radial cutoff_eps_r 1e-5' for order in (3, 4, 6)},
    'structure': 'STRU_FACTOR all cutoff_r 3.05 q_factor 1.7',
    'chemical': 'STRU_FACTOR all cutoff_r 3.05 q_factor 1.7 Chemical Chem_ctarget 1.8 Chem_sigma 1.1 Chem_map (1,1) (2,-0.7) (3,0.35)',
    'chemical_gaussian': 'STRU_FACTOR all cutoff_r 3.05 q_factor 2.3 Chemical Chem_ctarget 1.8 Chem_sigma 1.1 Chem_map (2,-0.7)',
    'weighted': 'WEIGHT_CHEM all cutoff_r 3.05 Chem_ctarget 1.8 Chem_sigma 1.1 Chem_map (1,1) (2,-0.7) (3,0.35)',
    'weighted_gaussian': 'WEIGHT_CHEM all cutoff_r 3.05 Chem_ctarget 1.8 Chem_sigma 1.1 Chem_map (3,0.4)',
    'css_binary': 'CSS all CHANNEL 1 radial 4.0 3.0 CHANNEL 2 radial 4.5 3.0 cutoff_eps 1e-5',
    'css_ternary': 'CSS all CHANNEL 1 radial 4.0 3.0 CHANNEL 2 radial 4.5 3.0 CHANNEL 3 radial 4.0 3.0 CHANNEL 2 outer 8.0 4.0 cutoff_eps 1e-5',
}
LINEAR = {
    'ql_linear': dict(zip(['q3','q4','q6','l3','l4','l6'], [.7,-.4,.25,-.6,.9,-.3])),
    'chemical_linear': dict(structure=.7, chemical=-.4, chemical_gaussian=.2, weighted=.08, weighted_gaussian=-.06, css_binary=.5, css_ternary=-.3),
    'all_linear': {name: (-1 if index % 2 else 1) * (.06 if name.startswith('weighted') else .3 + .03*index) for index,name in enumerate(CALCULATIONS)},
}
CASES = list(CALCULATIONS) + list(LINEAR) + ['mixed_groups', 'nonlinear_graph', 'distance', 'distance_linear', 'distance_graph', 'distance_graph_linear', 'distance_mixed']
STEPS = (1e-4, 3e-5, 1e-5, 3e-6)


def case_definition(case):
    if case.startswith('distance'):
        calculations = dict(distance='DISTANCE 1 2')
        if 'linear' in case: calculations['distance2'] = 'DISTANCE 1 17'
        graph = case not in ('distance', 'distance_linear')
        expression = 'distance'
        if 'linear' in case:
            expression = 'linear(distance,.7,0)-linear(distance2,.3,0)' if graph else '.7*distance-.3*distance2'
        if case == 'distance_mixed':
            calculations['structure'] = CALCULATIONS['structure']
            expression = 'distance*AVE(structure)+linear(distance,.2,0)'
        return calculations, [expression], 'GRAPH' if graph else 'LEGACY', 'RAW' if graph else 'DISTANCE'
    names = [case] if case in CALCULATIONS else list(LINEAR[case]) if case in LINEAR else ['q3','l6','structure','chemical','weighted','css_ternary']
    calculations = {name:CALCULATIONS[name] for name in names}
    if case in CALCULATIONS: expressions=[case]
    elif case in LINEAR: expressions=['+'.join(f'linear({name},{weight},0)' for name,weight in LINEAR[case].items())]
    elif case == 'mixed_groups':
        calculations = {name:definition.replace(' all ', ' tiny ' if index%2==0 else ' type2 ', 1) for index,(name,definition) in enumerate(calculations.items())}
        expressions=['+'.join(f'linear(AVE({name}),{.1*(-1)**index},0)' for index,name in enumerate(calculations))]
    else:
        expressions=['AVE(q3*l6)+linear(AVE(fermi(chemical,0.4,2)),0.3,0)+linear(AVE(css_ternary*structure),0.2,0)',
                     'linear(AVE(q3*l6),-0.4,0)+exp(linear(AVE(weighted),0.02,0))+linear(AVE(rational(l6,0.4,6,12)),0.2,0)']
    return calculations, expressions, 'GRAPH', 'RAW' if case in ('mixed_groups','nonlinear_graph') else 'AVE'


def worker(args):
    import numpy as np
    from mpi4py import MPI
    from lammps import lammps
    comm = MPI.COMM_WORLD
    rank = comm.rank
    directory = args.output.resolve()
    os.chdir(directory)
    box_length = 8.1
    positions = np.array([(4.05*(i+dx)+.23,4.05*(j+dy)+.27,4.05*(k+dz)+.21)
        for i in range(2) for j in range(2) for k in range(2)
        for dx,dy,dz in [(0,0,0),(.5,.5,0),(.5,0,.5),(0,.5,.5)]])
    ids = np.arange(1,len(positions)+1)
    positions += np.column_stack([.11*np.sin(.731*ids),.09*np.cos(.527*ids),.08*np.sin(.319*ids)])
    data = directory/'initial.data'
    if rank == 0:
        data.write_text(f'Finite difference fixture\n\n{len(ids)} atoms\n3 atom types\n\n'
            + ''.join(f'0 {box_length} {axis}lo {axis}hi\n' for axis in 'xyz')
            + '\nMasses\n\n1 1\n2 1\n3 1\n\nAtoms # atomic\n\n'
            + ''.join(f'{atom} {atom%3+1} {x:.17g} {y:.17g} {z:.17g}\n' for atom,(x,y,z) in zip(ids,positions)))
    comm.Barrier()
    calculations,expressions,executor,suffix = case_definition(args.case)
    declarations = [] if executor=='LEGACY' else ['CAL NAME radial SW_FUNC FERMI r_0 2.8 alpha -8', 'CAL NAME outer SW_FUNC FERMI r_0 3.4 alpha -8']
    declarations += [f'CAL NAME {name} {definition}' for name,definition in calculations.items()]
    dimensions = len(expressions)
    limits = (-20,20,2000) if dimensions==1 else (-3,3,600)
    configuration = f"""units lj
atom_style atomic
atom_modify map array sort 0 0.0
boundary p p p
read_data "{data}"
change_box all triclinic
pair_style zero 2.5
pair_coeff * *
neighbor 0.35 bin
neigh_modify every 1 delay 0 check yes
group tiny id 1 5 9
group type2 type 2
plugin load "{args.plugin}"
fix m all metad EXECUTOR {executor} GAUSSIAN 0.1 0.1 10 PACE 0 &
    {' &'.join(declarations)} &
    {' '.join(f'SYMBOL {name} {name}.{suffix}' for name in calculations)} &
    CV_dim {dimensions} {' '.join(f'DIM {index+1} {limits[0]} {limits[1]} {limits[2]} "{expression}"' for index,expression in enumerate(expressions))} &
    METAD_RESTART 0 WT 0
compute pressure all pressure NULL virial
thermo 1
thermo_style custom step {' '.join(f'f_m[{index+1}]' for index in range(dimensions))} f_m c_pressure[1] c_pressure[2] c_pressure[3] c_pressure[4] c_pressure[5] c_pressure[6]
thermo_modify norm no format float %.17g
run 0
"""
    configuration = configuration.replace(' &CAL ', ' &\n    CAL ')
    if args.weave_plugin:
        configuration = f'plugin load "{args.weave_plugin}"\nweave/batch_comm\n' + configuration
    if rank==0: (directory/'in.lammps').write_text(configuration)
    simulation = lammps(cmdargs=['-screen','none','-log',str(directory/'log.lammps'),*shlex.split(args.lammps_args)],comm=comm)
    simulation.commands_string(configuration)
    probe = ctypes.CDLL(str(args.probe))
    pointer_type = ctypes.POINTER(ctypes.c_double)
    for name in ['fd_bias_gradient','fd_add_hill']:
        function=getattr(probe,name); function.argtypes=[ctypes.c_void_p,pointer_type];function.restype=ctypes.c_int
    probe.fd_probe_error.restype=ctypes.c_char_p
    def probe_call(name, values):
        payload=(ctypes.c_double*len(values))(*values)
        if getattr(probe,name)(simulation.lmp,payload): raise RuntimeError(probe.fd_probe_error().decode())
        return np.array(payload)
    def outputs():
        return np.array([simulation.extract_fix('m',0,1,nrow=index) for index in range(dimensions)])
    # 在偏离当前 CV 的位置放一个 hill；PACE=0 保持整个差分扫描的势函数不变。
    probe_call('fd_add_hill',outputs()+np.array([.06*(-1)**i for i in range(dimensions)]))
    simulation.command('run 0')
    def state():
        local_count=simulation.extract_setting('nlocal')
        local=np.column_stack([simulation.numpy.extract_atom('id')[:local_count],
                               simulation.numpy.extract_atom('x')[:local_count],
                               simulation.numpy.extract_atom('f')[:local_count]])
        rows=np.concatenate(comm.allgather(local));rows=rows[np.argsort(rows[:,0])]
        return dict(rows=rows,values=outputs(),energy=simulation.extract_fix('m',0,0),
                    virial=simulation.numpy.extract_compute('pressure',0,1).copy()*simulation.get_thermo('vol'))
    baseline=state(); coordinates=baseline['rows'][:,1:4].copy()
    seed=probe_call('fd_bias_gradient',np.zeros(dimensions))
    # 保存失败前的局部布局和非有限力位置，便于定位旧 CV 的编号/通信问题。
    diagnostics=dict(rank_values=comm.allgather(baseline['values'].tolist()),
                     owned_counts=comm.allgather(simulation.extract_setting('nlocal')),
                     nonfinite_force_indices=np.argwhere(~np.isfinite(baseline['rows'][:,4:7])).tolist())
    if rank==0: (directory/'diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    assert np.all(np.isfinite(seed)) and np.max(np.abs(seed))>1e-5, ('zero bias seed',seed)
    assert not diagnostics['nonfinite_force_indices'], ('nonfinite forces',diagnostics)
    assert np.max(np.abs(baseline['rows'][:,4:7]))>1e-9, 'fixture has no force signal'
    if dimensions==1: assert abs(seed[0])>1e-5
    reference_error=0.0
    if args.case.startswith('distance') and args.case != 'distance_mixed':
        def distance(first,second):
            delta=coordinates[second-1]-coordinates[first-1];delta-=np.rint(delta/box_length)*box_length
            return np.linalg.norm(delta)
        expected=distance(1,2) if 'linear' not in args.case else .7*distance(1,2)-.3*distance(1,17)
        reference_error=float(abs(baseline['values'][0]-expected))
        # 比较每个 rank 的同一全局距离，与直接坐标计算的参考值。
        rank_values=comm.allgather(float(baseline['values'][0]))
    else: rank_values=comm.allgather(baseline['values'].tolist())
    def place(value):
        payload=(ctypes.c_double*value.size)(*value.ravel())
        simulation.scatter_atoms('x',1,3,payload)
        simulation.command('run 0')
        return outputs(),float(simulation.extract_fix('m',0,0))
    # 单 rank 遍历全部 3N 坐标；多 rank 抽取跨子域中心与组外邻居。
    selected=range(len(ids)) if comm.size==1 else [0,1,4,7,9,16,24,31]
    details=[];curves=[]
    def error_statistics(analytic,numerical,atol,rtol):
        analytic=np.array(analytic);numerical=np.array(numerical)
        error=np.abs(analytic-numerical)
        score=error/(atol+rtol*np.maximum(np.abs(analytic),np.abs(numerical)))
        return dict(max_abs_error=float(error.max()),max_scaled_error=float(score.max()),
                    rms_error=float(np.sqrt(np.mean(error**2))),analytic_peak=float(np.max(np.abs(analytic))))
    for step in args.steps:
        coordinate_actual=[];coordinate_numeric=[];cv_actual=[];cv_numeric=[]
        for atom in selected:
            for axis in range(3):
                plus=coordinates.copy();minus=coordinates.copy();plus[atom,axis]+=step;minus[atom,axis]-=step
                yp,ep=place(plus);ym,em=place(minus)
                gradient=(ep-em)/(2*step);analytic=-baseline['rows'][atom,4+axis]
                coordinate_actual.append(analytic);coordinate_numeric.append(gradient)
                record=dict(kind='coordinate',step=step,atom_id=atom+1,axis=axis,analytic=analytic,numerical=gradient)
                if dimensions==1:
                    cv_actual.append(analytic/seed[0]);cv_numeric.append((yp[0]-ym[0])/(2*step))
                    record.update(cv_analytic=cv_actual[-1],cv_numerical=cv_numeric[-1])
                details.append(record)
        place(coordinates)
        strain_actual=[];strain_numeric=[];cv_strain_actual=[];cv_strain_numeric=[]
        for component,(row,column) in enumerate([(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]):
            energies=[];values=[]
            for sign in (1,-1):
                matrix=np.eye(3);matrix[row,column]+=sign*step;cell=box_length*matrix
                simulation.command(f'change_box all x final 0 {cell[0,0]:.17g} y final 0 {cell[1,1]:.17g} z final 0 {cell[2,2]:.17g} xy final {cell[0,1]:.17g} xz final {cell[0,2]:.17g} yz final {cell[1,2]:.17g} units box')
                value,energy=place(coordinates@matrix.T);energies.append(energy);values.append(value)
            analytic=-baseline['virial'][component];numerical=(energies[0]-energies[1])/(2*step)
            strain_actual.append(analytic);strain_numeric.append(numerical)
            record=dict(kind='strain',step=step,component=component,analytic=analytic,numerical=numerical)
            if dimensions==1:
                cv_strain_actual.append(analytic/seed[0]);cv_strain_numeric.append((values[0][0]-values[1][0])/(2*step))
                record.update(cv_analytic=cv_strain_actual[-1],cv_numerical=cv_strain_numeric[-1])
            details.append(record)
        simulation.command(f'change_box all x final 0 {box_length} y final 0 {box_length} z final 0 {box_length} xy final 0 xz final 0 yz final 0 units box')
        place(coordinates)
        curve=dict(step=step,force=error_statistics(coordinate_actual,coordinate_numeric,1e-8,2e-5),
                   virial=error_statistics(strain_actual,strain_numeric,1e-7,3e-5))
        if dimensions==1:
            curve['cv_gradient']=error_statistics(cv_actual,cv_numeric,2e-7,2e-5)
            curve['cv_strain']=error_statistics(cv_strain_actual,cv_strain_numeric,2e-6,3e-5)
        curves.append(curve)
    restored=state()
    restore_error=max(float(np.max(np.abs(restored[key]-baseline[key]))) for key in ('rows','values','virial'))
    # 两个较小步长均须通过，避免只挑一个碰巧接近的差分步长。
    restore_passed = restore_error < 1e-10
    if executor == 'LEGACY':
        # exprtk::derivative 使用数值差分；坐标末位的舍入会被差分放大。
        # 坐标/CV 值仍严格恢复，力和 virial 沿用本测试的导数精度要求。
        restore_passed = (
            np.max(np.abs(restored['rows'][:,:4]-baseline['rows'][:,:4])) < 1e-10
            and np.max(np.abs(restored['values']-baseline['values'])) < 1e-10
            and error_statistics(restored['rows'][:,4:7], baseline['rows'][:,4:7],
                                 1e-8, 2e-5)['max_scaled_error'] <= 1
            and error_statistics(restored['virial'], baseline['virial'],
                                 1e-7, 3e-5)['max_scaled_error'] <= 1)
    passed=reference_error<2e-11 and restore_passed and all(
        measurement['max_scaled_error']<=1 for curve in curves[-2:]
        for name,measurement in curve.items() if name!='step')
    report=dict(case=args.case,ranks=comm.size,passed=passed,executor=executor,expressions=expressions,
                coordinates_checked=len(list(selected))*3,seed=seed.tolist(),values=baseline['values'].tolist(),
                reference_error=reference_error,rank_values=rank_values,restore_error=restore_error,restore_passed=bool(restore_passed),curves=curves)
    simulation.close()
    if rank==0:
        (directory/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        (directory/'details.json').write_text(json.dumps(details,indent=2)+'\n')
        saved={key:value.tolist() if hasattr(value,'tolist') else value for key,value in baseline.items()}
        if dimensions==1:
            saved['cv_gradient']=(-baseline['rows'][:,4:7]/seed[0]).tolist()
            saved['cv_strain']=(-baseline['virial']/seed[0]).tolist()
        (directory/'state.json').write_text(json.dumps(saved,indent=2)+'\n')
        print(json.dumps(report),flush=True)
    # close() 只销毁 LAMMPS 实例；独立 Python worker 负责结束 Kokkos 运行时。
    simulation.lib.lammps_kokkos_finalize()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plugin',type=Path,default=None)
    parser.add_argument('--weave-plugin',type=Path,help='Matching optional Weave communicator')
    parser.add_argument('--probe',type=Path)
    parser.add_argument('--mpi',type=Path,default=None)
    parser.add_argument('--lammps-src',type=Path,default=None)
    parser.add_argument('--mpicxx',type=Path)
    parser.add_argument('--mpi-args',default='')
    parser.add_argument('--lammps-args',default='',help='Extra LAMMPS arguments, e.g. -k on g 1 -sf kk -pk kokkos gpu/aware off')
    parser.add_argument('--np',type=int,nargs='+',default=[1,2,4])
    parser.add_argument('--steps',type=float,nargs='+',default=STEPS)
    parser.add_argument('--cases',nargs='+',choices=CASES,default=CASES)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--case',default='q3')
    args=parser.parse_args();args.output=args.output.resolve();args.plugin=plugin_path(args.plugin)
    if args.weave_plugin: args.weave_plugin=args.weave_plugin.resolve()
    if args.worker: return worker(args)
    if min(args.np)<1: parser.error('--np requires positive ranks')
    try:
        args.mpi,args.mpicxx=mpi_tools(args.plugin,args.mpi,args.mpicxx)
        args.lammps_src=lammps_source(args.plugin,args.lammps_src)
    except ValueError as error: parser.error(str(error))
    args.output.mkdir(parents=True,exist_ok=False)
    args.probe=args.output/'bias_probe.so'
    env=os.environ.copy();env.setdefault('OMPI_CXX',cmake_cache(args.plugin.parent).get('CLEAN_HOST_COMPILER','g++'));env.setdefault('OMP_NUM_THREADS','1')
    subprocess.run([str(args.mpicxx),'-std=c++17','-shared','-fPIC',str(Path(__file__).with_name('bias_probe.cpp')),
                    '-I'+str(args.lammps_src),'-I'+str(ROOT/'include/core'),str(args.plugin),'-o',str(args.probe)],env=env,check=True)
    results=[]
    for case in args.cases:
        for ranks in args.np:
            directory=args.output/f'{case}_np{ranks}';directory.mkdir()
            command=[str(args.mpi),*shlex.split(args.mpi_args),'-np',str(ranks),sys.executable,str(Path(__file__).resolve()),
                     '--worker','--case',case,'--output',str(directory),'--plugin',str(args.plugin),'--probe',str(args.probe),
                     '--lammps-args='+args.lammps_args, '--steps',*map(str,args.steps)]
            if args.weave_plugin: command += ['--weave-plugin',str(args.weave_plugin)]
            with (directory/'screen.txt').open('w') as output:
                try: process=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,timeout=240)
                except subprocess.TimeoutExpired: process=None
            path=directory/'result.json'
            result=json.loads(path.read_text()) if path.exists() else dict(case=case,ranks=ranks,passed=False,error=(directory/'screen.txt').read_text(errors='replace')[-3000:])
            result['exit_code']=process.returncode if process else 'timeout'
            result['passed']=result['passed'] and result['exit_code']==0
            results.append(result)
            (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
            print(case,ranks,'PASS' if result['passed'] else 'FAIL',flush=True)
    # 同一几何下独立 CV 梯度的加权和，与线性组合的一次 Graph 反向比较。
    import numpy as np
    linear_results=[]
    for case,coefficients in LINEAR.items():
        for ranks in args.np:
            required=[args.output/f'{name}_np{ranks}'/'state.json' for name in [case,*coefficients]]
            if not all(path.exists() for path in required): continue
            states=[json.loads(path.read_text()) for path in required]
            entry=dict(case=case,ranks=ranks)
            for key in ['values','cv_gradient','cv_strain']:
                expected=sum(weight*np.array(state[key]) for weight,state in zip(coefficients.values(),states[1:]))
                entry[key+'_error']=float(np.max(np.abs(np.array(states[0][key])-expected)))
            entry['passed']=max(entry[key+'_error'] for key in ['values','cv_gradient','cv_strain'])<2e-10
            linear_results.append(entry)
    (args.output/'linear_results.json').write_text(json.dumps(linear_results,indent=2)+'\n')
    metadata=dict(plugin_sha256=hashlib.sha256(args.plugin.read_bytes()).hexdigest(),steps=args.steps,
                  cases=len(results),passed=sum(row['passed'] for row in results),linear_checks=linear_results,
                  scope='Compiled non-ML CVs; DISTANCE covers LEGACY, Global Graph and mixed Global/PerAtom Graph')
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata),flush=True)
    return 0 if all(row['passed'] for row in results+linear_results) else 1


if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        if '--worker' in sys.argv:
            from mpi4py import MPI
            MPI.COMM_WORLD.Abort(1)
        raise
