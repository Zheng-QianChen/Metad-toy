#!/usr/bin/env python3
"""真实 GPUMD + libmetadtoy StruFactor 验证。
枚举多个周期像作为独立参考，核对值、总势能、力、逐原子 Full9 和应变差分。
Python 只生成输入与检查输出；GPUMD 每步计算全部由 C++/CUDA 完成。
"""
import argparse
import itertools
import json
import re
import tempfile
from pathlib import Path
import numpy as np
from run import read_xyz, run_executable, MODEL

def radial(radius, cutoff, wave):
    scaled = wave*radius
    value = np.sin(scaled)/scaled
    derivative = (scaled*np.cos(scaled)-np.sin(scaled))/(wave*radius*radius)
    t = max(0., (radius*radius-.8*cutoff*cutoff)/(.2*cutoff*cutoff))
    window = 1-t**3
    window_derivative = -6*t*t*radius/(.2*cutoff*cutoff)
    return value*window, derivative*window+value*window_derivative

def structure(positions, box, periodic, cutoff, wave, centers):
    values = np.ones(len(positions))
    pairs = []
    ranges = [range(-int(np.ceil(cutoff/length))-1,int(np.ceil(cutoff/length))+2)
              if p else [0] for length,p in zip(box,periodic)]
    # 独立逐像枚举，不使用生产候选表、skin 或 CSR。
    for center in centers:
        for neighbor in range(len(positions)):
            for shift in itertools.product(*ranges):
                if center==neighbor and not any(shift): continue
                delta=positions[neighbor]-positions[center]+box*np.array(shift)
                radius=np.linalg.norm(delta)
                if radius<1e-6 or radius>cutoff: continue
                value,derivative=radial(radius,cutoff,wave)
                values[center]+=value
                pairs.append((center,neighbor,delta,derivative/radius))
    return values,pairs

def reference(positions,box,periodic,kind):
    cutoff,wave=4.3,2.1
    members = [0] if kind in ('group','mixed') else [0,1]
    values,pairs=structure(positions,box,periodic,cutoff,wave,members)
    additional_pairs=[]
    distance_delta=positions[1]-positions[0]
    distance_delta-=box*np.rint(distance_delta/box)*periodic
    distance=np.linalg.norm(distance_delta)
    if kind=='mixed':
        second,additional_pairs=structure(positions,box,periodic,3.9,1.7,[1])
        value=float(values[0]*second[1]+distance)
        seeds=np.array([second[1],0.])
        additional_seed=values[0]
    elif kind=='raw':
        value=float(np.mean(values*values))
        seeds=values.copy()  # d mean(s_i*s_i)/d s_i = s_i for two atoms
    else:
        value=float(np.mean(values[members]))
        seeds=np.zeros(2);seeds[members]=1./len(members)
    offset=value-.4
    derivative=1.7*offset
    forces=np.zeros((2,3));virial=np.zeros((2,3,3))
    energy=.5*1.7*offset**2
    for center,neighbor,delta,derivative_over_radius in pairs:
        force=derivative*seeds[center]*derivative_over_radius*delta
        forces[center]+=force;forces[neighbor]-=force
        tensor=-.5*np.outer(delta,force)
        virial[center]+=tensor;virial[neighbor]+=tensor
    if kind=='mixed':
        for center,neighbor,delta,derivative_over_radius in additional_pairs:
            force=derivative*additional_seed*derivative_over_radius*delta
            forces[center]+=force;forces[neighbor]-=force
            tensor=-.5*np.outer(delta,force)
            virial[center]+=tensor;virial[neighbor]+=tensor
        force=derivative*distance_delta/distance
        forces[0]+=force;forces[1]-=force
        virial-=.5*np.outer(distance_delta,force)
    # 验证底势为 GPUMD 的两原子 Morse 夹具（按最小像定义）。
    delta=positions[1]-positions[0]
    delta-=box*np.rint(delta/box)*periodic
    radius=np.linalg.norm(delta)
    exponential=np.exp(-MODEL['alpha_inverse_A']*(radius-MODEL['equilibrium_A']))
    energy+=MODEL['depth_eV']*(1-exponential)**2-MODEL['depth_eV']
    slope=2*MODEL['depth_eV']*MODEL['alpha_inverse_A']*exponential*(1-exponential)
    force=slope*delta/radius
    forces[0]+=force;forces[1]-=force
    virial-=.5*np.outer(delta,force)
    return value,energy,forces,virial.reshape(2,9)

def case(executable,root,name,positions,box,periodic,kind='mean',skin=.3,steps=1):
    directory=root/name;directory.mkdir()
    lattice=np.diag(box).reshape(-1)
    header='Lattice="%s" pbc="%s" Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3' % (
        ' '.join('%.17g'%x for x in lattice),' '.join('T' if p else 'F' for p in periodic))
    (directory/'model.xyz').write_text('2\n'+header+'\n'+''.join(
        'O '+' '.join('%.17g'%x for x in position)+' 15.999 0 0 0\n' for position in positions))
    (directory/'morse.txt').write_text('metadtoy_morse 1 O\n5.2 2.65 1.21 2.0\n')
    group='left' if kind=='group' else 'all'
    reduction='raw' if kind=='raw' else 'mean'
    expression='(s*s).AVE' if kind=='raw' else 's'
    (directory/'sampling.in').write_text(
        'backend metadtoy\nstart fresh\noutput COLVAR 1\nneighbor_skin %.17g\n'%skin+
        ('group left atoms 0\n' if kind=='group' else '')+
        'cv s strufactor group %s cutoff 4.3 q 2.1 reduce %s\n'%(group,reduction)+
        'expression %s\nbias harmonic center .4 kappa 1.7\n'%expression)
    if kind=='mixed':
        (directory/'sampling.in').write_text(
            'backend metadtoy\nstart fresh\noutput COLVAR 1\nneighbor_skin %.17g\n'%skin+
            'group left atoms 0\ngroup right atoms 1\n'
            'cv a strufactor group left cutoff 4.3 q 2.1 reduce mean\n'
            'cv b strufactor group right cutoff 3.9 q 1.7 reduce mean\n'
            'cv r distance atoms 0 1\nexpression a*b+r\n'
            'bias harmonic center .4 kappa 1.7\n')
    (directory/'run.in').write_text('potential morse.txt\ntime_step 0.000001\nensemble nve\n'
        'enhanced_sampling sampling.in\ndump_xyz 1 state.xyz precision double force potential virial\nrun %d\n'%steps)
    run_executable(executable,directory)
    result=read_xyz(directory/'state.xyz')
    colvar=np.loadtxt(directory/'COLVAR',ndmin=2)
    expected=reference(result['pos'],box,np.asarray(periodic),kind)
    errors=dict(value=float(abs(colvar[1,1]-expected[0])),
        energy=float(abs(result['energy_atom'].sum()-expected[1])),
        force=float(np.max(abs(result['forces']-expected[2]))),
        virial=float(np.max(abs(result['virial']-expected[3]))))
    if max(errors.values())>3e-8: raise AssertionError((name,errors))
    log=(directory/'gpumd.log').read_text()
    rebuilds=re.search(r'\[MetaD-toy/neighbors\] (\d+) candidate rebuilds',log)
    return result,errors,colvar,int(rebuilds[1]) if rebuilds else None

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpumd',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    root=args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix='gpumd-strufactor-',dir=Path.cwd()))
    if args.output:root.mkdir(parents=True,exist_ok=False)
    executable=args.gpumd.resolve()
    positions=np.array([[.15,.25,.3],[1.19,.62,.47]])
    box=np.array([2.8,3.2,3.7]); periodic=np.array([1,1,1])
    metrics={}
    for kind in ('mean','group','mixed','raw'):
        result,errors,_,_=case(executable,root,kind,positions,box,periodic,kind)
        metrics[kind]=errors
        if kind=='raw':
            epsilon=2e-6
            finite=[]
            for axis in range(3):
                energies=[]
                for sign in (-1,1):
                    moved=positions.copy();moved[0,axis]+=sign*epsilon
                    output,*_=case(executable,root,'fd_%d_%d'%(axis,sign),moved,box,periodic,kind)
                    energies.append(output['energy_atom'].sum())
                finite.append(-(energies[1]-energies[0])/(2*epsilon))
            error=float(np.max(abs(np.array(finite)-result['forces'][0])))
            if error>3e-6: raise AssertionError(('force finite difference',error))
            metrics['finite_difference_force']=error
            # 同时缩放盒子和原子位置，包含自像对胞应变的贡献。
            strain_errors=[]
            for axis in range(3):
                energies=[]
                for sign in (-1,1):
                    moved=positions.copy();lengths=box.copy()
                    moved[:,axis]*=1+sign*epsilon;lengths[axis]*=1+sign*epsilon
                    output,*_=case(executable,root,'strain_%d_%d'%(axis,sign),moved,lengths,periodic,kind)
                    energies.append(output['energy_atom'].sum())
                numeric=-(energies[1]-energies[0])/(2*epsilon)
                strain_errors.append(abs(numeric-result['virial'][:,4*axis].sum()))
            if max(strain_errors)>3e-6: raise AssertionError(('strain virial',strain_errors))
            metrics['finite_difference_diagonal_virial']=max(strain_errors)
    _,metrics['nonperiodic'],_,_=case(executable,root,'nonperiodic',positions,box,[0,0,0])
    _,metrics['mixed_pbc'],_,_=case(executable,root,'mixed_pbc',positions,box,[1,0,1])
    _,_,cached,rebuilds=case(executable,root,'cached',positions,box,periodic,steps=200)
    _,_,rebuilt,always=case(executable,root,'always',positions,box,periodic,skin=0,steps=200)
    error=float(np.max(abs(cached-rebuilt)))
    if error>1e-10 or rebuilds!=1 or always!=201: raise AssertionError(('cache',error,rebuilds,always))
    metrics['cache']=dict(max_colvar_difference=error,cached_rebuilds=rebuilds,uncached_rebuilds=always)
    (root/'checks.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print('Results: '+str(root)); print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
