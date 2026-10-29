#!/usr/bin/env python3
"""GPUMD DISTANCE integration checks and O2/Morse WTMetaD fixture.

Runs the real gpumd_metadtoy executable in a newly created directory. All
simulation work is C++/CUDA; Python only prepares inputs and checks outputs.
"""
import argparse
import hashlib
import json
import re
import subprocess
import time
import tempfile
from pathlib import Path
import numpy as np

MODEL = dict(depth_eV=5.2, alpha_inverse_A=2.65, equilibrium_A=1.21, cutoff_A=8.0)
SAMPLING = dict(temperature_K=300.0, bias_factor=20.0, sigma_A=.015,
                height_eV=.02, pace=100, timestep_fs=.2, warmup_steps=20000,
                grid_min_A=.85, grid_max_A=3.0, grid_bins=4300,
                comparison_min_A=1.05, comparison_max_A=1.60,
                rmse_limit_eV=.025, max_error_limit_eV=.075)

def write_model(directory, positions, zero_velocities=True):
    lines = ['2', 'Lattice="20 0 0 0 20 0 0 0 20" pbc="T T T" '
             'Properties=species:S:1:pos:R:3:mass:R:1' + (':vel:R:3' if zero_velocities else '')]
    for position in positions:
        lines.append("O " + " ".join("%.17g" % v for v in position) + " 15.999" + (" 0 0 0" if zero_velocities else ""))
    (directory/'model.xyz').write_text("\n".join(lines)+"\n")
    (directory/'morse.txt').write_text("metadtoy_morse 1 O\n5.2 2.65 1.21 8.0\n")

def run_executable(executable, directory):
    start = time.perf_counter()
    with (directory/'gpumd.log').open('w') as log:
        result = subprocess.run([str(executable)], cwd=directory, stdout=log,
                                stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError("GPUMD failed in %s:\n%s" %
                           (directory, (directory/'gpumd.log').read_text()[-5000:]))
    return time.perf_counter()-start

def read_xyz(path):
    lines = path.read_text().splitlines()
    count = int(lines[0])
    properties = re.search(r"Properties=(\S+)", lines[1])[1].split(":")
    rows = [row.split() for row in lines[2:2+count]]
    result = {}
    offset = 0
    for name, kind, width in zip(properties[::3], properties[1::3], properties[2::3]):
        width = int(width)
        if kind != 'S':
            result[name] = np.array([[float(v) for v in row[offset:offset+width]] for row in rows])
        offset += width
    return result

def static_case(executable, root, name, positions, mode):
    directory = root/name
    directory.mkdir()
    write_model(directory, positions)
    if mode == 'native':
        config = ("output COLVAR 1\nstart fresh\ncv r distance atoms 0 1 pbc on\n"
                  "bias restraint harmonic arg r center 1.21 kappa 8\n")
    elif mode == 'none':
        config = None
    else:
        expression = {'bridge':'r', 'square':'r*r', 'shared':'a*b+a'}[mode]
        cvs = ("cv a distance atoms 0 1\ncv b distance atoms 0 1\n" if mode=='shared'
               else "cv r distance atoms 0 1\n")
        center, kappa = (1.21,8) if mode=='bridge' else (1.5,3)
        config = ("backend metadtoy\nstart fresh\noutput COLVAR 1\n" + cvs +
                  "expression %s\nbias harmonic center %g kappa %g\n" % (expression,center,kappa))
    if config:
        (directory/'sampling.in').write_text(config)
    (directory/'run.in').write_text("potential morse.txt\ntime_step 0.000001\nensemble nve\n" +
        ("enhanced_sampling sampling.in\n" if config else "") +
        "dump_xyz 1 state.xyz precision double force potential virial\nrun 1\n")
    run_executable(executable,directory)
    return read_xyz(directory/'state.xyz')

def morse_values(positions):
    delta = positions[1]-positions[0]
    delta -= 20*np.rint(delta/20)
    radius = np.linalg.norm(delta)
    exponential = np.exp(-MODEL['alpha_inverse_A']*(radius-MODEL['equilibrium_A']))
    energy = MODEL['depth_eV']*(1-exponential)**2-MODEL['depth_eV']
    derivative = 2*MODEL['depth_eV']*MODEL['alpha_inverse_A']*exponential*(1-exponential)
    return radius,delta,energy,derivative

def check_static(executable, root):
    positions=np.array([[9.3,10.1,9.6],[10.45,10.34,9.29]])
    modes={}
    metrics={}
    for mode in ('none','native','bridge','square','shared'):
        result=static_case(executable,root,mode,positions,mode)
        modes[mode]=result
        radius,delta,energy,derivative=morse_values(result['pos'])
        if mode in ('native','bridge'):
            energy+=4*(radius-1.21)**2
            derivative+=8*(radius-1.21)
        elif mode in ('square','shared'):
            value=radius**2+(radius if mode=='shared' else 0)
            chain=2*radius+(1 if mode=='shared' else 0)
            energy+=1.5*(value-1.5)**2
            derivative+=3*(value-1.5)*chain
        first_force=derivative*delta/radius
        forces=np.array([first_force,-first_force])
        tensor=-.5*np.outer(delta,first_force)
        # dump_xyz 将原生 Full9 转为行主序 xx xy xz yx yy yz zx zy zz。
        full9=tensor.reshape(-1)
        errors=dict(energy=float(abs(result['energy_atom'].sum()-energy)),
                    force=float(np.max(abs(result['forces']-forces))),
                    virial=float(np.max(abs(result['virial']-full9))))
        if max(errors.values())>2e-9:
            raise AssertionError((mode,errors))
        metrics[mode]=errors
    for key in ('pos','forces','energy_atom','virial'):
        error=float(np.max(abs(modes['native'][key]-modes['bridge'][key])))
        if error>2e-10: raise AssertionError(('native vs bridge',key,error))
    # 真实 GPUMD 总势能的中心差分，检查非线性 Graph 与两个 CV 共享原子时的反传。
    epsilon=1e-5
    fd=[]
    for axis in range(3):
        energies=[]
        for direction in (-1,1):
            displaced=positions.copy()
            displaced[0,axis]+=direction*epsilon
            result=static_case(executable,root,'fd_%d_%d'%(axis,direction),displaced,'shared')
            energies.append(result['energy_atom'].sum())
        fd.append(-(energies[1]-energies[0])/(2*epsilon))
    error=float(np.max(abs(np.array(fd)-modes['shared']['forces'][0])))
    if error>2e-6: raise AssertionError(('finite difference',error))
    metrics['finite_difference_force_max_eV_per_A']=error
    return metrics

def sample(executable, root, steps, seed, warmup):
    directory=root/('seed_%d'%seed)
    directory.mkdir()
    write_model(directory,np.array([[9.395,10,10],[10.605,10,10]]),zero_velocities=False)
    s=SAMPLING
    (directory/'sampling.in').write_text(
        "backend metadtoy\nstart fresh\noutput COLVAR 10\n"
        "cv r distance atoms 0 1\nexpression r\nwarmup %d\n" % warmup +
        "bias metad temperature 300 sigma .015 height .02 biasfactor 20 pace 100\n"
        "grid .85 3.0 4300\nhistory HILLS\n")
    (directory/'run.in').write_text(
        "potential morse.txt\nvelocity 300 seed %d\ntime_step .2\n"
        "ensemble nvt_lan 300 300 100\nenhanced_sampling sampling.in\n"
        "dump_thermo 1000\nrun %d\n" % (seed,steps))
    elapsed=run_executable(executable,directory)
    hills=np.loadtxt(directory/'HILLS',ndmin=2)
    colvar=np.loadtxt(directory/'COLVAR',ndmin=2)
    expected=np.arange((warmup//s['pace']+1)*s['pace'],steps+1,s['pace'])
    if not np.array_equal(hills[:,0],expected): raise AssertionError('Deposition lifecycle/PACE mismatch')
    if not np.isfinite(colvar).all() or not np.isfinite(hills).all(): raise AssertionError('Nonfinite sampling output')
    return dict(seed=seed,steps=steps,hills=len(hills),seconds=elapsed,
                hills_sha256=hashlib.sha256((directory/'HILLS').read_bytes()).hexdigest(),
                colvar_sha256=hashlib.sha256((directory/'COLVAR').read_bytes()).hexdigest(),
                ms_per_step=elapsed/steps*1000,
                distance_min_A=float(colvar[:,1].min()),distance_max_A=float(colvar[:,1].max()))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpumd',type=Path,required=True)
    parser.add_argument('--output',type=Path,help='New directory; omitted creates a unique directory in the working directory')
    parser.add_argument('--mode',choices=['smoke','fes'],default='smoke')
    parser.add_argument('--steps',type=int,default=2000000)
    parser.add_argument('--seed',type=int,default=73517)
    args=parser.parse_args()
    if args.seed<=0: parser.error('--seed must be positive')
    if args.mode=='fes' and args.steps<=SAMPLING['warmup_steps']+SAMPLING['pace']:
        parser.error('--steps must exceed warmup plus one deposition interval')
    root=args.output.resolve() if args.output else Path(tempfile.mkdtemp(prefix="gpumd-check-",dir=Path.cwd()))
    if args.output: root.mkdir(parents=True,exist_ok=False)
    executable=args.gpumd.resolve()
    (root/'model.json').write_text(json.dumps(MODEL,indent=2)+'\n')
    sampling=dict(SAMPLING)
    sampling['velocity_seed']=args.seed
    sampling['initial_velocities']='GPUMD velocity command (model.xyz omits velocities)'
    sampling['steps']=args.steps if args.mode=='fes' else 2000
    sampling['warmup_steps']=SAMPLING['warmup_steps'] if args.mode=='fes' else 200
    (root/'sampling.json').write_text(json.dumps(sampling,indent=2)+'\n')
    metrics={}
    if args.mode=='smoke': metrics['static']=check_static(executable,root)
    metrics['sampling']=sample(executable,root,sampling['steps'],args.seed,sampling['warmup_steps'])
    (root/'checks.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print("Results: "+str(root))
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':
    main()
