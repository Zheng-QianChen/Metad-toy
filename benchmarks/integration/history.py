#!/usr/bin/env python3
"""Global Q/LocalQL/CSS WT hills, NPT virial, and actual LAMMPS restart."""
import argparse,json,os,shlex,shutil
from pathlib import Path
import numpy as np
from mpi4py import MPI
from lammps import lammps

p=argparse.ArgumentParser(description=__doc__)
for name in ('fixture','output','plugin'): p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--lammps-args',default='')
a=p.parse_args();a.output=a.output.resolve();a.fixture=a.fixture.resolve();a.plugin=a.plugin.resolve()
comm=MPI.COMM_WORLD
if comm.rank==0:
    a.output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(a.fixture/'initial.data',a.output/'initial.data')
comm.Barrier();os.chdir(a.output)
prefix=(a.fixture/'in.lammps').read_text().split('fix bias all metad')[0]
prefix='\n'.join(f'plugin load {a.plugin}' if line.startswith('plugin load ') and
    Path(shlex.split(line)[2]).name=='fix_crystallize_plugin.so' else line for line in prefix.splitlines())+'\n'
configuration='''fix bias all metad EXECUTOR GRAPH BIAS HILLS GAUSSIAN 0.05 0.001 10 TEMP 100 PACE 2 &
 CAL NAME R SW_FUNC FERMI r_0 3.50 alpha -20 &
 CAL NAME C SW_FUNC FERMI r_0 3.30 alpha -20 &
 CAL NAME Q STEINH Q 6 Sc cutoff_eps_r 1e-12 SW_FUNC_r R &
 CAL NAME L STEINH L 6 Sc cutoff_eps_r 1e-12 SW_FUNC_r R &
 CAL NAME S CSS Sc CHANNEL 2 C 0 0.45 cutoff_eps 1e-12 &
 SYMBOL z Q.RAW SYMBOL q L.RAW SYMBOL c S.RAW &
 CV_dim 1 DIM_EXPR 1 "linear(AVE(z),0.2,0)+AVE(q*c)" &
 HILLS_FILE bias.hills WT 1 METAD_RESTART RESTART_FLAG
fix_modify bias energy yes virial yes
compute bp all pressure NULL fix
timestep 0.0002
fix integrate all npt temp 100 100 0.1 iso 0 0 1
thermo 1
thermo_style custom step pe f_bias f_bias[1] vol c_bp[1] c_bp[2] c_bp[3] c_bp[4] c_bp[5] c_bp[6]
thermo_modify norm no format float %.17g
'''
def create(text,name):
    s=lammps(cmdargs=['-screen','none','-log',name+'.log',*shlex.split(a.lammps_args)],comm=comm)
    s.commands_string(text)
    return s
def state(s):
    n=s.extract_setting('nlocal')
    rows=np.concatenate(comm.allgather(np.column_stack([s.numpy.extract_atom('id')[:n],
        s.numpy.extract_atom('x')[:n],s.numpy.extract_atom('f')[:n]])))
    rows=rows[np.argsort(rows[:,0])]
    return dict(rows=rows.tolist(),energy=float(s.extract_fix('bias',0,0)),
        cv=float(s.extract_fix('bias',0,1,nrow=0)),volume=float(s.get_thermo('vol')),
        virial=(s.numpy.extract_compute('bp',0,1)*s.get_thermo('vol')/s.extract_global('nktv2p')).tolist())
s=None
try:
    s=create(prefix+configuration.replace('RESTART_FLAG','0')+'velocity all create 100 872635 mom yes rot no\n','first')
    s.command('run 8');checkpoint=state(s)
    history=Path('bias.hills').read_bytes()
    s.command('run 0')
    if Path('bias.hills').read_bytes()!=history: raise AssertionError('run 0 deposited hills')
    s.command('write_restart checkpoint');s.command('run 4');continuous=state(s)
    final_history=Path('bias.hills').read_bytes()
    s.close();s=None
    if comm.rank==0: Path('bias.hills').write_bytes(history)
    comm.Barrier()
    resumed=prefix.replace('read_data initial.data','read_restart checkpoint')
    # Recreate NPT before the first run: an intervening run 0 discards unused
    # restart fix state (thermostat/barostat), changing the continued trajectory.
    resumed='\n'.join(line for line in resumed.splitlines() if not line.startswith('run '))+'\n'
    s=create(resumed+configuration.replace('RESTART_FLAG','1'),'restart')
    s.command('run 0')
    if Path('bias.hills').read_bytes()!=history: raise AssertionError('restart setup deposited hills')
    s.command('run 4');continued=state(s)
    errors={k:float(np.max(np.abs(np.array(continuous[k])-continued[k]))) for k in continuous}
    for k in continuous:
        if not np.allclose(continuous[k],continued[k],rtol=2e-10,atol=2e-10): raise AssertionError((k,errors))
    # Floating-point reductions may change final digits after atom redistribution.
    h1=np.loadtxt(final_history.decode().splitlines());h2=np.loadtxt('bias.hills')
    if h1.shape!=h2.shape or not np.allclose(h1,h2,rtol=2e-10,atol=2e-12): raise AssertionError('restart hills differ')
    if h2.shape[0]!=6: raise AssertionError(('hill count',h2.shape))
    if comm.rank==0:
        Path('result.json').write_text(json.dumps(dict(passed=True,ranks=comm.size,hills=6,
            restart_errors=errors,checkpoint=checkpoint,continuous=continuous,continued=continued),indent=2)+'\n')
        print('PASS global WT hills + NPT + restart',errors,flush=True)
finally:
    if s is not None:
        s.close();s.lib.lammps_kokkos_finalize()
