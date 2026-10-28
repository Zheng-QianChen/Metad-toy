#!/usr/bin/env python3
"""Real Al-Sc MEAM NPT with shared WT: SUM/MEAN, table oracle, run-0 and restart."""
import argparse,json,os,shlex,shutil
from pathlib import Path
import numpy as np
from mpi4py import MPI
from lammps import lammps
p=argparse.ArgumentParser(description=__doc__)
for name in ('fixture','output','plugin'):p.add_argument('--'+name,type=Path,required=True)
p.add_argument('--lammps-args',default='')
p.add_argument('--aggregation',choices=['SUM','MEAN'],default='MEAN')
p.add_argument('--single',action='store_true')
a=p.parse_args();a.output=a.output.resolve();a.fixture=a.fixture.resolve();a.plugin=a.plugin.resolve()
comm=MPI.COMM_WORLD
if comm.rank==0:
    a.output.mkdir(parents=True,exist_ok=False);shutil.copy2(a.fixture/'initial.data',a.output/'initial.data')
comm.Barrier();os.chdir(a.output)
prefix=(a.fixture/'in.lammps').read_text().split('fix bias all metad')[0]
prefix='\n'.join(f'plugin load {a.plugin}' if line.startswith('plugin load ') and Path(shlex.split(line)[2]).name=='fix_crystallize_plugin.so' else line for line in prefix.splitlines())+'\n'
if a.single:prefix=prefix.replace('group family type 2','group family id 1')
configuration=f'''fix bias all metad EXECUTOR GRAPH SHARED_ONLINE family -0.2 1.5 401 SHARED_AGGREGATION {a.aggregation} &
 GAUSSIAN 0.05 0.001 10 TEMP 100 PACE 2 &
 CAL NAME R SW_FUNC FERMI r_0 3.50 alpha -20 &
 CAL NAME C SW_FUNC FERMI r_0 3.30 alpha -20 &
 CAL NAME Q STEINH Q 6 Sc cutoff_eps_r 1e-12 SW_FUNC_r R &
 CAL NAME L STEINH L 6 Sc cutoff_eps_r 1e-12 SW_FUNC_r R &
 CAL NAME S CSS Sc CHANNEL 2 C 0 0.45 cutoff_eps 1e-12 &
 SYMBOL z Q.RAW SYMBOL q L.RAW SYMBOL c S.RAW &
 CV_dim 1 DIM_EXPR 1 "linear(z,0.2,0)+q*c" HILLS_FILE shared.history WT 1 METAD_RESTART RESTART_FLAG
fix_modify bias energy yes virial yes
compute raw all metad/atom bias Q.stein_q L.stein_q S.css
compute bp all pressure NULL fix
timestep 0.0002
fix integrate all npt temp 100 100 0.1 iso 0 0 1
thermo 1
thermo_style custom step pe f_bias f_bias[1] vol c_bp[1] c_bp[2] c_bp[3] c_bp[4] c_bp[5] c_bp[6]
thermo_modify norm no format float %.17g
'''
def create(text,name):
    if comm.rank==0:Path(name+'.in').write_text(text)
    s=lammps(cmdargs=['-screen','none','-log',name+'.log',*shlex.split(a.lammps_args)],comm=comm)
    try:s.commands_string(text)
    except BaseException:s.close();s.lib.lammps_kokkos_finalize();raise
    return s
def history(data=None):
    text=Path('shared.history').read_text() if data is None else data.decode()
    return np.array([[float(v) for v in line.split()[1:-1]] for line in text.splitlines() if line.startswith('STATE ')])
def state(s):
    n=s.extract_setting('nlocal')
    rows=np.concatenate(comm.allgather(np.column_stack([s.numpy.extract_atom('id')[:n],s.numpy.extract_atom('type')[:n],
        s.numpy.extract_atom('x')[:n],s.numpy.extract_atom('f')[:n],s.numpy.extract_compute('raw',1,2)[:n]])))
    rows=rows[np.argsort(rows[:,0])];mask=rows[:,0]==1 if a.single else rows[:,1]==2
    q=.2*rows[mask,8]+rows[mask,9]*rows[mask,10]
    table=history()[-1,2:].reshape(-1,2);spacing=1.7/400
    k=np.clip(np.floor((q+.2)/spacing).astype(int),0,399);t=(q-(-.2+k*spacing))/spacing
    values=(2*t**3-3*t**2+1)*table[k,0]+(t**3-2*t**2+t)*spacing*table[k,1]+(-2*t**3+3*t**2)*table[k+1,0]+(t**3-t**2)*spacing*table[k+1,1]
    expected=float(np.mean(values) if a.aggregation=='MEAN' else np.sum(values))
    energy=float(s.extract_fix('bias',0,0))
    if not np.isclose(energy,expected,rtol=2e-11,atol=2e-13):raise AssertionError(('shared energy',energy,expected))
    mean=float(s.extract_fix('bias',0,1,nrow=0))
    if not np.isclose(mean,np.mean(q),rtol=2e-11,atol=2e-13):raise AssertionError('family coordinate')
    return dict(rows=rows.tolist(),energy=energy,cv=mean,volume=float(s.get_thermo('vol')),
        virial=(s.numpy.extract_compute('bp',0,1)*s.get_thermo('vol')/s.extract_global('nktv2p')).tolist())
s=None;lib=None
try:
    s=create(prefix+configuration.replace('RESTART_FLAG','0')+'velocity all create 100 872635 mom yes rot no\n','first');lib=s.lib
    s.command('run 8');checkpoint=state(s);saved=Path('shared.history').read_bytes()
    # Recreate thermo: LAMMPS deliberately retains last_thermo() cache on a
    # same-step run 0 when only norm is changed.
    s.commands_string('thermo_style custom step f_bias\nthermo_modify norm yes\nrun 0')
    expected_thermo=checkpoint['energy']/(s.get_natoms() if a.aggregation=='SUM' else 1)
    if not np.isclose(s.last_thermo()['f_bias'],expected_thermo,rtol=1e-11,atol=1e-14):raise AssertionError('thermo extensivity')
    s.commands_string(next(line for line in configuration.splitlines() if line.startswith('thermo_style '))+'\nthermo_modify norm no format float %.17g')
    s.command('run 0');s.command('write_restart checkpoint')
    if Path('shared.history').read_bytes()!=saved:raise AssertionError('setup/write_restart deposited')
    s.command('run 4');continuous=state(s);final_history=Path('shared.history').read_bytes();s.close();s=None
    if comm.rank==0:
        Path('continuous.history').write_bytes(final_history);Path('shared.history').write_bytes(saved)
    comm.Barrier()
    resumed=prefix.replace('read_data initial.data','read_restart checkpoint')
    resumed='\n'.join(line for line in resumed.splitlines() if not line.startswith('run '))+'\n'
    s=create(resumed+configuration.replace('RESTART_FLAG','1'),'restart');s.command('run 0')
    if Path('shared.history').read_bytes()!=saved:raise AssertionError('restart setup deposited')
    s.command('run 4');continued=state(s)
    errors={k:float(np.max(np.abs(np.array(continuous[k])-continued[k]))) for k in continuous}
    for k in continuous:
        if not np.allclose(continuous[k],continued[k],rtol=2e-10,atol=2e-10):raise AssertionError((k,errors))
    h1,h2=history(final_history),history()
    if h1.shape!=(7,804) or h2.shape!=h1.shape or not np.allclose(h1,h2,rtol=2e-10,atol=2e-12):raise AssertionError(('history replay',h1.shape,h2.shape))
    if list(h2[1:,0])!=[2,4,6,8,10,12]:raise AssertionError('deposition cadence')
    if comm.rank==0:
        Path('result.json').write_text(json.dumps(dict(passed=True,ranks=comm.size,aggregation=a.aggregation,single=a.single,
            depositions=6,restart_errors=errors,history_max_error=float(np.max(np.abs(h1-h2))),checkpoint=checkpoint,continuous=continuous,continued=continued),indent=2)+'\n')
        print('PASS shared WT + NPT + restart',errors,flush=True)
finally:
    if s is not None:s.close()
    if lib is not None:lib.lammps_kokkos_finalize()
