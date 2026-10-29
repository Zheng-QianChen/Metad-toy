#!/usr/bin/env python3
"""Run the merged source against an explicitly pinned LAMMPS/Weave runtime.

Each invocation creates a new result directory. It never replaces a production
plugin. Run build.py first and supply the same dependency paths here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[2])
    for name in ('build','output','host-build','mpi-prefix','python-package','assets'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--weave-library',type=Path)
    p.add_argument('--cuda',type=Path,default=Path('/usr/local/cuda'))
    p.add_argument('--cmake',default='cmake')
    p.add_argument('--ctest',default='ctest')
    p.add_argument('--lammps-source',type=Path)
    p.add_argument('--phase',choices=['ctest','al3sc','probe','global','history'],required=True)
    p.add_argument('--profile',choices=['weave','fallback','kokkos','host'],default='weave')
    p.add_argument('--fixture',type=Path,help='Existing Al3Sc input directory for probe phase')
    p.add_argument('--np',type=int,nargs='+',default=[1,2])
    a=p.parse_args()
    if a.profile in ('weave','fallback') and a.weave_library is None:
        p.error('--profile weave/fallback requires --weave-library')
    a.output.mkdir(parents=True,exist_ok=False)
    plugin=a.build/'fix_crystallize_plugin.so'
    env=os.environ.copy()
    library_dirs=[str(a.host_build),str(a.mpi_prefix/'lib'),str(a.cuda/'lib64')]
    if a.profile in ('weave','fallback'):library_dirs.insert(0,str(a.weave_library.parent))
    env.update(PATH=f'{a.cuda}/bin:{a.mpi_prefix}/bin:'+env.get('PATH',''),
        LD_LIBRARY_PATH=':'.join(library_dirs),
        PYTHONPATH=str(a.python_package),LAMMPS_LIBRARY_PATH=str(a.host_build),
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMPI_MCA_btl='self,tcp',OMPI_MCA_pml='ob1',
        OMPI_MCA_opal_cuda_support='true',WEAVE_METAD_DEVICE_COMMIT='0' if a.profile=='fallback' else '1')
    kk='-k on g 1 -sf kk -pk kokkos comm device gpu/aware on neigh half newton on'
    largs='' if a.profile=='host' else kk
    weave=['--weave-plugin',str(a.weave_library)] if a.profile in ('weave','fallback') else []
    mpi=[str(a.mpi_prefix/'bin/mpirun')]
    results=[]
    def run(name,cmd,timeout=600):
        started=time.monotonic()
        with (a.output/(name+'.log')).open('w') as f:
            try: code=subprocess.run(list(map(str,cmd)),env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout).returncode
            except subprocess.TimeoutExpired: code='timeout'
        results.append(dict(name=name,command=list(map(str,cmd)),exit_code=code,seconds=time.monotonic()-started))
        (a.output/'validation.json').write_text(json.dumps(dict(profile=a.profile,phase=a.phase,
            plugin_sha256=hashlib.sha256(plugin.read_bytes()).hexdigest(),results=results),indent=2)+'\n')
        print(name,'PASS' if code==0 else 'FAIL',flush=True)
        if code: print((a.output/(name+'.log')).read_text(errors='replace')[-2500:],flush=True)
        return code==0
    if a.phase=='ctest':
        run('ctest',[a.ctest,'--test-dir',a.build,'--output-on-failure','--parallel','1'],timeout=1200)
    elif a.phase=='al3sc':
        for ranks in a.np:
            for kind in ('raw','gated','empty'):
                # Full FD for raw on all ranks and gated on one rank; empty rank
                # correctness is also exercised by the independent force probe.
                if kind=='gated' and ranks!=1: continue
                name=f'{kind}-np{ranks}'
                run(name,[*mpi,'-np',ranks,sys.executable,a.source/'benchmarks/shared_frozen/al3sc.py',
                    '--plugin',plugin,'--assets',a.assets,'--output',a.output/name,
                    '--lammps-args='+largs,'--steps','10',*weave,
                    *(['--raw-product'] if kind!='gated' else []),
                    *(['--single','--skip-fd'] if kind=='empty' else [])])
        first=a.output/f'raw-np{a.np[0]}'/'restart.final'
        if first.exists():
            run('restart',[*mpi,'-np',a.np[-1],sys.executable,a.source/'benchmarks/shared_frozen/al3sc.py',
                '--plugin',plugin,'--assets',a.assets,'--output',a.output/'restart',
                '--lammps-args='+largs,'--steps','10',*weave,'--raw-product','--skip-fd','--restart',first])
    elif a.phase=='probe':
        if not a.fixture: p.error('--fixture required for probe')
        if run('build-probe',[a.cmake,'--build',a.build,'--target','zqc_force_batch_probe','--parallel','4']):
            for ranks in a.np:
                for empty in (False,True):
                    name=f'probe-np{ranks}'+('-empty' if empty else '')
                    run(name,[*mpi,'-np',ranks,sys.executable,a.source/'benchmarks/finite_difference/force_batch.py',
                        '--plugin',plugin,'--probe',a.build/'zqc_force_batch_probe.so',
                        '--fixture',a.fixture,'--output',a.output/name,'--lammps-args='+largs,
                        *(['--single-center'] if empty else [])])
    elif a.phase=='history':
        if not a.fixture: p.error('--fixture required for history')
        for ranks in a.np:
            name=f'history-np{ranks}'
            run(name,[*mpi,'-np',ranks,sys.executable,a.source/'benchmarks/integration/history.py',
                '--plugin',plugin,'--fixture',a.fixture,'--output',a.output/name,'--lammps-args='+largs])
    else:
        if not a.lammps_source: p.error('--lammps-source required for global')
        run('global',[sys.executable,a.source/'benchmarks/finite_difference/run.py',
            '--plugin',plugin,'--output',a.output/'global','--lammps-src',a.lammps_source/'src',
            '--mpi',a.mpi_prefix/'bin/mpirun','--mpicxx',a.mpi_prefix/'bin/mpicxx',
            '--np',*a.np,'--steps','1e-5','3e-6','--lammps-args='+largs,*weave,
            '--cases','q6','l6','distance','distance_linear','distance_graph','distance_mixed','nonlinear_graph'],timeout=1500)
    return int(any(x['exit_code']!=0 for x in results))

if __name__=='__main__': raise SystemExit(main())
