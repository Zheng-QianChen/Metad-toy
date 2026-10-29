#!/usr/bin/env python3
"""Build a pinned integration profile without touching an installed runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[2])
    for name in ['output','lammps-source','host-build','cuda','mpi-prefix']:
        p.add_argument('--'+name,type=Path,required=True)
    for name in ['weave-source','weave-library']:
        p.add_argument('--'+name,type=Path)
    p.add_argument('--cmake',default='cmake');p.add_argument('--ninja',default='ninja')
    p.add_argument('--cxx',default='/usr/bin/g++');p.add_argument('--arch',required=True)
    p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--profile',choices=['weave','kokkos','host'],default='weave')
    args=p.parse_args()
    if args.profile=='weave' and (args.weave_source is None or args.weave_library is None):
        p.error('--profile weave requires --weave-source and --weave-library')
    args.output.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    library_dirs=[str(args.host_build),str(args.mpi_prefix/'lib'),str(args.cuda/'lib64')]
    if args.profile=='weave':library_dirs.insert(1,str(args.weave_library.parent))
    env.update(PATH=f'{args.cuda}/bin:{args.mpi_prefix}/bin:'+env.get('PATH',''),
               LD_LIBRARY_PATH=':'.join(library_dirs),
               OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    cmd=[args.cmake,'-S',str(args.source),'-B',str(args.output),'-G','Ninja',
         '-DCMAKE_MAKE_PROGRAM='+args.ninja,'-DCMAKE_BUILD_TYPE=Release',
         '-DZQC_ENABLE_CUDA=ON','-DZQC_BUILD_CVS=ON','-DZQC_BUILD_LAMMPS=ON',
         '-DZQC_BUILD_TESTS=ON','-DZQC_TEST_MPI=ON','-DUSE_ML_CVS=OFF',
         '-DZQC_BUILD_GPUMD=OFF','-DZQC_BUILD_GPUMD_RUNTIME=ON',
         '-DZQC_LAMMPS_KOKKOS='+('OFF' if args.profile=='host' else 'ON'),
         '-DZQC_LAMMPS_WEAVE='+('ON' if args.profile=='weave' else 'OFF'),
         '-DLAMMPS_SOURCE_DIR='+str(args.lammps_source/'src'),
         '-DLAMMPS_LIB='+str(args.host_build/'liblammps.so'),
         '-DCMAKE_CXX_COMPILER='+args.cxx,'-DCMAKE_CUDA_HOST_COMPILER='+args.cxx,
         '-DCMAKE_CUDA_COMPILER='+str(args.cuda/'bin/nvcc'),'-DCMAKE_CUDA_ARCHITECTURES='+args.arch,
         '-DMPI_CXX_COMPILER='+str(args.mpi_prefix/'bin/mpicxx'),
         '-DCMAKE_BUILD_RPATH='+env['LD_LIBRARY_PATH'].replace(':',';')]
    if args.profile!='host':
        cmd.append('-DKokkos_DIR='+str(args.host_build/'cmake_packages/Kokkos'))
    if args.profile=='weave':
        cmd.extend(['-DWEAVEMD_BATCH_SOURCE='+str(args.weave_source/'native/batch1'),
                    '-DWEAVEMD_PLUGIN_LIBRARY='+str(args.weave_library)])
    record={'profile':args.profile,'configure_command':cmd,'complete':False}
    def run(command,label):
        print(label,flush=True)
        with (args.output/(label+'.log')).open('w') as f:
            result=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT)
        record[label+'_exit']=result.returncode
        (args.output/'build-receipt.json').write_text(json.dumps(record,indent=2)+'\n')
        if result.returncode:raise RuntimeError((args.output/(label+'.log')).read_text()[-6000:])
    run(cmd,'configure')
    targets=[] if args.profile=='weave' else ['--target','fix_crystallize_plugin']
    run([args.cmake,'--build',str(args.output),*targets,'--parallel',str(args.jobs)],'compile')
    plugin=args.output/'fix_crystallize_plugin.so'
    record.update(complete=True,plugin_sha256=hashlib.sha256(plugin.read_bytes()).hexdigest(),
                  source_head=subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip(),
                  source_status=subprocess.check_output(['git','-C',str(args.source),'status','--short'],text=True))
    (args.output/'build-receipt.json').write_text(json.dumps(record,indent=2)+'\n')
    print('BUILD_COMPLETE',record['plugin_sha256'],flush=True)
if __name__=='__main__':main()
