#!/usr/bin/env python3
"""在 run.py 的 structure 夹具上检查真实适配器的六/九分量、Host/CUDA、AoS/SoA 和 ghost reverse。"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'benchmarks'))
from benchmark_common import cmake_cache, executable, lammps_source, mpi_tools, plugin_path


def worker(args):
    from mpi4py import MPI
    from lammps import lammps
    comm = MPI.COMM_WORLD
    os.chdir(args.output)
    simulation = lammps(cmdargs=['-screen','none','-log','log.lammps'], comm=comm)
    try:
        configuration = (args.fixture/'in.lammps').read_text()
        configuration = '\n'.join(f'plugin load "{args.plugin}"' if line.startswith('plugin load ') else line
                                  for line in configuration.splitlines())+'\n'
        simulation.commands_string(configuration)
        probe = ctypes.CDLL(str(args.probe))
        probe.engine_virial_check.argtypes = [ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,
                                             ctypes.POINTER(ctypes.c_double)]
        probe.engine_virial_check.restype = ctypes.c_int
        probe.engine_virial_error.restype = ctypes.c_char_p
        results = []
        for full9 in (0,1):
            for soa in (0,1):
                for cuda_memory in (0,1):
                    maximum_error = ctypes.c_double()
                    if probe.engine_virial_check(simulation.lmp,full9,soa,cuda_memory,ctypes.byref(maximum_error)):
                        raise RuntimeError(probe.engine_virial_error().decode())
                    results.append(dict(ranks=comm.size,components=9 if full9 else 6,layout='SoA' if soa else 'AoS',
                                        memory='CUDA' if cuda_memory else 'Host',max_error=maximum_error.value,passed=True))
        if comm.rank == 0:
            (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
            print(f'PASS: {len(results)} layout/representation cases on {comm.size} ranks',flush=True)
    finally:
        simulation.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture',type=Path,required=True,help='run.py 生成的 structure_npN 目录')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plugin',type=Path,required=True)
    parser.add_argument('--np',type=int,nargs='+',default=[1,2,4])
    parser.add_argument('--mpi',type=Path)
    parser.add_argument('--mpicxx',type=Path)
    parser.add_argument('--lammps-src',type=Path)
    parser.add_argument('--nvcc',type=Path)
    parser.add_argument('--probe',type=Path)
    parser.add_argument('--worker',action='store_true')
    args = parser.parse_args()
    args.output, args.fixture = args.output.resolve(), args.fixture.resolve()
    args.plugin = plugin_path(args.plugin)
    if args.worker:
        return worker(args)
    if min(args.np) < 1:
        parser.error('--np requires positive ranks')
    if not (args.fixture/'in.lammps').is_file():
        parser.error('--fixture must contain in.lammps')
    launcher, compiler = mpi_tools(args.plugin,args.mpi,args.mpicxx)
    headers = lammps_source(args.plugin,args.lammps_src)
    cache = cmake_cache(args.plugin.parent)
    nvcc = executable(args.nvcc,[cache.get('CMAKE_CUDA_COMPILER','nvcc')],'CUDA compiler')
    args.output.mkdir(parents=True,exist_ok=False)
    args.probe = args.output/'engine_virial_probe.so'
    environment = os.environ.copy()
    environment.setdefault('OMPI_CXX',cache.get('CLEAN_HOST_COMPILER','g++'))
    environment.setdefault('OMP_NUM_THREADS','1')
    subprocess.run([str(nvcc),'-ccbin',str(compiler),'-std=c++17','--shared','--cudart','shared','-Xcompiler=-fPIC',
                    str(Path(__file__).with_name('engine_virial_probe.cu')),'-I'+str(headers),
                    '-I'+str(ROOT/'include/core'),'-I'+str(ROOT/'include/utils'),str(args.plugin),
                    '-o',str(args.probe)],env=environment,check=True)
    results = []
    for ranks in args.np:
        directory = args.output/f'np{ranks}'
        directory.mkdir()
        command = [str(launcher),'-np',str(ranks),sys.executable,str(Path(__file__).resolve()),'--worker',
                   '--fixture',str(args.fixture),'--output',str(directory),'--plugin',str(args.plugin),
                   '--probe',str(args.probe)]
        with (directory/'screen.txt').open('w') as output:
            subprocess.run(command,env=environment,stdout=output,stderr=subprocess.STDOUT,timeout=120,check=True)
        rows = json.loads((directory/'results.json').read_text())
        results.extend(rows)
        print(f'{ranks} ranks: {len(rows)} cases PASS',flush=True)
    (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    (args.output/'metadata.json').write_text(json.dumps(dict(plugin_sha256=hashlib.sha256(args.plugin.read_bytes()).hexdigest(),
        cases=len(results),passed=len(results),scope='LAMMPS projection/reverse only; GPU full9 preservation covered by engine_geometry'),indent=2)+'\n')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        if '--worker' in sys.argv:
            from mpi4py import MPI
            MPI.COMM_WORLD.Abort(1)
        raise
