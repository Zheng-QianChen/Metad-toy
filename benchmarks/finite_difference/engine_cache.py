#!/usr/bin/env python3
"""真实 LAMMPS 缓存/需求测试。复用 run.py 的 structure 夹具，以计数器和冷重建结果作比较。"""
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
    import numpy as np
    comm = MPI.COMM_WORLD
    os.chdir(args.output)
    probe = ctypes.CDLL(str(args.probe))
    probe.engine_cache_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulonglong)]
    probe.engine_cache_check.argtypes = [ctypes.c_void_p]
    probe.engine_cache_error.restype = ctypes.c_char_p
    names = ['positions', 'types', 'ids', 'selection_scans', 'flags', 'neighbor_rebuilds', 'neighbor_uploads']

    def stats(simulation):
        values = (ctypes.c_ulonglong*7)()
        if probe.engine_cache_stats(simulation.lmp, values):
            raise RuntimeError(probe.engine_cache_error().decode())
        return dict(zip(names, values))

    def require(condition, message):
        if not comm.allreduce(bool(condition), op=MPI.LAND):
            raise RuntimeError(message)

    base = (args.fixture/'in.lammps').read_text().replace('&\n', ' ')
    base = '\n'.join(f'plugin load "{args.plugin}"' if line.startswith('plugin load ') else line
                     for line in base.splitlines())+'\n'
    extra = 'CAL NAME unused STEINH Q 3 type2 cutoff_r 3.30 SW_FUNC_r radial cutoff_eps_r 1e-5 SYMBOL unused unused.AVE '
    results = []
    for case in ['unused', 'shared', 'compute', 'localq', 'dynamic', 'dynamicq', 'moving', 'localq_moving', 'distance']:
        configuration = base
        if case in ('unused', 'compute'):
            configuration = configuration.replace('CV_dim 1', extra+'CV_dim 1')
        if case == 'shared':
            configuration = configuration.replace('STRU_FACTOR all', 'STRU_FACTOR type2')
            configuration = configuration.replace('CV_dim 1',
                'CAL NAME second STRU_FACTOR type2 cutoff_r 3.05 q_factor 2.1 SYMBOL second second.AVE CV_dim 1')
            configuration = configuration.replace('"structure"', '"structure+second"')
        if case == 'compute':
            configuration = configuration.replace('compute pressure',
                'compute unused_output all metad/atom m unused.stein_q\ncompute pressure')
        if case.startswith('localq'):
            configuration = configuration.replace('STRU_FACTOR all cutoff_r 3.05 q_factor 1.7',
                'STEINH L 6 all cutoff_r 3.30 SW_FUNC_r radial cutoff_eps_r 1e-5')
        if case.startswith('dynamic'):
            configuration = configuration.replace('fix m all',
                'variable membership atom "(id%2)==(step%2)"\ngroup moving dynamic all var membership every 1\nfix m all')
            configuration = configuration.replace('STRU_FACTOR all', 'STRU_FACTOR moving')
            if case == 'dynamicq':
                configuration = configuration.replace('STRU_FACTOR moving cutoff_r 3.05 q_factor 1.7',
                    'STEINH Q 3 moving cutoff_r 3.30 SW_FUNC_r radial cutoff_eps_r 1e-5')
        if case.endswith('moving'):
            configuration = configuration.replace('\nrun 0',
                '\nvariable vx atom "0.01*sin(id)"\nvelocity all set v_vx 0.0 0.0\nfix integrate all nve\nrun 0')
        if case == 'distance':
            configuration = configuration.replace('STRU_FACTOR all cutoff_r 3.05 q_factor 1.7', 'DISTANCE 1 2')
            configuration = configuration.replace('structure.AVE', 'structure.RAW')
        simulation = lammps(cmdargs=['-screen', 'none', '-log', f'log.{case}'], comm=comm)
        try:
            simulation.commands_string(configuration)
            initial = stats(simulation)
            if case == 'unused':
                require(initial['flags'] == initial['ids'] == initial['types'] == initial['selection_scans'] == 0,
                        'unused CV prepared metadata')
                require(initial['neighbor_rebuilds'] == 1, 'unused CV prepared a neighbor view')
            if case == 'shared':
                require(initial['selection_scans'] == 1 and initial['neighbor_rebuilds'] == 1,
                        'shared selection/list was prepared more than once')
            if case.startswith('localq'):
                require(initial['flags'] == 0 and initial['ids'] == 0, 'LocalQ requested unused group flags/IDs')
            if case == 'compute':
                data = simulation.numpy.extract_compute('unused_output', 1, 1)
                require(data is not None and np.isfinite(data).all(), 'compute-only CV unavailable')
                require(initial['flags'] > 0 and initial['ids'] > 0, 'compute demand was ignored')
            if case == 'distance':
                require(initial['positions'] == initial['flags'] == initial['selection_scans'] == initial['neighbor_rebuilds'] == 0,
                        'global distance prepared unused arrays')
            # 小位移及静止步进都保留原生 topology；每次 forward 仍读取本轮坐标。
            simulation.command('run 5 pre no post no')
            warm = stats(simulation)
            require(warm['positions'] == initial['positions']+(0 if case == 'distance' else 5), 'coordinate evaluation was skipped')
            if not case.startswith('dynamic'):
                require(warm['selection_scans'] == initial['selection_scans'], 'static group rescanned')
                require(warm['flags'] == initial['flags'], 'unchanged flags re-uploaded')
                if not case.endswith('moving'):
                    require(warm['neighbor_uploads'] == initial['neighbor_uploads'], 'unchanged CSR re-uploaded')
            else:
                require(warm['selection_scans'] == initial['selection_scans']+5, 'dynamic membership not refreshed')
            value = simulation.extract_fix('m', 0, 1, nrow=0)
            forces = np.array(simulation.numpy.extract_atom('f')[:simulation.extract_global('nlocal')], copy=True)
            # 同一步冷重建必须与缓存执行一致，覆盖 run 0 和动态组过滤。
            simulation.command('run 0')
            require(abs(simulation.extract_fix('m', 0, 1, nrow=0)-value) < 1e-11, 'cached CV differs from cold rebuild')
            fresh_forces = simulation.numpy.extract_atom('f')[:simulation.extract_global('nlocal')]
            require(np.allclose(forces, fresh_forces, rtol=1e-10, atol=1e-11), 'cached forces differ from cold rebuild')
            if case.endswith('moving'):
                # 真实迁移/排序/ghost 重建后再核对一次冷计算，防止只覆盖手工失效探针。
                before_migration = stats(simulation)
                simulation.command('atom_modify sort 1 0.0')
                simulation.command('displace_atoms type2 move 4.1 0 0 units box')
                simulation.command('run 1')
                require(stats(simulation)['neighbor_rebuilds'] > before_migration['neighbor_rebuilds'],
                        'migration did not invalidate topology')
                migration_value = simulation.extract_fix('m', 0, 1, nrow=0)
                simulation.command('run 0')
                require(abs(simulation.extract_fix('m', 0, 1, nrow=0)-migration_value) < 1e-11,
                        'migration value differs from cold rebuild')
            if case == 'unused':
                require(probe.engine_cache_check(simulation.lmp) == 0, probe.engine_cache_error().decode())
            if case == 'compute':
                simulation.command('uncompute unused_output')
                simulation.command('run 0')
                without_compute = stats(simulation)
                simulation.command('run 3 pre no post no')
                require(stats(simulation)['ids'] == without_compute['ids'], 'removed compute still requested IDs')
            results.append(dict(case=case, ranks=comm.size, before=initial, after=warm, passed=True))
        finally:
            simulation.close()
    if comm.rank == 0:
        (args.output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
        print(f'PASS: {len(results)} cache/demand cases on {comm.size} ranks', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--plugin', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--np', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--mpi', type=Path)
    parser.add_argument('--mpicxx', type=Path)
    parser.add_argument('--lammps-src', type=Path)
    parser.add_argument('--nvcc', type=Path)
    parser.add_argument('--probe', type=Path)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    args.fixture, args.output = args.fixture.resolve(), args.output.resolve()
    args.plugin = plugin_path(args.plugin)
    if args.worker:
        return worker(args)
    launcher, compiler = mpi_tools(args.plugin, args.mpi, args.mpicxx)
    headers = lammps_source(args.plugin, args.lammps_src)
    cache = cmake_cache(args.plugin.parent)
    nvcc = executable(args.nvcc, [cache.get('CMAKE_CUDA_COMPILER', 'nvcc')], 'CUDA compiler')
    args.output.mkdir(parents=True, exist_ok=False)
    args.probe = args.output/'engine_cache_probe.so'
    environment = os.environ.copy()
    environment.setdefault('OMPI_CXX', cache.get('CLEAN_HOST_COMPILER', 'g++'))
    environment.setdefault('OMP_NUM_THREADS', '1')
    subprocess.run([str(nvcc), '-ccbin', str(compiler), '-std=c++17', '--shared', '--cudart', 'shared',
                    '-Xcompiler=-fPIC', str(Path(__file__).with_name('engine_cache_probe.cu')),
                    '-I'+str(headers), '-I'+str(ROOT/'include/core'), '-I'+str(ROOT/'include/utils'),
                    str(args.plugin), '-o', str(args.probe)], env=environment, check=True)
    results = []
    for ranks in args.np:
        directory = args.output/f'np{ranks}'
        directory.mkdir()
        command = [str(launcher), '-np', str(ranks), sys.executable, str(Path(__file__).resolve()), '--worker',
                   '--fixture', str(args.fixture), '--plugin', str(args.plugin), '--output', str(directory),
                   '--probe', str(args.probe)]
        with (directory/'screen.txt').open('w') as output:
            subprocess.run(command, env=environment, stdout=output, stderr=subprocess.STDOUT, timeout=180, check=True)
        results.extend(json.loads((directory/'results.json').read_text()))
        print(f'{ranks} ranks: cache/demand checks PASS', flush=True)
    (args.output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    (args.output/'metadata.json').write_text(json.dumps(dict(
        plugin_sha256=hashlib.sha256(args.plugin.read_bytes()).hexdigest(), cases=len(results)), indent=2)+'\n')


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
