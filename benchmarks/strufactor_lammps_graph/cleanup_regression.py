#!/usr/bin/env python3
"""Stru_factor 整理回归：以保留的旧插件对照 Graph 和化学加权兼容路径。"""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from run import ROOT, input_text, run_case
from compare import compare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline-plugin', type=Path, required=True)
    ap.add_argument('--initial', type=Path, required=True, help='6x6x6 FCC 初态')
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    a.output = a.output.resolve()
    plugins = {'before': a.baseline_plugin.resolve(), 'after': ROOT/'build/graph-lammps/fix_crystallize_plugin.so'}
    comparisons = []
    for version, plugin in plugins.items():
        out = a.output/version
        out.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(a.initial, out/'initial.data')
        for np in [1, 2, 4]:
            args = SimpleNamespace(output=out, plugin=plugin, cells=6, steps=1, warmup=0,
                validate=True, motion='translate', mpi=Path('/home/zqc/apps/deepmd/bin/mpirun'),
                lmp=Path('/home/zqc/apps/deepmd/bin/lmp_mpi'))
            run_case(args, 'GRAPH', np, 0)
            # 化学加权仍由其子类维护标量兼容接口，检查共享环境整理后的数值。
            d = out/f'np{np}_chemical'
            d.mkdir()
            text = input_text(plugin, 'LEGACY', 6, 1, 0, True, out/'initial.data')
            text = text.replace('d_block_size 128 &', 'd_block_size 128 Chemical Chem_ctarget 1 Chem_sigma 1 &')
            (d/'in.lammps').write_text(text)
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1')
            with (d/'screen.txt').open('w') as f:
                result = subprocess.run([str(args.mpi), '--bind-to', 'none', '-np', str(np),
                    str(args.lmp), '-in', 'in.lammps'], cwd=d, env=env,
                    stdout=f, stderr=subprocess.STDOUT, timeout=60)
            if result.returncode:
                raise RuntimeError((d/'screen.txt').read_text(errors='replace')[-2500:])
    for np in [1, 2, 4]:
        for mode in ['graph_0', 'chemical']:
            comparisons.append(compare(a.output/'before'/f'np{np}_{mode}',
                                       a.output/'after'/f'np{np}_{mode}'))
        if np != 1:
            comparisons.append(compare(a.output/'after'/'np1_graph_0',
                                       a.output/'after'/f'np{np}_graph_0'))
    (a.output/'results.json').write_text(json.dumps(comparisons, indent=2))
    print(json.dumps(comparisons, indent=2))
    print('STRUFACTOR_CLEANUP_PASS')


if __name__ == '__main__':
    main()
