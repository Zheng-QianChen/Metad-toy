#!/usr/bin/env python3
"""在已有 structure 有限差分夹具上验证删除/重建 Fix 及非连续 group 编号。"""
import argparse
import os
from pathlib import Path
from mpi4py import MPI
from lammps import lammps

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--fixture', type=Path, required=True, help='run.py 产生的 structure_npN 目录')
parser.add_argument('--output', type=Path, required=True, help='新的输出目录')
args = parser.parse_args()
configuration = (args.fixture.resolve() / 'in.lammps').read_text()
# 保留 type2 的原始编号，同时删除前面的 tiny，使 group 编号出现空槽。
configuration = configuration.replace('group type2 type 2', 'group type2 type 2\ngroup tiny delete')
configuration = configuration.replace('STRU_FACTOR all ', 'STRU_FACTOR type2 ')
commands = configuration.replace('&\n', ' ').splitlines()
fix_command = next(line for line in commands if line.startswith('fix m '))
comm = MPI.COMM_WORLD
output = args.output.resolve()
if comm.rank == 0:
    output.mkdir(parents=True, exist_ok=False)
comm.Barrier()
os.chdir(output)
simulation = lammps(cmdargs=['-screen', 'none', '-log', 'log.lammps'], comm=comm)
try:
    simulation.commands_string(configuration)
    reference = simulation.extract_fix('m', 0, 1, nrow=0)
    # 重复使用同名 Fix，覆盖环境池、邻居句柄和适配器缓冲的销毁/重建。
    for iteration in range(3):
        simulation.command('unfix m')
        simulation.command(fix_command)
        simulation.command('run 0')
        current = simulation.extract_fix('m', 0, 1, nrow=0)
        if abs(current-reference) > 1e-12:
            raise RuntimeError(f'CV changed after Fix reconstruction: {reference} -> {current}')
    if comm.rank == 0:
        print(f'PASS: {comm.size} ranks, sparse group IDs, three Fix reconstructions, value={reference:.17g}')
finally:
    simulation.close()
