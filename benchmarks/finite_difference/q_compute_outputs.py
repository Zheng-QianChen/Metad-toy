#!/usr/bin/env python3
"""用单个 Q/Local Q 的有限差分夹具检查 compute 的三列梯度和 ghost 归并。"""
import argparse
import ctypes
import json
import os
from pathlib import Path

import numpy as np
from mpi4py import MPI
from lammps import lammps

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--fixture', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
fixture = args.fixture.resolve()
configuration = (fixture/'in.lammps').read_text()
cv_name = next(line.split()[2] for line in configuration.splitlines()
               if line.strip().startswith('CAL NAME ') and ' STEINH ' in line)
comm = MPI.COMM_WORLD
output = args.output.resolve()
if comm.rank == 0:
    output.mkdir(parents=True, exist_ok=False)
comm.Barrier()
os.chdir(output)
simulation = lammps(cmdargs=['-screen', 'none', '-log', 'log.lammps'], comm=comm)
try:
    simulation.commands_string(configuration)
    probe = ctypes.CDLL(str(fixture.parent/'bias_probe.so'))
    probe.fd_add_hill.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
    probe.fd_add_hill.restype = ctypes.c_int
    value = simulation.extract_fix('m', 0, 1, nrow=0)
    if probe.fd_add_hill(simulation.lmp, (ctypes.c_double*1)(value+.06)):
        raise RuntimeError('could not add the fixed hill')
    simulation.commands_string(f'''
compute gradient all metad/atom m {cv_name}.dcvdx_x {cv_name}.dcvdx_y {cv_name}.dcvdx_z
compute gradient_sum all reduce sum c_gradient[1] c_gradient[2] c_gradient[3]
thermo_style custom step c_gradient_sum[1] c_gradient_sum[2] c_gradient_sum[3]
run 0
''')
    maximum_error = 0.0
    # 读取不同分量及同一步重新计算，检查缓存只在本轮反向内复用。
    for iteration in range(2):
        if iteration:
            simulation.command('run 0')
        count = simulation.extract_setting('nlocal')
        force = simulation.numpy.extract_atom('f')[:count]
        gradient = simulation.numpy.extract_compute('gradient', 1, 2)[:count]
        local_error = float(np.max(np.abs(force+gradient))) if count else 0.0
        maximum_error = max(maximum_error, comm.allreduce(local_error, op=MPI.MAX))
    if maximum_error > 1e-11:
        raise RuntimeError(f'owner gradient != -force: {maximum_error}')
    if comm.rank == 0:
        result = dict(cv=cv_name, ranks=comm.size, force_gradient_error=maximum_error, passed=True)
        (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
finally:
    simulation.close()
