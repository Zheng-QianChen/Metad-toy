"""Read existing adapter counters around five steps; no timing is reported here."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys

from lammps import lammps

root = Path('/home/zqc/work/Metad-toy')
sys.path.insert(0, str(root/'benchmarks/runtime_performance'))
import run as benchmark

directory = root/'build/performance/neighbor_cache_probe'
plugin = root/'build/performance/current/fix_crystallize_plugin.so'
library = directory/'engine_cache_probe.so'
probe = ctypes.CDLL(str(library))
probe.engine_cache_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulonglong)]
probe.engine_cache_stats.restype = ctypes.c_int
probe.engine_cache_error.restype = ctypes.c_char_p
names = ['positions', 'types', 'ids', 'selection_scans', 'flags', 'neighbor_rebuilds', 'neighbor_uploads']


def counters(simulation):
    values = (ctypes.c_ulonglong*7)()
    if probe.engine_cache_stats(simulation.lmp, values):
        raise RuntimeError(probe.engine_cache_error().decode())
    return dict(zip(names, values))


results = []
for cutoff in [4.0, 5.0]:
    work = directory/f'pair{cutoff}'
    work.mkdir()
    os.chdir(work)
    data = root/'build/performance/uploaded_comparison_single_cvs/initial_16.data'
    configuration = benchmark.input_text(plugin, data, 'localq6', 5, 1, pair_cutoff=cutoff)
    # 首段执行至 warmup，之后用 pre no 连续运行，单独统计真实 step 的准备次数。
    configuration = configuration.removesuffix('run 5\n')
    (work/'in.lammps').write_text(configuration+'run 5 pre no post no\n')
    simulation = lammps(cmdargs=['-screen', 'none', '-log', str(work/'log.lammps')])
    try:
        simulation.commands_string(configuration.replace('&\n', ' '))
        before = counters(simulation)
        simulation.command('run 5 pre no post no')
        after = counters(simulation)
        results.append(dict(pair_cutoff=cutoff, steps=5, before=before, after=after,
                            delta={name: after[name]-before[name] for name in names}))
    finally:
        simulation.close()
(directory/'results.json').write_text(json.dumps(dict(
    plugin_sha256=hashlib.sha256(plugin.read_bytes()).hexdigest(),
    probe_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    cases=results), indent=2))
print(json.dumps(results, indent=2))
