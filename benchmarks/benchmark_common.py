"""Shared path discovery for user-facing benchmark runners (Python standard library)."""
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

def cmake_cache(build_directory):
    result = {}
    path = Path(build_directory)/'CMakeCache.txt'
    if path.is_file():
        for line in path.read_text().splitlines():
            if line and not line.startswith(('#', '//')) and ':' in line and '=' in line:
                key, value = line.split('=', 1)
                result[key.split(':', 1)[0]] = value
    return result

def executable(requested, candidates, label):
    # 显式指定的路径失败时立即报错，避免误用另一套 MPI/LAMMPS。
    for value in ([str(requested)] if requested else candidates):
        resolved = shutil.which(str(value))
        if resolved: return Path(resolved).absolute()
    raise ValueError(f'{label} not found: {requested or ", ".join(map(str, candidates))}')

def plugin_path(requested=None, build_directory=None):
    specified = requested or os.environ.get('METAD_PLUGIN')
    candidates = [Path(specified)] if specified else [Path(build_directory or ROOT/'build')/'fix_crystallize_plugin.so', ROOT/'build/graph-lammps/fix_crystallize_plugin.so']
    for path in candidates:
        if path.is_file(): return path.resolve()
    raise ValueError('Plugin not found; pass --plugin /path/fix_crystallize_plugin.so or METAD_PLUGIN')

def mpi_tools(plugin, launcher=None, compiler=None):
    cache = cmake_cache(Path(plugin).parent)
    mpi_home = cache.get('MPI_HOME') or os.environ.get('MPI_HOME')
    prefixes = [Path(mpi_home)/'bin'] if mpi_home else []
    launch_candidates = [directory/name for directory in prefixes for name in ('mpiexec','mpirun')]+['mpiexec','mpirun']
    launch = executable(launcher or os.environ.get('MPIEXEC'), launch_candidates, 'MPI launcher')
    compiler_candidates = [launch.parent/'mpicxx', launch.parent/'mpic++', 'mpicxx', 'mpic++']
    compile_command = executable(compiler or os.environ.get('MPICXX'), compiler_candidates, 'MPI C++ compiler') if compiler is not False else None
    return launch, compile_command

def lammps_executable(plugin, requested=None):
    cache = cmake_cache(Path(plugin).parent)
    library = cache.get('LAMMPS_LIB')
    prefix = Path(library).parent.parent/'bin' if library else None
    candidates = ([prefix/name for name in ('lmp','lmp_mpi')] if prefix else [])+['lmp','lmp_mpi']
    return executable(requested or os.environ.get('LAMMPS_EXECUTABLE'), candidates, 'LAMMPS executable')

def lammps_source(plugin, requested=None):
    source = requested or os.environ.get('LAMMPS_SOURCE_DIR') or cmake_cache(Path(plugin).parent).get('LAMMPS_SOURCE_DIR')
    if source and (Path(source)/'lammps.h').is_file(): return Path(source).resolve()
    raise ValueError('LAMMPS headers not found; pass --lammps-src /path/lammps/src')
