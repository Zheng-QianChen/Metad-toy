#!/usr/bin/env python3
"""复用生产 CV 夹具，检查 compute 的单列/多列、group 掩码与逐原子 stress。"""
import argparse
import hashlib
import json
import math
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'benchmarks'))
from benchmark_common import plugin_path, mpi_tools, lammps_executable


def load_fixture(name):
    spec = spec_from_file_location('output_fixture_'+name, ROOT/'benchmarks'/name/'run.py')
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dump_frames(path):
    lines = iter(path.read_text().splitlines())
    frames = []
    for line in lines:
        if line != 'ITEM: TIMESTEP':
            continue
        step = int(next(lines))
        assert next(lines) == 'ITEM: NUMBER OF ATOMS'
        count = int(next(lines))
        assert next(lines).startswith('ITEM: BOX BOUNDS')
        bounds = [list(map(float, next(lines).split())) for _ in range(3)]
        assert all(len(bound) == 2 for bound in bounds), 'fixture requires an orthogonal box'
        volume = math.prod(bound[1]-bound[0] for bound in bounds)
        header = next(lines).split()[2:]
        rows = {}
        for _ in range(count):
            values = list(map(float, next(lines).split()))
            assert all(math.isfinite(value) for value in values), 'non-finite compute output'
            row = dict(zip(header, values))
            rows[int(row['id'])] = row
        frames.append(dict(step=step, volume=volume, atoms=rows))
    assert frames, 'dump has no frames'
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--np', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--plugin', type=Path)
    parser.add_argument('--lmp', type=Path)
    parser.add_argument('--mpi', type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    plugin = plugin_path(args.plugin)
    launcher, _ = mpi_tools(plugin, args.mpi, compiler=False)
    executable = lammps_executable(plugin, args.lmp)
    results = []
    for case, module_name in [('mixed_shared','weighted_chem'), ('signed_maps','strufactor_chem'),
                              ('multi_radius','css')]:
        module = load_fixture(module_name)
        fixture = module if module_name == 'css' else module.fixture
        fixture.PLUGIN, fixture.LMP, fixture.MPI = plugin, executable, launcher
        data = args.output/(module_name+'.data')
        fixture.initial(data, True)
        if module_name == 'css':
            text, _ = module.script(data,case,module.PROFILES[case])
        else:
            text = module.script(data,case)
        # 已有 multi compute 的第一列作为同一属性的单列和 group 过滤对照。
        multi_line = next(line for line in text.splitlines() if ' metad/atom ' in line)
        tokens = multi_line.split()
        multi_name, property_name = tokens[1], tokens[5]
        declarations = (f'compute output_single all metad/atom m {property_name}\n'
                        f'compute output_group small metad/atom m {property_name}\n'
                        'compute output_stress all stress/atom NULL virial\n')
        text = text.replace('dump state all custom', declarations+'dump state all custom', 1)
        text = '\n'.join(line+' c_output_single c_output_group '+
            ' '.join(f'c_output_stress[{component}]' for component in range(1,7))
            if line.startswith('dump state ') else line for line in text.splitlines())+'\n'
        baseline = None
        for ranks in args.np:
            directory = args.output/f'{module_name}_np{ranks}'
            fixture.execute(directory,text,ranks)
            frames = dump_frames(directory/'atom_state.dump')
            assert len(frames) >= 2, 'expected output at setup and next timestep'
            single_error = group_error = mpi_error = 0.0
            for frame in frames:
                for atom_id, row in frame['atoms'].items():
                    raw_value = row[f'c_{multi_name}[1]']
                    single_error = max(single_error, abs(row['c_output_single']-raw_value))
                    expected = raw_value if atom_id in (1,5,9) else 0.0
                    group_error = max(group_error, abs(row['c_output_group']-expected))
            assert single_error < 1e-13 and group_error < 1e-13, (module_name,ranks,single_error,group_error)
            last = frames[-1]
            pressure = fixture.thermo(directory/'screen.txt')[3:9]
            assert len(pressure) == 6 and max(abs(value) for value in pressure) > 1e-10, 'missing virial signal'
            reconstructed = [-sum(row[f'c_output_stress[{component}]'] for row in last['atoms'].values())/
                             last['volume'] for component in range(1,7)]
            pressure_error = max(abs(first-second) for first,second in zip(pressure,reconstructed))
            assert pressure_error < 1e-9, (module_name,ranks,pressure_error)
            if baseline:
                assert [frame['step'] for frame in frames] == [frame['step'] for frame in baseline]
                for frame, reference in zip(frames,baseline):
                    assert frame['atoms'].keys() == reference['atoms'].keys()
                    for atom_id,row in frame['atoms'].items():
                        mpi_error = max(mpi_error,max(abs(value-reference['atoms'][atom_id][key]) for key,value in row.items()))
                assert mpi_error < 2e-8, (module_name,ranks,mpi_error)
            else:
                baseline = frames
            result = dict(case=module_name,ranks=ranks,frames=len(frames),single_column_error=single_error,
                          group_mask_error=group_error,stress_pressure_error=pressure_error,mpi_dump_error=mpi_error)
            results.append(result)
            print(json.dumps(result),flush=True)
            (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    (args.output/'metadata.json').write_text(json.dumps(dict(plugin=str(plugin),
        plugin_sha256=hashlib.sha256(plugin.read_bytes()).hexdigest(),cases=len(results),passed=len(results)),indent=2)+'\n')
    print('COMPUTE_OUTPUTS_PASS',flush=True)


if __name__ == '__main__':
    main()
