#!/usr/bin/env python3
"""化学结构因子的实际 LAMMPS/Graph/反向通信验证，按全局 ID 对照 CPU 与 MPI 分区。"""
import argparse
import json
import math
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = spec_from_file_location("css_fixture", ROOT / "benchmarks/css/run.py")
fixture = module_from_spec(_spec)
_spec.loader.exec_module(fixture)
CASES = ["signed_maps", "gaussian_fallback", "small_group", "wide_cutoff", "raw_nonlinear"]


def configuration(case):
    # 两个 CV 的中心组与高斯参数相同，Chem_map 不同，验证环境键区分权重表。
    maps = [{1: 1.0, 2: -0.7, 3: 0.35}, {1: -0.5, 2: 1.1, 3: 0.0}]
    if case == "gaussian_fallback":
        maps = [{2: -0.7}, {3: 0.4}]
    return maps, 4.5 if case == "wide_cutoff" else 3.3


def script(data, case):
    text = fixture.input_text(fixture.PLUGIN, "GRAPH", 4, 1, 0, True, data)
    text = text.replace("variable dx0", "group Sc type 2\ngroup small id 1 5 9\nvariable dx0", 1)
    group = "small" if case == "small_group" else "Sc"
    maps, cutoff = configuration(case)
    declarations = []
    for index, weights in enumerate(maps):
        mapping = " ".join(f"({atom_type},{weight})" for atom_type, weight in weights.items())
        declarations.append(f"CAL NAME S{index+1} STRU_FACTOR {group} cutoff_r {cutoff} "
                            f"q_factor {1.7+index*0.6} d_block_size 128 Chemical "
                            f"Chem_ctarget 1.8 Chem_sigma 1.1 Chem_map {mapping}")
    start = text.index("CAL NAME S1 STRU_FACTOR")
    end = text.index("SYMBOL s1", start)
    text = text[:start] + " &\n    ".join(declarations) + " &\n    " + text[end:]
    if case == "raw_nonlinear":
        text = text.replace("DIM 1 -5 5 500", "DIM 1 0 10 500")
        text = text.replace("S1.AVE", "S1.RAW").replace("S2.AVE", "S2.RAW")
        text = text.replace('"s1*s2"', '"AVE(s1*s1)+linear(AVE(s1*s2),0.3,0)"')
        text = text.replace('"s1+s2"', '"exp(linear(AVE(s2),0.1,0))"')
    if case == "wide_cutoff":
        # CV 半径大于 pair cutoff，邻居请求必须自行覆盖 CV 所需的 ghost 和邻居。
        text = text.replace("pair_style zero 4.0", "pair_style zero 2.5")
    text = text.replace("thermo 100000", "compute bias_pressure all pressure NULL virial\nthermo 100000")
    text = text.replace("f_m[2] f_m\n", "f_m[2] f_m " + " ".join(f"c_bias_pressure[{j}]" for j in range(1, 7)) + "\n")
    text = text.replace("dump state all custom", "compute chemical_data all metad/atom m S1.stru_f S2.stru_f\ndump state all custom")
    return text.replace("id x y z fx fy fz\n", "id x y z fx fy fz c_chemical_data[1] c_chemical_data[2]\n")


def reference(atoms, types, centers, weights, wave_number, cutoff):
    # 独立 CPU 定义：最小镜像位移、元素权重和径向平滑均直接从输入构建。
    resolved = {atom_type: weights.get(atom_type, math.exp(-0.5*((atom_type-1.8)/1.1)**2))
                for atom_type in set(types.values())}
    values = {}
    for center in centers:
        value = 1.0
        for neighbor, position in atoms.items():
            if center == neighbor:
                continue
            displacement = [position[c]-atoms[center][c] for c in range(3)]
            displacement = [v-round(v/16.2)*16.2 for v in displacement]
            distance_squared = sum(v*v for v in displacement)
            if distance_squared > cutoff*cutoff:
                continue
            angle = wave_number*math.sqrt(distance_squared)
            switch = 1.0
            if distance_squared > 0.8*cutoff*cutoff:
                scaled_distance = (distance_squared-0.8*cutoff*cutoff)/(0.2*cutoff*cutoff)
                switch = 1.0-scaled_distance**3
            value += resolved[types[center]]*resolved[types[neighbor]]*math.sin(angle)/angle*switch
        values[center] = value
    return values


def validate_invalid_inputs(output):
    data = output / "invalid_parameters.data"
    fixture.initial(data, True)
    valid = script(data, "signed_maps")
    cases = [
        ("zero_sigma", "Chem_sigma 1.1", "Chem_sigma 0", "Chem_sigma positive and finite"),
        ("bad_type", "(1,1.0)", "(0,1.0)", "Chem_map requires valid atom types"),
        ("infinite_weight", "(1,1.0)", "(1,inf)", "Chem_map requires valid atom types and finite weights"),
    ]
    for name, old, new, expected_error in cases:
        directory = output / name
        try:
            fixture.execute(directory, valid.replace(old, new, 1), 1)
        except RuntimeError:
            assert expected_error in (directory / "screen.txt").read_text(), name
        else:
            raise AssertionError(f"invalid input accepted: {name}")
        print(f"EXPECTED_ERROR_PASS {name}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--np", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for case in args.cases:
        data = args.output / f"{case}.data"
        types = fixture.initial(data, True)
        centers = [1, 5, 9] if case == "small_group" else [i for i, t in types.items() if t == 2]
        text = script(data, case)
        maps, cutoff = configuration(case)
        baseline = None
        for rank_count in args.np:
            run_directory = args.output / f"{case}_np{rank_count}"
            fixture.execute(run_directory, text, rank_count)
            atoms = fixture.last_dump(run_directory / "atom_state.dump")
            observed = fixture.thermo(run_directory / "screen.txt")
            raw = [reference(atoms, types, centers, maps[index], 1.7+index*0.6, cutoff) for index in range(2)]
            means = [sum(values.values())/len(centers) for values in raw]
            expected = [means[0]*means[1], means[0]+means[1]]
            if case == "raw_nonlinear":
                expected = [sum(raw[0][i]**2+0.3*raw[0][i]*raw[1][i] for i in centers)/len(centers),
                            math.exp(0.1*means[1])]
            cv_error = max(abs(a-b) for a, b in zip(observed[:2], expected))
            diagnostic_error = max(abs(row[6+index]-raw[index].get(atom_id, 0))
                                   for atom_id, row in atoms.items() for index in range(2))
            assert cv_error < 2e-11 and diagnostic_error < 2e-11, (case, rank_count, cv_error, diagnostic_error)
            force = max(abs(v) for row in atoms.values() for v in row[3:6])
            outside_force = max(abs(v) for atom_id, row in atoms.items() if atom_id not in centers for v in row[3:6])
            net_force = max(abs(sum(row[c] for row in atoms.values())) for c in range(3, 6))
            assert force > 1e-10 and outside_force > 1e-10 and net_force < 1e-10, (case, rank_count, force, outside_force, net_force)
            state_error = pressure_error = 0.0
            if baseline:
                state_error = max(abs(value-reference_value) for atom_id, row in atoms.items()
                                  for value, reference_value in zip(row, baseline[0][atom_id]))
                pressure_error = max(abs(value-reference_value) for value, reference_value in zip(observed[3:], baseline[1][3:]))
                assert state_error < 1e-10 and pressure_error < 1e-10, (case, rank_count, state_error, pressure_error)
            else:
                baseline = atoms, observed
            result = dict(case=case, ranks=rank_count, cv_cpu_error=cv_error, raw_cpu_error=diagnostic_error,
                          max_force=force, outside_group_force=outside_force, net_force=net_force,
                          mpi_state_error=state_error, mpi_pressure_error=pressure_error)
            results.append(result)
            print(json.dumps(result), flush=True)
            (args.output / "results.json").write_text(json.dumps(results, indent=2))
    validate_invalid_inputs(args.output)
    print("STRUFACTOR_CHEM_LAMMPS_PASS", flush=True)


if __name__ == "__main__":
    main()
