#!/usr/bin/env python3
"""WEIGHT_CHEM 的 Graph、MPI、逐原子力/virial 及独立 CPU 前向验证。"""
import argparse
import json
import math
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = spec_from_file_location("css_fixture", ROOT / "benchmarks/css/run.py")
fixture = module_from_spec(_spec)
_spec.loader.exec_module(fixture)
CASES = ["signed_maps", "gaussian_fallback", "small_group", "wide_cutoff", "raw_switch", "mixed_shared", "mixed_groups"]


def configuration(case):
    maps = [{1: 1.0, 2: -0.7, 3: 0.35}, {1: -0.5, 2: 1.1, 3: 0.0}]
    if case == "gaussian_fallback":
        maps = [{2: -0.7}, {3: 0.4}]
    if case == "mixed_shared":
        maps[1] = maps[0].copy()
    groups = ["small", "small"] if case in ("small_group", "wide_cutoff") else ["Sc", "Sc"]
    if case == "mixed_groups":
        groups = ["small", "Sc"]
    return maps, 4.5 if case == "wide_cutoff" else 3.05, groups


def script(data, case):
    text = fixture.input_text(fixture.PLUGIN, "GRAPH", 4, 1, 0, True, data)
    text = text.replace("variable dx0", "group Sc type 2\ngroup small id 1 5 9\nvariable dx0", 1)
    maps, cutoff, groups = configuration(case)
    declarations = []
    for index, weights in enumerate(maps):
        mapping = " ".join(f"({atom_type},{weight})" for atom_type, weight in weights.items())
        calculation = "STRU_FACTOR" if case == "mixed_shared" and index == 1 else "WEIGHT_CHEM"
        extra = "q_factor 1.7 Chemical " if calculation == "STRU_FACTOR" else ""
        declarations.append(f"CAL NAME S{index+1} {calculation} {groups[index]} cutoff_r {cutoff} "
                            f"{extra}d_block_size {64 if index else 128} "
                            f"Chem_ctarget 1.8 Chem_sigma 1.1 Chem_map {mapping}")
    start = text.index("CAL NAME S1 STRU_FACTOR")
    end = text.index("SYMBOL s1", start)
    text = text[:start] + " &\n    ".join(declarations) + " &\n    " + text[end:]
    text = text.replace('"s1*s2"', '"linear(s1*s2,0.02,0)"')
    text = text.replace('"s1+s2"', '"linear(s1+s2,0.1,0)"')
    if case == "raw_switch":
        text = text.replace("S1.AVE", "S1.RAW").replace("S2.AVE", "S2.RAW")
        text = text.replace('"linear(s1*s2,0.02,0)"', '"AVE(fermi(s1,-4,0.4))"')
        text = text.replace('"linear(s1+s2,0.1,0)"', '"linear(AVE(s1*s2),0.02,0)"')
    if case == "wide_cutoff":
        text = text.replace("pair_style zero 4.0", "pair_style zero 2.5")
    text = text.replace("thermo 100000", "compute bias_pressure all pressure NULL virial\nthermo 100000")
    text = text.replace("f_m[2] f_m\n", "f_m[2] f_m " + " ".join(f"c_bias_pressure[{j}]" for j in range(1, 7)) + "\n")
    second_property = "stru_f" if case == "mixed_shared" else "chem_pair_r"
    text = text.replace("dump state all custom", f"compute chemical_data all metad/atom m S1.chem_pair_r S2.{second_property}\n"
                        "compute atomic_stress all stress/atom NULL virial\ndump state all custom")
    return text.replace("id x y z fx fy fz\n", "id x y z fx fy fz c_chemical_data[1] c_chemical_data[2] "
                        + " ".join(f"c_atomic_stress[{j}]" for j in range(1, 7)) + "\n")


def reference(atoms, types, centers, weights, cutoff, with_structure_factor=False):
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
            switch = 1.0
            if distance_squared > 0.8*cutoff*cutoff:
                scaled_distance = (distance_squared-0.8*cutoff*cutoff)/(0.2*cutoff*cutoff)
                switch = 1.0-scaled_distance**3
            radial_value = switch
            if with_structure_factor:
                angle = 1.7*math.sqrt(distance_squared)
                radial_value *= math.sin(angle)/angle
            value += resolved[types[center]]*resolved[types[neighbor]]*radial_value
        values[center] = value
    return values


def validate_invalid_inputs(output):
    data = output / "invalid_parameters.data"
    fixture.initial(data, True)
    valid = script(data, "signed_maps")
    cases = [
        ("zero_sigma", "Chem_sigma 1.1", "Chem_sigma 0", "Chem_sigma positive and finite"),
        ("bad_type", "(1,1.0)", "(0,1.0)", "Chem_map requires valid atom types"),
        ("bad_mapping", "(1,1.0)", "(1)", "invalid Chem_map pair"),
        ("negative_cutoff", "cutoff_r 3.05", "cutoff_r -3", "cutoff_r must be finite and positive"),
        ("embedded_switch", "Chem_ctarget 1.8", "SW_func FERMI Chem_ctarget 1.8", "express switching functions in Graph"),
        ("legacy_executor", "EXECUTOR GRAPH", "EXECUTOR LEGACY", "WEIGHT_CHEM: use EXECUTOR GRAPH"),
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
        maps, cutoff, groups = configuration(case)
        centers = [[1, 5, 9] if group == "small" else [i for i, t in types.items() if t == 2] for group in groups]
        center_union = set(centers[0]) | set(centers[1])
        text = script(data, case)
        baseline = None
        for rank_count in args.np:
            run_directory = args.output / f"{case}_np{rank_count}"
            fixture.execute(run_directory, text, rank_count)
            atoms = fixture.last_dump(run_directory / "atom_state.dump")
            observed = fixture.thermo(run_directory / "screen.txt")
            raw = [reference(atoms, types, centers[index], maps[index], cutoff,
                             with_structure_factor=(case == "mixed_shared" and index == 1)) for index in range(2)]
            means = [sum(values.values())/len(values) for values in raw]
            expected = [0.02*means[0]*means[1], 0.1*(means[0]+means[1])]
            if case == "raw_switch":
                expected = [sum(1/(1+math.exp(-0.4*(raw[0][i]+4))) for i in centers[0])/len(centers[0]),
                            0.02*sum(raw[0][i]*raw[1][i] for i in centers[0])/len(centers[0])]
            cv_error = max(abs(a-b) for a, b in zip(observed[:2], expected))
            diagnostic_error = max(abs(row[6+index]-raw[index].get(atom_id, 0))
                                   for atom_id, row in atoms.items() for index in range(2))
            assert cv_error < 2e-11 and diagnostic_error < 2e-11, (case, rank_count, cv_error, diagnostic_error)
            force = max(abs(v) for row in atoms.values() for v in row[3:6])
            outside_force = max(abs(v) for atom_id, row in atoms.items() if atom_id not in center_union for v in row[3:6])
            net_force = max(abs(sum(row[c] for row in atoms.values())) for c in range(3, 6))
            assert force > 1e-10 and outside_force > 1e-10 and net_force < 1e-10, (case, rank_count, force, outside_force, net_force)
            # stress/atom 与全局 pressure 使用相同 LAMMPS 单位；总应力取负并除体积得到压力。
            stress_pressure = [-sum(row[8+component] for row in atoms.values())/(16.2**3) for component in range(6)]
            pressure_error = max(abs(value-observed[3+component]) for component, value in enumerate(stress_pressure))
            assert max(abs(value) for value in observed[3:]) > 1e-10, (case, "zero virial")
            assert pressure_error < 1e-9, (case, rank_count, "stress sum", pressure_error)
            state_error = mpi_pressure_error = mpi_stress_error = 0.0
            if baseline:
                state_error = max(abs(value-reference_value) for atom_id, row in atoms.items()
                                  for value, reference_value in zip(row[:8], baseline[0][atom_id][:8]))
                mpi_stress_error = max(abs(value-reference_value) for atom_id, row in atoms.items()
                                       for value, reference_value in zip(row[8:], baseline[0][atom_id][8:]))
                mpi_pressure_error = max(abs(value-reference_value) for value, reference_value in zip(observed[3:], baseline[1][3:]))
                assert state_error < 1e-10 and mpi_pressure_error < 1e-9 and mpi_stress_error < 2e-8, (case, rank_count, state_error, mpi_pressure_error, mpi_stress_error)
            else:
                baseline = atoms, observed
            result = dict(case=case, ranks=rank_count, cv_cpu_error=cv_error, raw_cpu_error=diagnostic_error,
                          max_force=force, outside_group_force=outside_force, net_force=net_force,
                          mpi_state_error=state_error, mpi_pressure_error=mpi_pressure_error,
                          mpi_stress_error=mpi_stress_error, stress_pressure_error=pressure_error)
            results.append(result)
            print(json.dumps(result), flush=True)
            (args.output / "results.json").write_text(json.dumps(results, indent=2))
    validate_invalid_inputs(args.output)
    print("WEIGHT_CHEM_LAMMPS_PASS", flush=True)


if __name__ == "__main__":
    main()
