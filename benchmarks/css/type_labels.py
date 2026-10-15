#!/usr/bin/env python3
"""核对 CSS 类型编号、类型标签和混合写法在真实 LAMMPS 中的等价性。"""
import argparse
import json
import math
import os
import subprocess
from pathlib import Path

from run import LMP, MPI, PROFILES, execute, initial, last_dump, script, thermo


def use_labels(input_text, quoted=True, mixed=False):
    type_labels = {1: 'Fe', 2: '"Fe3+"' if quoted else 'Fe3+', 3: 'O'}
    if mixed:
        type_labels.pop(1)
    for atom_type, type_label in type_labels.items():
        input_text = input_text.replace(f'CHANNEL {atom_type} ', f'CHANNEL {type_label} ')
    return input_text


def compare_outputs(reference_directory, test_directory):
    reference_atoms = last_dump(reference_directory/'atom_state.dump')
    test_atoms = last_dump(test_directory/'atom_state.dump')
    assert reference_atoms.keys() == test_atoms.keys()
    differences = [abs(value-reference_atoms[atom_id][component])
                   for atom_id, values in test_atoms.items()
                   for component, value in enumerate(values)]
    reference_thermo = thermo(reference_directory/'screen.txt')
    test_thermo = thermo(test_directory/'screen.txt')
    differences.extend(abs(actual-expected) for actual, expected in zip(test_thermo, reference_thermo))
    assert all(math.isfinite(error) for error in differences)
    maximum_error = max(differences)
    assert maximum_error < 1e-10, maximum_error
    return maximum_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--np', type=int, default=2)
    arguments = parser.parse_args()
    output_directory = arguments.output.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    data_path = output_directory/'types.data'
    initial(data_path, ternary=True)
    input_text, _ = script(data_path, 'small_group', PROFILES['small_group'])
    mapped_input = input_text.replace(f'read_data {data_path}',
                                     f'read_data {data_path}\nlabelmap atom 1 Fe 2 "Fe3+" 3 O')
    labeled_data_path = output_directory/'labeled.data'
    labeled_data_path.write_text(data_path.read_text().replace(
        'Masses\n\n', 'Atom Type Labels\n\n1 Fe\n2 Fe3+\n3 O\n\nMasses\n\n'))
    cases = {
        'numeric': input_text,
        'labels_quoted': use_labels(mapped_input),
        'labels_unquoted': use_labels(mapped_input, quoted=False),
        'mixed': use_labels(mapped_input, mixed=True),
        'data_labels': use_labels(input_text.replace(str(data_path), str(labeled_data_path))),
    }
    results = []
    for case_name, case_input in cases.items():
        case_directory = output_directory/case_name
        execute(case_directory, case_input, arguments.np)
        maximum_error = compare_outputs(output_directory/'numeric', case_directory)
        result = dict(case=case_name, np=arguments.np, maximum_error=maximum_error)
        results.append(result)
        print(json.dumps(result), flush=True)

    invalid_cases = {
        'unknown_label': (use_labels(mapped_input).replace('CHANNEL Fe ', 'CHANNEL Fe2+ ', 1), 'Fe2+'),
        'missing_labelmap': (use_labels(input_text), 'Expected integer parameter'),
        'type_out_of_range': (input_text.replace('CHANNEL 1 ', 'CHANNEL 4 ', 1), 'CSS: neighbor type is outside'),
    }
    for case_name, (case_input, expected_message) in invalid_cases.items():
        case_directory = output_directory/case_name
        case_directory.mkdir(exist_ok=False)
        (case_directory/'input.lmp').write_text(case_input)
        environment = os.environ.copy()
        environment.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1')
        with (case_directory/'screen.txt').open('w') as output:
            process = subprocess.run([str(MPI), '--bind-to', 'none', '-np', str(arguments.np),
                                      str(LMP), '-in', 'input.lmp'], cwd=case_directory,
                                     env=environment, stdout=output, stderr=subprocess.STDOUT, timeout=30)
        screen = (case_directory/'screen.txt').read_text(errors='replace')
        assert process.returncode != 0 and expected_message.lower() in screen.lower(), screen[-2000:]
        result = dict(case=case_name, np=arguments.np, rejected=True)
        results.append(result)
        print(json.dumps(result), flush=True)
    (output_directory/'results.json').write_text(json.dumps(results, indent=2))
    print('CSS_TYPE_LABELS_PASS')


if __name__ == '__main__':
    main()
