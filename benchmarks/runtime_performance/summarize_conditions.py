#!/usr/bin/env python3
"""Archive comparable run.py conditions and plot their complete timestep timings."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', action='append', required=True, help='LABEL=run.py output directory')
    parser.add_argument('--provenance', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    conditions = {}
    for specification in args.input:
        label, directory = specification.split('=', 1)
        if not re.fullmatch(r'[A-Za-z0-9_-]+', label) or label in conditions:
            parser.error('condition labels must be unique simple identifiers')
        conditions[label] = Path(directory).resolve(strict=True)

    # 所有条件复用同一组插件；每个条件单独核对终态，以免轨迹长度差异混入版本比较。
    plugin_hashes = None
    configurations = None
    archives = {}
    for label, directory in conditions.items():
        records = {name: json.loads((directory/f'{name}.json').read_text())
                   for name in ['metadata', 'results', 'summary', 'validation']}
        metadata, samples, summaries = records['metadata'], records['results'], records['summary']
        hashes = {name: plugin['sha256'] for name, plugin in metadata['plugins'].items()}
        keys = {(row['case'], row['cells'], row['ranks']) for row in summaries}
        if plugin_hashes is None:
            plugin_hashes, configurations = hashes, keys
        if hashes != plugin_hashes or keys != configurations or not keys:
            raise RuntimeError('conditions must cover the same plugins and configurations')
        if not records['validation'] or not all(row['passed'] for row in records['validation']):
            raise RuntimeError(f'{label}: initial validation failed')
        expected_count = len(keys)*len(hashes)*metadata['repeats']
        if len(samples) != expected_count:
            raise RuntimeError(f'{label}: incomplete measurement matrix')
        final_checks = []
        baseline = next(iter(hashes))
        for summary in summaries:
            key = summary['case'], summary['cells'], summary['ranks']
            subset = [row for row in samples if (row['case'], row['cells'], row['ranks']) == key]
            for variant in hashes:
                repeats = [row['repeat'] for row in subset if row['variant'] == variant]
                if sorted(repeats) != list(range(metadata['repeats'])):
                    raise RuntimeError(f'{label}: missing or duplicated repetitions')
            reference = next(row for row in subset if row['variant'] == baseline)
            for sample in subset:
                value_error = max(abs(a-b) for a, b in zip(sample['observables'][:3], reference['observables'][:3]))
                pressure_error = max(abs(a-b) for a, b in zip(sample['observables'][3:], reference['observables'][3:]))
                passed = value_error < 1e-10 and pressure_error < 1e-8
                final_checks.append(dict(case=sample['case'], atoms=sample['atoms'], ranks=sample['ranks'],
                                         variant=sample['variant'], repeat=sample['repeat'],
                                         cv_energy_error=value_error, pressure_error=pressure_error, passed=passed))
                if not passed:
                    raise RuntimeError(f'{label}: final observables differ: {final_checks[-1]}')
                if metadata['neighbor_rebuild'] == 'every-step' and sample['neighbor_builds'] != sample['steps']:
                    raise RuntimeError(f'{label}: forced neighbor rebuild was not observed on every step')
        records['final_validation'] = final_checks
        archives[label] = records

    args.output.mkdir(parents=True, exist_ok=False)
    for label, records in archives.items():
        destination = args.output/label
        destination.mkdir()
        for name, data in records.items():
            (destination/f'{name}.json').write_text(json.dumps(data, indent=2))
        calibration = conditions[label]/'calibration.json'
        if calibration.exists():
            shutil.copy2(calibration, destination/calibration.name)
        benchmark_source = conditions[label]/'benchmark_source.py'
        if benchmark_source.exists():
            if hashlib.sha256(benchmark_source.read_bytes()).hexdigest() != records['metadata']['script_sha256']:
                raise RuntimeError(f'{label}: archived benchmark source hash differs')
            shutil.copy2(benchmark_source, destination/benchmark_source.name)
    shutil.copy2(args.provenance, args.output/'baseline_provenance.json')
    (args.output/'analysis_metadata.json').write_text(json.dumps(dict(
        summarizer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        baseline_provenance_sha256=hashlib.sha256(args.provenance.read_bytes()).hexdigest()), indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    cases = sorted({key[0] for key in configurations})
    variants = list(plugin_hashes)
    provenance = json.loads(args.provenance.read_text())
    uploaded_label = 'Uploaded ' + provenance.get('commit', '')[:7]
    figure, axes = plt.subplots(len(archives), len(cases), figsize=(12, 4.2*len(archives)), squeeze=False)
    colors = ['#a9afb7', '#176a88', '#61768c']
    width = .7/len(variants)
    for condition_index, (label, records) in enumerate(archives.items()):
        for case_index, case in enumerate(cases):
            axis = axes[condition_index][case_index]
            summaries = [row for row in records['summary'] if row['case'] == case]
            positions = np.arange(len(summaries))
            for index, variant in enumerate(variants):
                medians = np.array([row['median_us'][variant]/1000 for row in summaries])
                low = medians-np.array([row['ranges_us'][variant][0]/1000 for row in summaries])
                high = np.array([row['ranges_us'][variant][1]/1000 for row in summaries])-medians
                axis.barh(positions+(index-(len(variants)-1)/2)*width, medians, width,
                          color=colors[index % len(colors)], xerr=np.vstack([low, high]), capsize=2,
                          label={'uploaded': uploaded_label.strip(), 'current': 'Current'}.get(variant, variant))
            axis.set_yticks(positions, [f'N={row["atoms"]:,} | ranks={row["ranks"]}' for row in summaries])
            axis.invert_yaxis()
            axis.set_xlabel('Timestep time (ms; median and min/max)')
            axis.set_title(f'{case} / {label}')
    figure.suptitle('Uploaded commit vs current: full LAMMPS timestep', fontsize=14)
    figure.legend(*axes[0][0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.5, .96), ncol=len(variants), frameon=False)
    figure.text(.08, .012, 'RTX 4080 SUPER / WSL2 / shared GPU / pair zero / 3 repeats / NVRTC excluded', fontsize=9)
    figure.tight_layout(rect=(0, .035, 1, .915))
    figure.savefig(args.output/'timestep.png', dpi=170)
    figure.savefig(args.output/'timestep.svg')
    plt.close(figure)
    print(json.dumps({label: dict(samples=len(records['results']), initial_checks=len(records['validation']),
                                 final_checks=len(records['final_validation'])) for label, records in archives.items()}, indent=2))


if __name__ == '__main__':
    main()
