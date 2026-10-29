#!/usr/bin/env python3
"""Archive performance samples, check final observables and render reproducible plots."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--leaf-csv', type=Path, required=True)
    parser.add_argument('--baselines', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--overwrite', action='store_true', help='refresh the known report files in an existing result directory')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.overwrite)
    (args.output/'analysis_metadata.json').write_text(json.dumps(dict(
        summarizer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        leaf_csv_sha256=hashlib.sha256(args.leaf_csv.read_bytes()).hexdigest()),indent=2))
    metadata = json.loads((args.input/'metadata.json').read_text())
    rows = json.loads((args.input/'results.json').read_text())
    summaries = json.loads((args.input/'summary.json').read_text())
    baseline = next(iter(metadata['plugins']))
    final_checks = []
    for summary in summaries:
        subset = [row for row in rows if (row['case'],row['cells'],row['ranks']) == (summary['case'],summary['cells'],summary['ranks'])]
        reference = next(row for row in subset if row['variant']==baseline)
        for row in subset:
            value_error = max(abs(a-b) for a,b in zip(row['observables'][:3],reference['observables'][:3]))
            pressure_error = max(abs(a-b) for a,b in zip(row['observables'][3:],reference['observables'][3:]))
            passed = value_error < 1e-10 and pressure_error < 1e-8
            final_checks.append(dict(case=row['case'],atoms=row['atoms'],ranks=row['ranks'],variant=row['variant'],repeat=row['repeat'],
                                     cv_energy_error=value_error,pressure_error=pressure_error,passed=passed))
        if any(len(values)!=metadata['repeats'] for values in summary['samples_us'].values()):
            raise RuntimeError('incomplete timing matrix')
    if not all(row['passed'] for row in final_checks):
        raise RuntimeError(f'final observables differ: {[row for row in final_checks if not row["passed"]]}')
    for name in ['metadata.json','results.json','summary.json','validation.json','calibration.json']:
        shutil.copy2(args.input/name,args.output/name)
    shutil.copy2(args.baselines,args.output/'baseline_manifest.json')
    shutil.copy2(args.leaf_csv,args.output/'leaf_layout.csv')
    (args.output/'final_validation.json').write_text(json.dumps(final_checks,indent=2))
    with args.leaf_csv.open() as source:
        leaf_rows = list(csv.DictReader(source))
    leaf_summary = []
    for count in sorted({int(row['items']) for row in leaf_rows}):
        layouts = {}
        for layout in ['contiguous','input_stride3','both_stride3','both_stride8']:
            values = [row for row in leaf_rows if int(row['items'])==count and row['layout']==layout]
            if len(values)!=3: raise RuntimeError('incomplete layout matrix')
            graph = [float(row['graph_forward_backward_us']) for row in values]
            staging = [float(row['staging_device_interval_us']) for row in values]
            layouts[layout] = dict(median_us=statistics.median(graph),min_us=min(graph),max_us=max(graph),
                                   staging_interval_us=statistics.median(staging),samples_us=graph)
        for layout,result in layouts.items():
            result['overhead_us']=result['median_us']-layouts['contiguous']['median_us']
            result['overhead_percent']=100*result['overhead_us']/layouts['contiguous']['median_us']
        leaf_summary.append(dict(items=count,layouts=layouts))
    (args.output/'leaf_summary.json').write_text(json.dumps(leaf_summary,indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    variants = list(metadata['plugins'])
    labels = {'before_cache':'Before cache','before_streams':'Cached, before streams','current':'Current'}
    colors = ['#a9afb7','#61768c','#176a88']
    cases = sorted({row['case'] for row in summaries})
    figure,axes = plt.subplots(1,len(cases),figsize=(12,5.2),squeeze=False)
    width = .22
    for axis,case in zip(axes[0],cases):
        subset = [row for row in summaries if row['case']==case]
        positions = np.arange(len(subset))
        for offset,variant in enumerate(variants):
            medians = np.array([row['median_us'][variant]/1000 for row in subset])
            low = medians-np.array([row['ranges_us'][variant][0]/1000 for row in subset])
            high = np.array([row['ranges_us'][variant][1]/1000 for row in subset])-medians
            axis.barh(positions+(offset-(len(variants)-1)/2)*width,medians,width,color=colors[offset%len(colors)],
                      label=labels.get(variant,variant),xerr=np.vstack([low,high]),capsize=2)
        axis.set_yticks(positions,[f'N={row["atoms"]:,} | ranks={row["ranks"]}' for row in subset])
        axis.invert_yaxis();axis.set_xlabel('Median timestep time (ms; min/max whiskers)')
        axis.set_ylabel('Atom count / MPI ranks')
        axis.set_title('Shared Stru_factor' if case=='shared' else 'Stru_factor + CSS + Q6 + Local Q6')
    figure.suptitle('LAMMPS full timestep: 3 repeated runs per version and configuration')
    figure.legend(*axes[0][-1].get_legend_handles_labels(),frameon=False,loc='upper center',bbox_to_anchor=(.5,.94),ncol=3,fontsize=9)
    figure.text(.12,.015,'RTX 4080 SUPER / WSL / shared GPU / Release sm_89 / pair zero / warmed fixed translation',fontsize=9)
    figure.tight_layout(rect=(0,.045,1,.87));figure.savefig(args.output/'timestep.png',dpi=170)
    figure.savefig(args.output/'timestep.svg');plt.close(figure)

    figure,axes = plt.subplots(1,2,figsize=(11,4.6))
    positions = np.arange(len(leaf_summary));width=.19
    layout_labels = [('contiguous','Contiguous'),('input_stride3','Input stride 3'),('both_stride3','Both stride 3'),('both_stride8','Both stride 8')]
    for offset,(layout,label) in enumerate(layout_labels):
        x = positions+(offset-1.5)*width
        times = [row['layouts'][layout]['median_us'] for row in leaf_summary]
        errors = np.array([[row['layouts'][layout]['median_us']-row['layouts'][layout]['min_us'] for row in leaf_summary],
                           [row['layouts'][layout]['max_us']-row['layouts'][layout]['median_us'] for row in leaf_summary]])
        axes[0].bar(x,times,width,label=label,color=['#a9afb7','#91adbe','#176a88','#173f58'][offset],yerr=errors,capsize=2)
        axes[1].bar(x,[row['layouts'][layout]['staging_interval_us'] for row in leaf_summary],width,
                    label=label,color=['#a9afb7','#91adbe','#176a88','#173f58'][offset])
    for axis in axes:
        axis.set_xticks(positions,[f'{row["items"]:,}' for row in leaf_summary]);axis.set_xlabel('Leaf item count')
    axes[0].set_ylabel('Median forward + backward time (microseconds)');axes[0].set_title('Complete Graph call (min/max whiskers)')
    axes[1].set_ylabel('Median event interval per iteration (microseconds)');axes[1].set_title('Leaf staging batch (includes submission gaps)')
    axes[0].legend(frameon=False,fontsize=8)
    figure.text(.12,.015,'Source: leaf_layout.csv / same AVE(x*x + 0.3*x) graph / 3 repetitions / NVRTC excluded',fontsize=9)
    figure.tight_layout(rect=(0,.045,1,1));figure.savefig(args.output/'leaf_layout.png',dpi=170)
    figure.savefig(args.output/'leaf_layout.svg');plt.close(figure)
    print(json.dumps(dict(full_timestep=summaries,leaf=leaf_summary,
                         initial_validations=len(json.loads((args.input/'validation.json').read_text())),
                         final_validations=len(final_checks)),indent=2))


if __name__ == '__main__':
    main()
