#!/usr/bin/env python3
"""Time frozen Al3Sc LocalQL x CSS inputs with production, uploaded and current plugins.

The preparation directory contains assets/, provenance.json and one <variant>_validate
input per version. Each measured run starts from the identical restart/HILLS snapshot.
This runner never signals or edits the separate production job.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import time

VARIANTS=('production','uploaded','current')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def parse_thermo(text):
    columns=None;rows=[]
    for line in text.splitlines():
        words=line.split()
        if words and words[0]=='Step' and 'f_bias[1]' in words:columns=words
        elif columns and len(words)==len(columns):
            try:values=list(map(float,words))
            except ValueError:continue
            if not all(math.isfinite(value) for value in values):raise RuntimeError('Nonfinite thermo')
            rows.append(dict(zip(columns,values)))
    return rows

def process_snapshot():
    result=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            command=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            if command.startswith('/home/zqc/apps/deepmd/bin/lmp_mpi') or 'campaign.py worker' in command:
                stat=(path/'stat').read_text().rsplit(') ',1)[1].split()
                result.append(dict(pid=int(path.name),state=stat[0],start_ticks=int(stat[19]),command=command))
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return result

def summarize(records):
    result={}
    for variant in VARIANTS:
        rows=[row for row in records if row['variant']==variant]
        if not rows:continue
        values=[row['ms_per_step'] for row in rows]
        result[variant]=dict(median_ms=statistics.median(values),minimum_ms=min(values),maximum_ms=max(values),
                            samples_ms=values,median_steps_per_second=1000/statistics.median(values),
                            median_ns_per_day=86.4/statistics.median(values),
                            median_stages_seconds={stage:statistics.median([row['stages'][stage]['seconds'] for row in rows])
                               for stage in rows[0]['stages']})
    if 'current' in result:
        for variant in ('production','uploaded'):
            if variant in result:
                result['current']['time_change_vs_'+variant]=result['current']['median_ms']/result[variant]['median_ms']-1
                result['current']['speedup_vs_'+variant]=result[variant]['median_ms']/result['current']['median_ms']
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=500)
    parser.add_argument('--warmup',type=int,default=100)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--timeout',type=int,default=180)
    parser.add_argument('--contention',choices=['production-paused','concurrent'],required=True)
    args=parser.parse_args()
    if min(args.steps,args.warmup,args.repeats)<1:parser.error('counts must be positive')
    args.prepared=args.prepared.resolve();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    assets=args.prepared/'assets';provenance=json.loads((args.prepared/'provenance.json').read_text())
    for name,entry in provenance['assets'].items():
        if digest(assets/name)!=entry['sha256']:raise RuntimeError('Frozen asset changed: '+name)
    environment=dict(os.environ,LD_LIBRARY_PATH='/home/zqc/apps/deepmd/lib:/usr/local/cuda/lib64:'+os.environ.get('LD_LIBRARY_PATH',''),
                     OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMPI_MCA_btl='self,tcp',OMPI_MCA_pml='ob1')
    metadata=dict(prepared=str(args.prepared),steps=args.steps,warmup=args.warmup,repeats=args.repeats,
                  contention=args.contention,processes_before=process_snapshot(),script_sha256=digest(Path(__file__)),
                  checkpoint_step=provenance['checkpoint_result']['end_step'],
                  plugins={name:provenance['assets'][name+'.so'] for name in VARIANTS},
                  conditions=dict(atoms=6912,sc_atoms=1728,mpi_ranks=1,temperature_K=1450,pressure_bar=1,
                                  timestep_ps=.001,neighbor_skin_A=2,communication_cutoff_A=12,pace=200,wt_factor=50),
                  timing='Last LAMMPS Loop time; physical MEAM+NPT+WT deposition+thermo; startup/NVRTC/warmup excluded')
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    shutil.copy2(__file__,args.output/'runner.py')
    records=[];randomizer=random.Random(20261022)
    for repeat in range(args.repeats):
        order=list(VARIANTS);randomizer.shuffle(order)
        for variant in order:
            active=process_snapshot()
            if args.contention=='production-paused' and any(row['state'] not in ('T','t') for row in active):
                raise RuntimeError('An external LAMMPS/production worker is running; refusing uncontended timing claim')
            directory=args.output/f'{repeat}_{variant}';directory.mkdir()
            for name in ['restart.start','HILLS']:shutil.copy2(assets/name,directory/name)
            template=(args.prepared/(variant+'_validate')/'input.lmp').read_text()
            # Retain production cadence for thermo and hill deposition. Trajectory interval
            # is 10000 steps; this short window does not reach its next output frame.
            configuration=template[:template.index('run 0\n')]+f'run {args.warmup}\nrun {args.steps} pre no\n'
            (directory/'input.lmp').write_text(configuration)
            started=time.monotonic()
            with (directory/'stdout.txt').open('w') as output:
                process=subprocess.run([provenance['lammps'],'-in','input.lmp','-log','log.lammps'],cwd=directory,
                                       env=environment,stdout=output,stderr=subprocess.STDOUT,timeout=args.timeout)
            wall=time.monotonic()-started;text=(directory/'stdout.txt').read_text(errors='replace')
            if process.returncode:raise RuntimeError(f'{directory}: exit {process.returncode}\n{text[-3000:]}')
            loops=re.findall(r'Loop time of ([0-9.eE+-]+) on .*? for (\d+) steps with (\d+) atoms',text)
            if not loops:raise RuntimeError('Missing Loop time: '+str(directory))
            seconds,steps,atoms=loops[-1]
            if int(steps)!=args.steps or int(atoms)!=6912:raise RuntimeError('Unexpected measured workload')
            stages={}
            for name,minimum,average,maximum,variance,percent in re.findall(
                r'^(\w+)\s*\|\s*([\d.eE+-]+)\s*\|\s*([\d.eE+-]+)\s*\|\s*([\d.eE+-]+)\s*\|\s*([\d.eE+-]+)\s*\|\s*([\d.eE+-]+)',text,re.M):
                stages[name]=dict(seconds=float(average),percent=float(percent))
            thermo=parse_thermo(text)
            if not thermo:raise RuntimeError('Missing thermo')
            compilation=re.findall(r'NVRTC ready: compile_ms=([0-9.]+) load_ms=([0-9.]+)',text)
            if variant!='production' and len(compilation)!=1:raise RuntimeError('Unexpected NVRTC compile count')
            if digest(assets/(variant+'.so'))!=provenance['assets'][variant+'.so']['sha256']:raise RuntimeError('Plugin changed')
            builds=re.findall(r'Neighbor list builds =\s*(\d+)',text)
            hills=[line for line in (directory/'HILLS').read_text().splitlines() if line.strip() and not line.startswith('#')]
            record=dict(variant=variant,repeat=repeat,order=order,steps=int(steps),loop_seconds=float(seconds),
                        ms_per_step=1000*float(seconds)/int(steps),wall_seconds=wall,stages=stages,
                        neighbor_builds=int(builds[-1]),initial=thermo[0],final=thermo[-1],
                        nvrtc_ms=float(compilation[0][0]) if compilation else None,
                        nvrtc_load_ms=float(compilation[0][1]) if compilation else None,
                        new_hills=len(hills)-provenance['hills_count'],
                        deposited_hills=[list(map(float,line.split())) for line in hills[provenance['hills_count']:]],processes_before=active,
                        processes_after=process_snapshot(),directory=str(directory))
            records.append(record)
            (directory/'result.json').write_text(json.dumps(record,indent=2)+'\n')
            (args.output/'results.json').write_text(json.dumps(records,indent=2)+'\n')
            (args.output/'summary.json').write_text(json.dumps(summarize(records),indent=2)+'\n')
            print(f'{repeat} {variant}: {record["ms_per_step"]:.4f} ms/step; neighbor builds {record["neighbor_builds"]}; hills +{record["new_hills"]}',flush=True)
    metadata['processes_after']=process_snapshot()
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('AL3SC_PERFORMANCE_COMPARISON_COMPLETE',flush=True)

if __name__=='__main__':main()
