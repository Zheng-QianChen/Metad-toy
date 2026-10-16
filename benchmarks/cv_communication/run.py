#!/usr/bin/env python3
"""对照通信改造前后插件，检查混合宽度、MPI 受力和实际发包量。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'benchmarks/strufactor_lammps_graph'))
from run import input_text
from compare import compare

CASES = {
    'strufactor': ('GRAPH', 'STRU_FACTOR all cutoff_r 3.30 q_factor 1.7 d_block_size 128',
                  'STRU_FACTOR all cutoff_r 3.30 q_factor 2.3 d_block_size 128', [], [9]),
    'q_graph': ('GRAPH', 'STEINH Q 4 all cutoff_r 3.30 d_block_size 128',
                'STEINH Q 6 all cutoff_r 3.30 d_block_size 128', [13, 17], [9]),
    'q_graph_odd': ('GRAPH', 'STEINH Q 3 all cutoff_r 3.30 d_block_size 128',
                    'STEINH Q 6 all cutoff_r 3.30 d_block_size 128', [11, 17], [9]),
    'local_graph': ('GRAPH', 'STEINH L 4 all cutoff_r 3.30 d_block_size 64',
                    'STEINH L 6 all cutoff_r 3.30 d_block_size 128', [], [9]),
    'q_local_graph': ('GRAPH', 'STEINH Q 4 all cutoff_r 3.30 d_block_size 128',
                      'STEINH L 6 all cutoff_r 3.30 d_block_size 128', [13], [9]),
    'chem_mix': ('GRAPH', 'STRU_FACTOR all cutoff_r 3.05 q_factor 1.7 d_block_size 128 Chemical Chem_ctarget 1 Chem_sigma 1',
                'WEIGHT_CHEM all cutoff_r 3.05 d_block_size 128 Chem_ctarget 1 Chem_sigma 1', [], [9]),
}


def run_case(args, version, plugin, case, np, probe):
    executor, cv1, cv2, _, _ = CASES[case]
    directory=args.output/version/f'{case}_np{np}'
    directory.mkdir(parents=True, exist_ok=False)
    text=input_text(plugin,executor,6,1,0,True,args.initial)
    text=text.replace('STRU_FACTOR all cutoff_r 3.30 q_factor 1.7 d_block_size 128',cv1)
    text=text.replace('STRU_FACTOR all cutoff_r 3.30 q_factor 2.3 d_block_size 128',cv2)
    # 化学加权配位数缩放到网格内；Graph 的常数缩放使用 linear 节点。
    if case=='chem_mix':
        text=text.replace('\"s1*s2\"','\"s1*linear(s2,0.05,0)\"').replace('\"s1+s2\"','\"s1+linear(s2,0.05,0)\"')
    # 压力六分量覆盖 reverse 包中的 virial 字段。
    text=text.replace('thermo_style custom step atoms f_m[1] f_m[2] f_m',
        'compute pfix all pressure NULL virial\nthermo_style custom step atoms f_m[1] f_m[2] f_m '+
        ' '.join(f'c_pfix[{k}]' for k in range(1,7)))
    if args.comm_style=='tiled': text=text.replace('read_data ', 'comm_style tiled\nread_data ',1)
    (directory/'in.lammps').write_text(text)
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',LD_PRELOAD=str(probe),
               ZQC_COMM_TRACE=str(directory/'comm'))
    with (directory/'screen.txt').open('w') as log:
        process=subprocess.run([str(args.mpi),'--bind-to','none','-np',str(np),str(args.lmp),
            '-in','in.lammps'],cwd=directory,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
    if process.returncode:
        raise RuntimeError(f'{directory}: exit {process.returncode}\n'+(directory/'screen.txt').read_text()[-3000:])
    traces=[json.loads((directory/f'comm.rank{rank}.json').read_text()) for rank in range(np)]
    if not all(trace['records'] for trace in traces): raise AssertionError('communication probe was not invoked')
    if version=='after':
        for trace in traces:
            for direction,expected in zip(['forward','reverse'],CASES[case][3:]):
                records=[x for x in trace['records'] if x['direction']==direction]
                assert sorted({x['width'] for x in records})==expected, (case,direction,records)
                assert all(x['capacity']==max(expected) for x in records), records
    print(f'{version} {case} np={np}: completed',flush=True)
    return directory,traces


def compare_observables(a, b):
    result=compare(a,b)
    result['pressure_errors']=result['cv_energy_errors'][3:]
    result['cv_energy_errors']=result['cv_energy_errors'][:3]
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--baseline-plugin',type=Path)
    ap.add_argument('--after-only',action='store_true',help='以改造后的 1 rank 为数值参考，检查其余 rank')
    ap.add_argument('--plugin',type=Path,default=ROOT/'build/graph-lammps/fix_crystallize_plugin.so')
    ap.add_argument('--initial',type=Path,required=True,help='固定编号的 6x6x6 FCC 初态')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--np',type=int,nargs='+',default=[1,2,4])
    ap.add_argument('--cases',nargs='+',choices=list(CASES),default=None)
    ap.add_argument('--comm-style',choices=['brick','tiled'],default='brick')
    ap.add_argument('--lmp',type=Path,default=Path('/home/zqc/apps/deepmd/bin/lmp_mpi'))
    ap.add_argument('--mpi',type=Path,default=Path('/home/zqc/apps/deepmd/bin/mpirun'))
    ap.add_argument('--lammps-src',type=Path,default=Path('/home/zqc/apps/lammps/lammps-stable_29Aug2024/src'))
    args=ap.parse_args()
    if args.cases is None:
        args.cases=list(CASES)
    if not args.after_only and args.baseline_plugin is None: ap.error('--baseline-plugin is required for a before/after comparison')
    if args.after_only and args.np[0]!=1: ap.error('--after-only requires --np to start with 1')
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    args.initial=args.initial.resolve()
    plugins={'after':args.plugin.resolve()}
    if not args.after_only: plugins['before']=args.baseline_plugin.resolve()
    probe=args.output/'trace_comm.so'
    env=os.environ.copy();env['OMPI_CXX']='/usr/bin/g++'
    subprocess.run([str(args.mpi.parent/'mpicxx'),'-std=c++17','-shared','-fPIC',
        str(Path(__file__).with_name('trace_comm.cpp')),'-I'+str(args.lammps_src),'-ldl','-o',str(probe)],env=env,check=True)
    results=[]
    for case in args.cases:
        reference=None
        for np in args.np:
            if args.after_only:
                after,at=run_case(args,'after',plugins['after'],case,np,probe)
                if reference is None: reference=after
                row={'case':case,'np':np,'reference':'after_np1','comparison':compare_observables(reference,after),'after':at}
            else:
                before,bt=run_case(args,'before',plugins['before'],case,np,probe)
                after,at=run_case(args,'after',plugins['after'],case,np,probe)
                row={'case':case,'np':np,'comparison':compare_observables(before,after),'before':bt,'after':at}
                # 通信轮数及邻居相同，检查缩窄包宽确实减少发出的 double 数。
                for rank in range(np):
                    for direction in ['forward','reverse']:
                        old=[x for x in bt[rank]['records'] if x['direction']==direction]
                        new=[x for x in at[rank]['records'] if x['direction']==direction]
                        assert sum(x['calls'] for x in old)==sum(x['calls'] for x in new)
                        assert sum(x['sends'] for x in old)==sum(x['sends'] for x in new)
                        assert sum(x['doubles'] for x in new)<=sum(x['doubles'] for x in old)
            results.append(row)
            (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
            print(f'{case} np={np}: force/CV/energy and communication checks passed',flush=True)
    metadata={'comm_style':args.comm_style,'plugins':{k:{'path':str(v),'sha256':hashlib.sha256(v.read_bytes()).hexdigest()} for k,v in plugins.items()}}
    (args.output/'environment.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('CV_COMMUNICATION_PASS')


if __name__=='__main__': main()
