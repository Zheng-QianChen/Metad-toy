#!/usr/bin/env python3
"""完整 LAMMPS timestep 对照；NVRTC 在 setup 完成，计时取最后一段 run 的 Loop time。"""
import argparse, json, os, re, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def input_text(plugin, executor, cells, steps, warmup, validate, initial, motion="translate"):
    header = f"""units metal
atom_style atomic
boundary p p p
read_data {initial}
variable dx0 atom 0.018*sin(0.731*id)
variable dy0 atom 0.016*cos(0.527*id)
variable dz0 atom 0.014*sin(0.319*id+0.2)
displace_atoms all move v_dx0 v_dy0 v_dz0 units box
pair_style zero 4.0
pair_coeff * *
neighbor 0.35 bin
neigh_modify every 1 delay 0 check yes
plugin load {plugin}
reset_timestep 99999
fix m all metad EXECUTOR {executor} GAUSSIAN 0.10 0.10 10.0 &
    PACE 100000 &
    CAL NAME S1 STRU_FACTOR all cutoff_r 3.30 q_factor 1.7 d_block_size 128 &
    CAL NAME S2 STRU_FACTOR all cutoff_r 3.30 q_factor 2.3 d_block_size 128 &
    SYMBOL s1 S1.AVE SYMBOL s2 S2.AVE &
    CV_dim 2 DIM 1 -5 5 500 "s1*s2" DIM 2 -5 5 500 "s1+s2" &
    METAD_RESTART 0 WT 0
velocity all set 0 0 0
fix integrate all nve
fix hold all setforce 0 0 0
thermo 100000
thermo_style custom step atoms f_m[1] f_m[2] f_m
thermo_modify format float %.17g
run 1
unfix hold
variable dx1 atom 0.07*sin(0.413*id+0.1)
variable dy1 atom 0.06*cos(0.293*id+0.4)
variable dz1 atom 0.05*sin(0.619*id+0.7)
displace_atoms all move v_dx1 v_dy1 v_dz1 units box
reset_timestep 100001
"""
    if validate:
        return header + """dump state all custom 1 atom_state.dump id x y z fx fy fz
dump_modify state sort id format float %.17g
run 1
"""
    if motion == "translate":
        # 规定相同平移轨迹，保留非零偏置力计算与跨 rank 迁移；受力不改变轨迹。
        # 避免长 NVE 轨迹分叉，或离开 hill 后 Graph 的零梯度捷径影响工作量。
        return header + f"""unfix integrate
fix integrate all move linear 0.10 0.05 0.025 units box
run {warmup}
run {steps}
"""
    # 可选 NVE 长轨迹，用于检查自然演化，不作为相同非零受力工作量的主计时。
    return header + f"""variable vx atom 0.1*sin(0.113*id)
variable vy atom 0.1*cos(0.193*id)
variable vz atom 0.1*sin(0.317*id)
velocity all set v_vx v_vy v_vz
run {warmup}
run {steps}
"""

def run_case(args, executor, np, repeat):
    d = args.output / f'np{np}_{executor.lower()}_{repeat}'
    d.mkdir(parents=True, exist_ok=False)
    (d/'in.lammps').write_text(input_text(args.plugin, executor, args.cells, args.steps, args.warmup, args.validate, args.output/"initial.data", args.motion))
    cmd=[str(args.mpi), '--bind-to','none','-np',str(np),str(args.lmp),'-in','in.lammps','-log','log.lammps']
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1')
    start=time.perf_counter()
    with (d/'screen.txt').open('w') as out:
        result=subprocess.run(cmd,cwd=d,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=900)
    wall=time.perf_counter()-start
    text=(d/'screen.txt').read_text(errors='replace')
    if result.returncode: raise RuntimeError(f'{d}: exit {result.returncode}\n{text[-3000:]}')
    loops=re.findall(r'Loop time of ([0-9.eE+-]+) on .*? for (\d+) steps with (\d+) atoms',text)
    if not loops: raise RuntimeError(f'{d}: no LAMMPS Loop time')
    seconds,steps,atoms=loops[-1]
    compile_times=re.findall(r'NVRTC ready: compile_ms=([0-9.]+) load_ms=([0-9.]+)',text)
    expected=1 if executor=='GRAPH' else 0
    if len(compile_times)!=expected: raise RuntimeError(f'{d}: expected {expected} compile, got {len(compile_times)}')
    row=dict(motion=args.motion,executor=executor,np=np,repeat=repeat,atoms=int(atoms),steps=int(steps),loop_s=float(seconds),
             us_per_step=1e6*float(seconds)/int(steps),wall_s=wall,nvrtc_ms=float(compile_times[0][0]) if compile_times else 0,
             load_ms=float(compile_times[0][1]) if compile_times else 0,directory=str(d))
    (d/'result.json').write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
    return row

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--plugin',type=Path,default=ROOT/'build/graph-lammps/fix_crystallize_plugin.so')
    ap.add_argument('--lmp',type=Path,default=Path('/home/zqc/apps/deepmd/bin/lmp_mpi'))
    ap.add_argument('--mpi',type=Path,default=Path('/home/zqc/apps/deepmd/bin/mpirun'))
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--cells',type=int,default=10)
    ap.add_argument('--steps',type=int,default=5000)
    ap.add_argument('--warmup',type=int,default=200)
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--np',type=int,nargs='+',default=[1,2,4])
    ap.add_argument('--executors',nargs='+',choices=['GRAPH','LEGACY'],default=['GRAPH'])
    ap.add_argument('--validate',action='store_true')
    ap.add_argument('--motion',choices=['translate','nve'],default='translate')
    args=ap.parse_args();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    # 固定全局 ID 与坐标，让不同 MPI 分解读取同一份物理初态。
    initial = args.output/'initial.data'
    if not initial.exists():
        atoms=[]
        for ix in range(args.cells):
            for iy in range(args.cells):
                for iz in range(args.cells):
                    for dx,dy,dz in [(0,0,0),(.5,.5,0),(.5,0,.5),(0,.5,.5)]:
                        atoms.append(f"{len(atoms)+1} 1 {4.05*(ix+dx):.17g} {4.05*(iy+dy):.17g} {4.05*(iz+dz):.17g}")
        length=4.05*args.cells
        initial.write_text(f"Fixed FCC atom IDs\n\n{len(atoms)} atoms\n1 atom types\n\n0 {length} xlo xhi\n0 {length} ylo yhi\n0 {length} zlo zhi\n\nMasses\n\n1 26.9815385\n\nAtoms # atomic\n\n"+'\n'.join(atoms)+'\n')
    rows=[]
    for np in args.np:
        for repeat in range(args.repeats):
            for executor in (args.executors if repeat%2==0 else args.executors[::-1]):
                rows.append(run_case(args,executor,np,repeat))
                (args.output/'results.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__': main()
