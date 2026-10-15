#!/usr/bin/env python3
"""真实 LAMMPS：不同/非连续组、空组 rank、组外邻居力及 MPI 一致性。"""
import argparse,json,math,os,subprocess
from pathlib import Path
from run import input_text,ROOT
from compare import read_dump,thermo,compare

def reference(atoms,ids,q,length):
    result=0.0;rc2=3.3**2
    for atom in ids:
        center=atoms[atom][:3];value=1.0
        for other,row in atoms.items():
            if other==atom: continue
            delta=[row[k]-center[k] for k in range(3)]
            delta=[v-round(v/length)*length for v in delta]
            r2=sum(v*v for v in delta)
            if r2>=rc2: continue
            x=q*math.sqrt(r2);u=(r2-.8*rc2)/(.2*rc2)
            smooth=1 if r2<=.8*rc2 else 1-u**3
            value+=math.sin(x)/x*smooth
        result+=value
    return result/len(ids)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--initial',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    rows=[]
    for np in [1,2,4]:
        d=a.output/f'np{np}';d.mkdir(parents=True,exist_ok=False)
        text=input_text(ROOT/'build/graph-lammps/fix_crystallize_plugin.so','GRAPH',6,1,0,True,a.initial.resolve())
        text=text.replace('mass 1 26.9815385','mass 1 26.9815385')
        text=text.replace('variable dx0', 'group small id 1:20\ngroup sparse id 31:300:3\nvariable dx0',1)
        text=text.replace('STRU_FACTOR all cutoff_r 3.30 q_factor 1.7','STRU_FACTOR small cutoff_r 3.30 q_factor 1.7')
        text=text.replace('STRU_FACTOR all cutoff_r 3.30 q_factor 2.3','STRU_FACTOR sparse cutoff_r 3.30 q_factor 2.3')
        (d/'in.lammps').write_text(text)
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1')
        with (d/'screen.txt').open('w') as out:
            r=subprocess.run(['/home/zqc/apps/deepmd/bin/mpirun','--bind-to','none','-np',str(np),'/home/zqc/apps/deepmd/bin/lmp_mpi','-in','in.lammps'],cwd=d,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=60)
        if r.returncode:raise RuntimeError((d/'screen.txt').read_text(errors='replace')[-2500:])
        atoms=read_dump(d/'atom_state.dump');v1=reference(atoms,list(range(1,21)),1.7,24.3);v2=reference(atoms,list(range(31,301,3)),2.3,24.3)
        actual=thermo(d/'screen.txt');err=max(abs(actual[0]-v1*v2),abs(actual[1]-v1-v2))
        assert err<1e-11, (actual,v1,v2,err)
        centers=set(range(1,21))|set(range(31,301,3))
        outside=max(abs(row[k]) for atom,row in atoms.items() if atom not in centers for k in range(3,6))
        total=[sum(row[k] for row in atoms.values()) for k in range(3,6)]
        assert outside>1e-8 and max(map(abs,total))<1e-11,(outside,total)
        row=dict(np=np,cv_cpu_error=err,max_outside_group_force=outside,net_force=total)
        if np!=1:row['mpi_comparison']=compare(a.output/'np1',d)
        rows.append(row);print(json.dumps(row),flush=True)
    (a.output/'results.json').write_text(json.dumps(rows,indent=2))
    print('LAMMPS_DISTINCT_GROUPS_PASS')
if __name__=='__main__':main()
