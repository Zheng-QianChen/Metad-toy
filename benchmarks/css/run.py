#!/usr/bin/env python3
"""CSS 全流程：独立 CPU 配位数、Graph、MPI、组外力和逐原子诊断。"""
import argparse,json,math,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/strufactor_lammps_graph'))
from importlib.util import spec_from_file_location,module_from_spec
_spec=spec_from_file_location('strufactor_fixture',ROOT/'benchmarks/strufactor_lammps_graph/run.py')
_fixture=module_from_spec(_spec);_spec.loader.exec_module(_fixture)
input_text=_fixture.input_text
from compare import thermo
LMP=Path('/home/zqc/apps/deepmd/bin/lmp_mpi')
MPI=Path('/home/zqc/apps/deepmd/bin/mpirun')
PLUGIN=ROOT/'build/graph-lammps/fix_crystallize_plugin.so'
EPS=1e-10
RADIAL={'first':(3.30,-8.),'outer':(4.50,-8.)}
PROFILES={
    'original':[(2,'first',0.,.45)],
    'ternary':[(1,'first',6.,2.),(2,'first',1.,1.2),(3,'first',2.,1.5)],
    'multi_radius':[(1,'first',6.,2.),(2,'first',1.,1.2),(3,'first',2.,1.5),(2,'outer',5.,2.)],
    'small_group':[(1,'first',6.,2.),(2,'first',1.,1.2),(3,'first',2.,1.5),(2,'outer',5.,2.)],
}

def initial(path,ternary):
    atoms=[];type_by_id={}
    for ix in range(4):
        for iy in range(4):
            for iz in range(4):
                for b,(dx,dy,dz) in enumerate([(0,0,0),(.5,.5,0),(.5,0,.5),(0,.5,.5)]):
                    typ=2 if b==0 else (3 if ternary and b==3 and (ix+iy+iz)%2==0 else 1)
                    aid=len(atoms)+1;type_by_id[aid]=typ
                    atoms.append(f'{aid} {typ} {4.05*(ix+dx):.17g} {4.05*(iy+dy):.17g} {4.05*(iz+dz):.17g}')
    nt=3 if ternary else 2
    path.write_text('CSS fixture\n\n256 atoms\n'+f'{nt} atom types\n\n0 16.2 xlo xhi\n0 16.2 ylo yhi\n0 16.2 zlo zhi\n\nMasses\n\n'+''.join(f'{i} 30.0\n' for i in range(1,nt+1))+'\nAtoms # atomic\n\n'+'\n'.join(atoms)+'\n')
    return type_by_id

def definition(name,group,channels):
    terms=' '.join(f'CHANNEL {t} {sw} {target:g} {sigma:g}' for t,sw,target,sigma in channels)
    return f'CAL NAME {name} CSS {group} {terms} cutoff_eps {EPS:g}'

def script(data,case,channels):
    s=input_text(PLUGIN,'GRAPH',4,1,0,True,data)
    group='small' if case=='small_group' else 'Sc'
    s=s.replace('variable dx0','group Sc type 2\ngroup small id 1 5 9\nvariable dx0',1)
    # 两个不同 CSS 共用同一组与最大 cutoff，验证共享环境和独立目标参数。
    second=[(t,sw,target+.3,sigma*1.1) for t,sw,target,sigma in channels]
    declarations=' &\n    '.join([f'CAL NAME {name} SW_FUNC FERMI r_0 {r:g} alpha {alpha:g}' for name,(r,alpha) in RADIAL.items()]+[definition('S1',group,channels),definition('S2',group,second)])
    start=s.index('CAL NAME S1 STRU_FACTOR');end=s.index('SYMBOL s1',start)
    s=s[:start]+declarations+' &\n    '+s[end:]
    s=s.replace('thermo 100000','compute bias_pressure all pressure NULL virial\nthermo 100000')
    s=s.replace('f_m[2] f_m\n','f_m[2] f_m '+ ' '.join(f'c_bias_pressure[{j}]' for j in range(1,7))+'\n')
    diagnostics=' '.join(['S1.css']+[f'S1.n{j+1}' for j in range(len(channels))])
    s=s.replace('dump state all custom',f'compute css_data all metad/atom m {diagnostics}\ndump state all custom')
    s=s.replace('id x y z fx fy fz\n','id x y z fx fy fz '+ ' '.join(f'c_css_data[{j}]' for j in range(1,len(channels)+2))+'\n')
    return s,second

def reference(atoms,types,centers,channels):
    values={};counts={}
    for i in centers:
        ns=[]
        for typ,sw,target,sigma in channels:
            r0,alpha=RADIAL[sw];cutoff=r0-math.log((1-EPS)/EPS)/alpha
            n=0.
            for j,row in atoms.items():
                if j==i or types[j]!=typ:continue
                dr=[row[k]-atoms[i][k] for k in range(3)];dr=[v-round(v/16.2)*16.2 for v in dr]
                r=math.sqrt(sum(v*v for v in dr))
                if r<=cutoff:n+=1/(1+math.exp(-alpha*(r-r0)))
            ns.append(n)
        values[i]=math.exp(-.5*sum(((n-c[2])/c[3])**2 for n,c in zip(ns,channels)))
        counts[i]=ns
    return sum(values.values())/len(centers),values,counts

def last_dump(path):
    lines=path.read_text().splitlines();starts=[i for i,line in enumerate(lines) if line.startswith('ITEM: ATOMS')]
    start=starts[-1];out={}
    for line in lines[start+1:]:
        if line.startswith('ITEM:'):break
        a=line.split();out[int(a[0])]=list(map(float,a[1:]))
    return out

def execute(d,text,np):
    d.mkdir(parents=True,exist_ok=False);(d/'input.lmp').write_text(text)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1')
    with (d/'screen.txt').open('w') as out:
        r=subprocess.run([str(MPI),'--bind-to','none','-np',str(np),str(LMP),'-in','input.lmp'],cwd=d,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=120)
    if r.returncode:raise RuntimeError(f'{d}: exit {r.returncode}\n'+(d/'screen.txt').read_text(errors='replace')[-3000:])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--np',type=int,nargs='+',default=[1,2,4]);a=ap.parse_args()
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True);results=[]
    for case,channels in PROFILES.items():
        data=a.output/f'{case}.data';types=initial(data,case!='original')
        centers=[1,5,9] if case=='small_group' else [i for i,t in types.items() if t==2]
        text,second=script(data,case,channels);baseline=None
        for np in a.np:
            d=a.output/f'{case}_np{np}';execute(d,text,np)
            details=last_dump(d/'atom_state.dump');atoms={i:row[:6] for i,row in details.items()};observed=thermo(d/'screen.txt')
            v1,vs,ns=reference(atoms,types,centers,channels);v2,_,_=reference(atoms,types,centers,second)
            cv_error=max(abs(observed[0]-v1*v2),abs(observed[1]-v1-v2))
            assert cv_error<2e-11,(case,np,observed[:2],v1,v2)
            diag_error=0.
            for i,row in details.items():
                expected=[vs.get(i,0)]+ns.get(i,[0]*len(channels))
                diag_error=max(diag_error,max(abs(x-y) for x,y in zip(row[6:],expected)))
            assert diag_error<2e-11,(case,np,diag_error)
            force=max(abs(v) for row in atoms.values() for v in row[3:6]);assert force>1e-10,(case,np,force)
            outside=max(abs(v) for i,row in atoms.items() if i not in centers for v in row[3:6])
            if case!='original':assert outside>1e-10
            net=max(abs(sum(row[k] for row in atoms.values())) for k in range(3,6));assert net<1e-11
            err=0.;pressure_error=0.
            if baseline:
                err=max(abs(v-w) for i,row in atoms.items() for v,w in zip(row,baseline[0][i]));assert err<1e-10,(case,np,err)
                pressure_error=max(abs(v-w) for v,w in zip(observed[3:],baseline[1][3:]));assert pressure_error<1e-10
            else:baseline=(atoms,observed)
            row=dict(case=case,np=np,cv_cpu_error=cv_error,diagnostic_error=diag_error,max_force=force,
                     outside_group_force=outside,net_force=net,mpi_state_error=err,mpi_pressure_error=pressure_error)
            results.append(row);print(json.dumps(row),flush=True)
            (a.output/'results.json').write_text(json.dumps(results,indent=2))
    print('CSS_LAMMPS_PASS',flush=True)
if __name__=='__main__':main()
