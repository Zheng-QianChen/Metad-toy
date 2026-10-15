#!/usr/bin/env python3
"""按全局 atom ID 比较实际 LAMMPS 受力、CV、偏置能量及 MPI 分解一致性。"""
import argparse,json,math,re
from pathlib import Path

def read_dump(path):
    text=path.read_text().split('ITEM: ATOMS id x y z fx fy fz\n')[-1]
    return {int(row[0]):list(map(float,row[1:])) for row in (line.split() for line in text.splitlines())}

def thermo(path):
    lines=path.read_text(errors='replace').splitlines()
    return [list(map(float,line.split())) for line in lines if re.match(r'^\s*100002\s+\d+\s',line)][-1][2:]

def compare(a,b):
    x,y=read_dump(a/'atom_state.dump'),read_dump(b/'atom_state.dump')
    assert x.keys()==y.keys()
    errors=[max(abs(x[tag][k]-y[tag][k]) for tag in x) for k in range(6)]
    cv_energy=[abs(v-w) for v,w in zip(thermo(a/'screen.txt'),thermo(b/'screen.txt'))]
    assert all(math.isfinite(v) for v in errors+cv_energy)
    force=max(errors[3:]);magnitude=max(abs(row[k]) for row in y.values() for k in range(3,6))
    assert magnitude>1e-8, 'zero forces are not a backward validation'
    assert force<1e-10 and max(errors[:3])<1e-10 and max(cv_energy)<1e-10, (errors,cv_energy)
    return dict(a=str(a),b=str(b),max_position_error=max(errors[:3]),max_force_error=force,max_force=magnitude,cv_energy_errors=cv_energy)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);args=ap.parse_args()
    rows=[]
    for np in [1,2,4]:
        rows.append(compare(args.directory/f'np{np}_legacy_0',args.directory/f'np{np}_graph_0'))
        if np!=1: rows.append(compare(args.directory/'np1_graph_0',args.directory/f'np{np}_graph_0'))
    (args.directory/'comparison.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2));print('LAMMPS_GRAPH_COMPARISON_PASS')
