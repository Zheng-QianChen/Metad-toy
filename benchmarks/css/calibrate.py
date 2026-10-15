#!/usr/bin/env python3
"""固定 FCC 几何：比较 L12 占位与 Al 中单个 Sc 的局部配位评分。"""
import argparse,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run import PLUGIN,initial,execute,last_dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True);rows=[]
    for case in ['l12','isolated_sc']:
        data=a.output/f'{case}.data';initial(data,False)
        if case=='isolated_sc':
            header,atoms=data.read_text().split('Atoms # atomic\n\n')
            updated=[]
            for line in atoms.splitlines():
                fields=line.split();fields[1]='2' if fields[0]=='1' else '1';updated.append(' '.join(fields))
            data.write_text(header+'Atoms # atomic\n\n'+'\n'.join(updated)+'\n')
        text=f'''units metal
atom_style atomic
boundary p p p
read_data {data}
pair_style zero 4.0
pair_coeff * *
neighbor 0.35 bin
neigh_modify every 1 delay 0 check yes
group Sc type 2
plugin load {PLUGIN}
fix m all metad EXECUTOR GRAPH GAUSSIAN 0.1 0.1 10 PACE 100000 &
 CAL NAME first SW_FUNC FERMI r_0 3.30 alpha -20 &
 CAL NAME outer SW_FUNC FERMI r_0 4.50 alpha -20 &
 CAL NAME old CSS Sc CHANNEL 2 first 0 0.45 &
 CAL NAME extended CSS Sc CHANNEL 1 first 12 2 CHANNEL 2 first 0 0.45 CHANNEL 2 outer 6 1.5 &
 SYMBOL old old.AVE SYMBOL new extended.AVE &
 CV_dim 2 DIM 1 -0.02 1.02 104 old DIM 2 -0.02 1.02 104 new &
 METAD_RESTART 0 WT 0
compute data all metad/atom m old.css extended.css extended.n1 extended.n2 extended.n3
thermo_style custom step f_m[1] f_m[2]
thermo_modify format float %.17g
run 0
write_dump all custom scores.dump id x y z fx fy fz c_data[1] c_data[2] c_data[3] c_data[4] c_data[5] modify sort id format float %.17g
'''
        d=a.output/case;execute(d,text,1)
        center=last_dump(d/'scores.dump')[1]
        row=dict(case=case,original_css=center[6],extended_css=center[7],Al_inner=center[8],Sc_inner=center[9],Sc_outer=center[10])
        rows.append(row);print(json.dumps(row),flush=True)
    assert min(r['original_css'] for r in rows)>.999
    assert rows[0]['extended_css']>.99 and rows[1]['extended_css']<.001
    (a.output/'results.json').write_text(json.dumps(rows,indent=2))
    print('CSS_CALIBRATION_PASS')
if __name__=='__main__':main()
