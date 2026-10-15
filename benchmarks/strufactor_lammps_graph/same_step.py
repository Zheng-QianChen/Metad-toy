#!/usr/bin/env python3
"""共享环境回归：同一 timestep 修改坐标后再次 run 0，两个 CV 都必须更新。"""
import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from run import ROOT, input_text
from compare import read_dump
from groups import reference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--initial', type=Path, required=True, help='6x6x6 FCC，盒长 24.3')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--plugin', type=Path, default=ROOT/'build/graph-lammps/fix_crystallize_plugin.so')
    a = ap.parse_args()
    rows = []
    for np in [1, 2, 4]:
        d = a.output / f'np{np}'
        d.mkdir(parents=True, exist_ok=False)
        text = input_text(a.plugin.resolve(), 'GRAPH', 6, 1, 0, True, a.initial.resolve())
        text = text.split('variable dx1', 1)[0]
        # 不增加 timestep；第二次 setup 必须使共享环境失效，但每轮只更新一次。
        text += """run 0
print "BEFORE_CV $(step:%.0f) $(f_m[1]:%.17g) $(f_m[2]:%.17g)"
write_dump all custom before.dump id x y z fx fy fz modify sort id format float %.17g
variable dx1 atom 0.07*sin(0.413*id+0.1)
variable dy1 atom 0.06*cos(0.293*id+0.4)
variable dz1 atom 0.05*sin(0.619*id+0.7)
displace_atoms all move v_dx1 v_dy1 v_dz1 units box
run 0
print "AFTER_CV $(step:%.0f) $(f_m[1]:%.17g) $(f_m[2]:%.17g)"
write_dump all custom after.dump id x y z fx fy fz modify sort id format float %.17g
"""
        (d/'in.lammps').write_text(text)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1')
        with (d/'screen.txt').open('w') as out:
            result = subprocess.run(['/home/zqc/apps/deepmd/bin/mpirun', '--bind-to', 'none',
                '-np', str(np), '/home/zqc/apps/deepmd/bin/lmp_mpi', '-in', 'in.lammps'],
                cwd=d, env=env, stdout=out, stderr=subprocess.STDOUT, timeout=60)
        screen = (d/'screen.txt').read_text(errors='replace')
        if result.returncode:
            raise RuntimeError(screen[-2500:])
        values = {}
        max_error = 0.0
        for phase in ['BEFORE', 'AFTER']:
            match = re.search(r'^' + phase + r'_CV (\d+) ([^ ]+) ([^\n]+)$', screen, re.M)
            assert match and int(match[1]) == 100000, 'timestep unexpectedly changed'
            values[phase] = [float(match[2]), float(match[3])]
            atoms = read_dump(d / (phase.lower() + '.dump'))
            v1 = reference(atoms, list(atoms), 1.7, 24.3)
            v2 = reference(atoms, list(atoms), 2.3, 24.3)
            max_error = max(max_error, abs(values[phase][0] - v1*v2), abs(values[phase][1] - v1-v2))
        assert max_error < 1e-11, (np, max_error, values)
        change = max(abs(x-y) for x, y in zip(values['BEFORE'], values['AFTER']))
        assert change > 1e-5, 'coordinates changed but CVs did not'
        row = dict(np=np, timestep=100000, cv_cpu_error=max_error, max_cv_change=change, values=values)
        rows.append(row)
        print(json.dumps(row), flush=True)
    (a.output/'results.json').write_text(json.dumps(rows, indent=2))
    print('LAMMPS_SAME_STEP_PASS')


if __name__ == '__main__':
    main()
