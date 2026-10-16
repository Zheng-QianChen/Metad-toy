#!/usr/bin/env python3
"""Reveal the reference only after blind reconstruction; save errors and a scientific plot."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def validate(directory):
    data=np.loadtxt(directory/'reconstructed.dat');grid,estimate,deviation=data.T
    reference=json.loads((directory/'reference_model.json').read_text())
    sampling=json.loads((directory/'sampling.json').read_text())
    target=reference['depth_eV']*(1-np.exp(-reference['alpha_inverse_A']*(grid-reference['equilibrium_A'])))**2
    # 比较区间在揭开参考前固定，位于三条轨迹共同覆盖的内部；零点统一在各曲线极小值。
    inside=(grid>=1.05)&(grid<=1.60)
    def metrics(values):
        difference=values[inside]-target[inside]
        return dict(rmse_eV=float(np.sqrt(np.mean(difference**2))),max_abs_error_eV=float(np.max(np.abs(difference))),
                    inferred_minimum_A=float(grid[np.argmin(values)]))
    records=[];curves=[]
    for run in sorted(directory.glob('seed_*')):
        record=dict(seed=run.name,checkpoints=[])
        for fraction in [.25,.5,.75,1.]:
            curve=np.loadtxt(run/f'reconstruction_{fraction:.2f}.dat')[:,2]
            record['checkpoints'].append(dict(fraction=fraction,**metrics(curve)))
            if fraction==1.:curves.append(curve)
        records.append(record)
    report=dict(interval_A=[1.05,1.60],reference_minimum_A=reference['equilibrium_A'],
                ensemble=metrics(estimate),runs=records,
                note='Finite-time WT-MetaD estimate; seed standard deviation is not a confidence interval. Morse parameters are an illustrative validation model.')
    # 宽于本机统计波动的验收阈值，检查默认长轨迹的重建质量及区间覆盖。
    coverage=json.loads((directory/'reconstruction.json').read_text())
    report['criteria']=dict(each_seed_rmse_eV=.025, each_seed_max_error_eV=.075, minimum_error_A=.01)
    report['passed']=all(row['distance_min']<=1.05 and row['distance_max']>=1.60 for row in coverage) and all(
        record['checkpoints'][-1]['rmse_eV']<.025 and record['checkpoints'][-1]['max_abs_error_eV']<.075
        and abs(record['checkpoints'][-1]['inferred_minimum_A']-reference['equilibrium_A'])<.01 for record in records)
    (directory/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savetxt(directory/'comparison.dat',np.column_stack([grid,target,estimate,deviation]),
               header='r_A reference_U_eV reconstructed_U_eV seed_std_eV')
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    ax=axes[0,0]
    ax.plot(grid,target,'k--',lw=1.8,label='Hidden Morse reference (revealed)')
    ax.plot(grid,estimate,color='#007d8a',label='WT-MetaD + radial entropy correction')
    ax.fill_between(grid,estimate-deviation,estimate+deviation,color='#007d8a',alpha=.2,label='Between-seed standard deviation')
    ax.axvspan(1.05,1.60,color='grey',alpha=.07)
    ax.set(xlim=(.99,1.83),ylim=(-.1,4.5),xlabel='O-O distance (Angstrom)',ylabel='Relative potential (eV)',title='Blind reconstruction from DISTANCE + HILLS')
    ax.legend(fontsize=8)
    ax=axes[0,1]
    for curve in curves:ax.plot(grid[inside],1000*(curve-target)[inside],alpha=.45,lw=1)
    ax.plot(grid[inside],1000*(estimate-target)[inside],color='k',lw=1.8,label='Three-seed mean')
    ax.axhline(0,color='grey',lw=.7);ax.set(xlim=(1.05,1.6),xlabel='O-O distance (Angstrom)',ylabel='Reconstruction error (meV)',title='Error in the stated comparison interval');ax.legend(fontsize=8)
    ax=axes[1,0]
    for run in sorted(directory.glob('seed_*')):
        trajectory=np.loadtxt(run/'trajectory.dat')[::200]
        ax.plot(trajectory[:,0]*sampling['timestep_ps']/1000,trajectory[:,1],alpha=.5,lw=.6,label=run.name)
    ax.set(xlabel='Time (ns)',ylabel='O-O distance (Angstrom)',title='Three independent sampling trajectories')
    ax=axes[1,1]
    for record in records:
        ax.plot([row['fraction']*sampling['steps']*sampling['timestep_ps']/1000 for row in record['checkpoints']],
                [1000*row['rmse_eV'] for row in record['checkpoints']],marker='o',label=record['seed'])
    ax.set(xlabel='Accumulated time (ns)',ylabel='Potential RMSE (meV)',title='Finite-time convergence, 1.05-1.60 Angstrom');ax.legend(fontsize=8)
    fig.suptitle(f"O-mass diatomic Morse test | {sampling['temperature_K']:g} K | bias factor {sampling['bias_factor']:g} | {len(curves)} x {sampling['steps']*sampling['timestep_ps']/1000:g} ns", fontsize=13)
    fig.savefig(directory/'reconstruction.png',dpi=180)
    fig.savefig(directory/'reconstruction.pdf')
    print(json.dumps(report,indent=2))
    if not report['passed']: raise SystemExit(1)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    validate(parser.parse_args().directory.resolve())
