#!/usr/bin/env python3
"""Blind radial FES reconstruction from HILLS and sampling metadata only."""
import argparse
import json
from pathlib import Path
import numpy as np

KB=8.617333262145e-5

def reconstruct(root):
    settings=json.loads((root/'sampling.json').read_text())
    grid=np.linspace(.95,1.85,901)
    curves=[]
    for run in sorted(root.glob('seed_*')):
        hills=np.loadtxt(run/'HILLS',ndmin=2)
        header=(run/'HILLS').read_text().splitlines()[:5]
        sigma=float(next(line for line in header if line.startswith('# SIGMA')).split()[2])
        if abs(sigma-settings['sigma_A'])>1e-14:
            raise ValueError('HILLS sigma disagrees with sampling metadata')
        accumulated=np.zeros_like(grid)
        previous=0
        for fraction in (.25,.5,.75,1.):
            stop=int(len(hills)*fraction)
            for start in range(previous,stop,500):
                chunk=hills[start:min(stop,start+500)]
                displacement=(grid[:,None]-chunk[None,:,1])/sigma
                accumulated+=(chunk[None,:,2]*np.exp(-.5*displacement**2)).sum(axis=1)
            previous=stop
            free_energy=-settings['bias_factor']/(settings['bias_factor']-1)*accumulated
            # 三维自由转动双原子的径向测度 r^2；此项将径向 F(r) 还原为 U(r)+C。
            potential=free_energy+2*KB*settings['temperature_K']*np.log(grid)
            free_energy-=free_energy.min()
            potential-=potential.min()
            data=np.column_stack([grid,free_energy,potential])
            np.savetxt(run/('reconstruction_%.2f.dat'%fraction),data,
                       header='r_A F_radial_eV U_reconstructed_eV')
            if fraction==1.: curves.append(data[:,1:])
    if not curves: raise ValueError('No sampling runs')
    np.savetxt(root/'reconstructed.dat',np.column_stack([grid,np.mean(curves,axis=0)]),
               header='r_A mean_F_radial_eV mean_U_reconstructed_eV')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    reconstruct(parser.parse_args().directory.resolve())
