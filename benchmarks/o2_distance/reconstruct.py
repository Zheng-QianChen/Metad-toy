#!/usr/bin/env python3
"""Blind reconstruction: read sampling.json, HILLS and distances; never read model/forces/PE."""
import argparse
import json
from pathlib import Path
import numpy as np

BOLTZMANN_EV = 8.617343e-5  # LAMMPS metal units

def reconstruct(directory):
    sampling = json.loads((directory/'sampling.json').read_text())
    grid = np.linspace(.95, 1.85, 901)
    snapshots = []; records = []
    for run in sorted(directory.glob('seed_*')):
        hills = np.loadtxt(run/'HILLS', ndmin=2)
        trajectory = np.loadtxt(run/'trajectory.dat', ndmin=2)
        accumulated = np.zeros_like(grid); previous = 0
        for fraction in [.25, .5, .75, 1.]:
            stop = int(len(hills)*fraction)
            for start in range(previous, stop, 500):
                chunk = hills[start:min(stop,start+500)]
                displacement = (grid[:,None]-chunk[None,:,1])/chunk[None,:,3]
                accumulated += np.sum(chunk[None,:,2]*np.exp(-.5*displacement**2), axis=1)
            previous = stop
            free_energy = -sampling['bias_factor']/(sampling['bias_factor']-1.)*accumulated
            potential = free_energy + 2.*BOLTZMANN_EV*sampling['temperature_K']*np.log(grid)
            potential -= potential.min(); free_energy -= free_energy.min()
            np.savetxt(run/f'reconstruction_{fraction:.2f}.dat', np.column_stack([grid,free_energy,potential]),
                       header='r_A F_eV U_reconstructed_eV')
            if fraction == 1.: snapshots.append(potential)
        # 每个区间的访问次数用于标出轨迹覆盖范围，不用于套用已知势函数。
        counts, _ = np.histogram(trajectory[:,1], bins=np.r_[grid-.0005,grid[-1]+.0005])
        records.append(dict(seed=run.name, hills=len(hills), distance_min=float(trajectory[:,1].min()),
                            distance_max=float(trajectory[:,1].max()),
                            inferred_minimum_A=float(grid[np.argmin(potential)])))
        np.savetxt(run/'visits.dat', np.column_stack([grid,counts]), header='r_A sample_count')
    curves = np.array(snapshots)
    mean = curves.mean(axis=0)
    deviation = curves.std(axis=0, ddof=1) if len(curves)>1 else np.zeros_like(mean)
    np.savetxt(directory/'reconstructed.dat', np.column_stack([grid, mean, deviation]),
               header='r_A mean_U_eV between_seed_std_eV')
    (directory/'reconstruction.json').write_text(json.dumps(records, indent=2)+'\n')
    print(json.dumps(records, indent=2))

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    reconstruct(parser.parse_args().directory.resolve())
