#!/usr/bin/env python3
"""Reveal the Morse reference after blind reconstruction and report fixed-range errors."""
import argparse
import json
from pathlib import Path
import numpy as np

def validate(root):
    model=json.loads((root/'model.json').read_text())
    settings=json.loads((root/'sampling.json').read_text())
    data=np.loadtxt(root/'reconstructed.dat')
    radius,free_energy,potential=data.T
    reference=model['depth_eV']*(1-np.exp(-model['alpha_inverse_A']*(radius-model['equilibrium_A'])))**2
    mask=(radius>=settings['comparison_min_A'])&(radius<=settings['comparison_max_A'])
    difference=potential-reference
    shift=difference[mask].mean()
    difference-=shift
    rmse=float(np.sqrt(np.mean(difference[mask]**2)))
    maximum=float(np.max(np.abs(difference[mask])))
    samples=np.concatenate([np.loadtxt(p/'COLVAR',ndmin=2)[:,1] for p in sorted(root.glob('seed_*'))])
    covered=bool(samples.min()<=settings['comparison_min_A'] and samples.max()>=settings['comparison_max_A'])
    result=dict(comparison_A=[settings['comparison_min_A'],settings['comparison_max_A']],
                rmse_eV=rmse,max_error_eV=maximum,coverage=covered,
                inferred_minimum_A=float(radius[np.argmin(potential)]),
                passed=bool(covered and rmse<settings['rmse_limit_eV'] and maximum<settings['max_error_limit_eV']))
    (root/'fes_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    axes[0].plot(radius,free_energy,label='Reconstructed radial F(r)')
    axes[0].plot(radius,potential-shift,label='Entropy-corrected U(r)')
    axes[0].plot(radius,reference,'--',label='Revealed Morse reference')
    axes[0].set(xlabel='Distance (A)',ylabel='Energy (eV)',xlim=(1.0,1.7),ylim=(-.1,3.3))
    axes[0].legend(fontsize=8)
    axes[1].plot(radius[mask],difference[mask]*1000)
    axes[1].axhline(0,color='k',linewidth=.5)
    axes[1].set(xlabel='Distance (A)',ylabel='U reconstruction error (meV)',
                title='RMSE %.2f meV'%(rmse*1000))
    fig.savefig(root/'o2_fes.png',dpi=160)
    print(json.dumps(result,indent=2))
    return result['passed']

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    raise SystemExit(0 if validate(args.directory.resolve()) else 1)
