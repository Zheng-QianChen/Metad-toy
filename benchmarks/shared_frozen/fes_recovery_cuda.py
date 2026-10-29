#!/usr/bin/env python3
"""Check the deposited physical-coordinate table through production CUDA bias."""
import argparse
import ctypes
import json
from pathlib import Path
import numpy as np
from kinetics import Bridge
from probability import pointer

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kinetics-library", type=Path, required=True)
    parser.add_argument("--cuda-library", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args=parser.parse_args()
    table=args.directory/"deposited.table"
    host=Bridge(args.kinetics_library, args.directory/"zero.table")
    cuda=ctypes.CDLL(str(args.cuda_library.resolve()))
    ptr=ctypes.POINTER(ctypes.c_double)
    cuda.shared_probability_cuda_evaluate.argtypes=[ctypes.c_char_p,ctypes.c_int,ptr,ptr,ptr]
    def evaluate(coordinates):
        coordinates=np.ascontiguousarray(coordinates)
        energy=np.empty(len(coordinates))
        gradient=np.empty_like(coordinates)
        status=cuda.shared_probability_cuda_evaluate(str(table).encode(),len(coordinates),
                pointer(coordinates),pointer(energy),pointer(gradient))
        if status:
            raise RuntimeError("production CUDA bias evaluation failed")
        return energy,gradient
    coordinates=np.random.default_rng(82310).uniform(-3.4,3.4,(512,2))
    coordinates[:6]=[[-3,3],[-3.01,3.01],[0,0],[-1,1],[-3+1e-8,3-1e-8],[.001,-.001]]
    measured_energy,measured_gradient=evaluate(coordinates)
    host_values=host.evaluate(table,3,coordinates)
    x,y=coordinates.T
    physical_energy=4*((x*x-1)**2+(y*y-1)**2)+(x-y)**2
    physical_force=np.column_stack([-16*x*(x*x-1)-2*(x-y),-16*y*(y*y-1)+2*(x-y)])
    energy_error=float(np.max(abs(measured_energy-(host_values[:,0]-physical_energy))))
    gradient_error=float(np.max(abs(measured_gradient-(physical_force-host_values[:,1:]))))
    swapped_energy,swapped_gradient=evaluate(coordinates[:,::-1])
    permutation_error=max(float(np.max(abs(swapped_energy-measured_energy))),
                          float(np.max(abs(swapped_gradient[:,::-1]-measured_gradient))))
    finite_difference_error=0.
    for component in (0,1):
        plus,minus=coordinates.copy(),coordinates.copy()
        delta=1e-6
        plus[:,component]+=delta
        minus[:,component]-=delta
        finite=(evaluate(plus)[0]-evaluate(minus)[0])/(2*delta)
        finite_difference_error=max(finite_difference_error,
                                     float(np.max(abs(finite-measured_gradient[:,component]))))
    assert energy_error<2e-12 and gradient_error<2e-11
    assert permutation_error<2e-12 and finite_difference_error<2e-6
    result=dict(status="passed",pairs=len(coordinates),energy_max=energy_error,
        gradient_max=gradient_error,permutation_max=permutation_error,
        finite_difference_max=finite_difference_error,
        host="same physical-CV interpolator as Langevin sampler",
        device="production FrozenSharedBias via CUDA benchmark bridge")
    (args.directory/"cuda_validation.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))

if __name__=="__main__":
    main()
