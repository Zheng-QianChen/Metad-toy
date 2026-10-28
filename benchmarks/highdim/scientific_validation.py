#!/usr/bin/env python3
"""Frozen-bias rejection/reweighting against correlated Gaussian ground truth.

Uses the compiled C++ probe, not the Python prototype, for bias evaluations.
FES RMSE is a held-out parametric Gaussian reconstruction, reported in kBT.
It is not a claim of nonparametric 128-dimensional density convergence.
"""
import argparse, ctypes, json
from pathlib import Path
import numpy as np

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=80000)
    args=parser.parse_args()
    library=ctypes.CDLL(str(args.library.resolve()))
    pointer=ctypes.POINTER(ctypes.c_double)
    library.bias_probe_create.argtypes=[ctypes.c_int,ctypes.c_int,pointer,pointer,pointer,ctypes.c_int]
    library.bias_probe_create.restype=ctypes.c_void_p
    library.bias_probe_destroy.argtypes=[ctypes.c_void_p]
    library.bias_probe_evaluate.argtypes=[ctypes.c_void_p,pointer,ctypes.c_int,pointer,pointer]
    library.bias_probe_error.restype=ctypes.c_char_p
    def ptr(array): return array.ctypes.data_as(pointer)
    def create(dimension,parallel,widths,centers,heights):
        result=library.bias_probe_create(dimension,parallel,ptr(widths),ptr(centers),ptr(heights),len(centers))
        if not result: raise RuntimeError(library.bias_probe_error().decode())
        return result
    def evaluate(bias,points):
        points=np.ascontiguousarray(points,dtype=np.float64)
        energies=np.empty(len(points))
        if library.bias_probe_evaluate(bias,ptr(points),len(points),ptr(energies),None):
            raise RuntimeError(library.bias_probe_error().decode())
        return energies
    def weighted_covariance(points,weights):
        weights=weights/weights.sum()
        mean=weights@points
        centered=points-mean
        covariance=centered.T@(weights[:,None]*centered)
        return mean,covariance
    def fes_rmse(test,mean,covariance,inverse_true):
        estimated=0.5*np.einsum('ij,ij->i',(test-mean)@np.linalg.inv(covariance),test-mean)
        exact=0.5*np.einsum('ij,ij->i',test@inverse_true,test)
        error=estimated-exact
        error-=error.mean() # FES 的任意加法常数用独立测试集统一对齐。
        return float(np.sqrt(np.mean(error*error)))
    random=np.random.default_rng(20261023)
    records=[]
    for dimension in [2,8,32,128]:
        correlation=0.65
        covariance=(1-correlation)*np.eye(dimension)+correlation*np.ones((dimension,dimension))
        inverse=np.linalg.inv(covariance)
        whiten=np.linalg.inv(np.linalg.cholesky(covariance))
        def correlated(count):
            return np.sqrt(1-correlation)*random.normal(size=(count,dimension))+np.sqrt(correlation)*random.normal(size=(count,1))
        # 固定的 PB 由三个一维 hill 批次构造；每维高度一致，组合势非负。
        widths=np.full(dimension,0.6)
        centers=np.ascontiguousarray(np.repeat(np.array([[-0.8],[0.1],[0.9]]),dimension,axis=1))
        heights=np.full((3,dimension),0.7)
        bias=create(dimension,1,widths,centers,heights)
        accepted=[];energy_chunks=[];count=0;proposals=0
        try:
            while count<args.samples:
                points=correlated(4096);energy=evaluate(bias,points)
                selected=random.random(len(points))<np.exp(-energy)
                accepted.append(points[selected]);energy_chunks.append(energy[selected])
                count+=int(selected.sum());proposals+=len(points)
            points=np.concatenate(accepted)[:args.samples]
            energy=np.concatenate(energy_chunks)[:args.samples]
        finally: library.bias_probe_destroy(bias)
        weights=np.exp(energy-energy.max())
        mean,recovered=weighted_covariance(points,weights)
        naive_mean,naive_covariance=weighted_covariance(points,np.ones(len(points)))
        test=correlated(10000)
        rmse=fes_rmse(test,mean,recovered,inverse)
        naive_rmse=fes_rmse(test,naive_mean,naive_covariance,inverse)
        whitened_error=float(np.linalg.norm(whiten@recovered@whiten.T-np.eye(dimension),'fro')/np.sqrt(dimension))
        effective=float(weights.sum()**2/(weights@weights))
        assert rmse<0.8 and whitened_error<0.08,(dimension,rmse,whitened_error)
        # 单个非周期 Gaussian hill 的采样协方差应恢复 diag(sigma^2)。
        # 提议分布仅略宽；宽化随 sqrt(D) 缩放，避免高维接受率指数崩溃。
        widths=np.linspace(0.5,1.1,dimension)
        center=np.zeros((1,dimension));height=np.ones(1)
        kernel=create(dimension,0,widths,center,height)
        factor=1+0.15/np.sqrt(dimension)
        samples=[];count=0;kernel_proposals=0
        try:
            while count<30000:
                standardized=random.normal(size=(4096,dimension))
                proposed=np.ascontiguousarray(standardized*(widths*factor))
                value=evaluate(kernel,proposed)
                proposal_shape=np.exp(-0.5*np.sum(standardized**2,axis=1))
                selected=random.random(len(value))<value/proposal_shape
                samples.append(proposed[selected]);count+=int(selected.sum());kernel_proposals+=len(value)
            kernel_points=np.concatenate(samples)[:30000]/widths
        finally: library.bias_probe_destroy(kernel)
        kernel_covariance=np.cov(kernel_points,rowvar=False,bias=True)
        kernel_error=float(np.linalg.norm(kernel_covariance-np.eye(dimension),'fro')/np.sqrt(dimension))
        assert kernel_error<0.1,(dimension,kernel_error)
        record=dict(dimension=dimension,correlation=correlation,samples=args.samples,
                    fes_method='Gaussian weighted-moment fit; held-out RMSE; additive constant removed',
                    fes_rmse_kbt=rmse,unweighted_fes_rmse_kbt=naive_rmse,
                    sampling_covariance_whitened_error=whitened_error,effective_samples=effective,
                    acceptance=args.samples/proposals,
                    hill_covariance_whitened_error=kernel_error,hill_acceptance=30000/kernel_proposals,
                    passed=True)
        records.append(record);print(json.dumps(record),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(records,indent=2)+'\n')
if __name__=='__main__': main()

