"""采样原型验证：解析导数、历史状态、窗口交换规则和统计重建。

前半部分使用固定点与有限差分，后半部分实际采样 U(x)=3*(x²-1)²。
这里 kBT=1，能量误差的数值也就是以 kBT 为单位的误差。
固定随机种子便于重现，WHAM 的统计误差和逐点导数误差分别采用自己的容差。
"""
import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.checks import finite_difference, write_result
from sampling.bias import HarmonicUmbrella, WellTemperedMetaD, exchange_log_ratio
from sampling.wham import reconstruct


def run(output):
    """生成导数检查结果、重启文件、umbrella 直方图和 WHAM 对照图。
    
    18 个窗口各运行 160000 步，前 10000 步用于平衡，之后每 5 步收集一次。
    重建用共同等宽 bin；对参考势的 RMSE 在 |CV|<1.5 的已访问区域计算，
    比较前消除自由能任意加性常数。窗口重叠指标检查直方图是否彼此衔接。
    """
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    # 不同中心和不同刚度同时验证分量对应关系及 dV/ds 的符号。
    umbrella=HarmonicUmbrella([-.2,.5],[2.,7.])
    point=np.array([.4,-.3])
    error=float(np.max(np.abs(umbrella.evaluate(point)[1]-finite_difference(lambda x:umbrella.evaluate(x)[0],point))))
    assert error<1e-8
    meta=WellTemperedMetaD([.3,.6],pace=5)
    for step in range(41): meta.update(step,np.array([np.sin(step*.1),np.cos(step*.1)]))
    # 检查求值保持历史长度，以及同一步重复 update 具有幂等性。
    before=len(meta.centers)
    meta.evaluate(point); assert len(meta.centers)==before
    assert not meta.update(40,point) and len(meta.centers)==before
    meta_error=float(np.max(np.abs(meta.evaluate(point)[1]-finite_difference(lambda x:meta.evaluate(x)[0],point))))
    assert meta_error<1e-8
    # 对同一后续 CV 序列继续沉积，两条历史应逐项完全相同。
    meta.save(output/'restart.json'); restored=WellTemperedMetaD.load(output/'restart.json')
    for step in range(41,80):
        value=np.array([np.sin(step*.1),np.cos(step*.1)])
        meta.update(step,value); restored.update(step,value)
    assert meta.heights==restored.heights and meta.centers==restored.centers
    # 正反交换的 log 权重比互为相反数，接受率比应满足详细平衡关系。
    for left,right in [(-1.,1.),(.8,-.7),(.2,.2)]:
        a,b=HarmonicUmbrella([-1.],4.),HarmonicUmbrella([1.],4.)
        ratio=exchange_log_ratio(a,b,[left],[right],1.)
        reverse=exchange_log_ratio(a,b,[right],[left],1.)
        assert abs(ratio+reverse)<1e-13
        assert abs((min(0.,ratio)-min(0.,reverse))-ratio)<1e-13
    # 每个窗口是独立的真实马尔可夫链；重建只读取计数和 umbrella 偏置。
    rng=np.random.default_rng(20261017)
    centers=np.linspace(-1.7,1.7,18); stiffness=20.
    coordinates=centers.copy()
    def base(x): return 3*(x*x-1)**2
    def total(x): return base(x)+.5*stiffness*(x-centers)**2
    energy=total(coordinates)
    edges=np.linspace(-2.,2.,161); bins=.5*(edges[1:]+edges[:-1])
    histograms=np.zeros((len(centers),len(bins)),dtype=np.int64)
    accepted=0; samples=0
    for step in range(160000):
        proposed=coordinates+rng.normal(0.,.23,size=len(centers))
        candidate=total(proposed)
        # 对称高斯提议下，Metropolis 接受率由总能量差决定；这里 beta=1。
        accept=np.log(rng.random(len(centers))) < energy-candidate
        coordinates=np.where(accept,proposed,coordinates); energy=np.where(accept,candidate,energy)
        accepted+=int(accept.sum())
        if step>=10000 and step%5==0:
            indices=np.searchsorted(edges,coordinates,side='right')-1
            valid=(indices>=0)&(indices<len(bins))
            np.add.at(histograms,(np.arange(len(centers))[valid],indices[valid]),1)
            samples+=len(centers)
    bias=.5*stiffness*(bins[None,:]-centers[:,None])**2
    fes,iterations=reconstruct(histograms,bias)
    reference=base(bins)
    mask=(np.abs(bins)<1.5)&np.isfinite(fes)
    # 平移一个共同常数后比较曲线形状，避免自由能零点选择影响 RMSE。
    shift=np.mean(fes[mask]-reference[mask]); residual=fes[mask]-shift-reference[mask]
    rmse=float(np.sqrt(np.mean(residual**2)))
    assert rmse<.12, f'WHAM RMSE {rmse}'
    # 两个归一化直方图逐 bin 取较小值后求和；1 为完全重合，0 为没有重叠。
    overlap=np.sum(np.minimum(histograms[:-1]/histograms[:-1].sum(axis=1)[:,None],histograms[1:]/histograms[1:].sum(axis=1)[:,None]),axis=1)
    assert overlap.min()>.1
    np.savez(output/'umbrella_samples.npz',histograms=histograms,bin_centers=bins,window_centers=centers,fes=fes,reference=reference)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axis=plt.subplots(figsize=(6,4)); axis.plot(bins,reference,label='Reference'); axis.plot(bins,fes-shift,label='Umbrella + WHAM')
    axis.set(xlim=(-1.6,1.6),ylim=(-.3,7),xlabel='CV',ylabel='Free energy / kBT'); axis.legend(); fig.tight_layout(); fig.savefig(output/'umbrella_wham.png',dpi=160); plt.close(fig)
    write_result(output,'sampling',{'umbrella_gradient_max_error':error,'metad_gradient_max_error':meta_error,'restart_exact':True,'samples':samples,'wham_rmse_kbt':rmse,'wham_iterations':iterations,'minimum_neighbor_overlap':float(overlap.min()),'acceptance':accepted/(160000*len(centers))})

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True); run(parser.parse_args().output)
