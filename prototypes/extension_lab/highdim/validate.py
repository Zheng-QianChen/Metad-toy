"""高维偏置的确定性导数检查与冻结偏置联合 FES 的统计验证。

导数检查覆盖维度、hill 分块、完整协方差、周期接缝及 PBMetaD 组合规则。
二维统计验证选用已知相关高斯，从固定 PB 偏置下生成样本，再独立检查恢复
的联合自由能和协方差。所有数据保存在 output，便于核对数值与图。
"""
import argparse
from pathlib import Path
import sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.checks import finite_difference,write_result
from highdim.bias import HillBias,ParallelBiasMetaD
from highdim.reweight import frozen_bias_fes


def run(output):
    """运行四种维度的核检查、PBMetaD 检查及二维重加权示例。
    
    高维点与 hill 中心按 1/sqrt(D) 缩放，使平方距离在不同维度仍为同一量级，
    避免高斯值几乎全为零导致导数测试失去辨识力。
    二维 FES 只在权重有效样本数超过 400 的 bin 上比较，消除共同常数后报告 RMSE。
    """
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(77123)
    dimensional=[]
    for dimension in (2,8,32,128):
        bias=HillBias(np.full(dimension,1.5),chunk_size=7)
        for center in rng.normal(0.,.4/np.sqrt(dimension),size=(64,dimension)):bias.add(center,.2)
        point=rng.normal(0.,.2/np.sqrt(dimension),size=dimension)
        # 记录一次求值成本，作为测试诊断；稳定性能评估需要另做预热和重复测量。
        start=time.perf_counter()
        energy,gradient=bias.evaluate(point)
        seconds=time.perf_counter()-start
        numerical=finite_difference(lambda value:bias.evaluate(value)[0],point)
        error=float(np.max(np.abs(gradient-numerical)));assert error<1e-7
        # 不同分块大小必须给出同样的总偏置和导数。
        bias.chunk_size=64
        alternate=bias.evaluate(point)
        np.testing.assert_allclose(alternate[0],energy,rtol=1e-13);np.testing.assert_allclose(alternate[1],gradient,rtol=1e-12,atol=1e-13)
        # numeric_hill_storage_bytes 只计中心和高度的 FP64 数值，不含 Python 对象开销。
        dimensional.append({'dimension':dimension,'hills':64,'gradient_max_error':error,'single_evaluation_seconds':seconds,'numeric_hill_storage_bytes':8*64*(dimension+1),'dense_100_bins_per_axis_cells':str(100**dimension)})
    # -pi 与 +pi 是同一个周期点；同时检查两端的能量、导数和跨接缝有限差分。
    periodic=HillBias([.4,.7],periods=[2*np.pi,0.]);periodic.add([2.9,.2],.8)
    left=periodic.evaluate(np.array([-np.pi,.4]));right=periodic.evaluate(np.array([np.pi,.4]))
    np.testing.assert_allclose(left[0],right[0],atol=1e-14);np.testing.assert_allclose(left[1],right[1],atol=1e-14)
    np.testing.assert_allclose(left[1],finite_difference(lambda x:periodic.evaluate(x)[0],[-np.pi,.4]),atol=1e-8)
    # 非零交叉协方差使核倾斜，可检查矩阵方向和非对角导数的实现。
    covariance=np.array([[.4,.12],[.12,.2]])
    anisotropic=HillBias([1.,1.],covariance=covariance);anisotropic.add([.1,-.2],.6)
    point=np.array([.2,.5]);full_error=float(np.max(np.abs(anisotropic.evaluate(point)[1]-finite_difference(lambda x:anisotropic.evaluate(x)[0],point))))
    assert full_error<1e-8
    anisotropic.save(output/'hill_restart.json');reloaded=HillBias.load(output/'hill_restart.json')
    np.testing.assert_array_equal(reloaded.evaluate(point)[1],anisotropic.evaluate(point)[1])
    invalid=False
    try: HillBias([1.,1.],covariance=[[1.,2.],[2.,1.]])
    except np.linalg.LinAlgError: invalid=True
    assert invalid
    pb=ParallelBiasMetaD([.5,.6],pace=5)
    # 空历史中两维的权重均为 1/2，所以首轮 0.1 的基础高度各分配 0.05。
    pb.update(0,[0.,0.]);np.testing.assert_allclose([b.heights[0] for b in pb.biases],[.05,.05])
    for step in range(1,51):pb.update(step,[np.sin(step/10),np.cos(step/10)])
    assert not pb.update(50,[0.,0.])
    point=np.array([.3,-.4])
    pb_error=float(np.max(np.abs(pb.evaluate(point)[1]-finite_difference(lambda x:pb.evaluate(x)[0],point))))
    assert pb_error<1e-8
    pb.save(output/'pb_restart.json');pb_copy=ParallelBiasMetaD.load(output/'pb_restart.json')
    for step in range(51,70):
        pb.update(step,point);pb_copy.update(step,point)
    np.testing.assert_array_equal(pb.evaluate(point)[1],pb_copy.evaluate(point)[1])
    # 已知相关高斯作为独立参考体系。拒绝采样精确地产生 p_b ∝ p_0 exp(-V_PB/kBT)。
    # 此处偏置全程冻结；重加权只使用样本和各样本处的偏置值。
    frozen=ParallelBiasMetaD([.6,.7])
    for index,component in enumerate(frozen.biases):
        for center in (-1.,0.,1.):component.add([center],.7+.1*index)
    target_covariance=np.array([[1.,.65],[.65,1.]])
    proposal=rng.multivariate_normal([0.,0.],target_covariance,size=700000)
    energies=frozen.evaluate(proposal)[0]
    # proposal 来自 p0，接受概率 exp(-V_PB)；本例 kBT=1 且 V_PB>=0，
    # 所以保留下来的独立样本严格服从 p0*exp(-V_PB) 的归一化分布。
    accepted=np.log(rng.random(len(proposal)))<-energies
    samples=proposal[accepted];energies=energies[accepted]
    edges=(np.linspace(-2.4,2.4,25),)*2
    fes,effective,weights=frozen_bias_fes(samples,energies,edges)
    centers=.5*(edges[0][1:]+edges[0][:-1]);xx,yy=np.meshgrid(centers,centers,indexing='ij')
    grid=np.stack((xx,yy),axis=-1)
    reference=.5*np.einsum('...i,ij,...j->...',grid,np.linalg.inv(target_covariance),grid)
    mask=(effective>400)&np.isfinite(fes)
    residual=fes[mask]-reference[mask];offset=residual.mean()
    rmse=float(np.sqrt(np.mean((residual-offset)**2)))
    weighted_mean=np.average(samples,axis=0,weights=weights)
    delta=samples-weighted_mean
    # 对全部接受样本求加权协方差，另行核对二维相关性，补充 histogram 的 FES 检查。
    recovered_covariance=np.einsum('ni,nj,n->ij',delta,delta,weights)/weights.sum()
    covariance_error=float(np.max(np.abs(recovered_covariance-target_covariance)))
    assert rmse<.12 and covariance_error<.025, (rmse,covariance_error)
    # 同一数据的两个一维边缘无法表示相关性：量化独立乘积分布的误差。
    independent=.5*(xx**2/recovered_covariance[0,0]+yy**2/recovered_covariance[1,1])
    independent_error=independent[mask]-reference[mask];independent_rmse=float(np.std(independent_error))
    assert independent_rmse>3*rmse
    np.savez(output/'joint_fes.npz',fes=fes-offset,reference=reference,centers=centers,effective_samples=effective,covariance=recovered_covariance)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(11,3.4))
    for axis,data,title in zip(axes,(reference,fes-offset,independent),('Reference joint F','Frozen PB reweighted F','Independent marginals')):
        picture=axis.pcolormesh(edges[0],edges[1],np.ma.masked_where(~mask,data).T,cmap='viridis',vmin=0,vmax=4,shading='flat')
        axis.set(title=title,xlabel='CV1',ylabel='CV2',aspect='equal')
    fig.colorbar(picture,ax=axes,label='F / kBT',fraction=.025,pad=.03);fig.savefig(output/'joint_fes.png',dpi=160,bbox_inches='tight');plt.close(fig)
    write_result(output,'highdim',{'dimensions':dimensional,'periodic_seam_continuous':True,'full_covariance_gradient_max_error':full_error,'pb_gradient_max_error':pb_error,'restart_exact':True,'frozen_bias_samples':len(samples),'joint_fes_valid_bins':int(mask.sum()),'joint_fes_rmse_kbt':rmse,'covariance_max_error':covariance_error,'independent_marginals_rmse_kbt':independent_rmse,'reweighting_scope':'equilibrium samples under a frozen bias'})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);run(parser.parse_args().output)
