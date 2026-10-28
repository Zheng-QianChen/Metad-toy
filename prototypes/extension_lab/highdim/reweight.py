"""从固定偏置下的平衡样本重建二维联合自由能。

若 p_b(s) 正比于 exp(-(F(s)+V(s))/kbt)，每个样本乘 exp(+V(s)/kbt)
即可去除偏置。本入口要求所有样本对应同一冻结偏置与温度；持续沉积的
轨迹需要另行处理时间归一化及平衡条件。
"""
import numpy as np


def frozen_bias_fes(samples,bias_energies,edges,kbt=1.):
    """返回二维 FES、各 bin 的权重有效样本数及逐样本相对权重。
    
    samples[N,2]：同一时刻配对的两个 CV，配对关系用于保留相关性。
    bias_energies[N]：这些样本处的完整组合偏置能，而非某一维的单独偏置。
    edges=(edges_x,edges_y)：两个坐标轴的 bin 边界，允许宽度不同。
    
    fes[Bx,By] 以第一个轴对应 CV1、第二个轴对应 CV2；减去有限值中的最小值。
    零计数 bin 为 NaN。effective[Bx,By]=(sum w)^2/sum(w^2) 表示权重造成的
    有效样本量，时间自相关还需额外评估。weights[N] 为共同缩放后的相对权重。
    """
    samples=np.asarray(samples,dtype=float);bias_energies=np.asarray(bias_energies,dtype=float)
    if samples.ndim!=2 or samples.shape[1]!=2 or bias_energies.shape!=(len(samples),) or kbt<=0:
        raise ValueError('Expected paired 2D samples and bias energies')
    log_weights=bias_energies/kbt
    # 所有权重除以同一常数，保持概率比及 FES 差值，同时控制指数数值范围。
    weights=np.exp(log_weights-log_weights.max())
    histogram,_,_=np.histogram2d(samples[:,0],samples[:,1],bins=edges,weights=weights)
    # 同时统计 sum(w^2)，以估计每个二维 bin 中权重分布的不均匀程度。
    squares,_,_=np.histogram2d(samples[:,0],samples[:,1],bins=edges,weights=weights**2)
    # 加权计数是 bin 的概率质量；除以面积才得到用于 -kbt*log 的概率密度。
    areas=np.diff(edges[0])[:,None]*np.diff(edges[1])[None,:]
    visited=histogram>0
    fes=np.full(histogram.shape,np.nan)
    fes[visited]=-kbt*np.log(histogram[visited]/areas[visited])
    # 自由能的共同常数可任选；将最小的已采样 bin 设为 0 便于作图比较。
    fes-=np.nanmin(fes)
    effective=np.divide(histogram**2,squares,out=np.zeros_like(histogram),where=squares>0)
    return fes,effective,weights
