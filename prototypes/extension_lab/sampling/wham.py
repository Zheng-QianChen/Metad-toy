"""同温度、等宽 histogram 的 WHAM 自洽重建。

各窗口可以有不同 umbrella 偏置和样本数。输出的是各 bin 的相对自由能；
等宽 bin 的宽度只贡献同一个加性常数，因此可在最后统一平移到最小值 0。
"""
import numpy as np
from scipy.special import logsumexp


def reconstruct(histograms, bias_energies, kbt=1., tolerance=1e-10, max_iterations=100000):
    """由窗口计数求无偏分布，并返回 (fes, iterations)。
    
    histograms[W,B] 是 W 个窗口在 B 个共同 bin 中的计数。
    bias_energies[W,B] 是各窗口在各 bin 中心的偏置能，单位与 kbt 相同。
    调用方负责提供足够的平衡采样和窗口重叠；每个窗口至少有一个样本。
    
    offsets[i]=f_i 是无量纲窗口自由能偏移。自洽关系为：
    p_b = sum_i H_ib / sum_i N_i*exp(f_i-V_ib/kbt)，
    f_i = -log(sum_b p_b*exp(-V_ib/kbt))。
    
    结果 fes[B] 的单位为能量，未访问 bin 保持 NaN。tolerance 控制相邻迭代
    中 f_i 的最大变化，max_iterations 限制迭代次数。这里的计数权重未作
    时间自相关修正；误差分析应在调用方按有效样本量或分块方法另行评估。
    """
    histograms = np.asarray(histograms, dtype=float)
    bias_energies = np.asarray(bias_energies, dtype=float)
    if histograms.shape != bias_energies.shape or np.any(histograms<0) or kbt<=0:
        raise ValueError('Invalid histogram, bias shape, or temperature')
    totals = histograms.sum(axis=1)
    if np.any(totals == 0):
        raise ValueError('Every window needs samples')
    # 只在有总计数的 bin 上取 log，保留空 bin 的未知状态。
    visited = histograms.sum(axis=0)>0
    log_counts = np.log(histograms[:,visited].sum(axis=0))
    reduced_bias = bias_energies[:,visited]/kbt
    # totals[i]=N_i，reduced_bias[i,b]=V_ib/kbt，均按窗口轴组织。
    offsets = np.zeros(len(totals))
    for iteration in range(max_iterations):
        # logsumexp 在对数域累加，避免窗口偏置相差很大时指数溢出。
        log_probability = log_counts - logsumexp(np.log(totals)[:,None]+offsets[:,None]-reduced_bias,axis=0)
        log_probability -= logsumexp(log_probability)
        next_offsets = -logsumexp(log_probability[None,:]-reduced_bias,axis=1)
        # f_i 只确定到共同常数；固定第一个窗口为 0，便于比较迭代收敛。
        next_offsets -= next_offsets[0]
        if np.max(np.abs(next_offsets-offsets)) < tolerance:
            break
        offsets = next_offsets
    else:
        raise RuntimeError('WHAM did not converge; check window overlap')
    fes = np.full(histograms.shape[1], np.nan)
    fes[visited] = -kbt*log_probability
    fes -= np.nanmin(fes)
    return fes, iteration+1
