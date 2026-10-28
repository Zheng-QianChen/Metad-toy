"""高维偏置的参考表示：hill 列表、平滑周期核和 PBMetaD。

所有 evaluate 方法都返回偏置能 V 与 CV 梯度 dV/ds。对角 Gaussian 的
每个 hill 保存 D 维中心和一个高度，数值存储量随 H*D 增长。
当前使用 NumPy 在 Host 求值，便于逐项核对公式和未来的设备实现。
"""
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp


class HillBias:
    """共享核形状的一组多维 Gaussian 或周期 hill。
    
    widths[D] 保存各方向 sigma，periods[D] 保存周期长度；periods[j]=0 表示
    普通实数坐标。centers[H][D] 和 heights[H] 按追加顺序保存 hill。
    
    covariance[D,D] 可指定一个对称正定协方差，所有 hill 共享此矩阵。
    此时核宽度由 covariance 完整决定，widths 仅用于维度与参数结构；
    precision 是其逆矩阵。完整协方差模式要求全部坐标为非周期坐标。
    
    chunk_size 规定每次处理的 hill 数，限制临时的 [...,chunk,D] 位移数组。
    批量样本轴由调用方控制，大轨迹可在调用方再按样本分批。
    """
    def __init__(self, widths, periods=None, covariance=None, chunk_size=64):
        """验证核参数，预计算协方差逆矩阵并建立空的 hill 历史。
        
        widths 和 periods 使用对应 CV 的单位，covariance[i,j] 使用两分量单位的
        乘积。Cholesky 分解用于验证正定性，保证高斯二次型具有有效的宽度。
        """
        self.widths=np.atleast_1d(np.asarray(widths,dtype=float))
        if not np.all(np.isfinite(self.widths)) or not np.all(self.widths>0):
            raise ValueError('Widths must be finite and positive')
        self.dimension=len(self.widths)
        # periods=0 表示非周期分量；其余分量用平滑周期核。
        self.periods=np.zeros(self.dimension) if periods is None else np.asarray(periods,dtype=float)
        if self.periods.shape!=self.widths.shape or not np.all(np.isfinite(self.periods)) or np.any(self.periods<0):
            raise ValueError('Invalid periods')
        self.covariance=None if covariance is None else np.asarray(covariance,dtype=float)
        self.precision=None
        if self.covariance is not None:
            if np.any(self.periods): raise ValueError('Full covariance currently uses nonperiodic coordinates')
            if self.covariance.shape!=(self.dimension,self.dimension) or not np.all(np.isfinite(self.covariance)) or not np.allclose(self.covariance,self.covariance.T):
                raise ValueError('Covariance must be a finite symmetric matrix')
            np.linalg.cholesky(self.covariance)
            self.precision=np.linalg.inv(self.covariance)
        if chunk_size<1: raise ValueError('Positive chunk size required')
        self.chunk_size=int(chunk_size)
        self.centers=[];self.heights=[]

    def add(self, center, height):
        """追加中心 center[D] 和能量高度 height，复制中心数值到历史列表。
        
        HillBias 也可表示有正有负的核展开，因此允许有限的负高度。
        PBMetaD 的沉积器在上层根据自己的规则生成正高度。
        """
        center=np.asarray(center,dtype=float)
        if center.shape!=self.widths.shape or not np.all(np.isfinite(center)) or not np.isfinite(height):
            raise ValueError('Invalid hill')
        self.centers.append(center.tolist());self.heights.append(float(height))

    def evaluate(self, coordinates):
        """在 coordinates[...,D] 上求能量和各分量梯度。
        
        返回 energy[...]、gradient[...,D]；单点输入 [D] 得到 0 维能量数组。
        每个 hill 定义为 h*exp(-q/2)。derivative 始终保存 (1/2)*dq/ds，
        因此最终梯度统一使用 -hill_value*derivative。
        
        非周期对角核 q=sum_j(delta_j/sigma_j)²；完整协方差核为
        q=delta^T*precision*delta。周期分量用余弦核替换对应的二次项，
        从而在一个周期的两端保持值和一阶导数连续。
        """
        coordinates=np.asarray(coordinates,dtype=float)
        if coordinates.shape[-1:]!=self.widths.shape: raise ValueError('Last axis must match CV dimension')
        energy=np.zeros(coordinates.shape[:-1]);gradient=np.zeros_like(coordinates)
        periodic=self.periods>0
        for begin in range(0,len(self.centers),self.chunk_size):
            centers=np.asarray(self.centers[begin:begin+self.chunk_size])
            # 插入一个 hill 轴：单点时为 [chunk,D]，批量时为 [...,chunk,D]。
            delta=coordinates[...,None,:]-centers
            if self.precision is not None:
                derivative=delta@self.precision
                quadratic=np.sum(delta*derivative,axis=-1)
            else:
                terms=(delta/self.widths)**2
                derivative=delta/self.widths**2
                if np.any(periodic):
                    # omega=2*pi/L，q_j=2*(1-cos(omega*delta))/(omega*sigma)^2。
                    # 小位移时展开为 delta^2/sigma^2；有限位移采用这一平滑周期核定义。
                    omega=2*np.pi/self.periods[periodic]
                    angle=delta[...,periodic]*omega
                    # 近原点退化为 Gaussian；周期边界的一阶导数连续。
                    scale=(omega*self.widths[periodic])**2
                    terms[...,periodic]=2*(1-np.cos(angle))/scale
                    derivative[...,periodic]=np.sin(angle)*omega/scale
                quadratic=terms.sum(axis=-1)
            contributions=np.asarray(self.heights[begin:begin+self.chunk_size])*np.exp(-.5*quadratic)
            energy+=contributions.sum(axis=-1)
            # 倒数第二轴是 hill 轴；只累加该轴，保留样本轴与 CV 分量轴。
            gradient-=np.sum(contributions[...,None]*derivative,axis=-2)
        return energy,gradient

    def save(self,path):
        """保存核参数、中心和高度；chunk_size 属于求值配置，可在重载后另设。
        """
        Path(path).write_text(json.dumps({'schema':1,'widths':self.widths.tolist(),'periods':self.periods.tolist(),'covariance':None if self.covariance is None else self.covariance.tolist(),'centers':self.centers,'heights':self.heights}))

    @classmethod
    def load(cls,path):
        """从 schema=1 的历史文件恢复核和 hills，并重新验证每个 hill。
        """
        state=json.loads(Path(path).read_text())
        if state['schema']!=1: raise ValueError('Unsupported restart version')
        bias=cls(state['widths'],state['periods'],state['covariance'])
        for center,height in zip(state['centers'],state['heights']): bias.add(center,height)
        return bias


class ParallelBiasMetaD:
    """以 D 个一维偏置组成一个 D 维 PBMetaD 偏置。
    
    biases[j] 持有第 j 个 CV 的一维 HillBias，记其值为 V_j(s_j)。组合形式为
    V_PB=-kbt*log((1/D)*sum_j exp(-V_j/kbt))。
    它比采用 sum 的常见定义多一个 kbt*log(D) 的常数，因此空历史时为 0。
    
    w_j=exp(-V_j/kbt)/sum_k exp(-V_k/kbt) 是当前点的归一化权重。
    力回传使用 dV_PB/ds_j=w_j*dV_j/ds_j；沉积高度还要乘 well-tempered
    衰减因子。w_j 较大只表示当前偏置值较低，不能单独判断原子力大小。
    """
    def __init__(self,widths,kbt=1.,hill_height=.1,bias_factor=10.,pace=10):
        """创建各维一维偏置和沉积时钟。
        
        widths[D] 为各维 sigma；kbt、hill_height 是能量；bias_factor 是 gamma；
        pace 是正整数沉积间隔。last_step 初始为 -1，使 step=0 可产生第一批 hill。
        """
        self.biases=[HillBias([width]) for width in widths]
        if not self.biases or kbt<=0 or hill_height<=0 or bias_factor<=1 or pace<1:
            raise ValueError('Invalid PBMetaD parameters')
        self.kbt=float(kbt);self.hill_height=float(hill_height)
        self.bias_factor=float(bias_factor);self.pace=int(pace);self.last_step=-1

    def components(self,coordinates):
        """返回每个一维偏置的值与自身导数。
        
        coordinates[...,D] 的第 j 列交给 biases[j]；两项返回值形状均为 [...,D]。
        这里保留单维长度 1 的轴，以满足 HillBias 的 [...,dimension] 输入约定。
        """
        coordinates=np.asarray(coordinates,dtype=float)
        if coordinates.shape[-1:]!=(len(self.biases),): raise ValueError('Wrong PBMetaD dimension')
        pairs=[bias.evaluate(coordinates[...,index:index+1]) for index,bias in enumerate(self.biases)]
        return np.stack([pair[0] for pair in pairs],axis=-1),np.concatenate([pair[1] for pair in pairs],axis=-1)

    def evaluate(self,coordinates):
        """组合各维偏置，并返回用于图反向传播的 D 分量梯度。
        
        normalizer=log(sum_j exp(-V_j/kbt)) 通过 logsumexp 稳定计算。
        对整个组合公式求导后恰好得到 w_j*dV_j/ds_j；权重随坐标的依赖已经
        包含在这一解析结果里。输入与返回值的批量布局遵循 HillBias.evaluate。
        """
        values,derivatives=self.components(coordinates)
        log_weights=-values/self.kbt
        normalizer=logsumexp(log_weights,axis=-1)
        # 加 kBT*log(M) 仅固定常数，使初始偏置为 0，力和接受率保持相同。
        energy=-self.kbt*(normalizer-np.log(len(self.biases)))
        gradient=np.exp(log_weights-normalizer[...,None])*derivatives
        return energy,gradient

    def update(self,step,coordinates):
        """在当前单点 coordinates[D] 处同步沉积各维 hill。
        
        第 j 维高度为 hill_height*w_j*exp(-V_j/[kbt*(gamma-1)])。
        w_j 和 tempering 均使用本轮沉积前的所有 V_j，确保结果与遍历顺序无关。
        返回值与 last_step 语义同 WTMetaD；同一步提交多次只处理第一次。
        """
        coordinates=np.asarray(coordinates,dtype=float)
        if coordinates.shape!=(len(self.biases),): raise ValueError('One CV vector required')
        if step<self.last_step: raise ValueError('Time moved backwards')
        if step==self.last_step: return False
        values,_=self.components(coordinates)
        self.last_step=int(step)
        if step%self.pace: return False
        log_weights=-values/self.kbt
        # conditional 的各分量总和为 1；它也就是 evaluate 中分配梯度的权重。
        conditional=np.exp(log_weights-logsumexp(log_weights))
        # tempering 按各自的一维偏置高度衰减，能量分母为 k_B*DeltaT。
        tempering=np.exp(-values/(self.kbt*(self.bias_factor-1)))
        # 先求全部旧偏置，再同时沉积，保证各维的分母取自同一时刻。
        for index,bias in enumerate(self.biases):
            bias.add([coordinates[index]],self.hill_height*conditional[index]*tempering[index])
        return True

    def save(self,path):
        """保存各维 hills、沉积参数和最近步号，便于恢复后继续同一偏置历史。
        """
        Path(path).write_text(json.dumps({'schema':1,'widths':[float(bias.widths[0]) for bias in self.biases],'kbt':self.kbt,'hill_height':self.hill_height,'bias_factor':self.bias_factor,'pace':self.pace,'last_step':self.last_step,'hills':[(bias.centers,bias.heights) for bias in self.biases]}))

    @classmethod
    def load(cls,path):
        """恢复一维偏置列表及 last_step；各维依照文件顺序对应原来的 CV 分量。
        """
        state=json.loads(Path(path).read_text())
        if state.pop('schema')!=1: raise ValueError('Unsupported restart version')
        hills=state.pop('hills');last_step=state.pop('last_step')
        bias=cls(**state);bias.last_step=last_step
        for component,(centers,heights) in zip(bias.biases,hills):
            for center,height in zip(centers,heights):component.add(center,height)
        return bias
