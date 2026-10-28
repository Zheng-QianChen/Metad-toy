"""三原子 XYZ 轨迹的模型 CV 与偏置力分析示例。

数据流为 coordinates → 三对距离 features → 模型 outputs → 标量偏置。
反向流为 dV/doutputs → 模型 VJP → dV/ddistance → 坐标力和 virial。
坐标按 Å 输入，本示例使用非周期的直接位移；模型输出约定为无量纲。
"""
from pathlib import Path
import numpy as np

# 这个示例的特征契约由 manifest 声明：r01、r12、r02，单位 Å。
PAIRS=((0,1),(1,2),(0,2))


def distances(coordinates):
    """从 coordinates[B,3,3] 构造三对距离及其方向。
    
    轴顺序是 [帧/样本,原子,xyz]；原子对顺序由 PAIRS 固定为 (0,1),(1,2),(0,2)。
    返回 values[B,3] 和 directions[B,3,3]，后者的第二个轴为原子对，第三个轴
    为 xyz。方向定义为 (r_left-r_right)/distance，也是距离对左原子坐标的导数。
    距离对右原子的导数取相反号；重合原子的导数没有确定方向，直接报错。
    """
    coordinates=np.asarray(coordinates,dtype=float)
    if coordinates.ndim!=3 or coordinates.shape[1:]!=(3,3):raise ValueError('Demo descriptor expects [batch,3 atoms,3 coordinates]')
    delta=np.stack([coordinates[:,left]-coordinates[:,right] for left,right in PAIRS],axis=1)
    values=np.linalg.norm(delta,axis=-1)
    if np.any(values<1e-12):raise ValueError('Overlapping atoms give an undefined distance derivative')
    return values,delta/values[...,None]


def evaluate_frame(model,coordinates):
    """执行一个批次的描述符、模型、标量偏置和坐标反向链。
    
    model 须满足三距离输入、两个无量纲输出的示例契约。
    返回 energies[B]、forces[B,3,3]、virial[B,3,3]、outputs[B,2]。
    forces 的第二轴为原子，virial 的两个尾轴均为空间方向。
    偏置能采用示例能量单位，力相应为能量/Å；应变为无量纲，virial 单位为能量。
    """
    if model.specification.get('features')!=['r01','r12','r02']:
        raise ValueError('Model feature order differs from this descriptor')
    if model.specification.get('units',{}).get('features')!='angstrom':
        raise ValueError('Demo coordinates use angstrom')
    features,directions=distances(coordinates)
    outputs=model.forward(features)
    # 两个 ML 输出共同参与一个标量偏置，演示顶部给出的 adjoint seed。
    energies=.7*outputs[:,0]*outputs[:,1]+np.exp(.1*outputs[:,0])
    # seeds[:,j]=dV/doutputs[:,j]，是传给模型 VJP 的两个顶部导数。
    seeds=np.stack((.7*outputs[:,1]+.1*np.exp(.1*outputs[:,0]),.7*outputs[:,0]),axis=1)
    feature_gradients=model.vjp(features,seeds)
    forces=np.zeros_like(coordinates)
    for index,(left,right) in enumerate(PAIRS):
        # 标量距离导数乘空间方向得到坐标梯度，再取负号形成力。
        # 一对原子得到相反的贡献，平移不变性对应总力为零。
        contribution=-feature_gradients[:,index,None]*directions[:,index]
        forces[:,left]+=contribution;forces[:,right]-=contribution
    # W_ab = sum_i F_ia r_ib，与 r'_a = r_a + epsilon_ab*r_b 的负能量导数相同。
    # 保留完整 3x3 张量以逐项做应变差分；接入引擎时再按其六分量约定排列。
    virial=np.einsum('bni,bnj->bij',forces,coordinates)
    return energies,forces,virial,outputs


def read_xyz(path):
    """按帧读取标准的 element x y z 文本记录，返回 [帧数,3,3] 数组。
    
    每帧依次包含原子数、注释行和三个原子记录。原子身份依赖固定的行顺序，
    元素序列检查可发现部分顺序变化；相同元素之间的交换需要调用方保证身份。
    当前把整条轨迹载入内存，适合验证短轨迹和描述符契约。
    """
    frames=[];elements=None
    with Path(path).open() as handle:
        while True:
            line=handle.readline()
            if not line:break
            count=int(line)
            if count!=3:raise ValueError('This descriptor requires three atoms')
            if not handle.readline():raise ValueError('Truncated XYZ comment')
            frame=[];names=[]
            for _ in range(count):
                fields=handle.readline().split()
                if len(fields)!=4:raise ValueError('Expected element x y z')
                names.append(fields[0]);frame.append([float(x) for x in fields[1:]])
            if elements is not None and names!=elements:raise ValueError('Atom order changed between frames')
            elements=names;frames.append(frame)
    if not frames:raise ValueError('Empty trajectory')
    return np.asarray(frames)


def analyze(model,input_path,output_path):
    """离线求值并保存逐帧结果，返回简要统计字典。
    
    CSV 列依次为帧号、两个模型 CV、偏置能以及九个力分量。力按原子顺序展开，
    每个原子内部为 x/y/z。统计包含 CV 均值、样本协方差、lag-1 自相关和最大总力。
    至少两帧才有有意义的样本协方差；自相关的 lag 单位是保存的帧间隔。
    """
    coordinates=read_xyz(input_path)
    energies,forces,virial,outputs=evaluate_frame(model,coordinates)
    table=np.column_stack((np.arange(len(coordinates)),outputs,energies,forces.reshape(len(coordinates),-1)))
    np.savetxt(output_path,table,delimiter=',',header='frame,ml_cv0,ml_cv1,bias_energy,'+','.join(f'force_{atom}_{axis}' for atom in range(3) for axis in 'xyz'),comments='')
    centered=outputs-outputs.mean(axis=0)
    # 对去均值序列计算相邻帧乘积，并以整个序列的平方和归一化。
    # 这是 lag=1 的相关指标；完整相关时间需要继续计算更长的 lag。
    lag_one=np.sum(centered[:-1]*centered[1:],axis=0)/np.maximum(np.sum(centered**2,axis=0),1e-30)
    return {'frames':len(coordinates),'cv_mean':outputs.mean(axis=0).tolist(),'cv_covariance':np.cov(outputs,rowvar=False).tolist(),'lag_one_autocorrelation':lag_one.tolist(),'maximum_total_force':float(np.max(np.abs(forces.sum(axis=1))))}

if __name__=='__main__':
    import argparse,json
    from runtime import ModelNode
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);parser.add_argument('--xyz',required=True);parser.add_argument('--csv',required=True)
    args=parser.parse_args();print(json.dumps(analyze(ModelNode(args.model),args.xyz,args.csv),indent=2))
