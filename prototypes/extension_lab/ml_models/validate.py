"""文件模型和坐标导数链的完整验证入口。

先导出并重新加载三种格式，再检查动态 batch、多输出 seeds、零 seeds 和非法
输入。随后分别以坐标扰动和应变扰动检查力与 virial，最后检查 XYZ→CSV 流程。
解析/自动微分结果与独立能量中心差分对照，避免只比较两个实现的输出值。
"""
import argparse,os
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.checks import finite_difference,write_result
from ml_models.export_demo import export_all
from ml_models.runtime import ModelNode
from ml_models.trajectory import evaluate_frame,analyze


def run(output):
    """生成测试模型、数值检查指标和 80 帧示例轨迹。
    
    TensorFlow 按需增长显存；各模型使用同一批随机特征及固定权重进行比较。
    所有差分使用 FP64 和小的绝对扰动，力/virial 的容差针对本示例的数值尺度。
    结果同时记录实际执行设备，以区分格式支持和设备执行情况。
    """
    import tensorflow as tf
    for device in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(device,True)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    directories=export_all(output/'models')
    rng=np.random.default_rng(43454)
    models=[ModelNode(directory) for directory in directories]
    errors={model.backend:0. for model in models}
    for batch in (1,7,64):
        features=rng.normal(size=(batch,3));seeds=rng.normal(size=(batch,2))
        baseline=models[0].forward(features);baseline_vjp=models[0].vjp(features,seeds)
        for model in models:
            np.testing.assert_allclose(model.forward(features),baseline,rtol=1e-12,atol=1e-13)
            np.testing.assert_allclose(model.vjp(features,seeds),baseline_vjp,rtol=1e-12,atol=1e-13)
            np.testing.assert_allclose(model.vjp(features,np.zeros_like(seeds)),0.,atol=1e-14)
            # 不同批量、不同输出 seed 的方向导数验证，避免仅验证某一个固定输出。
            direction=rng.normal(size=features.shape);step=1e-5
            # 比较 <seeds, J*direction> 与 <J^T*seeds, direction>，
            # 左侧由前向能量差分得到，右侧由模型 VJP 给出。
            numeric=np.sum((model.forward(features+step*direction)-model.forward(features-step*direction))*seeds)/(2*step)
            analytic=np.sum(model.vjp(features,seeds)*direction)
            error=abs(float(numeric-analytic));errors[model.backend]=max(errors[model.backend],error)
            assert error<1e-7
            # 分别触发 dtype、特征数量和非有限值检查，核对模型契约的报错边界。
            invalid=0
            for wrong in (features.astype(np.float32),features[:,:2],np.full_like(features,np.nan)):
                try:model.forward(wrong)
                except ValueError:invalid+=1
            assert invalid==3
    # 逐特征中心差分，验证完整的 feature adjoint。
    features=rng.normal(size=(2,3));seeds=rng.normal(size=(2,2))
    for model in models:
        numerical=finite_difference(lambda x:float(np.sum(model.forward(x)*seeds)),features)
        np.testing.assert_allclose(model.vjp(features,seeds),numerical,atol=1e-8,rtol=1e-7)
    coordinates=np.array([[[.1,.2,.3],[1.2,.1,.2],[.2,1.1,.4]]],dtype=np.float64)
    physical=[]
    for model in models:
        energy,forces,virial,outputs=evaluate_frame(model,coordinates)
        # 从最终标量偏置扰动每个原子坐标，覆盖距离描述符、模型和顶部偏置全链。
        numerical=-finite_difference(lambda x:float(evaluate_frame(model,x)[0].sum()),coordinates)
        force_error=float(np.max(np.abs(forces-numerical)));assert force_error<1e-8
        np.testing.assert_allclose(forces.sum(axis=1),0.,atol=1e-14)
        def deformed_energy(strain):
            """用一般小应变变换坐标，返回该变形后的偏置能。
            
            坐标存成行向量，因此 r'=(I+strain)*r 在代码中写为 coordinates@(I+strain).T。
            对 strain[a,b] 的负导数应等于 W_ab，用于逐项核对完整 3x3 virial。
            """
            transformed=coordinates@ (np.eye(3)+strain).T
            return float(evaluate_frame(model,transformed)[0].sum())
        numerical_virial=-finite_difference(deformed_energy,np.zeros((3,3)))
        virial_error=float(np.max(np.abs(virial[0]-numerical_virial)));assert virial_error<1e-8
        physical.append({'backend':model.backend,'force_max_error':force_error,'virial_max_error':virial_error,'device':model.device})
    # 构造平滑变化的短轨迹，写盘后重新读取；CSV 值应与原始内存批次一致。
    trajectory=output/'demo.xyz'
    frames=np.repeat(coordinates,80,axis=0)
    for frame in range(len(frames)):
        frames[frame,1,0]+=.08*np.sin(frame*.13)
        frames[frame,2,1]+=.05*np.cos(frame*.13)
    with trajectory.open('w') as handle:
        for index,frame in enumerate(frames):
            handle.write(f'3\nframe {index}\n')
            for atom,xyz in enumerate(frame):handle.write(('O' if atom==0 else 'H')+' '+' '.join(f'{value:.15g}' for value in xyz)+'\n')
    summary=analyze(models[-1],trajectory,output/'trajectory.csv')
    table=np.loadtxt(output/'trajectory.csv',delimiter=',',skiprows=1)
    np.testing.assert_allclose(table[:,1:3],evaluate_frame(models[-1],frames)[3],atol=1e-12)
    write_result(output,'ml_models',{'tensorflow':tf.__version__,'onnxruntime':__import__('onnxruntime').__version__,'batches':[1,7,64],'vjp_max_directional_errors':errors,'coordinate_chain':physical,'trajectory':summary,'contract':'float64 batch feature matrix and explicit multi-output VJP'})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);run(parser.parse_args().output)
