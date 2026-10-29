"""生成用于格式与导数检查的固定权重网络：3 个特征 → 5 个 tanh → 2 个输出。

三种格式使用同一组 FP64 参数。forward 和 VJP 一起导出，使运行时既能读取
模型输出，又能把任意上游 seeds 传回输入特征。网络参数是可复现的测试数据。
"""
import json
from pathlib import Path
import numpy as np


def export_all(directory):
    """在 directory 下生成 savedmodel、frozen_pb 和 onnx 三个模型目录。
    
    每个目录都有 manifest.json，保存维度、特征顺序、单位和文件/端点名称。
    函数返回这三个目录，验证器随后仅通过文件重新加载网络。
    TensorFlow 冻结步骤使用本环境版本的 convert_variables_to_constants_v2。
    """
    import tensorflow as tf
    import onnx
    from onnx import helper,numpy_helper,TensorProto
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(4815)
    # W1[3,5]、b1[5]、W2[5,2]、b2[2]；全部后端复用同一组参数。
    weights1=rng.normal(0.,.5,(3,5));bias1=rng.normal(0.,.1,(5,))
    weights2=rng.normal(0.,.4,(5,2));bias2=rng.normal(0.,.1,(2,))
    specification={'schema':1,'input_dimension':3,'output_dimension':2,'dtype':'float64','features':['r01','r12','r02'],'units':{'features':'angstrom','outputs':'dimensionless'}}
    class Model(tf.Module):
        """导出专用 TensorFlow 模块，显式提供前向和导数两个签名。
        """
        @tf.function(input_signature=[tf.TensorSpec([None,3],tf.float64,name='features')])
        def forward(self,features):
            """features[B,3] 经 X*W1+b1、tanh、H*W2+b2 得到 outputs[B,2]。
            """
            hidden=tf.math.tanh(tf.linalg.matmul(features,weights1)+bias1)
            return {'outputs':tf.linalg.matmul(hidden,weights2)+bias2}
        @tf.function(input_signature=[tf.TensorSpec([None,3],tf.float64,name='features'),tf.TensorSpec([None,2],tf.float64,name='seeds')])
        def vjp(self,features,seeds):
            """以 seeds[B,2] 为输出 cotangent，自动微分得到输入梯度 [B,3]。
            
            GradientTape 观察输入 features；output_gradients 让两路输出按用户给定
            seeds 合成同一个标量目标的导数。返回 feature_gradients 的命名签名。
            """
            with tf.GradientTape() as tape:
                tape.watch(features)
                outputs=self.forward(features)['outputs']
            return {'feature_gradients':tape.gradient(outputs,features,output_gradients=seeds)}
    model=Model()
    saved=directory/'savedmodel';saved.mkdir(exist_ok=True)
    tf.saved_model.save(model,str(saved/'model'),signatures={'forward':model.forward,'vjp':model.vjp})
    (saved/'manifest.json').write_text(json.dumps({**specification,'backend':'savedmodel','model':'model'},indent=2))
    # 冻结图将模型参数固化为常量；记录真实 tensor 名，供加载器恢复调用端点。
    frozen=directory/'frozen_pb';frozen.mkdir(exist_ok=True)
    endpoints={}
    for name,function in [('forward',model.forward),('vjp',model.vjp)]:
        concrete=convert_variables_to_constants_v2(function.get_concrete_function())
        (frozen/(name+'.pb')).write_bytes(concrete.graph.as_graph_def().SerializeToString())
        endpoints[name]={'file':name+'.pb','inputs':[tensor.name for tensor in concrete.inputs],'output':concrete.outputs[0].name}
    (frozen/'manifest.json').write_text(json.dumps({**specification,'backend':'frozen_pb','endpoints':endpoints},indent=2))
    exported=directory/'onnx';exported.mkdir(exist_ok=True)
    # ONNX initializer 保存权重；VJP 同时使用 W1^T、W2^T 和导数公式中的常数 1。
    constants=[numpy_helper.from_array(value,name) for name,value in [('W1',weights1),('b1',bias1),('W2',weights2),('b2',bias2),('W1T',weights1.T.copy()),('W2T',weights2.T.copy()),('one',np.array(1.,dtype=np.float64))]]
    forward_nodes=[helper.make_node('MatMul',['features','W1'],['linear1']),helper.make_node('Add',['linear1','b1'],['preactivation']),helper.make_node('Tanh',['preactivation'],['hidden']),helper.make_node('MatMul',['hidden','W2'],['linear2']),helper.make_node('Add',['linear2','b2'],['outputs'])]
    inputs=[helper.make_tensor_value_info('features',TensorProto.DOUBLE,['batch',3])]
    def save_graph(nodes,graph_inputs,output_name,output_dimension,filename,initializers):
        """把给定算子和常量写成 ONNX 模型，并先检查图结构。
        
        输入 batch 轴为动态维度；输出维度由 output_dimension 指定。
        固定 opset=17、IR=9 与本次验证的 ONNX Runtime 版本配合。
        """
        graph=helper.make_graph(nodes,filename,graph_inputs,[helper.make_tensor_value_info(output_name,TensorProto.DOUBLE,['batch',output_dimension])],initializers)
        proto=helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)],ir_version=9)
        onnx.checker.check_model(proto);onnx.save(proto,exported/filename)
    save_graph(forward_nodes,inputs,'outputs',2,'forward.onnx',constants[:4])
    # 显式 VJP：Gx=((seeds*W2^T) 逐元素乘 (1-hidden^2))*W1^T。
    # 先重算 hidden，再应用 tanh 导数；这张 ONNX 图与 forward 一起构成导数契约。
    reverse_nodes=forward_nodes[:3]+[helper.make_node('MatMul',['seeds','W2T'],['hidden_seed']),helper.make_node('Mul',['hidden','hidden'],['hidden_squared']),helper.make_node('Sub',['one','hidden_squared'],['tanh_derivative']),helper.make_node('Mul',['hidden_seed','tanh_derivative'],['linear_seed']),helper.make_node('MatMul',['linear_seed','W1T'],['feature_gradients'])]
    save_graph(reverse_nodes,inputs+[helper.make_tensor_value_info('seeds',TensorProto.DOUBLE,['batch',2])],'feature_gradients',3,'vjp.onnx',[constants[i] for i in (0,1,4,5,6)])
    (exported/'manifest.json').write_text(json.dumps({**specification,'backend':'onnx','forward':'forward.onnx','vjp':'vjp.onnx'},indent=2))
    return [saved,frozen,exported]
