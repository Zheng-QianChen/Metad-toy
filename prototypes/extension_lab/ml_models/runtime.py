"""文件模型的统一 forward / VJP 运行入口。

设 X[B,F] 为 B 个样本的 F 维特征，Y[B,O] 为模型的 O 个输出。
VJP 是 vector-Jacobian product：上游给出 S=dV/dY，返回 G=dV/dX，
本例各样本独立，逐样本满足 G_f=sum_o S_o*(dY_o/dX_f)。
模型文件须提供对应的 VJP 图；跨样本耦合需要按完整批量的 Jacobian 理解。
"""
import json
from pathlib import Path
import numpy as np


class ModelNode:
    """持有一个 manifest 和后端执行对象，表示图中的模型节点。
    
    specification 保存 schema、维度、dtype、文件名及描述符约定；functions
    保存可重复调用的 forward / vjp。TensorFlow SavedModel 对象额外保存在
    model 中，以维持其变量和资源的生命周期。
    
    device 是诊断信息：ONNX 记录 providers，TensorFlow 记录最近一次输出
    tensor 的设备。输入和返回值均为 Host NumPy 数组，TF 调用中的张量转换
    与 numpy() 构成当前原型的数据传输边界。
    """
    def __init__(self,directory):
        """读取 directory/manifest.json，加载 schema=1、FP64 的模型契约。
        
        ONNX 为 forward 和 vjp 分别创建 CPU Session。SavedModel 从命名签名取得
        两个函数；冻结 PB 则按 manifest 给出的 tensor 名截取可调用子图。
        维度检查负责输入输出形状，特征顺序与物理单位由描述符调用层核对。
        """
        self.directory=Path(directory)
        self.specification=json.loads((self.directory/'manifest.json').read_text())
        spec=self.specification
        if spec.get('schema')!=1 or spec.get('dtype')!='float64':
            raise ValueError('Expected schema 1, float64 model')
        self.input_dimension=spec['input_dimension'];self.output_dimension=spec['output_dimension']
        if self.input_dimension<1 or self.output_dimension<1:raise ValueError('Invalid model dimensions')
        self.backend=spec['backend'];self.device='unknown'
        if self.backend=='onnx':
            import onnxruntime as ort
            options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
            self.functions={name:ort.InferenceSession(str(self.directory/spec[name]),sess_options=options,providers=['CPUExecutionProvider']) for name in ('forward','vjp')}
            self.device=str(self.functions['forward'].get_providers())
        elif self.backend in ('savedmodel','frozen_pb'):
            import tensorflow as tf
            self.tf=tf
            if self.backend=='savedmodel':
                self.model=tf.saved_model.load(str(self.directory/spec['model']))
                self.functions={name:self.model.signatures[name] for name in ('forward','vjp')}
            else:
                self.functions={}
                for name,endpoint in spec['endpoints'].items():
                    definition=tf.compat.v1.GraphDef()
                    definition.ParseFromString((self.directory/endpoint['file']).read_bytes())
                    imported=tf.compat.v1.wrap_function(lambda:tf.compat.v1.import_graph_def(definition,name=''),[])
                    graph=imported.graph
                    # prune 以指定输入/输出 tensor 建立函数，加载时无需重新描述网络运算。
                    self.functions[name]=imported.prune([graph.get_tensor_by_name(tensor) for tensor in endpoint['inputs']],graph.get_tensor_by_name(endpoint['output']))
        else: raise ValueError('Unknown model backend')

    def _validate(self,value,dimension):
        """检查 Host 输入为有限的 float64 二维数组 [batch,dimension]。
        
        这里保持精度契约显式化；例如 float32 数据由调用方决定是否转换为 float64。
        """
        value=np.asarray(value)
        if value.dtype!=np.float64 or value.ndim!=2 or value.shape[1]!=dimension or not np.all(np.isfinite(value)):
            raise ValueError(f'Expected finite float64 [batch,{dimension}] tensor')
        return value

    def _invoke(self,name,features,seeds=None):
        """将规范输入交给指定后端，并统一为 NumPy 输出。
        
        name 为 forward 或 vjp；后者额外接收 seeds[B,O]。SavedModel 使用关键字
        参数和命名字典输出；冻结 PB 使用导出时记下的 tensor 顺序。
        输出还需满足 [B,O] 或 [B,F] 的形状及数值有限性约束。
        """
        arguments={'features':features}
        if seeds is not None:arguments['seeds']=seeds
        if self.backend=='onnx':
            result=self.functions[name].run(None,arguments)[0]
        else:
            arguments={key:self.tf.convert_to_tensor(value,dtype=self.tf.float64) for key,value in arguments.items()}
            if self.backend=='savedmodel':
                key='outputs' if name=='forward' else 'feature_gradients'
                tensor=self.functions[name](**arguments)[key]
            else:
                # GraphDef 的输入顺序由导出时记录的 tensor 名确定。
                names=self.specification['endpoints'][name]['inputs']
                # 本导出契约使用 features:0、seeds:0；去掉输出端口编号后映射逻辑参数。
                tensors=[arguments[tensor.split(':')[0]] for tensor in names]
                tensor=self.functions[name](*tensors)
            self.device=tensor.device
            # numpy() 把结果交回 Host，并等待本次结果就绪；跨设备零拷贝需要另一接口。
            result=tensor.numpy()
        dimension=self.output_dimension if name=='forward' else self.input_dimension
        if result.shape!=(len(features),dimension) or not np.all(np.isfinite(result)):
            raise ValueError('Model produced invalid output shape or nonfinite values')
        return result

    def forward(self,features):
        """输入 features[B,F]，返回模型输出 [B,O]。B 是样本批量，F/O 来自 manifest。
        """
        return self._invoke('forward',self._validate(features,self.input_dimension))

    def vjp(self,features,seeds):
        """输入 features[B,F] 和 seeds[B,O]，返回 feature_gradients[B,F]。
        
        seeds 对应同一批样本在模型输出处的能量导数。此结果继续交给描述符反向
        传播，即可把模型导数传到原子坐标；最后取负得到力。
        """
        features=self._validate(features,self.input_dimension);seeds=self._validate(seeds,self.output_dimension)
        if len(features)!=len(seeds):raise ValueError('Feature and seed batch sizes differ')
        return self._invoke('vjp',features,seeds)
