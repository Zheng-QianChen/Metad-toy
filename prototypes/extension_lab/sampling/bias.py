"""采样方法的最小契约与同温度窗口交换规则。

记 s 为 D 维 CV，V(s) 为偏置能。evaluate 返回 V 与 dV/ds，调用方将后者
作为 Graph.backward 的 seeds，再由坐标链式求导获得力 F=-dV/dr。
update 单独负责历史状态推进，便于引擎明确每个 timestep 的沉积时机。
本文件中的 kbt 表示 k_B*T，height 表示能量；CV 分量可以各有自己的单位。
"""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np


@dataclass
class HarmonicUmbrella:
    # center 和 stiffness 按 CV 分量存储；当前使用对角刚度矩阵。
    """对每个 CV 分量施加一个独立的谐和约束。
    
    center[D] 保存窗口中心；stiffness[D] 保存各分量的弹簧常数，单位为
    能量/CV单位²。传入单个 stiffness 时广播到全部分量。
    势能 V=1/2 sum_j stiffness[j]*(s[j]-center[j])²。
    """
    center: np.ndarray
    stiffness: np.ndarray

    def __post_init__(self):
        """将窗口参数整理为 NumPy 数组，复制广播后的刚度并检查有效范围。
        """
        self.center = np.atleast_1d(np.asarray(self.center, dtype=float))
        self.stiffness = np.broadcast_to(np.asarray(self.stiffness, dtype=float), self.center.shape).copy()
        if not np.all(np.isfinite(self.center)) or not np.all(self.stiffness > 0):
            raise ValueError('Umbrella needs finite centers and positive stiffness')

    def evaluate(self, coordinates):
        """求一个或一批 CV 的偏置能及其对 CV 的梯度。
        
        coordinates 形状为 [..., D]。返回 energy[...] 和 gradient[..., D]；
        省略批量轴时 energy 为 NumPy 标量。梯度符号为 dV/ds，转成力时取负。
        """
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.shape[-1:] != self.center.shape:
            raise ValueError('Last axis must match the CV dimension')
        displacement = coordinates - self.center
        return 0.5 * np.sum(self.stiffness * displacement**2, axis=-1), self.stiffness * displacement

    def update(self, step, coordinates):
        """固定窗口的参数在采样中保持恒定；返回 False 表示本步无历史更新。
        """
        return False


class WellTemperedMetaD:
    """通过历史 Gaussian hill 的叠加表示 well-tempered 偏置。
    
    widths[D] 是各 CV 方向的 sigma。centers[H][D] 与 heights[H] 按沉积顺序
    保存 H 个 hill；每项 height 都是当时衰减后的实际高度。
    bias_factor=gamma=(T+DeltaT)/T，因此能量尺度 k_B*DeltaT=kbt*(gamma-1)。
    last_step 记录最近处理过的步数，既包括沉积步，也包括普通步。
    """
    def __init__(self, widths, height=0.1, kbt=1., bias_factor=10., pace=10):
        """建立空的 hill 历史。
        
        height 是初始 hill 高度，kbt 是热能，bias_factor 必须大于 1。
        pace 是正整数步间隔；以 step % pace == 0 判断沉积，因此 step=0 也可沉积。
        """
        self.widths = np.atleast_1d(np.asarray(widths, dtype=float))
        if not np.all(np.isfinite(self.widths)) or not np.all(self.widths > 0):
            raise ValueError('Widths must be finite and positive')
        if height <= 0 or kbt <= 0 or bias_factor <= 1 or pace < 1:
            raise ValueError('Invalid MetaD parameters')
        self.height, self.kbt, self.bias_factor, self.pace = float(height), float(kbt), float(bias_factor), int(pace)
        self.centers, self.heights = [], []
        self.last_step = -1

    def evaluate(self, coordinates):
        """在单个 CV 向量 coordinates[D] 上计算历史 hill 的总和。
        
        每个 hill 为 h*exp(-1/2 sum_j((s_j-c_j)/sigma_j)²)，其 j 分量导数为
        -hill_value*(s_j-c_j)/sigma_j²。返回 float 能量和长度 D 的梯度数组。
        这里临时形成 [H,D] 位移矩阵，历史长度增加时求值成本同步增加。
        """
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.shape != self.widths.shape:
            raise ValueError('Expected one CV vector')
        if not self.centers:
            return 0., np.zeros_like(coordinates)
        displacement = coordinates - np.asarray(self.centers)
        contributions = np.asarray(self.heights) * np.exp(-0.5 * np.sum((displacement/self.widths)**2, axis=1))
        return float(np.sum(contributions)), np.sum(-contributions[:,None]*displacement/self.widths**2,axis=0)

    def update(self, step, coordinates):
        # 同一步重复读取/更新不会重复沉积；重启后步数继续单调递增。
        """处理一个整数时间步，并在沉积步追加当前坐标处的 hill。
        
        先求沉积前的 V，再按 h=height*exp(-V/[kbt*(gamma-1)]) 确定新高度。
        返回 True 表示追加了 hill；普通步或重复提交同一步返回 False。
        小于 last_step 的步数视为时间回退，调用方重启时需继续使用原有步号。
        """
        if step < self.last_step:
            raise ValueError('Time moved backwards')
        if step == self.last_step:
            return False
        energy, _ = self.evaluate(coordinates)
        self.last_step = int(step)
        if step % self.pace:
            return False
        self.centers.append(np.asarray(coordinates, dtype=float).tolist())
        self.heights.append(self.height * np.exp(-energy/(self.kbt*(self.bias_factor-1))))
        return True

    def save(self, path):
        """把参数、hill 历史和 last_step 保存为 schema=1 的 JSON。
        
        该状态用于恢复偏置；采样器的坐标和随机数生成器由调用方另外保存。
        """
        state = {**self.__dict__, 'widths': self.widths.tolist(), 'schema': 1}
        Path(path).write_text(json.dumps(state, indent=2))

    @classmethod
    def load(cls, path):
        """读取对应版本的 JSON，重建参数以及按原顺序排列的 hill 历史。
        """
        state = json.loads(Path(path).read_text())
        if state.pop('schema') != 1:
            raise ValueError('Unsupported restart version')
        bias = cls(**{key: state[key] for key in ('widths','height','kbt','bias_factor','pace')})
        bias.centers, bias.heights, bias.last_step = state['centers'], state['heights'], state['last_step']
        return bias


def exchange_log_ratio(left_bias, right_bias, left_state, right_state, kbt):
    """计算交换两窗口配置的对数 Boltzmann 权重比。
    
    left_bias/right_bias 是固定窗口；left_state/right_state 为交换前的 CV 向量。
    两个副本使用相同温度 kbt 和基础势，因此基础势的两项在交换前后抵消。
    返回 log_ratio=-DeltaV/kbt，其中 DeltaV 包含新配置两项减去旧配置两项。
    调用方用 log(u) < min(0, log_ratio) 决定交换，u 取 (0,1) 上的均匀数。
    """
    if kbt <= 0:
        raise ValueError('kBT must be positive')
    return float(-(left_bias.evaluate(right_state)[0] + right_bias.evaluate(left_state)[0]
                   - left_bias.evaluate(left_state)[0] - right_bias.evaluate(right_state)[0])/kbt)
