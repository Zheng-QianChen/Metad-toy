"""四项原型共用的有限差分与结果保存工具。

测试脚本负责提出容差并断言，通过后再调用 write_result 形成可回查的 JSON。
数值导数工具保留输入数组的 shape，以便同一接口检查 CV、坐标和应变。
"""
import json
from pathlib import Path
import numpy as np


def finite_difference(function, value, step=1e-5):
    """按输入的每个分量计算标量函数的中心差分导数。
    
    function(value) 必须返回标量；value 可为任意 shape 的实数数组。
    返回 derivative 与 value 同形状，每一项为 [f(x+h)-f(x-h)]/(2h)。
    step 是绝对扰动量，单位与被扰动分量相同；每个分量调用两次 function。
    当前工具适用于光滑函数，调用方应按变量尺度和浮点误差选择 step 与容差。
    """
    value = np.asarray(value, dtype=float)
    derivative = np.zeros_like(value)
    for index in np.ndindex(value.shape):
        plus, minus = value.copy(), value.copy()
        plus[index] += step
        minus[index] -= step
        derivative[index] = (function(plus) - function(minus)) / (2 * step)
    return derivative


def write_result(directory, name, result):
    """写入 directory/name.json 并向标准输出打印同一份结果。
    
    result 包含本项测试指标；默认 status 为 pass，调用方提供 status 时可覆盖。
    本函数记录已经完成的检查，所有决定通过与否的断言放在调用它之前。
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result = {"status": "pass", **result}
    (directory / (name + '.json')).write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, ensure_ascii=False))
