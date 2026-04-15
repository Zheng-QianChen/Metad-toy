#!/usr/bin/env python3
import numpy as np
import re
import sys

#===============================================================================
# --- 2. 文件 ---
LAMMPS_INPUT_FILE = 'run.in' 
data = np.loadtxt('HILLS')
cv_dim = 1
try:
    # 检查参数数量：至少需要脚本名、LAMMPS输入文件和 HILLS 文件 (共3个)
    if len(sys.argv) < 3:
        # 抛出错误并打印使用说明
        raise IndexError("缺少输入文件参数。")
    
    LAMMPS_INPUT_FILE = sys.argv[1] # 'run.in'
    HILLS_FILE = sys.argv[2]        # 'HILLS'

#    if len(sys.argv) >=3:
#        cv_dim = int(sys.argv[3])
    
except IndexError:
    print("错误: 请提供 LAMMPS 输入文件和 HILLS 文件路径。")
    print("用法: python plot.py <LAMMPS_Input_File> <HILLS_File>")
    sys.exit(1) # 退出脚本
#===============================================================================
# --- 2. 热力学参数 ---
# 注意: 请根据您的 LAMMPS/PLUMED 模拟设置调整这些值
T = 300.0  # MD 模拟温度 (K)
R = 8.3144621  # 气体常数 (J / mol K)
# 如果您的 FES 单位是 kJ/mol:
kBT = R * T / 1000.0  # k_B T (kJ/mol)
# 如果您的 FES 单位是 kcal/mol:
# kBT = (1.9872e-3) * T  # k_B T (kcal/mol)
print(f"使用的 kBT 值 (边缘积分常数): {kBT:.4f} kJ/mol")


def parse_metad_params(input_file_path):
    try:
        with open(input_file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file_path}")
        return None

    # 清理续行符并分词
    content = content.replace('&\n', ' ').replace('&', ' ')
    lines = content.split('\n')
    
    cleaned_tokens = []
    found_fix = False
    for line in lines:
        line = line.split('#')[0].strip() # 移除注释
        if not line: continue
        tokens = line.split()
        if tokens[0] == 'fix' and 'metad' in tokens:
            # 找到起始行，从 fix ID group style 之后开始抓取
            cleaned_tokens.extend(tokens[4:])
            found_fix = True
        elif found_fix:
            # 如果之前的行有续行符，接下来的 token 也会被加入
            # 注意：这里的逻辑简化了，假设 fix metad 是连续定义的
            cleaned_tokens.extend(tokens)

    if not found_fix: return None

    # 初始化参数
    params = {
        'sigma': 0.05, 'height0': 0.1, 'biasf': 10.0,
        'pace': 100, 'cv_dim': 1, 'nbin_num': 100,
        'continue': False,
        'bounds': {}  # 使用字典存储 {dim_idx: [lower, upper, bins]}
    }

    # 模拟 C++ 状态机解析
    i = 0
    while i < len(cleaned_tokens):
        arg = cleaned_tokens[i]
        
        if arg == "GAUSSIAN":
            params['sigma'] = float(cleaned_tokens[i+1])
            params['height0'] = float(cleaned_tokens[i+2])
            params['biasf'] = float(cleaned_tokens[i+3])
            i += 4
        elif arg == "PACE":
            params['pace'] = int(cleaned_tokens[i+1])
            i += 2
        elif arg == "CV_dim":
            params['cv_dim'] = int(cleaned_tokens[i+1])
            i += 2
        elif arg == "CAL":
            # 进入 CAL 块，跳过直到遇到 MetaD 的全局关键字
            i += 1
            while i < len(cleaned_tokens):
                if cleaned_tokens[i] in ["GAUSSIAN", "PACE", "CV_dim", "DIM", "METAD_RESTART", "WT", "CAL"]:
                    break
                i += 1 # 跳过 NAME, Q6, STEINH, Q, 6 等 CV 定义参数
        elif arg == "DIM":
            # DIM <index> <lower> <upper> <bins> <expr>
            try:
                dim_idx = int(cleaned_tokens[i+1])
                l_bound = float(cleaned_tokens[i+2])
                u_bound = float(cleaned_tokens[i+3])
                bins    = int(cleaned_tokens[i+4])
                # 存储该维度的配置
                params['bounds'][dim_idx] = [l_bound, u_bound, bins]
                # 更新全局 nbin_num (兼容旧逻辑)
                params['nbin_num'] = bins 
            except (ValueError, IndexError):
                print(f"警告: 解析 DIM {i} 时出错")
            i += 6
        elif arg == "METAD_RESTART":
            params['continue'] = (int(cleaned_tokens[i+1]) != 0)
            i += 2
        elif arg == "WT":
            i += 2
        else:
            i += 1
    return params



metad_params = parse_metad_params(LAMMPS_INPUT_FILE)

if cv_dim:
    metad_params['cv_dim']=cv_dim

if metad_params:
    if metad_params['cv_dim'] == 1:
        # 获取第 1 维的参数
        b1 = metad_params['bounds'].get(1, [0.0, 1.0, 100])
        cv1_min, cv1_max, nbin1 = b1[0], b1[1], b1[2]
        
        dx = (cv1_max - cv1_min) / nbin1
        X = np.linspace(cv1_min + dx/2, cv1_max - dx/2, nbin1)
        F = np.zeros_like(X)
        print(f"1D 网格：{cv1_min} 到 {cv1_max}, bins={nbin1}")

    elif metad_params['cv_dim'] == 2:
        # 分别获取第 1 维和第 2 维的参数
        b1 = metad_params['bounds'].get(1, [0.0, 1.0, 100])
        b2 = metad_params['bounds'].get(2, [0.0, 1.0, 100])
        
        cv1_min, cv1_max, nbin1 = b1[0], b1[1], b1[2]
        cv2_min, cv2_max, nbin2 = b2[0], b2[1], b2[2]
        
        dx = (cv1_max - cv1_min) / nbin1
        dy = (cv2_max - cv2_min) / nbin2
        
        x = np.linspace(cv1_min + dx/2, cv1_max - dx/2, nbin1)
        y = np.linspace(cv2_min + dy/2, cv2_max - dy/2, nbin2)
        X, Y = np.meshgrid(x, y, indexing='ij') 
        F = np.zeros_like(X)
        print(f"2D 网格：CV1[{cv1_min}, {cv1_max}] Bins={nbin1}")
        print(f"         CV2[{cv2_min}, {cv2_max}] Bins={nbin2}")

if metad_params['cv_dim']==1:
    for step, cv1, h, sigma in data:
        F += h * np.exp(-((X - cv1)**2) / (2 * sigma**2))
elif metad_params['cv_dim']==2:
    for step, cv1, cv2, h, sigma in data:
        F += h * np.exp(-((X - cv1)**2 + (Y - cv2)**2) / (2 * sigma**2))

# 保存
np.savetxt('FES.dat', F)

# 可选：画图
import matplotlib.pyplot as plt


if metad_params['cv_dim']==1:
    plt.plot(X, F, color='blue')
    plt.xlabel('CV')
    plt.ylabel('Marginal Free Energy')
    # plt.title(f'Marginal FES along CV1 (T={T} K)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('FES.png')
elif metad_params['cv_dim']==2:
    plt.contourf(X, Y, F, levels=50)
    plt.colorbar(label='Free energy [ε]')
    plt.xlabel('CV1'); plt.ylabel('CV2')
    plt.savefig('FES.png')

def calculate_marginal_fes(F_2D, kBT, axis, bin_size):
    """
    计算边缘自由能 F(CV_i)，即沿着 CV_j 积分。
    
    Args:
        F_2D (np.ndarray): 二维自由能曲面 F(CV_i, CV_j)。
        kBT (float): k_B * T。
        axis (int): 沿着哪个轴积分 (0 沿着 CV1 积分得到 F(CV2); 1 沿着 CV2 积分得到 F(CV1))。
        bin_size (float): 积分轴上的网格间距 (dx 或 dy)。
        
    Returns:
        np.ndarray: 一维边缘自由能 F(CV_i)。
    """
    # 玻尔兹曼因子
    # exp_term = e^{-F / kBT}
    exp_term = np.exp(-F_2D / kBT)
    
    # 沿着指定轴进行求和 (近似积分: sum(exp_term * bin_size))
    # 注意: 这里的 bin_size (dx 或 dy) 只是一个常数因子，不影响形状和归一化
    sum_exp = np.sum(exp_term, axis=axis) * bin_size 
    
    # 边缘自由能公式: F = -kBT * ln(sum_exp)
    F_marginal = -kBT * np.log(sum_exp)
    
    # 归一化: 将 F_min 设为 0
    F_marginal -= np.min(F_marginal)
    
    return F_marginal



if metad_params['cv_dim']==2:
    # A. 沿着 CV2 积分，得到 F(CV1)
    # 积分轴是 1 (对应 Y 轴/CV2)，网格大小是 dy
    F_CV1 = calculate_marginal_fes(F, kBT, axis=1, bin_size=dy)
    np.savetxt('Marginal_FES_CV1.dat', np.stack((x, F_CV1), axis=1), 
            header='CV1 FES_CV1', fmt='%.8e', comments='#')
    print("已计算并保存 F(CV1) 到 Marginal_FES_CV1.dat")
    # B. 沿着 CV1 积分，得到 F(CV2)
    # 积分轴是 0 (对应 X 轴/CV1)，网格大小是 dx
    F_CV2 = calculate_marginal_fes(F, kBT, axis=0, bin_size=dx)
    np.savetxt('Marginal_FES_CV2.dat', np.stack((y, F_CV2), axis=1), 
            header='CV2 FES_CV2', fmt='%.8e', comments='#')
    print("已计算并保存 F(CV2) 到 Marginal_FES_CV2.dat")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 绘制 F(CV1)
    ax1.plot(x, F_CV1, color='blue')
    ax1.set_xlabel('CV1')
    ax1.set_ylabel('Marginal Free Energy')
    ax1.set_title(f'Marginal FES along CV1 (T={T} K)')
    ax1.grid(True, linestyle='--', alpha=0.6)
    # 绘制 F(CV2)
    ax2.plot(y, F_CV2, color='red')
    ax2.set_xlabel('CV2')
    ax2.set_ylabel('Marginal Free Energy')
    ax2.set_title(f'Marginal FES along CV2 (T={T} K)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('Marginal_FES_comparison.png')
    print("边缘自由能图已保存为 Marginal_FES_comparison.png")
