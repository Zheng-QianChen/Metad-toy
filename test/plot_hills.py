#!/usr/bin/env python3
import numpy as np
import re
import sys

#===============================================================================
# --- 2. 文件 ---
LAMMPS_INPUT_FILE = 'run.in' 
data = np.loadtxt('HILLS')
try:
    # 检查参数数量：至少需要脚本名、LAMMPS输入文件和 HILLS 文件 (共3个)
    if len(sys.argv) < 3:
        # 抛出错误并打印使用说明
        raise IndexError("缺少输入文件参数。")
    
    LAMMPS_INPUT_FILE = sys.argv[1] # 'run.in'
    HILLS_FILE = sys.argv[2]        # 'HILLS'
    
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


def parse_metad_params(input_file_path, fix_id="metad"):
    """
    从 LAMMPS 输入文件中查找并解析 Metadynamics fix 命令。
    使用 C++ 构造函数的逻辑：基于关键字的无序解析。
    """
    
    # 1. 读取并清理文件内容 (处理多行连接符 & 和注释)
    try:
        with open(input_file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file_path}")
        return None

    # 将多行命令合并为单行，并去除注释行
    cleaned_tokens = []
    found_fix_metad = False
    content = content.replace('&\n', ' ').replace('&',' ')
    
    for line in content.split('\n'):
        stripped_line = line.strip()
        
        # 忽略注释和空行
        if not stripped_line or stripped_line.startswith('#'):
            continue
            
        # 移除行尾的续行符 & 并用空格代替
        cleaned_line = stripped_line
        
        # 将清理后的行分割成参数
        tokens = cleaned_line.split()
        
        if not tokens:
            continue
            
        # 查找 'fix metad all metad' 这一行，并提取所有后续参数
        # 注意: 假设 fix metad 命令的格式是: fix ID group style (例如: fix metad all metad ...)
        if not found_fix_metad:
            try:
                # 尝试找到 metad fix的起始位置
                start_index = tokens.index("metad", 3) # 从第4个参数开始找'metad'
                if tokens[0] == 'fix' and start_index == 3: # 严格检查
                    # 找到了 fix metad 命令的开头
                    cleaned_tokens.extend(tokens[4:]) # 仅保留 fix metad 后的所有参数
                    found_fix_metad = True
                else:
                    # 如果找到了 'metad' 关键字，但不是 fix 命令，则可能存在于其他地方
                    continue
            except ValueError:
                continue # 当前行不包含 metad fix 的开头
        # else:
        #     # 如果已经找到 fix metad 的开头，将后续行中的所有 token 都加进来
        #     cleaned_tokens.extend(tokens)
    print(cleaned_tokens)


    if not found_fix_metad:
        print(f"错误: 未在文件 '{input_file_path}' 中找到 'fix ... metad' 命令。")
        return None

    # 2. 模仿 C++ 构造函数，迭代解析参数
    
    # 初始化变量 (使用 C++ 默认值)
    params = {
        'sigma': 0.05, 
        'height0': 0.1, 
        'biasf': 10.0,
        'pace': 100,
        'cv_dim': 1,
        'nbin_num': 100,
        'continue': False,
        'lower_bound': None, # 确保 DIM 参数被提取
        'upper_bound': None
        # 假设 DIM 只针对 cv_dim=1 的情况
    }

    # 简化 nbin 数组的表示，用一个变量保存
    # nbin_num = 100 # Default
    
    i = 0
    num_args = len(cleaned_tokens)
    
    while i < num_args:
        arg = cleaned_tokens[i]
        print(arg)
        
        if arg == "GAUSSIAN":
            if i + 3 >= num_args: raise ValueError("GAUSSIAN requires 3 arguments.")
            params['sigma'] = float(cleaned_tokens[i+1])
            params['height0'] = float(cleaned_tokens[i+2])
            params['biasf'] = float(cleaned_tokens[i+3])
            i += 4
        
        elif arg == "PACE":
            if i + 1 >= num_args: raise ValueError("PACE requires 1 argument.")
            params['pace'] = int(cleaned_tokens[i+1])
            i += 2
            
        elif arg == "CV_dim":
            if i + 1 >= num_args: raise ValueError("CV_dim requires 1 argument.")
            params['cv_dim'] = int(cleaned_tokens[i+1])
            i += 2
            
        elif arg == "DISTANCE":
            if i + 2 >= num_args: raise ValueError("DISTANCE requires 2 atom IDs.")
            # 忽略 CV 定义，只跳过参数
            i += 3 
            
        elif arg == "STEINH":
            if i + 3 >= num_args: raise ValueError("STEINH requires at least 3 arguments (Q/L, num, group).")
            # STEINH <Q/L> <4/6/8/12> <group> ... (至少3个参数)
            
            # 找到 STEINH 块的结束位置 (即下一个关键字之前)
            j = i + 4 # 从第4个参数 (i+4) 开始查找可选参数或下一个关键字
            while j < num_args:
                next_arg = cleaned_tokens[j]
                # 检查下一个参数是否是已知的 Metad 关键字
                if next_arg in ["GAUSSIAN", "PACE", "CV_dim", "DISTANCE", "STEINH", "DIM", "METAD_RESTART", "WT"]:
                    break # 找到下一个关键字，跳出 STEINH 块
                # 检查 STEINH 的可选关键字，并跳过对应参数
                elif next_arg in ["cutoff_r", "cutoff_Natoms", "d_block_size"]:
                    j += 2 # 跳过关键字和数值
                else:
                    break # 遇到未知关键字或 STEINH 块结束
            
            i = j # 将 i 设置为 STEINH 块结束后的第一个参数
            
        elif arg == "DIM":
            if i + 4 >= num_args: raise ValueError("DIM requires 4 arguments (index, lower, upper, bins).")
            # DIM 1 0 40 400
            dim_index = int(cleaned_tokens[i+1]) - 1 # 0-based index
            # 注意：您的 C++ 代码只解析了一个 nbin_num，我们在这里假设 dim_index == 0
            if dim_index == 0:
                params['lower_bound'] = float(cleaned_tokens[i+2])
                params['upper_bound'] = float(cleaned_tokens[i+3])
                params['nbin_num'] = int(cleaned_tokens[i+4])
            
            i += 5
            
        elif arg == "METAD_RESTART":
            if i + 1 >= num_args: raise ValueError("METAD_RESTART requires 1 argument (0 or 1).")
            params['continue'] = (int(cleaned_tokens[i+1]) != 0)
            i += 2

        elif arg == "WT":
            # WT 1 (或 0)，您的 C++ 代码只检查了参数数量并跳过了
            if i + 1 >= num_args: raise ValueError("WT requires 1 argument (0 or 1).")
            i += 2
        
        else:
            # 遇到未知关键字
            # 您的 C++ 代码会抛出错误，这里我们直接返回 None 或抛出异常
            print(f"错误: 遇到 LAMMPS fix metad 无法识别的关键字: {arg}")
            return None
            
    # 3. 返回解析结果
    if params['lower_bound'] is None or params['upper_bound'] is None:
        print("警告: 未找到 DIM 边界设置。将使用默认值。")
    
    # 确保在 1D 模拟中 cv_dim 至少被设置过
    if params['cv_dim'] == 0:
        print("错误: CV_dim 未设置。")
        return None
        
    return params



metad_params = parse_metad_params(LAMMPS_INPUT_FILE)

if metad_params:
    # 假设您的 LAMMPS 模拟是 2D 或 1D，但您的 FES 绘图需要 nbin1 和 nbin2
    # 我们使用解析到的 nbin_num 作为两个维度的 bin 数
    if metad_params['cv_dim']==1:
        cv1_min,cv1_max = metad_params['lower_bound'], metad_params['upper_bound']
        nbin1 = metad_params['nbin_num']
        dx = (cv1_max - cv1_min) / nbin1
        X = np.linspace(cv1_min + dx/2, cv1_max - dx/2, nbin1)
        F = np.zeros_like(X)
        print(f"成功从 {LAMMPS_INPUT_FILE} 读取网格参数：nbin={nbin1}")
    elif metad_params['cv_dim']==2:
        cv1_min, cv1_max = 0.0, 40.0
        cv2_min, cv2_max = 0.0, 40.0
        nbin1 = metad_params['nbin_num']
        nbin2 = metad_params['nbin_num']
        dx = (cv1_max - cv1_min) / nbin1
        dy = (cv2_max - cv2_min) / nbin2
        x = np.linspace(cv1_min + dx/2, cv1_max - dx/2, nbin1)
        y = np.linspace(cv2_min + dy/2, cv2_max - dy/2, nbin2)
        X, Y = np.meshgrid(x, y, indexing='ij') 
        F = np.zeros_like(X)
        print(f"成功从 {LAMMPS_INPUT_FILE} 读取网格参数：nbin={nbin1}x{nbin2}")
else:
    print("无法初始化网格，请检查输入文件。")

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
