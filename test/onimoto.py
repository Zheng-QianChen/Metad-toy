#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
import os
import pandas as pd

# --- 1. 配置参数 (请根据您的模拟进行调整) ---
HILLS_FILE = 'HILLS'
OUTPUT_ANIMATION = 'HILLS_evolution.mp4'
TEMP = 300.0  # 模拟温度 (K)，用于设置图表标题

# 网格范围与点数 (必须与 FES 重构时一致)
cv1_min, cv1_max, nbin1 = 0.0, 40.0, 400
cv2_min, cv2_max, nbin2 = 0.0, 40.0, 400
dx = (cv1_max - cv1_min) / nbin1
dy = (cv2_max - cv2_min) / nbin2
x = np.linspace(cv1_min + dx/2, cv1_max - dx/2, nbin1)
y = np.linspace(cv2_min + dy/2, cv2_max - dy/2, nbin2)
X, Y = np.meshgrid(x, y, indexing='ij')

# --- 2. 加载 HILLS 数据 ---
try:
    # 假设列顺序：step, cv1, cv2, height, sigma
    data = pd.read_csv(HILLS_FILE, delim_whitespace=True, comment='#', 
                       names=['step', 'cv1', 'cv2', 'h', 'sigma'])
except FileNotFoundError:
    print(f"错误: 未找到 HILLS 文件 '{HILLS_FILE}'。")
    exit()

# 确保 sigma 列没有被忽略，如果它是常数，这里没问题
if (data['sigma'] == 0).all():
    # 如果所有 sigma 都是 0 (您的数据中有 0.15)，则需要检查读取是否正确
    print("警告: HILLS 文件中的 sigma 值似乎为零，请检查文件格式。")
    
# --- 3. 初始化 FES 数组 ---
F = np.zeros_like(X)
# 找出最大的能量，用于设置颜色条的范围
F_max = data['h'].max() * 5 # 初始估计，实际可能更高
# 如果是 WT-MetaD，F_max 可以更高，这里我们先取一个保守范围

# --- 4. 动画函数 ---

def update_fes(frame):
    """
    每一帧执行一次：累加一个新的高斯势垒，并更新绘图。
    """
    global F # 允许修改全局 F 数组
    
    # 提取当前步的 HILLS 数据 (step, cv1, cv2, h, sigma)
    try:
        current_hill = data.iloc[frame]
    except IndexError:
        return ax.collections # 如果帧数超过数据量，则停止
        
    cv1_c, cv2_c, h, sigma = current_hill['cv1'], current_hill['cv2'], current_hill['h'], current_hill['sigma']
    step = current_hill['step']
    
    # --- 计算并累加当前高斯势垒 ---
    # 假设 sigma1 = sigma2 = sigma
    if sigma != 0:
        new_F = h * np.exp(-((X - cv1_c)**2 + (Y - cv2_c)**2) / (2 * sigma**2))
    else:
        new_F = np.zeros_like(X)

    F += new_F
    
    # --- 绘图更新 ---
    ax.clear()
    
    # 绘制 FES (contourf)
    # levels 参数应根据 FES 范围动态调整或固定
    levels = np.linspace(0, F_max, 50) 
    
    # 使用 pcolormesh 或 contourf 绘制 FES
    c = ax.contourf(X, Y, F, levels=levels, cmap='viridis')
    
    # 标记当前沉积的高斯中心 (可选)
    ax.plot(cv1_c, cv2_c, 'rx', markersize=5, label='Current Hill')
    
    # 设置图表属性
    ax.set_title(f'Bias Potential Evolution (Step: {step})', fontsize=14)
    ax.set_xlabel(f'CV1 ({cv1_min:.1f} to {cv1_max:.1f})', fontsize=12)
    ax.set_ylabel(f'CV2 ({cv2_min:.1f} to {cv2_max:.1f})', fontsize=12)
    ax.set_xlim(cv1_min, cv1_max)
    ax.set_ylim(cv2_min, cv2_max)

    # 绘制 colorbar (只需要在第一帧设置，但FuncAnimation要求返回 Artist)
    # 这是一个简化方法，更复杂的方法需要手动管理 colorbar
    if frame == 0:
        fig.colorbar(c, ax=ax, label='Accumulated Bias Potential')
    
    return c.collections

# --- 5. 执行动画 ---

fig, ax = plt.subplots(figsize=(8, 7))

# Frame 总数等于 HILLS 文件中的行数
num_frames = len(data)

# interval: 每两帧之间的时间间隔 (毫秒)
# blit=False: 因为我们每次都重绘整个 FES 
anim = FuncAnimation(fig, update_fes, frames=num_frames, interval=50, blit=False)

# --- 6. 保存动画 ---
print(f"开始生成动画 ({num_frames} 帧)，请等待...")
# fps 越高，动画越快
anim.save(OUTPUT_ANIMATION, writer='ffmpeg', fps=20, dpi=100) 
print(f"动画生成完毕，文件已保存为: {OUTPUT_ANIMATION}")