import pandas as pd
import matplotlib.pyplot as plt

def plot_lammps_log(logfile):
    steps, temps = [], []
    with open(logfile, 'r') as f:
        start_reading = False
        for line in f:
            # 匹配表头，开始读取数据
            if "Step" in line and "Temp" in line:
                start_reading = True
                continue
            # 匹配数据结束（通常是 Loop time 或空行）
            if start_reading and ("Loop time" in line or not line.strip()):
                start_reading = False
                continue
            
            if start_reading:
                parts = line.split()
                if len(parts) >= 2 and parts[0].isdigit():
                    steps.append(int(parts[0]))
                    temps.append(float(parts[1]))

    # 绘图
    plt.figure(figsize=(8, 5))
    plt.plot(steps, temps, color='#1f77b4', label='Temperature')
    plt.axhline(y=300, color='r', linestyle='--', label='Target: 300K')
    plt.xlabel('Step')
    plt.ylabel('Temperature (K)')
    plt.title('LAMMPS Temperature Profile')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("temperature.png")

# 调用函数
plot_lammps_log('log.lammps')
