import numpy as np
import matplotlib.pyplot as plt

# 1. 读取数据
try:
    # 假设 fes.dat 只有两列：CV 和 FES
    data = np.loadtxt('fes.dat')
    cv = data[:, 0]
    fes = data[:, 1]
except IOError:
    print("错误：找不到 fes.dat 文件。请确保该文件存在于当前目录下。")
    exit()

# 2. 绘图设置
plt.figure(figsize=(10, 6))

# 3. 绘制 FES 曲线
plt.plot(cv, fes, label='FES', linewidth=2, color='tab:blue')

# 4. 设置标签和标题
plt.title('Reconstructed Free Energy Surface (FES) - Distance', fontsize=16)
# 注意：我们使用 LaTeX 渲染 Å 和 k_B T 符号，需要确保 matplotlib 支持。
plt.xlabel(r'Collective Variable (Distance $r_{12}$) / $\AA$', fontsize=14)
plt.ylabel(r'Free Energy / $k_B T$', fontsize=14)

# 5. 添加网格和图例
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# 6. 保存和显示
plt.tight_layout() # 调整布局，防止标签重叠
plt.savefig('fes_plot.png')
# plt.show() # 如果在交互式环境或桌面环境运行，可以显示窗口
print("图表已成功保存为 fes_plot.png")
