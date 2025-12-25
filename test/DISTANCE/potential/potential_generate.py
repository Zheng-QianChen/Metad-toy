import numpy as np

def custom_potential(r):
# 1. 背景容器势：让整体呈现一个大的凹槽，中心选在11附近
    # 这样可以保证 7-16 之间不会漂移到太高的能量
    v_rep = (4 / r)**12
    # v_bg = (r/4 - 2.5)**4
    v_bg = 0
    
    # 2. 定义三个势阱
    # 为了像图中那样平滑，我们将宽度 w 稍微调大一点
    wells = [
        {'r0': 6.0,  'H': 0.9, 'w': 1.8}, # 第一个阱
        {'r0': 10.0, 'H': 0.6, 'w': 1.2}, # 第二个阱
        {'r0': 14.0, 'H': 0.8, 'w': 1.4}, # 第三个阱（最深）
        {'r0': 18.0, 'H': -80.0, 'w': 0.8} # 第三个阱（最深）
    ]
    
    v_wells = 0
    for w in wells:
        v_wells -= w['H'] * np.exp(-(r - w['r0'])**2 / (2 * w['w']**2))
    
    return v_bg + v_wells + v_rep

# 生成表文件
r = np.linspace(0.1, 25, 2500) # 从1.0到25.0采样2000个点
v = custom_potential(r)
f = -np.gradient(v, r) # 力 = -dV/dr

import matplotlib.pyplot as plt
plt.plot(r,v)
plt.xlim(0,25)
plt.ylim(-1,1)
plt.savefig("potential.png")

with open("potential.table", "w") as fout:
    fout.write("# Custom Triple-Well Potential for LAMMPS\n")
    fout.write("\nMY_W_POTENTIAL\n")
    fout.write(f"N {len(r)}\n\n")
    for i in range(len(r)):
        fout.write(f"{i+1} {r[i]:.6f} {v[i]:.6f} {f[i]:.6f}\n")

print("Table file 'potential.table' has been generated.")
