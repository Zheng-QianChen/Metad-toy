import numpy as np
chunk_size = 0

# outer="_kongbaiduizhao"
outer=""

a = np.loadtxt("HILLS"+outer)

runstep=300001
row_indices = np.arange(len(a))

cv = a[:, 1]
# a[:,0] = a[:,0] - 10000
# a[:,0] = a[:,0] + (row_indices // runstep)*runstep
print(a[:,0])

if chunk_size==0:
    chunk_size = len(a) // 5
    print(chunk_size)

n_chunks = len(cv) // chunk_size
reshaped_cv = cv[:n_chunks*chunk_size].reshape(-1, chunk_size)
means = reshaped_cv.mean(axis=1)
print(means)

variances = reshaped_cv.var(axis=1)
stds = reshaped_cv.std(axis=1)
print(stds)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(row_indices, cv, label='CV')
plt.ylim(0.2,0.6)
plt.savefig(f"cv_plot{outer}.png")