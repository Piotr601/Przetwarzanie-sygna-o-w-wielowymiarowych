import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from skimage.draw import disk

# Reading images
soldier = plt.imread('Lab06/image2.jpg')

# Define figure
fig, ax = plt.subplots(2,2, figsize=(16,12))

# Ex 3

soldier_fft = np.fft.fftshift(np.fft.fft2(soldier))
soldier_log = np.log(np.abs(soldier_fft))

x = [240,175,295,110,360,175,295,240,110,50,175,
     50,110,175,295,360,410,295,360,410]
y = [150,235,235,320,320,405,405,490,150,240,80,
     405,490,560,560,490,405,80,150,235]
for i in range(int(len(x))):
    rr, cc = disk((x[i], y[i]), 9, shape=soldier_fft.shape)
    soldier_fft[rr,cc]=0

soldier_log2 = np.log(np.abs(soldier_fft))

soldier_inv = np.fft.ifft2(np.fft.ifftshift(soldier_fft)).real

# Plotting images
ax[0,0].imshow(soldier, cmap='binary_r')
ax[0,1].imshow(soldier_log, cmap='binary_r')

ax[1,0].imshow(soldier_log2, cmap='binary_r')
ax[1,1].imshow(soldier_inv, cmap='binary_r')
# Saving fig
plt.tight_layout()
plt.savefig('Lab06/lab06a.png')