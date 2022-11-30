import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon

# Define figure
fig, ax = plt.subplots(4,4, figsize=(16,16))

# Ex 3
n = [10, 30, 100, 1000]
shep = shepp_logan_phantom()
### shep = shep[50:350,50:350]

for pos, angle in enumerate(n):
    theta = np.linspace(0, 180, angle)
    sgram = radon(shep, theta=theta)
    
    i_sgram = iradon(sgram, theta=theta)
    diff = i_sgram - shep
    
    ax[pos,0].imshow(shep, cmap='binary_r')
    ax[pos,1].imshow(sgram, cmap='binary_r', aspect='auto', interpolation='nearest')
    ax[pos,2].imshow(i_sgram, cmap='binary_r')
    ax[pos,3].imshow(diff, cmap='binary_r')

# Saving fig
plt.tight_layout()
plt.savefig('Lab07/lab07a.png')
### plt.savefig('Lab07/lab07b.png')