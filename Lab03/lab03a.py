from pickletools import uint8
from skimage.data import moon
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate hist and distribuant
def hist(image):
    hist = np.unique(image, return_counts=True)
    vhist = np.zeros((L))
    vhist[hist[0]] = hist[1]
    vhist /= np.sum(vhist) 
    vdist = np.cumsum(vhist)
    return vhist, vdist

# Values
D = 8
L = np.power(2, D).astype(int)

# Define objects
fig, ax = plt.subplots(2, 3, figsize=(12,12))
raw_image = moon()
lut_base = np.arange(0,L)

# Img 1
ax[0,0].imshow(raw_image, cmap='binary_r')

# Img 2, 3
v, d = hist(raw_image)

ax[0,1].plot(v)
ax[0,2].plot(d)

# Img 4
lut = d
lut = (lut * (L-1)).astype(np.uint8)
ax[1,0].plot(lut)

# Img 5
new_img = lut[raw_image]
ax[1,1].imshow(new_img, cmap='binary_r')

# Img 6
v1, d1 = hist(new_img)
ax[1,2].plot(v1)

# Saving fig
plt.savefig('Lab03/lab03a.png')