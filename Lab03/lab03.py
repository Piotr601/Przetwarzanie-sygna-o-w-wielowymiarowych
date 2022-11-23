from pickletools import uint8
from skimage.data import chelsea
import numpy as np
import matplotlib.pyplot as plt

# Values
D = 8
L = np.power(2, D).astype(int)

# Define objects
fig, ax = plt.subplots(6, 3, figsize=(12,12))
raw_image = chelsea()
lut_base = np.arange(0,L)

# Other LUTS
lut_identity = np.linspace(0,L-1,L).astype(np.uint8)
lut_negative = np.linspace(L-1, 0, L).astype(np.uint8)
lut_treshould = np.zeros(L).astype(np.uint8)
lut_treshould[50:200] = L-1

# LUT Sinus
lut_sin = np.linspace(0, 2 * np.pi, L)
lut_sin = (((np.sin(lut_sin) +  1) / 2) * L-1)
lut_sin = lut_sin.astype(np.uint8)

# LUT Gamma 03
gamma_03 = 0.3
lut_gamma_03 = np.linspace(255, 0, L).astype(np.uint8)
lut_gamma_03 = (((lut_base/(L-1)) ** (1/gamma_03)) * (L-1)).astype(np.uint8)

# LUT Gamma 3
gamma_3 = 3
lut_gamma_3 = np.linspace(255, 0, L).astype(np.uint8)
lut_gamma_3 = (((lut_base/(L-1)) ** (1/gamma_3)) * (L-1)).astype(np.uint8)

# Definition to create histograms
def histograms(image, pos):
    hist_r = np.unique(image[:,:,0], return_counts=True)
    vhistr = np.zeros((L))
    vhistr[hist_r[0]] = hist_r[1]
    vhistr /= np.sum(vhistr) 
    ax[pos,2].plot(vhistr, c='r')
    
    hist_g = np.unique(image[:,:,1], return_counts=True)
    vhistg = np.zeros((L))
    vhistg[hist_g[0]] = hist_g[1]
    vhistg /= np.sum(vhistg) 
    ax[pos,2].plot(vhistg, c='g')

    hist_b = np.unique(image[:,:,2], return_counts=True)
    vhistb = np.zeros((L))
    vhistb[hist_b[0]] = hist_b[1]
    vhistb /= np.sum(vhistb) 
    ax[pos,2].plot(vhistb, c='b')
    

## Plotting information
# Img 1
identity_img = lut_identity[raw_image]
ax[0,0].plot(lut_identity, c='g')
ax[0,1].imshow(identity_img)
histograms(identity_img, 0)

# Img 2
negative_img = lut_negative[raw_image]
ax[1,0].plot(lut_identity, lut_negative, c='b')
ax[1,1].imshow(negative_img, cmap = 'binary_r')
histograms(negative_img, 1)

# Img 3
treshould_img = lut_treshould[raw_image]
ax[2,0].plot(lut_identity, lut_treshould, c='r')
ax[2,1].imshow(treshould_img)
histograms(treshould_img, 2)

# Img 4
sin_img = lut_sin[raw_image]
ax[3,0].plot(lut_identity, lut_sin, c='black')
ax[3,1].imshow(sin_img)
histograms(sin_img, 3)

# Img 5
gamma03_img = lut_gamma_03[raw_image]
ax[4,0].plot(lut_base, lut_gamma_03, c='magenta')
ax[4,1].imshow(gamma03_img)
histograms(gamma03_img, 4)

# Img 6
gamma3_img = lut_gamma_3[raw_image]
ax[5,0].plot(lut_base, lut_gamma_3, c='darkblue')
ax[5,1].imshow(gamma3_img)
histograms(gamma3_img, 5)

# Saving fig
plt.savefig('Lab03/lab03.png')