from pickletools import uint8
from skimage.data import chelsea
import numpy as np
import matplotlib.pyplot as plt

# Parametry obrazu
D = 8
L = np.power(2, D).astype(int)

def hist(image):
    hist = np.unique(image, return_counts=True)
    vhist = np.zeros((image))
    vhist[hist[0]] = hist[1]
    vhist /= np.sum(vhist) 
    vdist = np.cumsum(vhist)
    return vhist, vdist

def monochrome(raw_image, monochrome_transform=[1,1,1]):
    monochrome_transform = np.array(monochrome_transform) / np.sum(monochrome_transform)

    mono_image = raw_image * monochrome_transform
    mono_image = np.sum(mono_image, axis = 2).astype(np.uint8)
    return mono_image

# Przygotowanie przestrzeni
fig, ax = plt.subplots(4, 2, figsize=(12,12))

# Wczytywanie obrazu
raw_image = monochrome(chelsea())
ax[0,0].imshow(raw_image, cmap='binary_r')



lut_base = np.arange(0,L)
gamma = 1.5

lut_gamma = np.linspace(255, 0, L).astype(np.uint8)
lut_gamma = (((lut_base/(L-1)) ** (1/gamma)) * (L-1)).astype(np.uint8)

ax[0,1].scatter(lut_base[::8], lut_base[::8], c='black', marker ='x')
gamma_image = lut_gamma[raw_image]     # raw_image musi byc calkowito liczbowe INT


ax[1,1].scatter(lut_base[::8], lut_gamma[::8], c='black', marker ='x')
ax[1,0].imshow(gamma_image, cmap='binary_r')



# Zapis do pliku
plt.savefig('wyklad03b.png')