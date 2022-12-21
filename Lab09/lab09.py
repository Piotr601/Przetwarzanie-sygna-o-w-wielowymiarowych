import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk, erosion, dilation, opening, closing, rectangle

PROG = 0.6

fingerprint = plt.imread('Lab09/fingerprint.jpg')
fig, ax = plt.subplots(4,6, figsize=(18,12))

fingerprint_mean = np.mean(fingerprint, 2)

normalize   = (fingerprint_mean - np.min(fingerprint_mean))
normalize_2 = (normalize/np.max(normalize))

# Zadanie 1
def operations(row, footprint, cmap='binary'):
    after_prog = np.where(normalize_2 > PROG, 0, 1)

    fing_erosion = erosion(after_prog, footprint)
    fing_dilation = dilation(after_prog, footprint)
    fing_opening = opening(after_prog, footprint)
    fing_closing = closing(after_prog, footprint)
    
    # Zadanie 3
    fig_color = np.zeros(np.shape(fingerprint))
    fig_color[:,:,0] = fing_dilation
    fig_color[:,:,1] = fing_opening
    fig_color[:,:,2] = fing_closing
    # Koniec zadania 3
    
    ax[row,0].imshow(footprint, cmap)
    ax[row,1].imshow(fing_erosion, cmap='binary')
    ax[row,2].imshow(fing_dilation, cmap='binary')
    ax[row,3].imshow(fing_opening, cmap='binary')
    ax[row,4].imshow(fing_closing, cmap='binary')
    ax[row,5].imshow(fig_color, cmap='binary_r')

# Program
operations(0, disk(1))

# Zadanie 2
operations(1, disk(5))
operations(2, rectangle(10,1), cmap='binary_r')
operations(3, rectangle(1,10), cmap='binary_r')

# Zapis pliku
plt.tight_layout()
plt.savefig('Lab09/lab09.png')