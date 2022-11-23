import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

PROG_05 = 0.5 
PROG_08 = 0.8

# Reading images
house = plt.imread('Lab06/image1.jpeg')
house = np.mean(house[::, ::], 2)

# Define figure
fig, ax = plt.subplots(3,2, figsize=(12,18))

# Ex 1
house_fft = np.fft.fftshift(np.fft.fft2(house))
house_log = np.log(np.abs(house_fft))

house_fft_n = (house_log - np.min(house_log))
house_fft_n = (house_fft_n/np.max(house_fft_n))

house_3 = np.where(house_fft_n > PROG_05, 1, 0)
house_4 = np.where(house_fft_n > PROG_08, 1, 0)
    
white_points = np.argwhere(house_fft_n > PROG_08)

print(white_points)

# Ex 2
house_del = house_fft

house_del[75,:] = 0
house_del[93,:] = 0
house_del[131,:] = 0
house_del[149,:] = 0

house_r_log = np.log(np.abs(house_del))

house_inverted = np.fft.ifft2(np.fft.ifftshift(house_del)).real


# Plotting images
ax[0,0].imshow(house, cmap='binary_r')
ax[0,1].imshow(house_log, cmap ='binary_r')
ax[1,0].imshow(house_3, cmap ='binary_r')
ax[1,1].imshow(house_4, cmap ='binary_r')
ax[2,0].imshow(house_r_log, cmap='binary_r')
ax[2,1].imshow(house_inverted, cmap='binary_r')

# Saving fig
plt.tight_layout()
plt.savefig('Lab06/lab06.png')