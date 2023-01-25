import math
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import camera
from skimage.transform import resize

fig, ax = plt.subplots(2, 3, figsize=(12,8))

# Zadanie 1
img = camera()
img_128 = resize(img, (128,128))

gx = np.zeros((128,128))
gy = np.zeros((128,128))

def gradient_x(gx):
    sh_y, sh_x = np.shape(img_128)
    
    for y in range(sh_y):
        for x in range(sh_x):
            if x > 0 and x < 127:
                gx[x][y] = img_128[x+1][y] - img_128[x-1][y]

def gradient_y(gy):
    sh_y, sh_x = np.shape(img_128)
    
    for y in range(sh_y):
        for x in range(sh_x):
            if y > 0 and y < 127:
                gy[x][y] = img_128[x][y+1] - img_128[x][y-1]

gradient_x(gx)
gradient_y(gy)

mag = np.zeros((128,128))
angle = np.zeros((128,128))

for y in range(np.shape(mag)[0]):
    for x in range(np.shape(mag)[1]):
        mag[x][y] = math.sqrt((gx[x][y])**2 + (gy[x][y])**2)
        
angle = np.arctan(gy/gx) + np.pi/2
        
        
print(np.nanmin(angle), np.nanmax(angle))


ax[0,0].imshow(img_128, cmap='binary_r')
ax[0,1].imshow(gx, cmap='binary_r')
ax[0,2].imshow(gy, cmap='binary_r')
ax[1,0].imshow(mag, cmap='binary_r')
ax[1,1].imshow(angle, cmap='binary_r')

plt.tight_layout()
plt.savefig('Lab12/lab12.png')

# Zadanie 2

fig, ax = plt.subplots(2, 2, figsize=(12,8))
b = 0

s = 8
bins = 9
mask = np.zeros((np.shape(img_128)))

for i in range(16):
    for j in range(16):
        mask[i*s:i*s+s, j*s:j*s+s] =  b
        b += 1

hog = np.zeros((int((np.shape(mask)[0]/8*(np.shape(mask)[1]/8))),bins))
step = np.pi/bins

#ang_v = np.zeros()
#mag_v = np.zeros()

for val in np.unique(mask):
    #print(img_128[mask==val])
    break

ax[0,0].imshow(mask)
plt.tight_layout()
plt.savefig('Lab12/lab12a.png')