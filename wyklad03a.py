from skimage.data import chelsea
import numpy as np
import matplotlib.pyplot as plt

# Przestrzen na loty
# fig, ax = plt.subplot(221)
    # 1.618 - zlota proporcja
fig, ax = plt.subplots(4,2, figsize = (12,12/1.618)) 

# Wczytujemy obraz
D = 8
L = np.power(2, D).astype(int)
raw_image = chelsea()

ax[0,0].imshow(raw_image)

hist = np.unique(raw_image, return_counts=True)
ax[1,0].scatter(*hist, marker = '.')

vhist = np.zeros((L))
vhist[hist[0]] = hist[1]
vhist /= np.sum(vhist)

ax[2,0].plot(vhist)
ax[2,0].set_ylim(0,0.1)

vdist = np.cumsum(vhist)
ax[3,0].plot(vdist)

# Przygotowanie transformacji monochromatycznym
monochrome_transform = np.array([0,1,1])
monochrome_transform = monochrome_transform / np.sum(monochrome_transform)

# Dokonujemy transformacji
mono_image = raw_image * monochrome_transform
mono_image = np.sum(mono_image, axis = 2).astype(np.uint8)

ax[0,1].imshow(mono_image, cmap = 'binary_r')

# Przygotowanie histogramu
# Najprostszy estymator gestosci elementow
hist = np.unique(mono_image, return_counts=True)
# zbior
ax[1,1].scatter(*hist, c ='black', marker = '.')

# Wektorowa forma histogramu
vhist = np.zeros((L))
vhist[hist[0]] = hist[1]
vhist /= np.sum(vhist)      # bez tego wektor

# Prezentacja
ax[2,1].plot(vhist, c='black')
ax[2,1].set_ylim(0,.1)      # 10%

# Dystrybuanta / krzywa intensywnosci
vdist = np.cumsum(vhist)
ax[3,1].plot(vdist, c='black')

# Znalezenie punktu
''' 
    a = np.argwhere(mono_image == 208)
    # print(a)
    # ax [-1,-1].imshow(mono_image==208) 
'''

# Zapisanie plota
plt.savefig('wyklad03a.png')
#plt.savefig('wyklad03.eps')