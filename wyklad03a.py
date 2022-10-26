from skimage.data import chelsea
import numpy as np
import matplotlib.pyplot as plt

# Przestrzen na loty
# fig, ax = plt.subplot(221)
    # 1.618 - zlota proporcja
fig, ax = plt.subplots(2,4, figsize = (12,12/1.618)) 

# Wczytujemy obraz
D = 8
L = np.power(2, D).astype(int)
raw_image = chelsea()

ax[0,0].imshow(raw_image)

# Przygotowanie transformacji monochromatycznym
monochrome_transform = np.array([0,1,1])
monochrome_transform = monochrome_transform / np.sum(monochrome_transform)

# Dokonujemy transformacji
print(raw_image.shape, monochrome_transform[None, None].shape)
mono_image = raw_image * monochrome_transform
mono_image = np.sum(mono_image, axis = 2).astype(np.uint8)

ax[1,0].imshow(mono_image, cmap = 'binary_r')

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
ax[1,2].plot(vhist, c='black')
ax[1,2].set_ylim(0,.1)      # 10%

# Dystrybuanta / krzywa intensywnosci
vdist = np.cumsum(vhist)
ax[1,3].plot(vdist, c='black')

# Znalezenie punktu
''' 
    a = np.argwhere(mono_image == 208)
    # print(a)
    # ax [-1,-1].imshow(mono_image==208) 
'''

# Zapisanie plota
plt.savefig('wyklad03a.png')
#plt.savefig('wyklad03.eps')