import numpy as np
import matplotlib.pyplot as plt
from skimage.data import chelsea
from skimage.segmentation import slic, watershed, quickshift
from skimage.color import label2rgb
from skimage.feature import canny

fig, ax = plt.subplots(3,3, figsize=(18,12))
img = chelsea()


# Zadanie 1
slic_img = slic(img)
mean_img = np.mean(img, 2)
watershared_img = watershed(mean_img)
qucikshift_img = quickshift(img)

ax[0,0].imshow(slic_img, cmap='twilight')
ax[0,0].set_title(f"segments: {len(np.unique(slic_img))}")

ax[1,0].imshow(watershared_img, cmap='twilight')
ax[1,0].set_title(f"segments: {len(np.unique(watershared_img))}")

ax[2,0].imshow(qucikshift_img, cmap='twilight')
ax[2,0].set_title(f"segments: {len(np.unique(qucikshift_img))}")


# Zadanie 2
ax[0,1].imshow(label2rgb(label=slic_img, image=img, kind='overlay'))
ax[1,1].imshow(label2rgb(label=watershared_img, image=img, kind='overlay'))
ax[2,1].imshow(label2rgb(label=qucikshift_img, image=img, kind='overlay'))

img1 = label2rgb(label=slic_img, image=img, kind='avg')
img2 = label2rgb(label=watershared_img, image=img, kind='avg')
img3 = label2rgb(label=qucikshift_img, image=img, kind='avg')

ax[0,2].imshow(img1)
ax[1,2].imshow(img2)
ax[2,2].imshow(img3)


# Zadanie 3
edges_c = np.zeros(np.shape(img))
edges = canny(mean_img, sigma= 3)

for x in range(3):
    edges_c[:,:,x] = edges
    
ax[0,2].imshow(np.where(edges_c == 0, img1, 0), cmap = 'binary')
ax[1,2].imshow(np.where(edges_c == 0, img2, 0), cmap = 'binary')
ax[2,2].imshow(np.where(edges_c == 0, img3, 0), cmap = 'binary')

plt.tight_layout()
plt.savefig('Lab10/lab10.png')