import random
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import disk
from skimage.measure import label
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans, MiniBatchKMeans, Birch, DBSCAN

fig, ax = plt.subplots(2, 3, figsize=(12,8))

# Zadanie 1
image = np.zeros((100,100,3)).astype('int')
groundtruth = np.zeros((100,100))

for i in range(3):
    rad1 = random.randint(10,40)
    cnt1 = (random.randint(rad1, 100-rad1), random.randint(rad1, 100-rad1))

    rr, cc = disk(cnt1, rad1, shape=groundtruth.shape)
    canal = random.randint(0,2)
    val = random.randint(100,255)
    image[rr,cc, canal] += val
    groundtruth[rr,cc] += val
    
    x, y = random.randint(0,100), random.randint(0,100)


label(groundtruth)
k = np.random.normal(0, 16, size=(100,100,3)).astype('int')

new_img = image + k
np.clip(new_img, 0, 255)

ax[0,0].imshow(new_img)
ax[0,0].set_title('image')
ax[0,1].imshow(groundtruth)
ax[0,1].set_title('groundtruth')

# ZADANIE 2

def normalize(array):
    ark = (array - np.mean(array))
    arr = (ark/np.std(ark))
    return arr

X = np.reshape(new_img, (100*100,3))
xx, yy = np.meshgrid(np.arange(100), np.arange(100))

xx = xx.flatten()
yy = yy.flatten()

xx = np.expand_dims(xx, axis=1)
yy = np.expand_dims(yy, axis=1)

r = np.concatenate((X, xx, yy), axis=1)
X = normalize(r)

y = np.reshape(groundtruth, (100*100))

print(np.shape(X), np.shape(y))
print(X[0], y[0])


# ZADANIE 3

km = KMeans()
mbkm = MiniBatchKMeans()
bir = Birch()
dbscan = DBSCAN()

km_fit = km.fit_predict(X)
mbkm_fit = mbkm.fit_predict(X)
bir_fit = bir.fit_predict(X)
dbscan_fit = dbscan.fit_predict(X)

ax[0,2].imshow(np.reshape(km_fit, (100,100)))
ax[0,2].set_title('Kmeans: %.3f' % (adjusted_rand_score(km_fit, y)))
ax[1,0].imshow(np.reshape(mbkm_fit, (100,100)))
ax[1,0].set_title('MiniBatchKMeans: %.3f' % (adjusted_rand_score(mbkm_fit, y)))
ax[1,1].imshow(np.reshape(bir_fit, (100,100)))
ax[1,1].set_title('Birch: %.3f' % (adjusted_rand_score(bir_fit, y)))
ax[1,2].imshow(np.reshape(dbscan_fit, (100,100)))
ax[1,2].set_title('DBSCAN: %.3f' % (adjusted_rand_score(dbscan_fit, y)))

plt.tight_layout()
plt.savefig('Lab11/lab11.png')
