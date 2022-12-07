import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB

# Define figure
fig, ax = plt.subplots(2,3, figsize=(9,6))

mat_corrected = scipy.io.loadmat('Lab08/SalinasA_corrected.mat')['salinasA_corrected']

c10 = mat_corrected[:,:,10]
c100 = mat_corrected[:,:,100]
c200 = mat_corrected[:,:,200]


ax[0,0].imshow(c10, cmap='binary_r')
ax[0,0].set_title('Chanel 10')
ax[0,1].imshow(c100, cmap='binary_r')
ax[0,1].set_title('Chanel 100')
ax[0,2].imshow(c200, cmap='binary_r')
ax[0,2].set_title('Chanel 200')

ax[1,0].plot(mat_corrected[10,10,:])
ax[1,1].plot(mat_corrected[40,40,:])
ax[1,2].plot(mat_corrected[80,80,:])

# Saving fig
plt.tight_layout()
plt.savefig('Lab08/lab08.png')

# Zadanie 2
fig, ax = plt.subplots(1,2, figsize=(8,4))

red = mat_corrected[:,:,4]
blue = mat_corrected[:,:,26]
green = mat_corrected[:,:,12]

def normalize(r,g,b):
    red_n = (r - np.min(r))
    red_n = (red_n/np.max(red_n))

    green_n = (g - np.min(g))
    green_n = (green_n/np.max(green_n))

    blue_n = (b- np.min(b))
    blue_n = (blue_n/np.max(blue_n))

    rgb = np.zeros((83,86,3))
    rgb[:,:,0] = red_n
    rgb[:,:,1] = green_n
    rgb[:,:,2] = blue_n
    return rgb


resh = np.reshape(mat_corrected, (mat_corrected.shape[0]*mat_corrected.shape[1], mat_corrected.shape[2]))

pca = PCA(n_components=3)
principalComponents = pca.fit_transform(resh)

pca_img = np.reshape(principalComponents, (mat_corrected.shape[0], mat_corrected.shape[1], 3))

red_pca = pca_img[:,:,0]
blue_pca = pca_img[:,:,1]
green_pca = pca_img[:,:,2]

ax[0].imshow(normalize(red, green, blue), cmap='binary_r')
ax[1].imshow(normalize(red_pca, blue_pca, green_pca), cmap='binary_r')

plt.tight_layout()
plt.savefig('Lab08/lab08a.png')

# Zadanie 3

mat_gt = scipy.io.loadmat('Lab08/SalinasA_gt.mat')['salinasA_gt']

gt = np.reshape(mat_gt, (mat_gt.shape[0]*mat_gt.shape[1], 1))
#gt_wo0 = np.delete(np.where(mat_gt == 0))

# print(np.shape(mat_gt))
# print(mat_gt)
# k = np.where(mat_gt != 0)
#
# X = pca_img
# y = mat_gt
#
# skf = KFold(n_splits=3, shuffle=True, random_state=1234)
# clf = GaussianNB()
# scores = []
#
# for train_index, test_index in skf.split(X,y):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     clf.fit(X_train, y_train)
#     predict = clf.predict(X_test)
#     scores.append(accuracy_score(y_test, predict))

# mean_score = np.mean(scores)
# std_score = np.std(scores)
# print("Accuracy score: %.3f (%.3f)" % (mean_score, std_score))