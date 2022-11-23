import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import AffineTransform, warp
from scipy.interpolate import interp2d
from scipy.interpolate import NearestNDInterpolator

# Ex 1
fig, ax = plt.subplots(4,2, figsize=(10,13))

image = data.chelsea()
ax[0,0].imshow(image)

new_chelsea = np.mean(image[::8, ::8], 2)
ax[0,1].imshow(new_chelsea, cmap = 'binary_r')

def rotate(phi):
    matrix = np.zeros((3,3))
    matrix[0,0] = np.cos(phi)
    matrix[0,1] = -np.sin(phi)
    matrix[1,0] = np.sin(phi)
    matrix[1,1] = np.cos(phi)
    matrix[2,2] = 1
    return matrix

transform = AffineTransform(matrix = rotate(np.pi/12)) 
rot_image = warp(new_chelsea, transform.inverse)
ax[1,0].imshow(rot_image, cmap = 'binary_r')

def shear(cx, cy):
    matrix = np.zeros((3,3))
    matrix[0,0] = 1
    matrix[0,1] = cx
    matrix[1,0] = cy
    matrix[1,1] = 1
    matrix[2,2] = 1
    return matrix

transform = AffineTransform(matrix = shear(0.5, 0)) 
rot_image2 = warp(new_chelsea, transform.inverse)
ax[1,1].imshow(rot_image2, cmap = 'binary_r')

# Ex 2
x_s, y_s  = np.shape(rot_image)
x = np.linspace(0, 8 * x_s, x_s)
y = np.linspace(0, 8 * y_s, y_s)

x_new = np.arange(0, 8 * x_s, 1)
y_new = np.arange(0, 8 * y_s, 1)

iinter_img = interp2d(y, x, rot_image, kind='cubic')
iinter_img = iinter_img(y_new, x_new)

ax[2,0].imshow(iinter_img, cmap = 'binary_r')


x_s, y_s  = np.shape(rot_image2)
x = np.linspace(0, 8 * x_s, x_s)
y = np.linspace(0, 8 * y_s, y_s)

x_new = np.arange(0, 8 * x_s, 1)
y_new = np.arange(0, 8 * y_s, 1)

iinter_img2 = interp2d(y, x, rot_image2, kind='cubic')
iinter_img2 = iinter_img2(y_new, x_new)

ax[2,1].imshow(iinter_img2, cmap = 'binary_r')

print(np.round(iinter_img[0:15,0:15],1))

# Ex 3
# x_s, y_s  = np.shape(rot_image2)
# x = np.linspace(0, 8 * x_s, x_s)
# y = np.linspace(0, 8 * y_s, y_s)
#
# x_new = np.arange(0, 8 * x_s, 1)
# y_new = np.arange(0, 8 * y_s, 1)
#
#
# def nearby(X_axis, Y_axis):
#     x = np.abs(x-X_axis)
#     y = np.abs(y-Y_axis)
#     return y, x

plt.tight_layout()
plt.savefig("lab02")