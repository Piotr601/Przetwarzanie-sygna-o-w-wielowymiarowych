import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

# Reading image 
raw_image = plt.imread('Lab04/vessel.jpeg')
raw_image = np.mean(raw_image[::, ::], 2)

# Define objects
fig, ax = plt.subplots(3, 4, figsize=(18,12))

# Sobel operators
s1 = np.matrix('-1 0 1; -2 0 2; -1 0 1')
s2 = np.matrix('0 1 2; -1 0 1; -2 -1 0')
s3 = np.matrix('1 2 1; 0 0 0; -1 -2 -1')
s4 = np.matrix('2 1 0; 1 0 -1; 0 -1 -2')

# Convolving images
img_s1 =  ndimage.convolve(raw_image, s1)
img_s2 =  ndimage.convolve(raw_image, s2)
img_s3 =  ndimage.convolve(raw_image, s3)
img_s4 =  ndimage.convolve(raw_image, s4)

# Correlation
def correlation(image, sop):
    y, x = np.shape(image)
    y, x = y-2, x-2
    result = np.zeros((y,x))

    for ox in range(y):  
        for oy in range(x):
            result[ox,oy] = image[ox,oy]*sop[0,0] + image[ox+1,oy]*sop[1,0] + image[ox+2,oy]*sop[2,0] + image[ox,oy+1]*sop[0,1] + image[ox+1,oy+1]*sop[1,1] + image[ox+2,oy+1]*sop[2,1] + image[ox,oy+2]*sop[0,2] + image[ox+1,oy+2]*sop[1,2] + image[ox+2,oy+1]*sop[2,2]
             
    return result          

cor1 = correlation(raw_image, s1)
cor2 = correlation(raw_image, s2)
cor3 = correlation(raw_image, s3)
cor4 = correlation(raw_image, s4)

# Modified correlation
def modified_correlation(image, kernel):
    
    kernel = np.flipud(kernel)
    new_kernel = np.fliplr(kernel)
    
    img_y, img_x = np.shape(image)
    ker_y, ker_x = np.shape(new_kernel)
    
    if ker_y > img_y and ker_x > img_x:
        new_kernel, image = image, new_kernel
        img_y, img_x, ker_y, ker_x = ker_y, ker_x, img_y, img_x
    
    img_yborder = img_y-ker_y+1
    img_xborder = img_x-ker_x+1
    
    result = np.zeros((img_yborder, img_xborder))

    for ox in range(img_yborder):  
        for oy in range(img_xborder):
            arr_sum = new_kernel*image[ox:ox+ker_x, oy:oy+ker_y]
            result[ox,oy] = np.sum(arr_sum)
    return result  

    
m_cor1 = modified_correlation(raw_image, s1)
m_cor2 = modified_correlation(raw_image, s2)
m_cor3 = modified_correlation(raw_image, s3)
m_cor4 = modified_correlation(raw_image, s4)

## Plotting images
# Ex 1
ax[0,0].set_title('S1')
ax[0,0].imshow(img_s1, cmap = 'binary_r')
ax[0,1].set_title('S2')
ax[0,1].imshow(img_s2, cmap = 'binary_r')
ax[0,2].set_title('S3')
ax[0,2].imshow(img_s3, cmap = 'binary_r')
ax[0,3].set_title('S4')
ax[0,3].imshow(img_s4, cmap = 'binary_r')

# Ex2
ax[1,0].imshow(cor1, cmap = 'binary_r')
ax[1,1].imshow(cor2, cmap = 'binary_r')
ax[1,2].imshow(cor3, cmap = 'binary_r')
ax[1,3].imshow(cor4, cmap = 'binary_r')

# Ex3
ax[2,0].imshow(m_cor1, cmap = 'binary_r')
ax[2,1].imshow(m_cor2, cmap = 'binary_r')
ax[2,2].imshow(m_cor3, cmap = 'binary_r')
ax[2,3].imshow(m_cor4, cmap = 'binary_r')

# Saving fig
plt.tight_layout()
plt.savefig('Lab04/lab04.png')