import numpy as np
import matplotlib.pyplot as plt
import skimage.data as data

# Values
N = 100
D = 8
L = np.power(2, D)-1

# Define figure
fig, ax = plt.subplots(3,4, figsize=(16,16))

## Ex 1
x = np.linspace(0,11*np.pi,N)
sin = np.sin(x)
img = sin[:, np.newaxis]*sin[np.newaxis, :]

# Normalizing
img = (img - np.min(img))
img = (img/np.max(img))
img = np.rint(img*L)

# Fourier transformate
img_ft = np.fft.fftshift(np.fft.fft2(img))

## Ex 2
lin = np.linspace(0, 11*np.pi, 100)
x, y = np.meshgrid(lin, lin)

ampl = [5,3,4,7,8]
angl = [np.pi*2, np.pi*1.5, np.pi*4, np.pi*0.75, np.pi*3]
wave = [2,5,3,9,4]

zero = np.zeros((100,100))
        
for items in zip(ampl,angl,wave):
    zero += items[0]*np.sin(2*np.pi*(x*np.cos(items[1])+y*np.sin(items[1]))*(1/items[2]))

zero_ft = np.fft.fftshift(np.fft.fft2(zero))

camera = data.camera()
camera_ft = np.fft.fftshift(np.fft.fft2(camera))

# Ex 3
def redGreenBlue(image):
    x,y = np.shape(image)
    color_matrix = np.zeros((x,y,3))
    
    def Red(image):
        red = np.fft.ifftshift(np.real(image))
        red = np.fft.ifft2(red).real
        red = (red - np.min(red))
        red = (red / np.max(red))

        color_matrix[:,:, 0] = red
    
    def Green(image):
        green = np.fft.ifftshift(np.imag(image)*1j)
        green = np.fft.ifft2(green).real
        green = (green - np.min(green))
        green = (green / np.max(green))
        
        color_matrix[:,:, 1] = green
        
    
    def Blue(image):
        blue = np.fft.ifftshift(image)
        blue = np.fft.ifft2(blue).real
        blue = (blue - np.min(blue))
        blue = (blue / np.max(blue))
        
        color_matrix[:,:,2] = blue
        
    Red(image)
    Blue(image)
    Green(image)
    
    return color_matrix

# Plotting
ax[0,0].imshow(img, cmap='binary_r')
ax[0,1].imshow(np.abs(img_ft), cmap='binary_r')
ax[0,2].imshow(np.log(np.abs(img_ft)),cmap='binary_r')
ax[0,3].imshow(redGreenBlue(img_ft))

ax[1,0].imshow(zero, cmap='binary_r')
ax[1,1].imshow(np.abs(zero_ft), cmap='binary_r')
ax[1,2].imshow(np.log(np.abs(zero_ft)),cmap='binary_r')
ax[1,3].imshow(redGreenBlue(zero_ft))

ax[2,0].imshow(camera, cmap='binary_r')
ax[2,1].imshow(np.abs(camera_ft), cmap='binary_r')
ax[2,2].imshow(np.log(np.abs(camera_ft)),cmap='binary_r')
ax[2,3].imshow(redGreenBlue(camera_ft))

# Saving fig
plt.tight_layout()
plt.savefig('Lab05/lab05.png')