from tkinter import N
import numpy as np
import matplotlib.pyplot as plt

# ex 1
fig, ax = plt.subplots(3,3, figsize=(10,10))
x = np.linspace(0, 4 * np.pi, 40)
y = np.sin(x)

ax[0,0].plot(x, y)

sec = y[:, None]*y[None, :]
sec_min, sec_max = np.min(sec), np.max(sec)
sec_str = 'min: ' + str(round((sec_min),3)) + ", max: " + str(round((sec_max),3))
ax[0,1].imshow(sec, cmap = 'binary')

new_sec = (sec - np.min(sec_min))
new_sec = new_sec / np.max(new_sec)
new_sec_str = 'min: ' + str(round((np.min(new_sec)),3)) + ", max: " + str(round((np.max(new_sec)),3)) 
ax[0,2].imshow(new_sec, cmap ='binary')


# ex 2
def bit(arg):
    return (2**arg)-1

bit_2 = np.rint(new_sec * bit(2))
ax[1,0].imshow(bit_2, cmap = 'binary')

bit_4 = np.rint(new_sec * bit(4))
ax[1,1].imshow(bit_4, cmap = 'binary')

bit_8 = np.rint(new_sec * bit(8))
ax[1,2].imshow(bit_8, cmap = 'binary')


# ex 3
noise_img = np.add(new_sec, np.random.normal(size = np.shape(new_sec)))
ax[2,0].imshow(noise_img, cmap = 'binary')

def noise(length):
    noise_sum = new_sec
    for i in range(0,length):
        noise_image = np.add(new_sec, np.random.normal(size = np.shape(new_sec)))
        noise_sum = np.add(noise_sum, noise_image)
    noise_sum = (noise_sum - np.min(noise_sum))
    noise_sum = noise_sum / np.max(noise_sum)
    return noise_sum
    
noise_50 = noise(50)
ax[2,1].imshow(noise_50, cmap = 'binary')

noise_1000 = noise(1000)
ax[2,2].imshow(noise_1000, cmap = 'binary')


# setting titles
ax[0,0].set_title('sin function')
ax[0,1].set_title(sec_str)
ax[0,2].set_title(new_sec_str)

ax[1,0].set_title('min: %i, max: %i' %(np.min(bit_2), np.max(bit_2)))
ax[1,1].set_title('min: %i, max: %i' %(np.min(bit_4), np.max(bit_4)))
ax[1,2].set_title('min: %i, max: %i' %(np.min(bit_8), np.max(bit_8)))

ax[2,0].set_title("noised")
ax[2,1].set_title("n=50")
ax[2,2].set_title("n=1000")

plt.tight_layout()
plt.savefig("lab01")