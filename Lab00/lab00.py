import numpy as np
import matplotlib.pyplot as plt

# ex 1
mono = np.zeros((30,30))
mono = mono.astype(int)

mono[10:20, 10:20] = 1
mono[15:25, 15:25] = 2  
print(mono)

# ex 2
fig, ax = plt.subplots(2,2, figsize=(7,7))
ax[0,0].imshow(mono)
ax[0,1].imshow(mono, cmap = 'binary')

ax[0,0].set_title('mono')
ax[0,1].set_title('mono binary')

#plt.savefig("lab00")

# ex 3
color = np.zeros((30,30,3))
color = color.astype(float)

color[15:25, 5:15, 0] = 1
color[10:20, 10:20, 1] = 1 
color[5:15,  15:25, 2] = 1

ax[1,0].imshow(color)

negative = 1 - color
ax[1,1].imshow(negative)

ax[1,0].set_title('color')
ax[1,1].set_title('negative')

plt.tight_layout()
plt.savefig("lab00")
#print(color)