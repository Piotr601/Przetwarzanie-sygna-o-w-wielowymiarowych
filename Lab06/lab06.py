import numpy as np
import matplotlib.pyplot as plt

# Reading images
house = plt.imread('Lab06/image1.jpeg')
soldier = plt.imread('Lab06/image2.jpg')

# Define figure
fig, ax = plt.subplots(3,4, figsize=(18,12))





# Plotting images
ax[0,0].imshow(house, cmap='binary_r')
ax[0,1].imshow(soldier, cmap='binary_r')


# Saving fig
plt.tight_layout()
plt.savefig('Lab06/lab06.png')