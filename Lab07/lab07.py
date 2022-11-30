import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon

# Define figure
fig, ax = plt.subplots(1,4, figsize=(16,4))

# Ex 1
shep = shepp_logan_phantom()

theta = np.linspace(0,180,100)
sgram = radon(shep, theta=theta)

# Ex 2
i_sgram = iradon(sgram, theta=theta)
diff = i_sgram - shep

# Plotting
ax[0].imshow(shep, cmap='binary_r')
ax[1].imshow(sgram, cmap='binary_r', aspect='auto', interpolation='nearest')
ax[2].imshow(i_sgram, cmap='binary_r')
ax[3].imshow(diff, cmap='binary_r')

# Saving fig
plt.tight_layout()
plt.savefig('Lab07/lab07.png')