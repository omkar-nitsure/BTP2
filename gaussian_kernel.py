import numpy as np
import matplotlib.pyplot as plt

x_21 = np.linspace(-0.9, 0.9, 21)
x_105 = np.linspace(-0.9, 0.9, 210)
array_21 = np.ones(21)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].stem(x_21, array_21, linefmt='r-', markerfmt='ro', basefmt=' ', label='Sample timesteps')
axes[0, 0].plot(x_105, np.exp(-(x_105)**2/(2 * 0.09 **2)), label='Gaussian Kernel', color='blue')
axes[0, 0].legend()

axes[0, 1].stem(x_21, array_21, linefmt='r-', markerfmt='ro', basefmt=' ', label='Sample timesteps')
axes[0, 1].plot(x_105, np.exp(-(x_105 - 0.045)**2/(2 * 0.09 **2)), label='Gaussian Kernel', color='blue')
axes[0, 1].legend()

axes[1, 0].stem(x_21, array_21, linefmt='r-', markerfmt='ro', basefmt=' ', label='Sample timesteps')
axes[1, 0].plot(x_105, np.exp(-(x_105)**2/(2 * 0.045 **2)), label='Gaussian Kernel', color='blue')
axes[1, 0].legend()

axes[1, 1].stem(x_21, array_21, linefmt='r-', markerfmt='ro', basefmt=' ', label='Sample timesteps')
axes[1, 1].plot(x_105, np.exp(-(x_105 - 0.045)**2/(2 * 0.045 **2)), label='Gaussian Kernel', color='blue')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('gaussian_kernels.png')
