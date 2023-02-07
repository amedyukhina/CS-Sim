import numpy as np
from scipy import ndimage


def gaussian_kernel(sigma, scale: int = 4):
    sigma = np.array([sigma]).flatten()

    # calculate the size of the output array based on sigma and scale; make sure it is non-zero
    size = np.int_(np.round_(sigma))
    size[np.where(size < 1)] = 1
    size = size * 2 * scale + 1

    kernel = np.zeros(size)  # create an empty array
    kernel[tuple(np.int_(size / 2))] = 255.  # create a peak in the center of the kernel
    kernel = ndimage.gaussian_filter(kernel, sigma)  # smooth the peak with a Gaussian
    kernel = kernel / np.max(kernel)  # normalize the kernel
    return kernel
