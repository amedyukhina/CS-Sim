import numpy as np
from scipy.signal import fftconvolve

from .kernels import gaussian_kernel


def convolve(img, psf=None, sigma=None, mode='same'):
    ndim = len(img.shape)
    if psf is None:
        if sigma is not None:
            sigma = np.ravel([sigma])
            if len(sigma) < ndim:
                sigma = np.array([sigma[0]] * ndim)
            psf = gaussian_kernel(sigma)
        else:
            raise ValueError("Either psf or sigma must be provided")
    imgf = fftconvolve(img * 1., psf * 1., mode=mode)

    return imgf
