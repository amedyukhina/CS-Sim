import numpy as np
from scipy.signal import fftconvolve


def test_gauss(line_imgs, gauss_kernels):
    img = fftconvolve(line_imgs * 1., gauss_kernels * 1., mode='same')
    assert np.max(img) > 0
