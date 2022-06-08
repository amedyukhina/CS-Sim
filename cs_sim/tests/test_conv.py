import numpy as np
from ..corrupt.conv import convolve


def test_gauss(line_imgs, gauss_kernels):
    img = convolve(line_imgs, gauss_kernels)
    assert np.max(img) > 0
