import numpy as np
import pytest

from ..corrupt.conv import convolve
from ..corrupt.kernels import gaussian_kernel


@pytest.fixture(params=[1, 3, 10])
def sigmas(request):
    return request.param


def test_gauss(line_imgs, gauss_kernel_size):
    kernel = gaussian_kernel([gauss_kernel_size] * len(line_imgs.shape))
    img = convolve(line_imgs, psf=kernel)
    assert np.max(img) > 0


def test_gauss_with_sigma(line_imgs, sigmas):
    img = convolve(line_imgs, sigma=sigmas)
    assert np.max(img) > 0


def test_conv_error(line_imgs):
    with pytest.raises(ValueError):
        convolve(line_imgs)
