import numpy as np
import pytest

from ..corrupt.conv import convolve


@pytest.fixture(params=[1, 3, 10])
def sigmas(request):
    return request.param


def test_gauss(line_imgs, gauss_kernels):
    img = convolve(line_imgs, psf=gauss_kernels)
    assert np.max(img) > 0


def test_gauss_with_sigma(line_imgs, sigmas):
    img = convolve(line_imgs, sigma=sigmas)
    assert np.max(img) > 0


def test_conv_error(line_imgs):
    with pytest.raises(ValueError):
        convolve(line_imgs)
