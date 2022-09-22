import numpy as np
import pytest

from cs_sim.synth.filaments import generate_img_with_filaments
from ..corrupt.noise import poisson_noise, gaussian_noise, perlin_noise


@pytest.fixture(params=[1, 2, 5, 10])
def snr(request):
    return request.param


@pytest.fixture(params=[10, 20, 50, 100])
def perlin_size(request):
    return request.param


@pytest.fixture(params=[0.01, 0.03, 0.1, 0.5, 1])
def perlin_value(request):
    return request.param


def test_gauss_noise(snr, line_imgs):
    imgf = gaussian_noise(line_imgs, snr)
    assert np.min(imgf) >= 0


def test_poisson_noise(snr, line_imgs):
    imgf = poisson_noise(line_imgs, snr)
    assert np.min(imgf) >= 0


def test_perlin_noise(perlin_size, perlin_value):
    line_img = generate_img_with_filaments([10] * 3)
    imgf = perlin_noise(line_img, perlin_size, perlin_value)
    assert np.min(imgf) >= 0
