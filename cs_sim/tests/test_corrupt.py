import numpy as np
import pytest

from ..corrupt import corrupt_image
from ..corrupt.conv import convolve
from ..corrupt.noise import gaussian_noise, poisson_noise, perlin_noise
from ..synth.lines import generate_img_with_lines


@pytest.fixture
def corruption_steps():
    return [
        (perlin_noise, {'size': 50, 'value': 0.1}),
        (poisson_noise, {'snr': 2}),
        (convolve, {'sigma': 2}),
        (gaussian_noise, {'snr': 100})
    ]


def test_corrupt(corruption_steps):
    line_img = generate_img_with_lines([10] * 3)
    img = corrupt_image(line_img, corruption_steps)
    assert np.min(img) == 0
    assert np.max(img) > 0
