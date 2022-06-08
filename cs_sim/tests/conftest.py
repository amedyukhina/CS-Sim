import pytest

from ..corrupt.kernels import gaussian_kernel
from ..synth.lines import generate_img_with_lines


@pytest.fixture(scope='module', params=[dict(size=10, n_lines=10, maxval=255),
                                        dict(size=70, n_lines=15, maxval=20),
                                        dict(size=50, n_lines=20, maxval=16244),
                                        dict(size=20, n_lines=10, maxval=1)])
def line_params(request):
    return request.param


@pytest.fixture(scope='module')
def line_imgs(line_params):
    params = line_params.copy()
    size = params.pop('size')
    img = generate_img_with_lines(imgshape=[size] * 3, **params)
    return img


@pytest.fixture(scope='module', params=[1, 2, 5, 10])
def gauss_kernels(request):
    kernel = gaussian_kernel([request.param] * 3)
    return kernel


@pytest.fixture(scope='module')
def corruption_steps():
    return [
        ('perlin_noise', {'size': 50, 'value': 0.1}),
        ('poisson_noise', {'snr': 2}),
        ('convolve', {'sigma': 2}),
        ('gaussian_noise', {'snr': 100})
    ]
