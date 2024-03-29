import pytest

from ..synth.filaments import generate_img_with_filaments


@pytest.fixture(scope='module', params=[dict(curve_type='line', size=10, n_filaments=10, maxval=255),
                                        dict(curve_type='sine', size=10, n_filaments=10, maxval=255),
                                        dict(curve_type='line', distribution='random',
                                             size=70, n_filaments=15, maxval=20),
                                        dict(curve_type='sine', size=50, n_filaments=20, maxval=16244),
                                        dict(curve_type='line', size=20, n_filaments=10, maxval=1),
                                        dict(curve_type='sine', distribution='aster', size=10,
                                             n_filaments=10, maxval=255),
                                        dict(curve_type='line', distribution='aster', size=10,
                                             n_filaments=10, maxval=255),
                                        dict(curve_type='bezier', size=10, n_filaments=10, maxval=50, m=0.3, n=0.2),
                                        ])
def curve_params(request):
    return request.param


@pytest.fixture(scope='module')
def line_imgs(curve_params):
    size = curve_params['size']
    img = generate_img_with_filaments(imgshape=[size] * 3,
                                      **{key: curve_params[key] for key in curve_params.keys()
                                         if key != 'size'})
    return img


@pytest.fixture(scope='module', params=[1, 2, 5, 10])
def gauss_kernel_size(request):
    return request.param


@pytest.fixture(scope='module')
def corruption_steps():
    return [
        ('perlin_noise', {'size': 50, 'value': 0.1}),
        ('poisson_noise', {'snr': 2}),
        ('convolve', {'sigma': 2}),
        ('gaussian_noise', {'snr': 100})
    ]
