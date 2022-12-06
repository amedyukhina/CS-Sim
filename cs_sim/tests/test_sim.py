from cs_sim.synth.filaments import generate_img_with_filaments
import pytest


@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


def test_filaments(curve_params, dim):
    size = curve_params['size']
    img = generate_img_with_filaments(imgshape=[size] * dim,
                                      **{key: curve_params[key] for key in curve_params.keys()
                                         if key != 'size'})
    assert img.max() == curve_params['maxval']
    assert img.min() == 0
    if dim == 2 or curve_params['curve_type'] == 'bezier':
        assert img.shape == (size, size)
    else:
        assert img.shape == (size, size, size)
