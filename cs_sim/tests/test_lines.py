import pytest

from ..lines import generate_img_with_lines


@pytest.fixture(scope='module', params=[dict(size=10, n_lines=10, maxval=255),
                                        dict(size=100, n_lines=15, maxval=20),
                                        dict(size=50, n_lines=20, maxval=16244),
                                        dict(size=20, n_lines=10, maxval=1)])
def line_params(request):
    return request.param


def test_lines(line_params):
    size = line_params.pop('size')
    img = generate_img_with_lines(imgshape=[size] * 3, **line_params)
    assert img.max() == line_params['maxval']
    assert img.shape == (size, size, size)
