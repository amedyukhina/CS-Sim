import numpy as np

from ..corrupt import corrupt_image
from ..synth.filaments import generate_img_with_filaments


def test_corrupt(corruption_steps):
    line_img = generate_img_with_filaments([10] * 3)
    img = corrupt_image(line_img, corruption_steps)
    assert np.min(img) == 0
    assert np.max(img) > 0
