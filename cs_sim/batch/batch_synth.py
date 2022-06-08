import os

import numpy as np
from joblib import Parallel, delayed
from skimage import io

from ..synth.lines import generate_img_with_lines


def __get_type(img):
    maxval = np.max(img)
    if maxval <= 1 or maxval >= 2 ** 16 or np.min(img) < 0:
        return np.float32
    elif maxval <= 255:
        return np.uint8
    else:
        return np.uint16


def batch_generate_img_with_lines(n_img, dir_out, fn_base='line_img', fn_ext='.tif',
                                  n_jobs=1, **params):
    """
    Generate given number of images with straight lines with given parameters and save to a given directory.

    Parameters
    ----------
    n_img : int
        Number of images to generate.
    dir_out : str
        Directory to save the images.
    fn_base : str, optional
        Base image name.
        Default is 'line_img'.
    fn_ext : str, optional
        Image file extension.
        Default is '.tif'.
    n_jobs : int, optional
        If greater than 1, will run the processes in parallel.
        Default is 1.
    params : key value
        Parameters for the `generate_img_with_lines` function.

    """
    os.makedirs(dir_out, exist_ok=True)

    def __process(fn_out, **kwargs):
        img = generate_img_with_lines(**kwargs)
        io.imsave(fn_out, img.astype(__get_type(img)))

    fns = [os.path.join(dir_out, rf"{fn_base}_{i:05d}{fn_ext}") for i in range(n_img)]

    if n_jobs > 1:
        Parallel(n_jobs=n_jobs)(delayed(__process)(fn, **params) for fn in fns)
    else:
        for fn in fns:
            __process(fn, **params)
