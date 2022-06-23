import os
import warnings

import numpy as np
from joblib import Parallel, delayed
from skimage import io

from ..corrupt import corrupt_image


def batch_corrupt_image(dir_in, dir_out, corruption_steps, n_jobs=1):
    """
    Corrupt all images in a given directory with given parameters.

    Parameters
    ----------
    dir_in : str
        Input directory
    dir_out : str
        Output directory
    corruption_steps : list
        List of corruption steps with parameters.
    n_jobs : int, optional
        If greater than 1, will run the processes in parallel.
        Default is 1.

    """
    os.makedirs(dir_out, exist_ok=True)

    def __process(fn_in, fn_out, corr_stps):
        img = io.imread(fn_in)
        dtype = np.dtype(img.max())
        img = corrupt_image(img, corr_stps)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(fn_out, img.astype(dtype))

    fns = os.listdir(dir_in)
    fns_in = [os.path.join(dir_in, fn) for fn in fns]
    fns_out = [os.path.join(dir_out, fn) for fn in fns]

    if n_jobs > 1:
        Parallel(n_jobs=n_jobs)(delayed(__process)(fns_in[i], fns_out[i], corruption_steps)
                                for i in range(len(fns_in)))
    else:
        for i in range(len(fns_in)):
            __process(fns_in[i], fns_out[i], corruption_steps)
