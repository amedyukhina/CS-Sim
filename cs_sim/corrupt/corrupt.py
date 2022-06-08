import numpy as np
from cs_sim.corrupt.noise import gaussian_noise, poisson_noise, perlin_noise
from cs_sim.corrupt.conv import convolve

AVAILABLE_STEPS = [perlin_noise, poisson_noise, gaussian_noise, convolve]


def corrupt_image(img, corruption_steps):
    img = img.astype(float)
    maxval = np.max(img)
    for func, params in corruption_steps:
        if type(func) is str:
            funcs = [f for f in AVAILABLE_STEPS if f.__name__ == func]
            if len(funcs) > 0:
                func = funcs[0]
            else:
                raise ValueError(rf"{func} is not a valid step name; valid steps are:"
                                 rf"{[f.__name__ for f in AVAILABLE_STEPS]}")
        img = func(img, **params)

    img = img - np.min(img)
    img = img * maxval / np.max(img)

    return img
