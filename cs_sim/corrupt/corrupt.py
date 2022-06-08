import numpy as np


def corrupt_image(img, corruption_steps):
    img = img.astype(float)
    maxval = np.max(img)
    for func, params in corruption_steps:
        img = func(img, **params)

    img = img - np.min(img)
    img = img * maxval / np.max(img)

    return img
