import warnings

import numpy as np


def poisson_noise(img, snr):
    if snr is None:
        warnings.warn("SNR is None, returning the input image")
        return img
    else:
        img = img.astype(np.float32)
        imgmax = snr ** 2  # new image maximum to generate the right level of Poisson noise
        ratio = imgmax / img.max()  # keep the ratio of the new and old maximum to recover the old dynamic range
        img = img * ratio
        img = np.random.poisson(img)
        if ratio > 0:
            img = img / ratio
        return img


def gaussian_noise(img, snr=None):
    if img is None:
        raise ValueError('self.image is None! The image has to be initialized!')

    img = img.astype(np.float32)
    if snr is not None:
        sig = img.max() * 1. / snr
        noise = np.random.normal(0, sig, img.shape)
        img = img + noise
        img[np.where(img < 0)] = 0
    return img
