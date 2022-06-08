import warnings

import numpy as np
from scipy import ndimage

from .perlin import PerlinNoiseFactory


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


def perlin_noise(img, size, value, zoom=1):
    if zoom != 1:
        im = ndimage.interpolation.zoom(img, zoom=1. / zoom, order=1)
        size = np.array(size) / zoom
    else:
        im = img
    space_range = np.array(im.shape) / np.array(size)

    pnf = PerlinNoiseFactory(3, octaves=1, unbias=True, tile=space_range)
    noise_img = np.zeros(im.shape)
    for z in range(im.shape[0]):
        for y in range(im.shape[1]):
            for x in range(im.shape[2]):
                n = pnf(*np.array([z, y, x]) / np.array(size))
                noise_img[z, y, x] = int((n + 1) / 2 * 255 + 0.5)
    noise_img = noise_img / noise_img.max() * img.max() * value
    img = img + ndimage.interpolation.zoom(noise_img,
                                           zoom=np.array(img.shape) / np.array(noise_img.shape),
                                           order=1)
    img = img - np.min(img)
    img = img * 255. / img.max()
    return img
