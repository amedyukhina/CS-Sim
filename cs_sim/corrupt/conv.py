from scipy.signal import fftconvolve


def convolve(img, psf, mode='same'):
    imgf = fftconvolve(img * 1., psf * 1., mode=mode)
    return imgf
