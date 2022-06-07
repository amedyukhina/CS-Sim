import numpy as np


def generate_img_with_lines(imgshape, n_lines=10, maxval=255, nval=100):
    img = np.zeros(imgshape)
    for i in range(n_lines):
        coords = []
        for j in range(len(imgshape)):
            inds = np.random.randint(0, imgshape[j], 2)
            coords.append(np.linspace(inds[0], inds[1], nval, endpoint=True))
        img[np.int_(np.round_(coords))] = maxval
    return img
