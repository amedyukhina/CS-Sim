import numpy as np
from skimage import morphology


def generate_img_with_lines(imgshape, n_lines=10, maxval=255, n_points=None,
                            instance=False, thick=False):
    """
    Generate an image with straight lines.
    The start and the end of each line are chosen randomly.
    The line coordinates in between are interpolated with the given number of point (`n_points`)

    Parameters
    ----------
    imgshape : tuple
        Image shape.
        The number of inputs should correspond to the number of dimensions.
    n_lines : int, optional
        Number of lines to generate.
        Default is 10.
    maxval : scalar, optional
        The value to be assigned to the lines/foreground (the background value is 0).
        Default is 255.
    n_points : int, optional
        Number of points to represent each line.
        Should be on the order of image size.
        Increase if lines get disconnected.
        If None, the number of points is set to 2x the image dimension.
        Default is None.
    instance : bool, optional
        If True, assign unique intensity value to each line.
        Default is False.
    thick : bool, optional
        If True, will dilate the lines image to get thicker lines.
        Default is False.

    Returns
    -------
    np.ndarray:
        Image with straight lines.
    """
    if n_points is None:
        n_points = 2 * np.max(imgshape)
    img = np.zeros(imgshape)
    for i in range(n_lines):
        coords = []
        for j in range(len(imgshape)):
            inds = np.random.randint(0, imgshape[j], 2)
            coords.append(np.linspace(inds[0], inds[1], n_points, endpoint=True))
        curval = i + 1 if instance else maxval
        img[tuple(np.int_(np.round_(coords)))] = curval
    if thick:
        img = morphology.dilation(img)
    return img
