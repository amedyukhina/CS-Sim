import warnings

import numpy as np
from scipy.spatial import distance
from skimage import morphology

from .bezier import get_bezier_curve_coords
from .line import get_line_coords
from .sine import get_sine_curve_coords


def generate_img_with_filaments(imgshape, margin=0, curve_type='line', distribution='random', n_filaments=10,
                                maxval=255, n_points=None, instance=False, thick=False, **curve_kwargs):
    """
    Generate an image with straight lines.
    The start and the end of each line are chosen randomly.
    The line coordinates in between are interpolated with the given number of point (`n_points`)

    Parameters
    ----------
    imgshape : tuple
        Image shape.
        The number of inputs should correspond to the number of dimensions.
    margin : int, optional
        Margin at the edge of the image to to keep clear of the filaments.
        Default is 0.
    curve_type : str
        Type of the curve ('line', 'bezier' or 'sine').
        Default is 'line'.
    distribution : str
        Distribution of filaments ('random' or 'aster').
        Default is 'random'.
    n_filaments : int or tuple, optional
        Number of filaments to generate.
        If tuple, the number of filaments will be drawn randomly from the given range.
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
    curve_kwargs : key value
        Parameters for the curve generation function

    Returns
    -------
    np.ndarray:
        Image with straight lines.
    """
    if curve_type == 'line':
        get_coords = get_line_coords
    elif curve_type == 'sine':
        get_coords = get_sine_curve_coords
    elif curve_type == 'bezier':
        get_coords = get_bezier_curve_coords
        if len(imgshape) > 2:
            warnings.warn('Bezier curves for 3D are not yet supported. Generating 2D instead.')
            imgshape = imgshape[-2:]
    else:
        raise ValueError("Invalid value for curve_type! Must be 'line' or 'sine_curve'")

    if distribution == 'random':
        generate_coords = generate_random
    elif distribution == 'aster':
        generate_coords = generate_aster
    else:
        raise ValueError("Invalid value for distribution! Must be 'random' or 'aster'")

    if n_points is None:
        n_points = 2 * np.max(imgshape)
    n_filaments = np.ravel([n_filaments])
    if len(n_filaments) == 1:
        nfil = n_filaments[0]
    else:
        nfil = np.random.randint(n_filaments[0], n_filaments[1] + 1)

    coords, values = generate_coords(imgshape, margin, nfil, n_points,
                                     get_coords, instance, maxval, **curve_kwargs)
    img = np.zeros(imgshape)
    for coord, val in zip(coords, values):
        img[coord] = val
    if thick:
        img = morphology.dilation(img)
    return img


def generate_random(imgshape, margin, nfil, n_points, get_coords, instance, maxval, **curve_kwargs):
    all_coords = []
    values = []
    for i in range(nfil):
        start, stop = np.array([np.random.randint(margin, s - margin, 2) for s in imgshape]).transpose()
        coords = get_coords(start, stop, n_points, **curve_kwargs)
        coords = remove_out_of_shape(np.int_(np.round_(coords)), imgshape)
        curval = i + 1 if instance else maxval
        all_coords.append(tuple(coords.transpose()))
        values.append(curval)

    return all_coords, values


def generate_aster(imgshape, margin, nfil, n_points, get_coords, instance, maxval,
                   minlen=None, discard_fraction=0.1, **curve_kwargs):
    if minlen is None:
        minlen = int(imgshape[0] / 20)
    all_coords = []
    values = []
    start = np.array([np.random.randint(margin, s - margin, 1) for s in imgshape]).ravel()
    for i in range(nfil):
        stop = start.copy()
        while distance.euclidean(start, stop) < minlen:
            stop = np.array([np.random.randint(margin, s - margin, 1) for s in imgshape]).ravel()
        coords = get_coords(start, stop, n_points, **curve_kwargs)
        coords = remove_out_of_shape(np.int_(np.round_(coords)), imgshape)
        coords = coords[int(n_points * discard_fraction):]
        curval = i + 1 if instance else maxval
        all_coords.append(tuple(coords.transpose()))
        values.append(curval)
    return all_coords, values


def remove_out_of_shape(coords, imgshape):
    coords[np.where(coords < 0)] = 0
    for i in range(len(imgshape)):
        coords[:, i] = np.where(coords[:, i] >= imgshape[i], imgshape[i] - 1, coords[:, i])
    return coords
