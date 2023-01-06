import warnings

import numpy as np
from skimage import morphology

from .bezier import get_bezier_curve_coords
from .line import get_line_coords
from .sine import get_sine_curve_coords


def generate_img_with_filaments(imgshape, margin=0, curve_type='line', distribution='random', n_filaments=10,
                                maxval=255, n_points=None, instance=False, thick=False, **kwargs):
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
    kwargs : key value
        Parameters for the curve generation function and aster generation

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
    else:
        raise ValueError("Invalid value for curve_type! Must be 'line' or 'sine_curve'")

    if distribution == 'random':
        generate_coords = generate_random
    elif distribution == 'aster':
        generate_coords = generate_aster
    else:
        raise ValueError("Invalid value for distribution! Must be 'random' or 'aster'")

    if curve_type == 'bezier' or distribution == 'aster':
        if len(imgshape) > 2:
            warnings.warn('Bezier curves and aster distribution for 3D are not yet supported. Generating 2D instead.')
            imgshape = imgshape[-2:]

    if n_points is None:
        n_points = 2 * np.max(imgshape)
    n_filaments = np.ravel([n_filaments])
    if len(n_filaments) == 1:
        nfil = n_filaments[0]
    else:
        nfil = np.random.randint(n_filaments[0], n_filaments[1] + 1)

    coords, values = generate_coords(imgshape, margin, nfil, n_points,
                                     get_coords, instance, maxval, **kwargs)
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
                   minlen=5, discard_fraction=0.1, jitter=0,
                   center=None, r_mean=None, r_std=0.1, direction=None, angle_range=None,
                   **curve_kwargs):
    all_coords = []
    values = []

    if center is None:  # if center is not define, select randomly within margins
        center = np.array([np.random.randint(margin, s - margin, 1) for s in imgshape]).ravel()
    if direction is None:
        direction = 0
    if angle_range is None:  # if angle range is not defined, use all angles
        angle_range = 180

    for i in range(nfil):
        # start point
        start = np.array([c + np.random.randint(-jitter, jitter + 1, 1) for c in center]).ravel()

        # angle and radius
        a = to_radians(direction + np.random.rand() * 2 * angle_range - angle_range)
        if r_mean is None:
            r = np.random.rand() * (imgshape[0] / 2 - minlen) + minlen
        else:
            r = max(minlen, np.random.normal(r_mean, r_mean * r_std))

        # generate the end point from angle and radius, make sure it within the image border
        stop = np.int_(np.round_([r * np.sin(a) + start[0], r * np.cos(a) + start[1]]))
        stop = np.stack([np.stack([stop, np.array(imgshape) - 1]).min(0), np.zeros_like(stop)]).max(0)

        # get the entire curve coordinates, remove out of border, discard some start coordinates
        coords = get_coords(start, stop, n_points, **curve_kwargs)
        coords = remove_out_of_shape(np.int_(np.round_(coords)), imgshape)
        coords = coords[int(n_points * discard_fraction):]
        curval = i + 1 if instance else maxval
        all_coords.append(tuple(coords.transpose()))
        values.append(curval)
    return all_coords, values


def to_radians(x):
    return x / 180. * np.pi


def remove_out_of_shape(coords, imgshape):
    coords[np.where(coords < 0)] = 0
    for i in range(len(imgshape)):
        coords[:, i] = np.where(coords[:, i] >= imgshape[i], imgshape[i] - 1, coords[:, i])
    return coords
