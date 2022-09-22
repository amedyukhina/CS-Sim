import numpy as np
from skimage import morphology


def generate_img_with_filaments(imgshape, curve_type='line', n_filaments=10, maxval=255, n_points=None,
                                instance=False, thick=False, **curve_kwargs):
    """
    Generate an image with straight lines.
    The start and the end of each line are chosen randomly.
    The line coordinates in between are interpolated with the given number of point (`n_points`)

    Parameters
    ----------
    imgshape : tuple
        Image shape.
        The number of inputs should correspond to the number of dimensions.
    curve_type : str
        Type of the curve ('line' or 'curve').
        Default is 'line'.
    n_filaments : int, optional
        Number of filaments to generate.
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
    elif curve_type == 'sine_curve':
        get_coords = get_sine_curve_coords
    else:
        raise ValueError("Invalid value for curve_type!")
    if n_points is None:
        n_points = 2 * np.max(imgshape)
    img = np.zeros(imgshape)
    for i in range(n_filaments):
        start, stop = np.array([np.random.randint(0, s, 2) for s in imgshape]).transpose()
        coords = get_coords(start, stop, n_points, **curve_kwargs)
        coords = remove_out_of_shape(np.int_(np.round_(coords)), imgshape)
        curval = i + 1 if instance else maxval
        img[tuple(coords.transpose())] = curval
    if thick:
        img = morphology.dilation(img)
    return img


def remove_out_of_shape(coords, imgshape):
    coords[np.where(coords < 0)] = 0
    for i in range(len(imgshape)):
        coords[:, i] = np.where(coords[:, i] >= imgshape[i], imgshape[i] - 1, coords[:, i])
    return coords


def get_line_coords(start, stop, n_points, **_):
    return np.linspace(start, stop, n_points, endpoint=True)


def get_sine_curve_coords(start, stop, n_points, n_sines=10,
                          k_range=0.1, a_range=1, w_range=1, shift_amplitude=10):
    vectors = get_ortho_vectors(start, stop)
    curves = [get_sine_curve(n_points, a_range, k_range, w_range, n_sines=n_sines) for _ in range(len(vectors))]
    shifts = [curve.reshape((-1, 1)) @ vector.reshape((1, -1)) * shift_amplitude for curve, vector in
              zip(curves, vectors)]
    coords = np.linspace(start, stop, n_points, endpoint=True) + np.stack(shifts).sum(0)
    return coords


def get_sine_curve(n_points, a_range, k_range, w_range, n_sines=10):
    """
    Return a sum of `n_sines` sine waves specified by the sine function: A sine(Kt+W).
    A, K, and W are drawn randomly from a uniform distribution [-range, range),
    where "range" is a parameter (`a_range`, `k_range`, and `w_range` respectively).
    """
    t = np.arange(n_points)
    k = (np.random.rand(n_sines) - 0.5) * k_range
    a = (np.random.rand(n_sines) - 0.5) * a_range
    w = (np.random.rand(n_sines) - 0.5) * w_range

    x = np.array([a[i] * np.sin(k[i] * t + w[i]) for i in range(n_sines)])
    return np.sum(x, axis=0)


def get_ortho_vectors(start, stop):
    a = (stop - start)  # speed_vector
    if len(a) == 2:
        return _get_ortho_for_2d(a)
    elif len(a) == 3:
        return _get_ortho_for_3d(a)
    else:
        raise ValueError(rf"Vectors must have length 2 or 3, {len(a)} given")


def _get_ortho_for_2d(a):
    d = len(a)
    b = np.random.rand(d) - 0.5
    b[1] = - a[0] * b[0] / b[1]
    b = b / np.linalg.norm(b)
    return [b]


def _get_ortho_for_3d(a):
    d = len(a)
    # first ortho vector:
    b = np.random.rand(d) - 0.5
    b[2] = - (a[0] * b[0] + a[1] * b[1]) / a[2]
    b = b / np.linalg.norm(b)

    # second ortho vector:
    c = np.random.rand(d) - 0.5
    c[2] = c[0] * (a[0] * b[1] - a[1] * b[0]) / (a[1] * b[2] - a[2] * b[1])
    c[1] = - (b[0] * c[0] + b[2] * c[2]) / b[1]
    c = c / np.linalg.norm(c)

    return [b, c]
