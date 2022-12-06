import numpy as np


def get_line_coords(start, stop, n_points, **_):
    return np.linspace(start, stop, n_points, endpoint=True)
