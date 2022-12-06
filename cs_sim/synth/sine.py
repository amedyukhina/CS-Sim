import numpy as np


def get_sine_curve_coords(start, stop, n_points, n_sines=10, kmin=0.2, kmax=0.8, shift_amplitude=10, **_):
    vectors = get_ortho_vectors(start, stop)
    curves = [get_sine_curve(n_points, n_sines=n_sines, kmin=kmin, kmax=kmax) for _ in range(len(vectors))]
    shifts = [curve.reshape((-1, 1)) @ vector.reshape((1, -1)) * shift_amplitude for curve, vector in
              zip(curves, vectors)]
    coords = np.linspace(start, stop, n_points, endpoint=True) + np.stack(shifts).sum(0)
    return coords


def get_sine_curve(n_points, n_sines=10, kmin=0.2, kmax=0.8):
    """
    Return a sum of `n_sines` sine waves specified by the sine function: A sine(Kt+W).
    A is drawn randomly from the range [-1, 1].
    W is drawn randomly from the range [-pi, pi]
    K is drawn randomly from the range [-2 * pi * kmin, 2 * pi * kmax], where kmin and kmax are user defined
    """
    t = np.arange(n_points) / n_points

    kmin = np.pi * 2 * kmin
    kmax = np.pi * 2 * kmax
    k = np.random.rand(n_sines) * (kmax - kmin) + kmin
    a = (np.random.rand(n_sines) - 0.5) * 2
    w = (np.random.rand(n_sines) - 0.5) * np.pi / 2

    x = np.array([a[i] * np.sin(k[i] * t + w[i]) for i in range(n_sines)])
    return np.sum(x, axis=0) / np.sum(a)


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
