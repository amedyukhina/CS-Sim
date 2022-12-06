import numpy as np


def get_bezier_curve_coords(start, stop, n_points, m=0.5, n=0.1):
    cp = generate_bezier_points(start, stop, m=m, n=n)
    return bezier_curve_from_control_points(cp, n_points=n_points)


def generate_bezier_points(start, stop, m=0.5, n=0.1):
    # first and last
    p0 = np.array([0, 0])
    p3 = np.array([0, 1])

    # middle points
    x = (np.random.rand(2) - 0.5) * m * 2
    y = np.random.rand(2) * (1 - 2 * n) + n
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[1], y[1]])

    # concatenate points into array
    points = np.stack([p0, p1, p2, p3])
    points = np.concatenate([points, np.ones((4, 1))], axis=1).transpose()

    # rotate by 45 degrees
    theta = - np.pi / 4
    rot_transform = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    points = rot_transform @ points

    # affine transform to match the start and stop points
    c = start[1]
    d = start[0]
    a = (stop[1] - c) * np.sqrt(2)
    b = (stop[0] - d) * np.sqrt(2)
    affine = np.array([
        [a, 0, c],
        [0, b, d],
        [0, 0, 1]
    ])
    points = affine @ points
    points = np.fliplr(points.transpose()[:, :2])
    return points


def bezier_curve_from_control_points(cpoints, n_points=10):
    t = np.linspace(0, 1, n_points).reshape(-1, 1)
    b = (1 - t) ** 3 * cpoints[0] + 3 * (1 - t) ** 2 * t * cpoints[1] + \
        3 * (1 - t) * t ** 2 * cpoints[2] + t ** 3 * cpoints[3]
    return b
