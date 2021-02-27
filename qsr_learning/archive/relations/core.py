import numpy as np
from munch import Munch


def point_left_of_directed_line(point, directed_line):
    p = point
    q = directed_line.point
    d = directed_line.vector
    d_rot = np.array([-d[1], d[0]])  # d rotated by 90 degree
    # NOTE: As the y-axis is upside down, the sign is also switched.
    return (p - q) @ d_rot < 0


def point_right_of_directed_line(point, directed_line):
    p = point
    q = directed_line.point
    d = directed_line.vector
    d_rot = np.array([-d[1], d[0]])  # d rotated by 90 degree
    # NOTE: As the y-axis is upside down, the sign is also switched.
    return (p - q) @ d_rot > 0


def overlap(object1, object2, canvas_size=(224, 224), threshold=0.05):
    """
    Compute the overlap of objects.

    :param threshold: the fraction of the areas of any of the two objects that needs
        to be overlapped by the other object.
    """
    canvases = []
    for obj in [object1, object2]:
        canvas = np.zeros(canvas_size, dtype=bool)
        canvas[
            obj.bbox.top : obj.bbox.top + obj.mask.shape[0],
            obj.bbox.left : obj.bbox.left + obj.mask.shape[1],
        ] = obj.mask
        canvases.append(canvas)
    intersection_area = (canvases[0] & canvases[1]).sum()
    return (
        intersection_area / object1.area > threshold
        or intersection_area / object2.area > threshold
    )


def test_point_left_of_directed_line():
    point = np.array([1, 1])
    directed_line = Munch(point=np.array([2, -1]), vector=np.array([1, 1]))
    assert point_right_of_directed_line(point, directed_line)
    directed_line = Munch(point=np.array([2, -1]), vector=np.array([-1, 1]))
    assert not point_right_of_directed_line(point, directed_line)
