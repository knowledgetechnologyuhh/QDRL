import numpy as np
from munch import Munch


def point_left_of_directed_line(point, dline):
    p = point
    q = dline.point
    d = dline.vector
    d_rot = np.array([-d[1], d[0]])  # d rotated by 90 degree
    return (p - q) @ d_rot > 0


def point_right_of_directed_line(point, dline):
    p = point
    q = dline.point
    d = dline.vector
    d_rot = np.array([-d[1], d[0]])  # d rotated by 90 degree
    return (p - q) @ d_rot < 0


def left_of(entity1, entity2):
    dline = Munch(
        point=entity2.bottom_left, vector=entity2.top_left - entity2.bottom_left
    )
    return all(
        point_left_of_directed_line(p, dline)
        for p in (
            entity1.top_left,
            entity1.top_right,
            entity1.bottom_left,
            entity1.bottom_right,
        )
    )


def right_of(entity1, entity2):
    dline = Munch(
        point=entity2.bottom_right, vector=entity2.top_right - entity2.bottom_right
    )
    return all(
        point_right_of_directed_line(p, dline)
        for p in (
            entity1.top_left,
            entity1.top_right,
            entity1.bottom_left,
            entity1.bottom_right,
        )
    )


def above(entity1, entity2):
    dline = Munch(point=entity2.top_left, vector=entity2.top_right - entity2.top_left)
    return all(
        point_left_of_directed_line(p, dline)
        for p in (
            entity1.top_left,
            entity1.top_right,
            entity1.bottom_left,
            entity1.bottom_right,
        )
    )


def below(entity1, entity2):
    dline = Munch(
        point=entity2.bottom_left, vector=entity2.bottom_right - entity2.bottom_left
    )
    return all(
        point_right_of_directed_line(p, dline)
        for p in (
            entity1.top_left,
            entity1.top_right,
            entity1.bottom_left,
            entity1.bottom_right,
        )
    )
    