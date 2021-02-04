import numpy as np
from munch import Munch
from qsr_learning.relations.core import overlap


# TODO: Explain the meaning of the variables.
from qsr_learning.relations.core import (
    point_left_of_directed_line,
    point_right_of_directed_line,
)


def left_of(object1, object2):
    point_top_right = np.array([object1.bbox.right, object1.bbox.top])
    point_bottom_right = np.array([object1.bbox.right, object1.bbox.bottom])
    directed_line_right = Munch(
        point=np.array([object2.bbox.left, object2.bbox.top]), vector=np.array([-1, -1])
    )
    directed_line_left = Munch(
        point=np.array([object2.bbox.left, object2.bbox.bottom]),
        vector=np.array([-1, 1]),
    )
    constraint1 = point_left_of_directed_line(point_top_right, directed_line_right)
    constraint2 = point_right_of_directed_line(point_bottom_right, directed_line_left)
    # The latter constraint is needed for some boundary cases, where objec1 is much larger than object2.
    return (not overlap(object1, object2)) and (constraint1 and constraint2)


def right_of(object1, object2):
    point_top_left = np.array([object1.bbox.left, object1.bbox.top])
    point_bottom_left = np.array([object1.bbox.left, object1.bbox.bottom])
    directed_line_right = Munch(
        point=np.array([object2.bbox.right, object2.bbox.top]),
        vector=np.array([1, 1]),
    )
    directed_line_left = Munch(
        point=np.array([object2.bbox.right, object2.bbox.bottom]),
        vector=np.array([1, -1]),
    )
    constraint1 = point_left_of_directed_line(point_top_left, directed_line_right)
    constraint2 = point_right_of_directed_line(point_bottom_left, directed_line_left)
    return (not overlap(object1, object2)) and (constraint1 and constraint2)


def above(object1, object2):
    point_bottom_right = np.array([object1.bbox.right, object1.bbox.bottom])
    point_bottom_left = np.array([object1.bbox.left, object1.bbox.bottom])
    directed_line_right = Munch(
        point=np.array([object2.bbox.right, object2.bbox.top]),
        vector=np.array([1, -1]),
    )
    directed_line_left = Munch(
        point=np.array([object2.bbox.left, object2.bbox.top]),
        vector=np.array([-1, -1]),
    )
    constraint1 = point_left_of_directed_line(point_bottom_right, directed_line_right)
    constraint2 = point_right_of_directed_line(point_bottom_left, directed_line_left)
    return (not overlap(object1, object2)) and (constraint1 and constraint2)


def below(object1, object2):
    point_top_right = np.array([object1.bbox.right, object1.bbox.top])
    point_top_left = np.array([object1.bbox.left, object1.bbox.top])
    directed_line_right = Munch(
        point=np.array([object2.bbox.left, object2.bbox.bottom]),
        vector=np.array([-1, 1]),
    )
    directed_line_left = Munch(
        point=np.array([object2.bbox.right, object2.bbox.bottom]),
        vector=np.array([1, 1]),
    )
    constraint1 = point_left_of_directed_line(point_top_left, directed_line_right)
    constraint2 = point_right_of_directed_line(point_top_right, directed_line_left)
    return (not overlap(object1, object2)) and (constraint1 and constraint2)
