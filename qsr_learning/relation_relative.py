from qsr_learning import relation
from qsr_learning.entity import Entity
import numpy as np
from copy import deepcopy
import math


def rotate_bbox(entity1, entity2):
    """
    Return a copy of `entity1`, where the bottom side of its bounding box points to entity2.
    """
    v = (entity1.p + entity1.center) - (entity2.p + entity2.center)
    # The angle between v and the y-axis
    theta = (-np.arctan2(v[0], v[1])) % (2 * math.pi)
    entity1_rotated = deepcopy(entity1)
    entity1_rotated.frame_of_reference = "absolute"
    entity1_rotated.translate(-entity1_rotated.p)
    entity1_rotated.rotate(-theta)
    entity1_rotated.translate(entity1_rotated.p)
    entity1_rotated.frame_of_reference = "intrinsic"
    entity1_rotated.translate(-entity1_rotated.p)
    entity1_rotated.rotate(theta)
    entity1_rotated.translate(entity1_rotated.p)
    return entity1_rotated


def left_of(entity1: Entity, entity2: Entity, entity3: Entity):
    entity2_rotated = rotate_bbox(entity2, entity3)
    return relation.left_of(entity1, entity2_rotated)


def right_of(entity1: Entity, entity2: Entity, entity3: Entity):
    entity2_rotated = rotate_bbox(entity2, entity3)
    return relation.right_of(entity1, entity2_rotated)


def in_front_of(entity1: Entity, entity2: Entity, entity3: Entity):
    entity2_rotated = rotate_bbox(entity2, entity3)
    return relation.below(entity1, entity2_rotated)


def behind(entity1: Entity, entity2: Entity, entity3: Entity):
    entity2_rotated = rotate_bbox(entity2, entity3)
    return relation.above(entity1, entity2_rotated)
