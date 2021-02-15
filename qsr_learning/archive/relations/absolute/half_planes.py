def left_of(object1, object2):
    return object1.bbox.right < object2.bbox.left


def right_of(object1, object2):
    return object1.bbox.left > object2.bbox.right


def above(object1, object2):
    # Note: as the object moves up the y-value decreases
    return object1.bbox.bottom < object2.bbox.top


def below(object1, object2):
    # Note: as the object moves down the y-value increases
    return object1.bbox.top > object2.bbox.bottom
