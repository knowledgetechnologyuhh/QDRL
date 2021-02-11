import math
import random
from copy import deepcopy
from itertools import product
from typing import Callable, List, Tuple

import numpy as np
from munch import Munch


class Entity:
    def __init__(self, w, h, p=(0, 0), theta=0, name=None, color=None):
        self.bbox_float = np.array([[0, 0], [0, h], [w, h], [w, 0]])
        self.rotate(theta)
        self.translate(p)
        self.name = name
        self.color = color

    def rotate(self, theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        self.bbox_float = self.bbox_float @ R.transpose()

    def translate(self, p):
        self.bbox_float += np.array(p)

    @property
    def bbox(self):
        """Round the oringinal bbox coordinates."""
        return self.bbox_float.round().astype(int)

    @property
    def bottom_left(self):
        return self.bbox[0]

    @property
    def top_left(self):
        return self.bbox[1]

    @property
    def top_right(self):
        return self.bbox[2]

    @property
    def bottom_right(self):
        return self.bbox[3]

    def __repr__(self):
        return f"{self.name}: {', '.join(str(tuple(p)) for p in self.bbox)}"


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


def inside_canvas(entity: Entity, canvas_size: Tuple[int, int]) -> bool:
    """Check whether entity is in the canvas."""
    xs_inside_canvas = all(
        (0 < entity.bbox[:, 0]) & (entity.bbox[:, 0] < canvas_size[0])
    )
    ys_inside_canvas = all(
        (0 < entity.bbox[:, 1]) & (entity.bbox[:, 1] < canvas_size[1])
    )
    return xs_inside_canvas and ys_inside_canvas


def in_relation(
    entity1: Entity, entity2: Entity, relations: List[Callable[[Entity, Entity], bool]]
) -> bool:
    """Check whether entity1 and entity2 satisfy any of the given relations."""
    return any(relation(entity1, entity2) for relation in relations)


def generate_entities(
    num_entities: int = 5,
    w_range: Tuple[int, int] = (10, 30),
    h_range: Tuple[int, int] = (10, 30),
    canvas_size: Tuple[int, int] = (224, 224),
    relations: List[Callable[[Entity, Entity], bool]] = [
        left_of,
        right_of,
        above,
        below,
    ],
) -> List[Entity]:
    """
    :param canvas_size: (width, height)
    """

    # Sample entities
    entities_origin = []
    for i in range(num_entities):
        w = random.uniform(*w_range)
        h = random.uniform(*h_range)
        entities_origin.append(Entity(w, h, name=str(i)))

    # Rotate and translate the entities.
    entities_in_canvas = entities_in_relation = False
    while not (entities_in_canvas and entities_in_relation):
        entities = deepcopy(entities_origin)
        # Rotate entity
        for entity in entities:
            theta = random.uniform(0.0, 2 * math.pi)
            entity.rotate(theta)

        # Translate entity
        for entity in entities:
            p = (random.uniform(0, canvas_size[0]), random.uniform(0, canvas_size[1]))
            entity.translate(p)

        # Avoid boundary cases
        entities_in_canvas = all(
            inside_canvas(entity, canvas_size) for entity in entities
        )
        entities_in_relation = all(
            in_relation(entity1, entity2, relations)
            for entity1, entity2 in product(entities, repeat=2)
            if entity1 != entity2
        )
    return entities


def generate_positive_examples(
    entities, relations, size=1
) -> List[Tuple[str, str, str]]:
    """
    Generate positive examples from a list of entities.

    :param entities: a list of entities
    :param size: the number of positive examples to be generated

    :returns: a list of triples (entity1, relation, entity2)
    """
    assert size <= math.factorial(len(entities))
    all_positive_examples = []
    for entity1, entity2 in product(entities, entities):
        if entity1 != entity2:
            for rel in relations:
                if rel(entity1, entity2):
                    all_positive_examples.append(
                        (
                            entity1.name,
                            rel.__name__,
                            entity2.name,
                        )
                    )
    np.random.shuffle(all_positive_examples)
    return all_positive_examples[:size]


def generate_one_negative_example(
    entity_names, relation_names, positive_examples, negative_sample_type="relation"
):
    """
    Generate negative examples from a list of objects.

    :param objects: a list of objects
    :param size: the number of positive examples to be generated

    :returns: a list of triples (object1, relation, object2)
    """
    head, relation, tail = random.choice(list(positive_examples))
    if negative_sample_type == "head":
        entity_names = entity_names - {head, tail}
        head = random.choice(list(entity_names))
    elif negative_sample_type == "relation":
        relation_names = relation_names - {relation}
        relation = random.choice(list(relation_names))
    elif negative_sample_type == "tail":
        entity_names = entity_names - {head, tail}
        tail = random.choice(list(entity_names))
    else:
        raise ValueError
    negative_example = (head, relation, tail)
    return negative_example


def generate_negative_examples(
    entities,
    relations,
    positive_examples,
    size=None,
    mixture=Munch(head=1, relation=1, tail=1),
):
    if not size:
        size = len(positive_examples)
    entity_names = {obj.name for obj in entities}
    relation_names = {rel.__name__ for rel in relations}
    negative_examples = set()
    negative_sample_types = list(mixture.keys())
    p = np.array(list(mixture.values()))
    p = p / p.sum()
    while len(negative_examples) < size:
        negative_sample_type = np.random.choice(negative_sample_types, p=p)
        negative_example = generate_one_negative_example(
            entity_names, relation_names, positive_examples, negative_sample_type
        )
        negative_examples.add(negative_example)
    return negative_examples


# Tests
def test_left_of():
    entity1 = Entity(2, 3)
    entity2 = Entity(2, 3, p=(3, 3))
    assert left_of(entity1, entity2)


def test_right_of():
    entity1 = Entity(2, 3)
    entity2 = Entity(2, 3, p=(3, 3))
    assert right_of(entity2, entity1)


def test_entity():
    w = 2
    h = 3
    p = (1, 2)
    theta = np.pi / 2

    entity = Entity(w, h, p=p, theta=theta)

    assert np.allclose(entity.top_left, np.array([-2, 2]))
    assert np.allclose(entity.top_right, np.array([-2, 4]))
    assert np.allclose(entity.bottom_left, np.array([1, 2]))
    assert np.allclose(entity.bottom_right, np.array([1, 4]))
