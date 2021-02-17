import math
import random
from copy import deepcopy
from itertools import product
from typing import Callable, List, Tuple

import numpy as np
from munch import Munch
from entity import Entity
from relation import left_of, right_of, above, below


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