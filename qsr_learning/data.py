import math
import random
from copy import deepcopy
from itertools import product
from typing import Callable, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from qsr_learning.entity import Entity
from qsr_learning.relation import above, below, left_of, right_of


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
    entity_names,
    num_entities: int = 5,
    frame_of_reference: str = "absolute",
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

    entity_names_copy = deepcopy(entity_names)
    np.random.shuffle(entity_names_copy)

    # Rotate and translate the entities.
    entities_in_canvas = entities_in_relation = False
    while not (entities_in_canvas and entities_in_relation):
        entities = []
        for name in entity_names_copy[:num_entities]:
            theta = random.uniform(0.0, 2 * math.pi)
            p = (random.uniform(0, canvas_size[0]), random.uniform(0, canvas_size[1]))
            entity = Entity(
                name=name,
                frame_of_reference=frame_of_reference,
                p=p,
                theta=theta,
                size=(random.randint(*w_range), (random.randint(*h_range))),
            )
            entities.append(entity)
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


def generate_questions(entities, relations, size=None):
    """
    Generate positive examples from a list of entities.

    :param entities: a list of entities
    :param size: the number of examples to be generated

    :returns: a list of triples (entity1, relation, entity2)
    """
    # Generate positive examples
    positive_questions = []
    negative_questions = []
    for entity1, entity2 in product(entities, entities):
        if entity1 != entity2:
            for rel in relations:
                if rel(entity1, entity2):
                    positive_questions.append(
                        (
                            entity1.name,
                            rel.__name__,
                            entity2.name,
                        )
                    )
                else:
                    negative_questions.append(
                        (
                            entity1.name,
                            rel.__name__,
                            entity2.name,
                        )
                    )
    return positive_questions, negative_questions


def draw_entities(
    entities, canvas_size=(224, 224), show_bbox=True, orientation_marker=False
):
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 255))
    d = ImageDraw.Draw(canvas)
    d.polygon(
        [
            (0, 0),
            (0, canvas_size[1] - 1),
            (canvas_size[0] - 1, canvas_size[1] - 1),
            (canvas_size[0] - 1, 0),
        ],
        fill=None,
        outline="white",
    )
    for entity in entities:
        entity.draw(canvas, show_bbox=show_bbox, orientation_marker=orientation_marker)
    return canvas
