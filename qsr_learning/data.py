import math
import random
from copy import deepcopy
from itertools import product
from typing import Callable, Dict, List, Tuple

import numpy as np
import PIL
import torch
import torchvision
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, IterableDataset

import qsr_learning
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
    theta_range: Tuple[float, float] = (0.0, 2 * math.pi),
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
            theta = random.uniform(*theta_range)
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


# from qsr_learning.relation import above, below, left_of, right_of

# type aliases
Questions = List[Tuple[int, int, int]]
Answers = List[int]


def get_mean_and_std(
    entity_names,
    relation_names,
    num_entities,
    frame_of_reference,
    w_range,
    h_range,
    theta_range,
):
    """Get the mean and the std of the specific dataset."""
    drl_dataset = DRLDataset(
        entity_names=entity_names,
        relation_names=relation_names,
        num_entities=num_entities,
        frame_of_reference="absolute",
        w_range=(32, 32),
        h_range=(32, 32),
        theta_range=2 * math.pi,
        num_samples=len(entity_names) * num_entities,  # This is only a rough estimate
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
    )
    loader = DataLoader(drl_dataset, batch_size=8)
    channel_values = (
        torch.stack([img[0] for img, _, _ in loader], dim=0)
        .permute(1, 0, 2, 3)
        .reshape(3, -1)
    )
    return channel_values.mean(-1), channel_values.std(-1)


class DRLDataset(IterableDataset):
    def __init__(
        self,
        entity_names,
        relation_names,
        num_entities,
        frame_of_reference,
        w_range,
        h_range,
        theta_range,
        num_samples,
        show_bbox=False,
        orientation_marker=False,
        transform=None,
        shuffle=False,
    ):
        super().__init__()
        self.entity_names = entity_names
        self.relations = [
            getattr(qsr_learning.relation, relation_name)
            for relation_name in relation_names
        ]
        self.num_entities = num_entities
        self.w_range = (32, 32)
        self.h_range = (32, 32)
        self.frame_of_reference = frame_of_reference
        self.theta_range = (0, 2 * math.pi)
        self.num_samples = num_samples
        self.show_bbox = show_bbox
        self.orientation_marker = orientation_marker

        if not transform:
            self.mean, self.std = get_mean_and_std(
                entity_names,
                relation_names,
                num_entities,
                frame_of_reference,
                w_range,
                h_range,
                theta_range,
            )
            self.transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(self.mean, self.std),
                ]
            )
        else:
            self.transform = transform

        self.shuffle = shuffle

        self.idx2ent, self.ent2idx = {}, {}
        for idx, entity_name in enumerate(sorted(entity_names)):
            self.idx2ent[idx] = entity_name
            self.ent2idx[entity_name] = idx

        self.idx2rel, self.rel2idx = {}, {}
        for idx, relation_name in enumerate(sorted(relation_names)):
            self.idx2rel[idx] = relation_name
            self.rel2idx[relation_name] = idx

        self.image: Dict[int, PIL.Image] = {}
        self.questions: Dict[int, Questions] = {}
        self.answers: Dict[int, Answers] = {}

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            return iter(self.generate_sample(worker_id=0, num_samples=self.num_samples))
        else:
            worker_id = worker_info.id
            per_worker = math.ceil(self.num_samples / float(worker_info.num_workers))
            return iter(
                self.generate_sample(
                    worker_id=worker_id,
                    num_samples=int(
                        min((worker_id + 1) * per_worker, self.num_samples)
                        - worker_id * per_worker
                    ),
                )
            )

    def __len__(self):
        return self.num_samples

    def generate_sample(self, worker_id, num_samples):
        for i in range(num_samples):
            random.seed(i)
            np.random.seed(i)
            torch.manual_seed(i)
            if not self.questions.get(worker_id, []):
                image, questions, answers = self.generate_scene()
                qa_pairs = list(zip(questions, answers))
                random.shuffle(qa_pairs)
                questions, answers = zip(*qa_pairs)
                self.image[worker_id] = image
                if self.shuffle:
                    # For each generated image retrieve only one question-answer
                    # pair from the shuffled pairs.
                    self.questions[worker_id] = list(questions)[:1]
                    self.answers[worker_id] = list(answers)[:1]
                else:
                    self.questions[worker_id] = list(questions)
                    self.answers[worker_id] = list(answers)
            yield (
                self.transform(self.image[worker_id]),
                torch.tensor(self.questions[worker_id].pop()),
                torch.tensor(self.answers[worker_id].pop()),
            )

    def generate_scene(self):
        """Generate a scene and all questions with their answers."""
        entities = generate_entities(
            entity_names=self.entity_names,
            num_entities=self.num_entities,
            frame_of_reference=self.frame_of_reference,
            w_range=self.w_range,
            h_range=self.h_range,
            theta_range=self.theta_range,
        )

        image = draw_entities(
            entities,
            show_bbox=self.show_bbox,
            orientation_marker=self.orientation_marker,
        )
        background = Image.new("RGBA", image.size, (0, 0, 0))
        image = Image.alpha_composite(background, image).convert("RGB")

        positive_questions, negative_questions = generate_questions(
            entities, self.relations
        )

        def question2idx_tuple(question):
            entity1, relation, entity2 = question
            return (
                self.ent2idx[entity1],
                self.rel2idx[relation],
                self.ent2idx[entity2],
            )

        questions = [
            question2idx_tuple(question) for question in positive_questions
        ] + [question2idx_tuple(question) for question in negative_questions]

        answers = [1] * len(positive_questions) + [0] * len(negative_questions)
        return image, questions, answers
