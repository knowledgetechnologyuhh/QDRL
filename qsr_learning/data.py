import math
import random
from collections import namedtuple
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

Question = namedtuple("Question", ["head", "relation", "tail"])


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

    :returns: a list of `Question` instances `(head, relation, tail)`
    """
    # Generate positive examples
    positive_questions = []
    negative_questions = []
    for head, tail in product(entities, entities):
        if head != tail:
            for relation in relations:
                if relation(head, tail):
                    positive_questions.append(
                        Question(
                            head.name,
                            relation.__name__,
                            tail.name,
                        )
                    )
                else:
                    negative_questions.append(
                        Question(
                            head.name,
                            relation.__name__,
                            tail.name,
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
    w_range,
    h_range,
):
    """Get the mean and the std of the specific dataset."""
    drl_dataset = DRLDataset(
        entity_names=entity_names,
        relation_names=relation_names,
        num_entities=num_entities,
        fixed_entities=None,
        frame_of_reference="absolute",
        w_range=w_range,
        h_range=h_range,
        theta_range=(0, 0),
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
        fixed_entities,
        frame_of_reference,
        w_range,
        h_range,
        theta_range,
        num_samples,
        filter_fn=None,
        show_bbox=False,
        orientation_marker=False,
        transform=None,
        num_questions_per_image=1,
        random_seed=0,
    ):
        """
        :param num_questions_per_image: the (maximal) number of questions generated for each image.
        """
        super().__init__()
        self.entity_names = entity_names
        self.relations = [
            getattr(qsr_learning.relation, relation_name)
            for relation_name in relation_names
        ]
        self.num_entities = num_entities
        self.fixed_entities = fixed_entities  # predefined entites
        self.w_range = (32, 32)
        self.h_range = (32, 32)
        self.frame_of_reference = frame_of_reference
        self.theta_range = (0, 2 * math.pi)
        self.num_samples = num_samples
        self.filter_fn = filter_fn
        self.show_bbox = show_bbox
        self.orientation_marker = orientation_marker

        if not transform:
            self.mean, self.std = get_mean_and_std(
                entity_names,
                relation_names,
                num_entities,
                w_range,
                h_range,
            )
            self.transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(self.mean, self.std),
                ]
            )
        else:
            self.transform = transform

        self.num_questions_per_image = num_questions_per_image
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

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
            return iter(
                self.generate_sample(
                    worker_id=0,
                    num_samples=self.num_samples,
                    fixed_entities=self.fixed_entities,
                    filter_fn=self.filter_fn,
                )
            )
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
                    fixed_entities=self.fixed_entities,
                    filter_fn=self.filter_fn,
                )
            )

    def __len__(self):
        return self.num_samples

    def generate_sample(
        self, worker_id, num_samples, fixed_entities=None, filter_fn=None
    ):
        for i in range(num_samples):
            if not self.questions.get(worker_id, []):
                image, questions, answers = self.generate_scene(
                    fixed_entities=fixed_entities, filter_fn=filter_fn
                )
                qa_pairs = list(zip(questions, answers))
                random.shuffle(qa_pairs)
                questions, answers = zip(*qa_pairs)
                self.image[worker_id] = image
                self.questions[worker_id] = list(questions)[
                    : self.num_questions_per_image
                ]
                self.answers[worker_id] = list(answers)[: self.num_questions_per_image]
            yield (
                self.transform(self.image[worker_id]),
                torch.tensor(self.questions[worker_id].pop()),
                torch.tensor(self.answers[worker_id].pop()),
            )

    def generate_scene(self, fixed_entities=None, filter_fn=None):
        """Generate a scene and all questions with their answers.

        :param fixed_entities: entities that are predetermined and not randomly generated.
        """
        entities = generate_entities(
            entity_names=self.entity_names,
            num_entities=self.num_entities,
            frame_of_reference=self.frame_of_reference,
            w_range=self.w_range,
            h_range=self.h_range,
            theta_range=self.theta_range,
        )

        if fixed_entities:
            fixed_entity_names = [entity.name for entity in fixed_entities]
            entities = [
                entity for entity in entities if entity.name not in fixed_entity_names
            ]
            entities.extend(fixed_entities)

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

        if filter_fn:
            positive_questions = list(filter(filter_fn, positive_questions))
            negative_questions = list(filter(filter_fn, negative_questions))

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
