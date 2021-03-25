import math
import random
from collections import namedtuple
from copy import deepcopy
from itertools import product
from typing import Callable, List, Tuple, Union

import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import qsr_learning
from qsr_learning import binary_relation, ternary_relation
from qsr_learning.entity import Entity

BinaryQuestion = namedtuple("BinaryQuestion", ["entity1", "relation", "entity2"])
TernaryQuestion = namedtuple(
    "TernaryQuestion", ["entity1", "relation", "entity2", "as_seen_from", "entity3"]
)

binary_relations = [
    binary_relation.above,
    binary_relation.below,
    binary_relation.left_of,
    binary_relation.right_of,
]

ternary_relations = [
    ternary_relation.left_of,
    ternary_relation.right_of,
    ternary_relation.in_front_of,
    ternary_relation.behind,
]


def inside_canvas(entity: Entity, canvas_size: Tuple[int, int]) -> bool:
    """Check whether entity is in the canvas."""
    xs_inside_canvas = all(
        (0 < entity.bbox[:, 0]) & (entity.bbox[:, 0] < canvas_size[0])
    )
    ys_inside_canvas = all(
        (0 < entity.bbox[:, 1]) & (entity.bbox[:, 1] < canvas_size[1])
    )
    return xs_inside_canvas and ys_inside_canvas


# def generate_entities(
#     entity_names,
#     frame_of_reference: str = "absolute",
#     w_range: Tuple[int, int] = (32, 32),
#     h_range: Tuple[int, int] = (32, 32),
#     theta_range: Tuple[float, float] = (0.0, 2 * math.pi),
#     canvas_size: Tuple[int, int] = (224, 224),
#     relations: List[Callable[[Entity, Entity], bool]] = [
#         binary_relation.left_of,
#         binary_relation.right_of,
#         binary_relation.above,
#         binary_relation.below,
#     ],
#     entity_names_tuple: Union[Tuple[str, str], Tuple[str, str, str]] = None,
# ) -> List[Entity]:
#     """
#     Given a

#     :param canvas_size: (width, height)
#     """
#     # Shuffle the entity names, but make sure that the target entities come
#     # last, so that they are not potentially occuluded.
#     entity_names_copy = list(set(entity_names) - set(list(entity_names_tuple)))
#     random.shuffle(entity_names_copy)
#     entity_names_copy = entity_names_copy + list(entity_names_tuple)
#     entities_in_canvas = False
#     while not entities_in_canvas:
#         entities = []
#         for name in entity_names_copy:
#             # Rotate and translate the entities.
#             theta = random.uniform(*theta_range)
#             p = (random.uniform(0, canvas_size[0]), random.uniform(0, canvas_size[1]))
#             entity = Entity(
#                 name=name,
#                 frame_of_reference=frame_of_reference,
#                 p=p,
#                 theta=theta,
#                 size=(random.randint(*w_range), (random.randint(*h_range))),
#             )
#             entities.append(entity)
#         # Ensure that all entities are inside the canvas
#         entities_in_canvas = all(
#             inside_canvas(entity, canvas_size) for entity in entities
#         )
#     return entities


def generate_entities(
    entity_names,
    frame_of_reference: str = "absolute",
    w_range: Tuple[int, int] = (32, 32),
    h_range: Tuple[int, int] = (32, 32),
    theta_range: Tuple[float, float] = (0.0, 2 * math.pi),
    canvas_size: Tuple[int, int] = (224, 224),
    relations: List[Callable[[Entity, Entity], bool]] = [
        binary_relation.left_of,
        binary_relation.right_of,
        binary_relation.above,
        binary_relation.below,
    ],
    entity_names_tuple: Union[Tuple[str, str], Tuple[str, str, str]] = None,
) -> List[Entity]:
    """
    Given a

    :param canvas_size: (width, height)
    """
    # Shuffle the entity names, but make sure that the target entities come
    # last, so that they are not potentially occuluded.
    entity_names_copy = deepcopy(entity_names)
    random.shuffle(entity_names_copy)
    sampled_entities = []
    for entity_name in entity_names_copy:
        entity_is_valid = False
        while not entity_is_valid:
            # Rotate and translate the entities.
            theta = random.uniform(*theta_range)
            p = (random.uniform(0, canvas_size[0]), random.uniform(0, canvas_size[1]))
            entity = Entity(
                name=entity_name,
                frame_of_reference=frame_of_reference,
                p=p,
                theta=theta,
                size=(random.randint(*w_range), (random.randint(*h_range))),
            )
            entity_in_canvas = inside_canvas(entity, canvas_size)
            if entity_in_canvas:
                entity_is_valid = all(
                    any(r(entity, e) for r in binary_relations)
                    for e in sampled_entities
                )
        sampled_entities.append(entity)
    return sampled_entities


def draw_entities(entities, canvas_size=(224, 224), add_bbox=True, add_front=False):
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 255))
    for entity in entities:
        entity.draw(canvas, add_bbox=add_bbox, add_front=add_front)
    return canvas


def get_mean_and_std(
    vocab,
    entity_names,
    relation_names,
    frame_of_reference,
    num_entities,
    w_range,
    h_range,
    add_bbox,
    add_front,
    canvas_size,
):
    """Get the mean and the std of the specific dataset."""
    drl_dataset = DRLDataset(
        vocab=vocab,
        entity_names=entity_names,
        relation_names=relation_names,
        num_entities=num_entities,
        frame_of_reference=frame_of_reference,
        w_range=w_range,
        h_range=h_range,
        theta_range=(0, 0),
        add_bbox=add_bbox,
        add_front=add_front,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
        canvas_size=canvas_size,
        num_samples=len(entity_names),  # This is only a rough estimate
    )
    loader = DataLoader(drl_dataset, batch_size=8)
    channel_values = (
        torch.stack([img[0] for img, _, _ in loader], dim=0)
        .permute(1, 0, 2, 3)
        .reshape(3, -1)
    )
    return channel_values.mean(-1), channel_values.std(-1)


class DRLDataset(Dataset):
    def __init__(
        self,
        vocab,
        entity_names=["octopus", "trophy"],
        excluded_entity_names=[],
        relation_names=["above", "below", "left_of", "right_of"],
        excluded_relation_names=[],
        num_entities=2,
        frame_of_reference="absolute",
        w_range=(32, 32),
        h_range=(32, 32),
        theta_range=(0, 2 * math.pi),
        add_bbox=False,
        add_front=False,
        transform=None,
        canvas_size=(224, 224),
        num_samples=128,
        root_seed=0,
    ):
        super().__init__()
        self.relation_names = set(relation_names)
        self.excluded_relation_names = set(excluded_relation_names)
        self.allowed_relation_names = self.relation_names - self.excluded_relation_names
        if frame_of_reference == "absolute" or frame_of_reference == "intrinsic":
            self.arity = 2
            self.relations = [
                getattr(qsr_learning.binary_relation, relation_name)
                for relation_name in relation_names
            ]
            self.allowed_relations = [
                getattr(qsr_learning.binary_relation, relation_name)
                for relation_name in self.allowed_relation_names
            ]
        elif frame_of_reference == "relative":
            if num_entities < 3:
                raise ValueError(
                    "num_entities must be > 3 for relative frame of reference!"
                )
            self.arity = 3
            self.relations = [
                getattr(qsr_learning.ternary_relation, relation_name)
                for relation_name in relation_names
            ]
            self.allowed_relations = [
                getattr(qsr_learning.ternary_relation, relation_name)
                for relation_name in self.allowed_relation_names
            ]
            vocab = vocab + ["as_seen_from"]
        else:
            raise ValueError("the frame of reference does not exist!")
        self.entity_names = entity_names
        self.excluded_tuples = set(product(excluded_entity_names, repeat=self.arity))
        self.num_entities = num_entities
        self.w_range = w_range
        self.h_range = h_range
        self.frame_of_reference = frame_of_reference
        self.theta_range = theta_range
        self.add_bbox = add_bbox
        self.add_front = add_front

        if not transform:
            self.mean, self.std = get_mean_and_std(
                vocab,
                entity_names,
                relation_names,
                frame_of_reference,
                num_entities,
                w_range,
                h_range,
                add_bbox,
                add_front,
                canvas_size,
            )
            self.transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(self.mean, self.std),
                ]
            )
        else:
            self.transform = transform
        self.canvas_size = canvas_size
        self.num_samples = num_samples

        self.idx2word, self.word2idx = {}, {}
        for idx, word in enumerate(sorted(vocab)):
            self.idx2word[idx] = word
            self.word2idx[word] = idx

        self.root_seed = root_seed

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("sample index out of range")
        image, question, answer = self.gen_sample(idx)
        return (
            self.transform(image),
            torch.tensor([self.word2idx[w] for w in question], dtype=torch.long),
            torch.tensor(answer, dtype=torch.float),
        )

    def __len__(self):
        return self.num_samples

    def gen_sample(self, idx):
        random.seed(self.root_seed + idx)
        question_found = False
        while not question_found:
            entity_names = random.sample(self.entity_names, self.num_entities)
            entity_names_tuple = tuple(random.sample(entity_names, self.arity))
            if entity_names_tuple in self.excluded_tuples:
                if self.allowed_relation_names:
                    relation = random.choice(self.allowed_relations)
                    question_found = True
            else:
                relation = random.choice(self.relations)
                question_found = True
        if self.arity == 2:
            question = BinaryQuestion(
                entity_names_tuple[0], relation.__name__, entity_names_tuple[1]
            )
        elif self.arity == 3:
            question = TernaryQuestion(
                entity_names_tuple[0],
                relation.__name__,
                entity_names_tuple[1],
                "as_seen_from",
                entity_names_tuple[2],
            )
        answer = random.randint(0, 1)
        sample_found = False
        while not sample_found:
            entities = generate_entities(
                entity_names,
                self.frame_of_reference,
                w_range=self.w_range,
                h_range=self.h_range,
                theta_range=self.theta_range,
                canvas_size=self.canvas_size,
                relations=self.relations,
                entity_names_tuple=entity_names_tuple,
            )
            entity_tuple = [0] * self.arity
            for entity in entities:
                for i in range(self.arity):
                    if entity.name == entity_names_tuple[i]:
                        entity_tuple[i] = entity
            # Ensure that the a relations holds between entities in the tuple
            if any(
                r(*entity_tuple)
                for r in (binary_relations if self.arity == 2 else ternary_relations)
            ):
                sample_found = relation(*entity_tuple) == answer
        image = draw_entities(
            entities,
            canvas_size=self.canvas_size,
            add_bbox=self.add_bbox,
            add_front=self.add_front,
        )
        background = Image.new("RGBA", image.size, (0, 0, 0))
        image = Image.alpha_composite(background, image).convert("RGB")
        return image, question, answer
