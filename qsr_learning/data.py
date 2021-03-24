import math
import random
from collections import namedtuple
from copy import deepcopy
from itertools import combinations
from typing import Callable, List, Tuple

import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset

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


def generate_entities(
    entity_names,
    num_entities: int = 5,
    frame_of_reference: str = "absolute",
    w_range: Tuple[int, int] = (32, 32),
    h_range: Tuple[int, int] = (32, 32),
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
    random.shuffle(entity_names_copy)

    entities_in_canvas = False
    while not entities_in_canvas:
        entities = []
        for name in entity_names_copy[:num_entities]:
            # Rotate and translate the entities.
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
        # Ensure that all entities are inside the canvas
        entities_in_canvas = all(
            inside_canvas(entity, canvas_size) for entity in entities
        )
    return entities


def draw_entities(entities, canvas_size=(224, 224), add_bbox=True, add_front=False):
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 255))
    for entity in entities:
        entity.draw(canvas, add_bbox=add_bbox, add_front=add_front)
    return canvas


def get_mean_and_std(
    entity_names,
    relation_names,
    num_entities,
    w_range,
    h_range,
    add_bbox,
    add_front,
    canvas_size,
):
    """Get the mean and the std of the specific dataset."""
    drl_dataset = DRLDataset(
        entity_names=entity_names,
        relation_names=relation_names,
        num_entities=num_entities,
        frame_of_reference="absolute",
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
        self.entity_names = entity_names
        self.excluded_pairs = set(combinations(excluded_entity_names, 2))
        self.relation_names = set(relation_names)
        self.relations = [
            getattr(qsr_learning.relation, relation_name)
            for relation_name in relation_names
        ]
        self.excluded_relation_names = set(excluded_relation_names)
        self.allowed_relation_names = self.relation_names - self.excluded_relation_names
        self.allowed_relations = [
            getattr(qsr_learning.relation, relation_name)
            for relation_name in self.allowed_relation_names
        ]
        self.num_entities = num_entities
        self.w_range = w_range
        self.h_range = h_range
        self.frame_of_reference = frame_of_reference
        self.theta_range = theta_range
        self.add_bbox = add_bbox
        self.add_front = add_front

        if not transform:
            self.mean, self.std = get_mean_and_std(
                entity_names,
                relation_names,
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
        triple_found = False
        while not triple_found:
            entity_names = random.sample(self.entity_names, self.num_entities)
            head_name, tail_name = random.sample(entity_names, 2)
            pair_excluded = ((head_name, tail_name) in self.excluded_pairs) or (
                (tail_name, head_name) in self.excluded_pairs
            )
            if pair_excluded:
                if self.allowed_relation_names:
                    relation = random.choice(self.allowed_relations)
                    triple_found = True
            else:
                relation = random.choice(self.relations)
                triple_found = True
        answer = random.randint(0, 1)
        sample_found = False
        while not sample_found:
            entities = generate_entities(
                entity_names,
                self.num_entities,
                self.frame_of_reference,
                w_range=self.w_range,
                h_range=self.h_range,
                theta_range=self.theta_range,
                canvas_size=self.canvas_size,
                relations=self.relations,
            )
            for entity in entities:
                if entity.name == head_name:
                    head = entity
                elif entity.name == tail_name:
                    tail = entity
                else:
                    pass
            if any(r(head, tail) for r in [above, below, left_of, right_of]):
                sample_found = relation(head, tail) == answer
        image = draw_entities(
            entities,
            canvas_size=self.canvas_size,
            add_bbox=self.add_bbox,
            add_front=self.add_front,
        )
        background = Image.new("RGBA", image.size, (0, 0, 0))
        image = Image.alpha_composite(background, image).convert("RGB")

        question = Question(head.name, relation.__name__, tail.name)
        return image, question, answer
