import json
import math
import random
from itertools import product
from pathlib import Path
from typing import List, Tuple

import git
import numpy as np
import torch
from munch import Munch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import trange

repo = git.Repo(Path(".").absolute(), search_parent_directories=True)
ROOT = Path(repo.working_tree_dir)

# Path to the folder where emoji image files are stored
images_path = ROOT / "emoji-images" / "imgs"
# Extract the unicodes from the image file names
image_unicodes = set(p.stem for p in images_path.glob("*.png"))

# Path to the folder where emoji names and unicodes are stored
emojis_path = ROOT / "emojis" / "emojis.json"
# Load the emoji data
with open(emojis_path, "r") as f:
    emojis = json.load(f)["emojis"]

# A dictionary that assigns a name to each unicode with a corresponding image.
unicode2emoji_name = {
    unicode: entry["name"]
    for entry in emojis
    if (unicode := entry["unicode"].replace(" ", "-")) in image_unicodes
    and entry["name"]
}

emoji_name2unicode = {
    unicode2emoji_name[unicode]: unicode for unicode in unicode2emoji_name
}
emoji_names = list(emoji_name2unicode.keys())


def draw(objects, canvas_size=(224, 224)):
    """Draw the objects on a canvas."""
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 255))
    for obj in objects:
        canvas.alpha_composite(obj.image, (obj.bbox.left, obj.bbox.top))
    return canvas


def get_mask(image_array: np.array, threshold=None) -> np.array:
    """Return the mask of the image. The mask is obtained from the alpha channel.

    :param image_array: An array of shape (H x W x 4) representing an RGBA image.
    :param threshold: determines the threshold for deciding whether to assign a pixel
        to foreground or background. 4 is a good value. Higher values might not for
        example get the correct mask for the emoji "cigarette" (unicode: 1f6ac).
    :returns: The mask of the input image.
    """
    if threshold is None:
        threshold = 4
    mask_array = image_array[:, :, 3]
    mask_array[mask_array < threshold] = 0  # background
    mask_array[mask_array >= threshold] = 255  # foreground
    return mask_array.astype(bool)


def get_bbox(mask_array):
    """Get the bounding boxes of the mask_array."""
    x_proj = mask_array.any(axis=0)
    y_proj = mask_array.any(axis=1)
    bbox = Munch(
        left=x_proj.argmax(),
        top=y_proj.argmax(),
        right=len(x_proj) - 1 - x_proj[::-1].argmax(),
        bottom=len(y_proj) - 1 - y_proj[::-1].argmax(),
    )
    return bbox


def add_bbox(obj):
    """Add bounding box to the object."""
    obj.array[:, 0, :3] = 255
    obj.array[:, -1, :3] = 255
    obj.array[0, :, :3] = 255
    obj.array[-1, :, :3] = 255
    obj.array[:, 0, 3] = 255
    obj.array[:, -1, 3] = 255
    obj.array[0, :, 3] = 255
    obj.array[-1, :, 3] = 255
    obj.image = Image.fromarray(obj.array)


def crop_image(obj):
    """Crop the image of an object and updates affected attributes."""
    bbox = get_bbox(obj.mask)
    obj.array = obj.array[bbox.top : bbox.bottom + 1, bbox.left : bbox.right + 1]
    obj.image = Image.fromarray(obj.array)
    obj.mask = get_mask(obj.array)


def load_emoji(emoji_name, size=(64, 64)):
    """Load the emojis."""
    unicode = emoji_name2unicode[emoji_name]
    image = Image.open(images_path / (unicode + ".png")).convert("RGBA").resize(size)
    return image


def generate_objects(
    names=["hot dog", "pizza"],
    positions=[(0, 0), (0, 32)],
    rotations=[30, 45],
    image_size=(64, 64),
    crop=False,
    bbox=False,
):
    """Generate objects."""
    objects = []
    for name, position, rotation in zip(names, positions, rotations):
        obj = Munch()
        obj.name = name
        obj.rotation = rotation
        obj.image = load_emoji(obj.name, image_size).rotate(obj.rotation)
        obj.array = np.array(obj.image)
        obj.mask = get_mask(obj.array)
        obj.area = obj.mask.sum()
        if crop:
            crop_image(obj)
        obj.bbox = Munch(
            top=position[0],
            left=position[1],
            bottom=position[0] + obj.array.shape[0] - 1,
            right=position[1] + obj.array.shape[1] - 1,
        )
        if bbox:
            add_bbox(obj)
        objects.append(obj)
    return objects


def sample_objects(
    relations,
    canvas_size=(224, 224),
    num_objects=2,
    rotations=None,
    image_size=(32, 32),
):
    if not rotations:
        rotations = [0] * num_objects

    # Sample scenes until there a relation holds between any pair of objects.
    resample = True
    while resample:
        positions = list(
            zip(
                np.random.randint(0, canvas_size[0] - image_size[0], (num_objects,)),
                np.random.randint(0, canvas_size[1] - image_size[1], (num_objects,)),
            )
        )
        names = np.random.choice(emoji_names, size=(num_objects,), replace=False)
        objects = generate_objects(
            names=names,
            positions=positions,
            rotations=rotations,
            image_size=image_size,
            crop=False,
            bbox=False,
        )
        resample = False
        for object1, object2 in product(objects, objects):
            if object1 != object2 and all(
                [not relation(object1, object2) for relation in relations]
            ):
                resample = True
                break
    return objects


def generate_positive_examples(
    objects, relations, size=1
) -> List[Tuple[str, str, str]]:
    """
    Generate positive examples from a list of objects.

    :param objects: a list of objects
    :param size: the number of positive examples to be generated

    :returns: a list of triples (object1, relation, object2)
    """
    assert size <= math.factorial(len(objects))
    all_positive_examples = []
    for object1, object2 in product(objects, objects):
        if object1 != object2:
            for rel in relations:
                if rel(object1, object2):
                    all_positive_examples.append(
                        (
                            object1.name,
                            rel.__name__,
                            object2.name,
                        )
                    )
    np.random.shuffle(all_positive_examples)
    return all_positive_examples[:size]


def generate_one_negative_example(
    object_names, relation_names, positive_examples, negative_sample_type="relation"
):
    """
    Generate negative examples from a list of objects.

    :param objects: a list of objects
    :param size: the number of positive examples to be generated

    :returns: a list of triples (object1, relation, object2)
    """
    head, relation, tail = random.choice(list(positive_examples))
    if negative_sample_type == "head":
        object_names = object_names - {head, tail}
        head = random.choice(list(object_names))
    elif negative_sample_type == "relation":
        relation_names = relation_names - {relation}
        relation = random.choice(list(relation_names))
    elif negative_sample_type == "tail":
        object_names = object_names - {head, tail}
        tail = random.choice(list(object_names))
    else:
        raise ValueError
    negative_example = (head, relation, tail)
    return negative_example


def generate_negative_examples(
    objects,
    relations,
    positive_examples,
    size=None,
    mixture=Munch(head=1, relation=1, tail=1),
):
    if not size:
        size = len(positive_examples)
    object_names = {obj.name for obj in objects}
    relation_names = {rel.__name__ for rel in relations}
    negative_examples = set()
    negative_sample_types = list(mixture.keys())
    p = np.array(list(mixture.values()))
    p = p / p.sum()
    while len(negative_examples) < size:
        negative_sample_type = np.random.choice(negative_sample_types, p=p)
        negative_example = generate_one_negative_example(
            object_names, relation_names, positive_examples, negative_sample_type
        )
        negative_examples.add(negative_example)
    return negative_examples


class QSRData(Dataset):
    def __init__(
        self,
        relations,
        num_images=4,
        num_objects=2,
        num_pos_questions_per_image=2,
        num_neg_questions_per_image=2,
        mixture=Munch(head=1, relation=1, tail=1),
    ):
        self.images = []
        self.questions = []
        self.answers = []
        for i in trange(num_images):
            objects = sample_objects(relations, num_objects=num_objects)
            positive_examples = generate_positive_examples(
                objects, relations, num_pos_questions_per_image
            )
            negative_examples = generate_negative_examples(
                objects,
                relations,
                positive_examples,
                num_neg_questions_per_image,
                mixture,
            )
            self.images.append(np.array(draw(objects))[:, :, :3])
            for question in positive_examples:
                self.questions.append({"id": i, "question": question})
                self.answers.append({"id": i, "answer": True})
            for question in negative_examples:
                self.questions.append({"id": i, "question": question})
                self.answers.append({"id": i, "answer": False})
        super().__init__()
        relation_names = {"left_of", "right_of", "above", "below"}
        self.idx2word = dict(enumerate(sorted(set(emoji_names) | relation_names)))
        self.word2idx = {self.idx2word[idx]: idx for idx in self.idx2word}
        rgb_mean = 0.5
        rgb_std = 0.5
        self._transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(rgb_mean, rgb_std)]
        )

    def __getitem__(self, idx):
        image_id = self.questions[idx]["id"]
        return (
            self._transform(self.images[image_id]),
            torch.tensor(
                [self.word2idx[word] for word in self.questions[idx]["question"]]
            ),
            self.answers[idx]["answer"],
        )

    def __len__(self):
        # num_images * num_pos_questions_per_image * num_neg_questions_per_image
        return len(self.questions)
