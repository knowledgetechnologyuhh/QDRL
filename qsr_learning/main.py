"""This in the main file."""

import json
import math
from itertools import product
from pathlib import Path

import git
import numpy as np
from munch import Munch
from PIL import Image

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


def overlap(object1, object2, canvas_size=(224, 224), threshold=0.05):
    """
    Compute the overlap of objects.

    :param threshold: the fraction of the areas of any of the two objects that needs
        to be overlapped by the other object.
    """
    canvases = []
    for obj in [object1, object2]:
        canvas = np.zeros(canvas_size, dtype=bool)
        canvas[
            obj.bbox.top : obj.bbox.top + obj.mask.shape[0],
            obj.bbox.left : obj.bbox.left + obj.mask.shape[1],
        ] = obj.mask
        canvases.append(canvas)
    intersection_area = (canvases[0] & canvases[1]).sum()
    return (
        intersection_area / object1.area > threshold
        or intersection_area / object2.area > threshold
    )


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


def load_emoji(emoji_name, size=(64, 64)):
    """Load the emojis."""
    unicode = emoji_name2unicode[emoji_name]
    image = Image.open(images_path / (unicode + ".png")).convert("RGBA").resize(size)
    return image


def crop_image(obj):
    """Crop the image of an object and updates affected attributes."""
    bbox = get_bbox(obj.mask)
    obj.array = obj.array[bbox.top : bbox.bottom + 1, bbox.left : bbox.right + 1]
    obj.image = Image.fromarray(obj.array)
    obj.mask = get_mask(obj.array)


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


# TODO: Explain the logic behind the two functions.
def point_left_of_directed_line(point, directed_line):
    p = point
    q = directed_line.point
    d = directed_line.vector
    d_rot = np.array([-d[1], d[0]])  # d rotated by 90 degree
    # NOTE: As the y-axis is upside down, the sign is also switched.
    return (p - q) @ d_rot < 0


def point_right_of_directed_line(point, directed_line):
    p = point
    q = directed_line.point
    d = directed_line.vector
    d_rot = np.array([-d[1], d[0]])  # d rotated by 90 degree
    # NOTE: As the y-axis is upside down, the sign is also switched.
    return (p - q) @ d_rot > 0


def test_point_left_of_directed_line():
    point = np.array([1, 1])
    directed_line = Munch(point=np.array([2, -1]), vector=np.array([1, 1]))
    assert point_right_of_directed_line(point, directed_line)
    directed_line = Munch(point=np.array([2, -1]), vector=np.array([-1, 1]))
    assert not point_right_of_directed_line(point, directed_line)


# TODO: Explain the meaning of the variables.
def left_of(object1, object2):
    point_top_right = np.array([object1.bbox.right, object1.bbox.top])
    point_bottom_right = np.array([object1.bbox.right, object1.bbox.bottom])
    directed_line_right = Munch(
        point=np.array([object2.bbox.left, object2.bbox.top]), vector=np.array([-1, -1])
    )
    directed_line_left = Munch(
        point=np.array([object2.bbox.left, object2.bbox.bottom]),
        vector=np.array([-1, 1]),
    )
    constraint1 = point_left_of_directed_line(point_top_right, directed_line_right)
    constraint2 = point_right_of_directed_line(point_bottom_right, directed_line_left)
    # The latter constraint is needed for some boundary cases, where objec1 is much larger than object2.
    return (not overlap(object1, object2)) and (constraint1 and constraint2)


def right_of(object1, object2):
    point_top_left = np.array([object1.bbox.left, object1.bbox.top])
    point_bottom_left = np.array([object1.bbox.left, object1.bbox.bottom])
    directed_line_right = Munch(
        point=np.array([object2.bbox.right, object2.bbox.top]),
        vector=np.array([1, 1]),
    )
    directed_line_left = Munch(
        point=np.array([object2.bbox.right, object2.bbox.bottom]),
        vector=np.array([1, -1]),
    )
    constraint1 = point_left_of_directed_line(point_top_left, directed_line_right)
    constraint2 = point_right_of_directed_line(point_bottom_left, directed_line_left)
    return (not overlap(object1, object2)) and (constraint1 and constraint2)


def above(object1, object2):
    point_bottom_right = np.array([object1.bbox.right, object1.bbox.bottom])
    point_bottom_left = np.array([object1.bbox.left, object1.bbox.bottom])
    directed_line_right = Munch(
        point=np.array([object2.bbox.right, object2.bbox.top]),
        vector=np.array([1, -1]),
    )
    directed_line_left = Munch(
        point=np.array([object2.bbox.left, object2.bbox.top]),
        vector=np.array([-1, -1]),
    )
    constraint1 = point_left_of_directed_line(point_bottom_right, directed_line_right)
    constraint2 = point_right_of_directed_line(point_bottom_left, directed_line_left)
    return (not overlap(object1, object2)) and (constraint1 and constraint2)


def below(object1, object2):
    point_top_right = np.array([object1.bbox.right, object1.bbox.top])
    point_top_left = np.array([object1.bbox.left, object1.bbox.top])
    directed_line_right = Munch(
        point=np.array([object2.bbox.left, object2.bbox.bottom]),
        vector=np.array([-1, 1]),
    )
    directed_line_left = Munch(
        point=np.array([object2.bbox.right, object2.bbox.bottom]),
        vector=np.array([1, 1]),
    )
    constraint1 = point_left_of_directed_line(point_top_left, directed_line_right)
    constraint2 = point_right_of_directed_line(point_top_right, directed_line_left)
    return (not overlap(object1, object2)) and (constraint1 and constraint2)


def generate_positive_examples(objects, size=1):
    relations = [left_of, right_of, above, below]
    assert size <= math.factorial(len(objects))
    all_positive_examples = []
    for object1, object2 in product(objects, objects):
        if object1 != object2:
            for rel in [left_of, right_of, above, below]:
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


def generate_one_negative_example(object_names, relation_names, positive_examples):
    negative_example_found = False
    while not negative_example_found:
        head, tail = np.random.choice(object_names, size=(2,), replace=False)
        relation = np.random.choice(relation_names)
        negative_example_candidate = (head, relation, tail)
        if negative_example_candidate not in positive_examples:
            negative_example = negative_example_candidate
            negative_example_found = True
    return negative_example


def generate_negative_examples(objects, positive_examples, size=None):
    relations = [left_of, right_of, above, below]
    if not size:
        size = len(positive_examples)
    object_names = [obj.name for obj in objects]
    relation_names = [rel.__name__ for rel in [left_of, right_of, above, below]]
    negative_examples = set()
    while len(negative_examples) < size:
        negative_examples.add(
            generate_one_negative_example(
                object_names, relation_names, positive_examples
            )
        )
    return negative_examples


def sample_objects(
    canvas_size=(224, 224), num_objects=2, rotations=None, image_size=(32, 32)
):
    if not rotations:
        rotations = [0] * num_objects
    no_overlap = False
    while not no_overlap:
        positions = list(
            zip(
                np.random.randint(0, canvas_size[0] - image_size[0], (num_objects,)),
                np.random.randint(0, canvas_size[1] - image_size[1], (num_objects,)),
            )
        )
        names = np.random.choice(emoji_names, size=(num_objects,), replace=False)
        #     print(names)
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
            if object1 != object2 and (
                not (
                    left_of(object1, object2)
                    or right_of(object1, object2)
                    or above(object1, object2)
                    or below(object1, object2)
                )
            ):
                resample = True
                break
        if not resample:
            no_overlap = True
    return objects
