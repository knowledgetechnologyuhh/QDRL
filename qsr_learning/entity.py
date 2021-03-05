import json
import math
from pathlib import Path

import git
import numpy as np
from munch import Munch
from PIL import Image, ImageDraw
from functools import lru_cache

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


@lru_cache(len(emoji_names))  # cache the outputs as there are many repetitive calls
def load_emoji(emoji_name, size=None):
    """Load the emojis."""
    unicode = emoji_name2unicode[emoji_name]
    image = Image.open(images_path / (unicode + ".png")).convert("RGBA")
    if size:
        image = image.resize(size)
    return image


def image2array(image):
    # alway flip the y-axis when converting between image and array
    return np.array(image)[::-1, :]


def array2image(array):
    # alway flip the y-axis when converting between image and array
    return Image.fromarray(array[::-1, :])


def get_mask(image_array, threshold=None) -> np.array:
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


def get_bbox(image_array):
    """Get the bounding boxes of the mask_array."""
    mask_array = get_mask(image_array)
    x_proj = mask_array.any(axis=0)  # Does the column contain a non-zero entry?
    y_proj = mask_array.any(axis=1)  # Does the row contain a non-zero entry?
    bbox = Munch(
        left=x_proj.argmax(),
        bottom=y_proj.argmax(),
        right=len(x_proj) - 1 - x_proj[::-1].argmax(),
        top=len(y_proj) - 1 - y_proj[::-1].argmax(),
    )
    return bbox


def crop_image(image):
    image_array = image2array(image)
    bbox = get_bbox(image_array)
    image_array = image_array[bbox.bottom : bbox.top + 1, bbox.left : bbox.right + 1]
    image = array2image(image_array)
    x_max, y_max = image_array.shape[1] - 1, image_array.shape[0] - 1
    fbbox = np.array(
        [
            [0, 0],
            [0, y_max],
            [x_max, y_max],
            [x_max, 0],
        ]
    ).astype(float)
    return image, image_array, fbbox


class Entity:
    def __init__(
        self,
        name=None,
        frame_of_reference: str = "absolute",
        p=(0, 0),
        theta=0,
        size=(32, 32),
    ):
        self.name = name
        self.frame_of_reference = frame_of_reference

        self.image = load_emoji(name, size=size)
        self.image, self.image_array, self.flt_bbox = crop_image(self.image)

        self.p = p
        self.theta = theta
        self.center = (self.flt_bbox[2] - self.flt_bbox[0]) / 2 + self.flt_bbox[0]
        self.rotate(self.theta)
        self.translate(self.p)
        self.size = size

    def rotate(self, theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])

        self.image = self.image.rotate(
            360 * theta / (2 * math.pi),
            resample=3,
            expand=True,
        )
        self.image_array = image2array(self.image)
        if self.frame_of_reference == "absolute":
            self.image, self.image_array, self.flt_bbox = crop_image(self.image)
        elif (
            self.frame_of_reference == "intrinsic"
        ):  # objects have intrinsic orientations
            self.flt_bbox = (self.flt_bbox - self.center) @ R.transpose() + self.center
        else:
            raise ValueError("The chosen frame of reference is not defined!")

    def translate(self, p):
        self.flt_bbox += np.array(p)

    def draw(
        self,
        base,
        add_bbox=False,
        bbox_fill=None,
        bbox_outline="white",
        add_front=False,
    ):
        d = ImageDraw.Draw(base)

        # merge the entity image and the base
        base.alpha_composite(
            self.image,
            (self.bbox[:, 0].min(), base.size[1] - self.bbox[:, 1].max()),
        )

        if add_bbox:
            vertices = [
                self.bottom_left,
                self.top_left,
                self.top_right,
                self.bottom_right,
            ]
            d.polygon(
                [(p[0], base.size[1] - p[1]) for p in vertices],
                fill=bbox_fill,
                outline=bbox_outline,
            )

        if add_front:
            # color the top edge with red
            vertices = [self.top_left, self.top_right]
            d.polygon(
                [(p[0], base.size[1] - p[1]) for p in vertices],
                fill=None,
                outline="red",
            )
        return base

    @property
    def bbox(self):
        """Round the oringinal bbox coordinates."""
        return self.flt_bbox.round().astype(int)

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
