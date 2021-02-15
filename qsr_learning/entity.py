import json
import math
from pathlib import Path

import git
import numpy as np
from munch import Munch
from PIL import Image, ImageDraw

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


def load_emoji(emoji_name, size=(32, 32)):
    """Load the emojis."""
    unicode = emoji_name2unicode[emoji_name]
    image = Image.open(images_path / (unicode + ".png")).convert("RGBA").resize(size)
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
    )
    return image, image_array, fbbox


class Entity:
    def __init__(
        self,
        name=None,
        absolute_direction=False,
        p=(0, 0),
        theta=0,
        size=(32, 32),
    ):
        self.name = name
        self.absolute_direction = absolute_direction

        self.image = load_emoji(name, size=size)
        self.image, self.image_array, self.flt_bbox = crop_image(self.image)

        self.p = p
        self.theta = theta
        self.center = (self.flt_bbox[2] - self.flt_bbox[0]) / 2 + self.flt_bbox[0]
        self.rotate(self.theta)
        self.translate(self.p)

    def rotate(self, theta):
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])

        self.image = self.image.rotate(
            360 * theta / (2 * math.pi),
            expand=True,
        )
        self.image_array = image2array(self.image)
        if self.absolute_direction:
            self.image, self.image_array, self.flt_bbox = crop_image(self.image)
        else:  # objects have intrinsic orientations
            self.flt_bbox = (self.flt_bbox - self.center) @ R.transpose() + self.center

    def translate(self, p):
        self.flt_bbox += np.array(p)

    def draw(
        self,
        base,
        show_bbox=False,
        bbox_color=None,
        bbox_border=None,
        orientation_marker=False,
    ):
        d = ImageDraw.Draw(base)

        if show_bbox:
            d.polygon(
                [(p[0], base.size[1] - p[1]) for p in self.bbox],
                fill=bbox_color if bbox_color else None,
                outline=bbox_border if bbox_border else None,
            )

        if orientation_marker:
            # Use the tenth of the bounding box (from the top) for marking the front side of an self.
            bottom_left = (
                ((self.flt_bbox[0] - self.flt_bbox[1]) / 10 + self.flt_bbox[1])
                .round()
                .astype(int)
            )
            bottom_right = (
                ((self.flt_bbox[3] - self.flt_bbox[2]) / 10 + self.flt_bbox[2])
                .round()
                .astype(int)
            )
            vertices = [bottom_left, self.top_left, self.top_right, bottom_right]
            d.polygon(
                [(p[0], base.size[1] - p[1]) for p in vertices],
                fill="white",
            )
        # merge the entity image and the base
        base.alpha_composite(
            self.image,
            (self.bbox[:, 0].min(), base.size[1] - self.bbox[:, 1].max()),
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
