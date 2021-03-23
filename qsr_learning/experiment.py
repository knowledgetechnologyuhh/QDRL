import math
from pathlib import Path

import git
import pytorch_lightning as pl
from git.exc import RepositoryDirtyError
from munch import Munch
from pytorch_lightning import loggers
from torch.utils.data import DataLoader

from qsr_learning.data import DRLDataset
from qsr_learning.models import DRLNet

entity_names = [
    "books",
    "camel",
    "trophy",
    "ice cream",
    "person rowing boat",
    "ring",
    "tropical fish",
    "crown",
    "horse racing",
    "spiral shell",
    "watch",
    "rocket",
    "herb",
    "radio",
    "sun with face",
    "cat face",
    "soccer ball",
    "leopard",
    "bathtub",
    "closed umbrella",
    "folded hands",
    "person walking",
    "honey pot",
    "face with medical mask",
    "star",
    "sports medal",
    "megaphone",
    "backpack",
    "movie camera",
    "ox",
    "face screaming in fear",
    "mouse face",
    "bowling",
    "candy",
    "bicycle",
    "water buffalo",
]
excluded_entities = ["horse racing", "folded hands"]
#     "backpack",
#     "soccer ball",
#     "crown",
#     "face screaming in fear",
#     "bowling",
#     "books",
#     "star",
#     "water buffalo",
#     "rocket",
#     "person walking",
#     "closed umbrella",
#     "mouse face",
#     "watch",
#     "honey pot",
#     "ring",
#     "candy",
#     "bathtub",
# ]


config = Munch()
config.dataset = Munch(
    entity_names=entity_names,
    excluded_entities=[],
    relation_names=["left_of", "right_of", "above", "below"],
    num_entities=2,
    frame_of_reference="absolute",
    w_range=(8, 16),
    h_range=(8, 16),
    theta_range=(0, 2 * math.pi),
    add_bbox=False,
    add_front=False,
    transform=None,
    canvas_size=(128, 128),
    num_samples=10 ** 6,
    root_seed=0,
)
train_dataset = DRLDataset(
    **{
        **config.dataset,
        **dict(
            excluded_entities=excluded_entities,
            num_samples=10 ** 6,
            root_seed=0,
        ),
    }
)
validation_dataset_standard = DRLDataset(
    **{
        **config.dataset,
        **dict(
            excluded_entities=excluded_entities,
            num_samples=10 ** 4,
            root_seed=train_dataset.num_samples,
        ),
    }
)
validation_dataset_compositional = DRLDataset(
    **{
        **config.dataset,
        **dict(
            entity_names=excluded_entities,
            excluded_entities=[],
            num_samples=10 ** 4,
            root_seed=train_dataset.num_samples
            + validation_dataset_standard.num_samples,
        ),
    }
)


config.data_loader = Munch(
    batch_size=64,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
)
train_loader = DataLoader(train_dataset, **config.data_loader)
validation_loader_standard = DataLoader(
    validation_dataset_standard, **{**config.data_loader, "shuffle": False}
)
validation_loader_compositional = DataLoader(
    validation_dataset_compositional, **{**config.data_loader, "shuffle": False}
)

config.model = Munch(
    vision_model="resnet18",
    image_size=(3, *config.dataset.canvas_size),
    num_embeddings=len(train_dataset.word2idx),
    embedding_dim=64,
    question_len=train_dataset[0][1].shape.numel(),
    image_encoder_pretrained=False,
    freeze_image_encoder=False,
    lr=0.001,
)
model = DRLNet(**config.model)

repo = git.Repo(Path(".").absolute(), search_parent_directories=True)
if repo.is_dirty():
    raise RepositoryDirtyError(repo, "Have you forgotten to commit the changes?")
sha = repo.head.object.hexsha

ROOT = Path(repo.working_tree_dir)
tb_logger = loggers.TensorBoardLogger(
    save_dir=ROOT / "lightning_logs", name="", version=sha
)
config.trainer = Munch(
    gpus=[0],
    max_epochs=10,
    precision=32,
    limit_train_batches=1.0,
    limit_val_batches=1.0,
    val_check_interval=0.1,
)
trainer = pl.Trainer(**{**config.trainer, **dict(logger=tb_logger)})
trainer.fit(
    model, train_loader, [validation_loader_standard, validation_loader_compositional]
)
