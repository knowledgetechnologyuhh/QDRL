from pathlib import Path

import git
import pytorch_lightning as pl
from git.exc import RepositoryDirtyError
from munch import Munch
from pytorch_lightning import loggers
from torch.utils.data import DataLoader

from qsr_learning.data import DRLDataset
from qsr_learning.models import DRLNet

config = Munch()


# # Dataset


entity_names = [
    "slightly smiling face",
    "nerd face",
    "smiling face with halo",
    "expressionless face",
    "flushed face",
    "face with tears of joy",
    "neutral face",
    "smiling face with heart-eyes",
    "face with medical mask",
    "loudly crying face",
    "hugging face",
    "smiling face with smiling eyes",
    "squinting face with tongue",
    "face with steam from nose",
    "dog face",
    "cat face",
    "face screaming in fear",
    "pouting face",
    "pig face",
    "rabbit face",
    "tiger face",
    "monkey face",
    "cow face",
    "tired face",
    "mouse face",
    "dragon face",
    "face with tongue",
    "sun with face",
    "worried face",
    "dizzy face",
    "face with open mouth",
    "fearful face",
]
excluded_entity_names = [
    "hugging face",
    "fearful face",
    "face with steam from nose",
    "face with tongue",
    "nerd face",
    "expressionless face",
    "dragon face",
    "flushed face",
    "cow face",
    "smiling face with heart-eyes",
    "sun with face",
    "pig face",
    "pouting face",
    "smiling face with halo",
    "slightly smiling face",
    "worried face",
    "neutral face",
    "loudly crying face",
]
relation_names = ["left_of", "right_of", "above", "below"]
excluded_relation_names = ["above", "below"]
print(
    len(entity_names),
    len(excluded_entity_names),
    set(excluded_entity_names).issubset(set(entity_names)),
)
print(
    len(relation_names),
    len(excluded_relation_names),
    set(excluded_relation_names).issubset(set(relation_names)),
)


config.dataset = Munch(
    entity_names=entity_names,
    excluded_entity_names=excluded_entity_names,
    relation_names=relation_names,
    excluded_relation_names=excluded_relation_names,
    num_entities=2,
    frame_of_reference="absolute",
    w_range=(16, 16),
    h_range=(16, 16),
    theta_range=(0, 0),
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
            num_samples=10 ** 6,
            root_seed=0,
        ),
    }
)
validation_dataset_standard = DRLDataset(
    **{
        **config.dataset,
        **dict(
            num_samples=10 ** 4,
            root_seed=train_dataset.num_samples,
        ),
    }
)
validation_dataset_compositional = DRLDataset(
    **{
        **config.dataset,
        **dict(
            entity_names=excluded_entity_names,
            excluded_entity_names=[],
            relation_names=excluded_relation_names,
            excluded_relation_names=[],
            num_samples=10 ** 4,
            root_seed=train_dataset.num_samples
            + validation_dataset_standard.num_samples,
        ),
    }
)


# # Data Loader


config.data_loader = Munch(
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)


train_loader = DataLoader(train_dataset, **config.data_loader)
validation_loader_standard = DataLoader(
    validation_dataset_standard, **{**config.data_loader, "shuffle": False}
)
validation_loader_compositional = DataLoader(
    validation_dataset_compositional, **{**config.data_loader, "shuffle": False}
)


# # Model


config.model = Munch(
    vision_model="resnet18",
    image_size=(3, *config.dataset.canvas_size),
    num_embeddings=len(train_dataset.word2idx),
    embedding_dim=64,
    question_len=train_dataset[0][1].shape.numel(),
    image_encoder_pretrained=False,
    freeze_image_encoder=False,
)


model = DRLNet(**config.model)


# # Trainer


config.trainer = Munch(
    gpus=[0],
    max_epochs=10,
    precision=32,
    limit_train_batches=1.0,
    limit_val_batches=1.0,
    val_check_interval=0.1,
)


repo = git.Repo(Path(".").absolute(), search_parent_directories=True)
ROOT = Path(repo.working_tree_dir)


if repo.is_dirty():
    raise RepositoryDirtyError(repo, "Have you forgotten to commit the changes?")
sha = repo.head.object.hexsha


tb_logger = loggers.TensorBoardLogger(
    save_dir=ROOT / "lightning_logs", name="", version=sha
)
trainer = pl.Trainer(**{**config.trainer, **dict(logger=tb_logger)})
trainer.fit(
    model, train_loader, [validation_loader_standard, validation_loader_compositional]
)
