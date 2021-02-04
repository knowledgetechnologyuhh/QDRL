import os

import torch
from munch import Munch

from qsr_learning.relations.absolute.half_planes import left_of, right_of
from qsr_learning.train import train

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = Munch(
    model=Munch(
        embedding_dim=20,
    ),
    train=Munch(
        batch_size=64,
        num_epochs=10,
    ),
    data=Munch(
        negative_sample_mixture=Munch(head=1, relation=1, tail=1),
        train=Munch(
            relations=[left_of, right_of],
            num_images=128,
            num_objects=3,
            num_pos_questions_per_image=2,
            num_neg_questions_per_image=2,
        ),
        validation=Munch(
            relations=[left_of, right_of],
            num_images=16,
            num_objects=3,
            num_pos_questions_per_image=2,
            num_neg_questions_per_image=2,
        ),
        test=Munch(
            relations=[left_of, right_of],
            num_images=16,
            num_objects=3,
            num_pos_questions_per_image=2,
            num_neg_questions_per_image=2,
        ),
    ),
)

train(config, device)
