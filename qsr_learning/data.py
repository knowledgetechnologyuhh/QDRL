import numpy as np
import torch
from munch import Munch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import trange

from qsr_learning.main import (
    draw,
    emoji_names,
    generate_negative_examples,
    generate_positive_examples,
    sample_objects,
)


class QSRData(Dataset):
    def __init__(
        self,
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
            objects = sample_objects(num_objects=num_objects)
            positive_examples = generate_positive_examples(
                objects, num_pos_questions_per_image
            )
            negative_examples = generate_negative_examples(
                objects, positive_examples, num_neg_questions_per_image, mixture
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
