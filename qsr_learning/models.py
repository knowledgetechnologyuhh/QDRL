from copy import deepcopy

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import accuracy_score

from typing import Tuple


class ImageModule(nn.Module):
    def __init__(
        self, vision_model: str, image_size: Tuple[int, int, int], output_size: int
    ):
        """Process the image.

        :param vision_model: vision model name
        :param image_size: a (c, h, w) triple
        :param output_size: the size of the output
        """
        super().__init__()

        # Image encoder
        resnet = getattr(torchvision.models, vision_model)(pretrained=True)
        self.image_encoder = nn.Sequential(*deepcopy(list(resnet.children())[:-3]))
        del resnet
        # Freeze the image encoder weights
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        self.image_size = image_size
        self.output_size = output_size
        # Compute the size of image featutres
        image = torch.rand(
            (1, *self.image_size),
            device=list(set(p.device for p in self.parameters()))[0],
        )
        with torch.no_grad():
            _, c_in, h_in, w_in = self.image_encoder(image).shape

        self.image_module = nn.Sequential(
            nn.Conv2d(c_in, output_size, (h_in, w_in)),
            nn.BatchNorm2d(output_size),
            nn.ReLU(),
        )

    def forward(self, image):
        batch_size = image.shape[0]
        image_features = self.image_encoder(image)
        out = self.image_module(image_features)
        out = out.view(batch_size, -1)
        return out


class QuestionModule(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        question_len: int,
        output_size: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.question_len = question_len

        question = torch.randint(
            0,
            num_embeddings,
            (1, question_len),
            dtype=torch.int64,
            device=list(set(p.device for p in self.parameters()))[0],
        )
        with torch.no_grad():
            input_size = self.embedding(question).shape.numel()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, question):
        batch_size = question.shape[0]
        out = self.embedding(question)
        out = out.view(batch_size, -1)
        out = self.fc(out)
        return out


class DRLNet(pl.LightningModule):
    def __init__(
        self,
        vision_model: str,
        image_size: Tuple[int, int, int],
        num_embeddings: int,
        embedding_dim: int,
        question_len: int,
    ):
        super().__init__()
        image_feature_size = question_feature_size = 32

        # Image Module
        self.image_module = ImageModule(vision_model, image_size, image_feature_size)

        # Question module
        self.question_module = QuestionModule(
            num_embeddings, embedding_dim, question_len, question_feature_size
        )
        self.fc = nn.Linear(image_feature_size, 1)
        self.criterion = nn.BCELoss()

    def forward(self, images, questions):
        image_features = self.image_module(images)
        question_features = self.question_module(questions)
        # Fusion
        out = image_features * question_features
        out = self.fc(out)
        out = out.view(-1).sigmoid()
        return out

    def training_step(self, batch, batch_idx):
        self.image_encoder.eval()

        # Make prediction
        images, questions, answers = batch
        preds = self(images, questions)
        loss = self.criterion(preds, answers)

        # Logging
        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy_score(answers, preds))
        return loss

    def configure_optimizers(self):
        # Make sure to filter the parameters based on `requires_grad`
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters))
