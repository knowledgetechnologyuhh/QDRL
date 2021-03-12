from copy import deepcopy
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from pytorch_lightning.metrics import Accuracy


class VisionModule(nn.Module):
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
        c, h, w = self.image_encoder_output_size
        self.vision_module = nn.Sequential(
            nn.Conv2d(c, c, (3, 3), padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            nn.Conv2d(c, c, (3, 3), padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            nn.Conv2d(c, output_size, (h, w)),
            nn.BatchNorm2d(output_size),
            nn.ReLU(),
            nn.Conv2d(output_size, output_size, 1),
            nn.BatchNorm2d(output_size),
            nn.ReLU(),
        )

    def forward(self, image):
        batch_size = image.shape[0]
        image_features = self.image_encoder(image)
        out = self.vision_module(image_features)
        out = out.view(batch_size, -1)
        return out

    @property
    def image_encoder_output_size(self):
        image = torch.rand(
            (1, *self.image_size),
            device=list(set(p.device for p in self.parameters()))[0],
        )
        with torch.no_grad():
            _, c, h, w = self.image_encoder(image).shape
        return c, h, w

    @property
    def output_size(self):
        image = torch.rand(
            (2, *self.image_size),  # batchnorm requires batch_size > 1
            device=list(set(p.device for p in self.parameters()))[0],
        )
        with torch.no_grad():
            out = self(image).shape[1:].numel()
        return out


class LanguageModule(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        question_len: int,
        # output_size: int,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.question_len = question_len
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, question):
        out = self.embedding(question)
        _, hidden = self.gru(out)
        out = hidden.squeeze(0)
        return out

    @property
    def output_size(self):
        question = torch.randint(
            0,
            self.num_embeddings,
            (1, self.question_len),
            dtype=torch.int64,
            device=list(set(p.device for p in self.parameters()))[0],
        )
        with torch.no_grad():
            out = self(question).shape.numel()
        return out


class FusionModule(nn.Module):
    def __init__(self, image_feature_size, question_feature_size):
        super().__init__()
        self.fusion_module = nn.Sequential(
            nn.Linear(
                image_feature_size + question_feature_size,
                image_feature_size + question_feature_size,
            ),
            nn.BatchNorm1d(image_feature_size + question_feature_size),
            nn.ReLU(),
            nn.Linear(
                image_feature_size + question_feature_size,
                image_feature_size + question_feature_size,
            ),
            nn.BatchNorm1d(image_feature_size + question_feature_size),
            nn.ReLU(),
            nn.Linear(image_feature_size + question_feature_size, 1),
        )

    def forward(self, image_features: torch.Tensor, question_features: torch.Tensor):
        out = torch.cat((image_features, question_features), dim=-1)
        out = self.fusion_module(out)
        out = out.view(-1)
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
        self.save_hyperparameters()

        # Question module
        self.language_module = LanguageModule(
            num_embeddings, embedding_dim, question_len
        )
        # Image Module
        image_feature_size = self.language_module.output_size
        self.vision_module = VisionModule(vision_model, image_size, image_feature_size)
        self.fusion_module = FusionModule(
            self.vision_module.output_size, self.language_module.output_size
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.metric = Accuracy()

    def forward(self, images, questions):
        image_features = self.vision_module(images)
        question_features = self.language_module(questions)
        out = self.fusion_module(image_features, question_features)
        return out

    def training_step(self, batch, batch_idx):
        self.vision_module.image_encoder.eval()
        # Make prediction
        images, questions, answers = batch
        out = self(images, questions)
        loss = self.criterion(out, answers)

        # Logging
        self.log("train_loss", loss, prog_bar=True)
        self.log(
            "train_accuracy",
            self.metric(out.sigmoid(), answers.long()),
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        # Make prediction
        images, questions, answers = batch
        out = self(images, questions)
        loss = self.criterion(out, answers)
        # Logging
        self.log("validation_loss", loss, prog_bar=True)
        self.log(
            "validation_accuracy",
            self.metric(out.sigmoid(), answers.long()),
            prog_bar=True,
        )

    def configure_optimizers(self):
        # Make sure to filter the parameters based on `requires_grad`
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()))
