from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.metrics import Accuracy


def build_stem(
    feature_dim,
    stem_dim,
    module_dim,
    num_layers=2,
    with_batchnorm=True,
    kernel_size=[3],
    stride=[1],
    padding=None,
    subsample_layers=None,
    acceptEvenKernel=False,
):
    """Taken from the FiLM repository"""
    layers = []
    prev_dim = feature_dim

    if len(kernel_size) == 1:
        kernel_size = num_layers * kernel_size
    if len(stride) == 1:
        stride = num_layers * stride
    if padding is None:
        padding = num_layers * [None]
    if len(padding) == 1:
        padding = num_layers * padding
    if subsample_layers is None:
        subsample_layers = []

    for i, cur_kernel_size, cur_stride, cur_padding in zip(
        range(num_layers), kernel_size, stride, padding
    ):
        curr_out = module_dim if (i == (num_layers - 1)) else stem_dim
        if cur_padding is None:  # Calculate default padding when None provided
            if cur_kernel_size % 2 == 0 and not acceptEvenKernel:
                raise (NotImplementedError)
            cur_padding = cur_kernel_size // 2
        layers.append(
            nn.Conv2d(
                prev_dim,
                curr_out,
                kernel_size=cur_kernel_size,
                stride=cur_stride,
                padding=cur_padding,
                bias=not with_batchnorm,
            )
        )
        if with_batchnorm:
            layers.append(nn.BatchNorm2d(curr_out))
        layers.append(nn.ReLU(inplace=True))
        if i in subsample_layers:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        prev_dim = curr_out
    return nn.Sequential(*layers)


class VisionModule(nn.Module):
    def __init__(self, image_size: Tuple[int, int, int]):
        """Process the image.

        :param vision_model: vision model name
        :param image_size: a (c, h, w) triple
        :param output_size: the size of the output
        """
        super().__init__()

        # Image encoder
        self.image_encoder = build_stem(
            feature_dim=3,
            stem_dim=64,
            module_dim=64,
            num_layers=6,
            with_batchnorm=1,
            subsample_layers=(1, 3, 5),
        )

        self.image_size = image_size
        c, h, w = self.image_encoder_output_size

    def forward(self, image):
        image_features = self.image_encoder(image)
        out = image_features.view(image.shape[0], -1)
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
        hidden_size: int,
        question_len: int,
        # output_size: int,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.question_len = question_len
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
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
            nn.Linear(image_feature_size + question_feature_size, 1)
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
        hidden_size: int,
        question_len: int,
        lr=0.001,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vision_module = VisionModule(image_size)
        self.language_module = LanguageModule(
            num_embeddings, embedding_dim, hidden_size, question_len
        )
        self.fusion_module = FusionModule(
            self.vision_module.output_size, self.language_module.output_size
        )
        self.lr = lr

        self.criterion = nn.BCEWithLogitsLoss()
        self.metric = Accuracy()

    def forward(self, images, questions):
        image_features = self.vision_module(images)
        question_features = self.language_module(questions)
        out = self.fusion_module(image_features, question_features)
        return out

    def training_step(self, batch, batch_idx):
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

    def validation_step(self, batch, batch_idx, *args, **kwargs):
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
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr
        )
