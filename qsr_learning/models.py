import torch
import torch.nn as nn
from torchvision.models import resnet18


class HadarmardFusionNet(nn.Module):
    def __init__(self, config, dataset):
        super().__init__()
        self.image_encoder = resnet18(pretrained=config.pretrained)
        self.image_encoder.fc = nn.Linear(512, 2 * config.ent_dim + config.rel_dim)
        self.ent_embedding = nn.Embedding(len(dataset.idx2ent), config.ent_dim)
        self.rel_embedding = nn.Embedding(len(dataset.idx2rel), config.rel_dim)

        nn.init.xavier_uniform_(self.ent_embedding.weight.data)
        nn.init.xavier_uniform_(self.rel_embedding.weight.data)

    def forward(self, images, questions):
        # image_features.shape = (batch_size, 2 * config.ent_emb_dim + config.rel_emb_dim)
        image_features = self.image_encoder(images)
        head_features = self.ent_embedding(questions[:, 0])
        relation_features = self.rel_embedding(questions[:, 1])
        tail_features = self.ent_embedding(questions[:, 2])
        # question_features.shape = (batch_size, 2 * config.ent_emb_dim + config.rel_emb_dim)
        question_features = torch.cat(
            (head_features, relation_features, tail_features), dim=-1
        )
        # out.shape = (batch_size,)
        out = (image_features * question_features).sum(-1)
        return out.sigmoid()
