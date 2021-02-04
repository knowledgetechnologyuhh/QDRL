import torch
import torch.nn as nn
from torchvision.models import resnet18


class Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_encoder = resnet18(pretrained=True, progress=True)
        self.embedding = nn.Embedding(config.num_embeddings, config.embedding_dim)
        self.fc = nn.Linear(1000, 3 * config.embedding_dim)

        nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, images, questions):
        # image_features.shape = (batch_size, 3, 1000)
        image_features = self.image_encoder(images)
        # out.shape = (batch_size, self.embedding_dim)
        out = self.fc(image_features)
        # question_features.shape = (16, self.embedding_dim)
        head_features = self.embedding(questions[:, 0])
        relation_features = self.embedding(questions[:, 1])
        tail_features = self.embedding(questions[:, 2])
        question_features = torch.cat(
            (head_features, relation_features, tail_features), dim=-1
        )
        out = (out * question_features).sum(-1)
        return out.sigmoid()
