import torch
import torch.nn as nn
import torchvision
from copy import deepcopy


class HadarmardFusionNet(nn.Module):
    def __init__(
        self,
        dataset,
        ent_dim,
        rel_dim,
        cnn_model: str,
        pretrained,
    ):
        super().__init__()
        resnet = getattr(torchvision.models, cnn_model)(pretrained=pretrained)
        self.image_encoder = nn.Sequential(*deepcopy(list(resnet.children())[:-3]))
        del resnet
        # Get the size of the image featutres
        with torch.no_grad():
            device = list(self.parameters())[0].device
            output_size = (
                self.image_encoder(torch.rand((1, 3, 224, 224), device=device))
                .view(-1)
                .shape[0]
            )
        self.ent_embedding = nn.Embedding(len(dataset.idx2ent), ent_dim)
        self.rel_embedding = nn.Embedding(len(dataset.idx2rel), rel_dim)
        self.fc1 = nn.Linear(output_size, 2 * ent_dim + rel_dim)
        self.fc2 = nn.Linear(2 * (2 * ent_dim + rel_dim), 2 * ent_dim + rel_dim)
        self.fc3 = nn.Linear(2 * ent_dim + rel_dim, 1)

        nn.init.xavier_uniform_(self.ent_embedding.weight.data)
        nn.init.xavier_uniform_(self.rel_embedding.weight.data)

    def forward(self, images, questions):
        # image_features.shape = (batch_size, 2 * config.ent_emb_dim + config.rel_emb_dim)
        image_features = self.image_encoder(images).view(images.shape[0], -1)
        head_features = self.ent_embedding(questions[:, 0])
        relation_features = self.rel_embedding(questions[:, 1])
        tail_features = self.ent_embedding(questions[:, 2])
        # question_features.shape = (batch_size, 2 * config.ent_emb_dim + config.rel_emb_dim)
        question_features = torch.cat(
            (head_features, relation_features, tail_features), dim=-1
        )
        out = torch.cat((self.fc1(image_features), question_features), dim=-1)
        out = self.fc2(out).relu()
        out = self.fc3(out).view(-1).sigmoid()
        return out
