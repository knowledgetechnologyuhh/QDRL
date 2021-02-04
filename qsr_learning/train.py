import torch
import torch.nn as nn
from munch import Munch
from ray import tune
from torch.utils.data import DataLoader

from qsr_learning.data.data import QSRData
from qsr_learning.models.simple_baseline import Net


def report_result(epoch, phases, result, data_loader):
    log = dict(epoch=epoch)
    for phase in phases:
        log[phase + "_loss"] = result[phase].total_loss / len(
            data_loader[phase].dataset
        )
        log[phase + "_accuracy"] = result[phase].num_correct / len(
            data_loader[phase].dataset
        )
    tune.report(**log)


def step(model, criterion, optimizer, phase, batch, result, device):
    model.train() if phase == "train" else model.eval()
    torch.autograd.set_grad_enabled(phase == "train")
    images, questions, answers = (item.to(device) for item in batch)
    batch_size = images.shape[0]
    model.zero_grad()
    out = model(images, questions)
    loss = criterion(out, answers.float()) / batch_size
    result[phase].total_loss += loss.item()
    result[phase].num_correct += ((out > 0.5) == answers).sum().item()
    if phase == "train":
        loss.backward()
        optimizer.step()


# def train_epoch(phase, epoch, result, data_loader, model, criterion, optimizer, device):
#     result.total_loss[phase] = 0
#     result.num_correct[phase] = 0
#     for batch in data_loader[phase]:
#         train_step(model, criterion, optimizer, phase, batch, result, device)
#     print(epoch, format_result(phase, result, data_loader))


def train(config, device):
    phases = ["train", "validation"]
    data = Munch({phase: QSRData(**config.data[phase]) for phase in phases})
    config.model.num_embeddings = len(data.train.word2idx)
    data_loader = Munch(
        {
            phase: DataLoader(
                data[phase],
                batch_size=config.train.batch_size,
                shuffle=True,
                num_workers=4,
            )
            for phase in phases
        }
    )
    model = Net(config.model)
    model.to(device)
    criterion = nn.BCELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters())
    result = Munch()
    for epoch in range(config.train.num_epochs):
        for phase in phases:
            result[phase] = Munch()
            result[phase].total_loss = 0
            result[phase].num_correct = 0
            for batch in data_loader[phase]:
                step(model, criterion, optimizer, phase, batch, result, device)
        report_result(epoch, phases, result, data_loader)


# def test(config):
#     phases = ["test"]
#     data = Munch({phase: QSRData(**config.data[phase]) for phase in phases})
#     config.model.num_embeddings = len(data.train.word2idx)
#     data_loader = Munch(
#         {
#             phase: DataLoader(
#                 data[phase],
#                 batch_size=config.train.batch_size,
#                 shuffle=True,
#                 num_workers=4,
#             )
#             for phase in phases
#         }
#     )
#     result = Munch(
#         total_loss=Munch({phase: 0 for phase in phases}),
#         num_correct={phase: 0 for phase in phases},
#     )
#     for phase in phases:
#         for batch in data_loader[phase]:
#             train_step(model, criterion, optimizer, phase, batch, result, device)
#         print(epoch, format_result(phase, result, data_loader))
#     print(format_result(["test"], result, data_loader))
