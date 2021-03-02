import torch
import torch.nn as nn
from munch import Munch
from torch.utils.data import DataLoader
from tqdm.auto import trange

from qsr_learning.data import DRLDataset
from qsr_learning.models import HadarmardFusionNet


def report_result(epoch, phases, result, data_loader):
    log = dict(epoch=epoch)
    for phase in phases:
        log[phase + "_loss"] = result[phase].total_loss / len(
            data_loader[phase].dataset
        )
        log[phase + "_accuracy"] = result[phase].num_correct / len(
            data_loader[phase].dataset
        )
    # tune.report(**log)
    print(
        f"epoch: {log['epoch']:03d} "
        + " ".join(f"{key}: {{{key}:3.3f}}" for key in log if key != "epoch").format(
            **log
        )
    )


def step(model, criterion, optimizer, phase, batch, result, freeze, device):
    if phase == "train":
        model.train()
        if freeze == "all":
            model.image_encoder.eval()
        if freeze == "bn":  # Common in training object detection models
            for layer in model.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eval()
    else:
        model.eval()
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


def train(config):
    phases = ["train", "validation"]
    datasets = Munch({phase: DRLDataset(**config.data[phase]) for phase in phases})
    data_loader = Munch(
        {
            phase: DataLoader(
                datasets[phase],
                batch_size=config.train.batch_size,
                num_workers=4,
            )
            for phase in phases
        }
    )

    model = HadarmardFusionNet(datasets.train, **config.model)
    model.to(device)
    criterion = nn.BCELoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    result = Munch()
    for epoch in trange(config.train.num_epochs):
        for phase in phases:
            result[phase] = Munch()
            result[phase].total_loss = 0
            result[phase].num_correct = 0
            for batch in data_loader[phase]:
                step(
                    model,
                    criterion,
                    optimizer,
                    phase,
                    batch,
                    result,
                    config.train.freeze,
                    device,
                )
        report_result(epoch, phases, result, data_loader)
        torch.save(model.state_dict(), "model.pt")
