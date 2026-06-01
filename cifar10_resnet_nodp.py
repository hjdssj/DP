#!/usr/bin/env python3
"""
CIFAR-10 non-DP baseline using Res_Model (BatchNorm + residual blocks).
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class Conv_Block(nn.Module):
    def __init__(self, inchannel, outchannel, res=True):
        super().__init__()
        self.res = res
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, 3, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.left(x)
        if self.res:
            out = out + self.shortcut(x)
        return self.relu(out)


class Res_Model(nn.Module):
    def __init__(self, res=True):
        super().__init__()
        self.block1 = Conv_Block(3, 64, res)
        self.block2 = Conv_Block(64, 128, res)
        self.block3 = Conv_Block(128, 256, res)
        self.block4 = Conv_Block(256, 512, res)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.maxpool(self.block1(x))
        x = self.maxpool(self.block2(x))
        x = self.maxpool(self.block3(x))
        x = self.maxpool(self.block4(x))
        return self.classifier(x)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    correct = 0
    total = 0

    for data, target in tqdm(train_loader, desc=f"Epoch {epoch}"):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    avg_loss = np.mean(losses)
    acc = 100.0 * correct / total
    print(f"Train Epoch: {epoch} \tLoss: {avg_loss:.6f} \tAcc: {correct}/{total} ({acc:.2f}%)")


def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(f"\nTest set: Avg loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")
    return correct / len(test_loader.dataset)


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 ResNet Non-DP Baseline")
    parser.add_argument("-b", "--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=1024)
    parser.add_argument("-n", "--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr-decay-epochs", type=int, nargs="+", default=[50, 75],
                        help="Epochs at which to decay learning rate by 10x")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data-root", type=str, default="../cifar10")
    parser.add_argument("--save-model", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_dataset = datasets.CIFAR10(args.data_root, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(args.data_root, train=False, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                              shuffle=False, num_workers=2, pin_memory=True)

    model = Res_Model().to(device)
    d = sum(p.numel() for p in model.parameters())
    print(f"Model: Res_Model | Parameters: {d:,}")

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_epochs, gamma=0.1)

    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        acc = test(model, device, test_loader)
        scheduler.step()

        if acc > best_acc:
            best_acc = acc
            if args.save_model:
                torch.save(model.state_dict(), "cifar10_resnet_nodp_best.pt")

    print(f"\nBest test accuracy: {best_acc * 100:.2f}%")

    if args.save_model:
        torch.save(model.state_dict(), "cifar10_resnet_nodp_final.pt")


if __name__ == "__main__":
    main()
