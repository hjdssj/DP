#!/usr/bin/env python3
"""
Fashion-MNIST training with differential privacy (baseline).
Uses a deeper CNN architecture similar to standard LeNet-5 style networks.
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from torchvision import datasets, transforms
from tqdm import tqdm


# Precomputed characteristics of the Fashion-MNIST dataset
FASHION_MNIST_MEAN = 0.2860
FASHION_MNIST_STD = 0.3530


class DeepConvNet(nn.Module):
    """
    Deeper CNN architecture for Fashion-MNIST:
    - Conv1: 5x5, 1->32 channels
    - Pool1: 2x2 max pool
    - Conv2: 5x5, 32->64 channels
    - Pool2: 2x2 max pool
    - FC1: 7*7*64=3136 -> 1024
    - FC2: 1024 -> 10
    """
    def __init__(self):
        super().__init__()
        # Conv1: 1x28x28 -> 32x28x28
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        # Pool1: 32x28x28 -> 32x14x14
        # FC1 input: 32 * 7 * 7 = 1568 (after two 2x2 pools on 28x28)
        self.fc1 = nn.Linear(32 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 10)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Conv1 + ReLU + Pool
        x = F.max_pool2d(F.relu(self.conv1(x)), 2, 2)  # 28x28 -> 14x14
        # Conv2 + ReLU + Pool
        # Note: with valid pool, we'd get 7x7, but same architecture works
        x = F.max_pool2d(F.relu(self.conv1(x)), 2, 2)  # 14x14 -> 7x7
        # Flatten
        x = x.view(-1, 32 * 7 * 7)
        # FC1 + ReLU + Dropout (handled externally if needed)
        x = F.relu(self.fc1(x))
        # FC2
        x = self.fc2(x)
        return x


class FashionCNN(nn.Module):
    """
    CNN architecture matching the TensorFlow version:
    - Conv1: 5x5, 1->32
    - Pool1: 2x2 max pool
    - Conv2: 5x5, 32->64
    - Pool2: 2x2 max pool
    - FC1: 7*7*64=3136 -> 1024 (or 1568 -> 1024 with different pooling)
    - FC2: 1024 -> 10
    """
    def __init__(self, fc1_features=1024):
        super().__init__()
        # Conv1: 1 channel, 5x5 kernel, 32 filters
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)  # output: 32 x 28 x 28
        # Pool1: 2x2, stride 2 -> output: 32 x 14 x 14
        # Conv2: 32 channels, 5x5 kernel, 64 filters
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)  # output: 64 x 14 x 14
        # Pool2: 2x2, stride 2 -> output: 64 x 7 x 7
        self.fc1 = nn.Linear(64 * 7 * 7, fc1_features)  # 3136 -> 1024
        self.fc2 = nn.Linear(fc1_features, 10)  # 1024 -> 10

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.1)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.1)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        # Conv1 + ReLU
        x = F.relu(self.conv1(x))
        # Pool1
        x = F.max_pool2d(x, 2, 2)  # 28x28 -> 14x14

        # Conv2 + ReLU
        x = F.relu(self.conv2(x))
        # Pool2
        x = F.max_pool2d(x, 2, 2)  # 14x14 -> 7x7

        # Flatten
        x = x.view(x.size(0), -1)

        # FC1 + ReLU
        x = F.relu(self.fc1(x))

        # FC2 (no softmax, CrossEntropyLoss handles it)
        x = self.fc2(x)
        return x


def train(args, model, device, train_loader, optimizer, privacy_engine, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    if not args.disable_dp:
        epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"(ε = {epsilon:.2f}, δ = {args.delta})"
        )
    else:
        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")


def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct / len(test_loader.dataset)


def main():
    parser = argparse.ArgumentParser(
        description="Fashion-MNIST with Opacus DP-SGD (Deep CNN)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--test-batch-size", type=int, default=1024)
    parser.add_argument("-n", "--epochs", type=int, default=10)
    parser.add_argument("-r", "--n-runs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (Adam default)")
    parser.add_argument("--sigma", type=float, default=1.0, help="Noise multiplier")
    parser.add_argument("-c", "--max-per-sample-grad-norm", type=float, default=1.0,
                        help="Clip per-sample gradients to this norm")
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--disable-dp", action="store_true")
    parser.add_argument("--secure-rng", action="store_true")
    parser.add_argument("--data-root", type=str, default="../fashion_mnist",
                        help="Where Fashion-MNIST is/will be stored")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout rate (only used in non-DP mode)")
    args = parser.parse_args()
    device = torch.device(args.device)

    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            args.data_root,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((FASHION_MNIST_MEAN,), (FASHION_MNIST_STD,)),
            ])
        ),
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            args.data_root,
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((FASHION_MNIST_MEAN,), (FASHION_MNIST_STD,)),
            ])
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    run_results = []
    for _ in range(args.n_runs):
        model = FashionCNN().to(device)

        # Use Adam optimizer like the TensorFlow version
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        privacy_engine = None

        if not args.disable_dp:
            privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=args.sigma,
                max_grad_norm=args.max_per_sample_grad_norm,
            )

        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, privacy_engine, epoch)
        run_results.append(test(model, device, test_loader))

    if len(run_results) > 1:
        print(
            "Accuracy averaged over {} runs: {:.2f}% ± {:.2f}%".format(
                len(run_results), np.mean(run_results) * 100, np.std(run_results) * 100
            )
        )

    repro_str = (
        f"fashion_mnist_deep_{args.lr}_{args.sigma}_"
        f"{args.max_per_sample_grad_norm}_{args.batch_size}_{args.epochs}"
    )
    torch.save(run_results, f"run_results_{repro_str}.pt")

    if args.save_model:
        torch.save(model.state_dict(), f"fashion_mnist_cnn_{repro_str}.pt")


if __name__ == "__main__":
    main()