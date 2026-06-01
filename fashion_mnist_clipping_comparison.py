#!/usr/bin/env python3
"""
Fashion-MNIST: Fixed vs Adaptive Clipping Comparison

Fixed C experiments + Run adaptive script separately for comparison.
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


FASHION_MNIST_MEAN = 0.2860
FASHION_MNIST_STD = 0.3530


class FashionCNN(nn.Module):
    def __init__(self, fc1_features=1024):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, fc1_features)
        self.fc2 = nn.Linear(fc1_features, 10)
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
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_and_test(args, device, fixed_c):
    """Train with fixed clipping threshold and return (epsilon, accuracy)."""
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            args.data_root, train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((FASHION_MNIST_MEAN,), (FASHION_MNIST_STD,)),
            ])
        ),
        batch_size=args.batch_size, num_workers=0, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(
            args.data_root, train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((FASHION_MNIST_MEAN,), (FASHION_MNIST_STD,)),
            ])
        ),
        batch_size=args.test_batch_size, shuffle=True, num_workers=0, pin_memory=True,
    )

    model = FashionCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    privacy_engine = PrivacyEngine(secure_mode=False)
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=args.sigma,
        max_grad_norm=fixed_c,
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for data, target in tqdm(train_loader, leave=False):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Test
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100.0 * correct / len(test_loader.dataset)
    epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)

    return epsilon, accuracy


def main():
    parser = argparse.ArgumentParser(description="Fashion-MNIST Fixed C Comparison")
    parser.add_argument("-b", "--batch-size", type=int, default=150)
    parser.add_argument("--test-batch-size", type=int, default=1024)
    parser.add_argument("-n", "--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data-root", type=str, default="../fashion_mnist")
    args = parser.parse_args()
    device = torch.device(args.device)

    results = []

    # Different C values to compare
    c_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]

    print("=" * 70)
    print("Fashion-MNIST: Fixed C Comparison")
    print(f"sigma={args.sigma}, epochs={args.epochs}, batch_size={args.batch_size}")
    print("=" * 70)

    for c in c_values:
        print(f"\n--- C={c} ---")
        try:
            epsilon, accuracy = train_and_test(args, device, c)
            print(f"  epsilon={epsilon:.4f}, accuracy={accuracy:.2f}%")
            results.append({"C": c, "epsilon": epsilon, "accuracy": accuracy})
        except Exception as e:
            print(f"  Error: {e}")
            results.append({"C": c, "epsilon": float('nan'), "accuracy": float('nan')})

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'C':>8} | {'epsilon':>10} | {'accuracy':>10}")
    print("-" * 40)
    for r in results:
        if not np.isnan(r['accuracy']):
            print(f"{r['C']:>8.1f} | {r['epsilon']:>10.4f} | {r['accuracy']:>9.2f}%")
        else:
            print(f"{r['C']:>8.1f} | {'ERROR':>10} | {'ERROR':>10}")
    print("=" * 70)

    # Find optimal C
    valid_results = [r for r in results if not np.isnan(r['accuracy'])]
    if valid_results:
        best = max(valid_results, key=lambda x: x['accuracy'])
        print(f"\nOptimal C: {best['C']} with accuracy={best['accuracy']:.2f}%")

    # Save results
    repro_str = f"fixed_c_compare_{args.sigma}_{args.epochs}"
    torch.save(results, f"results_{repro_str}.pt")
    print(f"\nSaved to results_{repro_str}.pt")

    print("\n" + "=" * 70)
    print("ADAPTIVE CLIPPING EXPERIMENTS")
    print("=" * 70)
    print("To run adaptive clipping experiments, use:")
    print("  python fashion_mnist_adaptive_histogram.py -n 20 -b 150 --lr 0.0001 --sigma 1.0 -c 1.0 --target-ratio 0.3 --plot")
    print("  python fashion_mnist_adaptive_histogram.py -n 20 -b 150 --lr 0.0001 --sigma 1.0 -c 1.0 --target-ratio 0.5 --plot")
    print("  python fashion_mnist_adaptive_histogram.py -n 20 -b 150 --lr 0.0001 --sigma 1.0 -c 1.0 --target-ratio 0.7 --plot")


if __name__ == "__main__":
    main()