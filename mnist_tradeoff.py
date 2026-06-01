#!/usr/bin/env python3
"""
MNIST: Privacy (epsilon) vs Accuracy trade-off curve.

Fixed: epochs=20, batch=256, lr=0.1, C=0.4 (fixed) / initial_c=1.0 (adaptive)
Varied: sigma in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]

Produces two curves on the same plot:
  - Fixed C=0.4 baseline
  - Adaptive C (target_ratio=0.3)
"""

import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from torchvision import datasets, transforms
from tqdm import tqdm

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

EPOCHS = 20
BATCH_SIZE = 256
LR = 0.1
DELTA = 1e-5
FIXED_C = 0.4
INITIAL_C = 1.0
TARGET_RATIO = 0.3
SIGMAS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
DATA_ROOT = "../mnist"


class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=3)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 1)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 1)
        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class AdaptiveClipController:
    def __init__(self, initial_c, target_ratio=0.3, tolerance=0.05,
                 min_c=0.1, max_c=10.0, adjustment_speed=0.15):
        self.c = initial_c
        self.target_ratio = target_ratio
        self.tolerance = tolerance
        self.min_c = min_c
        self.max_c = max_c
        self.adjustment_speed = adjustment_speed

    def update(self, clipped_ratio):
        if clipped_ratio > self.target_ratio + self.tolerance:
            excess = clipped_ratio - (self.target_ratio + self.tolerance)
            new_c = self.c * (1 + self.adjustment_speed * (1 + excess * 2))
        elif clipped_ratio < self.target_ratio - self.tolerance:
            deficit = (self.target_ratio - self.tolerance) - clipped_ratio
            new_c = self.c * (1 - self.adjustment_speed * (1 + deficit * 2))
        else:
            new_c = self.c
        new_c = max(self.min_c, min(self.max_c, new_c))
        self.c = 0.5 * self.c + 0.5 * new_c
        return self.c


def make_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DATA_ROOT, train=True, download=True, transform=transform),
        batch_size=BATCH_SIZE, num_workers=0, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(DATA_ROOT, train=False, transform=transform),
        batch_size=1024, num_workers=0, pin_memory=True,
    )
    return train_loader, test_loader


def evaluate(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            correct += model(data).argmax(1).eq(target).sum().item()
    return correct / len(test_loader.dataset)


def run_fixed(sigma, device):
    """Train with fixed C=FIXED_C, return (epsilon, accuracy)."""
    train_loader, test_loader = make_loaders()
    model = SampleConvNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0)
    privacy_engine = PrivacyEngine(secure_mode=False)
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model, optimizer=optimizer, data_loader=train_loader,
        noise_multiplier=sigma, max_grad_norm=FIXED_C,
    )
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            criterion(model(data), target).backward()
            optimizer.step()
        eps = privacy_engine.accountant.get_epsilon(delta=DELTA)
        print(f"  [fixed  σ={sigma}] epoch {epoch:2d}/{EPOCHS} | ε={eps:.3f}")
    eps = privacy_engine.accountant.get_epsilon(delta=DELTA)
    acc = evaluate(model, device, test_loader)
    return eps, acc


def run_adaptive(sigma, device):
    """Train with adaptive C, return (epsilon, accuracy)."""
    train_loader, test_loader = make_loaders()
    model = SampleConvNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0)
    controller = AdaptiveClipController(INITIAL_C, TARGET_RATIO)
    privacy_engine = PrivacyEngine(secure_mode=False)
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model, optimizer=optimizer, data_loader=train_loader,
        noise_multiplier=sigma, max_grad_norm=INITIAL_C,
    )
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, clipped = 0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            criterion(model(data), target).backward()

            # Capture true per-sample norms before optimizer clips them
            per_param = []
            for p in model.parameters():
                if hasattr(p, 'grad_sample') and p.grad_sample is not None:
                    per_param.append(
                        p.grad_sample.reshape(p.grad_sample.shape[0], -1).norm(2, dim=1)
                    )
            if per_param:
                norms = torch.stack(per_param, dim=1).norm(2, dim=1)
                total += len(norms)
                clipped += int((norms >= controller.c).sum().item())

            optimizer.step()

        clipped_ratio = clipped / total if total > 0 else 0.0
        new_c = controller.update(clipped_ratio)
        optimizer.max_grad_norm = new_c
        eps = privacy_engine.accountant.get_epsilon(delta=DELTA)
        print(f"  [adapt  σ={sigma}] epoch {epoch:2d}/{EPOCHS} | ε={eps:.3f}"
              f" | C={new_c:.3f} | clipped={clipped_ratio:.1%}")
    eps = privacy_engine.accountant.get_epsilon(delta=DELTA)
    acc = evaluate(model, device, test_loader)
    return eps, acc


def save_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sigma', 'epsilon', 'accuracy', 'method'])
        w.writerows(rows)
    print(f"Saved {path}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Config: epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LR}, delta={DELTA}")
    print(f"Sigmas: {SIGMAS}\n")

    fixed_rows, adaptive_rows = [], []

    for sigma in SIGMAS:
        ckpt_fixed = f"tradeoff_fixed_s{sigma}.pt"
        ckpt_adaptive = f"tradeoff_adaptive_s{sigma}.pt"

        if os.path.exists(ckpt_fixed):
            d = torch.load(ckpt_fixed)
            eps, acc = d['epsilon'], d['accuracy']
            print(f"[fixed   σ={sigma}] loaded from cache: ε={eps:.3f}, acc={acc*100:.2f}%")
        else:
            print(f"\n--- Fixed C={FIXED_C}, sigma={sigma} ---")
            eps, acc = run_fixed(sigma, device)
            torch.save({'epsilon': eps, 'accuracy': acc}, ckpt_fixed)
            print(f"  => ε={eps:.3f}, acc={acc*100:.2f}%")
        fixed_rows.append([sigma, round(eps, 4), round(acc * 100, 2), 'fixed'])

        if os.path.exists(ckpt_adaptive):
            d = torch.load(ckpt_adaptive)
            eps, acc = d['epsilon'], d['accuracy']
            print(f"[adaptive σ={sigma}] loaded from cache: ε={eps:.3f}, acc={acc*100:.2f}%")
        else:
            print(f"\n--- Adaptive C (init={INITIAL_C}, target={TARGET_RATIO}), sigma={sigma} ---")
            eps, acc = run_adaptive(sigma, device)
            torch.save({'epsilon': eps, 'accuracy': acc}, ckpt_adaptive)
            print(f"  => ε={eps:.3f}, acc={acc*100:.2f}%")
        adaptive_rows.append([sigma, round(eps, 4), round(acc * 100, 2), 'adaptive'])

    save_csv('tradeoff_mnist.csv', fixed_rows + adaptive_rows)

    # Print summary table
    print("\n" + "=" * 60)
    print(f"{'sigma':>8} | {'epsilon':>8} | {'fixed acc':>10} | {'adapt acc':>10} | {'delta':>8}")
    print("=" * 60)
    for f_row, a_row in zip(fixed_rows, adaptive_rows):
        delta_acc = a_row[2] - f_row[2]
        print(f"{f_row[0]:>8} | {f_row[1]:>8.3f} | {f_row[2]:>9.2f}% | {a_row[2]:>9.2f}% | {delta_acc:>+7.2f}%")
    print("=" * 60)

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 6))
        f_eps = [r[1] for r in fixed_rows]
        f_acc = [r[2] for r in fixed_rows]
        a_eps = [r[1] for r in adaptive_rows]
        a_acc = [r[2] for r in adaptive_rows]

        ax.plot(f_eps, f_acc, 'bs-', markersize=8, linewidth=2, label=f'Fixed C={FIXED_C}')
        ax.plot(a_eps, a_acc, 'ro-', markersize=8, linewidth=2, label=f'Adaptive C (target={TARGET_RATIO})')

        for eps, acc, sigma in zip(f_eps, f_acc, SIGMAS):
            ax.annotate(f'σ={sigma}', (eps, acc), textcoords='offset points',
                        xytext=(5, 6), fontsize=8, color='blue')
        for eps, acc, sigma in zip(a_eps, a_acc, SIGMAS):
            ax.annotate(f'σ={sigma}', (eps, acc), textcoords='offset points',
                        xytext=(5, -14), fontsize=8, color='red')

        ax.set_xlabel('Privacy Budget ε', fontsize=13)
        ax.set_ylabel('Test Accuracy (%)', fontsize=13)
        ax.set_title('MNIST: Privacy-Utility Trade-off\n'
                     f'(epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LR})', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('tradeoff_mnist.png', dpi=150)
        print("Saved tradeoff_mnist.png")
    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == '__main__':
    main()
