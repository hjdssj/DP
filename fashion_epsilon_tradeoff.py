#!/usr/bin/env python3
"""
Fashion-MNIST: Privacy (epsilon) vs Accuracy trade-off curve.
Compares fixed C=0.4 baseline vs adaptive C (target_ratio=0.3).

Sweep: sigma in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
Fixed: epochs=20, batch=150, lr=0.0001 (Adam), delta=1e-5
"""

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torchvision import datasets, transforms
from tqdm import tqdm

FASHION_MNIST_MEAN = 0.2860
FASHION_MNIST_STD = 0.3530

SIGMAS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
FIXED_C = 0.4
EPOCHS = 20
LR = 0.0001
DELTA = 1e-5
TARGET_RATIO = 0.2


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
        return self.fc2(x)


def make_loaders(data_root, batch_size):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((FASHION_MNIST_MEAN,), (FASHION_MNIST_STD,)),
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(data_root, train=True, download=True, transform=tf),
        batch_size=batch_size, num_workers=0, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(data_root, train=False, transform=tf),
        batch_size=1024, num_workers=0, pin_memory=True,
    )
    return train_loader, test_loader


def compute_grad_norms(model):
    """Collect per-sample gradient norms from grad_sample (pre-clip, post-backward)."""
    per_param = []
    for p in model.parameters():
        if hasattr(p, 'grad_sample') and p.grad_sample is not None:
            gs = p.grad_sample
            per_param.append(gs.reshape(gs.shape[0], -1).norm(2, dim=1))
    if not per_param:
        return None
    return torch.stack(per_param, dim=1).norm(2, dim=1)


def train_epoch_fixed(model, optimizer, train_loader, device, criterion, max_physical_batch_size=16):
    model.train()
    losses = []
    with BatchMemoryManager(
        data_loader=train_loader,
        max_physical_batch_size=max_physical_batch_size,
        optimizer=optimizer
    ) as loader:
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return np.mean(losses)


def train_epoch_adaptive(model, optimizer, train_loader, device, criterion, controller, max_physical_batch_size=16):
    model.train()
    losses = []
    total, clipped = 0, 0

    with BatchMemoryManager(
        data_loader=train_loader,
        max_physical_batch_size=max_physical_batch_size,
        optimizer=optimizer
    ) as loader:
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()

            norms = compute_grad_norms(model)
            if norms is not None:
                c = controller['c']
                total += len(norms)
                clipped += int((norms >= c).sum().item())

            optimizer.step()
            losses.append(loss.item())

    clipped_ratio = clipped / total if total > 0 else 0.0
    old_c = controller['c']
    tol = 0.05
    speed = 0.15

    if clipped_ratio > TARGET_RATIO + tol:
        excess = clipped_ratio - (TARGET_RATIO + tol)
        new_c = old_c * (1 + speed * (1 + excess * 2))
    elif clipped_ratio < TARGET_RATIO - tol:
        deficit = (TARGET_RATIO - tol) - clipped_ratio
        new_c = old_c * (1 - speed * (1 + deficit * 2))
    else:
        new_c = old_c

    new_c = max(0.1, min(10.0, new_c))
    new_c = 0.5 * old_c + 0.5 * new_c
    controller['c'] = new_c

    if hasattr(optimizer, 'max_grad_norm'):
        optimizer.max_grad_norm = new_c

    return np.mean(losses), clipped_ratio, new_c


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return correct / len(test_loader.dataset)


def run_fixed(sigma, device, data_root, batch_size, max_physical_batch_size):
    train_loader, test_loader = make_loaders(data_root, batch_size)
    model = FashionCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    pe = PrivacyEngine(secure_mode=False)
    model, optimizer, train_loader = pe.make_private(
        module=model, optimizer=optimizer, data_loader=train_loader,
        noise_multiplier=sigma, max_grad_norm=FIXED_C,
    )

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch_fixed(model, optimizer, train_loader, device, criterion, max_physical_batch_size)
        eps = pe.accountant.get_epsilon(delta=DELTA)
        print(f"  [fixed  σ={sigma}] epoch {epoch:2d}/{EPOCHS}  loss={loss:.4f}  ε={eps:.2f}")

    eps = pe.accountant.get_epsilon(delta=DELTA)
    acc = evaluate(model, test_loader, device)
    return eps, acc


def run_adaptive(sigma, device, data_root, batch_size, max_physical_batch_size):
    train_loader, test_loader = make_loaders(data_root, batch_size)
    model = FashionCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    initial_c = 1.0
    controller = {'c': initial_c}

    pe = PrivacyEngine(secure_mode=False)
    model, optimizer, train_loader = pe.make_private(
        module=model, optimizer=optimizer, data_loader=train_loader,
        noise_multiplier=sigma, max_grad_norm=initial_c,
    )

    for epoch in range(1, EPOCHS + 1):
        loss, clip_ratio, new_c = train_epoch_adaptive(
            model, optimizer, train_loader, device, criterion, controller, max_physical_batch_size)
        eps = pe.accountant.get_epsilon(delta=DELTA)
        print(f"  [adapt  σ={sigma}] epoch {epoch:2d}/{EPOCHS}  loss={loss:.4f}"
              f"  ε={eps:.2f}  C={new_c:.3f}  clip%={clip_ratio:.1%}")

    eps = pe.accountant.get_epsilon(delta=DELTA)
    acc = evaluate(model, test_loader, device)
    print(f"  [adapt  σ={sigma}] final C={controller['c']:.4f}")
    return eps, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default="../fashion_mnist")
    parser.add_argument("--output", default="fashion_epsilon_tradeoff.csv")
    parser.add_argument("--batch-size", type=int, default=150)
    parser.add_argument("--max-physical-batch-size", type=int, default=16)
    args = parser.parse_args()
    device = torch.device(args.device)

    cache_file = args.output.replace('.csv', '_cache.pt')
    if os.path.exists(cache_file):
        cache = torch.load(cache_file, weights_only=False)
    else:
        cache = {}

    fixed_results = cache.get('fixed', {})
    adapt_results = cache.get('adaptive', {})

    for sigma in SIGMAS:
        key = str(sigma)

        if key not in fixed_results:
            print(f"\n{'='*55}\nFixed C={FIXED_C}, sigma={sigma}\n{'='*55}")
            eps, acc = run_fixed(sigma, device, args.data_root, args.batch_size, args.max_physical_batch_size)
            print(f"  => ε={eps:.3f}, acc={acc*100:.2f}%")
            fixed_results[key] = (eps, acc)
            cache['fixed'] = fixed_results
            torch.save(cache, cache_file)
        else:
            eps, acc = fixed_results[key]
            print(f"[cached] fixed  sigma={sigma}: ε={eps:.3f}, acc={acc*100:.2f}%")

        if key not in adapt_results:
            print(f"\n{'='*55}\nAdaptive C, sigma={sigma}\n{'='*55}")
            eps, acc = run_adaptive(sigma, device, args.data_root, args.batch_size, args.max_physical_batch_size)
            print(f"  => ε={eps:.3f}, acc={acc*100:.2f}%")
            adapt_results[key] = (eps, acc)
            cache['adaptive'] = adapt_results
            torch.save(cache, cache_file)
        else:
            eps, acc = adapt_results[key]
            print(f"[cached] adaptive sigma={sigma}: ε={eps:.3f}, acc={acc*100:.2f}%")

    # Write CSV
    with open(args.output, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['sigma', 'method', 'epsilon', 'accuracy'])
        for sigma in SIGMAS:
            key = str(sigma)
            if key in fixed_results:
                eps, acc = fixed_results[key]
                w.writerow([sigma, 'fixed', f'{eps:.4f}', f'{acc*100:.2f}'])
            if key in adapt_results:
                eps, acc = adapt_results[key]
                w.writerow([sigma, 'adaptive', f'{eps:.4f}', f'{acc*100:.2f}'])
    print(f"\nSaved results to {args.output}")

    # Print table
    print(f"\n{'sigma':>6} | {'method':>8} | {'epsilon':>8} | {'accuracy':>8}")
    print("-" * 40)
    for sigma in SIGMAS:
        key = str(sigma)
        for method, store in [('fixed', fixed_results), ('adaptive', adapt_results)]:
            if key in store:
                eps, acc = store[key]
                print(f"{sigma:>6} | {method:>8} | {eps:>8.3f} | {acc*100:>7.2f}%")

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 6))

        fx = sorted([(fixed_results[str(s)][0], fixed_results[str(s)][1]*100)
                     for s in SIGMAS if str(s) in fixed_results])
        ax_x = sorted([(adapt_results[str(s)][0], adapt_results[str(s)][1]*100)
                       for s in SIGMAS if str(s) in adapt_results])

        if fx:
            eps_f, acc_f = zip(*fx)
            ax.plot(eps_f, acc_f, 'bs-', markersize=8, linewidth=2, label=f'Fixed C={FIXED_C}')
            for e, a in zip(eps_f, acc_f):
                ax.annotate(f'{a:.1f}%', (e, a), textcoords='offset points',
                            xytext=(4, 5), fontsize=8)

        if ax_x:
            eps_a, acc_a = zip(*ax_x)
            ax.plot(eps_a, acc_a, 'ro-', markersize=8, linewidth=2,
                    label=f'Adaptive C (target={TARGET_RATIO:.0%})')
            for e, a in zip(eps_a, acc_a):
                ax.annotate(f'{a:.1f}%', (e, a), textcoords='offset points',
                            xytext=(4, -12), fontsize=8, color='red')

        ax.set_xlabel('Privacy Budget ε (δ=1e-5)', fontsize=12)
        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax.set_title('Fashion-MNIST: Privacy-Utility Trade-off\nFixed C vs Adaptive C', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_png = args.output.replace('.csv', '.png')
        plt.savefig(out_png, dpi=150)
        print(f"Saved plot to {out_png}")
    except ImportError:
        print("matplotlib not available, skip plot")


if __name__ == '__main__':
    main()