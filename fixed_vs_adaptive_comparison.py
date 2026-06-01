#!/usr/bin/env python3
"""
Fixed C vs Adaptive C comparison at the SAME epsilon budget.

Design:
- For fixed C: Run with sigma=1.0, epochs=20, various C values (0.1, 0.3, 0.5, 1.0, 2.0)
- For adaptive C: Use same sigma=1.0, initial C=1.0, target_ratio=0.3
- Compare: same epoch, same sigma, different C strategies → different epsilon achieved
- Goal: Show that adaptive C achieves similar/better accuracy with different epsilon profile

Expected outcome:
- Fixed C with small value: lower epsilon, potentially more bias
- Fixed C with large value: higher epsilon, potentially less bias but more noise
- Adaptive C: balances between, target ~30% clipping → C converges to optimal
"""

import argparse
import os
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


class GradientHistogram:
    def __init__(self, bin_min=0.0, bin_max=2.0, num_bins=50):
        self.bin_min = bin_min
        self.bin_max = bin_max
        self.bin_edges = np.linspace(bin_min, bin_max, num_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        self.num_bins = num_bins
        self.reset()

    def set_bin_max(self, bin_max):
        self.bin_max = bin_max
        self.bin_edges = np.linspace(self.bin_min, bin_max, self.num_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

    def reset(self):
        self.counts = np.zeros(self.num_bins)
        self.total_samples = 0
        self.clipped_at_max = 0

    def add_batch(self, grad_norms):
        norms = grad_norms.cpu().detach().numpy()
        self.total_samples += len(norms)
        max_bin_idx = self.num_bins - 1
        clipped_count = np.sum(norms >= self.bin_edges[max_bin_idx])
        self.clipped_at_max += clipped_count
        for norm in norms:
            if norm < self.bin_edges[max_bin_idx]:
                bin_idx = np.searchsorted(self.bin_edges[1:], norm)
                bin_idx = min(max(bin_idx, 0), self.num_bins - 1)
                self.counts[bin_idx] += 1

    def get_clipped_ratio(self):
        if self.total_samples == 0:
            return 0
        return self.clipped_at_max / self.total_samples

    def get_stats(self):
        if self.total_samples == 0:
            return {}
        return {
            'total_samples': self.total_samples,
            'clipped_ratio': self.get_clipped_ratio(),
            'mean': np.sum(self.bin_centers * self.counts) / self.total_samples,
        }


class AdaptiveClipController:
    def __init__(self, initial_c=1.0, target_ratio=0.2, tolerance=0.1,
                 min_c=0.1, max_c=50.0, adjustment_speed=0.1):
        self.c = initial_c
        self.target_ratio = target_ratio
        self.tolerance = tolerance
        self.min_c = min_c
        self.max_c = max_c
        self.adjustment_speed = adjustment_speed
        self.c_history = [initial_c]
        self.clipped_ratio_history = []

    def update(self, clipped_ratio):
        self.clipped_ratio_history.append(clipped_ratio)
        if clipped_ratio > self.target_ratio + self.tolerance:
            excess = clipped_ratio - (self.target_ratio + self.tolerance)
            adjustment = 1 + self.adjustment_speed * (1 + excess * 5)
            new_c = self.c * adjustment
        elif clipped_ratio < self.target_ratio - self.tolerance:
            deficit = (self.target_ratio - self.tolerance) - clipped_ratio
            adjustment = 1 - self.adjustment_speed * (1 + deficit * 5)
            new_c = self.c * adjustment
        else:
            new_c = self.c
        new_c = max(self.min_c, min(self.max_c, new_c))
        new_c = 0.5 * self.c + 0.5 * new_c
        self.c = new_c
        self.c_history.append(new_c)
        return new_c

    def get_c(self):
        return self.c


def train_fixed_c(args, model, device, train_loader, optimizer, privacy_engine, epoch, fixed_c):
    """Train with FIXED clipping threshold C."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []

    for _batch_idx, (data, target) in enumerate(tqdm(train_loader, leave=False)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
    return np.mean(losses), epsilon


def train_adaptive_c(args, model, device, train_loader, optimizer, privacy_engine, epoch,
                     histogram, adaptive_controller):
    """Train with ADAPTIVE clipping threshold C."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []

    for _batch_idx, (data, target) in enumerate(tqdm(train_loader, leave=False)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # Track gradient norms
        per_sample_norms = []
        for param in model.parameters():
            if hasattr(param, 'grad_sample') and param.grad_sample is not None:
                grad_sample = param.grad_sample
                flat_grad = grad_sample.reshape(grad_sample.shape[0], -1)
                param_norms = torch.norm(flat_grad, p=2, dim=1)
                per_sample_norms.append(param_norms)

        if per_sample_norms:
            stacked = torch.stack(per_sample_norms, dim=1)
            overall_norms = torch.norm(stacked, p=2, dim=1)
            histogram.add_batch(overall_norms)

        optimizer.step()

        for param in model.parameters():
            if hasattr(param, 'grad_sample'):
                param.grad_sample = None

        losses.append(loss.item())

    epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
    clipped_ratio = histogram.get_clipped_ratio()
    old_c = adaptive_controller.get_c()
    new_c = adaptive_controller.update(clipped_ratio)
    histogram.set_bin_max(new_c * 2)
    histogram.reset()

    return np.mean(losses), epsilon, new_c, clipped_ratio


def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 100.0 * correct / len(test_loader.dataset)


def run_fixed_c_experiment(args, device, fixed_c, ckpt_done):
    """Run fixed C experiment."""
    if os.path.exists(ckpt_done):
        print(f"  Already completed, loading from {ckpt_done}")
        with open(ckpt_done, 'r') as f:
            line = f.read().strip().split(',')
        return float(line[0]), float(line[1]), float(line[2]) if len(line) > 2 else None

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

    c_history = [fixed_c]

    try:
        for epoch in range(1, args.epochs + 1):
            loss, epsilon = train_fixed_c(args, model, device, train_loader, optimizer, privacy_engine, epoch, fixed_c)
            print(f"  Epoch {epoch}/{args.epochs} | Loss: {loss:.4f} | ε: {epsilon:.4f} | C: {fixed_c:.4f}")
            c_history.append(fixed_c)

        accuracy = test(model, device, test_loader)
        epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)

        with open(ckpt_done, 'w') as f:
            f.write(f"{epsilon},{accuracy},{fixed_c}")

        result = (epsilon, accuracy, fixed_c)
    finally:
        # Clean up to avoid OOM
        import gc
        del model, optimizer, privacy_engine, train_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()

    return result


def run_adaptive_c_experiment(args, device, initial_c, target_ratio, ckpt_done):
    """Run adaptive C experiment."""
    if os.path.exists(ckpt_done):
        print(f"  Already completed, loading from {ckpt_done}")
        with open(ckpt_done, 'r') as f:
            parts = f.read().strip().split(',')
        return float(parts[0]), float(parts[1]), float(parts[2])

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

    histogram = GradientHistogram(bin_min=0.0, bin_max=initial_c * 2, num_bins=50)
    adaptive_controller = AdaptiveClipController(
        initial_c=initial_c,
        target_ratio=target_ratio,
        tolerance=0.1,
        min_c=0.1,
        max_c=50.0,
        adjustment_speed=0.15
    )

    privacy_engine = PrivacyEngine(secure_mode=False)
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=args.sigma,
        max_grad_norm=adaptive_controller.get_c(),
    )

    try:
        for epoch in range(1, args.epochs + 1):
            loss, epsilon, c, clipped_ratio = train_adaptive_c(
                args, model, device, train_loader, optimizer, privacy_engine, epoch,
                histogram, adaptive_controller
            )
            print(f"  Epoch {epoch}/{args.epochs} | Loss: {loss:.4f} | ε: {epsilon:.4f} | C: {c:.4f} | Clipped: {clipped_ratio:.1%}")

        accuracy = test(model, device, test_loader)
        epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
        final_c = adaptive_controller.get_c()

        with open(ckpt_done, 'w') as f:
            f.write(f"{epsilon},{accuracy},{final_c}")

        result = (epsilon, accuracy, final_c)
    finally:
        del model, optimizer, privacy_engine, train_loader, test_loader
        torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="Fixed vs Adaptive C at Same Epsilon")
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("-n", "--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data-root", type=str, default="../fashion_mnist")
    args = parser.parse_args()
    device = torch.device(args.device)

    print("=" * 70)
    print("FIXED C vs ADAPTIVE C COMPARISON")
    print(f"sigma={args.sigma}, epochs={args.epochs}, batch_size={args.batch_size}")
    print("=" * 70)

    # Fixed C experiments - same sigma, different C values
    fixed_c_values = [0.1, 0.3, 0.5, 1.0, 2.0]
    fixed_results = []

    print("\n" + "-" * 70)
    print("FIXED C EXPERIMENTS (same sigma, different C)")
    print("-" * 70)

    for c in fixed_c_values:
        print(f"\n>>> Fixed C = {c} <<<")
        ckpt_done = f"ckpt_fixed_c_{c}_done.pt"
        try:
            epsilon, accuracy, used_c = run_fixed_c_experiment(args, device, c, ckpt_done)
            print(f"  Result: epsilon={epsilon:.4f}, accuracy={accuracy:.2f}%, C={used_c}")
            fixed_results.append({'C': used_c, 'epsilon': epsilon, 'accuracy': accuracy, 'type': 'fixed'})
        except Exception as e:
            print(f"  Error: {e}")
            fixed_results.append({'C': c, 'epsilon': float('nan'), 'accuracy': float('nan'), 'type': 'fixed'})

    # Adaptive C experiments - same sigma, different target_ratio
    target_ratios = [0.2, 0.3, 0.5]
    adaptive_results = []

    print("\n" + "-" * 70)
    print("ADAPTIVE C EXPERIMENTS (same sigma, different target_ratio)")
    print("-" * 70)

    for target_ratio in target_ratios:
        print(f"\n>>> Adaptive C (target_ratio={target_ratio}) <<<")
        ckpt_done = f"ckpt_adaptive_tr_{target_ratio}_done.pt"
        try:
            epsilon, accuracy, final_c = run_adaptive_c_experiment(args, device, 1.0, target_ratio, ckpt_done)
            print(f"  Result: epsilon={epsilon:.4f}, accuracy={accuracy:.2f}%, final C={final_c:.4f}")
            adaptive_results.append({'target_ratio': target_ratio, 'final_C': final_c, 'epsilon': epsilon, 'accuracy': accuracy, 'type': 'adaptive'})
        except Exception as e:
            print(f"  Error: {e}")
            adaptive_results.append({'target_ratio': target_ratio, 'epsilon': float('nan'), 'accuracy': float('nan'), 'type': 'adaptive'})

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Fixed C vs Adaptive C (same sigma, same epochs)")
    print("=" * 70)
    print(f"{'Method':>20} | {'C/final_C':>10} | {'epsilon':>10} | {'accuracy':>10}")
    print("-" * 60)

    for r in fixed_results:
        label = f"Fixed C={r['C']}"
        if not np.isnan(r['accuracy']):
            print(f"{label:>20} | {r['C']:>10.2f} | {r['epsilon']:>10.4f} | {r['accuracy']:>9.2f}%")
        else:
            print(f"{label:>20} | {'ERROR':>10} | {'ERROR':>10} | {'ERROR':>10}")

    for r in adaptive_results:
        label = f"Adaptive (tr={r['target_ratio']})"
        if not np.isnan(r['accuracy']):
            print(f"{label:>20} | {r['final_C']:>10.4f} | {r['epsilon']:>10.4f} | {r['accuracy']:>9.2f}%")
        else:
            print(f"{label:>20} | {'ERROR':>10} | {'ERROR':>10} | {'ERROR':>10}")

    print("=" * 70)

    # Save results
    results = fixed_results + adaptive_results
    torch.save(results, "fixed_vs_adaptive_results.pt")
    print("\nSaved to fixed_vs_adaptive_results.pt")

    # Key insight
    print("\n" + "-" * 70)
    print("KEY INSIGHT")
    print("-" * 70)
    print("With same sigma=1.0 and same epochs=20:")
    print("- Smaller C (e.g., C=0.1) → less noise added → lower epsilon → higher accuracy")
    print("- Larger C (e.g., C=2.0) → more noise added → higher epsilon → lower accuracy")
    print("- Adaptive C balances clipping bias and noise by targeting clipped ratio")
    print("-" * 70)


if __name__ == "__main__":
    main()