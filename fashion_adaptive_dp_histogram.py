#!/usr/bin/env python3
"""
Fashion-MNIST training with Opacus DP-SGD and DP histogram-based adaptive gradient clipping.

Supports two adaptive methods:
  --method ratio  : adjust C to keep clipped_ratio near target_ratio (DP-noisy)
  --method mse    : set C = argmin MSE(C) from DP histogram each epoch

Privacy accounting:
  - DP-SGD cost: tracked by Opacus RDP accountant
  - Histogram cost: epsilon_hist per epoch, composed additively (basic composition)
  - Total reported epsilon = epsilon_sgd + epsilon_hist * epochs
"""

import argparse
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


class GradientHistogram:
    """Tracks TRUE (pre-clip) gradient norm distribution with DP noise."""

    def __init__(self, bin_min=0.0, bin_max=2.0, num_bins=50):
        self.bin_min = bin_min
        self.bin_max = bin_max
        self.bin_edges = np.linspace(bin_min, bin_max, num_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        self.num_bins = num_bins
        self.current_c = bin_max / 2
        self.reset()

    def set_bin_max(self, bin_max):
        self.bin_max = bin_max
        self.bin_edges = np.linspace(self.bin_min, bin_max, self.num_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

    def set_current_c(self, c):
        self.current_c = c

    def reset(self):
        self.counts = np.zeros(self.num_bins)
        self.total_samples = 0
        self.clipped_count = 0

    def add_batch(self, grad_norms):
        norms = grad_norms.cpu().detach().numpy()
        self.total_samples += len(norms)
        self.clipped_count += int(np.sum(norms >= self.current_c))
        indices = np.clip(
            np.searchsorted(self.bin_edges[1:], norms), 0, self.num_bins - 1
        )
        for idx in indices:
            self.counts[idx] += 1

    def get_clipped_ratio(self):
        if self.total_samples == 0:
            return 0.0
        return self.clipped_count / self.total_samples

    def get_noisy_clipped_ratio(self, epsilon_hist):
        if self.total_samples == 0:
            return 0.0
        noisy_count = self.clipped_count + np.random.laplace(0, 1.0 / epsilon_hist)
        noisy_count = max(0.0, min(float(self.total_samples), noisy_count))
        return noisy_count / self.total_samples

    def get_stats(self):
        if self.total_samples == 0:
            return {}
        mean = np.sum(self.bin_centers * self.counts) / self.total_samples
        return {
            'total_samples': self.total_samples,
            'clipped_ratio': self.get_clipped_ratio(),
            'mean': mean,
        }


def compute_optimal_c_mse(bin_centers, noisy_counts, sigma, batch_size,
                           c_min=0.05, c_max=5.0, n_grid=200, var_weight=1.0):
    """Find C* = argmin [Bias²(C) + var_weight * Variance(C)] using DP histogram."""
    counts = np.maximum(noisy_counts, 0.0)
    N = counts.sum()
    if N <= 0:
        return (c_min + c_max) / 2, {}

    candidates = np.linspace(c_min, c_max, n_grid)
    mse_vals = np.empty(n_grid)
    bias_vals = np.empty(n_grid)
    var_vals = np.empty(n_grid)

    for i, C in enumerate(candidates):
        mask = bin_centers > C
        bias2 = np.sum(counts[mask] * (bin_centers[mask] - C) ** 2) / (N ** 2)
        variance = var_weight * (sigma * C) ** 2 / (batch_size ** 2)
        mse_vals[i] = bias2 + variance
        bias_vals[i] = bias2
        var_vals[i] = variance

    best = np.argmin(mse_vals)
    return candidates[best], {
        'candidates': candidates,
        'mse': mse_vals,
        'bias2': bias_vals,
        'variance': var_vals,
        'c_star': candidates[best],
    }


class AdaptiveClipController:
    def __init__(self, initial_c=1.0, target_ratio=0.2, tolerance=0.05,
                 min_c=0.1, max_c=10.0, adjustment_speed=0.15, method='ratio'):
        self.c = initial_c
        self.target_ratio = target_ratio
        self.tolerance = tolerance
        self.min_c = min_c
        self.max_c = max_c
        self.adjustment_speed = adjustment_speed
        self.method = method

        self.c_history = [initial_c]
        self.clipped_ratio_history = []
        self.mse_history = []
        self.bias2_history = []
        self.variance_history = []

    def update_ratio(self, clipped_ratio):
        self.clipped_ratio_history.append(clipped_ratio)
        if clipped_ratio > self.target_ratio + self.tolerance:
            excess = clipped_ratio - (self.target_ratio + self.tolerance)
            new_c = self.c * (1 + self.adjustment_speed * (1 + excess * 2))
        elif clipped_ratio < self.target_ratio - self.tolerance:
            deficit = (self.target_ratio - self.tolerance) - clipped_ratio
            new_c = self.c * (1 - self.adjustment_speed * (1 + deficit * 2))
        else:
            new_c = self.c
        new_c = max(self.min_c, min(self.max_c, new_c))
        new_c = 0.5 * self.c + 0.5 * new_c
        self.c = new_c
        self.c_history.append(new_c)
        return new_c

    def update_mse(self, bin_centers, noisy_counts, sigma, batch_size, var_weight=1.0):
        c_star, info = compute_optimal_c_mse(
            bin_centers, noisy_counts, sigma, batch_size,
            c_min=self.min_c, c_max=self.max_c, var_weight=var_weight,
        )
        new_c = 0.5 * self.c + 0.5 * c_star
        new_c = max(self.min_c, min(self.max_c, new_c))
        self.c = new_c
        self.c_history.append(new_c)
        if info:
            best = np.argmin(info['mse'])
            self.mse_history.append(info['mse'][best])
            self.bias2_history.append(info['bias2'][best])
            self.variance_history.append(info['variance'][best])
        return new_c, info

    def get_c(self):
        return self.c


class FashionCNN(nn.Module):
    def __init__(self, fc1_features=256):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, fc1_features)
        self.fc2 = nn.Linear(fc1_features, 10)
        self._init_weights()

    def _init_weights(self):
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


def compute_grad_norms(model):
    per_param = []
    for p in model.parameters():
        if hasattr(p, 'grad_sample') and p.grad_sample is not None:
            gs = p.grad_sample
            per_param.append(gs.reshape(gs.shape[0], -1).norm(2, dim=1))
    if not per_param:
        return None
    return torch.stack(per_param, dim=1).norm(2, dim=1)


def train_epoch(args, model, device, train_loader, optimizer, privacy_engine, epoch,
                histogram, controller, epsilon_hist_per_epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []

    with BatchMemoryManager(
        data_loader=train_loader,
        max_physical_batch_size=args.max_physical_batch_size,
        optimizer=optimizer,
    ) as loader:
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()

            norms = compute_grad_norms(model)
            if norms is not None:
                histogram.set_current_c(controller.get_c())
                histogram.add_batch(norms)

            optimizer.step()
            losses.append(loss.item())

    epsilon_sgd = privacy_engine.accountant.get_epsilon(delta=args.delta)
    epsilon_hist_total = epsilon_hist_per_epoch * epoch
    epsilon_total = epsilon_sgd + epsilon_hist_total
    print(
        f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f} "
        f"(ε_sgd={epsilon_sgd:.2f}, ε_hist={epsilon_hist_total:.2f}, "
        f"ε_total={epsilon_total:.2f}, δ={args.delta})"
    )

    old_c = controller.get_c()
    true_ratio = histogram.get_clipped_ratio()

    if controller.method == 'mse':
        noisy_counts = histogram.counts + np.random.laplace(
            0, 1.0 / epsilon_hist_per_epoch, histogram.num_bins)
        noisy_counts = np.maximum(noisy_counts, 0.0)
        new_c, mse_info = controller.update_mse(
            histogram.bin_centers, noisy_counts, args.sigma, args.batch_size,
            var_weight=args.mse_var_weight)
        c_star = mse_info.get('c_star', new_c) if mse_info else new_c
        print(
            f"  [MSE] C: {new_c:.4f} (was {old_c:.4f}, c*={c_star:.4f})"
            f" | True clipped: {true_ratio:.1%}"
            f" | MSE={controller.mse_history[-1]:.2e}"
            f" | Bias²={controller.bias2_history[-1]:.2e}"
            f" | Var={controller.variance_history[-1]:.2e}"
        )
    else:
        noisy_ratio = histogram.get_noisy_clipped_ratio(epsilon_hist_per_epoch)
        new_c = controller.update_ratio(noisy_ratio)
        print(
            f"  [ratio] C: {new_c:.4f} (was {old_c:.4f})"
            f" | Noisy clipped: {noisy_ratio:.1%}"
            f" | True clipped: {true_ratio:.1%}"
            f" | Target: {args.target_ratio:.0%}"
        )

    if hasattr(optimizer, 'max_grad_norm'):
        optimizer.max_grad_norm = new_c

    histogram.set_bin_max(new_c * 2)
    histogram.reset()
    return np.mean(losses)


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            correct += model(data).argmax(dim=1).eq(target).sum().item()
    return correct / len(test_loader.dataset)


def main():
    parser = argparse.ArgumentParser(
        description="Fashion-MNIST with Opacus DP-SGD and Adaptive Clipping",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-b", "--batch-size", type=int, default=150)
    parser.add_argument("--max-physical-batch-size", type=int, default=16)
    parser.add_argument("--test-batch-size", type=int, default=1024)
    parser.add_argument("-n", "--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("-c", "--initial-c", type=float, default=1.0)
    parser.add_argument("--target-ratio", type=float, default=0.2)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data-root", type=str, default="../fashion_mnist")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--epsilon-hist", type=float, default=1.0,
                        help="Per-epoch privacy budget for DP histogram query")
    parser.add_argument("--method", type=str, default="ratio", choices=["ratio", "mse"],
                        help="Adaptive C update method")
    parser.add_argument("--mse-var-weight", type=float, default=1.0,
                        help="Variance term weight in MSE objective (method=mse only)")
    parser.add_argument("--fc1-features", type=int, default=256,
                        help="FashionCNN fc1 hidden size (256 avoids OOM with grad_sample)")
    args = parser.parse_args()
    device = torch.device(args.device)

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((FASHION_MNIST_MEAN,), (FASHION_MNIST_STD,)),
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(args.data_root, train=True, download=True, transform=tf),
        batch_size=args.batch_size, num_workers=0, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST(args.data_root, train=False, transform=tf),
        batch_size=args.test_batch_size, num_workers=0, pin_memory=True,
    )

    histogram = GradientHistogram(bin_min=0.0, bin_max=args.initial_c * 2, num_bins=50)
    controller = AdaptiveClipController(
        initial_c=args.initial_c,
        target_ratio=args.target_ratio,
        tolerance=0.05,
        min_c=0.1,
        max_c=10.0,
        adjustment_speed=0.15,
        method=args.method,
    )

    model = FashionCNN(fc1_features=args.fc1_features).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    pe = PrivacyEngine(secure_mode=False)
    model, optimizer, train_loader = pe.make_private(
        module=model, optimizer=optimizer, data_loader=train_loader,
        noise_multiplier=args.sigma, max_grad_norm=controller.get_c(),
    )

    for epoch in range(1, args.epochs + 1):
        train_epoch(args, model, device, train_loader, optimizer, pe, epoch,
                    histogram, controller, args.epsilon_hist)

    acc = evaluate(model, test_loader, device)
    eps_sgd = pe.accountant.get_epsilon(delta=args.delta)
    eps_total = eps_sgd + args.epsilon_hist * args.epochs
    print(f"\nFinal: acc={acc*100:.2f}%  ε_sgd={eps_sgd:.3f}  ε_total={eps_total:.3f}")
    print(f"Final C: {controller.c:.4f}")
    print(f"C history: {[f'{c:.3f}' for c in controller.c_history]}")

    tag = f"fashion_{args.method}_s{args.sigma}_b{args.batch_size}_e{args.epochs}_vw{args.mse_var_weight}"
    torch.save({
        'accuracy': acc,
        'epsilon_sgd': eps_sgd,
        'epsilon_total': eps_total,
        'c_history': controller.c_history,
        'clipped_ratio_history': controller.clipped_ratio_history,
        'mse_history': controller.mse_history,
        'bias2_history': controller.bias2_history,
        'variance_history': controller.variance_history,
        'method': args.method,
        'args': vars(args),
    }, f"fashion_adaptive_results_{tag}.pt")

    if args.save_model:
        torch.save(model.state_dict(), f"fashion_cnn_{tag}.pt")


if __name__ == "__main__":
    main()
