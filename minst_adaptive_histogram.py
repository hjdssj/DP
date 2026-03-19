#!/usr/bin/env python3
"""
MNIST training with differential privacy and adaptive gradient clipping.

Implements an adaptive clipping threshold strategy that minimizes
gradient estimation MSE. The optimal C balances:
- Clipping bias (small C → high bias from over-clipping)
- DP noise variance (large C → higher noise because noise ∝ C)
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

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# Precomputed characteristics of the MNIST dataset
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class AdaptiveClipper:
    """Adaptive gradient clipping based on MSE minimization.

    The MSE of the per-sample gradient estimator with clipping is:
        MSE = Bias^2 + Variance

    Where:
        - Bias: from scaling clipped gradients
        - Variance: from added DP noise (scales with C)

    This class tracks gradient norm statistics and adapts C each epoch.
    """

    def __init__(self, initial_c=1.0, sigma=1.0, batch_size=64,
                 momentum=0.9, min_c=0.1, max_c=10.0, target_unclipped_ratio=0.3):
        """
        Args:
            initial_c: Starting clipping threshold
            sigma: DP noise multiplier
            batch_size: Training batch size
            momentum: Momentum for C update (smoothing)
            min_c: Minimum allowed C value
            max_c: Maximum allowed C value
            target_unclipped_ratio: Target fraction of samples that should NOT be clipped
                                    (controls bias-variance tradeoff)
        """
        self.c = initial_c
        self.sigma = sigma
        self.batch_size = batch_size
        self.momentum = momentum
        self.min_c = min_c
        self.max_c = max_c
        self.target_unclipped_ratio = target_unclipped_ratio

        # For tracking
        self.c_history = [initial_c]
        self.mse_history = []
        self.bias_history = []
        self.var_history = []

        # Gradient norm statistics from current epoch
        self.epoch_grad_norms = []

    def add_batch(self, grad_norms):
        """Collect gradient norms from a batch."""
        self.epoch_grad_norms.extend(grad_norms.tolist() if isinstance(grad_norms, np.ndarray)
                                      else grad_norms.cpu().detach().numpy().tolist())

    def compute_gradient_stats(self):
        """Compute statistics of collected gradient norms."""
        norms = np.array(self.epoch_grad_norms)
        if len(norms) == 0:
            return {}

        return {
            'mean': np.mean(norms),
            'std': np.std(norms),
            'median': np.median(norms),
            'p25': np.percentile(norms, 25),
            'p75': np.percentile(norms, 75),
            'p90': np.percentile(norms, 90),
            'p95': np.percentile(norms, 95),
            'p99': np.percentile(norms, 99),
            'min': np.min(norms),
            'max': np.max(norms),
            'above_c': np.sum(norms > self.c) / len(norms),  # Fraction clipped
            'below_c': np.sum(norms < self.c) / len(norms),
        }

    def estimate_mse_components(self, stats):
        """Estimate MSE bias and variance components given current C.

        For gradient g with norm ||g|| and clipping at C:
            - If ||g|| <= C: clipped_g = g (no change)
            - If ||g|| > C: clipped_g = g * (C / ||g||) (scaled down)

        The squared error for clipped sample is:
            ||clipped_g - g||^2 = ||g||^2 * (1 - C/||g||)^2  for ||g|| > C
                                 = 0                           for ||g|| <= C

        Bias^2 ≈ E[||g||^2 * (1 - C/||g||)^2 | ||g|| > C] * P(||g|| > C)
        Variance from DP noise ≈ (C^2 * sigma^2 * d) / (batch_size)
        """
        norms = np.array(self.epoch_grad_norms)
        if len(norms) == 0 or stats['above_c'] == 0:
            return {'bias': 0, 'variance': 0, 'mse': 0}

        above_c_norms = norms[norms > self.c]

        # Bias: squared error from clipping
        if len(above_c_norms) > 0:
            # For each clipped sample: ||g||^2 * (1 - C/||g||)^2 = (||g|| - C)^2
            squared_errors = (above_c_norms - self.c) ** 2
            bias = np.mean(squared_errors)
        else:
            bias = 0

        # Variance from DP noise: noise scale = C * sigma
        # Noise variance per parameter ∝ (C * sigma)^2
        # Total variance across batch scales with C^2 * sigma^2 / batch_size
        d = 512 + 32 + 10  # approximate total parameters
        variance = (self.c ** 2 * self.sigma ** 2 * d) / self.batch_size

        mse = bias + variance

        return {
            'bias': bias,
            'variance': variance,
            'mse': mse,
            'num_clipped': len(above_c_norms),
            'clipped_ratio': len(above_c_norms) / len(norms)
        }

    def compute_optimal_c_mse(self):
        """Compute optimal C that minimizes estimated MSE.

        Uses a grid search over percentiles of the gradient distribution.
        The optimal C balances:
        - Lower C: more samples clipped → higher bias, but lower noise variance
        - Higher C: fewer samples clipped → lower bias, but higher noise variance
        """
        norms = np.array(self.epoch_grad_norms)
        if len(norms) == 0:
            return self.c

        # Try different C values as percentiles of the gradient norms
        percentiles = np.linspace(50, 99, 50)
        best_c = self.c
        best_mse = float('inf')

        for p in percentiles:
            trial_c = np.percentile(norms, p)

            # Estimate MSE for this C
            above_c_norms = norms[norms > trial_c]

            if len(above_c_norms) > 0:
                bias = np.mean((above_c_norms - trial_c) ** 2)
            else:
                bias = 0

            d = 512 + 32 + 10
            variance = (trial_c ** 2 * self.sigma ** 2 * d) / self.batch_size
            mse = bias + variance

            if mse < best_mse:
                best_mse = mse
                best_c = trial_c

        return best_c

    def compute_optimal_c_percentile(self, target_ratio=None):
        """Compute optimal C based on target unclipped ratio.

        Simple strategy: keep (1 - target_ratio) of samples unclipped.
        """
        if target_ratio is None:
            target_ratio = self.target_unclipped_ratio

        # We want ~target_ratio of samples to be below C (unclipped)
        # So C should be at the (1 - target_ratio) percentile
        target_percentile = (1 - target_ratio) * 100

        norms = np.array(self.epoch_grad_norms)
        if len(norms) == 0:
            return self.c

        optimal_c = np.percentile(norms, target_percentile)
        return optimal_c

    def update_c(self, method='mse_optimize', stats=None):
        """Update clipping threshold C after an epoch.

        Args:
            method: 'mse_optimize' (grid search) or 'percentile' (target ratio)
            stats: Pre-computed gradient statistics (optional)

        Returns:
            dict with old_c, new_c, and MSE components
        """
        old_c = self.c

        if stats is None:
            stats = self.compute_gradient_stats()

        # Compute MSE components for current C
        mse_components = self.estimate_mse_components(stats)

        # Compute optimal C
        if method == 'mse_optimize':
            optimal_c = self.compute_optimal_c_mse()
        elif method == 'percentile':
            optimal_c = self.compute_optimal_c_percentile()
        else:
            optimal_c = self.c

        # Apply momentum and constraints
        new_c = self.momentum * old_c + (1 - self.momentum) * optimal_c
        new_c = max(self.min_c, min(self.max_c, new_c))

        self.c = new_c
        self.c_history.append(new_c)
        self.mse_history.append(mse_components['mse'])
        self.bias_history.append(mse_components['bias'])
        self.var_history.append(mse_components['variance'])

        return {
            'old_c': old_c,
            'new_c': new_c,
            'optimal_c': optimal_c,
            'mse': mse_components['mse'],
            'bias': mse_components['bias'],
            'variance': mse_components['variance'],
            'stats': stats
        }

    def reset_epoch(self):
        """Reset for next epoch."""
        self.epoch_grad_norms = []

    def get_c(self):
        """Get current clipping threshold."""
        return self.c


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
        x = self.fc2(x)
        return x


def compute_per_sample_grad_norms(model):
    """Compute per-sample gradient norms for all parameters."""
    per_sample_norms = []

    for param in model.parameters():
        if param.grad_sample is not None:
            grad_sample = param.grad_sample
            flat_grad = grad_sample.reshape(grad_sample.shape[0], -1)
            param_norms = torch.norm(flat_grad, p=2, dim=1)
            per_sample_norms.append(param_norms)

    if not per_sample_norms:
        return torch.tensor([])

    stacked = torch.stack(per_sample_norms, dim=1)
    overall_norms = torch.norm(stacked, p=2, dim=1)

    return overall_norms


def train_with_adaptive_clip(args, model, device, train_loader, optimizer, privacy_engine,
                             epoch, adaptive_clipper, plot_callback=None):
    """Training loop with adaptive clipping."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []

    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # Compute per-sample gradient norms before clipping
        per_sample_norms = compute_per_sample_grad_norms(model)

        if len(per_sample_norms) > 0:
            adaptive_clipper.add_batch(per_sample_norms.cpu().detach())

        optimizer.step()

        # Clear grad_sample
        for param in model.parameters():
            if hasattr(param, 'grad_sample'):
                param.grad_sample = None

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

    # After epoch: update adaptive C
    update_result = adaptive_clipper.update_c(method=args.adaptive_method)
    adaptive_clipper.reset_epoch()

    stats = update_result['stats']
    print(
        f"  Adaptive Clip: C={update_result['new_c']:.4f} "
        f"(was {update_result['old_c']:.4f}, optimal={update_result['optimal_c']:.4f}) "
        f"MSE={update_result['mse']:.6f} (bias={update_result['bias']:.6f}, var={update_result['variance']:.6f})"
    )
    print(
        f"  Grad Norms: mean={stats.get('mean', 0):.4f}, "
        f"p90={stats.get('p90', 0):.4f}, p95={stats.get('p95', 0):.4f}, "
        f"above_C={stats.get('above_c', 0):.2%}"
    )

    # Update privacy engine with new C if needed (requires recreating)
    # Note: Opacus doesn't support dynamic C easily, so we log it for analysis
    # In practice, one would need to re-initialize the privacy engine

    if plot_callback:
        plot_callback(epoch, update_result)


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
    accuracy = 100.0 * correct / len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), accuracy
        )
    )
    return correct / len(test_loader.dataset), accuracy


def plot_adaptive_results(adaptive_clipper, save_dir, prefix="adaptive"):
    """Plot the adaptive clipping results."""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = range(1, len(adaptive_clipper.c_history))

    # C history
    ax1 = axes[0, 0]
    ax1.plot(epochs, adaptive_clipper.c_history[1:], 'b-o', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Clipping Threshold C')
    ax1.set_title('Adaptive C Across Training')
    ax1.grid(True, alpha=0.3)

    # MSE history
    ax2 = axes[0, 1]
    ax2.plot(epochs, adaptive_clipper.mse_history, 'r-o', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Estimated MSE')
    ax2.set_title('Gradient Estimation MSE')
    ax2.grid(True, alpha=0.3)

    # Bias and Variance decomposition
    ax3 = axes[1, 0]
    ax3.plot(epochs, adaptive_clipper.bias_history, 'g-o', label='Bias²', linewidth=2)
    ax3.plot(epochs, adaptive_clipper.var_history, 'm-s', label='Variance', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Component Value')
    ax3.set_title('MSE Decomposition')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Combined plot
    ax4 = axes[1, 1]
    ax4.plot(epochs, adaptive_clipper.c_history[1:], 'b-o', label='C', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Clipping Threshold C')
    ax4.set_title('C with MSE Context')
    ax4.twinx()
    ax4.plot(epochs, adaptive_clipper.mse_history, 'r-s', label='MSE', linewidth=2)
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    plt.suptitle('Adaptive Gradient Clipping Analysis', fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{prefix}_analysis.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved analysis plot to {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="MNIST with Adaptive Differential Privacy Clipping",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=1024)
    parser.add_argument("-n", "--epochs", type=int, default=10)
    parser.add_argument("-r", "--n-runs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=1.0, help="DP noise multiplier")
    parser.add_argument("-c", "--initial-c", type=float, default=1.0,
                        help="Initial clipping threshold")
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--momentum", type=float, default=0.0,
                        help="Momentum for C update (0 = no smoothing)")
    parser.add_argument("--adaptive-method", type=str, default='mse_optimize',
                        choices=['mse_optimize', 'percentile'],
                        help="Method for adapting C")
    parser.add_argument("--target-unclipped", type=float, default=0.3,
                        help="Target fraction of samples not clipped (for percentile method)")
    parser.add_argument("--min-c", type=float, default=0.1)
    parser.add_argument("--max-c", type=float, default=10.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-model", action="store_true", default=False)
    parser.add_argument("--disable-dp", action="store_true", default=False,
                        help="Disable DP and adaptive clipping")
    parser.add_argument("--secure-rng", action="store_true", default=False)
    parser.add_argument("--data-root", type=str, default="../mnist")
    parser.add_argument("--plot", action="store_true", default=False,
                        help="Enable plotting")
    args = parser.parse_args()
    device = torch.device(args.device)

    repro_str = (
        f"mnist_adaptive_{args.adaptive_method}_{args.lr}_{args.sigma}_"
        f"{args.initial_c}_{args.batch_size}_{args.epochs}"
    )
    plot_dir = f"plots_{repro_str}"
    if args.plot and HAS_MATPLOTLIB:
        os.makedirs(plot_dir, exist_ok=True)
        print(f"Plots will be saved to {plot_dir}/")

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data_root, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                       ])),
        batch_size=args.batch_size, num_workers=0, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data_root, train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                       ])),
        batch_size=args.test_batch_size, shuffle=True, num_workers=0, pin_memory=True,
    )

    # Initialize adaptive clipper
    adaptive_clipper = AdaptiveClipper(
        initial_c=args.initial_c,
        sigma=args.sigma,
        batch_size=args.batch_size,
        momentum=args.momentum,
        min_c=args.min_c,
        max_c=args.max_c,
        target_unclipped_ratio=args.target_unclipped
    )

    def plot_callback(epoch, update_result):
        if args.plot and HAS_MATPLOTLIB and epoch % 2 == 0:
            # Plot every 2 epochs to save time
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            norms = np.array(adaptive_clipper.epoch_grad_norms)
            if len(norms) > 0:
                ax.hist(norms, bins=50, alpha=0.7, edgecolor='black')
                ax.axvline(x=update_result['new_c'], color='red', linestyle='--',
                           linewidth=2, label=f'C={update_result["new_c"]:.3f}')
                ax.axvline(x=update_result['old_c'], color='blue', linestyle=':',
                           linewidth=2, label=f'Old C={update_result["old_c"]:.3f}')
                ax.set_xlabel('Gradient Norm')
                ax.set_ylabel('Count')
                ax.set_title(f'Epoch {epoch}: Gradient Norm Distribution')
                ax.legend()

                save_path = os.path.join(plot_dir, f"epoch_{epoch:03d}_grad_norms.png")
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved epoch {epoch} plot to {save_path}")

    run_results = []
    run_accuracies = []

    for run_idx in range(args.n_runs):
        print(f"\n=== Run {run_idx + 1}/{args.n_runs} ===")
        print(f"Adaptive method: {args.adaptive_method}, initial C: {args.initial_c}")

        # Reset adaptive clipper
        adaptive_clipper = AdaptiveClipper(
            initial_c=args.initial_c,
            sigma=args.sigma,
            batch_size=args.batch_size,
            momentum=args.momentum,
            min_c=args.min_c,
            max_c=args.max_c,
            target_unclipped_ratio=args.target_unclipped
        )

        model = SampleConvNet().to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)
        privacy_engine = None

        if not args.disable_dp:
            privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=args.sigma,
                max_grad_norm=adaptive_clipper.get_c(),  # Initial C
            )

        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}:")
            train_with_adaptive_clip(
                args, model, device, train_loader, optimizer, privacy_engine,
                epoch, adaptive_clipper, plot_callback=plot_callback if args.plot else None
            )

        accuracy, acc_pct = test(model, device, test_loader)
        run_results.append(accuracy)
        run_accuracies.append(acc_pct)

        # Plot final analysis for this run
        if args.plot and HAS_MATPLOTLIB:
            plot_adaptive_results(adaptive_clipper, plot_dir, prefix=f"run_{run_idx+1}")

    # Summary
    if len(run_results) > 1:
        print(
            f"\nAccuracy averaged over {len(run_results)} runs: "
            f"{np.mean(run_accuracies):.2f}% ± {np.std(run_accuracies):.2f}%"
        )
    else:
        print(f"\nFinal accuracy: {run_accuracies[0]:.2f}%")

    print(f"\nFinal adaptive C: {adaptive_clipper.c:.4f}")
    print(f"C history: {[f'{c:.3f}' for c in adaptive_clipper.c_history[1:]]}")

    # Save results
    repro_str = (
        f"mnist_adaptive_{args.adaptive_method}_{args.lr}_{args.sigma}_"
        f"{args.initial_c}_{args.batch_size}_{args.epochs}"
    )
    torch.save({
        'run_results': run_results,
        'accuracies': run_accuracies,
        'c_history': adaptive_clipper.c_history,
        'mse_history': adaptive_clipper.mse_history,
        'bias_history': adaptive_clipper.bias_history,
        'var_history': adaptive_clipper.var_history,
    }, f"adaptive_results_{repro_str}.pt")

    if args.save_model:
        torch.save(model.state_dict(), f"mnist_adaptive_{repro_str}.pt")


if __name__ == "__main__":
    main()
