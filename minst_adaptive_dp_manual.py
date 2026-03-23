#!/usr/bin/env python3
"""
MNIST training with manual DP-SGD and adaptive gradient clipping.

This version implements DP-SGD manually (without Opacus) to have full control
over when clipping happens, enabling accurate gradient norm tracking for the
adaptive clipping algorithm based on MSE minimization.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm


# Precomputed characteristics of the MNIST dataset
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class AdaptiveClipper:
    """Adaptive gradient clipping based on MSE minimization.

    The MSE of the per-sample gradient estimator with clipping is:
        MSE = Bias^2 + Variance

    Where:
        - Bias: from scaling clipped gradients (clipped samples are scaled down)
        - Variance: from added DP noise (scales with C)

    This class tracks gradient norm statistics and adapts C each epoch.
    """

    def __init__(self, initial_c=1.0, sigma=1.0, batch_size=64,
                 momentum=0.9, min_c=0.1, max_c=10.0, target_ratio=0.1):
        """
        Args:
            initial_c: Starting clipping threshold
            sigma: DP noise multiplier
            batch_size: Training batch size
            momentum: Momentum for C update (smoothing)
            min_c: Minimum allowed C value
            max_c: Maximum allowed C value
            target_ratio: Target fraction of samples to clip (10% = 0.1)
        """
        self.c = initial_c
        self.sigma = sigma
        self.batch_size = batch_size
        self.momentum = momentum
        self.min_c = min_c
        self.max_c = max_c
        self.target_ratio = target_ratio

        # For tracking
        self.c_history = [initial_c]
        self.mse_history = []
        self.bias_history = []
        self.var_history = []

        # Gradient norm statistics from current epoch
        self.epoch_grad_norms = []

    def add_batch(self, grad_norms):
        """Collect gradient norms from a batch."""
        if isinstance(grad_norms, torch.Tensor):
            grad_norms = grad_norms.cpu().detach().numpy()
        self.epoch_grad_norms.extend(grad_norms.tolist())

    def compute_gradient_stats(self):
        """Compute statistics of collected gradient norms."""
        norms = np.array(self.epoch_grad_norms)
        if len(norms) == 0:
            return {}

        sorted_norms = np.sort(norms)
        return {
            'mean': np.mean(norms),
            'std': np.std(norms),
            'median': np.median(norms),
            'p50': np.percentile(norms, 50),
            'p90': np.percentile(norms, 90),
            'p95': np.percentile(norms, 95),
            'p99': np.percentile(norms, 99),
            'min': np.min(norms),
            'max': np.max(norms),
            'clipped_ratio': np.sum(norms >= self.c) / len(norms),
        }

    def compute_mse_components(self, stats):
        """Compute MSE components given current C and observed norms.

        For batch-level clipping:
        - Bias: squared error from clipping when ||g|| > C
        - Variance: simplified noise variance (C * sigma)^2 for the gradient norm
        """
        norms = np.array(self.epoch_grad_norms)
        if len(norms) == 0:
            return {'bias': 0, 'variance': 0, 'mse': 0}

        above_c = norms[norms > self.c]

        # Bias: mean squared error for clipped samples
        # If ||g|| > C, clipped value is C, error is ||g|| - C
        if len(above_c) > 0:
            squared_errors = (above_c - self.c) ** 2
            # Weight by fraction of clipped samples
            bias = np.mean(squared_errors) * len(above_c) / len(norms)
        else:
            bias = 0

        # Variance: simplified - noise std is C * sigma for gradient norm
        # This is an approximation for batch-level DP-SGD
        variance = (self.c * self.sigma) ** 2

        mse = bias + variance

        return {
            'bias': bias,
            'variance': variance,
            'mse': mse,
            'clipped_ratio': len(above_c) / len(norms) if len(norms) > 0 else 0,
        }

    def compute_optimal_c_mse(self):
        """Compute optimal C by grid search over percentiles.

        Finds the C that minimizes estimated MSE.
        """
        norms = np.array(self.epoch_grad_norms)
        if len(norms) == 0:
            return self.c

        # Try different C values as percentiles of observed norms
        percentiles = np.linspace(80, 99, 20)
        best_c = self.c
        best_mse = float('inf')

        for p in percentiles:
            trial_c = np.percentile(norms, p)
            if trial_c <= 0:
                continue

            # Compute MSE for this C
            above_c = norms[norms > trial_c]
            if len(above_c) > 0:
                bias = np.mean((above_c - trial_c) ** 2) * len(above_c) / len(norms)
            else:
                bias = 0

            d = 32050
            variance = (trial_c ** 2 * self.sigma ** 2 * d) / self.batch_size
            mse = bias + variance

            if mse < best_mse:
                best_mse = mse
                best_c = trial_c

        return best_c

    def update_c(self, method='mse_optimize'):
        """Update clipping threshold C after an epoch."""
        old_c = self.c
        stats = self.compute_gradient_stats()
        mse_components = self.compute_mse_components(stats)

        if method == 'mse_optimize':
            optimal_c = self.compute_optimal_c_mse()
        else:
            optimal_c = self.c

        # Apply momentum smoothing to prevent oscillation
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


def compute_per_sample_gradients(model, data, target, criterion, device):
    """Compute per-sample gradients using separate backward passes.

    This gives TRUE unclipped gradients by computing gradients per sample.
    All tensors stay on the same device as the model.
    """
    batch_size = data.shape[0]
    per_sample_grads = {}

    for name, param in model.named_parameters():
        per_sample_grads[name] = []

    for i in range(batch_size):
        model.zero_grad()

        single_data = data[i:i+1].to(device)
        single_target = target[i:i+1].to(device)

        output = model(single_data)
        loss = criterion(output, single_target)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                per_sample_grads[name].append(param.grad.clone())  # Keep on device

        model.zero_grad()

    # Stack per-sample gradients
    stacked_grads = {}
    for name, grads in per_sample_grads.items():
        if grads:
            stacked_grads[name] = torch.stack(grads, dim=0)  # [batch_size, *param_shape]

    return stacked_grads


def compute_grad_norms(per_sample_grads):
    """Compute per-sample gradient norms from per-sample gradient dict."""
    all_norms = []
    for name, grads in per_sample_grads.items():
        if grads.dim() > 1:
            batch_size = grads.shape[0]
            flat_grad = grads.reshape(batch_size, -1)
            norms = torch.norm(flat_grad, p=2, dim=1)
            all_norms.append(norms)
        else:
            # Bias or 1D weights
            norms = torch.abs(grads.squeeze())
            all_norms.append(norms)

    if not all_norms:
        return torch.tensor([])

    stacked = torch.stack(all_norms, dim=1)  # [batch_size, num_params]
    overall_norms = torch.norm(stacked, p=2, dim=1)  # [batch_size]
    return overall_norms.cpu()  # Return on CPU for collection


def clip_gradients(grads, C):
    """Clip gradient to norm C (batch-level clipping).

    For the average gradient g with norm ||g||:
        clipped_g = g * min(1, C / ||g||)

    This is standard gradient clipping at the batch level.
    """
    clipped = {}
    for name, grad in grads.items():
        if grad.dim() > 0:
            norm = torch.norm(grad)
            if norm > C:
                scale = C / norm
                clipped[name] = grad * scale
            else:
                clipped[name] = grad
        else:
            clipped[name] = grad
    return clipped


def add_gaussian_noise(grads_sum, sigma, C):
    """Add Gaussian noise to SUM of clipped gradients for DP.

    Standard DP-SGD: add noise to the sum of clipped per-sample gradients.
    """
    noise = {}
    for name, g in grads_sum.items():
        # Noise std = C * sigma, applied to the sum (not per-sample)
        noise_std = C * sigma
        noise[name] = torch.randn_like(g) * noise_std
    return noise


def apply_gradients(model, noisy_grads_sum, lr, batch_size):
    """Apply noisy gradients to model parameters.

    Standard DP-SGD: theta = theta - lr * (sum_of_clipped_grads + noise) / batch_size
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in noisy_grads_sum:
                # Average over samples and apply
                grad = noisy_grads_sum[name] / batch_size
                param.data -= lr * grad


def train_manual_dp(args, model, device, train_loader, optimizer, epoch,
                   adaptive_clipper, criterion, plot_callback=None):
    """Training loop with manual DP-SGD and batch-level adaptive clipping.

    This version:
    1. Standard forward+backward (single backward pass for the batch)
    2. Records gradient norms for adaptive C adjustment
    3. Clips gradient (batch-level)
    4. Adds DP noise
    5. Applies gradient
    """
    model.train()
    losses = []

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # Get gradient norms BEFORE clipping for histogram
        with torch.no_grad():
            grad_norms = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    norm = torch.norm(param.grad).item()
                    grad_norms.append(norm)
            if grad_norms:
                total_norm = np.sqrt(sum(n**2 for n in grad_norms))
                adaptive_clipper.add_batch(np.array([total_norm]))

        losses.append(loss.item())

        # Clip gradient using current C (batch-level clipping)
        C = adaptive_clipper.get_c()
        grads = {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None}
        clipped_grads = clip_gradients(grads, C)

        # Add DP noise (if not disabled)
        if not args.disable_dp:
            noise = add_gaussian_noise(clipped_grads, args.sigma, C)
            for name in clipped_grads:
                clipped_grads[name] = clipped_grads[name] + noise[name]

        # Apply noisy gradients
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in clipped_grads:
                    param.data -= args.lr * clipped_grads[name]

    # After epoch: update adaptive C
    update_result = adaptive_clipper.update_c(method=args.adaptive_method)
    adaptive_clipper.reset_epoch()

    stats = update_result['stats']
    print(
        f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}"
    )
    print(
        f"  Adaptive Clip: C={update_result['new_c']:.4f} "
        f"(was {update_result['old_c']:.4f}, optimal={update_result['optimal_c']:.4f}) "
        f"MSE={update_result['mse']:.2f} (bias={update_result['bias']:.2f}, var={update_result['variance']:.2f})"
    )
    print(
        f"  Grad Norms: mean={stats.get('mean', 0):.4f}, "
        f"p90={stats.get('p90', 0):.4f}, p95={stats.get('p95', 0):.4f}, "
        f"clipped={stats.get('clipped_ratio', 0):.2%}"
    )

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


def main():
    parser = argparse.ArgumentParser(description="MNIST with Manual DP-SGD and Adaptive Clipping")
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("-n", "--epochs", type=int, default=10)
    parser.add_argument("-r", "--n-runs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=1.0, help="DP noise multiplier")
    parser.add_argument("-c", "--initial-c", type=float, default=1.0, help="Initial clipping threshold")
    parser.add_argument("--momentum", type=float, default=0.5, help="Momentum for C update")
    parser.add_argument("--min-c", type=float, default=0.1)
    parser.add_argument("--max-c", type=float, default=10.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--disable-dp", action="store_true", help="Disable DP noise")
    parser.add_argument("--data-root", type=str, default="../mnist")
    parser.add_argument("--plot", action="store_true", help="Enable plotting")
    parser.add_argument("--adaptive-method", type=str, default="mse_optimize",
                        choices=["mse_optimize", "percentile"])

    args = parser.parse_args()
    device = torch.device(args.device)

    # Check for matplotlib
    HAS_MATPLOTLIB = True
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        HAS_MATPLOTLIB = False
        print("matplotlib not available, plotting disabled")

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data_root, train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                        ])),
        batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data_root, train=False,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                        ])),
        batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    criterion = nn.CrossEntropyLoss()

    # Plot directory
    repro_str = f"manual_dp_{args.lr}_{args.sigma}_{args.initial_c}_{args.batch_size}_{args.epochs}"
    plot_dir = f"plots_{repro_str}"
    if args.plot and HAS_MATPLOTLIB:
        os.makedirs(plot_dir, exist_ok=True)
        print(f"Plots will be saved to {plot_dir}/")

    run_results = []

    for run_idx in range(args.n_runs):
        print(f"\n=== Run {run_idx + 1}/{args.n_runs} ===")
        print(f"Adaptive method: {args.adaptive_method}, initial C: {args.initial_c}, sigma: {args.sigma}")

        # Initialize adaptive clipper
        adaptive_clipper = AdaptiveClipper(
            initial_c=args.initial_c,
            sigma=args.sigma,
            batch_size=args.batch_size,
            momentum=args.momentum,
            min_c=args.min_c,
            max_c=args.max_c
        )

        model = SampleConvNet().to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)

        for epoch in range(1, args.epochs + 1):
            train_manual_dp(args, model, device, train_loader, optimizer, epoch,
                          adaptive_clipper, criterion)

        acc, acc_pct = test(model, device, test_loader)
        run_results.append(acc)

        print(f"\nFinal adaptive C: {adaptive_clipper.c:.4f}")
        print(f"C history: {[f'{c:.3f}' for c in adaptive_clipper.c_history]}")

        # Plot results
        if args.plot and HAS_MATPLOTLIB:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            epochs = range(1, len(adaptive_clipper.c_history))

            axes[0, 0].plot(epochs, adaptive_clipper.c_history[1:], 'b-o')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('C')
            axes[0, 0].set_title('Clipping Threshold C over Epochs')
            axes[0, 0].grid(True)

            axes[0, 1].plot(epochs, adaptive_clipper.mse_history, 'r-o')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MSE')
            axes[0, 1].set_title('Estimated MSE over Epochs')
            axes[0, 1].grid(True)

            axes[1, 0].plot(epochs, adaptive_clipper.bias_history, 'g-o', label='Bias²')
            axes[1, 0].plot(epochs, adaptive_clipper.var_history, 'm-s', label='Variance')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Error')
            axes[1, 0].set_title('MSE Components')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

            axes[1, 1].plot(epochs, adaptive_clipper.c_history[1:], 'b-o', label='C')
            axes[1, 1].twinx().plot(epochs, adaptive_clipper.mse_history, 'r-s', label='MSE')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].set_title('C and MSE')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

            plt.suptitle('Adaptive Gradient Clipping Analysis (Manual DP-SGD)', fontsize=14)
            plt.tight_layout()
            save_path = os.path.join(plot_dir, f"run_{run_idx+1}_analysis.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved analysis plot to {save_path}")

    if len(run_results) > 1:
        print(f"\nAccuracy averaged over {len(run_results)} runs: {np.mean(run_results)*100:.2f}% ± {np.std(run_results)*100:.2f}%")

    # Save results
    torch.save(run_results, f"run_results_{repro_str}.pt")

    if args.save_model:
        torch.save(model.state_dict(), f"mnist_cnn_{repro_str}.pt")


if __name__ == "__main__":
    main()
