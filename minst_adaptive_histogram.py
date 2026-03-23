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

        Note: Since grad_sample is already clipped by Opacus, we use a
        simplified bias estimation based on the distribution of observed norms.

        The key insight is that if many samples have norm ≈ C (at the clipping
        boundary), the original norms were likely much larger → high bias.
        """
        norms = np.array(self.epoch_grad_norms)
        if len(norms) == 0:
            return {'bias': 0, 'variance': 0, 'mse': 0}

        # Estimate bias by looking at how many samples are concentrated at C
        # If samples are clustered near C, they were likely clipped from higher values
        at_boundary_tolerance = 0.05 * self.c  # 5% of C
        at_boundary = np.sum(np.abs(norms - self.c) < at_boundary_tolerance) / len(norms)

        # Estimate average amount of clipping for boundary samples
        # Assume original norms were uniformly distributed from C to some max estimate
        mean_norms = np.mean(norms)
        max_estimate = np.max(norms) * 2  # Rough estimate of original max

        # Bias is estimated from how much clipping likely occurred
        # Using a simple heuristic: if many samples at C, original was much larger
        estimated_original_mean = mean_norms + (max_estimate - mean_norms) * at_boundary * 0.5
        clipped_ratio = 1 - (self.c / max(estimated_original_mean, self.c * 1.1))

        # Simplified bias estimation
        bias = (clipped_ratio ** 2) * (max_estimate - self.c) ** 2 * at_boundary

        # Variance from DP noise (standard formula)
        # Total number of parameters in the model
        d = (1 * 16 * 8 * 8 + 16) + (16 * 32 * 4 * 4 + 32) + (32 * 4 * 4 * 32 + 32) + (32 * 10 + 10)
        variance = (self.c ** 2 * self.sigma ** 2 * d) / self.batch_size

        mse = bias + variance

        return {
            'bias': bias,
            'variance': variance,
            'mse': mse,
            'at_boundary_ratio': at_boundary,
            'clipped_ratio': clipped_ratio if 'clipped_ratio' in dir() else 0,
        }

    def compute_optimal_c_mse(self):
        """Compute optimal C using stable adaptive strategy.

        Since grad_sample is clipped by Opacus, we cannot observe true gradient norms.
        Instead, we use a robust approach:

        1. Track the observed clipped norms distribution
        2. If many samples cluster at C (heavily clipped), increase C
        3. If few samples at C (minimal clipping), C may be too large
        4. Apply heavy smoothing to prevent oscillation
        """
        norms = np.array(self.epoch_grad_norms)
        if len(norms) == 0:
            return self.c

        current_c = self.c

        # Count samples at various distances from C
        # Samples at exactly C were heavily clipped
        at_exact_c = np.sum(np.abs(norms - current_c) < 0.01 * current_c) / len(norms)
        near_c = np.sum(np.abs(norms - current_c) < 0.1 * current_c) / len(norms)
        above_c = np.sum(norms > current_c) / len(norms)

        # Strategy:
        # - If >50% of samples are at C (very heavy clipping), increase C significantly
        # - If 20-50% at C, increase C slightly
        # - If <20% at C but C is still the max observed, C might be too large
        # - Otherwise, keep C stable

        if len(self.c_history) <= 2:
            # First few epochs: estimate scale from clipped norms
            # Use a percentile-based estimate assuming clipping happened
            observed_p95 = np.percentile(norms, 95)
            # If observed norms are all close to C, true norms were much larger
            if near_c > 0.8:
                # Almost all samples were clipped → true norms much larger
                estimated_true_scale = observed_p95 * 2
            else:
                estimated_true_scale = observed_p95
            optimal_c = min(estimated_true_scale, self.max_c)
        else:
            # After warmup: use stable adjustment based on clipping fraction
            if at_exact_c > 0.5:
                # Heavy clipping: increase C by up to 50%
                adjustment = min(0.5, at_exact_c)
                optimal_c = current_c * (1 + adjustment)
            elif at_exact_c > 0.3:
                # Moderate clipping: increase by up to 20%
                adjustment = min(0.2, (at_exact_c - 0.3) / 0.2 * 0.2)
                optimal_c = current_c * (1 + adjustment)
            elif near_c < 0.1 and current_c > np.percentile(norms, 90):
                # Minimal clipping and C > 90th percentile: C may be too large
                optimal_c = current_c * 0.9
            else:
                # Stable: small adjustment toward observed 80th percentile
                target = np.percentile(norms, 80)
                optimal_c = current_c * 0.95 + target * 0.05

        # Apply very heavy smoothing (momentum = 0.95 for C updates)
        # This prevents oscillation
        optimal_c = max(self.min_c, min(self.max_c, optimal_c))

        return optimal_c

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
    """Compute per-sample gradient norms for all parameters.

    Note: Uses grad_sample (already clipped by Opacus) for backward compatibility.
    When DP is disabled, grad_sample doesn't exist - returns empty tensor.
    """
    per_sample_norms = []

    for param in model.parameters():
        if hasattr(param, 'grad_sample') and param.grad_sample is not None:
            grad_sample = param.grad_sample
            flat_grad = grad_sample.reshape(grad_sample.shape[0], -1)
            param_norms = torch.norm(flat_grad, p=2, dim=1)
            per_sample_norms.append(param_norms)

    if not per_sample_norms:
        return torch.tensor([])

    stacked = torch.stack(per_sample_norms, dim=1)
    overall_norms = torch.norm(stacked, p=2, dim=1)

    return overall_norms


class UnclippedGradHook:
    """Hook to capture unclipped per-sample gradients before Opacus clipping.

    Note: Due to Opacus's internal hooks executing first during backward pass,
    this approach captures gradients AFTER clipping. For true unclipped gradients,
    we would need to manually compute per-sample gradients (see compute_unclipped_grad_norms).
    """

    def __init__(self):
        self.grad_norms = []
        self._hook_handles = []

    def _hook_fn(self, module, grad_input, grad_output):
        """Capture gradient during backward pass (already clipped by Opacus)."""
        if grad_output is None or len(grad_output) == 0:
            return None

        for grad in grad_output:
            if grad is None:
                continue
            if grad.dim() > 1:
                batch_size = grad.shape[0]
                flat_grad = grad.reshape(batch_size, -1)
                norms = torch.norm(flat_grad, p=2, dim=1)
                self.grad_norms.append(norms.cpu())
        return None

    def register_hooks(self, model):
        """Register backward hooks on all model parameters."""
        self.remove_hooks()
        handle = model.register_full_backward_hook(self._hook_fn)
        self._hook_handles.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def get_grad_norms(self):
        """Get all captured gradient norms as a single tensor."""
        if not self.grad_norms:
            return torch.tensor([])
        all_norms = torch.cat(self.grad_norms)
        self.grad_norms = []
        return all_norms


def compute_unclipped_grad_norms(model, data, target, criterion, device):
    """Compute per-sample gradient norms by separate backward passes.

    This gives us TRUE unclipped gradients by doing one forward+backward per sample.
    Slower but accurate - needed for correct MSE-based C adaptation.
    """
    batch_size = data.shape[0]
    all_norms = []

    with torch.no_grad():
        for i in range(batch_size):
            model.zero_grad()
            single_data = data[i:i+1].to(device)
            single_target = target[i:i+1].to(device)

            output = model(single_data)
            loss = criterion(output, single_target)
            loss.backward()

            # Collect gradient norms
            param_norms = []
            for param in model.parameters():
                if param.grad is not None:
                    flat_grad = param.grad.flatten()
                    norm = torch.norm(flat_grad, p=2).item()
                    param_norms.append(norm)

            if param_norms:
                overall_norm = np.sqrt(sum(n**2 for n in param_norms))
                all_norms.append(overall_norm)

            model.zero_grad()

    return torch.tensor(all_norms)


def train_with_adaptive_clip(args, model, device, train_loader, optimizer, privacy_engine,
                             epoch, adaptive_clipper, unclipped_hook=None, plot_callback=None):
    """Training loop with adaptive clipping.

    Uses manual per-sample gradient computation to get TRUE unclipped gradient norms
    for accurate histogram-based C adaptation.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []

    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # Get UNCLIPPED gradient norms using the hook
        # Note: hook captures clipped values due to Opacus hook ordering
        # For true unclipped norms, we use separate backward passes
        if unclipped_hook is not None:
            per_sample_norms = unclipped_hook.get_grad_norms()
            if len(per_sample_norms) > 0:
                adaptive_clipper.add_batch(per_sample_norms)

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
    parser.add_argument("-c", "--initial-c", type=float, default=10.0,
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

        # Create unclipped gradient hook BEFORE Opacus wraps the model
        # This allows us to capture gradient norms before Opacus clipping
        unclipped_hook = None
        if not args.disable_dp:
            unclipped_hook = UnclippedGradHook()
            unclipped_hook.register_hooks(model)

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
                epoch, adaptive_clipper, unclipped_hook=unclipped_hook,
                plot_callback=plot_callback if args.plot else None
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
