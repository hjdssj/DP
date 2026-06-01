#!/usr/bin/env python3
"""
MNIST training with Opacus DP-SGD and DP histogram-based adaptive gradient clipping.

Extends minst_adaptive_histogram.py with a differentially private histogram:
- Each epoch, gradient norm counts are perturbed with Laplace noise before
  computing clipped_ratio, so the histogram query itself satisfies DP.
- The per-epoch histogram privacy cost (epsilon_hist) is tracked separately
  and added to the Opacus SGD cost to report the true total epsilon.

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
from torchvision import datasets, transforms
from tqdm import tqdm


# Precomputed characteristics of the MNIST dataset
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class GradientHistogram:
    """Tracks TRUE (pre-clip) gradient norm distribution with DP noise.

    grad_sample is read before optimizer.step(), so norms are unclipped.
    get_noisy_clipped_ratio() adds Laplace noise to satisfy epsilon_hist-DP
    before computing the clipped ratio used for adaptive C adjustment.

    Sensitivity of the clipped-count query = 1 (each sample contributes at
    most 1 to the count), so Laplace scale = 1 / epsilon_hist.
    """

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
        """Add TRUE per-sample gradient norms (before Opacus clipping)."""
        norms = grad_norms.cpu().detach().numpy()
        self.total_samples += len(norms)
        self.clipped_count += int(np.sum(norms >= self.current_c))
        indices = np.clip(
            np.searchsorted(self.bin_edges[1:], norms), 0, self.num_bins - 1
        )
        for idx in indices:
            self.counts[idx] += 1

    def get_clipped_ratio(self):
        """Return true (non-private) clipped ratio. For logging only."""
        if self.total_samples == 0:
            return 0.0
        return self.clipped_count / self.total_samples

    def get_noisy_clipped_ratio(self, epsilon_hist):
        """Return DP-noisy clipped ratio satisfying epsilon_hist-DP.

        Adds Laplace(0, 1/epsilon_hist) noise to the clipped count before
        dividing by total_samples. Clamps result to [0, 1].
        """
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
            'std': np.sqrt(
                np.sum(((self.bin_centers - mean) ** 2) * self.counts) / self.total_samples
            ),
        }


def compute_optimal_c_mse(bin_centers, noisy_counts, sigma, batch_size, d,
                           c_min=0.05, c_max=5.0, n_grid=200, var_weight=10.0):
    """Find C* = argmin MSE(C) using DP histogram.

    MSE(C) = Bias²(C) + Variance(C)
           = (1/N²) Σ_{r_k > C} n_k (r_k - C)²  +  σ²C²d / N²

    Args:
        bin_centers: np.ndarray [K], histogram bin centers
        noisy_counts: np.ndarray [K], DP-noisy bin counts (already clipped >=0)
        sigma: DP noise multiplier
        batch_size: logical batch size n
        d: total model parameter dimension
        c_min, c_max: search range for C
        n_grid: number of candidate C values

    Returns:
        c_star: float, MSE-minimizing C
        info: dict with mse/bias2/variance curves for plotting
    """
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
    """Adapts clipping threshold C based on observed clipping ratio or MSE minimization.

    method='ratio': adjust C to keep clipped_ratio near target_ratio
    method='mse':   set C = argmin MSE(C) from DP histogram each epoch
    """

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

    def update_mse(self, bin_centers, noisy_counts, sigma, batch_size, d, var_weight=10.0):
        c_star, info = compute_optimal_c_mse(
            bin_centers, noisy_counts, sigma, batch_size, d,
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

    def name(self):
        return "SampleConvNet"


def train(args, model, device, train_loader, optimizer, privacy_engine, epoch,
          histogram, adaptive_controller, epsilon_hist_per_epoch, d):
    """Training loop with DP histogram tracking and adaptive clipping."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []

    for (data, target) in tqdm(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        per_sample_norms = []
        for param in model.parameters():
            if hasattr(param, 'grad_sample') and param.grad_sample is not None:
                gs = param.grad_sample
                param_norms = gs.reshape(gs.shape[0], -1).norm(2, dim=1)
                per_sample_norms.append(param_norms)

        if per_sample_norms:
            overall_norms = torch.stack(per_sample_norms, dim=1).norm(2, dim=1)
            histogram.set_current_c(adaptive_controller.get_c())
            histogram.add_batch(overall_norms)

        optimizer.step()
        losses.append(loss.item())

    if not args.disable_dp:
        epsilon_sgd = privacy_engine.accountant.get_epsilon(delta=args.delta)
        epsilon_hist_total = epsilon_hist_per_epoch * epoch
        epsilon_total = epsilon_sgd + epsilon_hist_total
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"(ε_sgd={epsilon_sgd:.2f}, ε_hist={epsilon_hist_total:.2f}, "
            f"ε_total={epsilon_total:.2f}, δ={args.delta})"
        )
    else:
        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")

    old_c = adaptive_controller.get_c()
    true_clipped_ratio = histogram.get_clipped_ratio()

    if adaptive_controller.method == 'mse':
        # Build DP-noisy counts for MSE computation
        noisy_counts = histogram.counts + np.random.laplace(
            0, 1.0 / epsilon_hist_per_epoch, histogram.num_bins)
        noisy_counts = np.maximum(noisy_counts, 0.0)
        new_c, mse_info = adaptive_controller.update_mse(
            histogram.bin_centers, noisy_counts, args.sigma, args.batch_size, d,
            var_weight=args.mse_var_weight)
        c_star = mse_info.get('c_star', new_c) if mse_info else new_c
        print(
            f"  [MSE] C: {new_c:.4f} (was {old_c:.4f}, c*={c_star:.4f})"
            f" | True clipped: {true_clipped_ratio:.1%}"
            f" | MSE={adaptive_controller.mse_history[-1]:.2e}"
            f" | Bias²={adaptive_controller.bias2_history[-1]:.2e}"
            f" | Var={adaptive_controller.variance_history[-1]:.2e}"
        )
    else:
        noisy_clipped_ratio = histogram.get_noisy_clipped_ratio(epsilon_hist_per_epoch)
        new_c = adaptive_controller.update_ratio(noisy_clipped_ratio)
        print(
            f"  [ratio] C: {new_c:.4f} (was {old_c:.4f})"
            f" | Noisy clipped: {noisy_clipped_ratio:.1%}"
            f" | True clipped: {true_clipped_ratio:.1%}"
            f" | Target: {args.target_ratio:.0%}"
        )

    if not args.disable_dp and hasattr(optimizer, 'max_grad_norm'):
        optimizer.max_grad_norm = new_c

    histogram.set_bin_max(new_c * 2)
    histogram.reset()


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
    return correct / len(test_loader.dataset)


def main():
    parser = argparse.ArgumentParser(
        description="MNIST with Opacus DP-SGD and Adaptive Clipping",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--test-batch-size", type=int, default=1024)
    parser.add_argument("-n", "--epochs", type=int, default=10)
    parser.add_argument("-r", "--n-runs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=1.0, help="Noise multiplier")
    parser.add_argument("-c", "--initial-c", type=float, default=1.0,
                        help="Initial clipping threshold")
    parser.add_argument("--target-ratio", type=float, default=0.2,
                        help="Target fraction of samples to clip (0.0-1.0)")
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--disable-dp", action="store_true")
    parser.add_argument("--secure-rng", action="store_true")
    parser.add_argument("--data-root", type=str, default="../mnist")
    parser.add_argument("--plot", action="store_true", help="Enable histogram plotting")
    parser.add_argument("--epsilon-hist", type=float, default=1.0,
                        help="Per-epoch privacy budget for DP histogram query")
    parser.add_argument("--method", type=str, default="ratio", choices=["ratio", "mse"],
                        help="Adaptive C update method: ratio or mse")
    parser.add_argument("--mse-var-weight", type=float, default=10.0,
                        help="Variance term weight in MSE objective (method=mse only)")
    args = parser.parse_args()
    device = torch.device(args.device)

    # Try to import matplotlib
    HAS_MATPLOTLIB = True
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        HAS_MATPLOTLIB = False
        print("matplotlib not available, plotting disabled")

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data_root, train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
            ])
        ),
        batch_size=args.batch_size, num_workers=0, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data_root, train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
            ])
        ),
        batch_size=args.test_batch_size, shuffle=True, num_workers=0, pin_memory=True,
    )

    # Initialize histogram and adaptive controller
    histogram = GradientHistogram(
        bin_min=0.0,
        bin_max=args.initial_c * 2,
        num_bins=50
    )
    adaptive_controller = AdaptiveClipController(
        initial_c=args.initial_c,
        target_ratio=args.target_ratio,
        tolerance=0.05,
        min_c=0.1,
        max_c=10.0,
        adjustment_speed=0.15
    )

    repro_str = (
        f"mnist_adaptive_hist_{args.lr}_{args.sigma}_"
        f"{args.initial_c}_{args.batch_size}_{args.epochs}"
    )
    plot_dir = f"histogram_plots_{repro_str}"
    if args.plot and HAS_MATPLOTLIB:
        os.makedirs(plot_dir, exist_ok=True)

    run_results = []

    for run_idx in range(args.n_runs):
        print(f"\n=== Run {run_idx + 1}/{args.n_runs} ===")
        print(f"Adaptive Clipping: target_ratio={args.target_ratio}, initial C={args.initial_c}")

        # Reset
        histogram.reset()
        adaptive_controller = AdaptiveClipController(
            initial_c=args.initial_c,
            target_ratio=args.target_ratio,
            tolerance=0.05,
            min_c=0.1,
            max_c=10.0,
            adjustment_speed=0.15,
            method=args.method,
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
                max_grad_norm=adaptive_controller.get_c(),
            )

        d = sum(p.numel() for p in model.parameters())

        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, privacy_engine,
                  epoch, histogram, adaptive_controller, args.epsilon_hist, d)

            # Plot histogram every 2 epochs
            if args.plot and HAS_MATPLOTLIB and epoch % 2 == 0:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                # Left plot: full histogram with log scale
                bin_width = histogram.bin_edges[1] - histogram.bin_edges[0]
                axes[0].bar(histogram.bin_centers, histogram.counts,
                            width=bin_width * 0.9, alpha=0.7, edgecolor='black')
                axes[0].axvline(x=adaptive_controller.get_c(), color='red',
                                 linestyle='--', linewidth=2, label=f'C={adaptive_controller.get_c():.3f}')
                axes[0].set_xlabel('Gradient Norm')
                axes[0].set_ylabel('Count (log scale)')
                axes[0].set_yscale('log')
                axes[0].set_title(f'Epoch {epoch}: Gradient Norm Distribution')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

                # Right plot: zoomed view of low-norm region (0 to 0.3)
                zoom_max = 0.3
                zoom_indices = histogram.bin_centers < zoom_max
                zoom_centers = histogram.bin_centers[zoom_indices]
                zoom_counts = histogram.counts[zoom_indices]

                if len(zoom_counts) > 0 and zoom_counts.sum() > 0:
                    bar_width = (zoom_max / len(zoom_centers)) * 0.9 if len(zoom_centers) > 0 else 0.01
                    axes[1].bar(zoom_centers, zoom_counts,
                                width=bar_width, alpha=0.7, edgecolor='black', color='green')
                    axes[1].axvline(x=adaptive_controller.get_c(), color='red',
                                     linestyle='--', linewidth=2, label=f'C={adaptive_controller.get_c():.3f}')
                    axes[1].set_xlabel('Gradient Norm')
                    axes[1].set_ylabel('Count')
                    axes[1].set_title(f'Zoomed: Norm < {zoom_max}')
                    axes[1].legend()
                    axes[1].grid(True, alpha=0.3)

                plt.tight_layout()
                save_path = os.path.join(plot_dir, f"epoch_{epoch:03d}.png")
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved histogram to {save_path}")

        accuracy = test(model, device, test_loader)
        run_results.append(accuracy)

        # Final C summary
        print(f"\nFinal adaptive C: {adaptive_controller.c:.4f}")
        print(f"C history: {[f'{c:.3f}' for c in adaptive_controller.c_history]}")

        # Plot C and clipped ratio over training
        if args.plot and HAS_MATPLOTLIB:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            epochs = range(1, len(adaptive_controller.c_history))
            ax1.plot(epochs, adaptive_controller.c_history[1:], 'b-o')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Clipping Threshold C')
            ax1.set_title('Adaptive C Over Training')
            ax1.grid(True, alpha=0.3)

            ax2.plot(epochs, adaptive_controller.clipped_ratio_history, 'r-s')
            ax2.axhline(y=args.target_ratio, color='g', linestyle='--',
                        label=f'Target ({args.target_ratio:.0%})')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Clipped Ratio')
            ax2.set_title('Clipped Sample Ratio Over Training')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = os.path.join(plot_dir, f"run_{run_idx+1}_summary.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved summary to {save_path}")

    if len(run_results) > 1:
        print(
            f"\nAccuracy averaged over {len(run_results)} runs: "
            f"{np.mean(run_results) * 100:.2f}% ± {np.std(run_results) * 100:.2f}%"
        )

    torch.save({
        'run_results': run_results,
        'c_history': adaptive_controller.c_history,
        'clipped_ratio_history': adaptive_controller.clipped_ratio_history,
        'mse_history': adaptive_controller.mse_history,
        'bias2_history': adaptive_controller.bias2_history,
        'variance_history': adaptive_controller.variance_history,
        'method': args.method,
    }, f"adaptive_histogram_results_{repro_str}.pt")

    if args.save_model:
        torch.save(model.state_dict(), f"mnist_cnn_{repro_str}.pt")


if __name__ == "__main__":
    main()
