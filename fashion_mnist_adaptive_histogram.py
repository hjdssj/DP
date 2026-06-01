#!/usr/bin/env python3
"""
Fashion-MNIST training with Opacus DP-SGD and histogram-based adaptive gradient clipping.

Based on minst_adaptive_histogram.py with:
1. Gradient norm histogram tracking via Opacus's grad_sample
2. Adaptive C adjustment based on clipped sample ratio
3. Visualization of gradient distributions

Fashion-MNIST: 10 fashion categories (T-shirt, Trouser, Pullover, Dress, etc.)
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


# Precomputed characteristics of the Fashion-MNIST dataset
FASHION_MNIST_MEAN = 0.2860
FASHION_MNIST_STD = 0.3530


class GradientHistogram:
    """Tracks gradient norm distribution from Opacus's grad_sample.

    Note: grad_sample contains CLIPPED norms. If norm > C, we see C.
    We use this to estimate:
    - How many samples were clipped
    - Distribution of unclipped norms
    """

    def __init__(self, bin_min=0.0, bin_max=2.0, num_bins=50):
        self.bin_min = bin_min
        self.bin_max = bin_max
        self.bin_edges = np.linspace(bin_min, bin_max, num_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        self.num_bins = num_bins
        self.reset()

    def set_bin_max(self, bin_max):
        """Update bin edges when C changes."""
        self.bin_max = bin_max
        self.bin_edges = np.linspace(self.bin_min, bin_max, self.num_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

    def reset(self):
        self.counts = np.zeros(self.num_bins)
        self.total_samples = 0
        self.clipped_at_max = 0

    def add_batch(self, grad_norms):
        """Add gradient norms from a batch.

        grad_norms: tensor of per-sample gradient norms (from grad_sample)
        These are already clipped by Opacus to max_grad_norm.
        """
        norms = grad_norms.cpu().detach().numpy()
        self.total_samples += len(norms)

        # Count samples at the clip boundary (these were clipped)
        max_bin_idx = self.num_bins - 1
        clipped_count = np.sum(norms >= self.bin_edges[max_bin_idx])
        self.clipped_at_max += clipped_count

        # Bin the rest
        for norm in norms:
            if norm < self.bin_edges[max_bin_idx]:
                bin_idx = np.searchsorted(self.bin_edges[1:], norm)
                bin_idx = min(max(bin_idx, 0), self.num_bins - 1)
                self.counts[bin_idx] += 1

    def get_clipped_ratio(self):
        """Fraction of samples that hit the clip boundary."""
        if self.total_samples == 0:
            return 0
        return self.clipped_at_max / self.total_samples

    def get_stats(self):
        """Compute distribution statistics."""
        if self.total_samples == 0:
            return {}

        return {
            'total_samples': self.total_samples,
            'clipped_ratio': self.get_clipped_ratio(),
            'mean': np.sum(self.bin_centers * self.counts) / self.total_samples,
            'std': np.sqrt(
                np.sum(((self.bin_centers - np.sum(self.bin_centers * self.counts) / self.total_samples) ** 2) * self.counts) / self.total_samples
            ),
        }


class AdaptiveClipController:
    """Adapts clipping threshold C based on observed clipping ratio.

    Strategy:
    - Target: keep clipped_ratio within [min_ratio, max_ratio]
    - If clipped_ratio > max_ratio: increase C (less clipping bias)
    - If clipped_ratio < min_ratio: decrease C (less noise variance)
    - Use PID-like controller for smooth adjustment
    """

    def __init__(self, initial_c=1.0, target_ratio=0.2, tolerance=0.1,
                 min_c=0.1, max_c=10.0, adjustment_speed=0.1):
        """
        Args:
            initial_c: Starting clipping threshold
            target_ratio: Target fraction of samples to clip (20% default)
            tolerance: Acceptable range around target_ratio
            min_c, max_c: C bounds
            adjustment_speed: How fast to adjust C (0-1)
        """
        self.c = initial_c
        self.target_ratio = target_ratio
        self.tolerance = tolerance
        self.min_c = min_c
        self.max_c = max_c
        self.adjustment_speed = adjustment_speed

        self.c_history = [initial_c]
        self.clipped_ratio_history = []

    def update(self, clipped_ratio):
        """Update C based on observed clipped_ratio."""
        self.clipped_ratio_history.append(clipped_ratio)

        # Compute adjustment
        if clipped_ratio > self.target_ratio + self.tolerance:
            # 裁剪比例过高 → 增加C（减少裁剪）
            excess = clipped_ratio - (self.target_ratio + self.tolerance)
            adjustment = 1 + self.adjustment_speed * (1 + excess * 2)
            new_c = self.c * adjustment
        elif clipped_ratio < self.target_ratio - self.tolerance:
            # 裁剪比例过低 → 减小C（增加裁剪）
            deficit = (self.target_ratio - self.tolerance) - clipped_ratio
            adjustment = 1 - self.adjustment_speed * (1 + deficit * 2)
            new_c = self.c * adjustment
        else:
            # 在容忍区间内 → 保持稳定
            new_c = self.c

        # Apply bounds
        new_c = max(self.min_c, min(self.max_c, new_c))

        # Smooth change (momentum-like)
        new_c = 0.5 * self.c + 0.5 * new_c

        self.c = new_c
        self.c_history.append(new_c)

        return new_c

    def get_c(self):
        return self.c


class FashionCNN(nn.Module):
    """
    CNN architecture matching the TensorFlow version:
    - Conv1: 5x5, 1->32
    - Pool1: 2x2 max pool
    - Conv2: 5x5, 32->64
    - Pool2: 2x2 max pool
    - FC1: 7*7*64=3136 -> 1024
    - FC2: 1024 -> 10
    """
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

    def name(self):
        return "FashionCNN"


def train(args, model, device, train_loader, optimizer, privacy_engine, epoch,
          histogram, adaptive_controller):
    """Training loop with histogram tracking and adaptive clipping."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []

    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # Collect gradient norms from grad_sample (clipped by Opacus)
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

        # Clear grad_sample (Opacus requirement)
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

    # Update adaptive C based on observed clipping ratio
    clipped_ratio = histogram.get_clipped_ratio()
    old_c = adaptive_controller.get_c()
    new_c = adaptive_controller.update(clipped_ratio)

    # Sync histogram bin range with new C
    histogram.set_bin_max(new_c * 2)

    stats = histogram.get_stats()

    print(
        f"  Adaptive C: {new_c:.4f} (was {old_c:.4f})"
        f" | Clipped: {clipped_ratio:.1%} (target: {args.target_ratio:.0%})"
        f" | Grad mean: {stats.get('mean', 0):.4f}"
    )

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
        description="Fashion-MNIST with Opacus DP-SGD and Adaptive Clipping",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-b", "--batch-size", type=int, default=150, help="Batch size")
    parser.add_argument("--test-batch-size", type=int, default=1024)
    parser.add_argument("-n", "--epochs", type=int, default=20)
    parser.add_argument("-r", "--n-runs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--sigma", type=float, default=1.0, help="Noise multiplier")
    parser.add_argument("-c", "--initial-c", type=float, default=1.0,
                        help="Initial clipping threshold")
    parser.add_argument("--target-ratio", type=float, default=0.3,
                        help="Target fraction of samples to clip (0.0-1.0)")
    parser.add_argument("--tolerance", type=float, default=0.1,
                        help="Tolerance around target ratio")
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--disable-dp", action="store_true")
    parser.add_argument("--secure-rng", action="store_true")
    parser.add_argument("--data-root", type=str, default="../fashion_mnist")
    parser.add_argument("--plot", action="store_true", help="Enable histogram plotting")
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
            args.data_root, train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((FASHION_MNIST_MEAN,), (FASHION_MNIST_STD,)),
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
        tolerance=args.tolerance,
        min_c=0.1,
        max_c=10.0,
        adjustment_speed=0.15
    )

    repro_str = (
        f"fashion_mnist_adaptive_hist_{args.lr}_{args.sigma}_"
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
            tolerance=args.tolerance,
            min_c=0.1,
            max_c=10.0,
            adjustment_speed=0.15
        )

        model = FashionCNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
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

        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, privacy_engine,
                  epoch, histogram, adaptive_controller)

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
    }, f"adaptive_histogram_results_{repro_str}.pt")

    if args.save_model:
        torch.save(model.state_dict(), f"fashion_mnist_cnn_{repro_str}.pt")


if __name__ == "__main__":
    main()