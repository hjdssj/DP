#!/usr/bin/env python3
"""
CIFAR-10 training with Opacus DP-SGD and histogram-based adaptive gradient clipping.
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


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class GradientHistogram:
    """Tracks gradient norm distribution from Opacus's grad_sample."""

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
            'std': np.sqrt(
                np.sum(((self.bin_centers - np.sum(self.bin_centers * self.counts) / self.total_samples) ** 2) * self.counts) / self.total_samples
            ),
        }


class AdaptiveClipController:
    """Adapts clipping threshold C based on observed clipping ratio."""

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

        # 误差越大，调整幅度越大
        if clipped_ratio > self.target_ratio + self.tolerance:
            excess = clipped_ratio - (self.target_ratio + self.tolerance)
            # 更激进的调整：直接加上误差的比例
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


class Model(nn.Module):
    """CIFAR-10 CNN architecture."""
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=36, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(36 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 36 * 6 * 6)
        x = self.fc2(F.relu(self.fc1(x)))
        return x


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

    if not args.disable_dp:
        epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"(ε = {epsilon:.2f}, δ = {args.delta})"
        )
    else:
        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")

    clipped_ratio = histogram.get_clipped_ratio()
    old_c = adaptive_controller.get_c()
    new_c = adaptive_controller.update(clipped_ratio)
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
    parser = argparse.ArgumentParser(description="CIFAR-10 with Opacus DP-SGD and Adaptive Clipping")
    parser.add_argument("-b", "--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("-n", "--epochs", type=int, default=20)
    parser.add_argument("-r", "--n-runs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("-c", "--initial-c", type=float, default=1.0)
    parser.add_argument("--target-ratio", type=float, default=0.3)
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--disable-dp", action="store_true")
    parser.add_argument("--secure-rng", action="store_true")
    parser.add_argument("--data-root", type=str, default="../cifar10")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)

    HAS_MATPLOTLIB = True
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        HAS_MATPLOTLIB = False
        print("matplotlib not available, plotting disabled")

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            args.data_root, train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ])
        ),
        batch_size=args.batch_size, num_workers=0, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            args.data_root, train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ])
        ),
        batch_size=args.test_batch_size, shuffle=True, num_workers=0, pin_memory=True,
    )

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
        max_c=50.0,
        adjustment_speed=0.15
    )

    repro_str = f"cifar10_adaptive_{args.lr}_{args.sigma}_{args.initial_c}_{args.batch_size}_{args.epochs}"
    plot_dir = f"histogram_plots_{repro_str}"
    if args.plot and HAS_MATPLOTLIB:
        os.makedirs(plot_dir, exist_ok=True)

    run_results = []

    for run_idx in range(args.n_runs):
        print(f"\n=== Run {run_idx + 1}/{args.n_runs} ===")
        print(f"Adaptive Clipping: target_ratio={args.target_ratio}, initial C={args.initial_c}")

        histogram.reset()
        adaptive_controller = AdaptiveClipController(
            initial_c=args.initial_c,
            target_ratio=args.target_ratio,
            tolerance=args.tolerance,
            min_c=0.1,
            max_c=50.0,
            adjustment_speed=0.15
        )

        model = Model().to(device)
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

            if args.plot and HAS_MATPLOTLIB and epoch % 2 == 0:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

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

                zoom_max = 0.5
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

        print(f"\nFinal adaptive C: {adaptive_controller.c:.4f}")
        print(f"C history: {[f'{c:.3f}' for c in adaptive_controller.c_history]}")

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
        torch.save(model.state_dict(), f"cifar10_cnn_{repro_str}.pt")


if __name__ == "__main__":
    main()