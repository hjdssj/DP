#!/usr/bin/env python3
"""
MNIST training with Opacus DP-SGD and histogram-based adaptive gradient clipping.

Based on minst_baseline.py with added:
1. Gradient norm histogram tracking via Opacus's grad_sample
2. Adaptive C adjustment based on clipped sample ratio
3. Visualization of gradient distributions

Key insight: Opacus's grad_sample contains CLIPPED gradient norms.
We use this to estimate clipping ratio and adjust C accordingly.

The adaptive strategy:
- Track fraction of samples that were clipped (norm > C)
- If clipped_ratio > target_ratio (e.g., 20%), increase C
- If clipped_ratio < target_ratio (e.g., 10%), decrease C
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from torchvision import datasets, transforms
from tqdm import tqdm

from adaptive_clipping import GradientHistogram, AdaptiveClipController


# Precomputed characteristics of the MNIST dataset
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
RESULTS_DIR = Path(__file__).resolve().parent / "results"


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
          histogram, adaptive_controller):
    """Training loop with histogram tracking and adaptive clipping."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []

    for (data, target) in tqdm(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        if adaptive_controller is not None:
            # Capture TRUE (pre-clip) per-sample gradient norms from grad_sample.
            # grad_sample is populated by Opacus after backward() but BEFORE
            # optimizer.step() calls clip_and_accumulate().
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

    if adaptive_controller is None:
        if not args.disable_dp:
            epsilon_sgd = privacy_engine.accountant.get_epsilon(delta=args.delta)
            print(
                f"Train Epoch: {epoch} \t"
                f"Loss: {np.mean(losses):.6f} "
                f"(ε_sgd={epsilon_sgd:.2f}, ε_hist=0.00, "
                f"ε_total={epsilon_sgd:.2f}, δ={args.delta})"
            )
        else:
            print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")
        return

    # Update adaptive C based on histogram (ratio or mse mode)
    new_c, old_c = adaptive_controller.update(histogram)

    epsilon_sgd = 0.0
    epsilon_hist_total = adaptive_controller.epsilon_hist_spent
    epsilon_total = epsilon_hist_total
    if not args.disable_dp:
        epsilon_sgd = privacy_engine.accountant.get_epsilon(delta=args.delta)
        epsilon_total = epsilon_sgd + epsilon_hist_total
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"(ε_sgd={epsilon_sgd:.2f}, ε_hist={epsilon_hist_total:.2f}, "
            f"ε_total={epsilon_total:.2f}, δ={args.delta})"
        )
    else:
        print(
            f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f} "
            f"(ε_hist={epsilon_hist_total:.2f})"
        )

    # --- FIX: write new C back into the optimizer so Opacus actually uses it ---
    if not args.disable_dp and hasattr(optimizer, 'max_grad_norm'):
        optimizer.max_grad_norm = new_c

    # Sync histogram bin range with new C
    # MSE mode needs wide range to capture tail for accurate bias estimate
    if adaptive_controller.mode == 'mse':
        histogram.set_bin_max(max(new_c * 5, histogram.bin_max, 5.0))
    else:
        histogram.set_bin_max(max(new_c * 2, 2.0))

    stats = histogram.get_stats()
    clipped_ratio = adaptive_controller.clipped_ratio_history[-1]
    ratio_source = adaptive_controller.last_update_info.get('clipped_ratio_source', 'true')

    if adaptive_controller.mode == 'mse' and len(adaptive_controller.mse_history) > 0:
        print(
            f"  Adaptive C: {new_c:.4f} (was {old_c:.4f})"
            f" | Clipped: {clipped_ratio:.1%} [{ratio_source}]"
            f" | MSE={adaptive_controller.mse_history[-1]:.4f}"
            f" (bias={adaptive_controller.bias_history[-1]:.4f}"
            f" var={adaptive_controller.var_history[-1]:.4f})"
            f" | Grad mean: {stats.get('mean', 0):.4f}"
            + (f" | optimizer.max_grad_norm={optimizer.max_grad_norm:.4f}"
               if hasattr(optimizer, 'max_grad_norm') else "")
        )
    else:
        print(
            f"  Adaptive C: {new_c:.4f} (was {old_c:.4f})"
            f" | Clipped: {clipped_ratio:.1%} [{ratio_source}]"
            f" (target: {args.target_ratio:.0%})"
            f" | Grad mean: {stats.get('mean', 0):.4f}"
            + (f" | optimizer.max_grad_norm={optimizer.max_grad_norm:.4f}"
               if hasattr(optimizer, 'max_grad_norm') else "")
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
    parser.add_argument("--mode", type=str, default="ratio",
                        choices=["fixed", "ratio", "mse"],
                        help="Clipping mode: 'fixed' (static C), 'ratio' (clipped-ratio), or 'mse' (MSE minimization)")
    parser.add_argument("--use-dp-histogram", action="store_true",
                        help="Use Laplace-noisy histogram queries for adaptive clipping")
    parser.add_argument("--epsilon-hist", type=float, default=1.0,
                        help="Per-epoch privacy budget for histogram query")
    args = parser.parse_args()
    device = torch.device(args.device)

    # Compute model dimension d for MSE mode
    tmp_model = SampleConvNet()
    d = sum(p.numel() for p in tmp_model.parameters())
    del tmp_model
    print(f"Model parameter dimension d = {d}")

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

    repro_str = (
        f"mnist_adaptive_{args.mode}_{args.lr}_{args.sigma}_"
        f"{args.initial_c}_{args.batch_size}_{args.epochs}"
        f"_hist{args.epsilon_hist if args.use_dp_histogram else 0}"
    )
    plot_dir = f"histogram_plots_{repro_str}"
    if args.plot and HAS_MATPLOTLIB:
        os.makedirs(plot_dir, exist_ok=True)

    run_results = []

    for run_idx in range(args.n_runs):
        print(f"\n=== Run {run_idx + 1}/{args.n_runs} ===")
        if args.mode == "fixed":
            print(f"Fixed Clipping: C={args.initial_c}")
        else:
            print(f"Adaptive Clipping: target_ratio={args.target_ratio}, initial C={args.initial_c}")

        # Reset
        histogram = None
        adaptive_controller = None
        if args.mode in ("ratio", "mse"):
            hist_bin_max = 10.0 if args.mode == 'mse' else args.initial_c * 2
            histogram = GradientHistogram(
                bin_min=0.0,
                bin_max=hist_bin_max,
                num_bins=100 if args.mode == 'mse' else 50
            )
            adaptive_controller = AdaptiveClipController(
                mode=args.mode,
                initial_c=args.initial_c,
                target_ratio=args.target_ratio,
                tolerance=0.05,
                min_c=0.1,
                max_c=10.0,
                adjustment_speed=0.15,
                sigma=args.sigma,
                d=d,
                batch_size=args.batch_size,
                use_dp_histogram=args.use_dp_histogram,
                epsilon_hist=args.epsilon_hist,
            )

        model = SampleConvNet().to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0)
        privacy_engine = None
        clip_c = adaptive_controller.get_c() if adaptive_controller is not None else args.initial_c

        if not args.disable_dp:
            privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=args.sigma,
                max_grad_norm=clip_c,
            )

        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, privacy_engine,
                  epoch, histogram, adaptive_controller)

            # Plot histogram every 2 epochs
            if args.plot and HAS_MATPLOTLIB and adaptive_controller is not None and epoch % 2 == 0:
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

        epsilon_sgd = 0.0
        if not args.disable_dp:
            epsilon_sgd = privacy_engine.accountant.get_epsilon(delta=args.delta)
        epsilon_hist_total = (
            adaptive_controller.epsilon_hist_spent if adaptive_controller is not None else 0.0
        )
        epsilon_total = epsilon_sgd + epsilon_hist_total

        # Final C summary
        if adaptive_controller is not None:
            print(f"\nFinal adaptive C: {adaptive_controller.c:.4f}")
            print(f"C history: {[f'{c:.3f}' for c in adaptive_controller.c_history]}")
        else:
            print(f"\nFinal fixed C: {args.initial_c:.4f}")
        print(
            f"Privacy: ε_sgd={epsilon_sgd:.4f}, "
            f"ε_hist_total={epsilon_hist_total:.4f}, "
            f"ε_total={epsilon_total:.4f}, δ={args.delta}"
        )

        # Plot C and clipped ratio / MSE over training
        if args.plot and HAS_MATPLOTLIB and adaptive_controller is not None:
            if adaptive_controller.mode == 'mse' and len(adaptive_controller.mse_history) > 0:
                fig, axes = plt.subplots(3, 1, figsize=(12, 12))

                epochs = range(1, len(adaptive_controller.c_history))
                axes[0].plot(epochs, adaptive_controller.c_history[1:], 'b-o')
                axes[0].set_xlabel('Epoch')
                axes[0].set_ylabel('Clipping Threshold C')
                axes[0].set_title('Adaptive C (MSE mode) Over Training')
                axes[0].grid(True, alpha=0.3)

                axes[1].plot(epochs, adaptive_controller.clipped_ratio_history, 'r-s')
                axes[1].set_xlabel('Epoch')
                axes[1].set_ylabel('Clipped Ratio')
                axes[1].set_title('Clipped Sample Ratio Over Training')
                axes[1].grid(True, alpha=0.3)

                axes[2].plot(epochs, adaptive_controller.mse_history, 'r-o', label='MSE')
                axes[2].plot(epochs, adaptive_controller.bias_history, 'g-s', label='Bias')
                axes[2].plot(epochs, adaptive_controller.var_history, 'm-^', label='Variance')
                axes[2].set_xlabel('Epoch')
                axes[2].set_ylabel('Error')
                axes[2].set_title('MSE Components Over Training')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
            else:
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

    save_dict = {
        'run_results': run_results,
        'mode': args.mode,
        'use_dp_histogram': args.use_dp_histogram,
        'epsilon_hist_per_epoch': (
            args.epsilon_hist if args.use_dp_histogram and adaptive_controller is not None else 0.0
        ),
        'epsilon_hist_total': (
            adaptive_controller.epsilon_hist_spent if adaptive_controller is not None else 0.0
        ),
        'epsilon_sgd': (
            privacy_engine.accountant.get_epsilon(delta=args.delta)
            if privacy_engine is not None and not args.disable_dp else 0.0
        ),
        'epsilon_total': (
            (adaptive_controller.epsilon_hist_spent if adaptive_controller is not None else 0.0)
            + (
                privacy_engine.accountant.get_epsilon(delta=args.delta)
                if privacy_engine is not None and not args.disable_dp else 0.0
            )
        ),
        'histogram_query_count': (
            adaptive_controller.histogram_query_count if adaptive_controller is not None else 0
        ),
        'args': vars(args),
    }
    if adaptive_controller is not None:
        save_dict['c_history'] = adaptive_controller.c_history
        save_dict['clipped_ratio_history'] = adaptive_controller.clipped_ratio_history
    if args.mode == 'mse':
        save_dict['mse_history'] = adaptive_controller.mse_history
        save_dict['bias_history'] = adaptive_controller.bias_history
        save_dict['var_history'] = adaptive_controller.var_history
        save_dict['mse_curve_history'] = adaptive_controller.mse_curve_history
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / f"adaptive_histogram_results_{repro_str}.pt"
    torch.save(save_dict, result_path)
    print(f"Saved results to {result_path}")

    if args.save_model:
        model_path = RESULTS_DIR / f"mnist_cnn_{repro_str}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
