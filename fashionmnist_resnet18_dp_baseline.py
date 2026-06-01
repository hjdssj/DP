#!/usr/bin/env python3
"""
Fashion-MNIST training with DP-SGD using ResNet18 + Opacus.

ResNet18 adapted for 1x28x28 grayscale input:
  - Initial conv: 3x3, stride=1 (preserves 28x28)
  - No maxpool after conv1 (28x28 is too small for the standard 7x7 stride-2 design)
  - BatchNorm -> GroupNorm via Opacus ModuleValidator.fix()
  - BatchMemoryManager for grad_sample memory efficiency

Usage:
    # DP training with fixed C (baseline)
    python fashionmnist_resnet18_dp_baseline.py -n 20 -b 128 --sigma 1.0 -c 1.0

    # DP training with MSE adaptive clipping
    python fashionmnist_resnet18_dp_baseline.py -n 20 -b 128 --sigma 1.0 --mode mse --initial-c 1.0

    # DP training with ratio adaptive clipping
    python fashionmnist_resnet18_dp_baseline.py -n 20 -b 128 --sigma 1.0 --mode ratio --target-ratio 0.2

    # Disable DP (vanilla SGD baseline)
    python fashionmnist_resnet18_dp_baseline.py --disable-dp -n 20

    # Save model
    python fashionmnist_resnet18_dp_baseline.py -n 20 --save-model
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from torchvision import datasets, transforms
from tqdm import tqdm

from adaptive_clipping import GradientHistogram, AdaptiveClipController


# Fashion-MNIST normalization constants
FASHION_MNIST_MEAN = 0.2860
FASHION_MNIST_STD = 0.3530


# ---------------------------------------------------------------------------
# ResNet18 for Fashion-MNIST (1 x 28 x 28)
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    """ResNet BasicBlock with BatchNorm (will be replaced by GroupNorm)."""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity  # NOT += (inplace op conflicts with Opacus hooks)
        return F.relu(out)


class ResNet18(nn.Module):
    """ResNet-18 adapted for 1x28x28 Fashion-MNIST input.

    Differences from standard ImageNet ResNet-18:
    - in_channels=1 (grayscale, not 3)
    - conv1: 3x3 kernel, stride=1, padding=1 (not 7x7 stride=2)
    - No maxpool after conv1 (28x28 too small)
    - BatchNorm will be replaced by GroupNorm via ModuleValidator.fix()
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # Kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [BasicBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # [B, 64, 28, 28]
        x = self.layer1(x)                     # [B, 64, 28, 28]
        x = self.layer2(x)                     # [B, 128, 14, 14]
        x = self.layer3(x)                     # [B, 256, 7, 7]
        x = self.layer4(x)                     # [B, 512, 4, 4]
        x = self.avgpool(x)                    # [B, 512, 1, 1]
        x = torch.flatten(x, 1)               # [B, 512]
        return self.fc(x)                      # [B, 10]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args, model, device, train_loader, optimizer, privacy_engine, epoch,
          histogram, adaptive_controller):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []

    # BatchMemoryManager splits logical batch into smaller physical batches
    # to avoid OOM from per-sample gradient storage (grad_sample).
    # Logical batch size = args.batch_size (privacy budget unchanged).
    # Physical batch size = args.max_physical_batch_size (memory controlled).
    if not args.disable_dp:
        with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=args.max_physical_batch_size,
            optimizer=optimizer,
        ) as memory_safe_loader:
            for data, target in tqdm(memory_safe_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()

                # Capture TRUE (pre-clip) per-sample gradient norms.
                # grad_sample is populated by Opacus after backward() but BEFORE
                # optimizer.step() calls clip_and_accumulate().
                if adaptive_controller is not None:
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
    else:
        for data, target in tqdm(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    if not args.disable_dp:
        epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"(eps = {epsilon:.2f}, delta = {args.delta})"
        )
    else:
        print(f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f}")

    # Update adaptive C based on histogram
    if adaptive_controller is not None:
        new_c, old_c = adaptive_controller.update(histogram)

        # Write new C back into the optimizer so Opacus actually uses it
        if not args.disable_dp and hasattr(optimizer, 'max_grad_norm'):
            optimizer.max_grad_norm = new_c

        # Sync histogram bin range with new C
        if adaptive_controller.mode == 'mse':
            histogram.set_bin_max(max(new_c * 5, histogram.bin_max, 5.0))
        else:
            histogram.set_bin_max(max(new_c * 2, 2.0))

        stats = histogram.get_stats()
        clipped_ratio = histogram.get_clipped_ratio()

        if adaptive_controller.mode == 'mse' and len(adaptive_controller.mse_history) > 0:
            print(
                f"  Adaptive C: {new_c:.4f} (was {old_c:.4f})"
                f" | Clipped: {clipped_ratio:.1%}"
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
                f" | Clipped: {clipped_ratio:.1%} (target: {args.target_ratio:.0%})"
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fashion-MNIST ResNet18 with Opacus DP-SGD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-b", "--batch-size", type=int, default=128,
                        help="Logical batch size (privacy budget)")
    parser.add_argument("--max-physical-batch-size", type=int, default=32,
                        help="Physical batch size for BatchMemoryManager (memory control)")
    parser.add_argument("--test-batch-size", type=int, default=1024)
    parser.add_argument("-n", "--epochs", type=int, default=20)
    parser.add_argument("-r", "--n-runs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.05,
                        help="Learning rate")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="Noise multiplier for DP")
    parser.add_argument("-c", "--max-per-sample-grad-norm", type=float, default=1.0,
                        help="Clip per-sample gradients to this norm (C)")
    parser.add_argument("--mode", type=str, default="fixed",
                        choices=["fixed", "ratio", "mse"],
                        help="Clipping mode: 'fixed' (static C), 'ratio' (clipped-ratio adaptive), 'mse' (MSE minimization)")
    parser.add_argument("--initial-c", type=float, default=1.0,
                        help="Initial clipping threshold for adaptive modes")
    parser.add_argument("--target-ratio", type=float, default=0.2,
                        help="Target fraction of samples to clip (0.0-1.0, ratio mode only)")
    parser.add_argument("--delta", type=float, default=None,
                        help="Target delta (default: 1/N_train)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--disable-dp", action="store_true",
                        help="Disable DP, train with vanilla SGD")
    parser.add_argument("--secure-rng", action="store_true")
    parser.add_argument("--data-root", type=str, default="../fashion_mnist")
    args = parser.parse_args()
    device = torch.device(args.device)

    # Delta = 1/N is standard for DP-SGD
    # Computed after we know the dataset size
    train_dataset = datasets.FashionMNIST(
        args.data_root, train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((FASHION_MNIST_MEAN,), (FASHION_MNIST_STD,)),
        ]),
    )
    if args.delta is None:
        args.delta = 1.0 / len(train_dataset)
    print(f"delta = {args.delta:.2e} (1 / {len(train_dataset)})")

    test_dataset = datasets.FashionMNIST(
        args.data_root, train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((FASHION_MNIST_MEAN,), (FASHION_MNIST_STD,)),
        ]),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    run_results = []

    for run_idx in range(args.n_runs):
        print(f"\n{'='*50}")
        print(f"  Run {run_idx + 1}/{args.n_runs}")
        print(f"{'='*50}")

        # Build ResNet18 with BatchNorm, then fix for Opacus
        model = ResNet18(num_classes=10)

        # Validate and fix: replace BatchNorm with GroupNorm
        # This is REQUIRED for DP training correctness.
        # BatchNorm makes each sample's output depend on its batch peers,
        # violating per-sample DP guarantees.
        if not args.disable_dp:
            errors = ModuleValidator.validate(model, strict=False)
            if errors:
                print(f"Fixing {len(errors)} incompatible modules (BatchNorm -> GroupNorm)...")
                model = ModuleValidator.fix(model)
                assert ModuleValidator.validate(model, strict=False) == [], \
                    "ModuleValidator.fix() failed to resolve all issues"
                print("All modules now compatible with Opacus.")

        model = model.to(device)

        # Print model info
        d = sum(p.numel() for p in model.parameters())
        print(f"Model: ResNet18 | Parameters: {d:,}")

        # Initialize histogram and adaptive controller
        histogram = None
        adaptive_controller = None

        if args.mode in ('ratio', 'mse'):
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
            )
            print(f"Adaptive clipping: mode={args.mode}, initial_C={args.initial_c}"
                  + (f", target_ratio={args.target_ratio}" if args.mode == 'ratio' else ""))

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                              weight_decay=1e-4)
        privacy_engine = None

        # Determine the C value for Opacus
        clip_c = args.max_per_sample_grad_norm
        if adaptive_controller is not None:
            clip_c = adaptive_controller.get_c()

        if not args.disable_dp:
            privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=args.sigma,
                max_grad_norm=clip_c,
            )
            print(f"DP enabled: sigma={args.sigma}, C={clip_c}, "
                  f"batch_size={args.batch_size}, "
                  f"max_physical_batch_size={args.max_physical_batch_size}")

        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, privacy_engine, epoch,
                  histogram, adaptive_controller)

        accuracy = test(model, device, test_loader)
        run_results.append(accuracy)

        # Print adaptive C summary
        if adaptive_controller is not None:
            print(f"\nFinal adaptive C: {adaptive_controller.c:.4f}")
            print(f"C history: {[f'{c:.3f}' for c in adaptive_controller.c_history]}")

        # Print final summary
        print(f"\n--- Run {run_idx + 1} Summary ---")
        print(f"  Final test accuracy: {accuracy * 100:.2f}%")
        if not args.disable_dp and privacy_engine is not None:
            final_epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
            print(f"  Final epsilon: {final_epsilon:.2f} (delta = {args.delta:.2e})")

    if len(run_results) > 1:
        print(
            f"\nAccuracy averaged over {len(run_results)} runs: "
            f"{np.mean(run_results) * 100:.2f}% +/- {np.std(run_results) * 100:.2f}%"
        )

    # Save results
    repro_str = (
        f"fashion_resnet18_{args.mode}_{args.lr}_{args.sigma}_"
        f"{args.max_per_sample_grad_norm}_{args.batch_size}_{args.epochs}"
    )

    save_dict = {
        'run_results': run_results,
        'mode': args.mode,
    }
    if adaptive_controller is not None:
        save_dict['c_history'] = adaptive_controller.c_history
        save_dict['clipped_ratio_history'] = adaptive_controller.clipped_ratio_history
        if args.mode == 'mse':
            save_dict['mse_history'] = adaptive_controller.mse_history
            save_dict['bias_history'] = adaptive_controller.bias_history
            save_dict['var_history'] = adaptive_controller.var_history
    torch.save(save_dict, f"adaptive_histogram_results_{repro_str}.pt")

    if args.save_model:
        torch.save(model.state_dict(), f"fashion_resnet18_{repro_str}.pt")


if __name__ == "__main__":
    main()
