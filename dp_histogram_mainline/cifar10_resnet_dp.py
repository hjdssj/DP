#!/usr/bin/env python3
"""
CIFAR-10 training with DP-SGD using Res_Model + Opacus.

Res_Model with BatchNorm (auto-replaced by GroupNorm via ModuleValidator.fix()),
supports three clipping modes: fixed / ratio / mse.

Usage:
    # DP training with fixed C (baseline)
    python cifar10_resnet_dp.py -n 100 -b 1024 --sigma 1.0 -c 1.0

    # DP training with MSE adaptive clipping
    python cifar10_resnet_dp.py -n 100 -b 1024 --sigma 1.0 --mode mse --initial-c 1.0

    # DP training with ratio adaptive clipping
    python cifar10_resnet_dp.py -n 100 -b 1024 --sigma 1.0 --mode ratio --target-ratio 0.2

    # Non-DP baseline
    python cifar10_resnet_dp.py --disable-dp -n 100
"""

import argparse
from pathlib import Path

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


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
RESULTS_DIR = Path(__file__).resolve().parent / "results"


class Conv_Block(nn.Module):
    def __init__(self, inchannel, outchannel, res=True):
        super().__init__()
        self.res = res
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, 3, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.left(x)
        if self.res:
            out = out + self.shortcut(x)
        return self.relu(out)


class Res_Model(nn.Module):
    def __init__(self, res=True):
        super().__init__()
        self.block1 = Conv_Block(3, 64, res)
        self.block2 = Conv_Block(64, 128, res)
        self.block3 = Conv_Block(128, 256, res)
        self.block4 = Conv_Block(256, 512, res)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.maxpool(self.block1(x))
        x = self.maxpool(self.block2(x))
        x = self.maxpool(self.block3(x))
        x = self.maxpool(self.block4(x))
        return self.classifier(x)


def train(args, model, device, train_loader, optimizer, privacy_engine, epoch,
          histogram, adaptive_controller):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []

    if not args.disable_dp:
        with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=args.max_physical_batch_size,
            optimizer=optimizer,
        ) as memory_safe_loader:
            for data, target in tqdm(memory_safe_loader, desc=f"Epoch {epoch}"):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()

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
        for data, target in tqdm(train_loader, desc=f"Epoch {epoch}"):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    if adaptive_controller is not None:
        new_c, old_c = adaptive_controller.update(histogram)

        if not args.disable_dp and hasattr(optimizer, 'max_grad_norm'):
            optimizer.max_grad_norm = new_c

        if adaptive_controller.mode == 'mse':
            histogram.set_bin_max(max(new_c * 5, histogram.bin_max, 5.0))
        else:
            histogram.set_bin_max(max(new_c * 2, 2.0))

        stats = histogram.get_stats()
        clipped_ratio = adaptive_controller.clipped_ratio_history[-1]
        ratio_source = adaptive_controller.last_update_info.get('clipped_ratio_source', 'true')

    epsilon_sgd = 0.0
    epsilon_hist_total = (
        adaptive_controller.epsilon_hist_spent if adaptive_controller is not None else 0.0
    )
    epsilon_total = epsilon_hist_total
    if not args.disable_dp:
        epsilon_sgd = privacy_engine.accountant.get_epsilon(delta=args.delta)
        epsilon_total = epsilon_sgd + epsilon_hist_total
        print(
            f"Train Epoch: {epoch} \t"
            f"Loss: {np.mean(losses):.6f} "
            f"(eps_sgd={epsilon_sgd:.2f}, eps_hist={epsilon_hist_total:.2f}, "
            f"eps_total={epsilon_total:.2f}, delta={args.delta})"
        )
    else:
        print(
            f"Train Epoch: {epoch} \t Loss: {np.mean(losses):.6f} "
            f"(eps_hist={epsilon_hist_total:.2f})"
        )

    if adaptive_controller is not None:

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
        for data, target in tqdm(test_loader, desc="Test"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n"
    )
    return correct / len(test_loader.dataset)


def main():
    parser = argparse.ArgumentParser(
        description="CIFAR-10 Res_Model with Opacus DP-SGD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-b", "--batch-size", type=int, default=1024,
                        help="Logical batch size (privacy budget)")
    parser.add_argument("--max-physical-batch-size", type=int, default=32,
                        help="Physical batch size for BatchMemoryManager")
    parser.add_argument("--test-batch-size", type=int, default=1024)
    parser.add_argument("-n", "--epochs", type=int, default=100)
    parser.add_argument("-r", "--n-runs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr-decay-epochs", type=int, nargs="+", default=[50, 75],
                        help="Epochs at which to decay learning rate by 10x")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="Noise multiplier for DP")
    parser.add_argument("-c", "--max-per-sample-grad-norm", type=float, default=1.0,
                        help="Clip per-sample gradients to this norm (C)")
    parser.add_argument("--mode", type=str, default="fixed",
                        choices=["fixed", "ratio", "mse"],
                        help="Clipping mode")
    parser.add_argument("--initial-c", type=float, default=1.0,
                        help="Initial clipping threshold for adaptive modes")
    parser.add_argument("--target-ratio", type=float, default=0.2,
                        help="Target fraction of samples to clip (ratio mode)")
    parser.add_argument("--delta", type=float, default=None,
                        help="Target delta (default: 1/N_train)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--disable-dp", action="store_true",
                        help="Disable DP, train with vanilla SGD")
    parser.add_argument("--secure-rng", action="store_true")
    parser.add_argument("--data-root", type=str, default="../cifar10")
    parser.add_argument("--use-dp-histogram", action="store_true",
                        help="Use Laplace-noisy histogram queries for adaptive clipping")
    parser.add_argument("--epsilon-hist", type=float, default=1.0,
                        help="Per-epoch privacy budget for histogram query")
    args = parser.parse_args()
    device = torch.device(args.device)

    train_dataset = datasets.CIFAR10(
        args.data_root, train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]),
    )
    if args.delta is None:
        args.delta = 1.0 / len(train_dataset)
    print(f"delta = {args.delta:.2e} (1 / {len(train_dataset)})")

    test_dataset = datasets.CIFAR10(
        args.data_root, train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
    )

    run_results = []

    for run_idx in range(args.n_runs):
        print(f"\n{'='*50}")
        print(f"  Run {run_idx + 1}/{args.n_runs}")
        print(f"{'='*50}")

        model = Res_Model()

        if not args.disable_dp:
            errors = ModuleValidator.validate(model, strict=False)
            if errors:
                print(f"Fixing {len(errors)} incompatible modules (BatchNorm -> GroupNorm)...")
                model = ModuleValidator.fix(model)
                assert ModuleValidator.validate(model, strict=False) == [], \
                    "ModuleValidator.fix() failed to resolve all issues"
                print("All modules now compatible with Opacus.")

        model = model.to(device)

        d = sum(p.numel() for p in model.parameters())
        print(f"Model: Res_Model | Parameters: {d:,} | d/n^2 = {d/args.batch_size**2:.1f}")

        histogram = None
        adaptive_controller = None

        if args.mode in ('ratio', 'mse'):
            hist_bin_max = 10.0 if args.mode == 'mse' else args.initial_c * 2
            histogram = GradientHistogram(
                bin_min=0.0, bin_max=hist_bin_max,
                num_bins=100 if args.mode == 'mse' else 50,
            )
            adaptive_controller = AdaptiveClipController(
                mode=args.mode, initial_c=args.initial_c,
                target_ratio=args.target_ratio, tolerance=0.05,
                min_c=0.1, max_c=10.0, adjustment_speed=0.15,
                sigma=args.sigma, d=d, batch_size=args.batch_size,
                use_dp_histogram=args.use_dp_histogram,
                epsilon_hist=args.epsilon_hist,
            )
            print(f"Adaptive clipping: mode={args.mode}, initial_C={args.initial_c}"
                  + (f", target_ratio={args.target_ratio}" if args.mode == 'ratio' else "")
                  + (f", dp_hist_epsilon={args.epsilon_hist}" if args.use_dp_histogram else ""))

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                              weight_decay=5e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_decay_epochs, gamma=0.1)
        privacy_engine = None

        clip_c = args.max_per_sample_grad_norm
        if adaptive_controller is not None:
            clip_c = adaptive_controller.get_c()

        if not args.disable_dp:
            privacy_engine = PrivacyEngine(secure_mode=args.secure_rng)
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model, optimizer=optimizer, data_loader=train_loader,
                noise_multiplier=args.sigma, max_grad_norm=clip_c,
            )
            print(f"DP enabled: sigma={args.sigma}, C={clip_c}, "
                  f"batch_size={args.batch_size}, "
                  f"max_physical_batch_size={args.max_physical_batch_size}")

        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, privacy_engine, epoch,
                  histogram, adaptive_controller)
            scheduler.step()

        accuracy = test(model, device, test_loader)
        run_results.append(accuracy)

        if adaptive_controller is not None:
            print(f"\nFinal adaptive C: {adaptive_controller.c:.4f}")
            print(f"C history: {[f'{c:.3f}' for c in adaptive_controller.c_history]}")

        print(f"\n--- Run {run_idx + 1} Summary ---")
        print(f"  Final test accuracy: {accuracy * 100:.2f}%")
        final_epsilon_sgd = 0.0
        if not args.disable_dp and privacy_engine is not None:
            final_epsilon_sgd = privacy_engine.accountant.get_epsilon(delta=args.delta)
        final_epsilon_hist = (
            adaptive_controller.epsilon_hist_spent if adaptive_controller is not None else 0.0
        )
        final_epsilon_total = final_epsilon_sgd + final_epsilon_hist
        if not args.disable_dp and privacy_engine is not None:
            print(
                f"  Final epsilon: sgd={final_epsilon_sgd:.2f}, "
                f"hist={final_epsilon_hist:.2f}, total={final_epsilon_total:.2f} "
                f"(delta = {args.delta:.2e})"
            )

    if len(run_results) > 1:
        print(
            f"\nAccuracy averaged over {len(run_results)} runs: "
            f"{np.mean(run_results) * 100:.2f}% +/- {np.std(run_results) * 100:.2f}%"
        )

    repro_str = (
        f"cifar10_{args.mode}_{args.lr}_{args.sigma}_"
        f"{args.max_per_sample_grad_norm}_{args.batch_size}_{args.epochs}"
        f"_hist{args.epsilon_hist if args.use_dp_histogram else 0}"
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
            (
                privacy_engine.accountant.get_epsilon(delta=args.delta)
                if privacy_engine is not None and not args.disable_dp else 0.0
            )
            + (adaptive_controller.epsilon_hist_spent if adaptive_controller is not None else 0.0)
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
        model_path = RESULTS_DIR / f"cifar10_resnet_{repro_str}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
