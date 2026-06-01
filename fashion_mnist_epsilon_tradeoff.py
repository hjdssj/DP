#!/usr/bin/env python3
"""
Fashion-MNIST: Privacy (epsilon) vs Utility (Accuracy) Trade-off Curve

Uses FashionCNN architecture (matching TensorFlow version).
Supports checkpoint resume to handle OOM issues.
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


FASHION_MNIST_MEAN = 0.2860
FASHION_MNIST_STD = 0.3530


class FashionCNN(nn.Module):
    """Fashion-MNIST CNN matching TensorFlow version."""
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


def train_and_test(args, device, checkpoint_path="checkpoint.pt"):
    """Train with DP and return (epsilon, accuracy). Resume if checkpoint exists."""

    if os.path.exists(checkpoint_path):
        print(f"  Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path)
        model_state = ckpt['model_state']
        optimizer_state = ckpt['optimizer_state']
        start_epoch = ckpt['epoch'] + 1
        print(f"  Resuming from epoch {start_epoch}")
    else:
        start_epoch = 1

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
            args.data_root, train=False, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((FASHION_MNIST_MEAN,), (FASHION_MNIST_STD,)),
            ])
        ),
        batch_size=args.test_batch_size, shuffle=True, num_workers=0, pin_memory=True,
    )

    model = FashionCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if os.path.exists(checkpoint_path):
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    privacy_engine = PrivacyEngine(secure_mode=False)
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=args.sigma,
        max_grad_norm=args.max_grad_norm,
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, leave=False)):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)
        print(f"  Epoch {epoch}/{args.epochs} | Loss: {avg_loss:.4f} | ε: {epsilon:.2f}")

        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'args': args,
        }, checkpoint_path)

    # Test
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100.0 * correct / len(test_loader.dataset)
    epsilon = privacy_engine.accountant.get_epsilon(delta=args.delta)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return epsilon, accuracy


def main():
    parser = argparse.ArgumentParser(description="Fashion-MNIST Privacy-Utility Trade-off")
    parser.add_argument("-b", "--batch-size", type=int, default=150)
    parser.add_argument("--test-batch-size", type=int, default=1024)
    parser.add_argument("-n", "--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("-c", "--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data-root", type=str, default="../fashion_mnist")
    parser.add_argument("--output", type=str, default="epsilon_tradeoff_fashion.npz")
    args = parser.parse_args()
    device = torch.device(args.device)

    # Method 1: Fix sigma, vary epochs
    print("=" * 60)
    print("Method 1: Fix sigma=1.0, vary epochs (1, 2, 5, 10, 15, 20)")
    print("=" * 60)

    results_m1 = []
    for epochs in [1, 2, 5, 10, 15, 20]:
        args.epochs = epochs
        args.sigma = 1.0
        ckpt = f"ckpt_fashion_m1_e{epochs}.pt"
        print(f"\nEpochs={epochs}, sigma={args.sigma}")
        if os.path.exists(ckpt.replace('.pt', '_done.pt')):
            print(f"  Already completed, skipping")
            with open(ckpt.replace('.pt', '_done.pt'), 'r') as f:
                line = f.read().strip().split(',')
                results_m1.append((float(line[0]), float(line[1])))
            continue
        try:
            epsilon, accuracy = train_and_test(args, device, ckpt)
            print(f"  epsilon={epsilon:.2f}, accuracy={accuracy:.2f}%")
            results_m1.append((epsilon, accuracy))
            with open(ckpt.replace('.pt', '_done.pt'), 'w') as f:
                f.write(f"{epsilon},{accuracy}")
        except torch.OutOfMemoryError as e:
            print(f"  OOM! Reduce batch size and try again")
            torch.cuda.empty_cache()
            break

    # Method 2: Fix epochs=10, vary sigma
    print("\n" + "=" * 60)
    print("Method 2: Fix epochs=10, vary sigma (0.5, 1.0, 1.5, 2.0, 3.0)")
    print("=" * 60)

    results_m2 = []
    for sigma in [0.5, 1.0, 1.5, 2.0, 3.0]:
        args.epochs = 10
        args.sigma = sigma
        ckpt = f"ckpt_fashion_m2_s{int(sigma*10)}.pt"
        print(f"\nEpochs={args.epochs}, sigma={sigma}")
        if os.path.exists(ckpt.replace('.pt', '_done.pt')):
            print(f"  Already completed, skipping")
            with open(ckpt.replace('.pt', '_done.pt'), 'r') as f:
                line = f.read().strip().split(',')
                results_m2.append((float(line[0]), float(line[1])))
            continue
        try:
            epsilon, accuracy = train_and_test(args, device, ckpt)
            print(f"  epsilon={epsilon:.2f}, accuracy={accuracy:.2f}%")
            results_m2.append((epsilon, accuracy))
            with open(ckpt.replace('.pt', '_done.pt'), 'w') as f:
                f.write(f"{epsilon},{accuracy}")
        except torch.OutOfMemoryError as e:
            print(f"  OOM! Reduce batch size and try again")
            torch.cuda.empty_cache()
            break

    results = results_m1 + results_m2

    if not results:
        print("\nNo results collected.")
        return

    # Save results
    epsilons = np.array([r[0] for r in results])
    accuracies = np.array([r[1] for r in results])
    np.savez(args.output, epsilon=epsilons, accuracy=accuracies)
    print(f"\nSaved to {args.output}")

    # Export as CSV table
    csv_path = args.output.replace('.npz', '.csv')
    with open(csv_path, 'w') as f:
        f.write("epsilon,accuracy\n")
        for eps, acc in results:
            f.write(f"{eps:.4f},{acc:.2f}\n")
    print(f"Saved table to {csv_path}")

    # Print table
    print("\n" + "=" * 50)
    print(f"{'epsilon':>12} | {'accuracy':>12}")
    print("=" * 50)
    for eps, acc in results:
        print(f"{eps:>12.4f} | {acc:>12.2f}%")
    print("=" * 50)

    # Plot
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))

        eps1 = np.array([r[0] for r in results_m1])
        acc1 = np.array([r[1] for r in results_m1])
        if len(eps1) > 0:
            plt.plot(eps1, acc1, 'bo-', markersize=8, label='Vary epochs (sigma=1.0)')

        eps2 = np.array([r[0] for r in results_m2])
        acc2 = np.array([r[1] for r in results_m2])
        if len(eps2) > 0:
            plt.plot(eps2, acc2, 'rs-', markersize=8, label='Vary sigma (epochs=10)')

        plt.xlabel('Privacy Budget (epsilon)')
        plt.ylabel('Accuracy (%)')
        plt.title('Fashion-MNIST: Privacy-Utility Trade-off')
        plt.legend()
        plt.grid(True, alpha=0.3)

        for eps, acc in results_m1:
            plt.annotate(f'{eps:.1f}', (eps, acc), textcoords="offset points", xytext=(5,5), fontsize=8)
        for eps, acc in results_m2:
            plt.annotate(f'{eps:.1f}', (eps, acc), textcoords="offset points", xytext=(5,-10), fontsize=8)

        plt.tight_layout()
        plt.savefig('epsilon_tradeoff_fashion.png', dpi=150)
        print("Saved plot to epsilon_tradeoff_fashion.png")
    except ImportError:
        print("matplotlib not available, skipping plot")


if __name__ == "__main__":
    main()