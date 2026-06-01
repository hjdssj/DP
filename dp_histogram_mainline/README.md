# DP Histogram Mainline Sandbox

This directory is an isolated copy of the scripts needed to integrate DP histogram into the adaptive clipping pipeline. It is intentionally separated from the repository root so experiments can be modified without mixing with the existing validated scripts.

## Copied Files

| File | Role |
|---|---|
| `adaptive_clipping.py` | Shared histogram, MSE, and adaptive clipping controller logic. |
| `minst_adaptive_histogram.py` | First target for DP histogram integration and smoke testing. |
| `fashionmnist_resnet18_dp_baseline.py` | Later target for Fashion-MNIST ResNet18 experiments. |
| `cifar10_resnet_dp.py` | Later target for CIFAR-10 fixed / ratio / MSE comparison. |
| `run_all_experiments.py` | Later target for batch experiment orchestration. |
| `fashion_adaptive_dp_histogram_reference.py` | Reference implementation for Laplace-noisy histogram and privacy budget accounting. |

## Development Rule

Modify files in this directory first. After the DP histogram pipeline is verified by smoke tests and representative experiments, selected changes can be ported back to the root scripts.

## Output Location

New experiment outputs from the main sandbox scripts are saved under:

```text
dp_histogram_mainline/results/
```

This applies to MNIST, Fashion-MNIST, CIFAR-10 result `.pt` files and optional model checkpoints. The directory is created automatically when a script saves results.

## First Smoke Test Target

```bash
python dp_histogram_mainline/minst_adaptive_histogram.py -n 1 -b 64 --sigma 1.0 --mode mse --use-dp-histogram --epsilon-hist 0.05 --device cpu
```

The expected result is not high accuracy, but a complete run with stable clipping threshold updates and explicit privacy accounting:

```text
epsilon_total = epsilon_sgd + epsilon_hist_total
```

## Follow-up Smoke Tests

After the MNIST smoke test passes, use the same flags for the larger dataset scripts:

```bash
python dp_histogram_mainline/fashionmnist_resnet18_dp_baseline.py -n 1 -b 256 --max-physical-batch-size 32 --sigma 1.0 --mode mse --use-dp-histogram --epsilon-hist 0.05 --device cuda
python dp_histogram_mainline/cifar10_resnet_dp.py -n 1 -b 256 --max-physical-batch-size 32 --sigma 1.0 --mode mse --use-dp-histogram --epsilon-hist 0.05 --device cuda
```

For these smoke tests, only check that training completes, `C` remains finite, and the printed privacy line includes SGD, histogram, and total epsilon.

## Representative Experiment Runner

Use the runner below to execute the current representative experiments with resume support:

```bash
python dp_histogram_mainline/run_representative_experiments.py
```

It currently runs:

1. Fashion-MNIST ResNet18 MSE + DP histogram, bs=2048, 20 epochs.
2. Fashion-MNIST ResNet18 fixed C baseline, bs=2048, 20 epochs.
3. CIFAR-10 MSE + DP histogram, bs=1024, 20 epochs.

Useful options:

```bash
python dp_histogram_mainline/run_representative_experiments.py --list
python dp_histogram_mainline/run_representative_experiments.py --dry-run
python dp_histogram_mainline/run_representative_experiments.py --force
```

Resume behavior: if the expected result file already exists in `dp_histogram_mainline/results/`, that experiment is skipped.

## Full Trade-off Sweep Runner

Use the full sweep runner when the goal is complete epsilon-Accuracy curves rather than a small set of representative points:

```bash
python dp_histogram_mainline/run_full_tradeoff_sweeps.py
```

It covers fixed C and MSE + DP histogram for:

| Dataset | Sigma grid | Epochs | Batch |
|---|---|---:|---:|
| MNIST | 0.5, 0.8, 1.0, 1.2, 1.5, 2.0 | 10 | 64 |
| Fashion-MNIST | 3.383908, 1.929262, 1.212828, 0.856756, 0.647915, 0.503971 | 20 | 2048 |
| CIFAR-10 | 1.5, 1.2, 1.0, 0.8, 0.6, 0.5 | 20 | 1024 |

Useful commands:

```bash
python dp_histogram_mainline/run_full_tradeoff_sweeps.py --list
python dp_histogram_mainline/run_full_tradeoff_sweeps.py --dry-run
python dp_histogram_mainline/run_full_tradeoff_sweeps.py --dataset mnist
python dp_histogram_mainline/run_full_tradeoff_sweeps.py --dataset fashion
python dp_histogram_mainline/run_full_tradeoff_sweeps.py --dataset cifar
python dp_histogram_mainline/run_full_tradeoff_sweeps.py --method mse
```

After the sweep, regenerate trade-off figures:

```bash
python plot_tradeoff_curves.py
```

The main full-sweep figure is saved to:

```text
figures/full_dp_hist_tradeoff_sweeps.png
```
