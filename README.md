# Differentially Private SGD with Adaptive Gradient Clipping

This repository contains PyTorch and Opacus experiments for DP-SGD on MNIST,
Fashion-MNIST, and CIFAR-10. The main focus is adaptive gradient clipping:
instead of using one fixed clipping threshold `C` for the whole run, the scripts
estimate the gradient norm distribution during training and update `C` between
epochs.

The code supports three clipping modes:

- `fixed`: standard DP-SGD with a constant clipping threshold.
- `ratio`: adapt `C` to keep the clipped-sample ratio near a target value.
- `mse`: adapt `C` by minimizing an estimated clipping-noise mean squared error.

Experiment outputs, checkpoints, plots, notebooks, thesis files, and work logs
are intentionally ignored by Git. The repository is meant to track runnable
source code and lightweight project documentation only.

## Project Layout

| Path | Purpose |
|---|---|
| `adaptive_clipping.py` | Shared histogram, ratio, and MSE clipping controller logic. |
| `minst_baseline.py` | MNIST fixed-C DP-SGD baseline. |
| `minst_adaptive_histogram.py` | MNIST adaptive clipping with `ratio` or `mse` mode. |
| `fashion_mnist_*.py`, `fashionmnist_*.py` | Fashion-MNIST baselines and adaptive experiments. |
| `cifar10_*.py` | CIFAR-10 baselines, ResNet experiments, and trade-off scripts. |
| `run_all_experiments.py` | Batch orchestration for root-level experiments. |
| `dp_histogram_mainline/` | Isolated mainline scripts for MSE + DP histogram experiments. |
| `plot_*.py`, `make_*.py` | Plot and figure generation utilities. |

The `dp_histogram_mainline/` directory is the cleanest place to run the current
DP histogram version of the MSE method. It saves outputs under
`dp_histogram_mainline/results/`, which is ignored by Git.

## Setup

Activate the local virtual environment before running experiments:

```bash
source .venv/Scripts/activate
```

Install the main dependencies if needed:

```bash
pip install torch torchvision opacus numpy tqdm matplotlib
```

Use `--help` on any script to confirm its current options:

```bash
python minst_adaptive_histogram.py --help
python dp_histogram_mainline/cifar10_resnet_dp.py --help
```

## Quick Start

Fixed-C MNIST baseline:

```bash
python minst_baseline.py -n 10 -b 64 --sigma 1.0 -c 0.4
```

MNIST adaptive clipping with clipped-ratio control:

```bash
python minst_adaptive_histogram.py -n 10 -b 64 --sigma 1.0 -c 0.4 --mode ratio --target-ratio 0.3 --plot
```

MNIST adaptive clipping with MSE control:

```bash
python minst_adaptive_histogram.py -n 10 -b 64 --sigma 1.0 -c 1.0 --mode mse --plot
```

Fashion-MNIST ResNet18 with MSE control:

```bash
python fashionmnist_resnet18_dp_baseline.py -n 20 -b 128 --sigma 1.0 --mode mse --initial-c 1.0
```

CIFAR-10 ResNet with MSE control:

```bash
python cifar10_resnet_dp.py -n 20 -b 1024 --max-physical-batch-size 32 --sigma 1.0 --mode mse --initial-c 1.0
```

## Clipping Methods

### Fixed C

`fixed` mode uses Opacus DP-SGD with a constant per-sample gradient norm bound.
This is the standard baseline. It is simple and stable, but performance can be
sensitive to the chosen `C`.

Example:

```bash
python cifar10_resnet_dp.py -n 20 -b 1024 --sigma 1.0 --mode fixed -c 1.0
```

### Ratio-Based Adaptive C

`ratio` mode measures the fraction of samples whose true per-sample gradient
norm is above the current clipping threshold. It then updates `C` toward a
target clipped ratio:

- If too many samples are clipped, increase `C`.
- If too few samples are clipped, decrease `C`.
- If the ratio is within tolerance, keep `C` nearly unchanged.

This method is easy to interpret and usually gives smooth threshold updates.

Example:

```bash
python minst_adaptive_histogram.py -n 10 -b 64 --sigma 1.0 --mode ratio --target-ratio 0.2
```

### MSE-Based Adaptive C

`mse` mode chooses `C` by estimating the bias-variance trade-off caused by
clipping and DP noise. For candidate clipping thresholds, it estimates:

```text
MSE(C) = bias(C) + variance(C)

bias(C)     = (1 / N) * sum_i max(||g_i|| - C, 0)^2
variance(C) = sigma^2 * C^2 * d / batch_size^2
```

The bias term is approximated from a histogram of true per-sample gradient
norms. The variance term grows with `C`, the DP noise multiplier `sigma`, and
the model dimension `d`. The controller searches over candidate `C` values,
selects the one with the lowest estimated MSE, then smooths the update with the
previous threshold.

This is the main method for experiments that try to balance clipping bias
against DP noise automatically.

Example:

```bash
python fashionmnist_resnet18_dp_baseline.py -n 20 -b 2048 --max-physical-batch-size 32 --sigma 1.0 --mode mse --initial-c 1.0
```

## DP Histogram Mainline

The root adaptive scripts can compute the histogram directly from raw
per-sample norms. The `dp_histogram_mainline/` scripts additionally support a
private histogram query:

```bash
python dp_histogram_mainline/minst_adaptive_histogram.py -n 1 -b 64 --sigma 1.0 --mode mse --use-dp-histogram --epsilon-hist 0.05 --device cpu
```

With `--use-dp-histogram`, histogram counts are perturbed with Laplace noise.
Each epoch histogram query spends `epsilon_hist`, and the scripts report the
combined privacy accounting as:

```text
epsilon_total = epsilon_sgd + epsilon_hist_total
```

Useful mainline runners:

```bash
python dp_histogram_mainline/run_representative_experiments.py --list
python dp_histogram_mainline/run_representative_experiments.py --dry-run
python dp_histogram_mainline/run_full_tradeoff_sweeps.py --dataset mnist --method mse
```

After a full sweep, regenerate trade-off figures with:

```bash
python plot_tradeoff_curves.py
```

## Common Arguments

| Argument | Meaning |
|---|---|
| `-n`, `--epochs` | Number of training epochs. |
| `-b`, `--batch-size` | Logical batch size. |
| `--max-physical-batch-size` | Physical batch size for memory-efficient Opacus training. |
| `--sigma` | DP noise multiplier. |
| `-c`, `--initial-c` or `--max-per-sample-grad-norm` | Initial or fixed clipping threshold. |
| `--mode {fixed,ratio,mse}` | Clipping strategy, where supported by the script. |
| `--target-ratio` | Target clipped ratio for `ratio` mode. |
| `--use-dp-histogram` | Use noisy histogram counts in mainline scripts. |
| `--epsilon-hist` | Privacy budget per histogram query. |
| `--plot` | Save histogram and training summary plots when supported. |
| `--disable-dp` | Run the non-private baseline. |

## Outputs

Training scripts may create `.pt`, `.csv`, `.png`, `.log`, checkpoint, and plot
directories. These are ignored by `.gitignore` because they are experiment
artifacts rather than source code. Typical generated paths include:

```text
run_results_*.pt
adaptive_histogram_results_*.pt
ckpt_*_done.pt
experiment_results*/
histogram_plots*/
dp_histogram_mainline/results/
figures/
```

Keep privacy settings explicit in output names when adding new scripts:
include `sigma`, clipping threshold or mode, batch size, and epoch count.

## Smoke Tests

Use short runs before launching full experiments:

```bash
python minst_baseline.py -n 1 -b 64 --sigma 1.0 -c 0.4
python minst_adaptive_histogram.py -n 1 -b 64 --sigma 1.0 -c 1.0 --mode mse
python dp_histogram_mainline/minst_adaptive_histogram.py -n 1 -b 64 --sigma 1.0 --mode mse --use-dp-histogram --epsilon-hist 0.05 --device cpu
```

For smoke tests, check that training completes, `C` stays finite, and privacy
accounting is printed when DP is enabled.
